# from plotly import __version__
# from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#
# import plotly.plotly as py
# import plotly.graph_objs as go
# from plotly.graph_objs import Scatter, Figure, Layout
# from plotly.tools import FigureFactory as FF

import numpy as np
import pandas as pd
import logging
# import scipy
# import peakutils

from datetime import datetime, timedelta

import freqtrade.vendor.qtpylib.indicators as qtpylib
# from freqtrade.persistence import *
import freqtrade.persistence

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, String,
                        create_engine)
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.scoping import scoped_session
from sqlalchemy.orm.session import sessionmaker

from sklearn.cluster import MeanShift, estimate_bandwidth

from freqtrade.exchange import get_ticker_history

# from freqtrade.analyze import Analyze

import scripts.trendy_2 as trendy
from scripts import cactix

from pandas import Series

# from zigzag import *

logger = logging.getLogger('freqtrade')


def went_up(series: Series) -> Series:
    return series > series.shift(1)

def went_down(series: Series) -> Series:
    return series < series.shift(1)

def in_range(x, target, percent):
   start = target - target * percent
   end = target + target * percent
   check = (start <= x) & (end >= x)
   # print (check)
   return check

# def find_pivots(pair: str, interval: int=1, type: str='piv') -> pd.DataFrame:
    # return ''

def get_trend_lines(df: pd.DataFrame, pair: str, timerange: int=600, interval: str="1h", charts: bool=True) -> pd.DataFrame:
    main_trends, main_maxslope, main_minslope = cactix.gentrends(df, charts=charts)
    # main_trends, main_maxslope, main_minslope = trendy.gentrends(df.close, window=1/2, charts=charts)
    df['main_trend_max'] = main_trends['Max Line']
    df['main_trend_min'] = main_trends['Min Line']
    df['main_maxslope'] = main_maxslope
    df['main_minslope'] = main_minslope

    return df
            # short_trends['Max Line'], short_trends['Min Line'], \
            # short_maxslope, short_minslope

def bruno_pivots(pair: str, interval: int=1, piv_type: str='piv') -> pd.DataFrame:

    if piv_type == 'sup':
        # df = df[df['low'].value_counts()[df['low']] >= 3]
        df = df[(df['volume'] > df['volume'].rolling(window=10).mean() & df['close'] < df['open'])]
    elif piv_type == 'res':
        df = df[(df['volume'] > df['volume'].rolling(window=10).mean() & df['close'] > df['open'])]

def get_pivots(df: pd.DataFrame, pair: str, interval: int=1, piv_type: str='piv', full_df: pd.DataFrame=None) -> pd.DataFrame:
    # print ('find_pivots')

    quantile = 0.01
    cols = ['low', 'high']
    gap = 1.01


    if piv_type == 'sup':
        quantile = 0.01
        cols = ['low', 'high']
        gap = 1.05
        interval = 1
    elif piv_type == 'res':
        quantile = 0.01
        cols = ['high', 'low']
        gap = 0.96
        interval = 1

    # if 'USDT' in pair:
    #     if piv_type == 'sup':
    #         quantile = 0.01
    #         cols = ['low', 'high']
    #         gap = 1.05
    #         interval = 60
    #     elif piv_type == 'res':
    #         quantile = 0.01
    #         cols = ['high', 'low']
    #         gap = 0.95
    #         interval = 1

    # ticker_hist = get_ticker_history(pair, interval)
    # if not ticker_hist:
    #     logger.warning('Empty ticker history for pair %s', pair)
    #     return []  # return False ?
    #
    # try:
    #     dataframe = parse_ticker_dataframe(ticker_hist)
    # except ValueError as ex:
    #     logger.warning('Unable to analyze ticker for pair %s: %s', pair, str(ex))
    #     return []  # return False ?
    # except Exception as ex:
    #     logger.exception('Unexpected error when analyzing ticker for pair %s: %s', pair, str(ex))
    #     return []  # return False ?
    #
    # df = dataframe
    # print ('len df: ', len(df))
    # print ('quantile: ', quantile, type(quantile))
    # print ('bandwidth: ', int(len(df)) * quantile)

    # if piv_type == 'sup':
    #     # df = df[df['low'].value_counts()[df['low']] >= 3]
    #     df = df[(df['volume'] > df['volume'].rolling(window=10).mean() * 0.9) & (df['close'] < df['open'])]
    # elif piv_type == 'res':
    #     df = df[(df['volume'] > df['volume'].rolling(window=5).mean()) & (df['close'] > df['open'])]

    # if piv_type == 'sup':
    #     # df = df[df['low'].value_counts()[df['low']] >= 3]
    #     df = df[(df['low'] > df.iloc[-1]['close'] * 0.95)]
    # elif piv_type == 'res':
    #     df = df[(df['high'] < df.iloc[-1]['close'] * 1.05)]

    if len(full_df) <= len(df):
        full_df = df
    print ('len df: ', len(df))
    print ('len full_df: ', len(full_df))
    samples = len(full_df)

    print(len(full_df) * quantile)
    if full_df.empty:
        logger.warning('Empty dataframe for pair %s', pair)
        return []  # return False ?
    elif not len(full_df) * quantile > 1:
        print('dataframe too short: ', len(full_df))
        samples = len(full_df) * 0.1
    # full_df = full_df[(full_df['volume'] > full_df['volume'].rolling(window=10).mean())]


    data1 = full_df.as_matrix(columns=cols)
    # highest = full_df.high.rolling(window=len(full_df)).max()
    # lowest = full_df.low.rolling(window=len(full_df)).min()

    # print ('samples', samples)
    try:
        bandwidth1 = estimate_bandwidth(data1, quantile=quantile, n_samples=samples)
        ms1 = MeanShift(bandwidth=bandwidth1, bin_seeding=True)
        ms1.fit(data1)
    except Exception as ex:
        logger.exception('Unexpected error when analyzing ticker pivots for pair %s: %s', pair, str(ex))
        return []  # return False ?

    #Calculate Support/Resistance
    pivots = []
    # print ('labels', ms1.labels_)
    for k in range(len(np.unique(ms1.labels_))):
            my_members = ms1.labels_ == k
            values = data1[my_members, 0]
            # print (values)

            # find the edges
            if len(values) > 0:
                pivots.append(min(values))
                pivots.append(max(values))

    # print (pivots[0])
    pivots =  [ float(x) for x in pivots ]

    # print ("count pivots:", len(pivots))

    pivots = sorted(pivots)
    # print ("pivots:", pivots)
    # print (piv_type)
    piv_clean = {}

    # create supports and resistances in short dataframe

    # print ('first item: ', pivots[0])
    # print (pivots)
    if piv_type == 'sup':
        supports = []
        supports.append(pivots[0])
        for i in range(1, len(pivots)):
            # print ('for: ', i, pivots[i])
            # print ('for: ', i, piv_clean[-1] * gap)
            if pivots[i] >= (supports[-1] * gap):
                # print (pivots[i])
                supports.append(pivots[i])
            # print ('supports: ', supports)
        piv_clean['sup'] = supports
        def set_sup(row):
            # print (row)
            supports = sorted(piv_clean['sup'], reverse=True)
            # print (piv_clean['sup'])
            for sup in supports:
                # print ('sup: ', sup, 'low: ', row['low'])
                # print (row["low"] >= sup * 0.98)
                if row["low"] >= sup:
                    # print ('bingo: ', sup)
                    return sup
        def set_sup2(row):
            # print (row)
            # print (piv_clean['sup'])
            supports = sorted(piv_clean['sup'], reverse=True)
            for sup in supports:
                # print ('sup: ', sup, 'low: ', row['low'])
                # print (row["low"] >= sup * 0.98)
                if row["low"] >= sup and sup < row['s1'] :
                    # print ('bingo: ', sup)
                    return sup
        df = df.assign(s1=df.apply(set_sup, axis=1))
        df = df.assign(s2=df.apply(set_sup2, axis=1))

    # create resistances

    if piv_type == 'res':
        resistances = sorted(pivots, reverse=True)
        resistances.append(pivots[0])
        # print (piv_clean[-1])
        # print (pivots)
        # print ('first item: ', pivots[0])
        # print ('last item: ', pivots[-1])
        for i in range(1, len(pivots)):
            # print('last piv_clean', resistances[-1] * gap)
            # print('current pivot', pivots[i])
            # if i == 1:
                # print ('iter 1', resistances[-i])
            if pivots[i] <= (resistances[-1] * gap):
                resistances.append(pivots[i])
                # print('added', pivots[i])
        piv_clean['res'] = resistances
        def set_res(row):
            res = sorted(piv_clean['res'])
            # res.append(piv_clean['sup'])
            for r in res:
                # print ('res: ', row["s1"] * 1.01, res)
                #  and res >= row["s1"] * 1.02
                if row["high"] <= r and r >= row["s1"] * 1.02:
                    return r
        def set_res2(row):
            res = sorted(piv_clean['res'])
            # res.append(piv_clean['sup'])
            for r in res:
                # print ('res: ', row["s1"] * 1.01, res)
                if row["high"] <= r and row["r1"] < r and r >= row["s1"] * 1.02:
                    return r
        df = df.assign(r1=df.apply(set_res, axis=1))
        df = df.assign(r2=df.apply(set_res2, axis=1))

    return df

def support_resistance(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Pivots Points, first attemp ever.
    """

    w0 = 2
    w1 = 15
    w2 = 25
    w3 = 30
    w4 = 240
    w5 = 300
    w6 = 1000
    w7 = 3000

    shift1 = 1
    shift2 = 5

    df = dataframe.copy()

    prev_high_id = dataframe['high'].rolling(window=5000).max().idxmin()
    prev_low = dataframe['low'].rolling(window=5000).min()

    df['pivot'] = (dataframe['close'].shift(2) > dataframe['close'].shift(1)) & \
                (dataframe['close'].shift(3) > dataframe['close'].shift(2)) & \
                (dataframe['close'].shift(1) < dataframe['close'])
    df['s0'] = dataframe['low'].rolling(window=w0).min().shift()
    df['s1'] = dataframe['low'].rolling(window=w1).min().shift(0)
    df['s2'] = dataframe['low'].rolling(window=w2).min().shift(shift1)
    df['s3'] = dataframe['low'].rolling(window=w3).min().shift(shift1)
    df['s4'] = dataframe['low'].rolling(window=w4).min().shift(shift1)
    df['s5'] = dataframe['low'].rolling(window=w5).min().shift(shift2)
    df['s6'] = dataframe['low'].rolling(window=w6).min().shift(shift2)
    df['s7'] = dataframe['low'].rolling(window=w7).min().shift(shift2)
    df['r0'] = dataframe['high'].rolling(window=w0).max().shift()
    df['r1'] = dataframe['high'].rolling(window=w1).max().shift(0)
    df['r2'] = dataframe['high'].rolling(window=w2).max().shift(shift1)
    df['r3'] = dataframe['high'].rolling(window=w3).max().shift(shift1)
    df['r4'] = dataframe['high'].rolling(window=w4).max().shift(shift1)
    df['r5'] = dataframe['high'].rolling(window=w5).max().shift(shift2)
    df['r6'] = dataframe['high'].rolling(window=w6).max().shift(shift2)
    df['r7'] = dataframe['high'].rolling(window=w7).max().shift(shift2)

    return pd.DataFrame(
        index=df.index,
        data={
            'pivot': df['pivot'],

            # 3 supports
            's0': df['s0'],
            's1': df['s1'],
            's2': df['s2'],
            's3': df['s3'],
            's4': df['s4'],
            's5': df['s5'],
            's6': df['s6'],
            's7': df['s7'],

            # 3 resistances
            'r0': df['r0'],
            'r1': df['r1'],
            'r2': df['r2'],
            'r3': df['r3'],
            'r4': df['r4'],
            'r5': df['r5'],
            'r6': df['r6'],
            'r7': df['r7'],
        }
    )
