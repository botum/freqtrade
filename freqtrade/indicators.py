import numpy as np
import pandas as pd
import logging

from datetime import datetime, timedelta

import freqtrade.vendor.qtpylib.indicators as qtpylib
import freqtrade.persistence

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, String,
                        create_engine)
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.scoping import scoped_session
from sqlalchemy.orm.session import sessionmaker

from sklearn.cluster import MeanShift, estimate_bandwidth

from freqtrade.exchange import get_ticker_history


import scripts.trendy_2 as trendy
from scripts import cactix

from pandas import Series

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
    data1 = full_df.as_matrix(columns=cols)
    try:
        bandwidth1 = estimate_bandwidth(data1, quantile=quantile, n_samples=samples)
        ms1 = MeanShift(bandwidth=bandwidth1, bin_seeding=True)
        ms1.fit(data1)
    except Exception as ex:
        logger.exception('Unexpected error when analyzing ticker pivots for pair %s: %s', pair, str(ex))
        return []  # return False ?

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

    pivots =  [ float(x) for x in pivots ]

    pivots = sorted(pivots)
    piv_clean = {}

    if piv_type == 'sup':
        supports = []
        supports.append(pivots[0])
        for i in range(1, len(pivots)):
            if pivots[i] >= (supports[-1] * gap):
                supports.append(pivots[i])
        piv_clean['sup'] = supports
        def set_sup(row):
            supports = sorted(piv_clean['sup'], reverse=True)
            for sup in supports:
                if row["low"] >= sup:
                    return sup
        def set_sup2(row):
            supports = sorted(piv_clean['sup'], reverse=True)
            for sup in supports:
                if row["low"] >= sup and sup < row['s1'] :
                    return sup
        df = df.assign(s1=df.apply(set_sup, axis=1))
        df = df.assign(s2=df.apply(set_sup2, axis=1))

    # create resistances

    if piv_type == 'res':
        resistances = sorted(pivots, reverse=True)
        resistances.append(pivots[0])
        for i in range(1, len(pivots)):
            if pivots[i] <= (resistances[-1] * gap):
                resistances.append(pivots[i])
        piv_clean['res'] = resistances
        def set_res(row):
            res = sorted(piv_clean['res'])
            for r in res:
                if row["high"] <= r and r >= row["s1"] * 1.02:
                    return r
        def set_res2(row):
            res = sorted(piv_clean['res'])
            for r in res:
                if row["high"] <= r and row["r1"] < r and r >= row["s1"] * 1.02:
                    return r
        df = df.assign(r1=df.apply(set_res, axis=1))
        df = df.assign(r2=df.apply(set_res2, axis=1))

    return df
