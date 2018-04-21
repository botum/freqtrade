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
# from freqtrade.persistence import Pair

from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, String,
                        create_engine)
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.scoping import scoped_session
from sqlalchemy.orm.session import sessionmaker

from sklearn.cluster import MeanShift, estimate_bandwidth

from freqtrade.exchange import get_ticker_history

# from freqtrade.analyze import Analyze

import scripts.cactix as trendy

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

def get_trend_lines(live_df: pd.DataFrame, pair: str, timerange: int=600, interval: str="1h", charts: bool=False) -> pd.DataFrame:
    # trend_range = len(dataframe)
    # segments = 3
    # x_maxima, maxima, x_minima, minima = trendy.segtrends(df['close'][:trend_range], segments = segments)

    # ticker_hist = get_ticker_history(pair, interval)
    # if not ticker_hist:
    #     logger.warning('Empty ticker history for pair %s', pair)
    #     return []  # return False ?
    #
    # try:
    #     dataframe = Analyze.parse_ticker_dataframe(ticker_hist)
    # except ValueError as ex:
    #     logger.warning('Unable to analyze ticker for pair %s: %s', pair, str(ex))
    #     return []  # return False ?
    # except Exception as ex:
    #     logger.exception('Unexpected error when analyzing ticker for pair %s: %s', pair, str(ex))
    #     return []  # return False ?
    #
    # window = len(dataframe)
    main_trends, main_maxslope, main_minslope = trendy.gentrends(live_df['close'], window=1/2, charts=charts)


#     df = live_df['close']
#     print (len(dataframe))
    # print (df)
    # window = len(df)

    # pivots = peak_valley_pivots(live_df.close.values, 0.05, -0)
    # ts_pivots = pd.Series(live_df, index=live_df.index)
    # ts_pivots['close'] = pivots
    # print ('len_pivots: ', len(ts_pivots))
    # print ('pivots: ', ts_pivots)

    # short_trends, short_maxslope, short_minslope = trendy.gentrends(dataframe, window=1/2, charts=charts, pair=pair)

    # trends_x_max, trends_max, trends_x_min, trends_min = trendy.segtrends(df.close, segments = 10, charts = True)
    # maxline = np.linspace(a_max, b_max, len(x))  # Y values between max's
    # minline = np.linspace(a_min, b_min, len(x))  # Y values between min's
    #
    # trends = np.transpose(np.array((df['close'], maxline, minline)))
    # trends = pd.DataFrame(trends, index=np.arange(0, len(df)),
    #                       columns=['Data', 'Max Line', 'Min Line'])
    # len_data = len(live_df)
    # new_index = list(range(len_data-timerange, len_data))
    # print (new_index)
    # print (len(new_index))
    # print (len(short_trends))
    # short_trends.set_index([new_index])
    # print (short_trends)
    # x_maxima, maxima, x_minima, minima = trendy.segtrends(df, segments = 3)

    # dataframe = dataframe.join(trends)


    # trends = pd.Series(df, index=df.index)
    # trends = trends[pivots != 0]

    return main_trends['Max Line'], main_trends['Min Line'], main_maxslope, \
            main_minslope
            # short_trends['Max Line'], short_trends['Min Line'], \
            # short_maxslope, short_minslope

def bruno_pivots(pair: str, interval: int=1, piv_type: str='piv') -> pd.DataFrame:

    if piv_type == 'sup':
        # df = df[df['low'].value_counts()[df['low']] >= 3]
        df = df[(df['volume'] > df['volume'].rolling(window=10).mean() & df['close'] < df['open'])]
    elif piv_type == 'res':
        df = df[(df['volume'] > df['volume'].rolling(window=10).mean() & df['close'] > df['open'])]

def get_pivots(df: pd.DataFrame, pair: str, interval: int=1, piv_type: str='piv') -> pd.DataFrame:
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

    # print ('len df: ', len(df))
    samples = len(df)

    # print(len(df) * quantile)
    if df.empty:
        logger.warning('Empty dataframe for pair %s', pair)
        return []  # return False ?
    elif not len(df) * quantile > 1:
        print('dataframe too short: ', len(df))
        samples = len(df) * 0.1
    # df = df[(df['volume'] > df['volume'].rolling(window=10).mean())]


    data1 = df.as_matrix(columns=cols)
    # highest = df.high.rolling(window=len(df)).max()
    # lowest = df.low.rolling(window=len(df)).min()

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

    # create supports

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

    # print (pivots)



    # ticker_hist = get_ticker_history(pair, 60)
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
    # if dataframe.empty:
    #     logger.warning('Empty dataframe for pair %s', pair)
    #     return []  # return False ?
    # elif len(dataframe) < 50:
    #     return []
    # df = dataframe
    # # df = df[(df['volume'] > df['volume'].mean())]
    # data1 = df.as_matrix(columns=['low', 'high'])
    # # highest = df.high.rolling(window=len(df)).max()
    # # lowest = df.low.rolling(window=len(df)).min()
    #
    # bandwidth1 = estimate_bandwidth(data1, quantile=0.01, n_samples=len(df))
    # ms1 = MeanShift(bandwidth=bandwidth1, bin_seeding=True)
    # ms1.fit(data1)
    #
    # #Calculate Support/Resistance
    # # pivots = []
    # # print ('labels', ms1.labels_)
    # for k in range(len(np.unique(ms1.labels_))):
    #         my_members = ms1.labels_ == k
    #         values = data1[my_members, 0]
    #         # print (values)
    #
    #         # find the edges
    #         if len(values) > 0:
    #             pivots.append(min(values))
    #             pivots.append(max(values))


    #Adjust Calculation Set ; data2 size
    # df = dataframe
    #
    #
    # df = df[(df['volume'] > df['volume'].mean())]
    # data1 = df.as_matrix(columns=['high', 'low'])
    # bandwidth1 = estimate_bandwidth(data1, quantile=0.05, n_samples=len(df))
    # ms1 = MeanShift(bandwidth=bandwidth1, bin_seeding=True)
    # ms1.fit(data1)
    #
    # # print ('labels', ms1.labels_)
    # for k in range(len(np.unique(ms1.labels_))):
    #         my_members = ms1.labels_ == k
    #         values = data1[my_members, 0]
    #         # print (values)
    #
    #         # find the edges
    #         if len(values) > 0:
    #             pivots.append(min(values))
    #             pivots.append(max(values))

    # print (pivots)

    return df

def find_support_resistance(dataframe: pd.DataFrame, quantile: int, samples: int) -> pd.DataFrame:

    #Adjust Calculation Set ; data2 size
    df = dataframe
    data = dataframe.as_matrix(columns=['high','low'])
    # data2 = data[:len(data)*1]

    # Data = All Data ; Data2 = Adjusted DataSet
    bandwidth = estimate_bandwidth(data, quantile=quantile, n_samples=samples)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    # fit the data
    ms.fit(data)
    #Calculate Support/Resistance
    pivots = []
    # print ('labels', ms.labels_)
    for k in range(len(np.unique(ms.labels_))):
            my_members = ms.labels_ == k
            values = data[my_members, 0]
            # print (values)

            # find the edges
            if len(values) > 0:
                # pivots.append(min(values))
                pivots.append(max(values))

    # def set_sup(row):
    #     for sup in pivots:
    #         if row["close"] > sup:
    #             return sup
    # def set_res(row):
    #     for resistence in pivots:
    #         if row["close"] < resistence:
    #             return resistence
    #
    # df = df.assign(p1=df.apply(set_sup, axis=1))
    # df = df.assign(r1=df.apply(set_res, axis=1))

    # print(df)
    # print (pivots)
    return pivots


def support_resistance(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Pivots Points
    :param dataframe:
    :return: dataframe
    """
    # Todo: Use a window of many tickers

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



def tip_dip(dataframe: pd.DataFrame, timeperiod=40, levels=3, min_periods=1) -> pd.DataFrame:
    """
    Pivots Points
    Formula:
    Pivot = (Previous High + Previous Low + Previous Close)/3
    Resistance #1 = (2 x Pivot) - Previous Low
    Support #1 = (2 x Pivot) - Previous High
    Resistance #2 = (Pivot - Support #1) + Resistance #1
    Support #2 = Pivot - (Resistance #1 - Support #1)
    Resistance #3 = (Pivot - Support #2) + Resistance #2
    Support #3 = Pivot - (Resistance #2 - Support #2)
    ...
    :param dataframe:
    :param timeperiod: Period to compare (in ticker)
    :param levels: Num of support/resistance desired
    :return: dataframe
    """

    data = {}



    dataframe.loc[
                (
                     # (dataframe['high'] >= dataframe['high'].shift(1)) &
                     # (dataframe['high'] > dataframe['high'].shift(-1)) &
                     (dataframe['high'] >= (dataframe['low'].rolling(window=timeperiod).min() + dataframe['low'].rolling(window=timeperiod).min() * 0.008))
                     # (dataframe['volume'] > dataframe['volume'].rolling(window=timeperiod, min_periods = min_periods).mean() * 2)

                ),
               'tip'] = 1


    dataframe.loc[
                (
                    # (dataframe['close'] == dataframe['low']) &
                     # (dataframe['close'] <= (dataframe['low'] + dataframe['low'] * 0.005)) &
                     # (dataframe['close'] == dataframe['low']) &
                     # (dataframe['low'] < dataframe['low'].shift(-1))
                     # &
                     (dataframe['low'] <= (dataframe['high'].rolling(window=timeperiod).max() - dataframe['high'].rolling(window=timeperiod).max() * 0.05))
                     &
                     # (dataframe['close'] <= (dataframe['low'].rolling(window=1000).min() + dataframe['low'].rolling(window=1000).min() * 0.1))
                     # &
                     # (dataframe['low'] <= (dataframe['high'].rolling(window=100).max() - dataframe['high'].rolling(window=100).max() * 0.1)) &
                     # (dataframe['low'] <= (dataframe['high'].rolling(window=15).max() - dataframe['high'].rolling(window=15).max() * 0.02))
                     # &
                     (dataframe['volume'] > (dataframe['volume'].rolling(window=timeperiod).mean() * 1.0))

                ),
               'dip'] = 1


    return pd.DataFrame(
        index=dataframe.index,
        data=dataframe
    )

def s_r(dataframe: pd.DataFrame, timeperiod=30, levels=3, min_periods=1) -> pd.DataFrame:
    """
    Pivots Points
    Formula:
    Pivot = (Previous High + Previous Low + Previous Close)/3
    Resistance #1 = (2 x Pivot) - Previous Low
    Support #1 = (2 x Pivot) - Previous High
    Resistance #2 = (Pivot - Support #1) + Resistance #1
    Support #2 = Pivot - (Resistance #1 - Support #1)
    Resistance #3 = (Pivot - Support #2) + Resistance #2
    Support #3 = Pivot - (Resistance #2 - Support #2)
    ...
    :param dataframe:
    :param timeperiod: Period to compare (in ticker)
    :param levels: Num of support/resistance desired
    :return: dataframe
    """
    dataframe.loc[
                (
                     (dataframe['low'].rolling(window=1000, min_periods=5).min() == dataframe['low'])
                     &
                     (dataframe['volume'] > (dataframe['volume'].rolling(window=5).mean()))
                ),
               's1'] = 1

    dataframe.loc[
               (
                    dataframe['high'].rolling(window=100, min_periods=2).max() == dataframe['high']
               ),
              'r1'] = 1

    dataframe.loc[
                (
                     (dataframe['high'] >= dataframe['high'].shift(1)) &
                     # (dataframe['high'] > dataframe['high'].shift(-1)) &
                     (dataframe['close'] >= (dataframe['low'].rolling(window=timeperiod).min() + dataframe['low'].rolling(window=timeperiod).min() * 0.05))
                     # (dataframe['volume'] > dataframe['volume'].rolling(window=timeperiod, min_periods = min_periods).mean() * 2)

                ),
               'profit'] = 1


    return pd.DataFrame(
        index=dataframe.index,
        data=dataframe
    )


def sure(df: pd.DataFrame):
    #Adjust Calculation Set ; data2 size
    # print (df.head())
    data = df.as_matrix(columns=['Adj Close'])
    data2 = data[:len(data)*1]

    # Data = All Data ; Data2 = Adjusted DataSet
    bandwidth = estimate_bandwidth(data2, quantile=0.1, n_samples=100)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

    # fit the data
    ms.fit(data2)

    #Calculate Support/Resistance
    ml_results = []
    for k in range(len(np.unique(ms.labels_))):
            my_members = ms.labels_ == k
            values = data[my_members, 0]
            #print values

            # find the edges
            ml_results.append(min(values))
            ml_results.append(max(values))
    # print (ml_results.head())
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from zigzag import *

def peaks(dataframe: pd.DataFrame, timeperiod=30, levels=3, min_periods=1) -> pd.DataFrame:
    """
    Pivots Points
    Formula:
    Pivot = (Previous High + Previous Low + Previous Close)/3
    Resistance #1 = (2 x Pivot) - Previous Low
    Support #1 = (2 x Pivot) - Previous High
    Resistance #2 = (Pivot - Support #1) + Resistance #1
    Support #2 = Pivot - (Resistance #1 - Support #1)
    Resistance #3 = (Pivot - Support #2) + Resistance #2
    Support #3 = Pivot - (Resistance #2 - Support #2)
    ...
    :param dataframe:
    :param timeperiod: Period to compare (in ticker)
    :param levels: Num of support/resistance desired
    :return: dataframe
    """


    # np.random.seed(1997)
    # X = np.cumprod(1 + np.random.randn(100) * 0.01)
    pivots = peak_valley_pivots(dataframe.close, 0.03, -0.03)

    # dataframe['peak'] = peakdetect(dataframe['high'], lookahead=100)

    # def plot_pivots(X, pivots):
    #     plt.xlim(0, len(X))
    #     plt.ylim(X.min()*0.99, X.max()*1.01)
    #     plt.plot(np.arange(len(X)), X, 'k:', alpha=0.5)
    #     plt.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
    #     plt.scatter(np.arange(len(X))[pivots == 1], X[pivots == 1], color='g')
    #     plt.scatter(np.arange(len(X))[pivots == -1], X[pivots == -1], color='r')
    #     plot_pivots(X, pivots)
    # indices = peakutils.indexes(df[''], thres=df['low']*10, min_dist=1000)
    # plot_pivots(X, pivots)


    return pd.DataFrame(
        index=dataframe.index,
        data=pivots
    )


    #
    # # Pivot
    # data['pivot'] = qtpylib.rolling_mean(
    #     series=qtpylib.typical_price(dataframe),
    #     window=timeperiod
    # )
    #
    # # Resistance #1
    # data['r1'] = (2 * data['pivot']) - low
    #
    # # Resistance #2
    # data['s1'] = (2 * data['pivot']) - high
    #
    # # Calculate Resistances and Supports >1
    # for i in range(2, levels+1):
    #     prev_support = data['s' + str(i - 1)]
    #     prev_resistance = data['r' + str(i - 1)]
    #
    #     # Resitance
    #     data['r'+ str(i)] = (data['pivot'] - prev_support) + prev_resistance
    #
    #     # Support
    #     data['s' + str(i)] = data['pivot'] - (prev_resistance - prev_support)

    # df = dataframe
    #
    # time_series = df['close']
    # indices = peakutils.indexes(df['low'], thres=df['low']*10, min_dist=1000)
    #
    # trace = go.Scatter(
    #     x=[j for j in range(len(time_series))],
    #     y=time_series,
    #     mode='lines',
    #     name='Original Plot'
    # )
    #
    # trace2 = go.Scatter(
    #     x=indices,
    #     y=[time_series[j] for j in indices],
    #     mode='markers',
    #     marker=dict(
    #         size=8,
    #         color='rgb(255,0,0)',
    #         symbol='cross'
    #     ),
    #     name='Detected Peaks'
    # )
    # # print (trace, trace2)
    # data = [trace, trace2]
    # plot(data, filename='milk-production-plot-with-peaks')
    #
    # return pd.DataFrame(
    #     index=df.index,
    #     data=data
    # )
'''
