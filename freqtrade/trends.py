import copy
import sys
from typing import Dict, List, Tuple
from pandas import DataFrame, to_datetime, Series

from datetime import datetime


from scipy import linspace, polyval, polyfit, sqrt, stats, randn
from pylab import plot, title, show , legend

from freqtrade import (DependencyException, OperationalException, exchange, persistence)
from freqtrade.configuration import Configuration
# from freqtrade.persistence import Trade, Pair, Trend
from freqtrade.indicators import in_range
# import operator

import numpy as np
import matplotlib.pyplot as plt

# ZigZag

# This is inside your IPython Notebook
import pyximport
pyximport.install(reload_support=True)
from freqtrade.vendor.zigzag_hi_lo import *
# from zigzag import *

from matplotlib.pyplot import plot, grid, show, savefig

def plot_pivots(X, L, H, pivots):
    plt.plot(np.arange(len(X)), X, 'k:', alpha=0.2)
    plt.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
    plt.scatter(np.arange(len(X))[pivots == 1], H[pivots == 1], color='r', alpha=0.3)
    plt.scatter(np.arange(len(X))[pivots == -1], L[pivots == -1], color='g', alpha=0.3)
#     plt.show()
    pass

def plot_trends(df, filename: str=None):
    plt.figure(num=0, figsize=(20,10))
    plt.xlim(0, len(df.close))
    plt.ylim(df.low.min()*0.99, df.high.max()*1.01)

    plt.plot(df.index, df['rt'], 'r', label='resistance trend', linewidth=2)
    plt.plot(df.index, df['st'], 'g', label='support trend', linewidth=2)

    plt.plot(df.high, 'r', alpha=0.5)
    plt.plot(df.close, 'k', alpha=0.5)
    plt.plot(df.low, 'g', alpha=0.5)

    plt.plot(df.bb_lowerband, 'b', alpha=0.5, linewidth=2)
    plt.plot(df.bb_upperband, 'b', alpha=0.5, linewidth=2)

    pivot = [col for col in df if col.startswith('trend-')]
    # support = df[df['ids'].str.contains('ball', na = False)]
    # print(pivot)
    # print (pivots['sup'])
    for piv in pivot:
        plt.plot(df.index, df[piv], 'k', label='support trend', alpha=0.5, linewidth=1)

    # macd = go.Scattergl(x=data['date'], y=data['macd'], name='MACD')
    # macdsignal = go.Scattergl(x=data['date'], y=data['macdsignal'], name='MACD signal')
    # volume = go.Bar(x=data['date'], y=data['volume'], name='Volume')

    plot_pivots(df.close.values, df.low.values, df.high.values, df.pivots.values)

    plt.figure(num=0, figsize=(20,10))
    plt.xlim(0, len(df.close))
    plt.ylim(df.low.min()*0.99, df.high.max()*1.01)

    # print(pair)
    if not filename:
        filename = 'chart_plots/' + 'UNKNOWN-PAIR' + datetime.utcnow().strftime('-%m-%d-%Y-%H') + str(len(df)) + '.png'
    # print('saving file: ', filename)
    plt.savefig(filename)
    plt.close()
    # plot_pivots(df.close.values, df.low.values, df.high.values, pivots)
    # legend(['pivots','trend', 'close'])
    # plt.show()

def get_tests(df, trend_name, pt, first):

    trend = df[trend_name]

    if pt == 'res':
        tolerance = 0.001
        t_r = df.loc[df['pivots']==1 & in_range(df['high'],trend, tolerance)]
    if pt == 'sup':
        tolerance = 0.001
        t_r = df.loc[df['pivots']==-1 & in_range(df['low'],trend, tolerance)]

    trend_tests = len(t_r)
#     trends['trend'].append(trend)
#     print(trend_name)

    return trend_tests

def gentrends(self, df, charts=False, pair='default_filename_plot'):
    # config = Configuration.get_config(self)
    # persistence.init(config)
    #
    # current_trends = Trend.query.filter(Trend.pair.is_(pair)).all()
    # print ('current: ', current_trends)

    h = df.loc[df['pivots']==1]
    l = df.loc[df['pivots']==-1]

    df_orig = df

    print (pair)
    print (len(df))
    print (len(h))
    print (len(l))

    global trends
    trends = list()

    id_max = h[:-1].high.values.argmax()
    id_min = l[:-1].low.values.argmin()

    for i in range(0, len(h) -1):

        ax = h.index[i]
        ay = h.iloc[i].high
        bx = h.index[i+1]
        by = h.iloc[i+1].high
        t = df.index[ax:]
        # print (h)
        # print (id_max, len(h))


        slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
        trend_name = 't_r|'+str(ax)+'|'+str(ay)+'|'+str(bx)+'|'+str(by)
        trend = polyval([slope,intercept],t)
        df.loc[h.index[i]:,trend_name] = trend
        # plt.plot(df.index, df[trend_name], 'm', label='misterious', alpha=0.1)
#         print (df)

        # print(trends)

        next_waves = h[i+2:]
        is_last = len(next_waves[i:])==1
        trend_tests = get_tests(df, trend_name, 'res', True)
        # print (trend_name, '- A: ', ay, 'B: ', by, '-------------------------------------------------')

        trend = {'name':trend_name,
                'timeframe':'all',
                'a':[ax, ay, df.iloc[ax].date],
                'b':[bx, by, df.iloc[bx].date],
                'slope':slope,
                'conf_n':trend_tests,
                'type':'res'}
        trends.append(trend)
        # trend_obj = Trend(
        #     pair=pair,
        #     type = 'res',
        #     timeframe = '1m',
        #     a = (df.iloc[ax].date, ay),
        #     b = (df.iloc[bx].date, by),
        #     slope = slope,
        #     conf_n = trend_tests
        # )
        # Trend.session.add(trend_obj)
        # Trend.session.flush()


#         next_waves = df.loc[df['pivots']==1][i+1:]
        for ib in range(0, len(next_waves)):
#           print ('point A: ', ax, 'b: ', ay, '-------------------------------------------------')
            bx = next_waves.index[ib]
            by = next_waves.iloc[ib].high

#             print(df[df.high > df ])

#             print ('\t\tpoint B: ', bx, 'c: ', next_waves.iloc[ib].high, 'trend_point: ', df.loc[bx][trend_name])

            if by > df.loc[bx][trend_name]:
                t = df.index[h.index[i]:]
                slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
                trend_next_wave = polyval([slope,intercept],t)
                trend_name = 't_r|'+str(ax)+'|'+str(ay)+'|'+str(bx)+'|'+str(by)
                df.loc[h.index[i]:,trend_name] = trend_next_wave
#                 plt.plot(t, trend_next_wave, 'r', label=trend_name, alpha=0.3)
                trend_tests = get_tests(df, trend_name, 'res', False)
                trend = {'name':trend_name,
                        'timeframe':'all',
                        'a':[ax, ay, df.iloc[ax].date],
                        'b':[bx, by, df.iloc[bx].date],
                        'slope':slope,
                        'conf_n':trend_tests,
                        'type':'res'}
                trends.append(trend)

            trends[-1]['last'] = True

        if i == id_max:
            df_orig['rt'] = df[trend_name]

            trends[-1]['max'] = True

            # trend_obj = Trend(
            #     pair=pair,
            #     type = 'res',
            #     timeframe = '1m',
            #     a = (df.iloc[ax].date, ay),
            #     b = (df.iloc[bx].date, by),
            #     slope = slope,
            #     conf_n = trend_tests
            # )
            # Trend.session.add(trend_obj)
            # Trend.session.flush()


    for i in range(0, len(l) -1):

        ax = l.index[i]
        ay = l.iloc[i].low
#         print ('a', ax, ay)
        bx = l.index[i+1]
        by = l.iloc[i+1].low
        t = df.index[ax:]

        slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
        trend_l = polyval([slope,intercept],t)
        # plt.plot(t, trend, 'm', label='fitted line', alpha=0.5)
        trend_name = 't_s|'+str(ax)+'|'+str(ay)+'|'+str(bx)+'|'+str(by)
        df.loc[l.index[i]:,trend_name] = trend_l
        # plt.plot(df.index, df[trend_name], 'm', label='misterious', alpha=0.2)

        # plt.scatter(ax, ay, color='r')
        # plt.scatter(bx, by, color='r')
        # print (trend_name)
        next_waves = l[i+2:]
        is_last = len(next_waves[i:])==1
        trend_tests = get_tests(df, trend_name, 'sup', True)


        trend = {
            'name':trend_name,
            'timeframe':'all',
            'a':[ax, ay, df.iloc[ax].date],
            'b':[bx, by, df.iloc[bx].date],
            'slope':slope,
            'conf_n':trend_tests,
            'type':'sup'}
        trends.append(trend)
        # trend_obj = Trend(
        #     pair=pair,
        #     type = 'sup',
        #     timeframe = '1m',
        #     a = (df.iloc[ax].date, ay),
        #     b = (df.iloc[bx].date, by),
        #     slope = slope,
        #     conf_n = trend_tests
        # )
        # Trend.session.add(trend_obj)
        # Trend.session.flush()

    #     tolerance = 0.001
    #     test_r = df.loc[df['pivots']==-1 & in_range(df['low'],df[trend_name], tolerance)]
    #     trend_tests = len(test_r)
    # #     trends['trend'].append(trend)
    # #     print(trend_name)
    #     trends['name'].append(trend_name)
    #     trends['tests'].append(trend_tests)
    #     trends['type'].append('sup')

#         if len(t_r)-1 > tests:
#             plt.plot(t, trend, 'g', label=trend_name)
#             for it in range(0, len(t_r)):
#                 plt.scatter(t_r.index[it], t_r.iloc[it].low, color='c')

        for ib in range(0, len(next_waves)):
            bx = next_waves.index[ib]
            by = next_waves.iloc[ib].low
            if by < df.loc[bx][trend_name]:
                t = df.index[l.index[i]:]
                slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
                trend_next_wave = polyval([slope,intercept],t)
                trend_name = 't_s|'+str(ax)+'|'+str(ay)+'|'+str(bx)+'|'+str(by)
                df.loc[l.index[i]:,trend_name] = trend_next_wave
                trend_tests = get_tests(df, trend_name, 'sup', False)
                trend = {
                    'name':trend_name,
                    'timeframe':'all',
                    'a':[ax, ay, df.iloc[ax].date],
                    'b':[bx, by, df.iloc[bx].date],
                    'slope':slope,
                    'conf_n':trend_tests,
                    'type':'sup',
                    'last':True}
                trends.append(trend)

#                 plt.plot(t, trend_next_wave, 'g', label='fitted line', alpha=0.3)

        trends[-1]['last'] = True

        if i == id_min:
            # print ('sup: ', trend_name)
            df_orig['st'] = df[trend_name]
            # trends['last'][-1] = 1
            trends[-1]['min'] = True

            # trend_obj = Trend(
            #     pair=pair,
            #     type = 'sup',
            #     timeframe = '1m',
            #     a = (df.iloc[ax].date, ay),
            #     b = (df.iloc[bx].date, by),
            #     slope = slope,
            #     conf_n = trend_tests
            # )
            # Trend.session.add(trend_obj)
            # Trend.session.flush()

    # trends_df = DataFrame(data=trends).sort_values('tests', ascending=False)
    # print (trends['type'] == 'res')
#     above_mean = trends.loc[trends.tests > trends.tests.mean()][:10]
#     print(trends.tests)
    # print (trends)
    # for i, row in trends_df.iterrows():
    #     if row['last']:
    #         df_orig[row['name']] = df[row['name']]

    # if charts:
    #     filename = 'chart_plots/' + pair.replace('/', '-') + datetime.utcnow().strftime('-%m-%d-%Y-%H') + '.png'
    #     plot_trends(df_orig, filename)




    #     for i, row in trends_df.iterrows():
    # #         print (row)
    # #         print('row name ', row.tests)
    #         # linewidth = row['tests'] / max_tests
    #         linewidth = 1
    #         colour = 'k'
    #         # if row['first']:
    #         #     linewidth = 1
    #         #     colour = 'm'
    #         # if row['last']:
    #         #     linewidth = 1
    #         #     colour = 'b'
    #
    #         # print (linewidth)
    #         plt.plot(df_orig.index, df_orig[row['name']], colour, label=row['name'], linewidth=linewidth)

    return trends

# from math import sqrt


#     timeframe_volat = {
#                     '1d':0.02,
#                     '1h':0.015,
#                     '5m':0.008,
#                     '1m':0.005}

# timeframe_volat = {
#                 '1d':1.1,
#                 '1h':0.0009,
#                 '5m':4,
#                 '1m':4}

# volat_window = {
#                 '1d':1,
#                 '1h':2,
#                 '5m':10,
#                 '1m':30}
# Plots

# def plot_pivots(X, L, H, pivots):
#     plt.plot(np.arange(len(X)), X, 'k:', alpha=0.2)
#     plt.plot(np.arange(len(X))[pivots != 0], X[pivots != 0], 'k-')
#     plt.scatter(np.arange(len(X))[pivots == 1], H[pivots == 1], color='r', alpha=0.3)
#     plt.scatter(np.arange(len(X))[pivots == -1], L[pivots == -1], color='g', alpha=0.3)
# #     plt.show()
#     pass

# for t in timeframes:

#     df_long = dfs[t]
#     timeframe = 2000
#     df = df_long
# #     volat = sqrt(df['stddev'].mean()) * timeframe_volat[t]
# #     print (volat)
# #     volat = timeframe_volat[t]
# #     df['bb_exp'] = (df.bb_upperband - df.bb_lowerband) / df.bb_lowerband
#     window = volat_window[t]
#     df['bb_exp'] = (df.bb_upperband.rolling(window=window).max() - df.bb_lowerband.rolling(window=window).min()) / df.bb_upperband.rolling(window=window).max()
#     pivots = peak_valley_pivots(df.low.values, df.high.values, df.bb_exp.values)
#     df['pivots'] = np.transpose(np.array((pivots)))

# #     df['pct_change'] = df.close.pct_change()
# #     df['bb_upperband'] = np.log(df.bb_upperband)
# #     df['bb_middleband'] = np.log(df.bb_middleband)
# #     df['bb_lowerband'] = np.log(df.bb_lowerband)

# #     print (df.bb_exp)

# #     print(df)
#     # eje y en log scale


# #     plt.yscale('log')



#     df['high'] = np.log(df.high)
#     df['close'] = np.log(df.close)
#     df['open'] = np.log(df.open)
#     df['low'] = np.log(df.low)

#     plt.figure(num=0, figsize=(50,30))
# #     plt.xlim(0, len(df.close))
# #     plt.ylim(df.low.min()*0.99, df.high.max()*1.01)


#     gentrends(df)
#     plt.plot(df.high, 'r', alpha=0.5)
#     plt.plot(df.close, 'k', alpha=0.5)
#     plt.plot(df.low, 'g', alpha=0.5)

#     plot_pivots(df.close.values, df.low.values, df.high.values, pivots)
#     legend(['pivots','trend', 'close'])
#     plt.show()
