
# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from hyperopt import hp
from functools import reduce
from pandas import DataFrame
import pandas as pd

# --------------------------------

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from freqtrade.indicators import sure


# Update this variable if you change the class name
class_name = 'CactiX'


"""
Indicators for Freqtrade
author@: Gerald Lonlas
github@: https://github.com/glonlas/freqtrade-strategies
"""



class sr001(IStrategy):

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    # minimal_roi = {
    #    '30':   0.001,
    #    '10':   0.005,
    #    '5':   0.01,
    #    '3':   0.02,
    #    '0':   0.03
    # }
    minimal_roi = {
       '0': 0.01
    }

    # minimal_roi = {
    #     "40": 0.001,
    #     "30": 0.01,
    #     "20": 0.05,
    #     "10": 0.06
    # }
    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.5

    # Optimal ticker interval for the strategy
    ticker_interval = 1

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        #
        sr = sure(dataframe.dropna())
        dataframe['iv'] = qtpylib.implied_volatility(dataframe['close'])
        # dataframe['adx'] = ta.ADX(dataframe)
        # # dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)
        # dataframe['cci'] = ta.CCI(dataframe)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['mfi'] = ta.MFI(dataframe)
        # dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        # dataframe['minus_di'] = ta.MINUS_DI(dataframe)
        # dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        # dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        # dataframe['minus_di'] = ta.MINUS_DI(dataframe)
        # dataframe['roc'] = ta.ROC(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe)
        # # rsi = 0.1 * (dataframe['rsi'] - 50)
        # # dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)
        # # dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)
        # # stoch = ta.STOCH(dataframe)
        # # dataframe['slowd'] = stoch['slowd']
        # # dataframe['slowk'] = stoch['slowk']
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        # # stoch_rsi = ta.STOCHRSI(dataframe)
        # # dataframe['fastd_rsi'] = stoch_rsi['fastd']
        # # dataframe['fastk_rsi'] = stoch_rsi['fastk']
        # dataframe['fastminus_dm'] = ta.MINUS_DM(dataframe, timeperiod=1)
        # dataframe['longminus_dm'] = ta.MINUS_DM(dataframe, timeperiod=60)
        # dataframe['longlongminus_dm'] = ta.MINUS_DM(dataframe, timeperiod=200)
        # dataframe['fastplus_dm'] = ta.PLUS_DM(dataframe, timeperiod=1)
        # dataframe['longplus_dm'] = ta.PLUS_DM(dataframe, timeperiod=60)
        # dataframe['longlongplus_dm'] = ta.PLUS_DM(dataframe, timeperiod=200)
        # dataframe['direction'] = dataframe['plus_dm'] - dataframe['minus_dm']
        # dataframe['fastdirection'] = dataframe['fastplus_dm'] - dataframe['fastminus_dm']
        # dataframe['longdirection'] = dataframe['longplus_dm'] - dataframe['longminus_dm']
        # dataframe['longlongdirection'] = dataframe['longlongplus_dm'] - dataframe['longlongminus_dm']
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        # # dataframe['sar'] = ta.SAR(dataframe)
        # # dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        # hc31 = DataFrame(dataframe['close'])
        # hc32 = DataFrame(dataframe['high']).rename(columns={'high': 'close'})
        # hc33 = DataFrame(dataframe['low']).rename(columns={'low': 'close'})
        # ap = (hc31 + hc32 + hc33) / 3
        #
        # dataframe['esa'] = ta.EMA(ap, 17)
        #
        # esaframe = DataFrame(dataframe['esa']).rename(columns={'esa': 'close'})
        #
        # d1 = abs(ap - esaframe)
        # dataframe['d'] = ta.EMA(abs(d1), 17)
        # dframe = DataFrame(dataframe['d']).rename(columns={'d': 'close'})
        # ci = d1 / (0.015 * dframe)
        # dataframe['tci'] = ta.EMA(ci, 6)
        #
        # dataframe['wt1'] = dataframe['tci']
        # wt1frame = DataFrame(dataframe['wt1']).rename(columns={'wt1': 'close'})
        # dataframe['wt2'] = ta.SMA(wt1frame, 4)



        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        dataframe.loc[
           (
                # (qtpylib.crossed_below(dataframe['tema'],dataframe['bb_lowerband']))
               (dataframe['close'] < dataframe['bb_lowerband'] - (dataframe['bb_lowerband'] * 0.01))
               &
               # (dataframe['bb_upperband'] < dataframe['bb_upperband'].shift(2))
               # &
               #
               # # &
               # (dataframe['open'].shift(10) - (dataframe['open'].shift(10) * 0.04)
               # > dataframe['close'])
               # &
               # (dataframe['open'].shift(1) <= dataframe['close'].shift(1))
               #  &
                # (dataframe['open'].shift(2) <= dataframe['close'].shift(2))
                #  &
                (dataframe['volume'] > (dataframe['volume'].rolling(window=10).mean() * 1.02))
               # (dataframe['open'].shift(2) > dataframe['close'].shift(2)) &
               # (dataframe['mfi'] <= dataframe['mfi'].shift(1)) &
               # (dataframe['mfi'] < 10)
               # &
               # (dataframe['rsi'] < 5)
           ),
           'buy'] = 1
        # dataframe.loc[
        #   (
        #       # (dataframe['ema5'] < dataframe['ema10']) &
        #       # (dataframe['ema5'] < dataframe['ema10']) &
        #       (dataframe['open'].shift(1) > (dataframe['close'].shift(1) * 1.03))
        #   ),
        #   'buy'] = 1
        # dataframe.loc[
        #    (
        #    qtpylib.crossed_below(dataframe['wt2'], dataframe['wt1'])
        #    ),
        #    'buy'] = 1
        #Very good, works really well.
        #BTFD
        # dataframe.loc[
        #    (
        #        (dataframe['rsi'] < 50) &
        #        (dataframe['fastd'] < 50) &
        #        (dataframe['fastk'] < 50) &
        #        (dataframe['direction'].shift(3) < 0) &
        #        (dataframe['adx'].shift(3) > 20) &
        #        (dataframe['close'] > dataframe['open']) &
        #        (dataframe['close'].shift(1) > dataframe['open'].shift(1)) &
        #        (dataframe['close'].shift(2) > dataframe['open'].shift(2)) &
        #        (dataframe['close'] > dataframe['tema'])
        #    ),
        #    'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (

                # (dataframe['close'] < dataframe['open']) & (qtpylib.crossed_above(dataframe['open'],dataframe['bb_middleband']))
                # |
                # (dataframe['close'] < dataframe['open']) & (qtpylib.crossed_above(dataframe['close'],dataframe['bb_middleband']))
                # |
                # (qtpylib.crossed_above(dataframe['close'],dataframe['bb_upperband']))
                # |
                # (qtpylib.crossed_above(dataframe['close'],dataframe['bb_upperband']))
                # |
                # (qtpylib.crossed_below(dataframe['tema'],dataframe['bb_upperband']))
                # (
                #     (dataframe['high'].shift(1) >= dataframe['bb_upperband'].shift(1))
                #     &
                #     (dataframe['high'] < dataframe['bb_upperband'])
                # )
                # |
                # (
                # (dataframe['high'] >
                # dataframe['bb_upperband'] * 1.005)
                # )
            # qtpylib.crossed_below(dataframe['ema5'], dataframe['ema10'])
                # (dataframe['ema10'] < dataframe['ema10'].shift(1)) &
            # (dataframe['fastk'] > 80) &
            # (dataframe['btc_trend'] == 'pump')
            # |
            # (dataframe['btc_trend'] == 'dump')
            # |
            # (dataframe['profit'] == 1)
            # (dataframe['profit'] == 1)

            # (
            #     # % of high
            #     (
            #         dataframe['high'].rolling(window=(dataframe['buy'] == 1).idxmax()).max().shift(1)
            #         -
            #         dataframe['close'].shift((dataframe['buy'] == 1).idxmax())
            #     )
            # )
            # >
            # (
            #     #diff price from high
            #     dataframe['close'].shift((dataframe['buy'] == 1).idxmax())
            #      * 0.02
            # )
            ),
            'sell'] = 1
        return dataframe

    def hyperopt_space(self) -> List[Dict]:
        """
        Define your Hyperopt space for the strategy
        :return: Dict
        """
        space = {
            'ha_close_ema20': hp.choice('ha_close_ema20', [
                {'enabled': False},
                {'enabled': True}
            ]),
            'ha_open_close': hp.choice('ha_open_close', [
                {'enabled': False},
                {'enabled': True}
            ]),
            'trigger': hp.choice('trigger', [
                {'type': 'ema50_cross_ema100'},
                {'type': 'ema5_cross_ema10'},
            ]),
            'stoploss': hp.uniform('stoploss', -0.5, -0.01),
        }
        return space

    def buy_strategy_generator(self, params) -> None:
        """
        Define the buy strategy parameters to be used by hyperopt
        """
        def populate_buy_trend(dataframe: DataFrame) -> DataFrame:
            conditions = []
            # GUARDS AND TRENDS
            if 'ha_close_ema20' in params and params['ha_close_ema20']['enabled']:
                conditions.append(dataframe['ha_close'] > dataframe['ema20'])

            if 'ha_open_close' in params and params['ha_open_close']['enabled']:
                conditions.append(dataframe['ha_open'] < dataframe['ha_close'])


            # TRIGGERS
            triggers = {
                'ema20_cross_ema50': (qtpylib.crossed_above(dataframe['ema20'], dataframe['ema50'])),
                'ema50_cross_ema100': (qtpylib.crossed_above(dataframe['ema50'], dataframe['ema100'])),
            }
            conditions.append(triggers.get(params['trigger']['type']))

            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

            return dataframe

        return populate_buy_trend
