
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


# Update this variable if you change the class name
class_name = 'CactiX'


"""
Indicators for Freqtrade
author@: Gerald Lonlas
github@: https://github.com/glonlas/freqtrade-strategies
"""


class CactiX(IStrategy):

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi"
    minimal_roi = {
       '60':  0.0,
       '50':  0.01,
       '40':  0.02,
       '30':  0.03,
       '0':   0.04
    }

    # Optimal stoploss designed for the strategy
    # This attribute will be overridden if the config file contains "stoploss"
    stoploss = -0.3

    # Optimal ticker interval for the strategy
    ticker_interval = 1

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        pivots = pivots_points(dataframe)
        dataframe['pivot'] = pivots['pivot']
        dataframe['r1'] = pivots['r1']
        dataframe['r2'] = pivots['r2']
        dataframe['s1'] = pivots['s1']
        dataframe['s2'] = pivots['s2']
        dataframe['adx'] = ta.ADX(dataframe)
        dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)
        dataframe['cci'] = ta.CCI(dataframe)
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['mfi'] = ta.MFI(dataframe)
        dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)
        dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)
        dataframe['roc'] = ta.ROC(dataframe)
        dataframe['rsi'] = ta.RSI(dataframe)
        # rsi = 0.1 * (dataframe['rsi'] - 50)
        # dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)
        # dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)
        stoch = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch['slowd']
        dataframe['slowk'] = stoch['slowk']
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        stoch_rsi = ta.STOCHRSI(dataframe)
        dataframe['fastd_rsi'] = stoch_rsi['fastd']
        dataframe['fastk_rsi'] = stoch_rsi['fastk']
        dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        dataframe['fastminus_dm'] = ta.MINUS_DM(dataframe, timeperiod=1)
        dataframe['longminus_dm'] = ta.MINUS_DM(dataframe, timeperiod=60)
        dataframe['longlongminus_dm'] = ta.MINUS_DM(dataframe, timeperiod=200)
        dataframe['minus_di'] = ta.MINUS_DI(dataframe)
        dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        dataframe['fastplus_dm'] = ta.PLUS_DM(dataframe, timeperiod=1)
        dataframe['longplus_dm'] = ta.PLUS_DM(dataframe, timeperiod=60)
        dataframe['longlongplus_dm'] = ta.PLUS_DM(dataframe, timeperiod=200)
        dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        dataframe['direction'] = dataframe['plus_dm'] - dataframe['minus_dm']
        dataframe['fastdirection'] = dataframe['fastplus_dm'] - dataframe['fastminus_dm']
        dataframe['longdirection'] = dataframe['longplus_dm'] - dataframe['longminus_dm']
        dataframe['longlongdirection'] = dataframe['longlongplus_dm'] - dataframe['longlongminus_dm']
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']
        dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['sar'] = ta.SAR(dataframe)
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        
        #Very Good, use this.
        dataframe.loc[
            (
                (dataframe['rsi'] < 47) &
                (dataframe['rsi'] > 0) &
                (dataframe['fastd'] < 47) &
                (dataframe['fastd'] > 0) &
                (dataframe['close'] > dataframe['open']) &
                (dataframe['close'].shift(1) > dataframe['open'].shift(1)) &
                (dataframe['close'].shift(2) > dataframe['open'].shift(2)) &
                (dataframe['cci'] < -70) &
                (dataframe['cci'] > -500000000)
            ),
            'buy'] = 1

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        dataframe.loc[
            # (
            #     qtpylib.crossed_above(dataframe['ema50'], dataframe['ema100']) &
            #     (dataframe['ha_close'] < dataframe['ema20']) &
            #     (dataframe['ha_open'] > dataframe['ha_close'])  # red bar
            # )
            # |
            # best
            # (
            # (dataframe['close'] < (dataframe['r1'].shift((dataframe['buy'] == 1).idxmin())
            # -
            # (dataframe['r1'].shift((dataframe['buy'] == 1).idxmin()) * 0.01)))
            # &
            # in_range(dataframe['close'].rolling(window=5).max(), dataframe['r1'].shift((dataframe['buy'] == 1).idxmin()), 0.01)
            # )
            1
            ,
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
