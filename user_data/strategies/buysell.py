# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import talib.abstract as ta
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
# from scripts import trendy
from freqtrade.indicators import in_range, went_down, get_trend_lines, get_pivots
# from freqtrade.persistence import Pair

# from freqtrade.persistence import *

from freqtrade.trends import *

class_name = 'DefaultStrategy'

# pair = 'ETH/BTC'

class buysell(IStrategy):
    """
    Default Strategy provided by freqtrade bot.
    You can override it with your own strategy
    """

    # Minimal ROI designed for the strategy
    minimal_roi = {
        # "0":  0.01
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.6

    # Optimal ticker interval for the strategy
    ticker_interval = "1m"
    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:

        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """

        # dataframe['x_maxima'], dataframe['maxima'], dataframe['x_minima'], dataframe['minima'] = trendy.segtrends(dataframe['close'], segments = 5)

        # dataframe['trends'], maxslope, minslope = trendy.gentrends(dataframe['close'], window =10)

        # dataframe['xMax'], dataframe['yMax'], dataframe['xMin'], dataframe['yMin'] = trendy.minitrends(dataframe['close'], window = 30)
        #
        # print (pivots)


        # print (dataframe['sr'], "sr002.py")


        # dataframe['main_trend_max'], dataframe['main_trend_min'], main_maxslope, \
        # main_minslope = get_trend_lines(pair, dataframe, charts=True, interval="30m", timerange=200)

        # print ( main_maxslope, main_minslope, short_maxslope, short_minslope)
        # print (dataframe['trend_min'])

        # Momentum Indicator
        # ------------------------------------

        # ADX
        # dataframe['adx'] = ta.ADX(dataframe)

        # Awesome oscillator
        # dataframe['ao'] = qtpylib.awesome_oscillator(dataframe)
        """
        # Commodity Channel Index: values Oversold:<-100, Overbought:>100
        dataframe['cci'] = ta.CCI(dataframe)
        """
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)

        # Minus Directional Indicator / Movement
        # dataframe['minus_dm'] = ta.MINUS_DM(dataframe)
        # dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        # Plus Directional Indicator / Movement
        # dataframe['plus_dm'] = ta.PLUS_DM(dataframe)
        # dataframe['plus_di'] = ta.PLUS_DI(dataframe)
        # dataframe['minus_di'] = ta.MINUS_DI(dataframe)

        """
        # ROC
        dataframe['roc'] = ta.ROC(dataframe)
        """
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)
        """
        # Inverse Fisher transform on RSI, values [-1.0, 1.0] (https://goo.gl/2JGGoy)
        rsi = 0.1 * (dataframe['rsi'] - 50)
        dataframe['fisher_rsi'] = (numpy.exp(2 * rsi) - 1) / (numpy.exp(2 * rsi) + 1)
        # Inverse Fisher transform on RSI normalized, value [0.0, 100.0] (https://goo.gl/2JGGoy)
        dataframe['fisher_rsi_norma'] = 50 * (dataframe['fisher_rsi'] + 1)
        # Stoch
        stoch = ta.STOCH(dataframe)
        dataframe['slowd'] = stoch['slowd']
        dataframe['slowk'] = stoch['slowk']
        """
        # Stoch fast
        stoch_fast = ta.STOCHF(dataframe)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']
        """
        # Stoch RSI
        stoch_rsi = ta.STOCHRSI(dataframe)
        dataframe['fastd_rsi'] = stoch_rsi['fastd']
        dataframe['fastk_rsi'] = stoch_rsi['fastk']
        """

        # Overlap Studies
        # ------------------------------------

        # Previous Bollinger bands
        # Because ta.BBANDS implementation is broken with small numbers, it actually
        # returns middle band for all the three bands. Switch to qtpylib.bollinger_bands
        # and use middle band instead.
        dataframe['blower'] = ta.BBANDS(dataframe, nbdevup=2, nbdevdn=2)['lowerband']

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=1)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']


        dataframe['stddev'] = ta.STDDEV(dataframe)


        # # EMA - Exponential Moving Average
        # dataframe['ema3'] = ta.EMA(dataframe, timeperiod=3)
        # dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)
        # dataframe['ema10'] = ta.EMA(dataframe, timeperiod=10)
        # dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        # dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)

        # # SAR Parabol
        # dataframe['sar'] = ta.SAR(dataframe)
        #
        # # SMA - Simple Moving Average
        # dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)
        #
        # # TEMA - Triple Exponential Moving Average
        # dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
        #
        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """

        dataframe.loc[
            (
                (dataframe['close'] < dataframe['bb_lowerband']*1.01)
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
            (
                    (dataframe['close'] > dataframe['bb_middleband'])

            ),
            'sell'] = 1
        # print (dataframe.loc[dataframe['sell']==1].close)

        return dataframe



    def did_bought(self):
        """
        we are notified that a given pair was bought
        :param pair: the pair that was is concerned by the dataframe
        """

    def did_sold(self):
        """
        we are notified that a given pair was sold
        :param pair: the pair that was is concerned by the dataframe
        """

    def did_cancel_buy(self):
        """
        we are notified that a given pair buy was not filled
        :param pair: the pair that was is concerned by the dataframe
        """

    def did_cancel_sell(self):
        """
        we are notified that a given pair was not sold
        :param pair: the pair that was is concerned by the dataframe
        """
