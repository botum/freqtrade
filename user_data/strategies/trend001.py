# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement

import talib.abstract as ta
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
# from scripts import trendy
from freqtrade.indicators import in_range, find_pivots, went_down, get_trend_lines
# from freqtrade.persistence import Pair

# from freqtrade.persistence import *

class_name = 'DefaultStrategy'

# pair = 'ETH/BTC'

class DefaultStrategy(IStrategy):
    """
    Default Strategy provided by freqtrade bot.
    You can override it with your own strategy
    """

    # Minimal ROI designed for the strategy
    minimal_roi = {
        # "200":  0.001,
        # "120":  0.01,
        # "60":  0.02,
        # "15":  0.03,
        # "10":  0.04,
        # "0":  0.05,
        # "120":  0.05,
        # "60":  0.001,
        # "15":  0.005,
        # "10":  0.008,
        # "7":  0.0001,
        # "7":  0.001,
        # "6":  0.003,
        # "5":  0.006,
        # "4":  0.007,
        # "3":  0.008,
        # "2":  0.009,
        # "0":  0.01,
        # "2":  0.03,
        # "1":  0.04,
        # "0":  0.05
        # "30":  0.001,
        # "20":  0.005,
        # "10":  0.008,
        # "0":  0.0001
    }

    # Optimal stoploss designed for the strategy
    stoploss = -0.6

    # Optimal ticker interval for the strategy
    ticker_interval = "1m"

    def populate_cycle_trend(self, dataframe: DataFrame) -> DataFrame:


        # dataframe['main_trend_max'], dataframe['main_trend_min'], main_maxslope, \
        # main_minslope = get_trend_lines(pair, dataframe, timerange=48, interval="60m")
        # dataframe['trend_max'], dataframe['trend_min'], \
        # short_maxslope, short_minslope

        # MFI
        dataframe['mfi'] = ta.MFI(dataframe)
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe)

        # Overlap Studies
        # ------------------------------------

        # Previous Bollinger bands
        # Because ta.BBANDS implementation is broken with small numbers, it actually
        # returns middle band for all the three bands. Switch to qtpylib.bollinger_bands
        # and use middle band instead.
        dataframe['blower'] = ta.BBANDS(dataframe, nbdevup=2, nbdevdn=2)['lowerband']

        # Bollinger bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

        # EMA - Exponential Moving Average
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema200'] = ta.EMA(dataframe, timeperiod=200)

        # SMA - Simple Moving Average
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=40)

        # TEMA - Triple Exponential Moving Average
        dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)

        heikinashi = qtpylib.heikinashi(dataframe)
        dataframe['ha_open'] = heikinashi['open']
        dataframe['ha_close'] = heikinashi['close']

        dataframe.loc[
            (
                # (dataframe['ha_open'] > dataframe['ha_close'])  # green bar
                # &
                # crossed_below(dataframe['ema100'], dataframe['ema200'])
                # (dataframe['mfi'] < 10)

                # (dataframe['close'] < dataframe['bb_lowerband'] * 1.02)
                # &
                # (dataframe['close'] < dataframe['high'].rolling(window=192).max() * 0.75)  # green bar

                # (dataframe['close'] < dataframe['main_trend_min'] * 1.005)
                # |
                # (dataframe['close'] < dataframe['trend_min'] * 1.005)
                # &
                # (
                # (dataframe['main_trend_min'].shift(5) <= dataframe['main_trend_min'])
                # &
                # (dataframe['trend_min'].shift(5) <= dataframe['trend_min'])
                # )
                # &
                (
                (main_maxslope >= main_minslope)
                &
                (short_maxslope >= short_minslope)
                &
                (short_maxslope >= 0)
                )
            ),
            '24htrend'] = 1

        # dataframe.loc[
        #     (
        #         (dataframe['ha_open'] > dataframe['ha_close'])  # green bar
        #
        #     ),
        #     '24htrend'] = 1

        return dataframe


    def populate_indicators(self, dataframe: DataFrame, pivots: dict={}) -> DataFrame:
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
        if pivots and len(pivots) > 0:
            def set_sup(row):
                # print (row)
                supports = sorted(pivots['sup'], reverse=True)
                # print (pivots['sup'])
                for sup in supports:
                    # print ('sup: ', sup, 'low: ', row['low'])
                    # print (row["low"] >= sup * 0.98)
                    if row["low"] >= sup:
                        # print ('bingo: ', sup)
                        return sup
            def set_sup2(row):
                # print (row)
                # print (pivots['sup'])
                supports = sorted(pivots['sup'], reverse=True)
                for sup in supports:
                    # print ('sup: ', sup, 'low: ', row['low'])
                    # print (row["low"] >= sup * 0.98)
                    if row["low"] >= sup and sup < row['s1'] :
                        # print ('bingo: ', sup)
                        return sup
            def set_res(row):
                resistences = sorted(pivots['res'])
                # resistences.append(pivots['sup'])
                for res in resistences:
                    # print ('res: ', row["s1"] * 1.01, res)
                    #  and res >= row["s1"] * 1.02
                    if row["high"] <= res and res >= row["s1"] * 1.02:
                        return res
            def set_res2(row):
                resistences = sorted(pivots['res'])
                # resistences.append(pivots['sup'])
                for res in resistences:
                    # print ('res: ', row["s1"] * 1.01, res)
                    if row["high"] <= res and row["r1"] < res and res >= row["s1"] * 1.02:
                        return res
            dataframe = dataframe.assign(s1=dataframe.apply(set_sup, axis=1))
            dataframe = dataframe.assign(s2=dataframe.apply(set_sup2, axis=1))
            dataframe = dataframe.assign(r1=dataframe.apply(set_res, axis=1))
            dataframe = dataframe.assign(r2=dataframe.apply(set_res2, axis=1))
        else:
            dataframe['s1'] = 0
            dataframe['r1'] = 0
            dataframe['s2'] = 0
            dataframe['r2'] = 0


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
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2.5)
        dataframe['bb_lowerband'] = bollinger['lower']
        dataframe['bb_middleband'] = bollinger['mid']
        dataframe['bb_upperband'] = bollinger['upper']

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
                (
                # buy when going down and hit support
                # (dataframe['trend_min'] < dataframe['high'].rolling(window=5).mean().shift(1))
                # &
                (
                # (dataframe['close'] < dataframe['main_trend_min'] * 1.005)
                # |
                (dataframe['low'] < dataframe['main_trend_min'] * 1.01)
                )
                # &
                # (
                # (dataframe['main_trend_max'] > dataframe['main_trend_min']*1.02)
                # |
                # (dataframe['close'] > dataframe['main_trend_max'] * 1.03)
                # )
                # &
                # (dataframe['close'] > dataframe['trend_min'] * 0.995)
                # (
                # (dataframe['trend_max'] > dataframe['trend_min'])
                # )
                #
                # &
                # (dataframe['close'] <= dataframe['bb_lowerband'])
                # &
                # (dataframe['volume'] > (dataframe['volume'].shift(1) * 1.1))
                # &
                # (dataframe['volume'] > (dataframe['volume'].rolling(window=30).mean().shift(1) * 5))
                # &
                # (dataframe['close'] > dataframe['s1'])
                # &
                # (dataframe['close'] < dataframe['bb_middleband'])
                # &
                # (dataframe['ha_open'] < dataframe['ha_close'])  # green bar
                )
                # &
                # (
                # (dataframe['close'] > dataframe['trend_min'])
                # |
                # (dataframe['close'] > dataframe['trend_max'])
                # )
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
                # (dataframe['close'] >= dataframe['r1'].shift((dataframe['buy'] == 1).idxmax()) * 0.999)
                # |
                (dataframe['close'] >= dataframe['main_trend_max'] * 0.99)
                |
                # (dataframe['close'] >= dataframe['trend_max'] * 0.99)
                # |

                # (
                # (dataframe['ha_open'] > dataframe['ha_close'])  # red bar
                # &
                # (dataframe['ha_open'].shift(1) > dataframe['ha_close'].shift(1))  # red bar
                # )
                # |
                (dataframe['close'] <= dataframe['main_trend_min'] * 0.9)
                # |
                # (dataframe['close'] <= dataframe['trend_min'] * 0.95)
                # &
                # (
                #     (dataframe['close'].shift(1) > dataframe['bb_upperband'])
                #     &
                #     (dataframe['close'] <= dataframe['bb_upperband'] * 0.999)
                # )

            ),
            'sell'] = 1

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
