"""
Functions to analyze ticker data with indicators and produce buy and sell signals
"""
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Tuple

import arrow
from pandas import DataFrame, to_datetime

from freqtrade import (DependencyException, OperationalException, exchange, persistence)
from freqtrade.exchange import get_ticker_history
from freqtrade.logger import Logger
from freqtrade.persistence import Trade, Pair
from freqtrade.strategy.strategy import Strategy
from freqtrade.constants import Constants
from freqtrade.indicators import get_trend_lines, get_pivots, in_range
from freqtrade.trends import gentrends

# ZigZag

# This is inside your IPython Notebook
import pyximport
pyximport.install(reload_support=True)
from freqtrade.vendor.zigzag_hi_lo import *
# from zigzag import *


class SignalType(Enum):
    """
    Enum to distinguish between buy and sell signals
    """
    BUY = "buy"
    SELL = "sell"


class Analyze(object):
    """
    Analyze class contains everything the bot need to determine if the situation is good for
    buying or selling.
    """
    def __init__(self, config: dict) -> None:
        """
        Init Analyze
        :param config: Bot configuration (use the one from Configuration())
        """
        self.logger = Logger(name=__name__, level=config.get('loglevel')).get_logger()

        self.config = config
        self.strategy = Strategy(self.config)

    @staticmethod
    def parse_ticker_dataframe(ticker: list) -> DataFrame:
        """
        Analyses the trend for the given ticker history
        :param ticker: See exchange.get_ticker_history
        :return: DataFrame
        """
        cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        frame = DataFrame(ticker, columns=cols)

        frame['date'] = to_datetime(frame['date'],
                                    unit='ms',
                                    utc=True,
                                    infer_datetime_format=True)

        frame.sort_values('date', inplace=True)
        return frame

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """
        return self.strategy.populate_indicators(dataframe=dataframe)

    def populate_trend_lines(self, df: DataFrame, pair: str, interval: int) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """
        # dataframe['short_trend_max'], dataframe['short_trend_min'], dataframe['short_trend_max_slope'], dataframe['short_trend_min_slope'] = get_trend_lines(pair, dataframe)
        #
        #
        # # short time trends
        #
        # df60 = DataFrame
        # interval = "1m"
        # ticker_hist = get_ticker_history(pair, interval)
        # if not ticker_hist:
        #     self.logger.warning('Empty ticker history for pair %s, interval: %s', pair, interval)
        # try:
        #     df60 = self.analyze_ticker(ticker_hist, pair)
        # except ValueError as error:
        #     self.logger.warning(
        #         'Unable to analyze ticker for pair %s: %s',
        #         pair,
        #         str(error)
        #     )
        #
        # if df60.empty:
        #     self.logger.warning('Empty df60 for pair %s, interval: %s', pair, interval)

        # dataframe['main_trend_max'], dataframe['main_trend_min'], dataframe['main_trend_max_slope'], dataframe['main_trend_min_slope'] = get_trend_lines(dataframe, pair)
        # dataframe['main_trend_max'], dataframe['main_trend_min'], dataframe['main_trend_max_slope'], dataframe['main_trend_min_slope'] = get_trend_lines(dataframe, pair)

        # df60['main_trend_max'], df60['main_trend_min'], df60['main_trend_max_slope'], df60['main_trend_min_slope'] = get_trend_lines(pair, df60)


        timeframe_volat = {
                        '1d':1.1,
                        '1h':0.0009,
                        '5m':4,
                        '1m':1.2}

        volat_window = {
                        '1d':0.5,
                        '1h':2,
                        '5m':10,
                        '1m':5}
        print (interval)
        window = volat_window[interval]
        df['bb_exp'] = (df.bb_upperband.rolling(window=window).max() - df.bb_lowerband.rolling(window=window).min()) / df.bb_upperband.rolling(window=window).max() * timeframe_volat[interval]
        pivots = peak_valley_pivots(df.low.values, df.high.values, df.bb_exp.values)
        df['pivots'] = np.transpose(np.array((pivots)))

        df = gentrends(df, pair=pair, charts=True)
        # print (df)
        # print (df)
        # def set_sup(row):
        #     # print (row)
        #     supports = [col for col in df if col.startswith('t_')]
        #     # support = df[df['ids'].str.contains('ball', na = False)]
        #     # print(supports)
        #     # print (pivots['sup'])
        #     for sup in supports:
        #         # print (row["low"] >= sup * 0.98)
        #         # print (row)
        #         if row["low"] >= row[sup]:
        #             # print ('bingo: ')
        #             # print ('sup: ', row[sup], 'low: ', row['low'])
        #             # if in_range(row["close"],row[sup], 0.001):
        #             #     print ('buy zone: ')
        #             #     print ('sup: ', row[sup], 'close: ', row['close'], 'low: ', row['low'])
        #             return row[sup]
        # def set_res(row):
        #     resistences = [col for col in df if col.startswith('t_')]
        #     # resistences.append(pivots['sup'])
        #     for res in resistences:
        #         # print ('res: ', row[res] , row[high])
        #         #  and res >= row["s1"] * 1.02
        #         if row["high"] <= row[res]:
        #             return row[res]
        # df = df.assign(st=df.apply(set_sup, axis=1))
        # df = df.assign(rt=df.apply(set_res, axis=1))

#         dataframe['s1'] = df.filter(regex='trend-$', axis=1)[]
        # print (df.st, df.rt)

        return df

    def populate_pivots(self, dataframe: DataFrame, pair: str) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """
        # print (dataframe)
        # persistence.init(self.config)
        # exchange.init(self.config)
        # pair_obj = Pair.query.filter(Pair.pair.is_(_pair)).first()
        #
        # dataframe = pair_obj.get_pivots(dataframe)

        # dataframe = get_pivots(dataframe, pair, piv_type='sup')
        # dataframe = get_pivots(dataframe, pair, piv_type='res')

        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        return self.strategy.populate_buy_trend(dataframe=dataframe)

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :return: DataFrame with buy column
        """
        return self.strategy.populate_sell_trend(dataframe=dataframe)

    def get_ticker_interval(self) -> str:
        """
        Return ticker interval to use
        :return: Ticker interval value to use
        """
        return self.strategy.ticker_interval

    def analyze_ticker(self, ticker_history: List[Dict], pair: str, interval: int) -> DataFrame:
        """
        Parses the given ticker history and returns a populated DataFrame
        add several TA indicators and buy signal to it
        :return DataFrame with ticker data and indicator data
        """
        dataframe = self.parse_ticker_dataframe(ticker_history)
        dataframe = self.populate_indicators(dataframe)
        # dataframe = self.populate_pivots(dataframe, pair)
        dataframe = self.populate_trend_lines(dataframe, pair, interval)
        dataframe = self.populate_buy_trend(dataframe)
        dataframe = self.populate_sell_trend(dataframe)
        return dataframe

    def get_signal(self, pair: str, interval: str) -> Tuple[bool, bool]:
        """
        Calculates current signal based several technical analysis indicators
        :param pair: pair in format ANT/BTC
        :param interval: Interval to use (in min)
        :return: (Buy, Sell) A bool-tuple indicating buy/sell signal
        """
        print('interval : ',interval)
        ticker_hist = get_ticker_history(pair, interval)
        if not ticker_hist:
            self.logger.warning('Empty ticker history for pair %s', pair)
            return False, False

        try:
            dataframe = self.analyze_ticker(ticker_hist, pair, interval)
        except ValueError as error:
            self.logger.warning(
                'Unable to analyze ticker for pair %s: %s',
                pair,
                str(error)
            )
            return False, False
        except Exception as error:
            self.logger.exception(
                'Unexpected error when analyzing ticker for pair %s: %s',
                pair,
                str(error)
            )
            return False, False

        if dataframe.empty:
            self.logger.warning('Empty dataframe for pair %s', pair)
            return False, False

        latest = dataframe.iloc[-1]

        # Check if dataframe is out of date
        signal_date = arrow.get(latest['date'])
        interval_minutes = Constants.TICKER_INTERVAL_MINUTES[interval]
        if signal_date < arrow.utcnow() - timedelta(minutes=(interval_minutes + 5)):
            self.logger.warning(
                'Outdated history for pair %s. Last tick is %s minutes old',
                pair,
                (arrow.utcnow() - signal_date).seconds // 60
            )
            return False, False

        (buy, sell) = latest[SignalType.BUY.value] == 1, latest[SignalType.SELL.value] == 1
        self.logger.debug(
            'trigger: %s (pair=%s) buy=%s sell=%s',
            latest['date'],
            pair,
            str(buy),
            str(sell)
        )
        return buy, sell

    def should_sell(self, trade: Trade, rate: float, date: datetime, buy: bool, sell: bool) -> bool:
        """
        This function evaluate if on the condition required to trigger a sell has been reached
        if the threshold is reached and updates the trade record.
        :return: True if trade should be sold, False otherwise
        """
        # Check if minimal roi has been reached and no longer in buy conditions (avoiding a fee)
        if self.min_roi_reached(trade=trade, current_rate=rate, current_time=date):
            self.logger.debug('Required profit reached. Selling..')
            return True

        # Experimental: Check if the trade is profitable before selling it (avoid selling at loss)
        if self.config.get('experimental', {}).get('sell_profit_only', False):
            self.logger.debug('Checking if trade is profitable..')
            if trade.calc_profit(rate=rate) <= 0:
                return False

        if sell and not buy and self.config.get('experimental', {}).get('use_sell_signal', False):
            self.logger.debug('Sell signal received. Selling..')
            return True

        return False

    def min_roi_reached(self, trade: Trade, current_rate: float, current_time: datetime) -> bool:
        """
        Based an earlier trade and current price and ROI configuration, decides whether bot should
        sell
        :return True if bot should sell at current rate
        """
        current_profit = trade.calc_profit_percent(current_rate)
        if self.strategy.stoploss is not None and current_profit < self.strategy.stoploss:
            self.logger.debug('Stop loss hit.')
            return True

        # Check if time matches and current rate is above threshold
        time_diff = (current_time.timestamp() - trade.open_date.timestamp()) / 60
        for duration, threshold in self.strategy.minimal_roi.items():
            if time_diff <= duration:
                return False
            if current_profit > threshold:
                return True

        return False

    def tickerdata_to_dataframe(self, tickerdata: Dict[str, List]) -> Dict[str, DataFrame]:
        """
        Creates a dataframe and populates indicators for given ticker data
        """
        return {pair: self.populate_indicators(self.parse_ticker_dataframe(pair_data))
                for pair, pair_data in tickerdata.items()}
