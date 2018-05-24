"""
Functions to analyze ticker data with indicators and produce buy and sell signals
"""
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Tuple

import arrow
from pandas import DataFrame, to_datetime
from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, String,
                        create_engine)
from sqlalchemy.engine import Engine

from freqtrade import (DependencyException, OperationalException, exchange, persistence)
from freqtrade.configuration import Configuration
from freqtrade.exchange import get_ticker_history
from freqtrade import persistence
# from freqtrade.persistence import *
from freqtrade.persistence import Trade, Pair, Trend
from freqtrade.strategy.resolver import StrategyResolver
from freqtrade import constants
from freqtrade.indicators import get_trend_lines, get_pivots, in_range
from freqtrade.trends import gentrends, plot_trends

# ZigZag

# This is inside your IPython Notebook
import pyximport
pyximport.install(reload_support=True)
from freqtrade.vendor.zigzag_hi_lo import *
# from zigzag import *


logger = logging.getLogger(__name__)

def get_df(pair: str, interval: str) -> DataFrame:
    """
    Calculates current signal based several technical analysis indicators
    :param pair: pair in format ANT/BTC
    :param interval: Interval to use (in min)
    :return: (Buy, Sell) A bool-tuple indicating buy/sell signal
    """
    print('interval : ',interval)


    # ticker_hist = get_ticker_history(pair, interval)
    ticker_hist = load_tickerdata_file('freqtrade/tests/testdata/', pair, interval)
#     print (ticker_hist)
#     print (ticker_hist)
    if not ticker_hist:
        logger.warning('Empty ticker history for pair %s', pair)
        return None

    try:
        dataframe = analyze.analyze_ticker(ticker_hist, pair, interval)
    except ValueError as error:
        logger.warning(
            'Unable to analyze ticker for pair %s: %s',
            pair,
            str(error)
        )
        return None
    except Exception as error:
        logger.exception(
            'Unexpected error when analyzing ticker for pair %s: %s',
            pair,
            str(error)
        )
        return None

    return dataframe


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
        self.config = config
        self.strategy = StrategyResolver(self.config).strategy

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

        # group by index and aggregate results to eliminate duplicate ticks
        frame = frame.groupby(by='date', as_index=False, sort=True).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'max',
        })
        return frame

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        """
        return self.strategy.populate_indicators(dataframe=dataframe)

    def populate_trend_lines(self, df: DataFrame, pair: str, interval: int, trade: Trade=None) -> DataFrame:
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


        # df = gentrends(self, df, pair=pair, charts=False)

        config = Configuration.get_config(self)
        # engine = create_engine('sqlite:///tradesv3.trends.sqlite')
        persistence.init(config)

        print (pair)
        print (len(df))

        current_pair = Pair.query.filter(Pair.pair.is_(pair)).first()


        if current_pair == None:
            pair_obj = Pair(
                pair=pair
            )
            Pair.session.add(pair_obj)
            persistence.cleanup()
            current_pair = Pair.query.filter(Pair.pair.is_(pair)).first()

        print (current_pair)

        # df = current_pair.populate_trend_lines(df, interval)

        if trade:
            print ('-------------------- we are on a trade --------------------------------')
            res, sup = trade.res_trend, trade.sup_trend
        else:
            res, sup = current_pair.res_trend, current_pair.sup_trend
            print('first attempt: ', sup, res)

            if not (res and sup):
                update = True
            else:
                df = res.populate_to_df(df)
                df = sup.populate_to_df(df)
                if( df.iloc[-1]['close'] > df.iloc[-1]['max']) or (df.iloc[-1]['close'] < df.iloc[-1]['min']):
                    logger.info('%s max or min off trends /////////////////////////////////////////////////////', current_pair.pair)
                    update = True


            if update == True:
                current_pair.update_trend_lines(df, interval)
                # print ('max tren: ', Trend.query.filter_by(max = True, pair_id=self.id, interval=interval).first())
                current_pair.res_trend = Trend.query.filter_by(max = True, type = 'res', pair_id=current_pair.id, interval=interval).first()
                # print (current_pair.res_trend)
                # print ('min tren: ', Trend.query.filter_by(min = True, pair_id=current_pair.id, interval=interval).first())
                current_pair.sup_trend = Trend.query.filter_by(min = True, type = 'sup', pair_id=current_pair.id, interval=interval).first()
                # print (self.sup_trend)
                persistence.cleanup()
                current_pair = Pair.query.filter(Pair.pair.is_(pair)).first()
                res, sup = current_pair.res_trend, current_pair.sup_trend
                print(sup, res)

            df = res.populate_to_df(df)
            df = sup.populate_to_df(df)

            # print (df)
        print('current pair: ', current_pair)

        # print (df)
        # print (df)
        # def set_sup(row):
        #     # print (row)
        #     supports = [col for col in df if col.startswith('trend-')]
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
        #     resistences = [col for col in df if col.startswith('trend-')]
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
        filename = 'chart_plots/' + interval + '-' + pair.replace('/', '-') + datetime.utcnow().strftime('-%m-%d-%Y-%H') + '-backtesting.png'
        plot_trends(df, filename)

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

    def analyze_ticker(self, ticker_history: List[Dict], pair: str, interval: int, trade: Trade) -> DataFrame:
        """
        Parses the given ticker history and returns a populated DataFrame
        add several TA indicators and buy signal to it
        :return DataFrame with ticker data and indicator data
        """
        dataframe = self.parse_ticker_dataframe(ticker_history)
        dataframe = self.populate_indicators(dataframe)
        # dataframe = self.populate_pivots(dataframe, pair)
        dataframe = self.populate_trend_lines(dataframe, pair, interval, trade)
        dataframe = self.populate_buy_trend(dataframe)
        dataframe = self.populate_sell_trend(dataframe)
        return dataframe

    def get_signal(self, pair: str, interval: str, trade: Trade=None) -> Tuple[bool, bool]:
        """
        Calculates current signal based several technical analysis indicators
        :param pair: pair in format ANT/BTC
        :param interval: Interval to use (in min)
        :return: (Buy, Sell) A bool-tuple indicating buy/sell signal
        """
        print('interval : ',interval)
        ticker_hist = get_ticker_history(pair, interval)
        if not ticker_hist:
            logger.warning('Empty ticker history for pair %s', pair)
            return False, False

        try:
            dataframe = self.analyze_ticker(ticker_hist, pair, interval, trade)
        except ValueError as error:
            logger.warning(
                'Unable to analyze ticker for pair %s: %s',
                pair,
                str(error)
            )
            return False, False
        except Exception as error:
            logger.exception(
                'Unexpected error when analyzing ticker for pair %s: %s',
                pair,
                str(error)
            )
            return False, False

        if dataframe.empty:
            logger.warning('Empty dataframe for pair %s', pair)
            return False, False

        latest = dataframe.iloc[-1]

        # Check if dataframe is out of date
        signal_date = arrow.get(latest['date'])
        interval_minutes = constants.TICKER_INTERVAL_MINUTES[interval]
        if signal_date < arrow.utcnow() - timedelta(minutes=(interval_minutes + 5)):
            logger.warning(
                'Outdated history for pair %s. Last tick is %s minutes old',
                pair,
                (arrow.utcnow() - signal_date).seconds // 60
            )
            return False, False

        (buy, sell) = latest[SignalType.BUY.value] == 1, latest[SignalType.SELL.value] == 1
        logger.debug(
            'trigger: %s (pair=%s) buy=%s sell=%s',
            latest['date'],
            pair,
            str(buy),
            str(sell)
        )
        return buy, sell, dataframe

    def should_sell(self, trade: Trade, rate: float, date: datetime, buy: bool, sell: bool) -> bool:
        """
        This function evaluate if on the condition required to trigger a sell has been reached
        if the threshold is reached and updates the trade record.
        :return: True if trade should be sold, False otherwise
        """
        # Check if minimal roi has been reached and no longer in buy conditions (avoiding a fee)
        if self.min_roi_reached(trade=trade, current_rate=rate, current_time=date):
            logger.debug('Required profit reached. Selling..')
            return True

        # Experimental: Check if the trade is profitable before selling it (avoid selling at loss)
        if self.config.get('experimental', {}).get('sell_profit_only', False):
            logger.debug('Checking if trade is profitable..')
            if trade.calc_profit(rate=rate) <= 0:
                return False

        if sell and not buy and self.config.get('experimental', {}).get('use_sell_signal', False):
            logger.debug('Sell signal received. Selling..')
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
            logger.debug('Stop loss hit.')
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
