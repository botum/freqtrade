"""
This module contains the class to persist trades into SQLite
"""

import logging
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Dict, Optional

import arrow
from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, String,
                        create_engine)

from pandas import DataFrame, to_datetime
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.scoping import scoped_session
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.pool import StaticPool
from sqlalchemy_utils import ScalarListType

from freqtrade.exchange import get_ticker_history

from freqtrade.indicators import get_pivots

from freqtrade.trends import gentrends

from scipy import linspace, polyval, polyfit, sqrt, stats, randn
import numpy as np

logger = logging.getLogger(__name__)

_CONF = {}
_DECL_BASE = declarative_base()


def init(config: dict, engine: Optional[Engine] = None) -> None:
    """
    Initializes this module with the given config,
    registers all known command handlers
    and starts polling for message updates
    :param config: config to use
    :param engine: database engine for sqlalchemy (Optional)
    :return: None
    """
    _CONF.update(config)
    if not engine:
        if _CONF.get('dry_run', False):
            # the user wants dry run to use a DB
            if _CONF.get('dry_run_db', False):
                engine = create_engine('sqlite:///tradesv3.dry_run.sqlite')
            # Otherwise dry run will store in memory
            else:
                engine = create_engine('sqlite://',
                                       connect_args={'check_same_thread': False},
                                       poolclass=StaticPool,
                                       echo=False)
        else:
            engine = create_engine('sqlite:///tradesv3.sqlite')

    session = scoped_session(sessionmaker(bind=engine, autoflush=True, autocommit=True))
    Trade.session = session()
    Trade.query = session.query_property()
    Pair.session = session()
    Pair.query = session.query_property()
    Trend.session = session()
    Trend.query = session.query_property()
    _DECL_BASE.metadata.create_all(engine)

    # Clean dry_run DB
    if _CONF.get('dry_run', False) and _CONF.get('dry_run_db', False):
        clean_dry_run_db()


def cleanup() -> None:
    """
    Flushes all pending operations to disk.
    :return: None
    """
    Trade.session.flush()
    Pair.session.flush()
    Trend.session.flush()


def clean_dry_run_db() -> None:
    """
    Remove open_order_id from a Dry_run DB
    :return: None
    """
    for trade in Trade.query.filter(Trade.open_order_id.isnot(None)).all():
        # Check we are updating only a dry_run order not a prod one
        if 'dry_run' in trade.open_order_id:
            trade.open_order_id = None


class Trade(_DECL_BASE):
    """
    Class used to define a trade structure
    """
    __tablename__ = 'trades'

    id = Column(Integer, primary_key=True)
    exchange = Column(String, nullable=False)
    pair = Column(String, nullable=False)
    is_open = Column(Boolean, nullable=False, default=True)
    fee_open = Column(Float, nullable=False, default=0.0)
    fee_close = Column(Float, nullable=False, default=0.0)
    open_rate = Column(Float)
    close_rate = Column(Float)
    close_profit = Column(Float)
    stake_amount = Column(Float, nullable=False)
    amount = Column(Float)
    open_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    close_date = Column(DateTime)
    open_order_id = Column(String)

    def __repr__(self):
        return 'Trade(id={}, pair={}, amount={:.8f}, open_rate={:.8f}, open_since={})'.format(
            self.id,
            self.pair,
            self.amount,
            self.open_rate,
            arrow.get(self.open_date).humanize() if self.is_open else 'closed'
        )

    def update(self, order: Dict) -> None:
        """
        Updates this entity with amount and actual open/close rates.
        :param order: order retrieved by exchange.get_order()
        :return: None
        """
        # Ignore open and cancelled orders
        if order['status'] == 'open' or order['price'] is None:
            return

        logger.info('Updating trade (id=%d) ...', self.id)

        getcontext().prec = 8  # Bittrex do not go above 8 decimal
        if order['type'] == 'limit' and order['side'] == 'buy':
            # Update open rate and actual amount
            self.open_rate = Decimal(order['price'])
            self.amount = Decimal(order['amount'])
            logger.info('LIMIT_BUY has been fulfilled for %s.', self)
            self.open_order_id = None
        elif order['type'] == 'limit' and order['side'] == 'sell':
            self.close(order['price'])
        else:
            raise ValueError('Unknown order type: {}'.format(order['type']))
        cleanup()

    def close(self, rate: float) -> None:
        """
        Sets close_rate to the given rate, calculates total profit
        and marks trade as closed
        """
        self.close_rate = Decimal(rate)
        self.close_profit = self.calc_profit_percent()
        self.close_date = datetime.utcnow()
        self.is_open = False
        self.open_order_id = None
        logger.info(
            'Marking %s as closed as the trade is fulfilled and found no open orders for it.',
            self
        )

    def calc_open_trade_price(
            self,
            fee: Optional[float] = None) -> float:
        """
        Calculate the open_rate in BTC
        :param fee: fee to use on the open rate (optional).
        If rate is not set self.fee will be used
        :return: Price in BTC of the open trade
        """
        getcontext().prec = 8

        buy_trade = (Decimal(self.amount) * Decimal(self.open_rate))
        fees = buy_trade * Decimal(fee or self.fee_open)
        return float(buy_trade + fees)

    def calc_close_trade_price(
            self,
            rate: Optional[float] = None,
            fee: Optional[float] = None) -> float:
        """
        Calculate the close_rate in BTC
        :param fee: fee to use on the close rate (optional).
        If rate is not set self.fee will be used
        :param rate: rate to compare with (optional).
        If rate is not set self.close_rate will be used
        :return: Price in BTC of the open trade
        """
        getcontext().prec = 8

        if rate is None and not self.close_rate:
            return 0.0

        sell_trade = (Decimal(self.amount) * Decimal(rate or self.close_rate))
        fees = sell_trade * Decimal(fee or self.fee_close)
        return float(sell_trade - fees)

    def calc_profit(
            self,
            rate: Optional[float] = None,
            fee: Optional[float] = None) -> float:
        """
        Calculate the profit in BTC between Close and Open trade
        :param fee: fee to use on the close rate (optional).
        If rate is not set self.fee will be used
        :param rate: close rate to compare with (optional).
        If rate is not set self.close_rate will be used
        :return:  profit in BTC as float
        """
        open_trade_price = self.calc_open_trade_price()
        close_trade_price = self.calc_close_trade_price(
            rate=(rate or self.close_rate),
            fee=(fee or self.fee_close)
        )
        return float("{0:.8f}".format(close_trade_price - open_trade_price))

    def calc_profit_percent(
            self,
            rate: Optional[float] = None,
            fee: Optional[float] = None) -> float:
        """
        Calculates the profit in percentage (including fee).
        :param rate: rate to compare with (optional).
        If rate is not set self.close_rate will be used
        :param fee: fee to use on the close rate (optional).
        :return: profit in percentage as float
        """
        getcontext().prec = 8

        open_trade_price = self.calc_open_trade_price()
        close_trade_price = self.calc_close_trade_price(
            rate=(rate or self.close_rate),
            fee=(fee or self.fee_close)
        )

        return float("{0:.8f}".format((close_trade_price / open_trade_price) - 1))

class Pair(_DECL_BASE):
    __tablename__ = 'pairs'

    id = Column(Integer, primary_key=True)
    pair = Column(String, nullable=False)
    exchange = Column(String, nullable=True)
    trading = Column(Boolean, nullable=True, default=True)
    last_rate = Column(Float, default=0)
    supports = Column(ScalarListType, nullable=True)
    resistences = Column(ScalarListType, nullable=True)
    trends = Column(ScalarListType, nullable=True)
    day_trend = Column(ScalarListType, nullable=True)
    hour_trend = Column(ScalarListType, nullable=True)
    minute_trend = Column(ScalarListType, nullable=True)
    minute_trend = Column(ScalarListType, nullable=True)
    pivots_update_date = Column(DateTime, default=None)

    def __repr__(self):
        return 'Pair(pair={}, rate={:.8f}, sup={}, res={})'.format(
            self.pair,
            self.last_rate,
            str(self.supports),
            str(self.resistences)
        )



    # def get_trend_lines(self) -> list:
    #     # Check if is time to update pivots
    #     # return self.update_pivots()
    #
    #     # if self.current_rate < self.:
    #     #     logger.info('%s\'s pivots outdated by (%s), go find\'em now!',
    #     #                    self.pair, (datetime.utcnow() - self.pivots_update_date).seconds // 60)
    #     #     # print ('update pivots: ', self.pivots)
    #     #     return self.update_pivots()
    #     # else:
    #     #     pivots = {'sup': self.supports,
    #     #                 'res': self.resistences}
    #     #     return pivots
    #     # else:
    #     #     return self.update_pivots()
    #     self.day_trend, self.main_maxslope, self.main_minslope = cactix.gentrends(df1d)
    #     self.hour_trend, self.hour_maxslope, self.hour_minslope = cactix.gentrends(df1h)
    #     #
    #     return cactix.gentrends(df)

    def populate_trend_lines(self, df: DataFrame, timeframe: str) -> list:
        trends = Trend.query.filter(
                            (getattr(Trend,'pair')==self.pair) &
                            (getattr(Trend,'last') == True) &
                            (getattr(Trend,'conf_n') > 0) &
                            (getattr(Trend,'slope') > 0)).all()
        print ('Populating trends: ', len(trends))
        if len(trends) == 0:
            self.update_trend_lines(df)
            trends = Trend.query.filter(
                                (getattr(Trend,'pair')==self.pair) &
                                (getattr(Trend,'last') == True) &
                                (getattr(Trend,'conf_n') > 0) &
                                (getattr(Trend,'slope') > 0)).all()
            # print (trends)
        for t in trends:
            ax = t.ax
            ay = t.ay
            ad = t.ad
            bx = t.bx
            by = t.by
            bd = t.bd

            ind = len(df.loc[df.date == ad])

            # print (ind)

            if ind != None:
                # logger.info('Trend points in df range')
                # d = DataFrame(0, index=np.arange(len(df)), columns=['trend'])
                slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
                trend = polyval([slope,intercept],df.index)
                trend_name = 'trend-'+str(ay)+'|'+str(by)
                # print (trend)
                df.loc[:,trend_name] = trend
                # print ('update pivots: ', self.pivots)
            else:
                # logger.info('Trend points out df range')
                df_first_date = df.iloc[0].date.tz_localize(None).to_pydatetime()
                diff_min = (df_first_date - ad).total_seconds() / 60.0
                d = np.arange(len(df) + diff_min)
                slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
                trend = polyval([slope,intercept],d)[len(df):]
                trend_name = 'trend-'+str(ay)+'|'+str(by)
                df.loc[:,trend_name] = trend

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
        print (df.columns)

        return df

    # def get_trends(self) -> list:
    #     # Check if is time to update pivots
    #     # return self.update_pivots()
    #     if self.pivots_update_date:
    #         if self.pivots_update_date < datetime.utcnow() - timedelta(minutes=(30)):
    #             logger.info('%s\'s pivots outdated by (%s), go find\'em now!',
    #                            self.pair, (datetime.utcnow() - self.pivots_update_date).seconds // 60)
    #             # print ('update pivots: ', self.pivots)
    #             return self.update_pivots()
    #         else:
    #             pivots = {'sup': self.supports,
    #                         'res': self.resistences}
    #             return pivots
    #     else:
    #         return self.update_pivots()

    def update_trend_lines(self, df: DataFrame) -> list:
        """
        Updates this entity with amount and actual open/close rates.
        :param order: order retrieved by exchange.get_order()
        :return: None
        """

        logger.info('Updating pair %s trend lines...', self.pair)

        new_trends = gentrends(self, df, pair=self.pair, charts=True)
        if len(new_trends)>0:
            logger.info('Updating all trends for %s', self.pair)
            prev_trends = Trend.query.filter(Trend.pair.is_(self.pair)).all()
            for pt in prev_trends:
                pt.delete()
            for t in new_trends:
                ax = t['a'][0]
                ay = t['a'][1]
                ad = t['a'][2]
                bx = t['b'][0]
                by = t['b'][1]
                bd = t['b'][2]
                # prev_trend = Trend.query.filter(Trend.a.is_(a) and Trend.b.is_(b)).all()
                # if prev_trend:
                #     logger.info('Updating trend a: %s | b: %s', a, b)
                #     prev_trend.delete()
                trend_obj = Trend(
                    pair=self.pair,
                    type = t['type'],
                    timeframe = t['timeframe'],
                    ax = ax,
                    ay = ay,
                    ad = ad,
                    bx = bx,
                    by = by,
                    bd = bd,
                    slope = t['slope'],
                    conf_n = t['conf_n']
                )
                Trend.session.add(trend_obj)
                Trend.session.flush()

        return new_trends


    def get_pivots(self) -> list:
        # Check if is time to update pivots
        # return self.update_pivots()
        if self.pivots_update_date:
            if self.pivots_update_date < datetime.utcnow() - timedelta(minutes=(30)):
                logger.info('%s\'s pivots outdated by (%s), go find\'em now!',
                               self.pair, (datetime.utcnow() - self.pivots_update_date).seconds // 60)
                # print ('update pivots: ', self.pivots)
                return self.update_pivots()
            else:
                pivots = {'sup': self.supports,
                            'res': self.resistences}
                return pivots
        else:
            return self.update_pivots()

    def update_pivots(self) -> list:
        """
        Updates this entity with amount and actual open/close rates.
        :param order: order retrieved by exchange.get_order()
        :return: None
        """

        logger.info('Updating pair %s ...', self.pair)

        getcontext().prec = 8  # Bittrex do not go above 8 decimal

        # print (df)
        supports = get_pivots(self.pair, piv_type='sup')
        resistences = get_pivots(self.pair, piv_type='res')
        pivots = {'sup': supports,
                    'res': resistences}
        # print (pivots)
        self.supports = supports
        self.resistences = resistences
        self.pivot_update_date = datetime.utcnow()

        cleanup()
        return pivots


class Trend(_DECL_BASE):
    __tablename__ = 'trends'

    id = Column(Integer, primary_key=True)
    pair = Column(String, nullable=False)
    type = Column(String, nullable=False)
    last = Column(Boolean, nullable=False, default=True)
    timeframe = Column(String, nullable=False, default='all')
    ax = Column(Float, nullable=False)
    ay = Column(Float, nullable=False)
    ad = Column(DateTime, nullable=False)
    bx = Column(Float, nullable=False)
    by = Column(Float, nullable=False)
    bd = Column(DateTime, nullable=False)
    slope = Column(Integer, nullable=False)
    conf_n = Column(Integer, nullable=False, default=0)
    update_date = Column(DateTime, nullable=False, default=datetime.utcnow)

    def __repr__(self):
        return 'Trend(pair={}, type={}, last={}, timeframe={}, a={}, b={}, slope={}, confirm_n={}, last update={})'.format(
            self.pair,
            self.type,
            self.last,
            self.timeframe,
            str([self.ax,self.ay]),
            str([self.bx,self.by]),
            str(self.slope),
            str(self.conf_n),
            arrow.get(self.update_date).humanize()
        )



    # def transpose_last_trends(self, pair, df) -> list:
    #     # self.a[0]
    #     df.loc[h.index[i]:,trend_name] = trend_next_wave
    #     #
    #     return cactix.gentrends(df)
