"""
This module contains the class to persist trades into SQLite
"""

import logging
from datetime import datetime
from decimal import Decimal, getcontext
from typing import Dict, Optional, Any

import arrow
from sqlalchemy import (Boolean, Column, DateTime, Float, Integer, String,
                        create_engine, ForeignKey)

from pandas import DataFrame, to_datetime
from sqlalchemy import inspect
from sqlalchemy.exc import NoSuchModuleError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm.scoping import scoped_session
from sqlalchemy.orm.session import sessionmaker
from sqlalchemy.orm import relationship, backref
from sqlalchemy.pool import StaticPool

from sqlalchemy_utils import ScalarListType

from freqtrade.exchange import get_ticker_history
from freqtrade.optimize import load_tickerdata_file


from freqtrade.indicators import get_pivots

from freqtrade.trends import gentrends, plot_trends
from freqtrade.vendor.zigzag_hi_lo import *

from scipy import linspace, polyval, polyfit, sqrt, stats, randn
import numpy as np

from freqtrade import OperationalException

logger = logging.getLogger(__name__)

_CONF = {}
_DECL_BASE: Any = declarative_base()


def init(config: Dict) -> None:
    """
    Initializes this module with the given config,
    registers all known command handlers
    and starts polling for message updates
    :param config: config to use
    :return: None
    """
    _CONF.update(config)

    db_url = _CONF.get('db_url', None)
    kwargs = {}

    # Take care of thread ownership if in-memory db
    if db_url == 'sqlite://':
        kwargs.update({
            'connect_args': {'check_same_thread': False},
            'poolclass': StaticPool,
            'echo': False,
        })

    try:
        engine = create_engine(db_url, **kwargs)
    except NoSuchModuleError:
        error = 'Given value for db_url: \'{}\' is no valid database URL! (See {}).'.format(
            db_url, 'http://docs.sqlalchemy.org/en/latest/core/engines.html#database-urls'
        )
        raise OperationalException(error)

    session = scoped_session(sessionmaker(bind=engine, autoflush=False, autocommit=True))
    Trade.session = session()
    Trade.query = session.query_property()
    Pair.session = session()
    Pair.query = session.query_property()
    Trend.session = session()
    Trend.query = session.query_property()
    _DECL_BASE.metadata.create_all(engine)
    check_migrate(engine)

    # Clean dry_run DB if the db is not in-memory
    if _CONF.get('dry_run', False) and db_url != 'sqlite://':
        clean_dry_run_db()


def has_column(columns, searchname: str) -> bool:
    return len(list(filter(lambda x: x["name"] == searchname, columns))) == 1


def check_migrate(engine) -> None:
    """
    Checks if migration is necessary and migrates if necessary
    """
    inspector = inspect(engine)

    cols = inspector.get_columns('trades')

    if not has_column(cols, 'fee_open'):
        # Schema migration necessary
        engine.execute("alter table trades rename to trades_bak")
        # let SQLAlchemy create the schema as required
        _DECL_BASE.metadata.create_all(engine)

        # Copy data back - following the correct schema
        engine.execute("""insert into trades
                (id, exchange, pair, is_open, fee_open, fee_close, open_rate,
                open_rate_requested, close_rate, close_rate_requested, close_profit,
                stake_amount, amount, open_date, close_date, open_order_id)
            select id, lower(exchange),
                case
                    when instr(pair, '_') != 0 then
                    substr(pair,    instr(pair, '_') + 1) || '/' ||
                    substr(pair, 1, instr(pair, '_') - 1)
                    else pair
                    end
                pair,
                is_open, fee fee_open, fee fee_close,
                open_rate, null open_rate_requested, close_rate,
                null close_rate_requested, close_profit,
                stake_amount, amount, open_date, close_date, open_order_id
                from trades_bak
             """)

        # Reread columns - the above recreated the table!
        inspector = inspect(engine)
        cols = inspector.get_columns('trades')

    if not has_column(cols, 'open_rate_requested'):
        engine.execute("alter table trades add open_rate_requested float")
    if not has_column(cols, 'close_rate_requested'):
        engine.execute("alter table trades add close_rate_requested float")


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
    open_rate_requested = Column(Float)
    close_rate = Column(Float)
    close_rate_requested = Column(Float)
    close_profit = Column(Float)
    stake_amount = Column(Float, nullable=False)
    amount = Column(Float)
    open_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    close_date = Column(DateTime)
    open_order_id = Column(String)
    # trend_id = Column(Integer, ForeignKey('trends.id'))
    sup_trend = Column(Integer, nullable=False)
    res_trend = Column(Integer, nullable=False)

    def __repr__(self):
        return 'Trade(id={}, pair={}, amount={:.8f}, open_rate={:.8f}, open_since={}, max_t={}, min_t={})'.format(
            self.id,
            self.pair,
            self.amount,
            self.open_rate,
            arrow.get(self.open_date).humanize() if self.is_open else 'closed',
            self.res_trend,
            self.sup_trend
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
    pivots_update_date = Column(DateTime, default=None)
    res_trend = Column(Integer, nullable=True)
    sup_trend = Column(Integer, nullable=True)

    def __repr__(self):
        return 'Pair(pair={}, rate={:.8f}, sup={}, res={}, tmax={}, tmin={})'.format(
            self.pair,
            self.last_rate,
            str(self.supports),
            str(self.resistences),
            str(self.res_trend),
            str(self.sup_trend)
        )



    def get_trade_trends(self, df: DataFrame, interval: str) -> list:

        # trends = Trend.query.filter_by(
        #                     pair_id=self.id
        #                     ).all()
        #
        # print(trends)
        max_trend = self.res_trend
        min_trend = self.sup_trend

        return max_trend, min_trend
        # print(df)


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

    # def populate_trend_lines(self, df: DataFrame, interval: str, update: bool=False) -> list:
    #
    #     res, sup = self.res_trend, self.sup_trend
    #     df = res.populate_to_df(df)
    #     df = sup.populate_to_df(df)
    #
    #     return df

    def update_trend_lines(self, df: DataFrame, interval: str):
        """
        Updates this entity with amount and actual open/close rates.
        :param order: order retrieved by exchange.get_order()
        :return: None
        """

        logger.info('Persistence update_trend_lines: Updating pair %s trend lines...', self.pair)

        # df = get_df(self.pair, interval)
        # print (interval)


        interval_volat = {
                    '1d':1,
                    '1h':1,
                    '30m':1,
                    '5m':1,
                    '1m':1}

        prop = int(len(df)/50)
        volat_window = {
                    '1d':10 ,
                    '1h':20,
                    '30m':100,
                    '5m':100,
                    '1m':prop
                    }

        window = volat_window[interval]
        df['bb_exp'] = (df.bb_upperband.rolling(window=window).max() - df.bb_lowerband.rolling(window=window).min()) / df.bb_upperband.rolling(window=window).max() * interval_volat[interval]
        # df['bb_exp'] = (df.bb_upperband - df.bb_lowerband) / df.bb_upperband  * interval_volat[interval]
        pivots = peak_valley_pivots(df.low.values, df.high.values, df.bb_exp.values)
        df['pivots'] = np.transpose(np.array((pivots)))

        new_trends = gentrends(self, df, pair=self.pair, charts=True, interval=interval)
        if len(new_trends)>0:
            logger.info('Persistence: update_trend_lines: Updating all trends for %s', self.pair)
            prev_trends = Trend.query.filter_by(pair=self.pair, interval=interval).all()
            # self.res_trend = None
            # self.sup_trend = None
            # print (prev_trends)
            if prev_trends:
                for pt in prev_trends:
                    Trend.session.delete(pt)
                    cleanup()
            for t in new_trends:
                # print(t)
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
                    last = t['last'],
                    max_trend = t['max'],
                    min_trend = t['min'],
                    interval = t['interval'],
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
                cleanup()
        else:
            print ('no trends')
        return self

    def get_pivots(self, df: DataFrame) -> list:
        # Check if is time to update pivots
        # return self.update_pivots()
        if self.pivots_update_date:
            if self.pivots_update_date < datetime.utcnow() - timedelta(minutes=(30)):
                logger.info('%s\'s pivots outdated by (%s), go find\'em now!',
                               self.pair, (datetime.utcnow() - self.pivots_update_date).seconds // 60)
                # print ('update pivots: ', self.pivots)
                return self.update_pivots(df)
            else:
                pivots = {'sup': self.supports,
                            'res': self.resistences}
                return pivots
        else:
            return self.update_pivots()

    def update_pivots(self, df: DataFrame) -> list:
        """
        Updates this entity with amount and actual open/close rates.
        :param order: order retrieved by exchange.get_order()
        :return: None
        """

        logger.info('Updating pair %s ...', self.pair)

        getcontext().prec = 8  # Bittrex do not go above 8 decimal

        # print (df)
        supports = get_pivots(df, self.pair, piv_type='sup')
        resistences = get_pivots(df, self.pair, piv_type='res')
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
    last = Column(Boolean, default=False)
    max_trend = Column(Boolean, default=False)
    min_trend = Column(Boolean, default=False)
    interval = Column(String, nullable=False, default='all')
    ax = Column(Float, nullable=False)
    ay = Column(Float, nullable=False)
    ad = Column(DateTime, nullable=False)
    bx = Column(Float, nullable=False)
    by = Column(Float, nullable=False)
    bd = Column(DateTime, nullable=False)
    slope = Column(Integer, nullable=False)
    conf_n = Column(Integer, nullable=False, default=0)
    update_date = Column(DateTime, nullable=False, default=datetime.utcnow)
    # trades_id = Column(Integer, ForeignKey('trades.id'))
    # trades_sup = Column(Integer, nullable=False)
    # trades_res = Column(Integer, nullable=False)
    # trades_sup = relationship("Trade", backref="sup_trend")
    # trades_res = relationship("Trade", backref="res_trend")

    def __repr__(self):
        return 'Trend(pair={}, type={}, last={}, max={}, min={}, interval={}, a={}, b={}, slope={}, confirm_n={}, last update={})'.format(
            self.pair,
            self.type,
            self.last,
            self.max_trend,
            self.min_trend,
            self.interval,
            str([self.ax,self.ay]),
            str([self.bx,self.by]),
            str(self.slope),
            str(self.conf_n),
            arrow.get(self.update_date).humanize()
        )


    def populate_to_df(self, df: DataFrame) -> DataFrame:
        ax = self.ax
        ay = self.ay
        ad = self.ad
        bx = self.bx
        by = self.by
        bd = self.bd
        df['date'] = to_datetime(df['date']).dt.tz_localize(None)

        # print (df.date)
        # print (ad)

        ind = df.index[df.date == ad]

        # print (ind)

        # if self.conf_n > 2:
            # print (t)

        if ind != None:
            slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
            trend = polyval([slope,intercept],df.index)
            trend_name = 'trend-'+str(ay)+'|'+str(by)
        else:
            df_first_date = df.iloc[0].date
            diff_min = (df_first_date - ad).total_seconds() / 60.0
            d = np.arange(len(df) + diff_min)
            # print (d)
            slope, intercept, r_value, p_value, std_err = stats.linregress([ax, bx], [ay, by])
            trend = polyval([slope,intercept],d)[-len(df):]
            trend_name = 'trend-'+str(ay)+'|'+str(by)
            # print ('different size dataframes')
            # print ('diff mins: ', diff_min)
            # print ('len df: ', len(df))
            # print ('len trend: ', len(trend))

        # print (self)
        # print (self.max_trend, self.min_trend)
        if self.max_trend:
            df.loc[:,'max'] = trend
        elif self.min_trend:
            df.loc[:,'min'] = trend
        else:
            df.loc[:,trend_name] = trend
        return df

    # def transpose_last_trends(self, pair, df) -> list:
    #     # self.a[0]
    #     df.loc[h.index[i]:,trend_name] = trend_next_wave
    #     #
    #     return cactix.gentrends(df)
