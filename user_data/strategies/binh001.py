# --- Do not remove these libs ---
from freqtrade.strategy.interface import IStrategy
from typing import Dict, List
from hyperopt import hp
from functools import reduce
from pandas import DataFrame
# --------------------------------

# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
import numpy # noqa


# Update this variable if you change the class name
class_name = 'MyStrategy'


# This class is a sample. Feel free to customize it.
class binh001(IStrategy):
    minimal_roi = {
        "60":  0.0,
        "50":  0.001,
        "40":  0.005,
        "30":  0.015,
        "0":  0.02
    }
    stoploss = -0.50
    ticker_interval = "5m"

    def populate_indicators(self, dataframe: DataFrame) -> DataFrame:
       dataframe['rsi'] = ta.RSI(dataframe, timeperiod=5)
       rsiframe = DataFrame(dataframe['rsi']).rename(columns={'rsi': 'close'})
       dataframe['emarsi'] = ta.EMA(rsiframe, timeperiod=5)
       macd = ta.MACD(dataframe)
       dataframe['macd'] = macd['macd']
       dataframe['adx'] = ta.ADX(dataframe)
       dataframe['tema'] = ta.TEMA(dataframe, timeperiod=9)
       dataframe['ema5'] = ta.EMA(dataframe, timeperiod=5)

       return dataframe

    def populate_buy_trend(self, dataframe: DataFrame) -> DataFrame:
       dataframe.loc[
           (
             dataframe['adx'].gt(20) &
             dataframe['emarsi'].le(40) &
             dataframe['macd'].lt(0) &
             dataframe['macd'].lt(dataframe['macd'].shift(1))
           ),
           'buy'] = 1

       return dataframe

    def populate_sell_trend(self, dataframe: DataFrame) -> DataFrame:
       dataframe.loc[
           (
             dataframe['adx'].gt(25) &
             dataframe['macd'].gt(0) &
             dataframe['emarsi'].ge(70) &
             (dataframe['ema5'] <= dataframe['ema5'].shift(1))
           ),
           'sell'] = 1
       return dataframe

    def hyperopt_space(self) -> List[Dict]:
       return False

    def buy_strategy_generator(self, params) -> None:
       return False
