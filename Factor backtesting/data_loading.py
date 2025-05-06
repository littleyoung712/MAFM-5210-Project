import pandas as pd
from typing import (
    List,
    Optional,
)
import akshare as ak
from datetime import datetime as dt


class kline(object):  # 返回
    PATH = 'sqlite:///' + r"D:\大学文档\MAFM\Mathematical Models of Investment\project1\ashare_price.db"  # path of db storage
    Data = None

    def __init__(self, codes: Optional[List[str]] = None, start: Optional[str] = None, end: Optional[str] = None):
        trading_days = self.get_trading_days(start, end)
        self.Data = self.load_data(codes, trading_days)
        pass

    def __getitem__(self, date: pd.Timestamp) -> pd.DataFrame:
        return self.Data.loc[date]

    @staticmethod
    def get_trading_days(start: Optional[str] = None, end: Optional[str] = None) -> Optional[List[str]]:
        if start:
            if end:
                trading_days = pd.to_datetime(ak.stock_zh_index_daily_em('sz399300', start, end)['date'])
            else:
                trading_days = pd.to_datetime(ak.stock_zh_index_daily_em('sz399300', start)['date'])
            return trading_days.map(lambda x: dt.strftime(x, '%Y%m%d')).tolist()
        else:
            if end:
                trading_days = pd.to_datetime(ak.stock_zh_index_daily_em('sz399300', end_date=end)['date'])
                return trading_days.map(lambda x: dt.strftime(x, '%Y%m%d')).tolist()
            else:
                return None


    @staticmethod
    def generate_sql_syntax(codes: Optional[List[str]] = None, dates: Optional[List[str]] = None) -> str:
        if codes:
            if dates:
                syntax = ('SELECT S_INFO_WINDCODE,TRADE_DT,S_DQ_ADJCLOSE,S_DQ_TRADESTATUSCODE,S_DQ_CLOSE,S_DQ_HIGH,S_DQ_LOW,'
                          f'S_DQ_AVGPRICE FROM AShareEODPrices WHERE TRADE_DT in {tuple(dates)} and S_INFO_WINDCODE in {tuple(codes)}')
            else:
                syntax = ('SELECT S_INFO_WINDCODE,TRADE_DT,S_DQ_ADJCLOSE,S_DQ_TRADESTATUSCODE,S_DQ_CLOSE,S_DQ_HIGH,S_DQ_LOW,'
                          f'S_DQ_AVGPRICE FROM AShareEODPrices WHERE S_INFO_WINDCODE in {tuple(codes)}')
        else:
            if dates:
                syntax = ('SELECT S_INFO_WINDCODE,TRADE_DT,S_DQ_ADJCLOSE,S_DQ_TRADESTATUSCODE,S_DQ_CLOSE,S_DQ_AVGPRICE,S_DQ_HIGH,S_DQ_LOW '
                          f'FROM AShareEODPrices WHERE TRADE_DT in {tuple(dates)}')
            else:
                syntax = ('SELECT S_INFO_WINDCODE,TRADE_DT,S_DQ_ADJCLOSE,S_DQ_TRADESTATUSCODE,S_DQ_CLOSE,S_DQ_AVGPRICE,S_DQ_HIGH,S_DQ_LOW '
                          'FROM AShareEODPrices')

        return syntax

    def load_data(self, codes: Optional[List[str]] = None, dates: Optional[List[str]] = None) -> pd.DataFrame:
        # 需要输入的是获取的所有代码,以及回测时间:
        sql_syntax = self.generate_sql_syntax(codes, dates)
        data = pd.read_sql(
            sql_syntax
            , self.PATH)
        data['TRADE_DT'] = data['TRADE_DT'].astype(str)
        data.columns = ['st_code', 'trade_date', 'close_adj', 'trade_state', 'close', 'avg_price','high','low']
        return data

    def get_data(self, indicator: str) -> pd.DataFrame:
        try:
            pivot_data = pd.pivot_table(data=self.Data, values=indicator, index='trade_date', columns='st_code')
            return pivot_data
        except Exception as e:
            print(f"Error in get_data method: {e}")
            print("Indicator must be one of 'close_adj','trade_state','close','avg_price','high','low'")


if __name__ == '__main__':
    my_data = kline(start='20240101',end='20250101')
    b=my_data.get_data('high')


