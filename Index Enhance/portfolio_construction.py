import copy
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from data_loading import Kline
from typing import (
    List,
    Optional,
)
start_date = '20200101'
end_date = '20250401'


class Multifactors_Select(object):
    SAVE_PATH = r"../../result/Backtest"  # save_path
    shift_dict = {'D': 1, 'W': 5, 'M': 21, 'Y': 252}
    def __init__(self, factor_df: pd.DataFrame, factor_name: Optional[str] = None, codes: Optional[List[str]] = None,
                 start: Optional[str] = None,
                 end: Optional[str] = None, freq: str = 'D',pool:Optional[pd.DataFrame]=None):
        self.portfolio = None
        self.factors = factor_df
        self.freq = self.shift_dict[freq]
        self.name = factor_name
        self.codes = codes
        self.Kline = Kline(codes, start, end)
        self.trading_day = pd.to_datetime(self.Kline.get_trading_days(start, end))
        self.trading_day = self.trading_day[self.trading_day.isin(self.factors.index)][:-1]  # 因子信号发出后一天进行调仓，因此最后因子不用，没法进行调调仓
        if pool is not None:
           self.pool = pool[self.trading_day]
        else:
            self.pool = None
        self.factors = self.align(original_factor=self.factors, dates=self.trading_day.tolist(), codes=self.codes)
        self.trade_state_next = self.align(self.get_trading_state_next(), dates=self.trading_day.tolist(),
                                           codes=self.codes)  # 只提取有因子数据的

    @staticmethod
    def align(original_factor: pd.DataFrame, dates: Optional[List[pd.Timestamp]] = None,
              codes: Optional[List[str]] = None) -> pd.DataFrame:  # 只选用在池子内，且在交易日期内的factor
        if codes:
            if dates:
                use_factor = original_factor.loc[dates, codes]
            else:
                use_factor = original_factor[codes]
        else:
            if dates:
                use_factor = original_factor.loc[dates]
            else:
                use_factor = original_factor
        return use_factor


    def get_trading_state_next(self) -> pd.DataFrame:
        trade_state = self.Kline.get_data('trade_state')
        high = self.Kline.get_data('high')
        low = self.Kline.get_data('low')
        trade_state.index, low.index, high.index = pd.to_datetime(trade_state.index), pd.to_datetime(
            trade_state.index), pd.to_datetime(trade_state.index)
        flag = high != low
        trade_permission = flag * trade_state
        trade_permission[trade_permission == -1] = 1
        trade_permission[trade_permission != 1] = 0
        trade_state_next = (trade_permission.shift(-1)).dropna(how='all', axis=0)
        return trade_state_next

    def st_quantile_sel(self, num:int=50, flag:int=-1,
                        agg_holding:pd.Series=None,
                        ):
        # freq为频率，日频为1，周频为5，月频21，年252
        if agg_holding is None:
            agg_holding = pd.Series(index=self.trading_day, data=[1 for i in range(len(self.trading_day))])
        factor_df = pd.DataFrame()  # 需要去除没有交易的日期,换不同的因子需要重新调整.-1为反向因子，1为正向
        data = copy.deepcopy(self.factors)
        use_stock = data.columns
        date_range = self.trading_day[::self.freq]
        for date in tqdm(date_range):
            if date in data.index:
                if self.pool is not None:
                    use_stock = self.pool[date].dropna()
                use_stock = self.trade_state_next.columns[self.trade_state_next.columns.isin(use_stock)]  # 筛选交易的，非退市的
                trade_stock = self.trade_state_next.loc[date,use_stock]
                selected_factor = data.loc[[date], trade_stock[trade_stock==1].index].T.dropna()
                if len(selected_factor) > 0:
                    quantile = min(num / len(selected_factor), 1) if num > 1 else num
                    if flag == -1:
                        factor_quantiles = selected_factor.quantile([0, quantile, 1])
                        selected_factor = selected_factor[
                            (selected_factor <= factor_quantiles.iloc[1]) & (
                                    selected_factor != 0)].dropna()
                        selected_factor = selected_factor.sort_values(by=date,ascending=True)
                    else:
                        factor_quantiles = selected_factor.quantile([0, 1-quantile, 1])
                        selected_factor = selected_factor[
                            (selected_factor >= factor_quantiles.iloc[1]) & (
                                    selected_factor != 0)].dropna()
                        selected_factor = selected_factor.sort_values(by=date, ascending=False)
                    next_pos = np.where(self.trading_day == date)[0][0] + 1
                    if next_pos<len(self.trading_day):
                        selected_factor.loc[:, 'trade_date'] = [
                            self.trading_day[np.where(self.trading_day == date)[0][0] + 1] for j in
                            range(len(selected_factor))]  # 信号发出后一天持仓
                        selected_factor.loc[:, 'weight'] = [agg_holding[date] / len(selected_factor) for i in
                                                            range(len(selected_factor))]
                        selected_factor = selected_factor.rename(columns={date:'values'})
                        factor_df = pd.concat([factor_df, selected_factor], axis=0)
        factor_df.index.name = 'st_code'
        factor_df.columns.name = ''
        self.portfolio = factor_df
        return factor_df
    def save_data(self,freq:str,file_name:str):
        self.portfolio.to_csv(os.path.join(self.SAVE_PATH ,f'{freq}', fr'{file_name}.csv'), encoding='gbk')










if __name__ == '__main__':
    acf = pd.read_parquet(
        r"../../data/Backtest/acf_factor.parquet")  # your file path
    acf = pd.pivot_table(index='TRADE_DT', columns='S_INFO_WINDCODE', values='acf', data=acf)
    acf.index = pd.to_datetime(acf.index)
    zz1000_cons = pd.read_csv(f"../../data/Backtest/中证1000成分股.csv")
    zz1000_cons.columns = pd.to_datetime(zz1000_cons.columns)
    all_cons = list(zz1000_cons.unstack().dropna().unique())
    f = 'W'
    ms = Multifactors_Select(acf,'acf',start=start_date,end=end_date,pool=zz1000_cons,codes=all_cons,freq=f)
    wt = ms.st_quantile_sel(50,1)
    ms.save_data(f,'acf因子等权组合')
