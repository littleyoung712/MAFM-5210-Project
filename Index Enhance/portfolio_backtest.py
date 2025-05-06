import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from data_loading import Kline
from datetime import datetime as dt
from tqdm import tqdm


start = '20200101'
end = '20250401'

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
fee = 0.0002  # 回测前调整手续费






class PortfolioBacktester(object):
    def __init__(self,start_date:str, end_date:str,portfolio_weight:pd.DataFrame):  # portfolio_weight: three columns:st_code, trade_date, weight
        self.start_date = start_date
        self.end_date = end_date
        self.buy_df = pd.pivot_table(index='trade_date',columns='st_code',values='weight',data=portfolio_weight)
        self.buy_df.index = pd.to_datetime(self.buy_df.index)
        self.buy_df = self.buy_df.sort_index()
        use_codes = list(self.buy_df.columns)
        self.Kline = Kline(start=self.start_date,end=self.end_date,codes=use_codes)
        self.trading_day = Kline.get_trading_days(start=self.start_date,end=self.end_date)

    @staticmethod
    def calculate_return(portfolio:pd.Series, capital:float, trade_price:pd.Series, multi:int, flag:int=0):  # 如果是可转债，则为1，否则0
        operation_df = pd.DataFrame(index=portfolio.index, columns=['diff_volume', 'buy_sign', 'sell_sign'])
        operation_df['diff_volume'] = portfolio
        operation_df.loc[operation_df['diff_volume'] > 0, 'buy_sign'] = 1 + fee
        operation_df.loc[operation_df['diff_volume'] <= 0, 'buy_sign'] = 0
        operation_df.loc[operation_df['diff_volume'] < 0, 'sell_sign'] = 1 - fee * flag
        operation_df.loc[operation_df['diff_volume'] >= 0, 'sell_sign'] = 0
        bp_sp_df = pd.concat([trade_price, trade_price], axis=1).loc[operation_df.index]  # 这里需要调整，调整购买和卖出的价格

        x = (bp_sp_df.values * operation_df[['buy_sign', 'sell_sign']].values).sum(axis=1)
        capital -= x @ operation_df['diff_volume'] * multi
        return capital

    def backtest(self,trade_as:str='open',settle_as:str='close',capital:int=10000000,show_everyday:bool=False):
        multi = 100
        flag = 0
        date_range = self.buy_df.index[self.buy_df.index.isin(self.trading_day)]
        if show_everyday:
            date_range = pd.to_datetime(self.trading_day)
            date_range = date_range[date_range<=self.buy_df.index[-1]]
        cut_range = pd.to_datetime(np.quantile(date_range, [0, 0.33, 0.66, 1]))  # 分割检索更快
        cut_range = [dt(year=i.year, month=i.month, day=i.day) for i in cut_range]
        money_list = []
        date_list = []
        portfolio = pd.Series(index=self.buy_df.columns, data=[0 for i in range(len(self.buy_df.columns))])
        j = 0
        trade_prices = self.Kline.get_data(trade_as)
        settle_price = self.Kline.get_data(settle_as)
        trade_prices.index,settle_price.index = pd.to_datetime(trade_prices.index), pd.to_datetime(settle_price.index)
        for i in tqdm(range(len(cut_range))):
            if i != len(cut_range) - 1:
                select_range = date_range[(date_range >= cut_range[i]) & (date_range < cut_range[i + 1])]
            else:
                select_range = date_range[(date_range >= cut_range[i])]
            select_buy_df = self.buy_df.loc[self.buy_df.index.isin(select_range), :]
            for date in tqdm(select_range):
                daily_settle_price = settle_price.loc[date]
                if date in select_buy_df.index:
                    daily_trade_prices = trade_prices.loc[date]
                    new_portfolio = pd.Series(index=self.buy_df.columns,
                                              data=[0 for i in range(len(self.buy_df.columns))])
                    buy_list = select_buy_df.loc[date].dropna()

                    if j < len(date_range) - 1:
                        next_date = date_range[j + 1]
                        next_settle_price = settle_price.loc[next_date]
                        # 计算每只股票可以购买的数量
                        normal_st = list(set(daily_settle_price.dropna().index) & set(next_settle_price.dropna().index))
                        # 正常股，去除退市，st，同时保证两个交易日连续两个交易日都在交易，即不持有第二天停止交易的资产(
                        origin_sum = buy_list.sum()
                        buy_list = buy_list.loc[buy_list.index.isin(normal_st)]
                        # 不是最后一天，则需要调仓，最后一天平仓
                        asset = portfolio[portfolio != 0] @ daily_trade_prices.loc[
                            portfolio[portfolio != 0].index] * multi  # 这里需要改，看买的时候是在收盘还是开盘
                        buy_count = (capital + asset) * buy_list * origin_sum / buy_list.sum()
                        # 按照 buy_list 进行购买
                        for code in buy_list.index:
                            print('\r' + code, end='已完成', flush=True)
                            buy_price = daily_trade_prices.loc[code]
                            buy_volume = buy_count[code] // (buy_price * (1 + fee) * multi)
                            new_portfolio[code] = buy_volume
                        diff_portfolio = new_portfolio - portfolio
                        diff_portfolio = diff_portfolio[diff_portfolio != 0]
                        if len(diff_portfolio) > 0:
                            capital = self.calculate_return(diff_portfolio, capital,
                                                      daily_trade_prices, multi,
                                                       flag=flag)
                        portfolio = new_portfolio
                asset = portfolio[portfolio != 0] @ daily_settle_price.loc[portfolio[portfolio != 0].index] * multi  # 这里要改，算净值
                money_list.append(capital + asset)
                date_list.append(date)
                print(date)
                j += 1
        return money_list, date_list

    def index_backtest(self,index_data:pd.Series,capital:int=10000000,show_everyday:bool=False):
        date_range = self.buy_df.index[self.buy_df.index.isin(self.trading_day)]
        if show_everyday:
            date_range = pd.to_datetime(self.trading_day)
            date_range = date_range[date_range <= self.buy_df.index[-1]]
        use_index = index_data.loc[date_range]
        index_nv = use_index/use_index.iloc[0]*capital
        return index_nv,date_range












if __name__ == '__main__':
    PATH = r'../../'
    f = 'M'
    acf_ew_portfolio = pd.read_csv(os.path.join(PATH,'result','Backtest',rf'{f}\acf因子等权组合.csv'), encoding='gbk',)
    acf_mvo_portfolio = pd.read_csv(os.path.join(PATH,'result','Backtest',rf'{f}\acf因子MVO优化.csv'), encoding='gbk',)
    acf_te_portfolio =  pd.read_csv(os.path.join(PATH,'result','Backtest',rf'{f}\acf因子te优化.csv'), encoding='gbk',)
    # acf_te_portfolio['trade_date'] = pd.to_datetime(acf_te_portfolio['trade_date'])
    # acf_te_portfolio['weight'] = acf_te_portfolio.groupby(by='trade_date').apply(lambda x:x['weight']/x['weight'].sum()).values
    # acf_te_portfolio.groupby(by='trade_date').apply(lambda x: x['weight'].sum())

    zz1000_sr = pd.read_excel(os.path.join(PATH,'data','Backtest',"K线导出_000852_日线数据.xlsx"),index_col='trade_date',parse_dates=True)['close']
    PBCT_ew = PortfolioBacktester(portfolio_weight=acf_ew_portfolio,start_date=start,end_date=end)
    money_ew, date_ew = PBCT_ew.backtest(trade_as='avg_price',settle_as='close',show_everyday=False)
    PBCT_mvo = PortfolioBacktester(portfolio_weight=acf_mvo_portfolio,start_date=start,end_date=end)
    money_mvo, date_mvo = PBCT_mvo.backtest(trade_as='avg_price',settle_as='close',show_everyday=False)
    PBCT_te = PortfolioBacktester(portfolio_weight=acf_te_portfolio,start_date=start,end_date=end)
    money_te, date_te = PBCT_te.backtest(trade_as='avg_price',settle_as='close',show_everyday=False)

    ind_money,ind_date = PBCT_ew.index_backtest(zz1000_sr,show_everyday=False)
    ew_nv = pd.Series(index=date_ew,data=money_ew)
    mvo_nv = pd.Series(index=date_mvo,data=money_mvo)
    te_nv = pd.Series(index=date_te,data=money_te)
    ind_nv = pd.Series(index=ind_date,data=ind_money)
    nv_df = pd.concat([ew_nv,mvo_nv,te_nv,ind_nv],axis=1)
    nv_df.columns = ['Equal_Weighted','MVO_Optimized','TE_Optimized','CSI_1000']
    nv_df.plot()
    plt.title('Backtest Curve',fontsize=18)
    plt.show()
    nv_df.to_csv(os.path.join(PATH,'result','Backtest',rf'{f}\Three_Strategy_NV.csv'),encoding='gbk')


    # ew_nv.to_csv(os.path.join(PATH,'result',f'{f}','EW_Strategy_Performance.csv'))
    # excess_rnt = (ew_nv.pct_change() - ind_nv.pct_change()).dropna()
    # excess_nv = (1+excess_rnt).cumprod()*ind_money[0]
    # excess_nv.plot(label='Excess Return')
    # plt.plot(date_ew, money_ew, label='Strategy Performance')
    # plt.plot(ind_date, ind_money, label='CSI 1000')
    # plt.xlabel('Date')
    # plt.ylabel('Capital')
    # plt.title('Backtest Curve', fontsize=16)
    # plt.legend()
    # # plt.savefig(os.path.join(PATH,'result',f"{f}\Strategy_Performance.png"))
    # plt.show(block=True)
    #



