import os

import numpy as np
import pandas as pd
from typing import (
    List,
    Optional,
    Tuple,
)
from datetime import datetime as dt

from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt

from data_loading import kline
import warnings

warnings.filterwarnings('ignore')


class Factor_Analysis(object):  # change your save_path
    SAVE_PATH = r"D:\大学文档\MAFM\Mathematical Models of Investment\project1\result\neutral"  # save_path
    BENCHMARK_PATH = "D:\大学文档\MAFM\Mathematical Models of Investment\project1\choice_a.xlsx" # benchmark_path
    POOL = 'A Share'
    shift_dict = {'D': 1, 'W': 5, 'M': 21, 'Y': 252}

    def __init__(self, factor_df: pd.DataFrame, factor_name: Optional[str] = None, codes: Optional[List[str]] = None,
                 start: Optional[str] = None,
                 end: Optional[str] = None, freq: str = 'D'):
        self.SAVE_PATH = os.path.join(self.SAVE_PATH, freq)
        self.profit = None
        self.hierarchical_constitute = None  # 后freq日购买的股票
        self.freq = self.shift_dict[freq]
        self.codes = codes
        self.kline = kline(codes, start, end)
        self.trading_day = pd.to_datetime(kline.get_trading_days(start, end))
        self.trading_day = self.trading_day[self.trading_day.isin(factor_df.index)][
                           :-self.freq]  #  最后N天的因子不用，因为回测用下N日数据
        self.dates = self.backtest_dates()
        self.factor_df = self.align(original_factor=factor_df, dates=self.trading_day.tolist(), codes=self.codes)
        self.std_factor_df = self.preprocess()
        self.r_next = self.align(self.get_r_next(), dates=self.trading_day.tolist(),
                                 codes=self.std_factor_df.columns.tolist())  # 只提取有因子数据的
        self.trade_state_next = self.align(self.get_trading_state_next(), dates=self.trading_day.tolist(),
                                           codes=self.std_factor_df.columns.tolist())  # 只提取有因子数据的
        self.benchmark_r_next = pd.read_excel(self.BENCHMARK_PATH,
                                              index_col=2, parse_dates=True)['收盘价'].pct_change(
            periods=self.freq).shift(-self.freq).dropna()
        self.name = factor_name

    @staticmethod
    def three_sigma(cs_factor: pd.Series) -> pd.Series:  # 截面去极值

        mean = np.nanmean(cs_factor)
        std = np.nanstd(cs_factor)
        cs_factor = cs_factor.loc[
            (cs_factor > mean - 3 * std) & (cs_factor < mean + 3 * std)]
        return cs_factor

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
        trade_state = self.kline.get_data('trade_state')
        high = self.kline.get_data('high')
        low = self.kline.get_data('low')
        trade_state.index, low.index, high.index = pd.to_datetime(trade_state.index), pd.to_datetime(
            trade_state.index), pd.to_datetime(trade_state.index)
        flag = high != low
        trade_permission = flag * trade_state
        trade_permission[trade_permission == -1] = 1
        trade_permission[trade_permission != 1] = 0
        trade_state_next = (trade_permission.shift(-self.freq)).dropna(how='all', axis=0)
        return trade_state_next

    def get_r_next(self) -> pd.DataFrame:
        close_adj = self.kline.get_data('close_adj')
        r_adj = close_adj.pct_change(periods=self.freq)
        r_adj.index = pd.to_datetime(r_adj.index)
        return (r_adj.shift(-self.freq)).dropna(how='all', axis=0)

    def standard(self, cs_factor: pd.Series, type: Optional[str] = 'Minmax') -> pd.Series:  # 截面标准化
        cs_factor = self.three_sigma(cs_factor)
        if type == 'Minmax':
            col_min = np.nanmin(cs_factor)
            col_max = np.nanmax(cs_factor)
            cs_factor = (cs_factor - col_min) / (col_max - col_min)
            return cs_factor
        elif type == 'Normal':
            mean = np.nanmean(cs_factor)
            std = np.nanstd(cs_factor)
            cs_factor = (cs_factor - mean) / std
            return cs_factor

    def preprocess(self) -> pd.DataFrame:  # 因子截面标准化+去极值
        standard_factor = self.factor_df.apply(lambda x: self.standard(x), axis=1).dropna(axis=1, how='all')
        return standard_factor

    def backtest_dates(self) -> List[pd.Timestamp]:
        if self.freq == 1:
            date_range = self.trading_day
            profit_dates = date_range.tolist() + [date_range[-1] + relativedelta(days=1)]
        elif self.freq == 5:
            date_range = self.trading_day[::5]
            profit_dates = date_range.tolist() + [date_range[-1] + relativedelta(weeks=1)]
        elif self.freq == 21:
            date_range = self.trading_day[::21]
            profit_dates = date_range.tolist() + [date_range[-1] + relativedelta(months=1)]
        else:
            date_range = self.trading_day[::252]
            profit_dates = date_range.tolist() + [date_range[-1] + relativedelta(years=1)]
        return profit_dates

    def st_filter(self, date: pd.Timestamp) -> pd.Index:
        trade_state = self.trade_state_next.loc[date]
        trading_stock = trade_state[trade_state == 1].index
        return trading_stock

    def Stratify_r(self, date: pd.Timestamp) -> Tuple:  # 返回分层收益率(下一交易日)和分层成分
        print(date)
        trading_stock = self.st_filter(date)
        daily_factor = self.std_factor_df.loc[date]
        daily_next_r = pd.DataFrame(self.r_next.loc[date, trading_stock])
        factor_quantiles = daily_factor.quantile([0, 1 / 5, 2 / 5, 3 / 5, 4 / 5, 1])
        if len(factor_quantiles) != len(factor_quantiles.drop_duplicates()):
            print('Data concentrated！')
            dup_pos = np.where(factor_quantiles.duplicated())[0][0]
            factor_quantiles.iloc[dup_pos] = (factor_quantiles.iloc[dup_pos] + factor_quantiles.iloc[dup_pos + 1]) / 2

        st_layer = pd.cut(daily_factor, bins=factor_quantiles,
                          labels=['1', '2', '3', '4', '5'],
                          include_lowest=True, right=True)
        daily_next_r['level'] = st_layer.loc[daily_next_r.index]
        layer_return = daily_next_r.groupby(by='level').apply(np.nanmean)

        layer_return.loc['1-5'] = layer_return.loc['1'] - layer_return.loc['5']
        layer_return.loc['5-1'] = layer_return.loc['5'] - layer_return.loc['1']

        return layer_return, daily_next_r['level']

    def Period_NV_Backtest(self) -> pd.DataFrame:  # 返回一段期间的给定name的净值变化

        period_factor_NV = pd.DataFrame()
        hierarchical_constitute = pd.DataFrame()
        for date in self.trading_day:
            factor_r, daily_constitute = self.Stratify_r(date)  # 后一日的收益率
            factor_NV = factor_r + 1
            period_factor_NV = pd.concat([period_factor_NV, factor_NV], axis=1)
            hierarchical_constitute = pd.concat([hierarchical_constitute, daily_constitute], axis=1)
        period_factor_NV.columns = self.trading_day
        hierarchical_constitute.columns = self.trading_day
        self.hierarchical_constitute = hierarchical_constitute
        return period_factor_NV  # 返回的格式是单因子分层的时序区间收益率

    def Backtest(self):  # 计算区间内的单因子回测情况
        NV = self.Period_NV_Backtest()
        profit = pd.DataFrame(index=NV.index,
                              data=np.ones([len(NV.index), 1]))
        date_range = self.trading_day[::self.freq]
        for date in date_range:
            select_NV = NV[date]
            profit = pd.concat([profit, profit.iloc[:, -1] * select_NV], axis=1)
        profit.columns = self.dates
        label = '1-5' if profit.iloc[-2, -1] > profit.iloc[-1, -1] else '5-1'
        profit.loc['Long-Short', :] = profit.loc[label, :]
        self.profit = profit
        return profit

    def Rank_IC(self) -> pd.Series:
        Rank_IC_sr = self.std_factor_df.corrwith(self.r_next, method='spearman', axis=1)
        return Rank_IC_sr

    def IC(self) -> pd.Series:
        IC_sr = self.std_factor_df.corrwith(self.r_next, method='pearson', axis=1)
        return IC_sr

    def turnover(self) -> pd.DataFrame:
        date_range = self.trading_day[::self.freq]
        turnover_high_list = [0]
        turnover_low_list = [0]
        for i in range(len(date_range) - 1):
            position_old = self.hierarchical_constitute[date_range[i]]
            position_new = self.hierarchical_constitute[date_range[i + 1]]
            position_old_high = set(position_old[position_old == '5'].index.tolist())
            position_old_low = set(position_old[position_old == '1'].index.tolist())
            position_new_high = set(position_new[position_new == '5'].index.tolist())
            position_new_low = set(position_new[position_new == '1'].index.tolist())
            turnover_high = len(position_new_high.difference(position_old_high)) / len(position_old_high)
            turnover_low = len(position_new_low.difference(position_old_low)) / len(position_old_low)
            turnover_high_list.append(turnover_high)
            turnover_low_list.append(turnover_low)
        turnover_high_sr = pd.Series(data=turnover_high_list, index=self.dates[1:])
        turnover_low_sr = pd.Series(data=turnover_low_list, index=self.dates[1:])
        turnover_df = pd.concat([turnover_high_sr, turnover_low_sr], axis=1)
        turnover_df.columns = ['layer5 turnover', 'layer1 turnover']
        return turnover_df

    def hit_rate(self) -> pd.DataFrame:
        date_range = self.trading_day[::self.freq]
        layer_1_hit_list = []
        layer_5_hit_list = []
        for i in range(len(date_range)):
            daily_cons = self.hierarchical_constitute[date_range[i]]
            daily_return = self.r_next.loc[date_range[i]]
            layer_1_cons = daily_cons[daily_cons == '1']
            layer_5_cons = daily_cons[daily_cons == '5']
            benchmark_return = self.benchmark_r_next.loc[date_range[i]]
            layer_1_return = daily_return.loc[layer_1_cons.index]
            layer_5_return = daily_return.loc[layer_5_cons.index]
            layer_5_hit_rate = (layer_5_return > benchmark_return).value_counts(1).loc[True]
            layer_1_hit_rate = (layer_1_return > benchmark_return).value_counts(1).loc[True]
            layer_1_hit_list.append(layer_1_hit_rate)
            layer_5_hit_list.append(layer_5_hit_rate)
        hit_high_sr = pd.Series(data=layer_5_hit_list, index=self.dates[1:])
        hit_low_sr = pd.Series(data=layer_1_hit_list, index=self.dates[1:])
        hit_df = pd.concat([hit_high_sr, hit_low_sr], axis=1)
        hit_df.columns = ['layer5 hit_rate', 'layer1 hit_rate']
        return hit_df

    def Backtest_Plot(self):
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        Period_profit = self.profit
        Period_profit.loc[['1', '2', '3', '4', '5', 'Long-Short']].T.plot()
        plt.legend()
        plt.title('{} Factor Performance'.format(self.name))
        plt.savefig(self.SAVE_PATH + r'\{}因子分层回测 开始于{} frequency={}.png'.format(self.name,
                                                                                         dt.strftime(
                                                                                             self.trading_day[0],
                                                                                             '%Y%m%d'),
                                                                                         self.freq))
        plt.close()

    def line_plot(self, data: pd.Series, figure_name: str = 'IC'):
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=200)
        ax.plot(data.index, data.values, label='{}'.format(figure_name), lw=1)
        ax.text(.05, .95, "Mean %.3f \n Std. %.3f" % (data.mean(), data.std()),
                fontsize=16,
                bbox={'facecolor': 'white', 'alpha': 1, 'pad': 5},
                transform=ax.transAxes,
                verticalalignment='top')
        ax.set(xlabel="", title='{} {} Factor {} Distribution'.format(self.POOL, self.name, figure_name))
        ax.set_ylabel('{}'.format(figure_name), color='blue')

        ax.tick_params(axis='y', labelcolor='blue')
        cum_data = data.cumsum()
        ax1 = ax.twinx()
        ax1.set_ylabel('Cumulative {}'.format(figure_name), color='red')
        ax1.plot(data.index, cum_data, color='red', ls='-', alpha=0.8, label='Cumulative {}'.format(figure_name))
        ax1.tick_params(axis='y', labelcolor='red')
        fig.legend(loc='upper right')

        plt.savefig(
            self.SAVE_PATH + r'\{}内{}因子{} 开始于{} frequency={}.png'.format(self.POOL, self.name, figure_name,
                                                                               dt.strftime(self.trading_day[0],
                                                                                           '%Y%m%d'), self.freq
                                                                               ))
        plt.close()

    def bar_plot(self, figure_name: str = 'turnover'):
        function_dict = {'turnover': self.turnover(), 'hit_rate': self.hit_rate()}
        try:
            df = function_dict[figure_name]
            df.to_csv(self.SAVE_PATH+rf'\{figure_name}.csv',encoding='gbk')
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].bar(df.index, df[f'layer5 {figure_name}'], width=10)
            ax[0].set_title(f'Layer 5 {figure_name}', fontsize=16)
            ax[1].bar(df.index, df[f'layer1 {figure_name}'], width=10)
            ax[1].set_title(f'Layer 1 {figure_name}', fontsize=16)
            plt.show(block=True)
        except Exception as e:
            print('Encounter error:', e)
            print('Figure name must be one of "turnover" and "hit_rate"')
        # plt.savefig(self.SAVE_PATH+rf'\{self.name}因子换手率分布 开始于{dt.strftime(self.trading_day[0],'%Y%m%d')}.png')

    def save_data(self):
        self.profit.to_csv(self.SAVE_PATH + '\Factor_NV.csv', encoding='gbk')
        self.hierarchical_constitute.to_csv(self.SAVE_PATH + '\Hierarchical_Holding.csv', encoding='gbk')


if __name__ == '__main__':
    acf = pd.read_parquet(r"D:\大学文档\MAFM\Mathematical Models of Investment\project1\acf_industry_neutral.parquet")  # your file path
    acf = pd.pivot_table(index='TRADE_DT', columns='S_INFO_WINDCODE', values='acf_industry_neutral', data=acf)
    acf.index = pd.to_datetime(acf.index)
    backtester = Factor_Analysis(acf, 'Neutral_TCF', start='20200101', freq='M')
    strategy_r = backtester.Backtest()
    backtester.Backtest_Plot()
    IC = backtester.IC()
    Rank_IC = backtester.Rank_IC()
    backtester.line_plot(IC, 'IC')
    backtester.line_plot(Rank_IC, 'Rank_IC')
    # backtester.bar_plot('turnover')
    # backtester.bar_plot('hit_rate')
    backtester.save_data()