import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
DATA_PATH = r'../../result/Backtest/M/'


class Evaluation(object):
    def __init__(self, NV:pd.Series):
        self.NV = NV
        pass

    def get_annual_profit(self):
        days = (self.NV.index[-1] - self.NV.index[0]).days
        annual_profit = np.power(self.NV.iloc[-1] / self.NV.iloc[0], 1 / (days / 365)) - 1
        return annual_profit

    def get_max_withdraw(self, ):
        max_withdraw = 0
        max_withdraw_date = None
        peak = self.NV.iloc[0]
        i = 0
        for price in self.NV:
            if price > peak:
                peak = price
            withdraw = (peak - price) / peak
            if withdraw > max_withdraw:
                max_withdraw = withdraw
                max_withdraw_date = self.NV.index[i]
            i += 1

        return max_withdraw, max_withdraw_date

    def get_annual_volatility(self):

        daily_returns = (self.NV - self.NV.shift(1)) / self.NV.shift(1)
        daily_volatility = np.std(daily_returns)
        annual_volatility = daily_volatility * np.sqrt(252)
        return annual_volatility

    def get_sharpe(self):
        Er = self.get_annual_profit() - 0.025
        sigma = self.get_annual_volatility()
        return Er / sigma

    def get_kamma(self):
        Er = self.get_annual_profit()
        withdraw, withdraw_date = self.get_max_withdraw()
        return Er / withdraw

    def generate_info(self):
        r = (self.NV.iloc[-1] - self.NV.iloc[0]) / self.NV.iloc[0]
        annual_r = self.get_annual_profit()
        sigma = self.get_annual_volatility()
        sharpe = self.get_sharpe()
        kamma = self.get_kamma()
        max_withdraw, max_withdraw_date = self.get_max_withdraw()
        return pd.Series(data=[r, annual_r, sigma, max_withdraw, max_withdraw_date, sharpe, kamma],
                         index=['Period Return', 'Annualized Return', 'Annualized Volatility',
                                'Max Drawdown', 'Max Drawdown Date', 'Sharpe', 'Calmar'])


if __name__ == '__main__':

    data_path = os.path.join(DATA_PATH,'Three_Strategy_NV.csv')
    NV_df = pd.read_csv(data_path,index_col=0,parse_dates=True)
    performance = NV_df.apply(lambda x: Evaluation(x).generate_info(), axis=0)
    performance.to_csv(os.path.join(DATA_PATH,'Three_Strategy_Performance.csv'))


    #
