Our portfolio backtest project contains three main modules:
1:portfolio construction;2:portfolio backtest;3:data loading
There are several tips for you to use these three modules:
1:Find your database path:ashare_price.db, it will be uploaded in the zip file data, you need to change the attribution in class Kline from file data_loading
2:Kline class is very convenience. To use it, you are required to enter three parameters: code, which can be your target pool, if none, return all A share;
 start_date: when your data starts; end_date: when your data ends. Once initialized a class. you can use get_data methods to obtain EOD datas for your target.
3:portfolio_construction file is aimed to construct a portfolio, its output will be used for the file: portfolio_backtest as the input. To use this file, you
need to change three file paths:1:your original factor path, in this project is acf_factor.parquet;2:index daily constitution, which is 中证1000成分股.csv;3:
SAVE_PATH, this can be arbitrary path, just for your convenience.
4:The core part of file portfolio_construction is Class Multifactors_Select. To use it, you need to initialize several parameters, which is clearly show in the
__init__ method.
    def __init__(self, factor_df: pd.DataFrame, factor_name: Optional[str] = None, codes: Optional[List[str]] = None,
                 start: Optional[str] = None,
                 end: Optional[str] = None, freq: str = 'D',pool:Optional[pd.DataFrame]=None):
factor_df is your factor, date for index, stock codes for columns;factor_name is the name for your factor;codes is your target pool;freq is your strategy frequency,
can be one of 'D','W','M'.
Once you have initialized this class, you can easily use embedded function st_quantile_sel to select your stocks
    def st_quantile_sel(self, num:int=50, flag:int=-1,
                        agg_holding:pd.Series=None,
                        ):
num is the number of stock in your portfolio. flag means the direction of your factor, 1 for positive factor and -1 for negative factor. agg_holding is not so important but
still can be given if you like, this parameter is to control your aggregate position for your strategy, if not given, then it would be full position every day. If you want
to reduce your total position to 0.8 or less, you can try it.
5:portfolio_backtest part can be used for your backtest, which is also very convenience. Before your usage, two paths need to be adjusted:1:PATH:the path of your project;
2:acf因子TE优化.csv: the name of your portfolio file, this should be within the directory 'result';3:K线导出_000852_日线数据.xlsx, index data, this should be in the directory
'data'.
6:Congratulations, now you can backtest your portfolio! Firstly, initialize a Class PortfolioBacktester, just three parameters:start, end, portfolio_weight:
three columns:st_code, trade_date, weight. Once initialized, use function: PBCT.backtest(trade_as='avg_price',settle_as='close',show_everyday=False) to get backtest.
trade_as parameters is the price we buy or sell our stocks, default as open price, settle_as is the price we show our strategy performance, usually we settle our strategy
performance after market close, so we default it as close price. show_everyday decide whether your want to see your portfolio performance every day or just every rebalance
day, if True, then show it every day(however, we still just trading at every rebalance day). What's more, we have added transaction cost to this class, which is fee at the top
of this file, we default it as 0.02%, you can adjust it as well.

These are all for this project, hope you can get familiar with it quickly!

