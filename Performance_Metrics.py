import pandas as pd
import numpy as np
from scipy import linalg
import warnings
import scipy.stats as ss
from dateutil.relativedelta import relativedelta
import scipy
import statistics
from MLFIN_FUNC import *

import warnings
warnings.filterwarnings("ignore")


def index_performance(INDEX):
    ###################################################################################################################
    # Print Preliminaries Stat on console
    ##################################################################################################################

    print("/////////////////////////////////////////////////////////////////////////////////////////////////")
    print("Index Cumulated Return", round(INDEX['log_ret'].sum(), 3), "Index Std", round(INDEX['log_ret'].std(), 4))
    print("/////////////////////////////////////////////////////////////////////////////////////////////////")
    print('Overnight_RET', round(INDEX['O Ret'].sum(), 3), 'Overnight_Std', round(INDEX['O Ret'].std(), 4))
    print('Intraday_RET', round(INDEX['I Ret'].sum(), 3), 'Intraday_Std', round(INDEX['I Ret'].std(), 4))
    #print('T_RET', round(INDEX['T Ret'].sum(), 3), 'Total_Std', round(INDEX['T Ret'].std(), 4))
    print("/////////////////////////////////////////////////////////////////////////////////////////////////")
    print("Strategy Intraday Performance at the index level")
    print("/////////////////////////////////////////////////////////////////////////////////////////////////")
    print('Str_Long', round(INDEX['Str Long'].sum(), 3), 'Str_Long_Std', round(INDEX['Str Long'].std(), 4))
    print('Str_Short', round(INDEX['Str Short'].sum(), 3), 'Str_Short_Std', round(INDEX['Str Short'].std(), 4))
    print('Index CR', round(INDEX['log_ret'].sum(), 3), 'Strat CR',
          round(INDEX['O Ret'].sum() + INDEX['Str Long'].sum() + INDEX['Str Short'].sum(), 3))
    print('Index CR/Std', round(INDEX['log_ret'].sum() / INDEX['log_ret'].std(), 0), 'Strat CR/Std', round(
        (INDEX['O Ret'].sum() + INDEX['Str Long'].sum() + INDEX['Str Short'].sum()) / (
                    INDEX['O Ret'].var() + INDEX['Str Long'].var() + INDEX['Str Short'].var()) ** (1 / 2), 0))
    print("/////////////////////////////////////////////////////////////////////////////////////////////////")
    print('Check T_RET', round(INDEX['T Ret'].sum(), 3), " O_RET+I_RET",
          round(INDEX['O Ret'].sum() + INDEX['I Ret'].sum(), 3))
    print("/////////////////////////////////////////////////////////////////////////////////////////////////")

    #############################################################################################################
    # Compute Returns
    ############################################################################################################

    INDEX["date_time"] = INDEX["Date"] +' '+ INDEX["Time"]
    INDEX.index = pd.to_datetime(INDEX['date_time'])
    INDEX = INDEX.drop('date_time', axis=1)

    Cumulated_return = INDEX['log_ret'].sum()
    Cumulated_return_Positive = INDEX['log_ret'].where(INDEX['log_ret'] > 0).sum(0)
    Cumulated_return_Neagative = INDEX['log_ret'].where(INDEX['log_ret'] < 0).sum(0)
    Profitability_Ratio = Cumulated_return_Positive/abs(Cumulated_return_Neagative)

    Cumulated_return_O_Ret=INDEX['O Ret'].sum()
    Cumulated_return_Positive_O_Ret=INDEX['O Ret'].where(INDEX['O Ret'] > 0).sum(0)
    Cumulated_return_Neagative_O_Ret=INDEX['O Ret'].where(INDEX['O Ret'] < 0).sum(0)
    Profitability_Ratio_O_Ret=Cumulated_return_Positive_O_Ret/abs(Cumulated_return_Neagative_O_Ret)

    Cumulated_return_I_Ret=INDEX['I Ret'].sum()
    Cumulated_return_Positive_I_Ret=INDEX['I Ret'].where(INDEX['I Ret'] > 0).sum(0)
    Cumulated_return_Neagative_I_Ret=INDEX['I Ret'].where(INDEX['I Ret'] < 0).sum(0)
    Profitability_Ratio_I_Ret=Cumulated_return_Positive_I_Ret/abs(Cumulated_return_Neagative_I_Ret)

    print("/////////////////////////////////////////////////////////////////////////////////////////////////")
    print('Profitability_Ratio_Index', Profitability_Ratio)
    print('Profitability_Ratio_Overnight', Profitability_Ratio_O_Ret)
    print('Profitability_Ratio_Intraday', Profitability_Ratio_I_Ret)
    print("/////////////////////////////////////////////////////////////////////////////////////////////////")

    INDEX['Date'] = pd.to_datetime(INDEX['Date'])
    Start_Date = INDEX['Date'].iloc[1]
    End_Date = INDEX['Date'].iloc[-1]
    difference_in_years = relativedelta(End_Date, Start_Date).years
    Average_Yearly_Return = Cumulated_return/difference_in_years
    Average_Yearly_Return_I = Cumulated_return_I_Ret/difference_in_years
    Average_Yearly_Return_O = Cumulated_return_O_Ret/difference_in_years

    print("/////////////////////////////////////////////////////////////////////////////////////////////////")
    print('difference_in_years', difference_in_years, 'Average_Yearly_Return', Average_Yearly_Return)
    print('Average_Yearly_Return_I', Average_Yearly_Return_I, 'Average_Yearly_Return_O', Average_Yearly_Return_O)
    print("/////////////////////////////////////////////////////////////////////////////////////////////////")
    INDEX['Cum_ret'] = INDEX.log_ret.cumsum()
    INDEX['AUM']=100*np.exp(INDEX['Cum_ret'])
    Daily_Index=INDEX['T Ret']!=0
    RET=INDEX[Daily_Index]
    logret_series = np.exp(RET['T Ret'])
    logret_series=np.log(logret_series)
    pos_concentr, neg_concentr, hourly_concentr = all_bets_concentration(logret_series, frequency='W')

    print("/////////////////////////////////////////////////////////////////////////////////////////////////")
    print('HHI index on positive log returns is' , pos_concentr)
    print('HHI index on negative log returns is' , neg_concentr)
    print('HHI index on log returns divided into Weekly bins is' , hourly_concentr)
    print("/////////////////////////////////////////////////////////////////////////////////////////////////")
    drawdown, tuw = drawdown_and_time_under_water(RET['AUM'], dollars = False)
    drawdown_dollars, _ = drawdown_and_time_under_water(RET['AUM'], dollars = True)
    print('The 99th percentile of Drawdown is' , drawdown.quantile(.99))
    print('The 99th percentile of Drawdown in dollars is' , drawdown_dollars.quantile(.99))
    print('The 99th percentile of Time under water' , tuw.quantile(.99))
    print("/////////////////////////////////////////////////////////////////////////////////////////////////")

    #Also looking at returns grouped by days
    logret_by_days = logret_series.groupby(pd.Grouper(freq='D')).sum()
    logret_by_days = logret_by_days[logret_by_days!=0]

    print('Average log return from positive bars is' , logret_series[logret_series>0].mean(),
          'and counter is', logret_series[logret_series>0].count())
    print('Average log return from negative bars is' , logret_series[logret_series<0].mean(),
         'and counter is', logret_series[logret_series<0].count())
    print("/////////////////////////////////////////////////////////////////////////////////////////////////")
    #Uning mlfinlab package function to get SR
    annualized_sr = sharpe_ratio(logret_by_days, entries_per_year=252, risk_free_rate=0)
    print('Annualized Sharpe Ratio is' , annualized_sr)
    print("/////////////////////////////////////////////////////////////////////////////////////////////////")

    print("////////////////////////////////////////////////////////////////////////////////")
    print("General Summary")
    print("////////////////////////////////////////////////////////////////////////////////")
    #print("Start", Date_Start, "End", Date_End)
    print("Index Cumulated Return", round(INDEX['log_ret'].sum(),3), "Index Std", round(INDEX['log_ret'].std(),4))
    print('Number of Years', difference_in_years)
    print('Average_Yearly_Return', Average_Yearly_Return)
    print('Profitability_Ratio', Profitability_Ratio)
    print('Annualized Sharpe Ratio is' , annualized_sr)

    print("////////////////////////////////////////////////////////////////////////////////")
    print("Return Decomposition")
    print("////////////////////////////////////////////////////////////////////////////////")
    print('Overnight_RET', round(INDEX['O Ret'].sum(), 3), 'Overnight_Std', round(INDEX['O Ret'].std(), 4))
    print('Intraday_RET', round(INDEX['I Ret'].sum(), 3), 'Intraday_Std', round(INDEX['I Ret'].std(), 4))
    print('Total_RET', round(INDEX['T Ret'].sum(), 3) ,'Total_Std', round(INDEX['T Ret'].std(), 4))
    print('Average log return from positive bars is' , logret_series[logret_series>0].mean(),
          'and counter is', logret_series[logret_series>0].count())
    print('Average log return from negative bars is' , logret_series[logret_series<0].mean(),
         'and counter is', logret_series[logret_series<0].count())

    print("////////////////////////////////////////////////////////////////////////////////")
    print("Concentration")
    print("////////////////////////////////////////////////////////////////////////////////")
    print('HHI index on positive log returns is' , pos_concentr)
    print('HHI index on negative log returns is' , neg_concentr)
    print('HHI index on log returns divided into Weekly bins is' , hourly_concentr)

    print("////////////////////////////////////////////////////////////////////////////////")
    print("Drawdown")
    print("////////////////////////////////////////////////////////////////////////////////")
    print('The 99th percentile of Drawdown is' , drawdown.quantile(.99))
    print('The 99th percentile of Drawdown in dollars is' , drawdown_dollars.quantile(.99))
    print('The 99th percentile of Time under water' , tuw.quantile(.99))

    print("////////////////////////////////////////////////////////////////////////////////")
    print("Strategy Performance at the index level")
    print("////////////////////////////////////////////////////////////////////////////////")
    print('Str_Long', round(INDEX['Str Long'].sum(), 3) , 'Str_Long_Std', round(INDEX['Str Long'].std(), 4))
    print('Str_Short', round(INDEX['Str Short'].sum(), 3) , 'Str_Short_Std', round(INDEX['Str Short'].std(), 4))
    print('Index CR', round(INDEX['log_ret'].sum(), 3), 'Strat CR', round(INDEX['O Ret'].sum()+INDEX['Str Long'].sum()+INDEX['Str Short'].sum(),3) )
    print('Index CR/Std', round(INDEX['log_ret'].sum()/INDEX['log_ret'].std(), 0), 'Strat CR/Std', round((INDEX['O Ret'].sum()+INDEX['Str Long'].sum()+INDEX['Str Short'].sum())/(INDEX['O Ret'].var()+INDEX['Str Long'].var()+INDEX['Str Short'].var())**(1/2),0))

def Perfomance_Analysis_Strategy(STRAT_RETURNS, vix, STRATEGY):
    #############################################################################################
    # First Analysis
    #############################################################################################

    Cumulated_return_strategy = STRAT_RETURNS['Trades'].sum()
    Cumulated_return_Positive_Trades = STRAT_RETURNS['Trades'].where(STRAT_RETURNS['Trades'] > 0).sum(0)
    Cumulated_return_Neagative_Trades = STRAT_RETURNS['Trades'].where(STRAT_RETURNS['Trades'] < 0).sum(0)
    Cumulated_return_High_Vol = STRAT_RETURNS['Trades'].where(vix > 25).sum(0)
    Cumulated_return_Low_Vol = STRAT_RETURNS['Trades'].where(vix < 25).sum(0)

    BULL_TRADES = 0
    BEAR_TRADES = 0
    LATERAL_TRADES = 0

    for i in range(0, len(STRAT_RETURNS['Trades'])):

        if STRAT_RETURNS['Trades'].iloc[i] != 0:

            if (STRAT_RETURNS["POS"].iloc[i] > 0):
                BULL_TRADES = BULL_TRADES + STRAT_RETURNS['Trades'].iloc[i]

            elif (STRAT_RETURNS["POS"].iloc[i] < 0):
                BEAR_TRADES = BEAR_TRADES + STRAT_RETURNS['Trades'].iloc[i]

    Profitability_Ratio = Cumulated_return_Positive_Trades / abs(Cumulated_return_Neagative_Trades)

    Start_Date = STRAT_RETURNS['DATE_F'].iloc[1]
    End_Date = STRAT_RETURNS['DATE_F'].iloc[-1]
    difference_in_years = relativedelta(End_Date, Start_Date).years
    Average_Yearly_Return = Cumulated_return_strategy / difference_in_years

    ####################################################################
    # Create time series of returns of the trades
    ####################################################################

    A = STRAT_RETURNS['Trades'] != 0
    TRADES = STRAT_RETURNS[A]

    #############################################################################################
    # Trade Analysis
    #############################################################################################

    Number_of_Trades = TRADES['Trades'].count()
    Number_of_Negative_Trades = TRADES['Trades'].where(TRADES['Trades'] < 0).count()
    Hit_Ratio = (Number_of_Trades - Number_of_Negative_Trades) / Number_of_Trades
    Frequency_of_bets = Number_of_Trades / difference_in_years
    Average_Trade_Return = TRADES['Trades'].mean()
    Average_Return_from_hits = TRADES['Trades'].where(TRADES['Trades'] > 0).mean()
    Average_Return_from_misses = TRADES['Trades'].where(TRADES['Trades'] < 0).mean()

    #########################################################################################################
    # Return Distribution
    ########################################################################################################

    Trades_Variance = statistics.variance(TRADES['Trades'])
    Trades_Standard_Deviation = statistics.stdev(TRADES['Trades'])
    Trades_Skwness = scipy.stats.skew(TRADES['Trades'])
    Trades_Kurtosis = scipy.stats.kurtosis(TRADES['Trades'])

    Cumulated_Return_Long = TRADES['Long_Trades'].sum()
    Number_of_Trades_Long = TRADES['Long_Trades'].where(TRADES['Long_Trades'] != 0).count()
    Ratio_of_longs = Number_of_Trades_Long / Number_of_Trades
    Number_of_Positive_Trades_Long = TRADES['Long_Trades'].where(TRADES['Long_Trades'] > 0).count()
    Hit_Ratio_Long = Number_of_Positive_Trades_Long / Number_of_Trades_Long
    Average_Trade_Return_Long = TRADES['Long_Trades'].mean()
    Average_Return_from_hits_Long = TRADES['Long_Trades'].where(TRADES['Long_Trades'] > 0).mean()
    Average_Return_from_misses_Long = TRADES['Long_Trades'].where(TRADES['Long_Trades'] < 0).mean()
    Trades_Variance_Long = statistics.variance(TRADES['Long_Trades'])
    Trades_Standard_Deviation_Long = statistics.stdev(TRADES['Long_Trades'])
    Trades_Skwness_Long = scipy.stats.skew(TRADES['Long_Trades'])
    Trades_Kurtosis_Long = scipy.stats.kurtosis(TRADES['Long_Trades'])
    Average_Yearly_Return_Long = Cumulated_Return_Long / difference_in_years

    Cumulated_Return_Short = TRADES['Short_Trades'].sum()
    Number_of_Trades_Short = TRADES['Short_Trades'].where(TRADES['Short_Trades'] != 0).count()
    Number_of_Positive_Trades_Short = TRADES['Short_Trades'].where(TRADES['Short_Trades'] > 0).count()
    Hit_Ratio_Short = Number_of_Positive_Trades_Short / Number_of_Trades_Short
    Average_Trade_Return_Short = TRADES['Short_Trades'].mean()
    Average_Return_from_hits_Short = TRADES['Short_Trades'].where(TRADES['Short_Trades'] > 0).mean()
    Average_Return_from_misses_Short = TRADES['Short_Trades'].where(TRADES['Short_Trades'] < 0).mean()
    Trades_Variance_Short = statistics.variance(TRADES['Short_Trades'])
    Trades_Standard_Deviation_Short = statistics.stdev(TRADES['Short_Trades'])
    Trades_Skwness_Short = scipy.stats.skew(TRADES['Short_Trades'])
    Trades_Kurtosis_Short = scipy.stats.kurtosis(TRADES['Short_Trades'])
    Average_Yearly_Return_Short = Cumulated_Return_Short / difference_in_years

    CAPITAL = 100
    AUM = 100
    LEVERAGE = AUM / CAPITAL
    MAXIMUM_DOLLAR_POSITION_SIZE = AUM  # Fixed baet in this strategy
    Average_AUM = AUM * (STRAT_RETURNS['POS'].where(STRAT_RETURNS['POS'] != 0).count() / len(STRAT_RETURNS['POS']))

    Lev_Average_Yearly_Return = (Cumulated_return_strategy / difference_in_years) * LEVERAGE
    Lev_Average_Trade_Return = TRADES['Trades'].mean() * LEVERAGE
    Lev_Average_Return_from_hits = TRADES['Trades'].where(TRADES['Trades'] > 0).mean() * LEVERAGE
    Lev_Average_Return_from_misses = TRADES['Trades'].where(TRADES['Trades'] < 0).mean() * LEVERAGE
    Lev_Trades_Variance = statistics.variance(TRADES['Trades']) * LEVERAGE
    Lev_Trades_Standard_Deviation = statistics.stdev(TRADES['Trades']) * LEVERAGE ** (1 / 2)

    print(
        '////////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('PERFORMANCE SUMMARY')
    print(
        '////////////////////////////////////////////////////////////////////////////////////////////////////////////')

    print('Cumulated return strategy', round(Cumulated_return_strategy, 2))
    print('Cumulated return Winning Trades', round(Cumulated_return_Positive_Trades, 2))
    print('Cumulated_return_Negative_Trades', round(Cumulated_return_Neagative_Trades, 2))
    print('Profitability Ratio', round(Profitability_Ratio, 2))
    print("Cumulated_return_High_Vol", round(Cumulated_return_High_Vol, 2), "Cumulated_return_Low_Vol",
          round(Cumulated_return_Low_Vol, 2))
    print("TOTAL", round(STRAT_RETURNS['Trades'].sum(), 2), "BULL_TRADES", round(BULL_TRADES, 2), "BEAR_TRADES",
          round(BEAR_TRADES, 2), "LATERAL_TRADES", round(STRAT_RETURNS['Trades'].sum() - BULL_TRADES - BEAR_TRADES, 2))
    print('Start_Date', Start_Date, 'End_Date', End_Date, 'Number of Years :', round(difference_in_years, 2))
    print('Average Yearly Return', round(Average_Yearly_Return, 2))
    print('Number of Trades', round(Number_of_Trades, 2))
    print('Number of Negative_Trades', round(Number_of_Negative_Trades, 2))
    print('Hit Ratio', round(Hit_Ratio, 2))
    print('Frequency of_bets', round(Frequency_of_bets, 2))
    print('Average Trade_Return', round(Average_Trade_Return, 4))
    print('Average Return from hits', round(Average_Return_from_hits, 4))
    print('Average Return from misses', round(Average_Return_from_misses, 4))
    print('Trades Variance', round(Trades_Variance, 4))
    print('Trades Standard Deviation', round(Trades_Standard_Deviation, 4))
    print('Trades Skwness', round(Trades_Skwness, 2))
    print('Trades Kurtosis', round(Trades_Kurtosis, 2))

    print(
        '////////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('LONG TRADES SUMMARY')
    print(
        '////////////////////////////////////////////////////////////////////////////////////////////////////////////')

    print('Cumulated Return Long', round(Cumulated_Return_Long, 2))
    print('Number of Trades Long', round(Number_of_Trades_Long, 2))
    print('LONGS/TOTAL', round(Ratio_of_longs, 2))
    print('Number of Winning Trades Long', round(Number_of_Positive_Trades_Long, 2))
    print('Hit Ratio Long', round(Hit_Ratio_Long, 2))
    print('Average Trade Return Long', round(Average_Trade_Return_Long, 4))
    print('Average Return from hits Long', round(Average_Return_from_hits_Long, 4))
    print('Average Return from misses Long', round(Average_Return_from_misses_Long, 4))
    print('Trades Variance Long', round(Trades_Variance_Long, 4))
    print('Trades Standard Deviation Long', round(Trades_Standard_Deviation_Long, 4))
    print('Trades Skwness Long', round(Trades_Skwness_Long, 2))
    print('Trades Kurtosis Long', round(Trades_Kurtosis_Long, 2))
    print('Average Yearly Return Long', round(Average_Yearly_Return_Long, 2))

    print(
        '////////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('SHORT TRADES SUMMARY')
    print(
        '////////////////////////////////////////////////////////////////////////////////////////////////////////////')

    print('Cumulated Return Short', round(Cumulated_Return_Short, 2))
    print('Number of Trades Short', round(Number_of_Trades_Short, 2))
    print('Number of Winning Trades Short', round(Number_of_Positive_Trades_Short, 2))
    print('Hit Ratio Short', round(Hit_Ratio_Short, 2))
    print('Average Trade Return Short', round(Average_Trade_Return_Short, 4))
    print('Average Return from hits Short', round(Average_Return_from_hits_Short, 4))
    print('Average Return from misses Short', round(Average_Return_from_misses_Short, 4))
    print('Trades Variance Short', round(Trades_Variance_Short, 4))
    print('Trades Standard Deviation Short', round(Trades_Standard_Deviation_Short, 4))
    print('Trades Skwness Short', round(Trades_Skwness_Short, 2))
    print('Trades Kurtosis Short', round(Trades_Kurtosis_Short, 2))
    print('Average Yearly Return Short', round(Average_Yearly_Return_Short, 2))


    return 1

def Perfomance_Analysis_Strategy_Advanced(STRATEGY, STRAT_RETURNS, trading_days, risk_free_ratio, objective_Sharpe, n_startegies_testes, correaltion_strategies):
    print(
        '////////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('Advanced Performance Analysis')
    print(
        '////////////////////////////////////////////////////////////////////////////////////////////////////////////')
    # Using mlfinlab package function for detailed concentration output
    STRATEGY["date_time"] = STRATEGY["Date_x"] + ' ' + STRATEGY["Time_x"]
    STRATEGY.index = pd.to_datetime(STRATEGY['date_time'])
    STRAT_RETURNS.index = pd.to_datetime(STRATEGY['date_time'])
    STRATEGY = STRATEGY.drop('date_time', axis=1)

    A = STRAT_RETURNS['Trades'] != 0
    TRADES2 = STRAT_RETURNS[A]
    logret_series= np.exp(TRADES2['Trades'])
    logret_series= np.log(logret_series)

    logret_series_benchmark = np.exp(STRATEGY['log_ret_x'])
    logret_series_benchmark = np.log(logret_series_benchmark)

    pos_concentr, neg_concentr, hourly_concentr = all_bets_concentration(logret_series, frequency='W')
    print('Concentration')
    print('HHI index on positive log returns is', round(pos_concentr, 4))
    print('HHI index on negative log returns is', round(neg_concentr, 4))
    print('HHI index on log returns divided into hourly bins is', round(hourly_concentr, 4))

    # Getting series of prices to represent the value of one long portfolio
    TRADES2['perc_ret'] = TRADES2.Trades.cumsum()
    # TRADES2['perc_ret'].plot()
    TRADES2['AUM'] = 100 * np.exp(TRADES2['perc_ret'])
    # TRADES2['AUM'].plot()
    # plt.show()
    # Using mlfinlab package function to get drawdowns and time under water series
    drawdown, tuw = drawdown_and_time_under_water(TRADES2['AUM'], dollars=False)
    drawdown_dollars, _ = drawdown_and_time_under_water(TRADES2['AUM'], dollars=True)
    print(
        '////////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('Drawdown')
    print(
        '////////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('The 99th percentile of Drawdown is', round(drawdown.quantile(.99), 2))
    print('The 99th percentile of Drawdown in dollars is', round(drawdown_dollars.quantile(.99), 2))
    print('The 99th percentile of Time under water', round(tuw.quantile(.99), 2))

    # Using simple formula for annual return calculation
    days_observed = (TRADES2['AUM'].index[-1] - TRADES2['AUM'].index[0]) / np.timedelta64(1, 'D')
    cumulated_return = TRADES2['AUM'][-1] / TRADES2['AUM'][0]

    # Using 365 days instead of 252 as days observed are calculated as calendar
    # days between the first observation and the last
    annual_return = (cumulated_return) ** (trading_days / days_observed) - 1
    print(
        '////////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('Annualized average return from the portfolio is', round(annual_return, 2))
    print(
        '////////////////////////////////////////////////////////////////////////////////////////////////////////////')

    # Also looking at returns grouped by days
    logret_by_days = logret_series.groupby(pd.Grouper(freq='D')).sum()
    logret_by_days = logret_by_days[logret_by_days != 0]

    logret_by_days_benchmark = logret_series_benchmark.groupby(pd.Grouper(freq='D')).sum()
    logret_by_days_benchmark = logret_by_days_benchmark[logret_by_days_benchmark != 0]

    print('Average log return from positive bars grouped by day is',
          round(logret_by_days[logret_by_days > 0].mean(), 4), 'and counter is',
          round(logret_by_days[logret_by_days > 0].count(), 4))

    print('Average log return from negative bars grouped by day is', round(logret_series[logret_series < 0].mean(), 4),
          'and counter is', round(logret_series[logret_series < 0].count(), 4))

    #############################################################################
    # Uning mlfinlab package function to get SR
    #############################################################################

    # Stating the risk-free ratio and trading days per year
    annualized_sr = sharpe_ratio(logret_by_days, entries_per_year = trading_days, risk_free_rate = risk_free_ratio)
    print(
        '////////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('Sharpe')
    print(
        '////////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('Strategy Annualized Sharpe Ratio is', round(annualized_sr, 2))
    annualized_sr_benchmark = sharpe_ratio(logret_by_days_benchmark, entries_per_year = trading_days, risk_free_rate = risk_free_ratio)
    print('Benchmark Annualized Sharpe Ratio is', round(annualized_sr_benchmark, 2))

    # Calculating excess returns above the risk-free ratio
    # This means subtracting the benchmark from daily returns

    # Daily returns adjusted for taking compounding effect into account
    daily_risk_free_ratio = (1 + risk_free_ratio) ** (1 / trading_days) - 1
    log_daily_risk_free_ratio = np.log(1 + daily_risk_free_ratio)

    excess_returns = logret_by_days - logret_by_days_benchmark
    information_r = sharpe_ratio(excess_returns, trading_days)
    print('Information ratio (with yearly risk-free rate assumed to be 0%) is', round(information_r, 2))


    # Using mlfinlab package function to get PSR
    probabilistic_sr = probabilistic_sharpe_ratio(observed_sr=annualized_sr,
                                                  benchmark_sr=objective_Sharpe,
                                                  number_of_returns=days_observed,
                                                  skewness_of_returns=logret_by_days.skew(),
                                                  kurtosis_of_returns=logret_by_days.kurt())
    print('Probabilistic Sharpe Ratio (a value above 0.95 is good) is', round(probabilistic_sr, 2))

    benchmark_sr_dsr = deflated_sharpe_ratio(observed_sr=annualized_sr,
                                             sr_estimates=[0.005 ** (1 / 2), 3],
                                             number_of_returns=days_observed,
                                             skewness_of_returns=logret_by_days.skew(),
                                             kurtosis_of_returns=logret_by_days.kurt(),
                                             estimates_param=True, benchmark_out=True)

    print('Benchmark Sharpe ratio used in DSR is', round(benchmark_sr_dsr, 2))
    # Using mlfinlab package function to get DSR. Passing standard deviation of trails and
    # number of trails as a parameter, also flag estimates_param.

    deflated_sr = deflated_sharpe_ratio(observed_sr=annualized_sr,
                                        sr_estimates=[0.05 ** (1 / 2), 5],
                                        number_of_returns=days_observed,
                                        skewness_of_returns=logret_by_days.skew(),
                                        kurtosis_of_returns=logret_by_days.kurt(),
                                        estimates_param=True)
    print('Deflated Sharpe Ratio (a value above 0.95 is good) ', round(deflated_sr, 2))

    AUTOCORRELATION = estimated_autocorrelation(logret_by_days)
    print('Daily AUTOCORRELATION of the Strategy Returns', AUTOCORRELATION[0:5])

    # Creating an object and specifying the desired level of simulations to do
    # for Haircut Sharpe Ratios and Profit Hurdle in the Holm and BHY methods.
    backtesting = CampbellBacktesting(simulations = 5000)

    # Calculating the adjusted Sharpe ratios and the haircuts.
    haircuts = backtesting.haircut_sharpe_ratios(sampling_frequency='D', num_obs=days_observed,
                                                 sharpe_ratio=annualized_sr,
                                                 annualized=True, autocorr_adjusted=False, rho_a=AUTOCORRELATION[1],
                                                 num_mult_test=n_startegies_testes, rho=correaltion_strategies)

    # Adjusted Sharpe ratios by the method used.
    print(
        '////////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('The adjusted Sharpe ratio ')
    print(
        '////////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('The adjusted Sharpe ratio using the Bonferroni method is', round(haircuts[1][0], 4))
    print('The adjusted Sharpe ratio using the Holm method is', round(haircuts[1][1], 4))
    print('The adjusted Sharpe ratio using the BHY method is', round(haircuts[1][2], 4))
    print('The average adjusted Sharpe ratio of the methods is', round(haircuts[1][3], 4))
    # Sharpe ratio haircuts.
    print(
        '////////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('The Sharpe ratio haircut ')
    print(
        '////////////////////////////////////////////////////////////////////////////////////////////////////////////')
    print('The Sharpe ratio haircut using the Bonferroni method is', round(haircuts[2][0], 2))
    print('The Sharpe ratio haircut using the Holm method is', round(haircuts[2][1], 2))
    print('The Sharpe ratio haircut using the BHY method is', round(haircuts[2][2], 2))
    print('The average Sharpe ratio haircut of the methods is', round(haircuts[2][3], 2))

    # Calculating the Minimum Average Monthly Returns.
    #monthly_ret = backtesting.profit_hurdle(num_mult_test=n_startegies_testes, num_obs=days_observed, alpha_sig=0.01,
    #                                        vol_anu=0.05, rho=correaltion_strategies)

    # Minimum Average Monthly Returns by the method used.
    #print(
    #    '////////////////////////////////////////////////////////////////////////////////////////////////////////////')
    #print("Required Minimum Average Monthly Returns")
    #print(
    #    '////////////////////////////////////////////////////////////////////////////////////////////////////////////')
    #print('Required Minimum Average Monthly Returns using the Bonferroni method is', monthly_ret[0])
    #print('Required Minimum Average Monthly Returns using the Holm method is', monthly_ret[1])
    #print('Required Minimum Average Monthly Returns using the BHY method is', monthly_ret[2])
    #print('Required Minimum Average Monthly Returns using the average of the methods is', monthly_ret[3])

    return 1
