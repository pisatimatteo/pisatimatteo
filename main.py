####################################################################################################
# Inport Libraries
####################################################################################################

from datetime import datetime, date, timedelta
import time
import pandas as pd
import numpy as np
from threading import Timer
import csv
import requests
import re
import math
import matplotlib.pyplot as plt
import openpyxl
import statistics
from scipy.stats import skew,kurtosis
from dateutil.relativedelta import relativedelta
import scipy.stats as ss
import matplotlib.pyplot as plt

#############################################################################################################
# Functions Needed for the Index analysis
#############################################################################################################

def Preparation(index):
    index['Close'] = index['Close'].astype(str)
    index['Close'] = [x.replace(',', '.') for x in index['Close']]
    index['Close'] = index['Close'].astype(float)

    index['Open'] = index['Open'].astype(str)
    index['Open'] = [x.replace(',', '.') for x in index['Open']]
    index['Open'] = index['Open'].astype(float)

    index['High'] = index['High'].astype(str)
    index['High'] = [x.replace(',', '.') for x in index['High']]
    index['High'] = index['High'].astype(float)

    index['Low'] = index['Low'].astype(str)
    index['Low'] = [x.replace(',', '.') for x in index['Low']]
    index['Low'] = index['Low'].astype(float)

    index['Close'] = pd.to_numeric(index['Close'], errors='coerce')
    index['Open'] = pd.to_numeric(index['Open'], errors='coerce')
    index['Low'] = pd.to_numeric(index['Low'], errors='coerce')
    index['High'] = pd.to_numeric(index['High'], errors='coerce')

    index['log_ret'] = np.log(index.Close) - np.log(index.Close.shift(1))

    return index

# Function to compute Daily Tot, overnight, and intraday returns
def overnight_ret(index, Over_tresh):
    index['O-E Prices'] = 0
    index['O Ret'] = 0
    index['I Ret'] = 0
    index['T Ret'] = 0
    index['Str Long'] = 0
    index['Str Short'] = 0

    p_open = index.iloc[0, 4]  # column of open bars
    index.iloc[0, 9] = p_open  # column of open days

    p_close = index.iloc[-1, 7]  # column of close
    index.iloc[-1, 9] = p_close  # column of close days
    overn_ret=0

    for i in range(1, index['Date'].size - 1):

        if index.iloc[i, 0] != index.iloc[i - 1, 0]:  # change in day,new day arrived

            p_open = index.iloc[i, 4]  # column of open bars
            index.iloc[i, 9] = p_open  # column of open days
            index.iloc[i, 10] = np.log(p_open) - np.log(p_close)  # overnight ret
            overn_ret=index.iloc[i, 10]

        elif index.iloc[i, 0] != index.iloc[i + 1, 0]:  # change in day, old day finished

            p_close = index.iloc[i, 7]  # column of close
            index.iloc[i, 9] = p_close  # column of close days
            index.iloc[i, 11] = np.log(p_close) - np.log(p_open)  # intra ret
            index.iloc[i, 12] = overn_ret+index.iloc[i, 11]   # tot ret

            if overn_ret>Over_tresh:
                index.iloc[i, 13]=index.iloc[i, 11]
            else:
                index.iloc[i, 14]=index.iloc[i, 11]*-1

"""
Implements statistics related to:
- flattening and flips
- average period of position holding
- concentration of bets
- drawdowns
- various Sharpe ratios
- minimum track record length
"""
import warnings
import pandas as pd
import scipy.stats as ss
import numpy as np


def timing_of_flattening_and_flips(target_positions: pd.Series) -> pd.DatetimeIndex:
    """
    Advances in Financial Machine Learning, Snippet 14.1, page 197

    Derives the timestamps of flattening or flipping trades from a pandas series
    of target positions. Can be used for position changes analysis, such as
    frequency and balance of position changes.

    Flattenings - times when open position is bing closed (final target position is 0).
    Flips - times when positive position is reversed to negative and vice versa.

    :param target_positions: (pd.Series) Target position series with timestamps as indices
    :return: (pd.DatetimeIndex) Timestamps of trades flattening, flipping and last bet
    """

    empty_positions = target_positions[(target_positions == 0)].index  # Empty positions index
    previous_positions = target_positions.shift(1)  # Timestamps pointing at previous positions

    # Index of positions where previous one wasn't empty
    previous_positions = previous_positions[(previous_positions != 0)].index

    # FLATTENING - if previous position was open, but current is empty
    flattening = empty_positions.intersection(previous_positions)

    # Multiplies current position with value of next one
    multiplied_posions = target_positions.iloc[1:] * target_positions.iloc[:-1].values

    # FLIPS - if current position has another direction compared to the next
    flips = multiplied_posions[(multiplied_posions < 0)].index
    flips_and_flattenings = flattening.union(flips).sort_values()
    if target_positions.index[-1] not in flips_and_flattenings:  # Appending with last bet
        flips_and_flattenings = flips_and_flattenings.append(target_positions.index[-1:])

    return flips_and_flattenings


def average_holding_period(target_positions: pd.Series) -> float:
    """
    Advances in Financial Machine Learning, Snippet 14.2, page 197

    Estimates the average holding period (in days) of a strategy, given a pandas series
    of target positions using average entry time pairing algorithm.

    Idea of an algorithm:

    * entry_time = (previous_time * weight_of_previous_position + time_since_beginning_of_trade * increase_in_position )
      / weight_of_current_position
    * holding_period ['holding_time' = time a position was held, 'weight' = weight of position closed]
    * res = weighted average time a trade was held

    :param target_positions: (pd.Series) Target position series with timestamps as indices
    :return: (float) Estimated average holding period, NaN if zero or unpredicted
    """

    holding_period = pd.DataFrame(columns=['holding_time', 'weight'])
    entry_time = 0
    position_difference = target_positions.diff()

    # Time elapsed from the starting time for each position
    time_difference = (target_positions.index - target_positions.index[0]) / np.timedelta64(1, 'D')
    for i in range(1, target_positions.size):

        # Increased or unchanged position
        if float(position_difference.iloc[i] * target_positions.iloc[i - 1]) >= 0:
            if float(target_positions.iloc[i]) != 0:  # And not an empty position
                entry_time = (entry_time * target_positions.iloc[i - 1] +
                              time_difference[i] * position_difference.iloc[i]) / target_positions.iloc[i]

        # Decreased
        if float(position_difference.iloc[i] * target_positions.iloc[i - 1]) < 0:
            hold_time = time_difference[i] - entry_time

            # Flip of a position
            if float(target_positions.iloc[i] * target_positions.iloc[i - 1]) < 0:
                weight = abs(target_positions.iloc[i - 1])
                holding_period.loc[target_positions.index[i], ['holding_time', 'weight']] = (hold_time, weight)
                entry_time = time_difference[i]  # Reset entry time

            # Only a part of position is closed
            else:
                weight = abs(position_difference.iloc[i])
                holding_period.loc[target_positions.index[i], ['holding_time', 'weight']] = (hold_time, weight)

    if float(holding_period['weight'].sum()) > 0:  # If there were closed trades at all
        avg_holding_period = float((holding_period['holding_time'] * \
                                    holding_period['weight']).sum() / holding_period['weight'].sum())
    else:
        avg_holding_period = float('nan')

    return avg_holding_period


def bets_concentration(returns: pd.Series) -> float:
    """
    Advances in Financial Machine Learning, Snippet 14.3, page 201

    Derives the concentration of returns from given pd.Series of returns.

    Algorithm is based on Herfindahl-Hirschman Index where return weights
    are taken as an input.

    :param returns: (pd.Series) Returns from bets
    :return: (float) Concentration of returns (nan if less than 3 returns)
    """

    if returns.size <= 2:
        return float('nan')  # If less than 3 bets
    weights = returns / returns.sum()  # Weights of each bet
    hhi = (weights ** 2).sum()  # Herfindahl-Hirschman Index for weights
    hhi = float((hhi - returns.size ** (-1)) / (1 - returns.size ** (-1)))

    return hhi


def all_bets_concentration(returns: pd.Series, frequency: str = 'M') -> tuple:
    """
    Advances in Financial Machine Learning, Snippet 14.3, page 201

    Given a pd.Series of returns, derives concentration of positive returns, negative returns
    and concentration of bets grouped by time intervals (daily, monthly etc.).
    If after time grouping less than 3 observations, returns nan.

    Properties or results:

    * low positive_concentration ⇒ no right fat-tail of returns (desirable)
    * low negative_concentration ⇒ no left fat-tail of returns (desirable)
    * low time_concentration ⇒ bets are not concentrated in time, or are evenly concentrated (desirable)
    * positive_concentration == 0 ⇔ returns are uniform
    * positive_concentration == 1 ⇔ only one non-zero return exists

    :param returns: (pd.Series) Returns from bets
    :param frequency: (str) Desired time grouping frequency from pd.Grouper
    :return: (tuple of floats) Concentration of positive, negative and time grouped concentrations
    """

    # Concentration of positive returns per bet
    positive_concentration = bets_concentration(returns[returns >= 0])

    # Concentration of negative returns per bet
    negative_concentration = bets_concentration(returns[returns < 0])

    # Concentration of bets/time period (month by default)
    time_concentration = bets_concentration(returns.groupby(pd.Grouper(freq=frequency)).count())

    return (positive_concentration, negative_concentration, time_concentration)


def drawdown_and_time_under_water(returns: pd.Series, dollars: bool = False) -> tuple:
    """
    Advances in Financial Machine Learning, Snippet 14.4, page 201

    Calculates drawdowns and time under water for pd.Series of either relative price of a
    portfolio or dollar price of a portfolio.

    Intuitively, a drawdown is the maximum loss suffered by an investment between two consecutive high-watermarks.
    The time under water is the time elapsed between an high watermark and the moment the PnL (profit and loss)
    exceeds the previous maximum PnL. We also append the Time under water series with period from the last
    high-watermark to the last return observed.

    Return details:

    * Drawdown series index is the time of a high watermark and the value of a
      drawdown after it.
    * Time under water index is the time of a high watermark and how much time
      passed till the next high watermark in years. Also includes time between
      the last high watermark and last observation in returns as the last element.

    :param returns: (pd.Series) Returns from bets
    :param dollars: (bool) Flag if given dollar performance and not returns.
                    If dollars, then drawdowns are in dollars, else as a %.
    :return: (tuple of pd.Series) Series of drawdowns and time under water
    """

    frame = returns.to_frame('pnl')
    frame['hwm'] = returns.expanding().max()  # Adding high watermarks as column

    # Grouped as min returns by high watermarks
    high_watermarks = frame.groupby('hwm').min().reset_index()
    high_watermarks.columns = ['hwm', 'min']

    # Time high watermark occurred
    high_watermarks.index = frame['hwm'].drop_duplicates(keep='first').index

    # Picking ones that had a drawdown after high watermark
    high_watermarks = high_watermarks[high_watermarks['hwm'] > high_watermarks['min']]
    if dollars:
        drawdown = high_watermarks['hwm'] - high_watermarks['min']
    else:
        drawdown = 1 - high_watermarks['min'] / high_watermarks['hwm']

    time_under_water = ((high_watermarks.index[1:] - high_watermarks.index[:-1]) / np.timedelta64(1, 'Y')).values

    # Adding also period from last High watermark to last return observed.
    time_under_water = np.append(time_under_water,
                                 (returns.index[-1] - high_watermarks.index[-1]) / np.timedelta64(1, 'Y'))

    time_under_water = pd.Series(time_under_water, index=high_watermarks.index)

    return drawdown, time_under_water


def sharpe_ratio(returns: pd.Series, entries_per_year: int = 252, risk_free_rate: float = 0) -> float:
    """
    Calculates annualized Sharpe ratio for pd.Series of normal or log returns.

    Risk_free_rate should be given for the same period the returns are given.
    For example, if the input returns are observed in 3 months, the risk-free
    rate given should be the 3-month risk-free rate.

    :param returns: (pd.Series) Returns - normal or log
    :param entries_per_year: (int) Times returns are recorded per year (252 by default)
    :param risk_free_rate: (float) Risk-free rate (0 by default)
    :return: (float) Annualized Sharpe ratio
    """

    sharpe_r = (returns.mean() - risk_free_rate) / returns.std() * (entries_per_year) ** (1 / 2)

    return sharpe_r


def information_ratio(returns: pd.Series, benchmark: float = 0, entries_per_year: int = 252) -> float:
    """
    Calculates annualized information ratio for pd.Series of normal or log returns.

    Benchmark should be provided as a return for the same time period as that between
    input returns. For example, for the daily observations it should be the
    benchmark of daily returns.

    It is the annualized ratio between the average excess return and the tracking error.
    The excess return is measured as the portfolio’s return in excess of the benchmark’s
    return. The tracking error is estimated as the standard deviation of the excess returns.

    :param returns: (pd.Series) Returns - normal or log
    :param benchmark: (float) Benchmark for performance comparison (0 by default)
    :param entries_per_year: (int) Times returns are recorded per year (252 by default)
    :return: (float) Annualized information ratio
    """

    excess_returns = returns - benchmark
    information_r = sharpe_ratio(excess_returns, entries_per_year)

    return information_r


def probabilistic_sharpe_ratio(observed_sr: float, benchmark_sr: float, number_of_returns: int,
                               skewness_of_returns: float = 0, kurtosis_of_returns: float = 3) -> float:
    """
    Calculates the probabilistic Sharpe ratio (PSR) that provides an adjusted estimate of SR,
    by removing the inflationary effect caused by short series with skewed and/or
    fat-tailed returns.

    Given a user-defined benchmark Sharpe ratio and an observed Sharpe ratio,
    PSR estimates the probability that SR ̂is greater than a hypothetical SR.
    - It should exceed 0.95, for the standard significance level of 5%.
    - It can be computed on absolute or relative returns.

    :param observed_sr: (float) Sharpe ratio that is observed
    :param benchmark_sr: (float) Sharpe ratio to which observed_SR is tested against
    :param number_of_returns: (int) Times returns are recorded for observed_SR
    :param skewness_of_returns: (float) Skewness of returns (0 by default)
    :param kurtosis_of_returns: (float) Kurtosis of returns (3 by default)
    :return: (float) Probabilistic Sharpe ratio
    """

    test_value = ((observed_sr - benchmark_sr) * np.sqrt(number_of_returns - 1)) / \
                  ((1 - skewness_of_returns * observed_sr + (kurtosis_of_returns - 1) / \
                    4 * observed_sr ** 2)**(1 / 2))

    if np.isnan(test_value):
        warnings.warn('Test value is nan. Please check the input values.', UserWarning)
        return test_value

    if isinstance(test_value, complex):
        warnings.warn('Output is a complex number. You may want to check the input skewness (too high), '
                      'kurtosis (too low), or observed_sr values.', UserWarning)

    if np.isinf(test_value):
        warnings.warn('Test value is infinite. You may want to check the input skewness, '
                      'kurtosis, or observed_sr values.', UserWarning)

    probab_sr = ss.norm.cdf(test_value)

    return probab_sr


def deflated_sharpe_ratio(observed_sr: float, sr_estimates: list, number_of_returns: int,
                          skewness_of_returns: float = 0, kurtosis_of_returns: float = 3,
                          estimates_param: bool = False, benchmark_out: bool = False) -> float:
    """
    Calculates the deflated Sharpe ratio (DSR) - a PSR where the rejection threshold is
    adjusted to reflect the multiplicity of trials. DSR is estimated as PSR[SR∗], where
    the benchmark Sharpe ratio, SR∗, is no longer user-defined, but calculated from
    SR estimate trails.

    DSR corrects SR for inflationary effects caused by non-Normal returns, track record
    length, and multiple testing/selection bias.
    - It should exceed 0.95, for the standard significance level of 5%.
    - It can be computed on absolute or relative returns.

    Function allows the calculated SR benchmark output and usage of only
    standard deviation and number of SR trails instead of full list of trails.

    :param observed_sr: (float) Sharpe ratio that is being tested
    :param sr_estimates: (list) Sharpe ratios estimates trials list or
        properties list: [Standard deviation of estimates, Number of estimates]
        if estimates_param flag is set to True.
    :param  number_of_returns: (int) Times returns are recorded for observed_SR
    :param skewness_of_returns: (float) Skewness of returns (0 by default)
    :param kurtosis_of_returns: (float) Kurtosis of returns (3 by default)
    :param estimates_param: (bool) Flag to use properties of estimates instead of full list
    :param benchmark_out: (bool) Flag to output the calculated benchmark instead of DSR
    :return: (float) Deflated Sharpe ratio or Benchmark SR (if benchmark_out)
    """

    # Calculating benchmark_SR from the parameters of estimates
    if estimates_param:
        benchmark_sr = sr_estimates[0] * \
                       ((1 - np.euler_gamma) * ss.norm.ppf(1 - 1 / sr_estimates[1]) +
                        np.euler_gamma * ss.norm.ppf(1 - 1 / sr_estimates[1] * np.e ** (-1)))

    # Calculating benchmark_SR from a list of estimates
    else:
        benchmark_sr = np.array(sr_estimates).std() * \
                       ((1 - np.euler_gamma) * ss.norm.ppf(1 - 1 / len(sr_estimates)) +
                        np.euler_gamma * ss.norm.ppf(1 - 1 / len(sr_estimates) * np.e ** (-1)))

    deflated_sr = probabilistic_sharpe_ratio(observed_sr, benchmark_sr, number_of_returns,
                                             skewness_of_returns, kurtosis_of_returns)

    if benchmark_out:
        return benchmark_sr

    return deflated_sr


def minimum_track_record_length(observed_sr: float, benchmark_sr: float,
                                skewness_of_returns: float = 0,
                                kurtosis_of_returns: float = 3,
                                alpha: float = 0.05) -> float:
    """
    Calculates the minimum track record length (MinTRL) - "How long should a track
    record be in order to have statistical confidence that its Sharpe ratio is above
    a given threshold?”

    If a track record is shorter than MinTRL, we do not  have  enough  confidence
    that  the  observed Sharpe ratio ̂is above the designated Sharpe ratio threshold.

    MinTRLis expressed in terms of number of observations, not annual or calendar terms.

    :param observed_sr: (float) Sharpe ratio that is being tested
    :param benchmark_sr: (float) Sharpe ratio to which observed_SR is tested against
    :param  number_of_returns: (int) Times returns are recorded for observed_SR
    :param skewness_of_returns: (float) Skewness of returns (0 by default)
    :param kurtosis_of_returns: (float) Kurtosis of returns (3 by default)
    :param alpha: (float) Desired significance level (0.05 by default)
    :return: (float) Minimum number of track records
    """

    track_rec_length = 1 + (1 - skewness_of_returns * observed_sr +
                            (kurtosis_of_returns - 1) / 4 * observed_sr ** 2) * \
                       (ss.norm.ppf(1 - alpha) / (observed_sr - benchmark_sr)) ** (2)

    return track_rec_length

################################################################################################
# Import Data
################################################################################################
FTSEMIB= pd.read_csv (r'/home/pisati/Desktop/INDEXES/$000001-SHG_5M.csv', index_col=False,  sep=',')

FTSEMIB.drop('Tick Range', axis=1, inplace=True)
FTSEMIB.drop('MA', axis=1, inplace=True)
FTSEMIB.drop('MA.1', axis=1, inplace=True)
FTSEMIB.drop('Vol', axis=1, inplace=True)
index=FTSEMIB.to_numpy() # convert to numpy

# Fix Date interval
Date_Start = "01/03/2010"
Date_End = "11/06/2021"

TIME=pd.to_datetime(index[:,0])
ts = (TIME - np.datetime64('1990-01-01T00:00:00Z')) / np.timedelta64(1, 'h')
ts=ts.to_numpy()
Signals=pd.DataFrame([Date_Start, Date_End])
Signals2=Signals.to_numpy()
Signals3=pd.to_datetime(Signals2[:,0])
Sig = (Signals3 - np.datetime64('1990-01-01T00:00:00Z')) / np.timedelta64(1, 'h')
Sig=Sig.to_numpy()
A=(ts>Sig[0])
NEW=index[A]
FTSEMIB2= pd.DataFrame(NEW, columns = ['Date', 'Time', 'Bar#', 'Bar Index','Open', 'High', 'Low', 'Close'])
Preparation(FTSEMIB2)

ts=ts[A]
index=index[A]
B=(ts<=Sig[1])
NEW2=index[B]
FTSEMIB3= pd.DataFrame(NEW2, columns = ['Date', 'Time', 'Bar#', 'Bar Index','Open', 'High', 'Low', 'Close'])

# Compute index returns
Preparation(FTSEMIB3)

# Compute intraday, overnaight and daily returns +strategy
overnight_ret(FTSEMIB3, 0.002)

###################################################################################################################
# Print Preliminaries Stat on console
##################################################################################################################
print("/////////////////////////////////////////////////////////////////////////////////////////////////")
print("Index Cumulated Return", round(FTSEMIB3['log_ret'].sum(),3), "Index Std", round(FTSEMIB3['log_ret'].std(),4))
print("/////////////////////////////////////////////////////////////////////////////////////////////////")
print('Overnight_RET', round(FTSEMIB3['O Ret'].sum(), 3), 'Overnight_Std', round(FTSEMIB3['O Ret'].std(), 4))
print('Intraday_RET', round(FTSEMIB3['I Ret'].sum(), 3), 'Intraday_Std', round(FTSEMIB3['I Ret'].std(), 4))
print('T_RET', round(FTSEMIB3['T Ret'].sum(), 3) ,'Total_Std', round(FTSEMIB3['T Ret'].std(), 4))
print("/////////////////////////////////////////////////////////////////////////////////////////////////")
print("Strategy Performance at the index level")
print("/////////////////////////////////////////////////////////////////////////////////////////////////")
print('Str_Long', round(FTSEMIB3['Str Long'].sum(), 3) , 'Str_Long_Std', round(FTSEMIB3['Str Long'].std(), 4))
print('Str_Short', round(FTSEMIB3['Str Short'].sum(), 3) , 'Str_Short_Std', round(FTSEMIB3['Str Short'].std(), 4))
print('Index CR', round(FTSEMIB3['log_ret'].sum(), 3), 'Strat CR', round(FTSEMIB3['O Ret'].sum()+FTSEMIB3['Str Long'].sum()+FTSEMIB3['Str Short'].sum(),3) )
print('Index CR/Std', round(FTSEMIB3['log_ret'].sum()/FTSEMIB3['log_ret'].std(), 0), 'Strat CR/Std', round((FTSEMIB3['O Ret'].sum()+FTSEMIB3['Str Long'].sum()+FTSEMIB3['Str Short'].sum())/(FTSEMIB3['O Ret'].var()+FTSEMIB3['Str Long'].var()+FTSEMIB3['Str Short'].var())**(1/2),0))
print("/////////////////////////////////////////////////////////////////////////////////////////////////")
print('Check T_RET', round(FTSEMIB3['T Ret'].sum(), 3), " O_RET+I_RET", round(FTSEMIB3['O Ret'].sum()+FTSEMIB3['I Ret'].sum(),3))
print("/////////////////////////////////////////////////////////////////////////////////////////////////")

#############################################################################################################
# Compute Returns
############################################################################################################

FTSEMIB3["date_time"] = FTSEMIB3["Date"] +' '+ FTSEMIB3["Time"]
FTSEMIB3.index = pd.to_datetime(FTSEMIB3['date_time'])
FTSEMIB3 = FTSEMIB3.drop('date_time', axis=1)

Cumulated_return=FTSEMIB3['log_ret'].sum()
Cumulated_return_Positive=FTSEMIB3['log_ret'].where(FTSEMIB3['log_ret'] > 0).sum(0)
Cumulated_return_Neagative=FTSEMIB3['log_ret'].where(FTSEMIB3['log_ret'] < 0).sum(0)
Profitability_Ratio=Cumulated_return_Positive/abs(Cumulated_return_Neagative)

print("/////////////////////////////////////////////////////////////////////////////////////////////////")
print('Profitability_Ratio', Profitability_Ratio)
print("/////////////////////////////////////////////////////////////////////////////////////////////////")

FTSEMIB3['Date'] = pd.to_datetime(FTSEMIB3['Date'])
Start_Date=FTSEMIB3['Date'].iloc[1]
End_Date=FTSEMIB3['Date'].iloc[-1]
difference_in_years = relativedelta(End_Date, Start_Date).years
Average_Yearly_Return=Cumulated_return/difference_in_years
print('difference_in_years', difference_in_years, 'Average_Yearly_Return', Average_Yearly_Return)
print("/////////////////////////////////////////////////////////////////////////////////////////////////")

#Getting series of prices to represent the value of one long portfolio
FTSEMIB3['Cum_ret'] = FTSEMIB3.log_ret.cumsum()
FTSEMIB3['Cum_ret'].plot()
plt.show()
FTSEMIB3['AUM']=100*np.exp(FTSEMIB3['Cum_ret'])
FTSEMIB3['AUM'].plot()
plt.show()

# HHI
Daily_Index=FTSEMIB3['T Ret']!=0
RET=FTSEMIB3[Daily_Index]
logret_series = np.exp(RET['T Ret'])
logret_series=np.log(logret_series)
pos_concentr, neg_concentr, hourly_concentr = all_bets_concentration(logret_series, frequency='W')

print("/////////////////////////////////////////////////////////////////////////////////////////////////")
print('HHI index on positive log returns is' , pos_concentr)
print('HHI index on negative log returns is' , neg_concentr)
print('HHI index on log returns divided into Weekly bins is' , hourly_concentr)
print("/////////////////////////////////////////////////////////////////////////////////////////////////")

# Drawdown
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
print('Annualized Sharpe Ratio is', annualized_sr)
print("/////////////////////////////////////////////////////////////////////////////////////////////////")

# Final version of the repo
print("////////////////////////////////////////////////////////////////////////////////")
print("General Summary")
print("////////////////////////////////////////////////////////////////////////////////")
print("Start", Date_Start, "End", Date_End)
print("Index Cumulated Return", round(FTSEMIB3['log_ret'].sum(),3), "Index Std", round(FTSEMIB3['log_ret'].std(),4))
print('Number of Years', difference_in_years)
print('Average_Yearly_Return', Average_Yearly_Return)
print('Profitability_Ratio', Profitability_Ratio)
print('Annualized Sharpe Ratio is' , annualized_sr)

print("////////////////////////////////////////////////////////////////////////////////")
print("Return Decomposition")
print("////////////////////////////////////////////////////////////////////////////////")
print('Overnight_RET', round(FTSEMIB3['O Ret'].sum(), 3), 'Overnight_Std', round(FTSEMIB3['O Ret'].std(), 4))
print('Intraday_RET', round(FTSEMIB3['I Ret'].sum(), 3), 'Intraday_Std', round(FTSEMIB3['I Ret'].std(), 4))
print('Total_RET', round(FTSEMIB3['T Ret'].sum(), 3) ,'Total_Std', round(FTSEMIB3['T Ret'].std(), 4))
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
print('Str_Long', round(FTSEMIB3['Str Long'].sum(), 3) , 'Str_Long_Std', round(FTSEMIB3['Str Long'].std(), 4))
print('Str_Short', round(FTSEMIB3['Str Short'].sum(), 3) , 'Str_Short_Std', round(FTSEMIB3['Str Short'].std(), 4))
print('Index CR', round(FTSEMIB3['log_ret'].sum(), 3), 'Strat CR', round(FTSEMIB3['O Ret'].sum()+FTSEMIB3['Str Long'].sum()+FTSEMIB3['Str Short'].sum(),3) )
print('Index CR/Std', round(FTSEMIB3['log_ret'].sum()/FTSEMIB3['log_ret'].std(), 0), 'Strat CR/Std', round((FTSEMIB3['O Ret'].sum()+FTSEMIB3['Str Long'].sum()+FTSEMIB3['Str Short'].sum())/(FTSEMIB3['O Ret'].var()+FTSEMIB3['Str Long'].var()+FTSEMIB3['Str Short'].var())**(1/2),0))
