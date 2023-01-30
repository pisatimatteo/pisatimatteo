##################################################################################################################################
# Libraries
##################################################################################################################################

from Preparation_Comb_CV import *
from Performance_Metrics import *
from Strategy_Night import *
from Data_Import import *

import numba as nb
import pandas as pd
import numpy as np
import scipy.stats
import statistics
import warnings
warnings.filterwarnings("ignore")

###########################################################################
# Import Data
###########################################################################

# Import Index
INDEX = pd.read_csv(r'/home/pisati/Desktop/Backtesting_Night/FTSEMIB')
# Import Future
FUTURE = pd.read_csv(r'/home/pisati/Desktop/Backtesting_Night/%FTMIB')
# Import VIX
VIX = pd.read_csv(r'/home/pisati/Desktop/Backtesting_Night/$VIX', delimiter=";")

##############################################################################
# Data Preparation
##############################################################################

INDEX = Dataframe_Creation(INDEX)
Len = INDEX['Time'].size

# Find Overnight Returns
INDEX = overnight_ret(INDEX, Len)

#################################################################################
# Return Future
#################################################################################
FUTURE = FUTURE_Prepare(FUTURE)

###################################################################################
# Create a common key and merge the index and Future database
##################################################################################
FUTURE["Times"] = FUTURE["Date"] + FUTURE["Time"]
INDEX["Times"] = INDEX["Date"] + INDEX["Time"]
JOINT_FUT_INDEX = pd.merge(FUTURE, INDEX, on='Times', how='left')

##################################################################################
# VIX Preparation
#################################################################################
VIX = VIX_Prepare(VIX)

##################################################################################
# Merge with the VIX
##################################################################################
VIX["Times"] = VIX["Date"] + VIX["Time"]
MERGED_DATABASE = pd.merge(JOINT_FUT_INDEX, VIX, on='Times', how='left')

#Delate unneded columns
MERGED_DATABASE.drop(columns=['Bar#_x', 'Bar Index_x', 'High_x', 'Low_x', 'Times', 'Bar#', 'Bar Index', 'Open', 'High', 'Low', 'Date', 'Time',  "Date_y", 'Time_y', 'log_ret_y', 'Open_y', 'Close_y', 'Bar#_y', 'Bar Index_y','High_y', 'Low_y' ], axis=1, inplace=True)

# Find start of the strategy
start = MERGED_DATABASE.index.get_loc(MERGED_DATABASE['Over'].first_valid_index())
#Initial_Date = MERGED_DATABASE["Date_x"].iloc[start]

Initial_Date = '2012-03-21'
STRATEGY = compute_Future_Returns_on_open_market_time(MERGED_DATABASE, Initial_Date)

#################################################################################################################
# Find the Results of the Unoptimized Strategy
#################################################################################################################

# Prepare for Numba and set timing of entry exit points for the strategy (hard coded inside the function)

dt, ts, ret, over, vix, Sig_8, Sig_7, Enter, RESULTS = relative_strategy_times(STRATEGY)

# Basic version not optimized
threshold_Over = 0.002
threshold_VIX = 35
cost = 0.0000

RETURNS = Relative_Strategy_njit(dt, ts, ret, over, vix, Sig_8, Sig_7, Enter, RESULTS, 0.002, threshold_VIX, cost)

STRAT_RETURNS = []
STRAT_RETURNS = pd.DataFrame(RETURNS, columns = ['Date','Time','Returns', 'Over','POS','CUM_RET','Trades', 'Long_Trades', 'Short_Trades'])
STRAT_RETURNS['DATE_F'] = pd.DataFrame(RETURNS[:, 0]*np.timedelta64(1, 'm')+np.datetime64('1990-01-01T00:00:00Z'), columns = ['DATE_F'])
STRAT_RETURNS['TIME_F'] = pd.DataFrame(RETURNS[:, 1]*np.timedelta64(1, 'm')+np.datetime64('1990-01-01T00:00:00Z'), columns = ['TIME_F'])
print("Cum Ret Strategy", STRAT_RETURNS['Trades'].sum())
print("Cum Ret Future", STRATEGY['log_ret_x'].sum())

################################################################################################
# Detailed Performance Analysis of the unoptimized Strategy
################################################################################################

PAS = Perfomance_Analysis_Strategy(STRAT_RETURNS, vix, STRATEGY)

trading_days = 252
risk_free_ratio = 0.00
objective_Sharpe = 1
n_startegies_testes = 2
correaltion_strategies = 0.9

P_AN_AD = Perfomance_Analysis_Strategy_Advanced(STRATEGY, STRAT_RETURNS, trading_days, risk_free_ratio, objective_Sharpe, n_startegies_testes, correaltion_strategies)


trading_days = 252
risk_free_ratio = 0
min_param = -0.01
max_param = 0.015
increment = 0.001

##################################################################################################################
# Brute unidimensional optimization
##################################################################################################################
from scipy import optimize

threshold_VIX = 20
threshold_Over = 0.002
#rranges = (slice(20, 60, 10), slice(0, 0.5, 0.01) )
args = (dt, ts, ret, over, vix, Sig_8, Sig_7, Enter, RESULTS, threshold_Over, threshold_VIX, cost)
GRID = optim_grid_1D(Relative_Strategy_njit, trading_days, risk_free_ratio, min_param, max_param, increment, STRATEGY, *args)

##################################################################################################################
# Grid unidimensional optimization
##################################################################################################################

trading_days = 252
risk_free_ratio = 0
min_param = -0.01
max_param = 0.015
increment = 0.001
cost = 0.0000

RESULTS = np.zeros((ts.shape[0], 9))
OPTIM = []
OPTIM = pd.DataFrame(np.zeros((int((max_param-min_param)/increment), 4)), columns=['Parameter', 'Profit_Ratio', 'Cumulated_Returns', "Sharpe_Ratio"])
j = 0

for threshold_Over in np.arange(min_param, max_param, increment):

    RETURNS = Relative_Strategy_njit(dt, ts, ret, over, vix, Sig_8, Sig_7, Enter, RESULTS, threshold_Over, threshold_VIX,
                                 cost)
    STRAT_RETURNS = pd.DataFrame(RETURNS, columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades', 'Long_Trades',
                                           'Short_Trades'])
    Cumulated_return_Positive_Trades = STRAT_RETURNS['Trades'].where(STRAT_RETURNS['Trades'] > 0).sum(0)
    Cumulated_return_Neagative_Trades = STRAT_RETURNS['Trades'].where(STRAT_RETURNS['Trades'] < 0).sum(0)
    OPTIM.iloc[j, 0] = threshold_Over
    OPTIM.iloc[j, 1] = Cumulated_return_Positive_Trades / abs(Cumulated_return_Neagative_Trades)
    OPTIM.iloc[j, 2] = STRAT_RETURNS['Trades'].sum()
    STRATEGY["date_time"] = STRATEGY["Date_x"] + ' ' + STRATEGY["Time_x"]
    STRATEGY.index = pd.to_datetime(STRATEGY['date_time'])
    STRAT_RETURNS.index = pd.to_datetime(STRATEGY['date_time'])
    A = STRAT_RETURNS['Trades'] != 0
    TRADES2 = STRAT_RETURNS[A]
    logret_series = np.exp(TRADES2['Trades'])
    logret_series = np.log(logret_series)
    logret_by_days = logret_series.groupby(pd.Grouper(freq='D')).sum()
    logret_by_days = logret_by_days[logret_by_days != 0]
    annualized_sr = sharpe_ratio(logret_by_days, entries_per_year = trading_days, risk_free_rate = risk_free_ratio)
    OPTIM.iloc[j, 3] = annualized_sr
    j = j + 1

# Resulsts Profitability Ratio
Param_Best_Perf_PR = OPTIM.iloc[OPTIM.loc[:, "Profit_Ratio"].argmax(), 0]
PR_Best_Perf_PR = OPTIM.iloc[OPTIM.loc[:, "Profit_Ratio"].argmax(), 1]
CR_Best_Perf_PR = OPTIM.iloc[OPTIM.loc[:, "Profit_Ratio"].argmax(), 2]
SR_Best_Perf_PR = OPTIM.iloc[OPTIM.loc[:, "Profit_Ratio"].argmax(), 3]

# Resulsts Cum_Ret
Param_Best_Perf_CR = OPTIM.iloc[OPTIM.loc[:, "Cumulated_Returns"].argmax(), 0]
PR_Best_Over_CR = OPTIM.iloc[OPTIM.loc[:, "Cumulated_Returns"].argmax(), 1]
CR_Cum_Ret_CR = OPTIM.iloc[OPTIM.loc[:, "Cumulated_Returns"].argmax(), 2]
SR_Cum_Ret_CR = OPTIM.iloc[OPTIM.loc[:, "Cumulated_Returns"].argmax(), 3]

# Resulsts Cum_Ret
Param_Best_Perf_SR = OPTIM.iloc[OPTIM.loc[:, "Sharpe_Ratio"].argmax(), 0]
PR_Best_Over_SR = OPTIM.iloc[OPTIM.loc[:, "Sharpe_Ratio"].argmax(), 1]
CR_Cum_Ret_SR = OPTIM.iloc[OPTIM.loc[:, "Sharpe_Ratio"].argmax(), 2]
SR_Cum_Ret_SR = OPTIM.iloc[OPTIM.loc[:, "Sharpe_Ratio"].argmax(), 3]

# Print Results
print("/////////////////////////////////////////////////////////////////////////////////////////////////")
print("Param_Best_Perf_PR", round(Param_Best_Perf_PR, 4) , "PR_Best_Perf_PR", round(PR_Best_Perf_PR, 4), "CR_Best_Perf_PR", round(CR_Best_Perf_PR, 4), "SR_Best_Perf_PR", round(SR_Best_Perf_PR, 4))
print("Param_Best_Perf_CR", round(Param_Best_Perf_CR, 4) , "PR_Best_Over_CR", round(PR_Best_Over_CR, 4) , "CR_Cum_Ret_CR", round(CR_Cum_Ret_CR, 4), "SR_Cum_Ret_CR", round(SR_Cum_Ret_CR, 4))
print("Param_Best_Perf_SR", round(Param_Best_Perf_SR, 4) , "PR_Best_Over_SR", round(PR_Best_Over_SR, 4) , "CR_Cum_Ret_SR", round(CR_Cum_Ret_SR, 4), "SR_Cum_Ret_SR", round(SR_Cum_Ret_SR, 4))
print("/////////////////////////////////////////////////////////////////////////////////////////////////")
print(OPTIM)

####################################################################################################################
# Grid 2d optimization CR
####################################################################################################################

trading_days = 252
risk_free_ratio = 0

# overnight
min_param_1 = -0.01
max_param_1 = 0.015
increment_1 = 0.001
cost = 0.0000

# vix
min_param_2 = 25
max_param_2 = 50
increment_2 = 1
cost = 0.0000

RESULTS = np.zeros((ts.shape[0], 9))
STRAT_RETURNS = []
OPTIM_2D = []

OPTIM_2D = pd.DataFrame(np.zeros((int((max_param_1-min_param_1)/increment_1)*int((max_param_2-min_param_2)/increment_2), 5)), columns=['Parameter_1', 'Parameter_2', 'Profit_Ratio', 'Cumulated_Returns', "Sharpe_Ratio"])

j = 0

for threshold_VIX in np.arange(min_param_2, max_param_2, increment_2):

    for threshold_Over in np.arange(min_param_1, max_param_1, increment_1):

        STRAT_RETURNS = []
        RETURNS = Relative_Strategy_njit(dt, ts, ret, over, vix, Sig_8, Sig_7, Enter, RESULTS, threshold_Over, threshold_VIX, cost)
        STRAT_RETURNS = pd.DataFrame(RETURNS, columns = ['Date','Time','Returns', 'Over','POS','CUM_RET','Trades','Long_Trades', 'Short_Trades'])

        OPTIM_2D.iloc[j, 0] = threshold_VIX
        OPTIM_2D.iloc[j, 1] = threshold_Over
        OPTIM_2D.iloc[j, 2] = STRAT_RETURNS['Trades'].sum()

        Cumulated_return_Positive_Trades = STRAT_RETURNS['Trades'].where(STRAT_RETURNS['Trades'] > 0).sum(0)
        Cumulated_return_Neagative_Trades = STRAT_RETURNS['Trades'].where(STRAT_RETURNS['Trades'] < 0).sum(0)
        OPTIM_2D.iloc[j, 3] = Cumulated_return_Positive_Trades/abs(Cumulated_return_Neagative_Trades)

        STRATEGY["date_time"] = STRATEGY["Date_x"] + ' ' + STRATEGY["Time_x"]
        STRATEGY.index = pd.to_datetime(STRATEGY['date_time'])
        STRAT_RETURNS.index = pd.to_datetime(STRATEGY['date_time'])

        A = STRAT_RETURNS['Trades'] != 0
        TRADES2 = STRAT_RETURNS[A]
        logret_series = np.exp(TRADES2['Trades'])
        logret_series = np.log(logret_series)
        logret_by_days = logret_series.groupby(pd.Grouper(freq='D')).sum()
        logret_by_days = logret_by_days[logret_by_days != 0]
        annualized_sr = sharpe_ratio(logret_by_days, entries_per_year=trading_days, risk_free_rate=risk_free_ratio)

        OPTIM_2D.iloc[j, 4] = annualized_sr
        RESULTS = np.zeros((ts.shape[0], 9))
        j = j+1

######################################################################################
# Resulsts Profitability Ratio
#####################################################################################

Param_1_Best_Perf_PR = OPTIM_2D.iloc[OPTIM_2D.loc[:, "Profit_Ratio"].argmax(), 0]
Param_2_Best_Perf_PR = OPTIM_2D.iloc[OPTIM_2D.loc[:, "Profit_Ratio"].argmax(), 1]
CR_Best_Perf_PR = OPTIM_2D.iloc[OPTIM_2D.loc[:, "Profit_Ratio"].argmax(), 2]
PR_Best_Perf_PR = OPTIM_2D.iloc[OPTIM_2D.loc[:, "Profit_Ratio"].argmax(), 3]
SR_Best_Perf_PR = OPTIM_2D.iloc[OPTIM_2D.loc[:, "Profit_Ratio"].argmax(), 4]

######################################################################################
# Resulsts Cum_Ret
######################################################################################

Param_1_Best_Perf_CR = OPTIM_2D.iloc[OPTIM_2D.loc[:, "Cumulated_Returns"].argmax(), 0]
Param_2_Best_Perf_CR = OPTIM_2D.iloc[OPTIM_2D.loc[:, "Cumulated_Returns"].argmax(), 1]
CR_Best_Over_CR = OPTIM_2D.iloc[OPTIM_2D.loc[:, "Cumulated_Returns"].argmax(), 2]
PR_Cum_Ret_CR = OPTIM_2D.iloc[OPTIM_2D.loc[:, "Cumulated_Returns"].argmax(), 3]
SR_Cum_Ret_CR = OPTIM_2D.iloc[OPTIM_2D.loc[:, "Cumulated_Returns"].argmax(), 4]

######################################################################################
# Resulsts Cum_Ret
######################################################################################

Param_1_Best_Perf_SR = OPTIM_2D.iloc[OPTIM_2D.loc[:, "Sharpe_Ratio"].argmax(), 0]
Param_2_Best_Perf_SR = OPTIM_2D.iloc[OPTIM_2D.loc[:, "Sharpe_Ratio"].argmax(), 0]
CR_Best_Over_SR = OPTIM_2D.iloc[OPTIM_2D.loc[:, "Sharpe_Ratio"].argmax(), 1]
PR_Cum_Ret_SR = OPTIM_2D.iloc[OPTIM_2D.loc[:, "Sharpe_Ratio"].argmax(), 2]
SR_Cum_Ret_SR = OPTIM_2D.iloc[OPTIM_2D.loc[:, "Sharpe_Ratio"].argmax(), 3]

######################################################################################
# Print Results
######################################################################################

print("/////////////////////////////////////////////////////////////////////////////////////////////////")
print("Param_Best_1_Perf_PR", round(Param_1_Best_Perf_PR, 4), "Param_Best_2_Perf_PR", round(Param_2_Best_Perf_PR, 4), "PR_Best_Perf_PR", round(PR_Best_Perf_PR, 4), "CR_Best_Perf_PR", round(CR_Best_Perf_PR, 4), "SR_Best_Perf_PR", round(SR_Best_Perf_PR, 4))
print("Param_Best_1_Perf_CR", round(Param_1_Best_Perf_CR, 4), "Param_Best_2_Perf_CR", round(Param_2_Best_Perf_CR, 4), "PR_Best_Over_CR", round(PR_Best_Over_CR, 4), "CR_Cum_Ret_CR", round(CR_Cum_Ret_CR, 4), "SR_Cum_Ret_CR", round(SR_Cum_Ret_CR, 4))
print("Param_Best_1_Perf_SR", round(Param_1_Best_Perf_SR, 4), "Param_Best_2_Perf_SR", round(Param_1_Best_Perf_SR, 4), "PR_Best_Over_SR", round(PR_Best_Over_SR, 4), "CR_Cum_Ret_SR", round(CR_Cum_Ret_SR, 4), "SR_Cum_Ret_SR", round(SR_Cum_Ret_SR, 4))
print("/////////////////////////////////////////////////////////////////////////////////////////////////")
print(OPTIM_2D)


#############################################################################################################################
# Walkforward Expanding training window with grid optim
#############################################################################################################################

BASE = round((STRATEGY.shape[0] / 5))
SET = round((STRATEGY.shape[0] / 10))
OOS = pd.DataFrame(np.zeros((5, 3)))
CUM_RET_OOS = 0
a = np.zeros(shape=(8, 7))
PARTIAL = pd.DataFrame(a, columns=['Start Date', 'End Date', 'OO S', 'CUM RET OOS', 'IN S', 'Best VIX', 'Best over'])
cost = 0.0000

for m in range(0, 8):

    # First optimization with half data
    RESULTS = np.zeros((BASE + SET * m, 9))
    PROVA = []
    OPTIM = []
    OPTIM = pd.DataFrame(np.zeros((625, 3)))
    j = 0

    for threshold_VIX in np.arange(25, 50, 1):

        for threshold_Over in np.arange(-0.01, 0.015, 0.001):
            RETURNS = Relative_Strategy_njit(dt[0:BASE + SET * m], ts[0:BASE + SET * m], ret[0:BASE + SET * m],
                                         over[0:BASE + SET * m], vix[0:BASE + SET * m], Sig_8, Sig_7, Enter, RESULTS,
                                         threshold_Over, threshold_VIX, cost)
            PROVA = pd.DataFrame(RETURNS,
                                 columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades', 'Long_Trades',
                                          'Short_Trades'])
            Cumulated_return_Positive_Trades = PROVA['Trades'].where(PROVA['Trades'] > 0).sum(0)
            Cumulated_return_Neagative_Trades = PROVA['Trades'].where(PROVA['Trades'] < 0).sum(0)
            Profitability_Ratio = Cumulated_return_Positive_Trades / abs(Cumulated_return_Neagative_Trades)
            OPTIM.iloc[j, 0] = PROVA['Trades'].sum()
            OPTIM.iloc[j, 1] = threshold_VIX
            OPTIM.iloc[j, 2] = threshold_Over
            RESULTS = np.zeros((ts.shape[0], 9))
            PROVA = []
            j = j + 1

    Best_Perf = OPTIM.iloc[OPTIM[0].argmax(), 0]
    Best_VIX = OPTIM.iloc[OPTIM[0].argmax(), 1]
    Best_over = OPTIM.iloc[OPTIM[0].argmax(), 2]

    # First out of sample
    RESULTS = np.zeros((SET + 1, 9))
    PROVA = []
    RETURNS = Relative_Strategy_njit(dt[BASE + SET * m:BASE + SET * (m + 1)], ts[BASE + SET * m:BASE + SET * (m + 1)],
                                 ret[BASE + SET * m:BASE + SET * (m + 1)], over[BASE + SET * m:BASE + SET * (m + 1)],
                                 vix[BASE + SET * m:BASE + SET * (m + 1)], Sig_8, Sig_7, Enter, RESULTS, Best_over,
                                 Best_VIX, cost)
    PROVA = pd.DataFrame(RETURNS, columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades', 'Long_Trades',
                                           'Short_Trades'])
    PROVA['DATE_F'] = pd.DataFrame(RETURNS[:, 0] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'),
                                   columns=['DATE_F'])
    PROVA['TIME_F'] = pd.DataFrame(RETURNS[:, 1] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'),
                                   columns=['TIME_F'])
    Start_Date = PROVA['DATE_F'].iloc[1]
    End_Date = PROVA['DATE_F'].iloc[SET - 1]
    CUM_RET_OOS = CUM_RET_OOS + PROVA['Trades'].sum()

    PARTIAL['Start Date'][m] = Start_Date
    PARTIAL['End Date'][m] = End_Date
    PARTIAL['OO S'][m] = round(PROVA['Trades'].sum(), 2)
    PARTIAL['CUM RET OOS'][m] = round(CUM_RET_OOS, 2)
    PARTIAL['IN S'][m] = round(Best_Perf, 2)
    PARTIAL['Best VIX'][m] = round(Best_VIX, 0)
    PARTIAL['Best over'][m] = round(Best_over, 4)

    if m == 0:
        OOS_RES = PROVA
    elif m > 0:
        OOS_RES = OOS_RES.append(PROVA, ignore_index=True)

    if m == 7:
        End_Date = PROVA['DATE_F'].iloc[SET - 10]
        PARTIAL['End Date'][m] = End_Date

    if m == 0:
        print('Walkforward growing in sample:')

    print('Iteration number:', m)
    print("IN_S", round(Best_Perf, 2), 'Best_VIX', round(Best_VIX, 0), 'Best_over', round(Best_over, 4))
    print('Start_Date', Start_Date, 'End_Date', End_Date)
    print("OO_S", round(PROVA['Trades'].sum(), 2), 'CUM_RET_OOS', round(CUM_RET_OOS, 2))


##################################################################################################################
# Walkforward Fix training window with grid optim
##################################################################################################################

BASE = round((STRATEGY.shape[0] / 5))
SET = round((STRATEGY.shape[0] / 10))
OOS = pd.DataFrame(np.zeros((5, 3)))
CUM_RET_OOS = 0
OOS_RES = []
a = np.zeros(shape=(8, 7))
PARTIAL = pd.DataFrame(a, columns=['Start Date', 'End Date', 'OO S', 'CUM RET OOS', 'IN S', 'Best VIX', 'Best over'])
cost = 0.0003

for m in range(0, 8):

    # First optimization with half data
    RESULTS = np.zeros((BASE + SET * m, 9))
    PROVA = []
    OPTIM = []
    OPTIM = pd.DataFrame(np.zeros((625, 3)))
    j = 0

    for threshold_VIX in np.arange(25, 50, 1):

        for threshold_Over in np.arange(-0.01, 0.015, 0.001):
            RETURNS = Relative_Strategy_njit(dt[SET * m:SET * (m + 2)], ts[SET * m:SET * (m + 2)],
                                         ret[SET * m:SET * (m + 2)], over[SET * m:SET * (m + 2)],
                                         vix[SET * m:SET * (m + 2)], Sig_8, Sig_7, Enter, RESULTS, threshold_Over,
                                         threshold_VIX, cost)
            PROVA = pd.DataFrame(RETURNS,
                                 columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades', 'Long_Trades',
                                          'Short_Trades'])
            OPTIM.iloc[j, 0] = PROVA['Trades'].sum()
            OPTIM.iloc[j, 1] = threshold_VIX
            OPTIM.iloc[j, 2] = threshold_Over
            RESULTS = np.zeros((ts.shape[0], 9))
            PROVA = []
            j = j + 1

    Best_Perf = OPTIM.iloc[OPTIM[0].argmax(), 0]
    Best_VIX = OPTIM.iloc[OPTIM[0].argmax(), 1]
    Best_over = OPTIM.iloc[OPTIM[0].argmax(), 2]

    # First out of sample
    RESULTS = np.zeros((SET + 1, 9))
    PROVA = []
    RETURNS = Relative_Strategy_njit(dt[SET * (m + 2):SET * (m + 3)], ts[SET * (m + 2):SET * (m + 3)],
                                 ret[SET * (m + 2):SET * (m + 3)], over[SET * (m + 2):SET * (m + 3)],
                                 vix[SET * (m + 2):SET * (m + 3)], Sig_8, Sig_7, Enter, RESULTS, Best_over, Best_VIX,
                                 cost)
    PROVA = pd.DataFrame(RETURNS, columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades', 'Long_Trades',
                                           'Short_Trades'])
    PROVA['DATE_F'] = pd.DataFrame(RETURNS[:, 0] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'),
                                   columns=['DATE_F'])
    PROVA['TIME_F'] = pd.DataFrame(RETURNS[:, 1] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'),
                                   columns=['TIME_F'])
    Start_Date = PROVA['DATE_F'].iloc[1]
    End_Date = PROVA['DATE_F'].iloc[SET - 1]
    CUM_RET_OOS = CUM_RET_OOS + PROVA['Trades'].sum()

    PARTIAL['Start Date'][m] = Start_Date
    PARTIAL['End Date'][m] = End_Date
    PARTIAL['OO S'][m] = round(PROVA['Trades'].sum(), 2)
    PARTIAL['CUM RET OOS'][m] = round(CUM_RET_OOS, 2)
    PARTIAL['IN S'][m] = round(Best_Perf, 2)
    PARTIAL['Best VIX'][m] = round(Best_VIX, 0)
    PARTIAL['Best over'][m] = round(Best_over, 4)

    if m == 0:
        OOS_RES = PROVA

    elif m > 0:
        OOS_RES = OOS_RES.append(PROVA, ignore_index=True)

    if m == 7:
        End_Date = PROVA['DATE_F'].iloc[SET - 10]
        PARTIAL['End Date'][m] = End_Date

    if m == 0:
        print("Walkforward fixed training 20% test 10%")

    print('Iteration number:', m)
    print("IN_S", round(Best_Perf, 2), 'Best_VIX', round(Best_VIX, 0), 'Best_over', round(Best_over, 4))
    print('Start_Date', Start_Date, 'End_Date', End_Date)
    print("OO_S", round(PROVA['Trades'].sum(), 2), 'CUM_RET_OOS', round(CUM_RET_OOS, 2))


######################################################################################################################
# Cross Validation with embargo and pruning
######################################################################################################################

TS = pd.Series(ts)
cv_gen = PurgedKFold(n_splits=10, samples_info_sets=TS, pct_embargo=0.0001)
TRAIN = []
TEST = []
CUM_RET_OOS = 0
cost = 0.0003
i = 0

for train, test in cv_gen.split(X=STRATEGY, y=TS):

    dt_train = dt[train]
    ts_train = ts[train]
    ret_train = ret[train]
    over_train = over[train]
    vix_train = vix[train]

    dt_test = dt[test]
    ts_test = ts[test]
    ret_test = ret[test]
    over_test = over[test]
    vix_test = vix[test]

    # Initialization for grid optimization
    RESULTS = np.zeros((ts_train.shape[0], 9))
    PROVA = []
    OPTIM = []
    OPTIM = pd.DataFrame(np.zeros((625, 3)))
    j = 0

    for threshold_VIX in np.arange(25, 50, 1):
        for threshold_Over in np.arange(-0.01, 0.015, 0.001):
            RETURNS = Relative_Strategy_njit(dt_train, ts_train, ret_train, over_train, vix_train, Sig_8, Sig_7, Enter,
                                         RESULTS, threshold_Over, threshold_VIX, cost)
            PROVA = pd.DataFrame(RETURNS,
                                 columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades', 'Long_Trades',
                                          'Short_Trades'])
            OPTIM.iloc[j, 0] = PROVA['Trades'].sum()
            OPTIM.iloc[j, 1] = threshold_VIX
            OPTIM.iloc[j, 2] = threshold_Over
            RESULTS = np.zeros((ts_train.shape[0], 9))
            PROVA = []
            j = j + 1

    Best_Perf = OPTIM.iloc[OPTIM[0].argmax(), 0]
    Best_VIX = OPTIM.iloc[OPTIM[0].argmax(), 1]
    Best_over = OPTIM.iloc[OPTIM[0].argmax(), 2]

    # out of sample
    RESULTS = np.zeros((ts_test.shape[0], 9))
    PROVA = []
    RETURNS = Relative_Strategy_njit(dt_test, ts_test, ret_test, over_test, vix_test, Sig_8, Sig_7, Enter, RESULTS,
                                 Best_over, Best_VIX, cost)
    PROVA = pd.DataFrame(RETURNS, columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades', 'Long_Trades',
                                           'Short_Trades'])
    PROVA['DATE_F'] = pd.DataFrame(RETURNS[:, 0] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'),
                                   columns=['DATE_F'])
    PROVA['TIME_F'] = pd.DataFrame(RETURNS[:, 1] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'),
                                   columns=['TIME_F'])
    CUM_RET_OOS = CUM_RET_OOS + PROVA['Trades'].sum()

    if i == 0:
        print("Ten fold cross validation with embargo and pruning")

    print("N", i, "IN_S", round(Best_Perf, 2), 'Best_VIX', round(Best_VIX, 0), 'Best_over', round(Best_over, 4))
    print("N", i, "OO_S", round(PROVA['Trades'].sum(), 2), 'CUM_RET_OOS', round(CUM_RET_OOS, 2))
    i = i+1


#######################################################################################################################
# Combinatorial Cross-Validation
#######################################################################################################################

TS = pd.Series(ts)
n_splits = 6
n_test_splits = 2
pct_embargo = 0
test_ranges: [(int, int)] = [(ix[0], ix[-1] + 1) for ix in np.array_split(np.arange(TS.shape[0]), n_splits)]
splits_indices = {}
for index, [start_ix, end_ix] in enumerate(test_ranges):
    splits_indices[index] = [start_ix, end_ix]

N_PATHS = get_number_of_backtest_paths(n_splits, n_test_splits)
PATHS = pd.DataFrame(index=range(N_PATHS), columns=range(n_splits))
# Possible test splits for each fold
combinatorial_splits = list(combinations(list(splits_indices.keys()), n_test_splits))
combinatorial_test_ranges = []  # List of test indices formed from combinatorial splits

for combination in combinatorial_splits:
    temp_test_indices = []  # Array of test indices for current split combination
    for int_index in combination:
        temp_test_indices.append(splits_indices[int_index])
    combinatorial_test_ranges.append(temp_test_indices)

cv_gen2 = CombinatorialPurgedKFold(n_splits=6, n_test_splits=2, samples_info_sets=TS, pct_embargo=0.00)
TRAIN = []
TEST = []
CUM_RET_OOS = 0
cost = 0.0003
l = 0

for train, test in cv_gen2.split(X=STRATEGY, y=TS):

    # print("test", test)

    main_list = list(set(train) - set(test))

    dt_train = dt[main_list]
    ts_train = ts[main_list]
    ret_train = ret[main_list]
    over_train = over[main_list]
    vix_train = vix[main_list]

    # Initialization for grid optimization
    RESULTS = np.zeros((ts_train.shape[0], 9))
    PROVA = []
    OPTIM = []
    OPTIM = pd.DataFrame(np.zeros((625, 3)))
    j = 0

    for threshold_VIX in np.arange(25, 50, 1):
        for threshold_Over in np.arange(-0.01, 0.015, 0.001):
            RETURNS = Relative_Strategy_njit(dt_train, ts_train, ret_train, over_train, vix_train, Sig_8, Sig_7, Enter,
                                         RESULTS, threshold_Over, threshold_VIX, cost)
            PROVA = pd.DataFrame(RETURNS,
                                 columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades', 'Long_Trades',
                                          'Short_Trades'])
            OPTIM.iloc[j, 0] = PROVA['Trades'].sum()
            OPTIM.iloc[j, 1] = threshold_VIX
            OPTIM.iloc[j, 2] = threshold_Over
            RESULTS = np.zeros((ts_train.shape[0], 9))
            PROVA = []
            j = j + 1

    Best_Perf = OPTIM.iloc[OPTIM[0].argmax(), 0]
    Best_VIX = OPTIM.iloc[OPTIM[0].argmax(), 1]
    Best_over = OPTIM.iloc[OPTIM[0].argmax(), 2]
    print("N", l, "IN_S", round(Best_Perf, 2), 'Best_VIX', round(Best_VIX, 0), 'Best_over', round(Best_over, 4))

    # out of sample
    for n in range(0, n_test_splits):
        test2 = np.arange(combinatorial_test_ranges[l][n][0], combinatorial_test_ranges[l][n][1], 1)
        # print("test2", test2)

        dt_test = dt[test2]
        ts_test = ts[test2]
        ret_test = ret[test2]
        over_test = over[test2]
        vix_test = vix[test2]

        RESULTS = np.zeros((ts_test.shape[0], 9))
        PROVA = []
        RETURNS = Relative_Strategy_njit(dt_test, ts_test, ret_test, over_test, vix_test, Sig_8, Sig_7, Enter, RESULTS,
                                     Best_over, Best_VIX, cost)
        PROVA = pd.DataFrame(RETURNS,
                             columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades', 'Long_Trades',
                                      'Short_Trades'])
        PROVA['DATE_F'] = pd.DataFrame(RETURNS[:, 0] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'),
                                       columns=['DATE_F'])
        PROVA['TIME_F'] = pd.DataFrame(RETURNS[:, 1] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'),
                                       columns=['TIME_F'])
        cum_ret = PROVA['Trades'].sum()
        CUM_RET_OOS = CUM_RET_OOS + PROVA['Trades'].sum()
        print("N", l, "OO_S", round(PROVA['Trades'].sum(), 2), 'CUM_RET_OOS', round(CUM_RET_OOS, 2))

        for m in range(0, N_PATHS, 1):
            Indicator = combinatorial_splits[l][n]
            if pd.isnull(PATHS[Indicator][m]):
                #print("Indicator", Indicator, "m", m)
                PATHS[Indicator][m] = cum_ret
                #print(PATHS)
                break
    l = l + 1

OOS_PATHS = PATHS.sum(axis=1, skipna=True)
print("OOS_PATHS", OOS_PATHS)
print("Number of Paths", get_number_of_backtest_paths(n_splits, n_test_splits))
print("Average OOS per path ", round(CUM_RET_OOS / get_number_of_backtest_paths(n_splits, n_test_splits), 2))
