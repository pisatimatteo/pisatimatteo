##################################################################################################################################
# Libraries
##################################################################################################################################

from Preparation_Comb_CV import *
from Performance_Metrics import *
from Strategy_Night import *
import pandas as pd
import numpy as np
import numba as nb

import matplotlib
import math
import csv
import re
import requests
import time
from datetime import datetime, date, timedelta
from threading import Timer


###################################################################################################################################
# Functions
###################################################################################################################################

def overnight_ret(index):
    p_open = index.iloc[0, 4]  # column of open bars
    index.iloc[0, 8] = p_open  # column of open days
    index.iloc[0, 8] = index.iloc[0, 8].replace(',', '.')
    index.iloc[0, 8] = pd.to_numeric(index.iloc[0, 8])
    p_open = index.iloc[0, 8]

    p_close = index.iloc[Len - 1, 7]  # column of close
    index.iloc[Len - 1, 9] = p_close  # column of close days
    index.iloc[Len - 1, 9] = index.iloc[Len - 1, 9].replace(',', '.')
    index.iloc[Len - 1, 9] = pd.to_numeric(index.iloc[Len - 1, 9])
    p_close = index.iloc[Len - 1, 9]

    for i in range(1, index['Date'].size - 1):

        if index.iloc[i, 0] != index.iloc[i - 1, 0]:  # change in day,new day arrived

            p_open = index.iloc[i, 4]  # column of open bars
            index.iloc[i, 8] = p_open  # column of open days
            index.iloc[i, 8] = index.iloc[i, 8].replace(',', '.')
            index.iloc[i, 8] = pd.to_numeric(index.iloc[i, 8])
            p_open = index.iloc[i, 8]
            index.iloc[i, 10] = np.log(p_open) - np.log(p_close)  # overnight ret
            # rint(index.iloc[i,10])

        elif index.iloc[i, 0] != index.iloc[i + 1, 0]:  # change in day, old day finished

            p_close = index.iloc[i, 7]  # column of close
            index.iloc[i, 9] = p_close  # column of close days
            index.iloc[i, 9] = index.iloc[i, 9].replace(',', '.')
            index.iloc[i, 9] = pd.to_numeric(index.iloc[i, 9])
            p_close = index.iloc[i, 9]


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


####################################################################################################################
# Import Data filter and preparation
####################################################################################################################
ECONOMICS = pd.read_excel(r'/home/pisati/Desktop/Backtesting_Night/ECONOMICS.xlsx')
FTSEMIB = pd.read_csv(r'/home/pisati/Desktop/Backtesting_Night/FTSEMIB')

# Reverse Sort and Turn Time Date into strings
FTSEMIB = FTSEMIB.iloc[::-1]
FTSEMIB['Time'] = FTSEMIB['Time'].astype(str)
FTSEMIB['Date'] = FTSEMIB['Date'].astype(str)

# Create columns for analyses
Len = FTSEMIB['Time'].size
FTSEMIB['Start'] = [0] * Len
FTSEMIB['End'] = [0] * Len
FTSEMIB['Over'] = [0] * Len
FTSEMIB['Position'] = [0] * Len
FTSEMIB['Start'] = FTSEMIB['Start'].astype(str)
FTSEMIB['End'] = FTSEMIB['End'].astype(str)

# Find first Open
p_open = FTSEMIB.iloc[0, 4]
FTSEMIB.iloc[0, 8] = p_open
FTSEMIB.iloc[0, 8] = FTSEMIB.iloc[0, 8].replace(',', '.')
FTSEMIB.iloc[0, 8] = pd.to_numeric(FTSEMIB.iloc[0, 8])
p_open = FTSEMIB.iloc[0, 8]

# Find last Close
p_close = FTSEMIB.iloc[Len - 1, 7]
FTSEMIB.iloc[Len - 1, 9] = p_close
FTSEMIB.iloc[Len - 1, 9] = FTSEMIB.iloc[Len - 1, 9].replace(',', '.')
FTSEMIB.iloc[Len - 1, 9] = pd.to_numeric(FTSEMIB.iloc[Len - 1, 9])
p_close = FTSEMIB.iloc[Len - 1, 9]

# Total Return in the period
TOTALRETURN = np.log(p_close) - np.log(p_open)
print('Total Return in the Period', FTSEMIB.iloc[0, 0], '/', FTSEMIB.iloc[Len - 1, 0], '=', TOTALRETURN)

# Find Overnight Returns
overnight_ret(FTSEMIB, Len)
# Import Future Database
FTSEMIBFUT= pd.read_csv (r'/home/pisati/Desktop/Backtesting_Night/%FTMIB')
FTSEMIBFUT[['Date','Time', "Bar#", 'Bar Index', 'Open', 'High', 'Low', 'Close']] = FTSEMIBFUT['Date;Time;Bar#;Bar Index;Open;High;Low;Close'].str.split(';',expand=True)
del FTSEMIBFUT['Date;Time;Bar#;Bar Index;Open;High;Low;Close']

# Find close bar return for the Future
FTSEMIBFUT = FTSEMIBFUT.iloc[::-1]
Preparation(FTSEMIBFUT)

# Create a common key and merge the index and uture database
FTSEMIBFUT["Times"] = FTSEMIBFUT["Date"] + FTSEMIBFUT["Time"]
FTSEMIB["Times"] = FTSEMIB["Date"] + FTSEMIB["Time"]
FTSEMIBFUT = pd.merge(FTSEMIBFUT, FTSEMIB, on='Times', how='left')

# Import VIX
VIX= pd.read_csv (r'/home/pisati/Desktop/Backtesting_Night/$VIX', delimiter=";")
VIX = VIX.iloc[::-1]
Preparation(VIX)

#Merge with the VIX
VIX["Times"] = VIX["Date"] + VIX["Time"]
FTSEMIBFUT = pd.merge(FTSEMIBFUT, VIX, on='Times', how='left')
LIST_COLUMNS=list(FTSEMIBFUT.columns)

####################################################################################################################
# Create indicators
####################################################################################################################

Moving_Average_90 = FTSEMIBFUT["Close_x"].rolling(900).mean()
Moving_Average_30 = FTSEMIBFUT["Close_x"].rolling(300).mean()

INDIC = pd.DataFrame(columns=['Momentum'], index=range(FTSEMIBFUT["Times"].shape[0]))

for i in range(90, FTSEMIBFUT["Close_x"].shape[0]):

    if FTSEMIBFUT["Close_x"].iloc[i] > Moving_Average_30.iloc[i] > Moving_Average_90.iloc[i]:
        INDIC.iloc[i] = 1
    elif FTSEMIBFUT["Close_x"].iloc[i] < Moving_Average_30.iloc[i] < Moving_Average_90.iloc[i]:
        INDIC.iloc[i] = -1
    else:
        INDIC.iloc[i] = 0

#Delate unneded columns to speed up
del FTSEMIBFUT[ 'Bar#_x']
del FTSEMIBFUT[ 'Bar Index_x']
del FTSEMIBFUT[ 'Open_x']
del FTSEMIBFUT['High_x']
del FTSEMIBFUT['Low_x']
del FTSEMIBFUT['Times']
del FTSEMIBFUT['Bar#_y']
del FTSEMIBFUT['Bar Index_y']
del FTSEMIBFUT['High_y']
del FTSEMIBFUT['Low_y']
del FTSEMIBFUT['Bar#']
del FTSEMIBFUT['Bar Index']
del FTSEMIBFUT['Open']
del FTSEMIBFUT['High']
del FTSEMIBFUT['Low']
del FTSEMIBFUT['Date']
del FTSEMIBFUT['Time']
del FTSEMIBFUT['Date_y']
del FTSEMIBFUT['Time_y']
del FTSEMIBFUT['Start']
del FTSEMIBFUT['End']
del FTSEMIBFUT['log_ret_y']
del FTSEMIBFUT['Open_y']
del FTSEMIBFUT['Close_y']
del FTSEMIBFUT['Close_x']

# Find start of the strategy
#start=FTSEMIBFUT.index.get_loc(FTSEMIBFUT['Over'].first_valid_index())
#STRATEGY=FTSEMIBFUT[start:-1]
Starting=FTSEMIBFUT.index[FTSEMIBFUT['Date_x'] =='2012-03-21' ].tolist()
STARTING=Starting[0]
STRATEGY=FTSEMIBFUT[STARTING:]
STRATEGY.insert(6, 'Cum_Ret', 0)
STRATEGY.insert(7, 'Trades', 0)
STRATEGY['Position']=0
INDIC=INDIC[STARTING:]


###################################################################################################
# Turn to numpy for numba
###################################################################################################
index=STRATEGY.to_numpy()
TIME=pd.to_datetime(index[:,1])
ts = (TIME - np.datetime64('1990-01-01T00:00:00Z')) / np.timedelta64(1, 'm')
ts=ts.to_numpy()
ts=np.round(ts,0)

DATE=pd.to_datetime(index[:,0])
dt = (DATE - np.datetime64('1990-01-01T00:00:00Z')) / np.timedelta64(1, 'm')
dt=dt.to_numpy()
dt=np.round(dt,0)

RET=STRATEGY['log_ret_x'].fillna(0)
ret=RET.to_numpy()

OVER=STRATEGY['Over']
OVER= OVER.fillna(0)
over=OVER.to_numpy()

VIX=STRATEGY['Close']
VIX = VIX.fillna(0).replace(to_replace=0, method='ffill')
vix=VIX.to_numpy()

INDIC=INDIC.fillna(0)
indic=INDIC.to_numpy()

RESULTS=np.zeros((ts.shape[0], 9))

##################################################################################################################
# Strategy calibration
##################################################################################################################
Time_Entry_1 = "08:00:00"
Time_Exit_1 = "16:30:00"
Time_Entry_2 = Time_Exit_1
Time_Exit_2 = "23:55:00"
Time_Entry_3 = "00:00:00"
Time_Exit_3 = Time_Entry_1
VIX_Time="15:30:00"
Signals=pd.DataFrame([Time_Entry_1, Time_Exit_1, Time_Entry_2, Time_Exit_2, Time_Entry_3, Time_Exit_3, VIX_Time])
Signals2=Signals.to_numpy()
Signals3=pd.to_datetime(Signals2[:,0])
Sig = (Signals3 - np.datetime64('1990-01-01T00:00:00Z')) / np.timedelta64(1, 'm')
Sig=Sig.to_numpy()
Sig_8=np.round(Sig,0)


# Signals 2
Time_Entry_1 = "07:00:00"
Time_Exit_1 = "15:30:00"
Time_Entry_2 = Time_Exit_1
Time_Exit_2 = "23:55:00"
Time_Entry_3 = "00:00:00"
Time_Exit_3 = Time_Entry_1
VIX_Time="15:30:00"
Signals2=pd.DataFrame([Time_Entry_1, Time_Exit_1, Time_Entry_2, Time_Exit_2, Time_Entry_3, Time_Exit_3, VIX_Time])
Signals2=Signals2.to_numpy()
Signals3=pd.to_datetime(Signals2[:,0])
Sig2 = (Signals3 - np.datetime64('1990-01-01T00:00:00Z')) / np.timedelta64(1, 'm')
Sig2=Sig2.to_numpy()
Sig_7=np.round(Sig2,0)

# Entry
Time_Entry_A = "07:00:00"
Time_Entry_B = "08:00:00"
ENTRY=pd.DataFrame([Time_Entry_A, Time_Entry_B])
ENTRY2=ENTRY.to_numpy()
ENTRY3=pd.to_datetime(ENTRY2[:,0])
Enter = (ENTRY3 - np.datetime64('1990-01-01T00:00:00Z')) / np.timedelta64(1, 'm')
Enter = Enter.to_numpy()
Enter=np.round(Enter,0)

# Basic version for debugging only
threshold_Over=0.002
threshold_VIX=35
cost=0.000

###########################################################################################################################
# Strategy profiling and testing with unoptimized parameters
###########################################################################################################################

RESULTS=np.zeros((ts.shape[0], 9))
PROVA=[]


@numba.njit(
    nb.typeof(RESULTS)(nb.typeof(dt), nb.typeof(ts), nb.typeof(ret), nb.typeof(over), nb.typeof(vix), nb.typeof(Sig_8),
                       nb.typeof(Sig_7), nb.typeof(Enter), nb.typeof(RESULTS), nb.typeof(threshold_Over),
                       nb.typeof(threshold_VIX), nb.typeof(cost)), cache=True, fastmath=True)
def Times_analyse_njit(Dates, Times, Returns, OVER, VIX, Signals_8, Signals_7, ENTER, RESULTS, threshold_Over,
                       threshold_VIX, cost):
    # Initialize values
    CURRENT_OVER = 0
    CURRENT_VIX = 0
    CRET = 0
    POS = 0
    POSL = 0
    Signals = Signals_8

    for i in range(1, Times.shape[0]):

        RESULTS[i, 0] = Dates[i]
        RESULTS[i, 1] = Times[i]
        RESULTS[i, 2] = Returns[i]
        RESULTS[i, 3] = OVER[i]

        if Times[i] == Signals[6]:  # update VIX
            CURRENT_VIX = VIX[i]  # VIX close bar

        if OVER[i] != 0:
            CURRENT_OVER = OVER[i]  # over bar

            if Times[i] == ENTER[0]:  # Hour change

                Signals = Signals_7

            elif Times[i] == ENTER[1]:

                Signals = Signals_8

                # Find Positions
        if CURRENT_OVER > threshold_Over and threshold_VIX > CURRENT_VIX:

            if Signals[1] > Times[i] >= Signals[0]:
                POSL = POS
                POS = 1
                RESULTS[i, 4] = 1

        elif CURRENT_OVER < threshold_Over and threshold_VIX > CURRENT_VIX:

            if Signals[1] > Times[i] >= Signals[0]:
                POSL = POS
                POS = -1
                RESULTS[i, 4] = -1

        if CURRENT_OVER > threshold_Over and threshold_VIX < CURRENT_VIX:  # only for fast track
            POSL = POS
            POS = 0
            RESULTS[i, 4] = 0

        elif CURRENT_OVER < threshold_Over and threshold_VIX < CURRENT_VIX:  # only for fast track
            POSL = POS
            POS = 0
            RESULTS[i, 4] = 0

        if Signals[3] >= Times[i] >= Signals[2] and threshold_VIX > CURRENT_VIX:
            POSL = POS
            POS = 1
            RESULTS[i, 4] = 1

        elif Signals[5] > Times[i] >= Signals[4] and threshold_VIX > CURRENT_VIX:
            POSL = POS
            POS = 1
            RESULTS[i, 4] = 1

        # Cumulate Returns
        if POSL == 0 and POS == 0:
            CRET = 0

        elif POSL == 0:

            if POS == 1:
                CRET = Returns[i]
                RESULTS[i, 5] = Returns[i]

            if POS == -1:
                CRET = -Returns[i]
                RESULTS[i, 5] = -Returns[i]

        elif POSL == 1 and POS == 1:
            CRET = Returns[i] + CRET
            RESULTS[i, 5] = Returns[i] + RESULTS[i - 1, 5]

        elif POSL == -1 and POS == -1:
            CRET = -Returns[i] + CRET
            RESULTS[i, 5] = -Returns[i] + RESULTS[i - 1, 5]


        elif POSL == 1 and POS == 0:  # data entry morning received with delay

            RESULTS[i, 6] = CRET + Returns[i] - cost
            # Long
            RESULTS[i, 7] = RESULTS[i, 6]
            CRET = 0

        elif POSL == -1 and POS == 0:
            RESULTS[i, 6] = CRET - Returns[i] - cost
            # Long
            RESULTS[i, 8] = RESULTS[i, 6]
            CRET = 0

        elif POSL == -1 and POS == 1:
            RESULTS[i, 6] = CRET - Returns[i] - cost
            # Long
            RESULTS[i, 8] = RESULTS[i, 6]
            CRET = 0

        elif POSL == 1 and POS == -1:  # data entry morning received with delay
            RESULTS[i, 6] = CRET + Returns[i] - cost
            # Long
            RESULTS[i, 7] = RESULTS[i, 6]
            CRET = 0

    # Performance metrics
    return RESULTS


RETURNS = Times_analyse_njit(dt, ts, ret, over, vix, Sig_8, Sig_7, Enter, RESULTS, 0.002, threshold_VIX, cost)
PROVA = pd.DataFrame(RETURNS, columns = ['Date','Time','Returns', 'Over','POS','CUM_RET','Trades', 'Long_Trades', 'Short_Trades'])
PROVA['DATE_F'] = pd.DataFrame(RETURNS[:,0]*np.timedelta64(1, 'm')+np.datetime64('1990-01-01T00:00:00Z'), columns = ['DATE_F'])
PROVA['TIME_F'] = pd.DataFrame(RETURNS[:,1]*np.timedelta64(1, 'm')+np.datetime64('1990-01-01T00:00:00Z'), columns = ['TIME_F'])
print("Cum Ret",PROVA['Trades'].sum())
#cProfile.run('re.compile("Times_analyse_njit")')

#############################################################################################
# First Analysis
#############################################################################################

Cumulated_return_strategy = PROVA['Trades'].sum()
Cumulated_return_Positive_Trades = PROVA['Trades'].where(PROVA['Trades'] > 0).sum(0)
Cumulated_return_Neagative_Trades = PROVA['Trades'].where(PROVA['Trades'] < 0).sum(0)
Cumulated_return_High_Vol = PROVA['Trades'].where(vix > 25).sum(0)
Cumulated_return_Low_Vol = PROVA['Trades'].where(vix < 25).sum(0)
BULL_TRADES = 0
BEAR_TRADES = 0
LATERAL_TRADES = 0

for i in range(0, len(PROVA['Trades'])):

    if PROVA['Trades'].iloc[i] != 0:
        if (INDIC.iloc[i] > 0).bool():
            BULL_TRADES = BULL_TRADES + PROVA['Trades'].iloc[i]

    if PROVA['Trades'].iloc[i] != 0:
        if (INDIC.iloc[i] < 0).bool():
            BEAR_TRADES = BEAR_TRADES + PROVA['Trades'].iloc[i]
Profitability_Ratio=Cumulated_return_Positive_Trades/abs(Cumulated_return_Neagative_Trades)

Start_Date=PROVA['DATE_F'].iloc[1]
End_Date=PROVA['DATE_F'].iloc[-1]
difference_in_years = relativedelta(End_Date, Start_Date).years
Average_Yearly_Return=Cumulated_return_strategy/difference_in_years

A=PROVA['Trades']!=0
TRADES=PROVA[A]
#TRADES['Trades'].hist(bins=50)

#############################################################################################
# Trade Analysis
#############################################################################################

Number_of_Trades=TRADES['Trades'].count()
Number_of_Negative_Trades=TRADES['Trades'].where(TRADES['Trades'] < 0).count()
Hit_Ratio=(Number_of_Trades-Number_of_Negative_Trades)/Number_of_Trades
Frequency_of_bets=Number_of_Trades/difference_in_years
Average_Trade_Return=TRADES['Trades'].mean()
Average_Return_from_hits=TRADES['Trades'].where(TRADES['Trades'] > 0).mean()
Average_Return_from_misses=TRADES['Trades'].where(TRADES['Trades'] < 0).mean()
Trades_Variance=statistics.variance(TRADES['Trades'])
Trades_Standard_Deviation=statistics.stdev(TRADES['Trades'])
Trades_Skwness=skew(TRADES['Trades'])
Trades_Kurtosis=kurtosis(TRADES['Trades'])

Cumulated_Return_Long=TRADES['Long_Trades'].sum()
Number_of_Trades_Long=TRADES['Long_Trades'].where(TRADES['Long_Trades'] != 0).count()
Ratio_of_longs=Number_of_Trades_Long/Number_of_Trades
Number_of_Positive_Trades_Long=TRADES['Long_Trades'].where(TRADES['Long_Trades'] > 0).count()
Hit_Ratio_Long=Number_of_Positive_Trades_Long/Number_of_Trades_Long
Average_Trade_Return_Long=TRADES['Long_Trades'].mean()
Average_Return_from_hits_Long=TRADES['Long_Trades'].where(TRADES['Long_Trades'] > 0).mean()
Average_Return_from_misses_Long=TRADES['Long_Trades'].where(TRADES['Long_Trades'] < 0).mean()
Trades_Variance_Long=statistics.variance(TRADES['Long_Trades'])
Trades_Standard_Deviation_Long=statistics.stdev(TRADES['Long_Trades'])
Trades_Skwness_Long=skew(TRADES['Long_Trades'])
Trades_Kurtosis_Long=kurtosis(TRADES['Long_Trades'])
Average_Yearly_Return_Long=Cumulated_Return_Long/difference_in_years

Cumulated_Return_Short=TRADES['Short_Trades'].sum()
Number_of_Trades_Short=TRADES['Short_Trades'].where(TRADES['Short_Trades'] != 0).count()
Number_of_Positive_Trades_Short=TRADES['Short_Trades'].where(TRADES['Short_Trades'] > 0).count()
Hit_Ratio_Short=Number_of_Positive_Trades_Short/Number_of_Trades_Short
Average_Trade_Return_Short=TRADES['Short_Trades'].mean()
Average_Return_from_hits_Short=TRADES['Short_Trades'].where(TRADES['Short_Trades'] > 0).mean()
Average_Return_from_misses_Short=TRADES['Short_Trades'].where(TRADES['Short_Trades'] < 0).mean()
Trades_Variance_Short=statistics.variance(TRADES['Short_Trades'])
Trades_Standard_Deviation_Short=statistics.stdev(TRADES['Short_Trades'])
Trades_Skwness_Short=skew(TRADES['Short_Trades'])
Trades_Kurtosis_Short=kurtosis(TRADES['Short_Trades'])
Average_Yearly_Return_Short=Cumulated_Return_Short/difference_in_years

#######################################################################################################
# AUM and Leverage
#######################################################################################################

CAPITAL=100000
AUM=120000
LEVERAGE=AUM/CAPITAL
MAXIMUM_DOLLAR_POSITION_SIZE=AUM #Fixed baet in this strategy
Average_AUM=Number_of_Positive_Trades=AUM*(PROVA['POS'].where(PROVA['POS'] !=0).count()/len(PROVA['POS']))

Lev_Average_Yearly_Return=(Cumulated_return_strategy/difference_in_years)*LEVERAGE
Lev_Average_Trade_Return=TRADES['Trades'].mean()*LEVERAGE
Lev_Average_Return_from_hits=TRADES['Trades'].where(TRADES['Trades'] > 0).mean()*LEVERAGE
Lev_Average_Return_from_misses=TRADES['Trades'].where(TRADES['Trades'] < 0).mean()*LEVERAGE
Lev_Trades_Variance = statistics.variance(TRADES['Trades'])*LEVERAGE
Lev_Trades_Standard_Deviation=statistics.stdev(TRADES['Trades'])*LEVERAGE**(1/2)

print('////////////////////////////////////////////////////////////////////////////////////////////////////////////')
print('PERFORMANCE SUMMARY')
print('////////////////////////////////////////////////////////////////////////////////////////////////////////////')

print('Cumulated return strategy', round(Cumulated_return_strategy,2))
print('Cumulated return Winning Trades', round(Cumulated_return_Positive_Trades,2))
print('Cumulated_return_Negative_Trades', round(Cumulated_return_Neagative_Trades,2))
print('Profitability Ratio', round(Profitability_Ratio, 2))
print("Cumulated_return_High_Vol", round(Cumulated_return_High_Vol,2), "Cumulated_return_Low_Vol", round(Cumulated_return_Low_Vol,2))
print("TOTAL", round(PROVA['Trades'].sum(),2), "BULL_TRADES", round(BULL_TRADES, 2) , "BEAR_TRADES", round(BEAR_TRADES, 2), "LATERAL_TRADES", round(PROVA['Trades'].sum()-BULL_TRADES-BEAR_TRADES,2))
print('Start_Date', Start_Date, 'End_Date', End_Date, 'Number of Years :', round(difference_in_years, 2))
print('Average Yearly Return', round(Average_Yearly_Return,2))
print('Number of Trades', round(Number_of_Trades,2))
print('Number of Negative_Trades', round(Number_of_Negative_Trades,2))
print('Hit Ratio', round(Hit_Ratio,2))
print('Frequency of_bets', round(Frequency_of_bets,2))
print('Average Trade_Return', round(Average_Trade_Return,4))
print('Average Return from hits', round(Average_Return_from_hits,4))
print('Average Return from misses', round(Average_Return_from_misses,4))
print('Trades Variance', round(Trades_Variance,4))
print('Trades Standard Deviation', round(Trades_Standard_Deviation,4))
print('Trades Skwness', round(Trades_Skwness,2))
print('Trades Kurtosis', round(Trades_Kurtosis,2))

print('////////////////////////////////////////////////////////////////////////////////////////////////////////////')
print('LONG SUMMARY')
print('////////////////////////////////////////////////////////////////////////////////////////////////////////////')

print('Cumulated Return Long', round(Cumulated_Return_Long,2))
print('Number of Trades Long', round(Number_of_Trades_Long,2))
print('LONGS/TOTAL', round(Ratio_of_longs,2))
print('Number of Winning Trades Long', round(Number_of_Positive_Trades_Long,2))
print('Hit Ratio Long', round(Hit_Ratio_Long,2))
print('Average Trade Return Long', round(Average_Trade_Return_Long,4))
print('Average Return from hits Long', round(Average_Return_from_hits_Long,4))
print('Average Return from misses Long', round(Average_Return_from_misses_Long,4))
print('Trades Variance Long', round(Trades_Variance_Long,4))
print('Trades Standard Deviation Long', round(Trades_Standard_Deviation_Long,4))
print('Trades Skwness Long', round(Trades_Skwness_Long,2))
print('Trades Kurtosis Long', round(Trades_Kurtosis_Long,2))
print('Average Yearly Return Long', round(Average_Yearly_Return_Long,2))

print('////////////////////////////////////////////////////////////////////////////////////////////////////////////')
print('SHORT SUMMARY')
print('////////////////////////////////////////////////////////////////////////////////////////////////////////////')

print('Cumulated Return Short', round(Cumulated_Return_Short,2))
print('Number of Trades Short', round(Number_of_Trades_Short,2))
print('Number of Winning Trades Short', round(Number_of_Positive_Trades_Short,2))
print('Hit Ratio Short', round(Hit_Ratio_Short,2))
print('Average Trade Return Short', round(Average_Trade_Return_Short,4))
print('Average Return from hits Short', round(Average_Return_from_hits_Short,4))
print('Average Return from misses Short', round(Average_Return_from_misses_Short,4))
print('Trades Variance Short', round(Trades_Variance_Short,4))
print('Trades Standard Deviation Short', round(Trades_Standard_Deviation_Short,4))
print('Trades Skwness Short', round(Trades_Skwness_Short,2))
print('Trades Kurtosis Short', round(Trades_Kurtosis_Short,2))
print('Average Yearly Return Short', round(Average_Yearly_Return_Short,2))

#Using mlfinlab package function for detailed concentration output
STRATEGY["date_time"] = STRATEGY["Date_x"] +' '+ STRATEGY["Time_x"]
STRATEGY.index = pd.to_datetime(STRATEGY['date_time'])
PROVA.index = pd.to_datetime(STRATEGY['date_time'])
STRATEGY = STRATEGY.drop('date_time', axis=1)

A=PROVA['Trades']!=0
TRADES2=PROVA[A]
logret_series = np.exp(TRADES2['Trades'])
logret_series=np.log(logret_series)

pos_concentr, neg_concentr, hourly_concentr = all_bets_concentration(logret_series, frequency='M')
print('HHI index on positive log returns is' , round(pos_concentr,4))
print('HHI index on negative log returns is' , round(neg_concentr,4))
print('HHI index on log returns divided into hourly bins is' , round(hourly_concentr,4))

#Getting series of prices to represent the value of one long portfolio
TRADES2['perc_ret'] = TRADES2.Trades.cumsum()
#TRADES2['perc_ret'].plot()
TRADES2['AUM']=100*np.exp(TRADES2['perc_ret'])
#TRADES2['AUM'].plot()
#plt.show()
#Using mlfinlab package function to get drawdowns and time under water series
drawdown, tuw = drawdown_and_time_under_water(TRADES2['AUM'], dollars = False)
drawdown_dollars, _ = drawdown_and_time_under_water(TRADES2['AUM'], dollars = True)
print('The 99th percentile of Drawdown is', round(drawdown.quantile(.99),2))
print('The 99th percentile of Drawdown in dollars is', round(drawdown_dollars.quantile(.99),2))
print('The 99th percentile of Time under water', round(tuw.quantile(.99),2))

#Using simple formula for annual return calculation
days_observed = (TRADES2['AUM'].index[-1] - TRADES2['AUM'].index[0]) / np.timedelta64(1, 'D')
cumulated_return = TRADES2['AUM'][-1]/TRADES2['AUM'][0]

#Using 365 days instead of 252 as days observed are calculated as calendar
#days between the first observation and the last
annual_return = (cumulated_return)**(365/days_observed) - 1
print('Annualized average return from the portfolio is' , round(annual_return,2))

#Also looking at returns grouped by days
logret_by_days = logret_series.groupby(pd.Grouper(freq='D')).sum()
logret_by_days = logret_by_days[logret_by_days!=0]

print('Average log return from positive bars grouped by days is' ,
      round(logret_by_days[logret_by_days>0].mean(),2), 'and counter is',
      round(logret_by_days[logret_by_days>0].count(),2))
print('Average log return from positive bars is' , round(logret_series[logret_series>0].mean(),4),
      'and counter is', logret_series[logret_series>0].count())
print('Average log return from negative bars is' , round(logret_series[logret_series<0].mean(),4),
     'and counter is', round(logret_series[logret_series<0].count(),2))

#Uning mlfinlab package function to get SR
annualized_sr = sharpe_ratio(logret_by_days, entries_per_year=252, risk_free_rate=0)
print('Annualized Sharpe Ratio is' , round(annualized_sr,2))

#Stating the risk-free ratio and trading days per year
risk_free_ratio = 0.00
trading_days = 252

#Calculating excess returns above the risk-free ratio
#This means subtracting the benchmark from daily returns

#Daily returns adjusted for taking compounding effect into account
daily_risk_free_ratio = (1 + risk_free_ratio)**(1/trading_days) - 1
log_daily_risk_free_ratio = np.log(1 + daily_risk_free_ratio)

#Using mlfinlab package function to get Information ratio
information_ratio = information_ratio(logret_by_days,
                       log_daily_risk_free_ratio, entries_per_year=trading_days)
print('Information ratio (with yearly risk-free rate assumed to be 1.5%) is' , round(information_ratio,2))

#Using mlfinlab package function to get PSR
probabilistic_sr = probabilistic_sharpe_ratio(observed_sr=annualized_sr,
                                                                     benchmark_sr=0.4,
                                                                     number_of_returns=days_observed,
                                                                     skewness_of_returns=logret_by_days.skew(),
                                                                     kurtosis_of_returns=logret_by_days.kurt())
print('Probabilistic Sharpe Ratio with benchmark SR of 0.8 is' , round(probabilistic_sr,2))

#Using mlfinlab package function to get DSR. Passing standard deviation of trails and
#number of trails as a parameter, also flag estimates_param.
deflated_sr = deflated_sharpe_ratio(observed_sr=annualized_sr,
                                                           sr_estimates=[0.005**(1/2), 5],
                                                           number_of_returns=days_observed,
                                                           skewness_of_returns=logret_by_days.skew(),
                                                           kurtosis_of_returns=logret_by_days.kurt(),
                                                           estimates_param=True)
print('Deflated Sharpe Ratio with 10 trails and 0.5 variance is' , round(deflated_sr,2))

benchmark_sr_dsr = deflated_sharpe_ratio(observed_sr=annualized_sr,
                                                                sr_estimates=[0.005**(1/2), 5],
                                                                number_of_returns=days_observed,
                                                                skewness_of_returns=logret_by_days.skew(),
                                                                kurtosis_of_returns=logret_by_days.kurt(),
                                                                estimates_param=True, benchmark_out=True)

print('Benchmark Sharpe ratio used in DSR is' , round(benchmark_sr_dsr,2))
benchmark_sr_dsr_adj = deflated_sharpe_ratio(observed_sr=annualized_sr,
                                                                    sr_estimates=[0.05**(1/2), 3],
                                                                    number_of_returns=days_observed,
                                                                    skewness_of_returns=logret_by_days.skew(),
                                                                    kurtosis_of_returns=logret_by_days.kurt(),
                                                                    estimates_param=True, benchmark_out=True)
print('Benchmark Sharpe ratio if number of trails is decreased to 3 is' , round(benchmark_sr_dsr_adj,2))

AUTOCORRELATION=estimated_autocorrelation(logret_by_days)

# Creating an object and specifying the desired level of simulations to do
# for Haircut Sharpe Ratios and Profit Hurdle in the Holm and BHY methods.
backtesting = CampbellBacktesting(simulations=5000)

# Calculating the adjusted Sharpe ratios and the haircuts.
haircuts = backtesting.haircut_sharpe_ratios(sampling_frequency='D', num_obs=days_observed, sharpe_ratio=annualized_sr,
                                             annualized=True, autocorr_adjusted=False, rho_a=AUTOCORRELATION[1],
                                             num_mult_test=3, rho=0.95)

# Adjusted Sharpe ratios by the method used.
print('The adjusted Sharpe ratio using the Bonferroni method is', round(haircuts[1][0],4))
print('The adjusted Sharpe ratio using the Holm method is', round(haircuts[1][1],4))
print('The adjusted Sharpe ratio using the BHY method is', round(haircuts[1][2],4))
print('The average adjusted Sharpe ratio of the methods is', round(haircuts[1][3],4))
# Sharpe ratio haircuts.
print('The Sharpe ratio haircut using the Bonferroni method is', round(haircuts[2][0],2))
print('The Sharpe ratio haircut using the Holm method is', round(haircuts[2][1],2))
print('The Sharpe ratio haircut using the BHY method is', round(haircuts[2][2],2))
print('The average Sharpe ratio haircut of the methods is', round(haircuts[2][3],2))

# Calculating the Minimum Average Monthly Returns.
monthly_ret = backtesting.profit_hurdle(num_mult_test=30, num_obs=1200, alpha_sig=0.05,
                                        vol_anu=0.2, rho=0.7)

# Minimum Average Monthly Returns by the method used.
print('Required Minimum Average Monthly Returns using the Bonferroni method is', monthly_ret[0])
print('Required Minimum Average Monthly Returns using the Holm method is', monthly_ret[1])
print('Required Minimum Average Monthly Returns using the BHY method is', monthly_ret[2])
print('Required Minimum Average Monthly Returns using the average of the methods is', monthly_ret[3])

#################################################################################################################
# Economic Analysis
###################################################################################################################

logret_by_months= logret_series.groupby(pd.Grouper(freq='M')).sum()
logret_by_months = logret_by_months[logret_by_months!=0]
INDEX_STRAT=logret_by_months.index.tolist()
LIST=logret_by_months.tolist()
df=pd.DataFrame(INDEX_STRAT, columns=['Date'])
df['Trades']=LIST
df.index=INDEX_STRAT
Strat_Trades_Date=df['Date'].iloc[0]

ECONOMICS=ECONOMICS[1:]
Starting=ECONOMICS.index[ECONOMICS['Frequency: Monthly'] ==Strat_Trades_Date ].tolist()
ECONOMICS['Frequency: Monthly']=pd.to_datetime(ECONOMICS['Frequency: Monthly'])
ECONOMICS.index=ECONOMICS['Frequency: Monthly']
mask = (ECONOMICS['Frequency: Monthly'] > '2012-02-01') & (ECONOMICS['Frequency: Monthly'] < '2020-06-01')
ECONOMICS2=ECONOMICS.loc[mask]
df.reset_index(drop=True, inplace=True)
ECONOMICS2.reset_index(drop=True, inplace=True)
dfn = pd.concat( [df, ECONOMICS2], axis=1)

# create the regressions
Y = pd.DataFrame()
X = pd.DataFrame()
Y = dfn['Trades'].iloc[1:]

X = dfn.iloc[:,2:]
X=X.iloc[:,1:-1]
X_Ret=pd.DataFrame()
X_Ret = X.pct_change()
X_Ret=X_Ret.iloc[1:, :]
X_Ret['Risk_Spread']=X_Ret.iloc[:,3]-X_Ret.iloc[:,5]

# Kitchen ink model preliminary
X_Ret = sm.add_constant(X_Ret) ## let's add an intercept (beta_0) to our model
model = sm.OLS(Y, X_Ret).fit()
predictions = model.predict(X_Ret)
print(model.summary())

# Univariate regressions returns
Y = pd.DataFrame()
X = pd.DataFrame()
Y = dfn['Trades'].iloc[1:]

X = dfn.iloc[:, 2:]
X = X.iloc[:, 1:-1]
X_Ret = pd.DataFrame()
X_Ret = X.pct_change()
X_Ret = X_Ret.iloc[1:, :]
X_Ret['Risk_Spread'] = X_Ret.iloc[:, 3] - X_Ret.iloc[:, 5]

Uni_Regress_Betas = []
Uni_Regress_pval = []
X_Ret = sm.add_constant(X_Ret)  ## let's add an intercept (beta_0) to our model

for i in range(1, X_Ret.shape[1]):
    model = sm.OLS(Y, X_Ret.iloc[:, [0, i]]).fit()
    model.summary()
    SUMMAR = pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0]
    Uni_Regress_Int = SUMMAR['coef'].values[0]
    Uni_Regress_Beta = SUMMAR['coef'].values[1]
    Uni_pval_Int = SUMMAR['P>|t|'].values[0]
    Uni_pval_Beta = SUMMAR['P>|t|'].values[1]
    SUMMAR2 = pd.read_html(model.summary().tables[0].as_html(), header=0, index_col=0)[0]
    Uni_R2 = list(SUMMAR2)[2]
    print('Univariate regression, predictor :', list(ECONOMICS2)[i])
    print(" __R2__ :", Uni_R2)
    print('Beta intercept', Uni_Regress_Int, 'p-val', Uni_pval_Int)
    print('Beta independent', Uni_Regress_Beta, 'p-val', Uni_pval_Beta)

# Lasso
normalized_Y=(Y-Y.mean())/Y.std()
X = dfn.iloc[:,2:]
X=X.iloc[:,1:-1]
X_Ret=pd.DataFrame()
X_Ret = X.pct_change()
X_Ret=X_Ret.iloc[1:, :]
X_Ret['Risk_Spread']=X_Ret.iloc[:,3]-X_Ret.iloc[:,5]
normalized_X_Ret=(X_Ret-X_Ret.mean())/X_Ret.std()

lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = False)
lassocv.fit(normalized_X_Ret, normalized_Y)
lasso = Lasso(max_iter = 10000, normalize = False)
lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(normalized_X_Ret, normalized_Y)
mean_squared_error(normalized_Y, lasso.predict(normalized_X_Ret))
# Some of the coefficients are now reduced to exactly zero.
pd.Series(lasso.coef_, index=normalized_X_Ret.columns)

# Univariate regressions levels
Y = pd.DataFrame()
Y = dfn['Trades'].iloc[1:]
Y = list(Y)

X = pd.DataFrame()
X = dfn.iloc[1:, 3:]
X['Risk_Spread'] = X.iloc[:, 3] - X.iloc[:, 5]

Uni_Regress_Betas = []
Uni_Regress_pval = []
X = sm.add_constant(X)  ## let's add an intercept (beta_0) to our model

for i in range(1, X.shape[1]):
    model = sm.OLS(Y, X.iloc[:, [0, i]].astype(float)).fit()
    model.summary()
    SUMMAR = pd.read_html(model.summary().tables[1].as_html(), header=0, index_col=0)[0]
    Uni_Regress_Int = SUMMAR['coef'].values[0]
    Uni_Regress_Beta = SUMMAR['coef'].values[1]
    Uni_pval_Int = SUMMAR['P>|t|'].values[0]
    Uni_pval_Beta = SUMMAR['P>|t|'].values[1]
    SUMMAR2 = pd.read_html(model.summary().tables[0].as_html(), header=0, index_col=0)[0]
    Uni_R2 = list(SUMMAR2)[2]
    print('Univariate regression, predictor :', list(X)[i])
    print(" __R2__ :", Uni_R2)
    print('Beta intercept', Uni_Regress_Int, 'p-val', Uni_pval_Int)
    print('Beta independent', Uni_Regress_Beta, 'p-val', Uni_pval_Beta)

# Multivariate
# create the regressions
Y = pd.DataFrame()
X = pd.DataFrame()
Y = dfn['Trades'].iloc[1:]

X = dfn.iloc[:,2:]
X=X.iloc[1:,1:-1]
X['Risk_Spread']=X.iloc[:,3]-X.iloc[:,5]

# Kitchen ink model preliminary
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model
model = sm.OLS(Y, X.astype(float)).fit()
print(model.summary())

# Lasso
normalized_Y=(Y-Y.mean())/Y.std()
X = dfn.iloc[:,2:]
X = X.iloc[1:,1:-1]

X_Ret=pd.DataFrame()
X_Ret = X.pct_change()
X_Ret=X_Ret.iloc[1:, :]
X_Ret['Risk_Spread']=X.iloc[:,3]-X.iloc[:,5]

lassocv = LassoCV(alphas = None, cv = 10, max_iter = 100000, normalize = True)
lassocv.fit(X, Y)
lasso = Lasso(max_iter = 10000, normalize = True)
lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X, Y)
mean_squared_error(normalized_Y, lasso.predict(X))

# Some of the coefficients are now reduced to exactly zero.
print(pd.Series(lasso.coef_, index=X.columns))

# Best worst Level
# Best worst Level
from scipy import stats

n_relevant = 10
Y.index = dfn['Date'].iloc[1:]
Best_Strat_Ret = Y.nlargest(n_relevant)
Worst_Strat_Ret = Y.nsmallest(n_relevant)

# Analysis of the Worst
dfn2 = dfn
dfn2.index = dfn2['Date']
Worst_Strat_Ret = pd.concat([Worst_Strat_Ret, dfn2], axis=1)
Worst_Strat_Ret["Spread"] = X.iloc[:, 3] - X.iloc[:, 5]
Perc = pd.DataFrame(index=range(n_relevant), columns=range(Worst_Strat_Ret.shape[1]))

m = 0

for i in range(0, Worst_Strat_Ret.shape[0]):

    if pd.notnull(Worst_Strat_Ret.iloc[i, 0]):

        for j in range(4, Worst_Strat_Ret.shape[1]):
            Perc.iloc[m, j] = stats.percentileofscore(Worst_Strat_Ret.iloc[:, j], Worst_Strat_Ret.iloc[i, j])
            # print('Percentile', j, round(Perc.iloc[m, j],2))

        m = m + 1

Perc_Median_Worst_Lev = round(Perc.iloc[:, 4:].median(), 2)
Ls = list(Worst_Strat_Ret)
Ls = Ls[4:]

# Analysis of the Bests
dfn2 = dfn
dfn2.index = dfn2['Date']
Best_Strat_Ret = pd.concat([Best_Strat_Ret, dfn2], axis=1)
Best_Strat_Ret["Spread"] = X.iloc[:, 3] - X.iloc[:, 5]

Perc = pd.DataFrame(index=range(n_relevant), columns=range(Best_Strat_Ret.shape[1]))
m = 0

for i in range(0, Best_Strat_Ret.shape[0]):

    # print("i", i)

    if pd.notnull(Best_Strat_Ret.iloc[i, 0]):

        for j in range(4, Best_Strat_Ret.shape[1]):
            Perc.iloc[m, j] = stats.percentileofscore(Best_Strat_Ret.iloc[:, j], Best_Strat_Ret.iloc[i, j])

        m = m + 1

Perc_Median_Best_Lev = round(Perc.iloc[:, 4:].median(), 2)

# Best worst percentages
n_relevant = 10
n_pred = 16
Y.index = dfn['Date'].iloc[1:]
Best_Strat_Ret = Y.nlargest(n_relevant)
Worst_Strat_Ret = Y.nsmallest(n_relevant)

# Worst
Worst_Strat_Ret = pd.concat([Worst_Strat_Ret, X_Ret], axis=1)
Perc = pd.DataFrame(index=range(n_relevant), columns=range(Worst_Strat_Ret.shape[1]))
m = 0

for i in range(0, Worst_Strat_Ret.shape[0]):

    # print("i", i)

    if pd.notnull(Worst_Strat_Ret.iloc[i, 0]):

        for j in range(4, Worst_Strat_Ret.shape[1]):
            Perc.iloc[m, j] = stats.percentileofscore(Worst_Strat_Ret.iloc[:, j], Worst_Strat_Ret.iloc[i, j])
            # print('Percentile', round(Perc.iloc[m, j],2))

        m = m + 1

Perc_Median_Worst_Ret = round(Perc.iloc[:, 4:].median(), 2)
Ls = list(Worst_Strat_Ret)
Ls = Ls[4:]

# Best
Best_Strat_Ret = pd.concat([Best_Strat_Ret, X_Ret], axis=1)
Perc = pd.DataFrame(index=range(n_relevant), columns=range(Best_Strat_Ret.shape[1]))
m = 0

for i in range(0, Best_Strat_Ret.shape[0]):

    # print("i", i)

    if pd.notnull(Best_Strat_Ret.iloc[i, 0]):

        for j in range(4, Best_Strat_Ret.shape[1]):
            Perc.iloc[m, j] = stats.percentileofscore(Best_Strat_Ret.iloc[:, j], Best_Strat_Ret.iloc[i, j])
            # print('Percentile', round(Perc.iloc[m, j],2))

        m = m + 1

Perc_Median_Best_Ret = round(Perc.iloc[:, 4:].median(), 2)
Ls = list(Worst_Strat_Ret)
Ls = Ls[4:]

# Print results
SUMMARY=pd.DataFrame(index=range(len(Ls)),columns=range(5) )
SUMMARY.columns = ["Indicator", "Worst Lev", "Worst Ret", "Best Lev", "Best Ret"]
SUMMARY.iloc[:,0]=Ls

for i in range(0, len(Ls) ):

    SUMMARY.iloc[i,1]=Perc_Median_Worst_Lev.iloc[i]
    SUMMARY.iloc[i,2]=Perc_Median_Worst_Ret.iloc[i]
    SUMMARY.iloc[i,3]=Perc_Median_Best_Lev.iloc[i]
    SUMMARY.iloc[i,4]=Perc_Median_Best_Ret.iloc[i]

print(SUMMARY)

# Current Level percentiles available
# Current Level percentiles available
ECONOMICS = pd.read_excel(r'/home/pisati/Desktop/Backtesting_Night/ECONOMICS.xlsx')
ECONOMICS = ECONOMICS[1:]
Last = ECONOMICS.iloc[-1, :]
Actual_Lev = pd.DataFrame(index=range(ECONOMICS.shape[1] - 1), columns=range(1))

for i in range(1, ECONOMICS.shape[1]):

    if pd.notnull(ECONOMICS.iloc[-1, i]):

        Actual_Lev.loc[i, 0] = round(stats.percentileofscore(ECONOMICS.iloc[:, i], ECONOMICS.iloc[-1, i]), 2)

    else:

        Actual_Lev.loc[i, 0] = "NA"

Actual_Lev = Actual_Lev.rename(columns={0: 'Last_Level'})

# Current return percentiles available
ECONOMICS = pd.read_excel(r'/home/pisati/Desktop/Backtesting_Night/ECONOMICS.xlsx')
ECONOMICS=ECONOMICS[1:]
Starting=ECONOMICS.index[ECONOMICS['Frequency: Monthly'] ==Strat_Trades_Date ].tolist()
ECONOMICS['Frequency: Monthly']=pd.to_datetime(ECONOMICS['Frequency: Monthly'])
ECONOMICS.index=ECONOMICS['Frequency: Monthly']
mask = (ECONOMICS['Frequency: Monthly'] > '2012-02-01')
ECONOMICS2=ECONOMICS.loc[mask]
df.reset_index(drop=True, inplace=True)
ECONOMICS2.reset_index(drop=True, inplace=True)
dfn = pd.concat( [df, ECONOMICS2], axis=1)
X = dfn.iloc[:, 3:]
X_Ret = pd.DataFrame()
X_Ret = X.pct_change()
X_Ret = X_Ret.iloc[1:, :]
X_Ret['Unnamed: 16'] = X_Ret.iloc[:, 3] - X_Ret.iloc[:, 5]
Actual_Perc = pd.DataFrame(index=range(X.shape[1]), columns=range(1))

for i in range(1, X.shape[1]):

    if pd.notnull(X.iloc[-1, i]):

        Actual_Perc.loc[i, 0] = round(stats.percentileofscore(X.iloc[:, i], X.iloc[-1, i]), 2)

    else:

        Actual_Perc.loc[i, 0] = "NA"

Actual_Perc = Actual_Perc.rename(columns={0: 'Last_Perc'})
#
SUMMARY2 = pd.concat([Actual_Lev, Actual_Perc], axis=1, ignore_index=True)
SUMMARY2 = SUMMARY2.rename(columns={0: 'Last_Level', 1: 'Last_Perc'})
print(SUMMARY2)

##################################################################################################################
# Grid unidimensional optimization
##################################################################################################################

RESULTS = np.zeros((ts.shape[0], 9))
PROVA = []
OPTIM = []
OPTIM = pd.DataFrame(np.zeros((25, 3)))
cost = 0.0004
j = 0

for threshold_Over in np.arange(-0.01, 0.015, 0.001):
    PROVA = []
    RETURNS = Times_analyse_njit(dt, ts, ret, over, vix, Sig_8, Sig_7, Enter, RESULTS, threshold_Over, threshold_VIX,
                                 cost)
    PROVA = pd.DataFrame(RETURNS, columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades', 'Long_Trades',
                                           'Short_Trades'])
    Cumulated_return_Positive_Trades = PROVA['Trades'].where(PROVA['Trades'] > 0).sum(0)
    Cumulated_return_Neagative_Trades = PROVA['Trades'].where(PROVA['Trades'] < 0).sum(0)
    OPTIM.iloc[j, 0] = PROVA['Trades'].sum()
    OPTIM.iloc[j, 0] = Cumulated_return_Positive_Trades / abs(Cumulated_return_Neagative_Trades)
    OPTIM.iloc[j, 1] = threshold_Over
    OPTIM.iloc[j, 2] = PROVA['Trades'].sum()
    RESULTS = np.zeros((ts.shape[0], 9))
    j = j + 1

# Resulsts Profitability ratio
Best_Perf_PR = OPTIM.iloc[OPTIM[0].argmax(), 0]
Best_Over_PF = OPTIM.iloc[OPTIM[0].argmax(), 1]
Cum_Ret_PF = OPTIM.iloc[OPTIM[0].argmax(), 2]

# Resulsts Cum_Ret
Best_Perf_CR = OPTIM.iloc[OPTIM[2].argmax(), 0]
Best_Over_CR = OPTIM.iloc[OPTIM[2].argmax(), 1]
Cum_Ret_CR = OPTIM.iloc[OPTIM[2].argmax(), 2]

# Print Results
print("/////////////////////////////////////////////////////////////////////////////////////////////////")
print('Best_Perf_PR', round(Best_Perf_PR, 3), 'Best_Param_PF', round(Best_Over_PF, 4), 'Cum_Ret_PF',
      round(Cum_Ret_PF, 3))
print('Best_Perf_CR', round(Best_Perf_CR, 3), 'Best_Param_CR', round(Best_Over_CR, 4), 'Cum_Ret_CR',
      round(Cum_Ret_CR, 3))
print("/////////////////////////////////////////////////////////////////////////////////////////////////")

####################################################################################################################
# Grid 2d optimization CR
####################################################################################################################

RESULTS=np.zeros((ts.shape[0], 9))
PROVA=[]
OPTIM=[]
OPTIM=pd.DataFrame(np.zeros((625, 4)))
j=0

for threshold_VIX in np.arange(25, 50, 1):

    for threshold_Over in np.arange(-0.01, 0.015, 0.001):

        RETURNS=Times_analyse_njit(dt, ts, ret, over, vix, Sig_8, Sig_7, Enter, RESULTS, threshold_Over, threshold_VIX, cost)
        PROVA = pd.DataFrame(RETURNS, columns = ['Date','Time','Returns', 'Over','POS','CUM_RET','Trades','Long_Trades', 'Short_Trades'])
        OPTIM.iloc[j,0]=PROVA['Trades'].sum()
        OPTIM.iloc[j,1]=threshold_VIX
        OPTIM.iloc[j,2]=threshold_Over
        Cumulated_return_Positive_Trades=PROVA['Trades'].where(PROVA['Trades'] > 0).sum(0)
        Cumulated_return_Neagative_Trades=PROVA['Trades'].where(PROVA['Trades'] < 0).sum(0)
        OPTIM.iloc[j,3]=Cumulated_return_Positive_Trades/abs(Cumulated_return_Neagative_Trades)
        RESULTS=np.zeros((ts.shape[0], 9))
        PROVA=[]
        j=j+1

Best_Perf_CR=OPTIM.iloc[OPTIM[0].argmax(),0]
Best_VIX_CR=OPTIM.iloc[OPTIM[0].argmax(),1]
Best_over_CR=OPTIM.iloc[OPTIM[0].argmax(),2]
PR=OPTIM.iloc[OPTIM[0].argmax(),3]

Best_Perf_PR=OPTIM.iloc[OPTIM[3].argmax(),0]
Best_VIX_PR=OPTIM.iloc[OPTIM[3].argmax(),1]
Best_over_PR=OPTIM.iloc[OPTIM[3].argmax(),2]
CR=OPTIM.iloc[OPTIM[3].argmax(),3]

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

print('Best Performance achievable in-sample')
print("/////////////////////////////////////////////////////////////////////////////////////////////////")
print('Best_Perf_CR', round(Best_Perf_CR,3), 'Perf_PR', round(PR,3), namestr(threshold_VIX, globals()), round(Best_VIX_CR,3), namestr(threshold_Over, globals()), round(Best_over_CR,3))
print('Best_Perf_PR', round(CR,3), 'Perf_CR', round(Best_Perf_PR,3), namestr(threshold_VIX, globals()), round(Best_VIX_PR,3), namestr(threshold_Over, globals()), round(Best_over_PR,3))
print("/////////////////////////////////////////////////////////////////////////////////////////////////")

########################################################################################################################################
# OOS Permutation test
#########################################################################################################################################

RESULTS = np.zeros((ts.shape[0], 9))
PROVA = []
OPTIM = []
n_permut = 10
OPTIM = pd.DataFrame(np.zeros((625, 3)))
BASE = round((STRATEGY.shape[0] / 2))
SET = round((STRATEGY.shape[0] / 10))
SIMUL = pd.DataFrame(np.zeros((n_permut, 1)))
cost = 0.000
j = 0

for threshold_VIX in np.arange(25, 50, 1):

    for threshold_Over in np.arange(-0.01, 0.015, 0.001):
        RETURNS = Times_analyse_njit(dt[0:BASE], ts[0:BASE], ret[0:BASE], over[0:BASE], vix[0:BASE], Sig_8, Sig_7,
                                     Enter, RESULTS, threshold_Over, threshold_VIX, cost)
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
print(Best_Perf, Best_VIX, Best_over)

# OOS Original
index = STRATEGY.to_numpy()
TIME_2 = pd.to_datetime(index[BASE:, 1])
ts2 = (TIME_2 - np.datetime64('1990-01-01T00:00:00Z')) / np.timedelta64(1, 'm')
ts2 = ts2.to_numpy()
ts2 = np.round(ts2, 0)

DATE_2 = pd.to_datetime(index[BASE:, 0])
dt2 = (DATE_2 - np.datetime64('1990-01-01T00:00:00Z')) / np.timedelta64(1, 'm')
dt2 = dt2.to_numpy()
dt2 = np.round(dt2, 0)

RET_2 = STRATEGY['log_ret_x'].iloc[BASE:].fillna(0)
ret2 = RET_2.to_numpy()

OVER_2 = STRATEGY['Over'].iloc[BASE:]
OVER_2 = OVER_2.fillna(0)
over2 = OVER_2.to_numpy()

VIX_2 = STRATEGY['Close'].iloc[BASE:]
VIX_2 = VIX_2.fillna(0).replace(to_replace=0, method='ffill')
vix2 = VIX_2.to_numpy()

RESULTS = np.zeros((ts.shape[0], 9))
RETURNS_TS = Times_analyse_njit(dt2, ts2, ret2, over2, vix2, Sig_8, Sig_7, Enter, RESULTS, Best_over, Best_VIX, cost)
PROVA = pd.DataFrame(RETURNS_TS, columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades', 'Long_Trades',
                                          'Short_Trades'])
PROVA['DATE_F'] = pd.DataFrame(RETURNS_TS[:, 0] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'),
                               columns=['DATE_F'])
PROVA['TIME_F'] = pd.DataFrame(RETURNS_TS[:, 1] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'),
                               columns=['TIME_F'])
Start_Date = PROVA['DATE_F'].iloc[1]
End_Date = PROVA['DATE_F'].iloc[SET - 1]
ORIGINAL = PROVA['Trades'].sum()
PROVA = []
print("ORIGINAL", ORIGINAL)

Permutation_Matr_2 = []
Permutation_Matr_2 = pd.DataFrame(np.zeros((len(VIX_2), n_permut)))

for i in np.arange(0, n_permut, 1):
    PERM = np.random.permutation(ret2)
    Permutation_Matr_2.iloc[:, i] = PERM

oos = Permutation_Matr_2.fillna(0).to_numpy()

for i in range(0, n_permut, 1):
    RETURNS_TS = Times_analyse_njit(dt2, ts2, oos[:, i], over2, vix2, Sig_8, Sig_7, Enter, RESULTS, Best_over, Best_VIX,
                                    cost)
    PROVA = pd.DataFrame(RETURNS_TS,
                         columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades', 'Long_Trades',
                                  'Short_Trades'])
    PROVA['DATE_F'] = pd.DataFrame(RETURNS_TS[:, 0] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'),
                                   columns=['DATE_F'])
    PROVA['TIME_F'] = pd.DataFrame(RETURNS_TS[:, 1] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'),
                                   columns=['TIME_F'])
    Start_Date = PROVA['DATE_F'].iloc[1]
    End_Date = PROVA['DATE_F'].iloc[SET - 1]
    SIMUL.iloc[i] = PROVA['Trades'].sum()
    # print(PROVA['Trades'].sum())

Rand_Fitting = SIMUL.mean()
Skill = ORIGINAL - Rand_Fitting
print("Rand_Fitting", Rand_Fitting, "ORIGINAL ", ORIGINAL, "Skill", Skill)

##############################################################################################################################
# In sample Training Bias
##############################################################################################################################
# Permutations

n_permut = 10
RETURNS = STRATEGY['log_ret_x'].dropna().to_numpy()
Permutation_Matr = []
Permutation_Matr = pd.DataFrame(np.zeros((len(RETURNS), n_permut)))
IN_SAMP_PERM = pd.DataFrame(index=range(n_permut), columns=range(2))

for i in np.arange(0, n_permut, 1):
    PERM = np.random.permutation(RETURNS)
    Permutation_Matr.iloc[:, i] = PERM

# Create realized volatility
Squared_Ret = Permutation_Matr ** 2
Realized_Variance = Squared_Ret.rolling(48).sum().dropna()
Realized_Vol = Realized_Variance ** (1 / 2)

pm = Permutation_Matr.fillna(0).to_numpy()
rv = Realized_Vol.fillna(0).to_numpy()

# Grid 2d optimization in sample
RESULTS=np.zeros((ts.shape[0], 9))
PROVA=[]
OPTIM=[]
OPTIM=pd.DataFrame(np.zeros((625, 3)))
j=0

for threshold_VIX in np.arange(25, 50, 1):

    for threshold_Over in np.arange(-0.01, 0.015, 0.001):

        RETURNS = Times_analyse_njit(dt, ts, ret, over, vix, Sig_8, Sig_7, Enter, RESULTS, threshold_Over, threshold_VIX, cost)
        PROVA = pd.DataFrame(RETURNS, columns = ['Date','Time','Returns', 'Over','POS','CUM_RET','Trades','Long_Trades', 'Short_Trades'])
        OPTIM.iloc[j,0] = PROVA['Trades'].sum()
        OPTIM.iloc[j,1] = threshold_VIX
        OPTIM.iloc[j,2] = threshold_Over
        RESULTS=np.zeros((ts.shape[0], 9))
        PROVA = []
        j=j+1

Best_Perf = OPTIM.iloc[OPTIM[0].argmax(), 0]
Best_VIX = OPTIM.iloc[OPTIM[0].argmax(), 1]
Best_over = OPTIM.iloc[OPTIM[0].argmax(), 2]
print("In-Sample Best Performance", Best_Perf, Best_VIX, Best_over)

# Optimizations of the permuted returns
Perm_In_Sample_Best = pd.DataFrame(np.zeros((n_permut, 1)))
m = 0

for i in np.arange(0, n_permut, 1):

    PROVA = []
    OPTIM = []
    OPTIM = pd.DataFrame(np.zeros((625, 3)))
    j = 0

    for threshold_VIX in np.arange(25, 50, 1):

        for threshold_Over in np.arange(-0.01, 0.015, 0.001):
            RETURNS = Times_analyse_njit(dt, ts, pm[:, i], over, vix, Sig_8, Sig_7, Enter, RESULTS, threshold_Over,
                                         threshold_VIX, cost)
            PROVA = pd.DataFrame(RETURNS,
                                 columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades', 'Long_Trades',
                                          'Short_Trades'])
            PROVA['DATE_F'] = pd.DataFrame(
                RETURNS[:, 0] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'), columns=['DATE_F'])
            PROVA['TIME_F'] = pd.DataFrame(
                RETURNS[:, 1] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'), columns=['TIME_F'])
            # print(PROVA['Trades'].sum())
            OPTIM.iloc[j, 0] = PROVA['Trades'].sum()
            OPTIM.iloc[j, 1] = threshold_VIX
            OPTIM.iloc[j, 2] = threshold_Over
            RESULTS = np.zeros((ts.shape[0], 9))
            PROVA = []
            j = j + 1

    Best_Perf_CR = OPTIM.iloc[OPTIM[0].argmax(), 0]
    Best_VIX_CR = OPTIM.iloc[OPTIM[0].argmax(), 1]
    Best_over_CR = OPTIM.iloc[OPTIM[0].argmax(), 2]
    Perm_In_Sample_Best.iloc[m, 0] = Best_Perf_CR
    #print(Perm_In_Sample_Best)
    m = m + 1

Rand_Fitting=Perm_In_Sample_Best.mean()
Skill = Best_Perf-Rand_Fitting
print("Rand_Fitting", Rand_Fitting, "Best_Perf", Best_Perf, "Skill", Skill)

#############################################################################################################################
# Walkforward Expanding training window with grid optim
#############################################################################################################################

BASE = round((STRATEGY.shape[0] / 5))
SET = round((STRATEGY.shape[0] / 10))
OOS = pd.DataFrame(np.zeros((5, 3)))
CUM_RET_OOS = 0
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
            RETURNS = Times_analyse_njit(dt[0:BASE + SET * m], ts[0:BASE + SET * m], ret[0:BASE + SET * m],
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
    RETURNS = Times_analyse_njit(dt[BASE + SET * m:BASE + SET * (m + 1)], ts[BASE + SET * m:BASE + SET * (m + 1)],
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
            RETURNS = Times_analyse_njit(dt[SET * m:SET * (m + 2)], ts[SET * m:SET * (m + 2)],
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
    RETURNS = Times_analyse_njit(dt[SET * (m + 2):SET * (m + 3)], ts[SET * (m + 2):SET * (m + 3)],
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

##################################################################################################################
# Test Walkforward Factory
##################################################################################################################
##############################################################################################################################
# Test the Trading Factory
##############################################################################################################################
n_permut = 10
cost = 0.0003
con = 0
BASE = round((STRATEGY.shape[0] / 5))
SET = round((STRATEGY.shape[0] / 10))
SIMUL = pd.DataFrame(np.zeros((n_permut, 1)))

RETURNS = STRATEGY['log_ret_x'].iloc[0:BASE].dropna().to_numpy()
Permutation_Matr_1 = []
Permutation_Matr_1 = pd.DataFrame(np.zeros((len(RETURNS), n_permut)))

for i in np.arange(0, n_permut, 1):
    PERM = np.random.permutation(RETURNS)
    Permutation_Matr_1.iloc[:, i] = PERM

pm = Permutation_Matr_1.fillna(0).to_numpy()

RETURNS = STRATEGY['log_ret_x'].iloc[BASE:].dropna().to_numpy()
Permutation_Matr_2 = []
Permutation_Matr_2 = pd.DataFrame(np.zeros((len(RETURNS), n_permut)))

for i in np.arange(0, n_permut, 1):
    PERM = np.random.permutation(RETURNS)
    Permutation_Matr_2.iloc[:, i] = PERM

oos = Permutation_Matr_2.fillna(0).to_numpy()

for i in range(0, n_permut, 1):

    if i == 0:

        OOS = pd.DataFrame(np.zeros((5, 3)))
        CUM_RET_OOS = 0
        a = np.zeros(shape=(8, 7))
        PARTIAL = pd.DataFrame(a, columns=['Start Date', 'End Date', 'OO S', 'CUM RET OOS', 'IN S', 'Best VIX',
                                           'Best over'])

        for m in range(0, 8):

            # First optimization with half data
            RESULTS = np.zeros((BASE + SET * m, 9))
            PROVA = []
            OPTIM = []
            OPTIM = pd.DataFrame(np.zeros((625, 3)))
            j = 0

            for threshold_VIX in np.arange(25, 50, 1):

                for threshold_Over in np.arange(-0.01, 0.015, 0.001):
                    RETURNS = Times_analyse_njit(dt[0:BASE + SET * m], ts[0:BASE + SET * m], ret[0:BASE + SET * m],
                                                 over[0:BASE + SET * m], vix[0:BASE + SET * m], Sig_8, Sig_7, Enter,
                                                 RESULTS, threshold_Over, threshold_VIX, cost)
                    PROVA = pd.DataFrame(RETURNS,
                                         columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades',
                                                  'Long_Trades', 'Short_Trades'])
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
            RETURNS = Times_analyse_njit(dt[BASE + SET * m:BASE + SET * (m + 1)],
                                         ts[BASE + SET * m:BASE + SET * (m + 1)],
                                         ret[BASE + SET * m:BASE + SET * (m + 1)],
                                         over[BASE + SET * m:BASE + SET * (m + 1)],
                                         vix[BASE + SET * m:BASE + SET * (m + 1)], Sig_8, Sig_7, Enter, RESULTS,
                                         Best_over, Best_VIX, cost)
            PROVA = pd.DataFrame(RETURNS,
                                 columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades', 'Long_Trades',
                                          'Short_Trades'])
            PROVA['DATE_F'] = pd.DataFrame(
                RETURNS[:, 0] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'), columns=['DATE_F'])
            PROVA['TIME_F'] = pd.DataFrame(
                RETURNS[:, 1] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'), columns=['TIME_F'])
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

            # print('Iteration number:', m)
            # print("IN_S", round(Best_Perf,2), 'Best_VIX', round(Best_VIX,0),'Best_over', round(Best_over,4))
            # print('Start_Date', Start_Date, 'End_Date', End_Date)
            # print("OO_S", round(PROVA['Trades'].sum(),2), 'CUM_RET_OOS', round(CUM_RET_OOS,2))

        ORIGINAL = CUM_RET_OOS
        print("Original", ORIGINAL)

    if i > 0:

        OOS = pd.DataFrame(np.zeros((5, 3)))
        CUM_RET_OOS = 0
        a = np.zeros(shape=(8, 7))
        PARTIAL = pd.DataFrame(a, columns=['Start Date', 'End Date', 'OO S', 'CUM RET OOS', 'IN S', 'Best VIX',
                                           'Best over'])

        for m in range(0, 8):

            # First optimization with half data
            RESULTS = np.zeros((BASE + SET * m, 9))
            PROVA = []
            OPTIM = []
            OPTIM = pd.DataFrame(np.zeros((625, 3)))
            j = 0
            ins = np.concatenate((pm[:, i], oos[0:SET * m, i]), axis=0)

            for threshold_VIX in np.arange(25, 50, 1):

                for threshold_Over in np.arange(-0.01, 0.015, 0.001):
                    RETURNS = Times_analyse_njit(dt[0:BASE + SET * m], ts[0:BASE + SET * m], ins,
                                                 over[0:BASE + SET * m], vix[0:BASE + SET * m], Sig_8, Sig_7, Enter,
                                                 RESULTS, threshold_Over, threshold_VIX, cost)
                    PROVA = pd.DataFrame(RETURNS,
                                         columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades',
                                                  'Long_Trades', 'Short_Trades'])
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
            RETURNS = Times_analyse_njit(dt[BASE + SET * m:BASE + SET * (m + 1)],
                                         ts[BASE + SET * m:BASE + SET * (m + 1)], oos[SET * m:SET * (m + 1), i],
                                         over[BASE + SET * m:BASE + SET * (m + 1)],
                                         vix[BASE + SET * m:BASE + SET * (m + 1)], Sig_8, Sig_7, Enter, RESULTS,
                                         Best_over, Best_VIX, cost)
            PROVA = pd.DataFrame(RETURNS,
                                 columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades', 'Long_Trades',
                                          'Short_Trades'])
            PROVA['DATE_F'] = pd.DataFrame(
                RETURNS[:, 0] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'), columns=['DATE_F'])
            PROVA['TIME_F'] = pd.DataFrame(
                RETURNS[:, 1] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'), columns=['TIME_F'])
            Start_Date = PROVA['DATE_F'].iloc[1]
            End_Date = PROVA['DATE_F'].iloc[SET - 1]
            CUM_RET_OOS = CUM_RET_OOS + PROVA['Trades'].sum()

        SIMUL.iloc[i] = CUM_RET_OOS
        print("SIMUL", i, CUM_RET_OOS)

Rand_Fitting = SIMUL.mean()
Skill = ORIGINAL - Rand_Fitting
print("Rand_Fitting", Rand_Fitting, "ORIGINAL", ORIGINAL, "Skill", Skill)

###############################################################################################################
# Basic Cross-Validation with Grid Optim
###############################################################################################################

N = 10  # Number splits
SET = round((STRATEGY.shape[0] / N))
OOS = pd.DataFrame(np.zeros((N, 3)))
CUM_RET_OOS = 0
a = np.zeros(shape=(10, 7))
PARTIAL = pd.DataFrame(a, columns=['Start Date', 'End Date', 'OO S', 'CUM RET OOS', 'IN S', 'Best VIX', 'Best over'])

for m in range(0, N):

    dt_test = dt[SET * m:SET * (m + 1)]
    dt_train_1 = dt[0:SET * m]
    dt_train_2 = dt[SET * (m + 1):-1]
    dt_train = np.concatenate((dt_train_1, dt_train_2), axis=0)

    ts_test = ts[SET * m:SET * (m + 1)]
    ts_train_1 = ts[0:SET * m]
    ts_train_2 = ts[SET * (m + 1):-1]
    ts_train = np.concatenate((ts_train_1, ts_train_2), axis=0)

    ret_test = ret[SET * m:SET * (m + 1)]
    ret_train_1 = ret[0:SET * m]
    ret_train_2 = ret[SET * (m + 1):-1]
    ret_train = np.concatenate((ret_train_1, ret_train_2), axis=0)

    over_test = over[SET * m:SET * (m + 1)]
    over_train_1 = over[0:SET * m]
    over_train_2 = over[SET * (m + 1):-1]
    over_train = np.concatenate((over_train_1, over_train_2), axis=0)

    vix_test = vix[SET * m:SET * (m + 1)]
    vix_train_1 = vix[0:SET * m]
    vix_train_2 = vix[SET * (m + 1):-1]
    vix_train = np.concatenate((vix_train_1, vix_train_2), axis=0)

    # Initialization for grid optimization
    RESULTS = np.zeros((ts_train.shape[0], 9))
    PROVA = []
    OPTIM = []
    OPTIM = pd.DataFrame(np.zeros((625, 3)))
    j = 0

    for threshold_VIX in np.arange(25, 50, 1):
        for threshold_Over in np.arange(-0.01, 0.015, 0.001):
            RETURNS = Times_analyse_njit(dt_train, ts_train, ret_train, over_train, vix_train, Sig_8, Sig_7, Enter,
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
    RETURNS = Times_analyse_njit(dt_test, ts_test, ret_test, over_test, vix_test, Sig_8, Sig_7, Enter, RESULTS,
                                 Best_over, Best_VIX, cost)
    PROVA = pd.DataFrame(RETURNS, columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades', 'Long_Trades',
                                           'Short_Trades'])
    PROVA['DATE_F'] = pd.DataFrame(RETURNS[:, 0] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'),
                                   columns=['DATE_F'])
    PROVA['TIME_F'] = pd.DataFrame(RETURNS[:, 1] * np.timedelta64(1, 'm') + np.datetime64('1990-01-01T00:00:00Z'),
                                   columns=['TIME_F'])
    Start_Date = PROVA['DATE_F'].iloc[1]
    End_Date = PROVA['DATE_F'].iloc[ts_test.shape[0] - 1]
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

    if m == 0:
        print('Basic 10 fold cross-validation:')

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
            RETURNS = Times_analyse_njit(dt_train, ts_train, ret_train, over_train, vix_train, Sig_8, Sig_7, Enter,
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
    RETURNS = Times_analyse_njit(dt_test, ts_test, ret_test, over_test, vix_test, Sig_8, Sig_7, Enter, RESULTS,
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

N_PATHS = _get_number_of_backtest_paths(n_splits, n_test_splits)
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
            RETURNS = Times_analyse_njit(dt_train, ts_train, ret_train, over_train, vix_train, Sig_8, Sig_7, Enter,
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
        RETURNS = Times_analyse_njit(dt_test, ts_test, ret_test, over_test, vix_test, Sig_8, Sig_7, Enter, RESULTS,
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
print("Number of Paths", _get_number_of_backtest_paths(n_splits, n_test_splits))
print("Average OOS per path ", round(CUM_RET_OOS / _get_number_of_backtest_paths(n_splits, n_test_splits), 2))
