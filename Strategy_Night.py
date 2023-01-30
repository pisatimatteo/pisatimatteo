##################################################################################################################################
# Libraries
##################################################################################################################################

import pandas as pd
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from Performance_Metrics import sharpe_ratio

# Prova Git

def overnight_ret_index(index, Over_tresh):

    # Add needed columns
    index['O-E Prices'] = 0
    index['O Ret'] = 0
    index['I Ret'] = 0
    index['T Ret'] = 0
    index['Str Long'] = 0
    index['Str Short'] = 0
    j = 0 # skip first day

    # Initialization of the columns with the first-opening and last-close prices
    p_open = index.iloc[0, 4]  # compy first value in the column of open prices
    index.iloc[0, 9] = p_open  # we copy this value int the column O-E Prices

    p_close = index.iloc[-1, 7]  # compy last value in the column of close prices
    index.iloc[-1, 9] = p_close  # we copy this value int the column O-E Prices
    overn_ret = 0 # set first overnight return to zero

    for i in range(1, index['Date'].size - 1):

        if index.iloc[i, 0] != index.iloc[i - 1, 0]:  # change in day, new day arrived

            p_open = index.iloc[i, 4]  # column of open bars
            index.iloc[i, 9] = p_open  # column of open days
            index.iloc[i, 10] = np.log(p_open) - np.log(p_close)  # overnight ret
            overn_ret = index.iloc[i, 10]
            j = 1 # flag first day finished

        elif index.iloc[i, 0] != index.iloc[i + 1, 0]:  # change in day, old day finished

            p_close = index.iloc[i, 7]  # column of close
            index.iloc[i, 9] = p_close  # column of close days
            index.iloc[i, 11] = np.log(p_close) - np.log(p_open)  # intra ret
            index.iloc[i, 12] = overn_ret + index.iloc[i, 11]  # tot ret

            if j == 1:  # do not comput the first day (no overnight return is obviously available)
                if overn_ret > Over_tresh:
                    index.iloc[i, 13] = index.iloc[i, 11]
                else:
                    index.iloc[i, 14] = index.iloc[i, 11] * -1


def overnight_ret(index, Len):

    p_open = index["Open"].iloc[0]  # column of open bars
    index["Start"].iloc[0] = p_open  # column of open days
    index["Start"].iloc[0] = index["Start"].iloc[0].replace(',', '.')
    index["Start"].iloc[0] = pd.to_numeric(index["Start"].iloc[0])
    p_open = index["Start"].iloc[0]

    p_close = index["Close"].iloc[Len - 1]  # column of close
    index["End"].iloc[Len - 1] = p_close  # column of close days
    index["End"].iloc[Len - 1] = index["End"].iloc[Len - 1].replace(',', '.')
    index["End"].iloc[Len - 1] = pd.to_numeric(index["End"].iloc[Len - 1])
    p_close = index["End"].iloc[Len - 1]

    ########################################################################################
    # Total Return in the period
    ########################################################################################
    TOTALRETURN = np.log(p_close) - np.log(p_open)
    print('Total Return in the Period for the INdex', index.iloc[0, 0], '/', index.iloc[Len - 1, 0], '=', TOTALRETURN)

    for i in range(1, index['Date'].size - 1):

        if index["Date"].iloc[i] != index["Date"].iloc[i - 1]:  #  new day arrived

            p_open = index["Open"].iloc[i]    # column of open bars
            index["Start"].iloc[i] = p_open  # column of open days
            index["Start"].iloc[i] = index["Start"].iloc[i].replace(',', '.')
            index["Start"].iloc[i] = pd.to_numeric(index["Start"].iloc[i])
            p_open = index["Start"].iloc[i]
            index["Over"].iloc[i] = np.log(p_open) - np.log(p_close)  # overnight ret
            # rint(index.iloc[i,10])

        elif index["Date"].iloc[i] != index["Date"].iloc[i + 1]:  #  old day finished

            p_close = index["Close"].iloc[i]  # column of close
            index["End"].iloc[i] = p_close  # column of close days
            index["End"].iloc[i] = index["End"].iloc[i].replace(',', '.')
            index["End"].iloc[i] = pd.to_numeric(index["End"].iloc[i])
            p_close = index["End"].iloc[i]
            index["Intra"].iloc[i] = np.log(p_close) - np.log(p_open)  # intraday ret

    return index


def compute_Future_Returns_on_open_market_time(MERGED_DATABASE, Initial_Date):
    Starting = MERGED_DATABASE.index[MERGED_DATABASE['Date_x'] == Initial_Date].tolist()
    STARTING = Starting[0]
    STRATEGY = MERGED_DATABASE[STARTING:]
    STRATEGY['Start'] = STRATEGY['Start'].fillna(0).astype(int)
    STRATEGY['End'] = STRATEGY['End'].fillna(0).astype(int)
    STRATEGY['Cum_Ret'] = 0
    STRATEGY['Trades'] = 0
    Len = STRATEGY['Time_x'].size
    STRATEGY['Start_F'] = [0] * Len
    STRATEGY['End_F'] = [0] * Len
    STRATEGY['Over_F'] = [0] * Len
    STRATEGY['Intra_F'] = [0] * Len

    # Find overnight return for the Futures
    p_open = STRATEGY["Open_x"].iloc[0]  # column of open bars
    STRATEGY['Start_F'].iloc[0] = p_open  # column of open days
    p_close = STRATEGY['Close_x'].iloc[Len - 1]  # column of close
    STRATEGY['End_F'].iloc[Len - 1] = p_close  # column of close days

    # STRATEGY['Position']=STRATEGY['Position'].fillna(0)
    # STRATEGY['End']=STRATEGY['End'].fillna(0)
    # STRATEGY['Start']=STRATEGY['Start'].fillna(0)

    for i in range(1,
                   Len - 1):  # this is calculating the intraday and overnight returns for the future when the index is open

        if STRATEGY['Start'].iloc[i] != 0:  # change in day,new day arrived

            p_open = STRATEGY["Open_x"].iloc[i]  # column of open bars
            STRATEGY['Start_F'].iloc[i] = p_open  # column of open days
            STRATEGY['Over_F'].iloc[i] = np.log(p_open) - np.log(p_close)  # overnight ret

        elif STRATEGY['End'].iloc[i] != 0:  # change in day, old day finished

            p_close = STRATEGY["Close_x"].iloc[i]  # column of close
            STRATEGY['End_F'].iloc[i] = p_close  # column of close days
            STRATEGY['Intra_F'].iloc[i] = np.log(p_close) - np.log(p_open)  # overnight ret
    return STRATEGY


def Intraday_Long(Overnight_Ret, Intra_Ret, LIST_EQUITY_INDEXES, Treshold):

    LONG_INTRA = pd.DataFrame(index=range(Overnight_Ret.shape[0]), columns=range(int(Overnight_Ret.shape[1] / 7)))
    A = (Overnight_Ret > Treshold)
    LONG_INTRA = Intra_Ret[A]
    LONG_INTRA.columns = LIST_EQUITY_INDEXES

    return LONG_INTRA

def Intraday_Short(Overnight_Ret, Intra_Ret, LIST_EQUITY_INDEXES, Treshold):

    SHORT_INTRA = pd.DataFrame(index=range(Overnight_Ret.shape[0]), columns=range(int(Overnight_Ret.shape[1] / 7)))
    A = (Overnight_Ret < Treshold)
    SHORT_INTRA = -Intra_Ret[A]
    SHORT_INTRA.columns = LIST_EQUITY_INDEXES

    return SHORT_INTRA

def plot_Cum_Ret(Ticker, Tot_Ret, Intra_Ret, Overnight_Ret, LONG_INTRA, SHORT_INTRA):

    Cum_SUM_Tot_Ret = Tot_Ret.cumsum()
    Cum_SUM_Intra_Ret = Intra_Ret.cumsum()
    Cum_SUM_Overnight_Ret = Overnight_Ret.cumsum()
    Cum_SUM_LONG_INTRA = LONG_INTRA.cumsum()
    Cum_SUM_SHORT_INTRA = SHORT_INTRA.cumsum()

    Cum_Sum_Tot_Ret_Ticker = Cum_SUM_Tot_Ret.loc[:, Ticker].dropna()
    Cum_SUM_Intra_Ticker = Cum_SUM_Intra_Ret.loc[:, Ticker].dropna()
    Cum_SUM_Overnight_Ticker = Cum_SUM_Overnight_Ret.loc[:, Ticker].dropna()
    Cum_SUM_LONG_Ticker = Cum_SUM_LONG_INTRA.loc[:, Ticker].dropna()
    Cum_SUM_SHORT_Ticker = Cum_SUM_SHORT_INTRA.loc[:, Ticker].dropna()

    plt.figure(figsize=(16, 8), dpi=150)
    Cum_Sum_Tot_Ret_Ticker.plot(label='Cum Sum Tot Ret', color='red')
    Cum_SUM_Intra_Ticker.plot(label='Cum SUM Intra', color='green')
    Cum_SUM_Overnight_Ticker.plot(label='Cum SUM Over', color='pink')
    Cum_SUM_LONG_Ticker.plot(label='Cum SUM LONG Ticker', color='blue')
    Cum_SUM_SHORT_Ticker.plot(label='Cum SUM SHORT Ticker', color='brown')

    plt.title('{} Cum Ret Strategy Components'.format(Ticker))
    plt.xlabel('Years')
    plt.legend()
    plt.show()

    return 0


def relative_strategy_times(STRATEGY):

    index = STRATEGY.to_numpy()
    TIME = pd.to_datetime(index[:, 1])
    ts = (TIME - np.datetime64('1990-01-01T00:00:00Z')) / np.timedelta64(1, 'm')
    ts = ts.to_numpy()
    ts = np.round(ts, 0)

    DATE = pd.to_datetime(index[:, 0])
    dt = (DATE - np.datetime64('1990-01-01T00:00:00Z')) / np.timedelta64(1, 'm')
    dt = dt.to_numpy()
    dt = np.round(dt, 0)

    RET = STRATEGY['log_ret_x'].fillna(0)
    ret = RET.to_numpy()

    OVER = STRATEGY['Over_F']
    OVER = OVER.fillna(0)
    over = OVER.to_numpy()

    VIX = STRATEGY['Close']
    VIX = VIX.fillna(0).replace(to_replace=0, method='ffill')
    vix = VIX.to_numpy()

    RESULTS = np.zeros((ts.shape[0], 9))

    # Signals
    Time_Entry_1 = "08:00:00"
    Time_Exit_1 = "16:30:00"
    Time_Entry_2 = Time_Exit_1
    Time_Exit_2 = "23:55:00"
    Time_Entry_3 = "00:00:00"
    Time_Exit_3 = Time_Entry_1
    VIX_Time = "17:30:00"
    Signals = pd.DataFrame([Time_Entry_1, Time_Exit_1, Time_Entry_2, Time_Exit_2, Time_Entry_3, Time_Exit_3, VIX_Time])
    Signals2 = Signals.to_numpy()
    Signals3 = pd.to_datetime(Signals2[:, 0])
    Sig = (Signals3 - np.datetime64('1990-01-01T00:00:00Z')) / np.timedelta64(1, 'm')
    Sig = Sig.to_numpy()
    Sig_8 = np.round(Sig, 0)

    # Signals 2
    Time_Entry_1 = "07:00:00"
    Time_Exit_1 = "15:30:00"
    Time_Entry_2 = Time_Exit_1
    Time_Exit_2 = "23:55:00"
    Time_Entry_3 = "00:00:00"
    Time_Exit_3 = Time_Entry_1
    VIX_Time = "15:30:00"
    Signals2 = pd.DataFrame([Time_Entry_1, Time_Exit_1, Time_Entry_2, Time_Exit_2, Time_Entry_3, Time_Exit_3, VIX_Time])
    Signals2 = Signals2.to_numpy()
    Signals3 = pd.to_datetime(Signals2[:, 0])
    Sig2 = (Signals3 - np.datetime64('1990-01-01T00:00:00Z')) / np.timedelta64(1, 'm')
    Sig2 = Sig2.to_numpy()
    Sig_7 = np.round(Sig2, 0)

    # Entry
    Time_Entry_A = "07:00:00"
    Time_Entry_B = "08:00:00"
    ENTRY = pd.DataFrame([Time_Entry_A, Time_Entry_B])
    ENTRY2 = ENTRY.to_numpy()
    ENTRY3 = pd.to_datetime(ENTRY2[:, 0])
    Enter = (ENTRY3 - np.datetime64('1990-01-01T00:00:00Z')) / np.timedelta64(1, 'm')
    Enter = Enter.to_numpy()
    Enter = np.round(Enter, 0)

    return dt, ts, ret, over, vix, Sig_8, Sig_7, Enter, RESULTS


#@nb.njit
def Relative_Strategy_njit(Dates, Times, Returns, OVER, VIX, Signals_8, Signals_7, ENTER, RESULTS, threshold_Over,
                       threshold_VIX, cost):

    # Initialize values
    CURRENT_OVER = 0 # Current Overrnght return
    CURRENT_VIX = 0 # Last value Vix
    CRET = 0   # Cumulated Return
    POS = 0    # Current position
    POSL = 0   # Past Position
    Signals = Signals_8 # entry hours

    for i in range(1, Times.shape[0]):
        # Create the needed reults columns
        RESULTS[i, 0] = Dates[i]
        RESULTS[i, 1] = Times[i]
        RESULTS[i, 2] = Returns[i]
        RESULTS[i, 3] = OVER[i]

        if Times[i] == Signals[6]:  # update VIX once per day
            CURRENT_VIX = VIX[i]  # VIX close bar

        if OVER[i] != 0: # if we are ath the opening moment
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
            RESULTS[i, 6] = CRET - Returns[i] - cost*2
            # Long
            RESULTS[i, 8] = RESULTS[i, 6]
            CRET = 0

        elif POSL == 1 and POS == -1:  # data entry morning received with delay
            RESULTS[i, 6] = CRET + Returns[i] - cost*2
            # Long
            RESULTS[i, 7] = RESULTS[i, 6]
            CRET = 0

    # Performance metrics
    return RESULTS

def optim_grid_1D(strat_func, target_variable, trading_days, risk_free_ratio, min_param, max_param, increment, STRATEGY, *args):


    OPTIM = pd.DataFrame(np.zeros((int((max_param-min_param)/increment), 4)), columns=['Parameter', 'Profit_Ratio', 'Cumulated_Returns', "Sharpe_Ratio"])
    j = 0

    for target_variable in np.arange(min_param, max_param, increment):

        RETURNS = strat_func(*args)

        STRAT_RETURNS = pd.DataFrame(RETURNS, columns=['Date', 'Time', 'Returns', 'Over', 'POS', 'CUM_RET', 'Trades',
                                                       'Long_Trades',
                                                       'Short_Trades'])
        Cumulated_return_Positive_Trades = STRAT_RETURNS['Trades'].where(STRAT_RETURNS['Trades'] > 0).sum(0)
        Cumulated_return_Neagative_Trades = STRAT_RETURNS['Trades'].where(STRAT_RETURNS['Trades'] < 0).sum(0)
        OPTIM.iloc[j, 0] = target_variable
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
        annualized_sr = sharpe_ratio(logret_by_days, entries_per_year=trading_days, risk_free_rate=risk_free_ratio)
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
    print("Param_Best_Perf_PR", round(Param_Best_Perf_PR, 4), "PR_Best_Perf_PR", round(PR_Best_Perf_PR, 4),
          "CR_Best_Perf_PR", round(CR_Best_Perf_PR, 4), "SR_Best_Perf_PR", round(SR_Best_Perf_PR, 4))
    print("Param_Best_Perf_CR", round(Param_Best_Perf_CR, 4), "PR_Best_Over_CR", round(PR_Best_Over_CR, 4),
          "CR_Cum_Ret_CR", round(CR_Cum_Ret_CR, 4), "SR_Cum_Ret_CR", round(SR_Cum_Ret_CR, 4))
    print("Param_Best_Perf_SR", round(Param_Best_Perf_SR, 4), "PR_Best_Over_SR", round(PR_Best_Over_SR, 4),
          "CR_Cum_Ret_SR", round(CR_Cum_Ret_SR, 4), "SR_Cum_Ret_SR", round(SR_Cum_Ret_SR, 4))
    print("/////////////////////////////////////////////////////////////////////////////////////////////////")
    print(OPTIM)
    return OPTIM