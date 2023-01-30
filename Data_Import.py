
import pandas as pd
import numpy as np
import yfinance as yf
from yahoo_fin.stock_info import *


def download_data(LIST_EQUITY_INDEXES):

    DATA_INDEXES = get_data(LIST_EQUITY_INDEXES[0])

    for elem in LIST_EQUITY_INDEXES:
        DATA_INDEXES_2 = get_data(elem)
        DATA_INDEXES = pd.concat([DATA_INDEXES, DATA_INDEXES_2], axis=1)
    DATA_INDEXES = DATA_INDEXES.iloc[:,7:]

    return DATA_INDEXES


def Total_returns(OHLCV, LIST_EQUITY_INDEXES):
    OHLCV.replace(0, np.nan, inplace=True)
    Tot_Ret = pd.DataFrame(index=range(OHLCV.shape[0]), columns=range(int(OHLCV.shape[1] / 7)))
    Tot_Ret = np.log(OHLCV.close) - np.log(OHLCV.close.shift(1))  # compute log returns
    Tot_Ret.columns = LIST_EQUITY_INDEXES
    return Tot_Ret


def Intraday_returns(OHLCV, LIST_EQUITY_INDEXES):
    Intra_Ret = pd.DataFrame(index=range(OHLCV.shape[0]), columns=range(int(OHLCV.shape[1] / 7)))
    Log_Close = np.log(OHLCV.close)  # compute log returns
    Log_Open = np.log(OHLCV.open)
    Log_Open.columns = Log_Close.columns
    Intra_Ret = Log_Close - Log_Open
    Intra_Ret.columns = LIST_EQUITY_INDEXES
    Intra_Ret.replace(0, np.nan, inplace=True)
    return Intra_Ret


def Overnight_returns(OHLCV, LIST_EQUITY_INDEXES):
    Over_Ret = pd.DataFrame(index=range(OHLCV.shape[0]), columns=range(int(OHLCV.shape[1] / 7)))
    Log_Close = np.log(OHLCV.close)  # compute log returns
    LOG_Lag_Close = np.log(OHLCV.close.shift(1))
    Log_Open = np.log(OHLCV.open)
    Log_Open.columns = Log_Close.columns
    Over_Ret = Log_Open - LOG_Lag_Close
    Over_Ret.columns = LIST_EQUITY_INDEXES
    return Over_Ret



def Preparation(index):

    index['Close'] = index['Close'].astype(str)
    index['Close'] = [x.replace(',', '.') for x in index['Close']]  # Call preparation function to change , into . and to compute returns
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

    index['Close'] = pd.to_numeric(index['Close'], errors='coerce') # turn to numeric the string and remove errors when needed
    index['Open'] = pd.to_numeric(index['Open'], errors='coerce')
    index['Low'] = pd.to_numeric(index['Low'], errors='coerce')
    index['High'] = pd.to_numeric(index['High'], errors='coerce')

    index['log_ret'] = np.log(index.Close) - np.log(index.Close.shift(1)) # compute log returns

    return index # return a dataframe which can be used for computational purposes

def Index_Cleaning(INDEX, Date_Start, Date_End):

    INDEX = INDEX[["Date", "Time", "Bar#", "Bar Index",  "Open", "High", "Low", "Close"]]
    index = INDEX.to_numpy() # convert to numpy

    # Fix Date interval
    TIME = pd.to_datetime(index[:,0])  # convert index to datetime format
    ts = (TIME - np.datetime64('1990-01-01T00:00:00Z')) / np.timedelta64(1, 'h') # turn datetime into a numeric version
    ts = ts.to_numpy() # convert to numpy
    Signals = pd.DataFrame([Date_Start, Date_End]) # convert dataframe
    Signals = Signals.to_numpy() # convert to numpy
    Signals = pd.to_datetime(Signals[:,0]) # convert dataframe the
    Sig = (Signals - np.datetime64('1990-01-01T00:00:00Z')) / np.timedelta64(1, 'h') # convert the two critical dates into numeric format
    Sig = Sig.to_numpy() # convert to numpy
    A = (ts > Sig[0]) #select dates above the first Critical date
    NEW = index[A] #apply this selection to the dataframe
    INDEX = pd.DataFrame(NEW, columns = ['Date', 'Time', 'Bar#', 'Bar Index','Open', 'High', 'Low', 'Close']) # turn into dataframe  name the columns

    Preparation(INDEX) # Call preparation function to change , into . and to compute returns
    ts = ts[A]
    index = index[A]
    B = (ts<=Sig[1])
    NEW2 = index[B]
    INDEX = pd.DataFrame(NEW2, columns = ["Date", "Time", "Bar#", "Bar Index", "Open", "High", "Low", "Close"])
    Preparation(INDEX)

    return INDEX


def Index_Cleaning_D(INDEX, Date_Start, Date_End):

    INDEX = INDEX[["Date",  "Bar#", "Bar Index",  "Open", "High", "Low", "Close"]]
    index=INDEX.to_numpy() # convert to numpy

    # Fix Date interval
    Date_Start = "01/03/2010"
    Date_End = "01/01/2020"
    TIME=pd.to_datetime(index[:,0])
    ts = (TIME - np.datetime64('1990-01-01T00:00:00Z')) / np.timedelta64(1, 'h')
    ts=ts.to_numpy()
    Signals=pd.DataFrame([Date_Start, Date_End])
    Signals=Signals.to_numpy()
    Signals=pd.to_datetime(Signals[:,0])
    Sig = (Signals - np.datetime64('1990-01-01T00:00:00Z')) / np.timedelta64(1, 'h')
    Sig = Sig.to_numpy()
    A = (ts>Sig[0])
    NEW = index[A]
    INDEX = pd.DataFrame(NEW, columns = ['Date', 'Bar#', 'Bar Index','Open', 'High', 'Low', 'Close'])

    Preparation(INDEX)
    ts = ts[A]
    index = index[A]
    B = (ts<=Sig[1])
    NEW2 = index[B]
    INDEX = pd.DataFrame(NEW2, columns = ["Date",  "Bar#", "Bar Index", "Open", "High", "Low", "Close"])
    Preparation(INDEX)
    return INDEX


def Dataframe_Creation(INDEX):
    # Reverse Sort and Turn Time Date into strings
    INDEX = INDEX.iloc[::-1]
    INDEX['Time'] = INDEX['Time'].astype(str)
    INDEX['Date'] = INDEX['Date'].astype(str)

    # Create columns for analyses
    Len = INDEX['Time'].size
    INDEX['Start'] = [0] * Len
    INDEX['End'] = [0] * Len
    INDEX['Over'] = [0] * Len
    INDEX['Intra'] = [0] * Len
    INDEX['Start'] = INDEX['Start'].astype(str)
    INDEX['End'] = INDEX['End'].astype(str)

    return INDEX


def FUTURE_Prepare(FUTURE):
    FUTURE[['Date', 'Time', "Bar#", 'Bar Index', 'Open', 'High', 'Low', 'Close']] = FUTURE[
        'Date;Time;Bar#;Bar Index;Open;High;Low;Close'].str.split(';', expand=True)
    del FUTURE['Date;Time;Bar#;Bar Index;Open;High;Low;Close']

    # Find close bar return for the Future
    FUTURE = FUTURE.iloc[::-1]
    FUTURE = Preparation(FUTURE)
    return FUTURE

def VIX_Prepare(VIX):

    VIX = VIX.iloc[::-1]
    Preparation(VIX)
    return VIX