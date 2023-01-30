##################################################################################################################################
# Libraries
##################################################################################################################################

from Strategy_Night import *
from Data_Import import *
from Performance_Metrics import *
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

#####################################################################################################################
#  Select list of Markets
#####################################################################################################################

LIST_EQUITY_INDEXES = ["^GSPC", "^IXIC", "^DJI", "^RUT", "^GDAXI", "^FCHI", "FTSEMIB.MI"]

#####################################################################################################################
# Import the Dtabase
#####################################################################################################################

OHLCV = download_data(LIST_EQUITY_INDEXES)

#####################################################################################################################
# Compute Total, Intraday and Overnight  Returns
#####################################################################################################################

Tot_Ret = Total_returns(OHLCV, LIST_EQUITY_INDEXES)
Intra_Ret = Intraday_returns(OHLCV, LIST_EQUITY_INDEXES)
Overnight_Ret = Overnight_returns(OHLCV, LIST_EQUITY_INDEXES)

#############################################################################################################
# Fix treshold
#############################################################################################################

Treshold = 0.002
LONG_INTRA = Intraday_Long(Overnight_Ret, Intra_Ret, LIST_EQUITY_INDEXES, Treshold)
SHORT_INTRA = Intraday_Short(Overnight_Ret, Intra_Ret, LIST_EQUITY_INDEXES, Treshold)

#############################################################################################################
# Plot cumulated returns of the fifferent time series
#############################################################################################################

# Select an index you vant to visualize
Ticker = "^FCHI"
plot_Cum_Ret( Ticker, Tot_Ret, Intra_Ret, Overnight_Ret, LONG_INTRA, SHORT_INTRA)

A=1
