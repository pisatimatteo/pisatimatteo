import pyfolio as pf
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')
from bs4 import BeautifulSoup

print("########################################################################################################")
print("#                                            S&P 500                                                   #")
print("########################################################################################################")

fb = yf.Ticker('^GSPC')
history = fb.history('20Y')
history.index = history.index.tz_localize('utc')
returns = history.Close.pct_change().dropna()
data = pf.create_returns_tear_sheet(returns, live_start_date='2020-8-1')

A=1