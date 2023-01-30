import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from typing import Union, Tuple
import sys
import time
import datetime as dt
import multiprocessing as mp
from numba import njit
import statistics
from sklearn.linear_model import LinearRegression

def tick_rule(tick_prices):
    """
    Applies the tick rule to classify trades as buy-initiated or sell-initiated

    :param tick_prices: a series of tick prices
    :return: a series of tick signs
    """
    price_change = tick_prices.diff()
    aggressor = pd.Series(index=tick_prices.index, data=np.nan)

    aggressor.iloc[0] = 1.
    aggressor[price_change < 0] = -1.
    aggressor[price_change > 0] = 1.
    aggressor = aggressor.fillna(method='ffill')
    return aggressor

def roll_model(prices):
    """
    Estimates 1/2*(bid-ask spread) and unobserved noise based on price sequences

    :param prices: a series of prices
    :return: a tuple with estimated of values of (spread, unobserved noise)
    """
    price_change = prices.diff()
    autocorr = price_change.autocorr(lag=1)
    spread_squared = np.max([-autocorr, 0])
    spread = np.sqrt(spread_squared)
    noise = price_change.var() - 2 * (spread ** 2)
    return spread, noise

def high_low_estimator(high, low, window):
    """
    Estimates volatility using Parkinson's method

    :param high: a series of high prices
    :param low: a series of low prices
    :param window: length of the rolling estimation window
    :return: volatility estimate
    """
    log_high_low = np.log(high / low)
    volatility = log_high_low.rolling(window=window).mean() / np.sqrt(8. / np.pi)
    return volatility

class CorwinShultz:
    """
    A class that encapsulates all the functions for Corwin and Shultz estimator
    """

    @staticmethod
    def get_beta(high, low, sample_length):
        """
        Computes beta in Corwin and Shultz bid-ask spread estimator

        :param high: a series of high prices
        :param low: a series of low prices
        :param sample_length: number of values per sample
        :return: beta estimate
        """
        log_high_low = np.log(high / low) ** 2
        sum_neighbors = log_high_low.rolling(window=2).sum()
        beta = sum_neighbors.rolling(window=sample_length).mean()
        return beta

    @staticmethod
    def get_gamma(high, low):
        """
        Computes gamma in Corwin and Shultz bid-ask spread estimator

        :param high: a series of high prices
        :param low: a series of low prices
        :return: gamma estimate
        """
        high_over_2_bars = high.rolling(window=2).max()
        low_over_2_bars = low.rolling(window=2).min()
        gamma = np.log(high_over_2_bars / low_over_2_bars) ** 2
        return gamma

    @staticmethod
    def get_alpha(beta, gamma):
        """
        Computes alpha in Corwin and Shultz bid-ask spread estimator

        :param beta: Corwin and Shultz beta estimate
        :param gamma: Corwin and Shultz gamma estimate
        :return: aplha estimate
        """
        denominator = 3 - 2 ** 1.5
        beta_term = (np.sqrt(2) - 1) * np.sqrt(beta) / denominator
        gamma_term = np.sqrt(gamma / denominator)
        alpha = beta_term - gamma_term
        alpha[alpha < 0] = 0
        return alpha

    @staticmethod
    def get_becker_parkinson_volatility(beta, gamma):
        """
        Computes Becker-Parkinson implied volatility

        :param beta: Corwin and Shultz beta estimate
        :param gamma: Corwin and Shultz gamma estimate
        :return: volatility estimate
        """
        k2 = np.sqrt(8 / np.pi)
        denominator = 3 - 2 ** 1.5
        beta_term = (2 ** (-.5) -1) * np.sqrt(beta) / (k2 * denominator)
        gamma_term = np.sqrt(gamma / (k2 ** 2 * denominator))
        volatility = beta_term + gamma_term
        volatility[volatility < 0] = 0
        return volatility



def corwin_shultz_spread(high, low, sample_length=1):
    """
    Computes an estimate of the bid-ask spread according to Corwin and Shultz estimator

    :param high: a series of high prices
    :param low: a series of low prices
    :param sample_length: number of values per sample
    :return: spread estimate
    """
    beta = CorwinShultz.get_beta(high, low, sample_length)
    gamma = CorwinShultz.get_gamma(high, low)
    alpha = CorwinShultz.get_alpha(beta, gamma)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread

def becker_parkinson_volatility(high, low, sample_length=1):
    """
    Computes implied volatility according Becker-Parkinson method

    :param high: a series of high prices
    :param low: a series of low prices
    :param sample_length: number of values per sample
    :return: volatility estimate
    """
    beta = CorwinShultz.get_beta(high, low, sample_length)
    gamma = CorwinShultz.get_gamma(high, low)
    volatility = CorwinShultz.get_becker_parkinson_volatility(beta, gamma)
    return volatility

def kyles_lambda(tick_prices, tick_volumes, tick_signs, regressor=LinearRegression()):
    """
    Estimates price impact coefficient based on Kyle's model

    :param tick_prices: a series of tick prices
    :param tick_volumes: a series of associated tick volumes
    :param regressor: a regressor to fit the estimate
    :return: Kyle's lambda
    """
    price_change = tick_prices.diff()
    net_order_flow = tick_signs * tick_volumes
    x_val   = net_order_flow.values[1:].reshape(-1, 1)
    y_val = price_change.dropna().values
    lambda_ = regressor.fit(x_val  , y_val)
    return lambda_.coef_[0]

def amihuds_lambda(close, dollar_volume, regressor=LinearRegression()):
    """
    Estimates price impact coefficient based on Amihud's model

    :param close: a series of clsoe prices
    :param dollar_volume: a series of associated dollar volumes
    :param regressor: a regressor to fit the estimate
    :return: Amihud's lambda
    """
    log_close = np.log(close)
    abs_change = np.abs(log_close.diff())
    x_val   = dollar_volume.values[1:].reshape(-1, 1)
    y_val = abs_change.dropna()
    lambda_ = regressor.fit(x_val  , y_val)
    return lambda_.coef_[0]

def hasbroucks_lambda(close, hasb_flow, regressor=LinearRegression()):
    """
    Estimates price impact coefficient based on Hasbrouck's model

    :param close: a series of clsoe prices
    :param hasb_flow: a series of net square root dollar volume of the form sum(tick_sign * sqrt(tick_price * tick_volume))
    :param regressor: a regressor to fit the estimate
    :return: Hasbrouck's lambda
    """
    ratio = pd.Series(index=close.index[1:], data=close.values[1:]/close.values[:-1])
    log_ratio = np.log(ratio)
    x_val   = hasb_flow.values[1:].reshape(-1, 1)
    y_val = log_ratio
    lambda_ = regressor.fit(x_val  , y_val)
    return lambda_.coef_[0]


def hasbroucks_flow(tick_prices, tick_volumes, tick_sings):
    """
    A helper function that computes net square root doolar volume

    :param tick_prices: a series of tick prices
    :param tick_volumes: a series of associated tick volumes
    :param tick_signs: a series of associated tick signs
    :return: net square root doolar volume
    """
    return (np.sqrt(tick_prices * tick_volumes) * tick_sings).sum()

def vpin(buy_volumes, sell_volumes, volume, num_bars):
    """
    Estimates Volume-Synchronized Probability of Informed Trading

    :param buy_volumes: a series of total volume from buy-initiated trades in each bar
    :param sell_volumes: a series of total volume from sell-initiated trades in each bar
    :param volume: volume sampling threshold used with volume bars
    :param num_bars: the size of the rolling window for the estimate
    :return: volume-synchronized probability of informed trading
    """
    abs_diff = (buy_volumes - sell_volumes).abs()
    estimated_vpin = abs_diff.rolling(window=num_bars).mean() / volume
    return estimated_vpin

def dollar_volume(tick_prices, tick_volumes):
    """
    Computes total dollar volume

    :param tick_prices: a series of tick prices
    :param tick_volumes: a series of tick_volumes
    :return: total dollar volume
    """
    return (tick_prices * tick_volumes).sum()

"""
A base class for the various bar types. Includes the logic shared between classes, to minimise the amount of
duplicated code.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, Generator, Iterable, Optional

import numpy as np
import pandas as pd

from numba import jit
from numba import float64
from numba import int64


@jit((float64[:], int64), nopython=False, nogil=True)
def ewma(arr_in, window):  # pragma: no cover
    """
    Exponentially weighted moving average specified by a decay ``window`` to provide better adjustments for
    small windows via:
        y[t] = (x[t] + (1-a)*x[t-1] + (1-a)^2*x[t-2] + ... + (1-a)^n*x[t-n]) /
               (1 + (1-a) + (1-a)^2 + ... + (1-a)^n).

    :param arr_in: (np.ndarray), (float64) A single dimensional numpy array
    :param window: (int64) The decay window, or 'span'
    :return: (np.ndarray) The EWMA vector, same length / shape as ``arr_in``
    """
    arr_length = arr_in.shape[0]
    ewma_arr = np.empty(arr_length, dtype=float64)
    alpha = 2 / (window + 1)
    weight = 1
    ewma_old = arr_in[0]
    ewma_arr[0] = ewma_old
    for i in range(1, arr_length):
        weight += (1 - alpha)**i
        ewma_old = ewma_old * (1 - alpha) + arr_in[i]
        ewma_arr[i] = ewma_old / weight

    return ewma_arr
def _crop_data_frame_in_batches(df: pd.DataFrame, chunksize: int) -> list:
    # pylint: disable=invalid-name
    """
    Splits df into chunks of chunksize

    :param df: (pd.DataFrame) Dataframe to split
    :param chunksize: (int) Number of rows in chunk
    :return: (list) Chunks (pd.DataFrames)
    """
    generator_object = []
    for _, chunk in df.groupby(np.arange(len(df)) // chunksize):
        generator_object.append(chunk)
    return generator_object

# pylint: disable=too-many-instance-attributes


class BaseBars(ABC):
    """
    Abstract base class which contains the structure which is shared between the various standard and information
    driven bars. There are some methods contained in here that would only be applicable to information bars but
    they are included here so as to avoid a complicated nested class structure.
    """

    def __init__(self, metric: str, batch_size: int = 2e7):
        """
        Constructor

        :param metric: (str) Type of imbalance bar to create. Example: dollar_imbalance.
        :param batch_size: (int) Number of rows to read in from the csv, per batch.
        """

        # Base properties
        self.metric = metric
        self.batch_size = batch_size
        self.prev_tick_rule = 0

        # Cache properties
        self.open_price, self.prev_price, self.close_price = None, None, None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}
        self.tick_num = 0  # Tick number when bar was formed

        # Batch_run properties
        self.flag = False  # The first flag is false since the first batch doesn't use the cache


    def batch_run(self, file_path_or_df: Union[str, Iterable[str], pd.DataFrame], verbose: bool = True, to_csv: bool = False,
                  output_path: Optional[str] = None) -> Union[pd.DataFrame, None]:
        """
        Reads csv file(s) or pd.DataFrame in batches and then constructs the financial data structure in the form of a DataFrame.
        The csv file or DataFrame must have only 3 columns: date_time, price, & volume.

        :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing
                                raw tick data  in the format[date_time, price, volume]
        :param verbose: (bool) Flag whether to print message on each processed batch or not
        :param to_csv: (bool) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
        :param output_path: (bool) Path to results file, if to_csv = True

        :return: (pd.DataFrame or None) Financial data structure
        """

        if to_csv is True:
            header = True  # if to_csv is True, header should written on the first batch only
            open(output_path, 'w').close()  # clean output csv file

        if verbose:  # pragma: no cover
            print('Reading data in batches:')

        # Read csv in batches
        count = 0
        final_bars = []
        cols = ['date_time', 'tick_num', 'open', 'high', 'low', 'close', 'volume', 'cum_buy_volume', 'cum_ticks',
                'cum_dollar_value']
        for batch in self._batch_iterator(file_path_or_df):
            if verbose:  # pragma: no cover
                print('Batch number:', count)

            list_bars = self.run(data=batch)

            if to_csv is True:
                pd.DataFrame(list_bars, columns=cols).to_csv(output_path, header=header, index=False, mode='a')
                header = False
            else:
                # Append to bars list
                final_bars += list_bars
            count += 1

        if verbose:  # pragma: no cover
            print('Returning bars \n')

        # Return a DataFrame
        if final_bars:
            bars_df = pd.DataFrame(final_bars, columns=cols)
            return bars_df

        # Processed DataFrame is stored in .csv file, return None
        return None

    def _batch_iterator(self, file_path_or_df: Union[str, Iterable[str], pd.DataFrame]) -> Generator[pd.DataFrame, None, None]:
        """
        :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame
                                containing raw tick data in the format[date_time, price, volume]
        """
        if isinstance(file_path_or_df, (list, tuple)):
            # Assert format of all files
            for file_path in file_path_or_df:
                self._read_first_row(file_path)
            for file_path in file_path_or_df:
                for batch in pd.read_csv(file_path, chunksize=self.batch_size, parse_dates=[0]):
                    yield batch

        elif isinstance(file_path_or_df, str):
            self._read_first_row(file_path_or_df)
            for batch in pd.read_csv(file_path_or_df, chunksize=self.batch_size, parse_dates=[0]):
                yield batch

        elif isinstance(file_path_or_df, pd.DataFrame):
            for batch in _crop_data_frame_in_batches(file_path_or_df, self.batch_size):
                yield batch

        else:
            raise ValueError('file_path_or_df is neither string(path to a csv file), iterable of strings, nor pd.DataFrame')

    def _read_first_row(self, file_path: str):
        """
        :param file_path: (str) Path to the csv file containing raw tick data in the format[date_time, price, volume]
        """
        # Read in the first row & assert format
        first_row = pd.read_csv(file_path, nrows=1)
        self._assert_csv(first_row)

    def run(self, data: Union[list, tuple, pd.DataFrame]) -> list:
        """
        Reads a List, Tuple, or Dataframe and then constructs the financial data structure in the form of a list.
        The List, Tuple, or DataFrame must have only 3 attrs: date_time, price, & volume.

        :param data: (list, tuple, or pd.DataFrame) Dict or ndarray containing raw tick data in the format[date_time, price, volume]

        :return: (list) Financial data structure
        """

        if isinstance(data, (list, tuple)):
            values = data

        elif isinstance(data, pd.DataFrame):
            values = data.values

        else:
            raise ValueError('data is neither list nor tuple nor pd.DataFrame')

        list_bars = self._extract_bars(data=values)

        # Set flag to True: notify function to use cache
        self.flag = True

        return list_bars

    @abstractmethod
    def _extract_bars(self, data: pd.DataFrame) -> list:
        """
        This method is required by all the bar types and is used to create the desired bars.

        :param data: (pd.DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (list) Bars built using the current batch.
        """

    @abstractmethod
    def _reset_cache(self):
        """
        This method is required by all the bar types. It describes how cache should be reset
        when new bar is sampled.
        """

    @staticmethod
    def _assert_csv(test_batch: pd.DataFrame):
        """
        Tests that the csv file read has the format: date_time, price, and volume.
        If not then the user needs to create such a file. This format is in place to remove any unwanted overhead.

        :param test_batch: (pd.DataFrame) The first row of the dataset.
        """
        assert test_batch.shape[1] == 3, 'Must have only 3 columns in csv: date_time, price, & volume.'
        assert isinstance(test_batch.iloc[0, 1], float), 'price column in csv not float.'
        assert not isinstance(test_batch.iloc[0, 2], str), 'volume column in csv not int or float.'

        try:
            pd.to_datetime(test_batch.iloc[0, 0])
        except ValueError:
            raise ValueError('csv file, column 0, not a date time format:',
                             test_batch.iloc[0, 0])

    def _update_high_low(self, price: float) -> Union[float, float]:
        """
        Update the high and low prices using the current price.

        :param price: (float) Current price
        :return: (tuple) Updated high and low prices
        """
        if price > self.high_price:
            high_price = price
        else:
            high_price = self.high_price

        if price < self.low_price:
            low_price = price
        else:
            low_price = self.low_price

        return high_price, low_price

    def _create_bars(self, date_time: str, price: float, high_price: float, low_price: float, list_bars: list) -> None:
        """
        Given the inputs, construct a bar which has the following fields: date_time, open, high, low, close, volume,
        cum_buy_volume, cum_ticks, cum_dollar_value.
        These bars are appended to list_bars, which is later used to construct the final bars DataFrame.

        :param date_time: (str) Timestamp of the bar
        :param price: (float) The current price
        :param high_price: (float) Highest price in the period
        :param low_price: (float) Lowest price in the period
        :param list_bars: (list) List to which we append the bars
        """
        # Create bars
        open_price = self.open_price
        high_price = max(high_price, open_price)
        low_price = min(low_price, open_price)
        close_price = price
        volume = self.cum_statistics['cum_volume']
        cum_buy_volume = self.cum_statistics['cum_buy_volume']
        cum_ticks = self.cum_statistics['cum_ticks']
        cum_dollar_value = self.cum_statistics['cum_dollar_value']

        # Update bars
        list_bars.append(
            [date_time, self.tick_num, open_price, high_price, low_price, close_price, volume, cum_buy_volume,
             cum_ticks,
             cum_dollar_value])

    def _apply_tick_rule(self, price: float) -> int:
        """
        Applies the tick rule as defined on page 29 of Advances in Financial Machine Learning.

        :param price: (float) Price at time t
        :return: (int) The signed tick
        """
        if self.prev_price is not None:
            tick_diff = price - self.prev_price
        else:
            tick_diff = 0

        if tick_diff != 0:
            signed_tick = np.sign(tick_diff)
            self.prev_tick_rule = signed_tick
        else:
            signed_tick = self.prev_tick_rule

        self.prev_price = price  # Update previous price used for tick rule calculations
        return signed_tick

    def _get_imbalance(self, price: float, signed_tick: int, volume: float) -> float:
        """
        Advances in Financial Machine Learning, page 29.

        Get the imbalance at a point in time, denoted as Theta_t

        :param price: (float) Price at t
        :param signed_tick: (int) signed tick, using the tick rule
        :param volume: (float) Volume traded at t
        :return: (float) Imbalance at time t
        """
        if self.metric == 'tick_imbalance' or self.metric == 'tick_run':
            imbalance = signed_tick
        elif self.metric == 'dollar_imbalance' or self.metric == 'dollar_run':
            imbalance = signed_tick * volume * price
        elif self.metric == 'volume_imbalance' or self.metric == 'volume_run':
            imbalance = signed_tick * volume
        else:
            raise ValueError('Unknown imbalance metric, possible values are tick/dollar/volume imbalance/run')
        return imbalance


class BaseImbalanceBars(BaseBars):
    """
    Base class for Imbalance Bars (EMA and Const) which implements imbalance bars calculation logic
    """

    def __init__(self, metric: str, batch_size: int,
                 expected_imbalance_window: int, exp_num_ticks_init: int,
                 analyse_thresholds: bool):
        """
        Constructor

        :param metric: (str) Type of imbalance bar to create. Example: dollar_imbalance.
        :param batch_size: (int) Number of rows to read in from the csv, per batch.
        :param expected_imbalance_window: (int) Window used to estimate expected imbalance from previous trades
        :param exp_num_ticks_init: (int) Initial estimate for expected number of ticks in bar.
                                         For Const Imbalance Bars expected number of ticks equals expected number of ticks init
        :param analyse_thresholds: (bool) Flag to return thresholds values (theta, exp_num_ticks, exp_imbalance) in a
                                          form of Pandas DataFrame
        """
        BaseBars.__init__(self, metric, batch_size)

        self.expected_imbalance_window = expected_imbalance_window

        self.thresholds = {'cum_theta': 0, 'expected_imbalance': np.nan, 'exp_num_ticks': exp_num_ticks_init}

        # Previous bars number of ticks and previous tick imbalances
        self.imbalance_tick_statistics = {'num_ticks_bar': [], 'imbalance_array': []}

        if analyse_thresholds is True:
            # Array of dicts: {'timestamp': value, 'cum_theta': value, 'exp_num_ticks': value, 'exp_imbalance': value}
            self.bars_thresholds = []
        else:
            self.bars_thresholds = None

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for imbalance bars
        """
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}
        self.thresholds['cum_theta'] = 0

    def _extract_bars(self, data: Tuple[dict, pd.DataFrame]) -> list:
        """
        For loop which compiles the various imbalance bars: dollar, volume, or tick.

        :param data: (pd.DataFrame) Contains 3 columns - date_time, price, and volume.
        :return: (list) Bars built using the current batch.
        """

        # Iterate over rows
        list_bars = []
        for row in data:
            # Set variables
            date_time = row[0]
            self.tick_num += 1
            price = np.float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            if self.open_price is None:
                self.open_price = price

            # Update high low prices
            self.high_price, self.low_price = self._update_high_low(price)

            # Bar statistics calculations
            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_dollar_value'] += dollar_value
            self.cum_statistics['cum_volume'] += volume
            if signed_tick == 1:
                self.cum_statistics['cum_buy_volume'] += volume

            # Imbalance calculations
            imbalance = self._get_imbalance(price, signed_tick, volume)
            self.imbalance_tick_statistics['imbalance_array'].append(imbalance)
            self.thresholds['cum_theta'] += imbalance

            # Get expected imbalance for the first time, when num_ticks_init passed
            if not list_bars and np.isnan(self.thresholds['expected_imbalance']):
                self.thresholds['expected_imbalance'] = self._get_expected_imbalance(
                    self.expected_imbalance_window)

            if self.bars_thresholds is not None:
                self.thresholds['timestamp'] = date_time
                self.bars_thresholds.append(dict(self.thresholds))

            # Check expression for possible bar generation
            if (np.abs(self.thresholds['cum_theta']) > self.thresholds['exp_num_ticks'] * np.abs(
                    self.thresholds['expected_imbalance']) if ~np.isnan(self.thresholds['expected_imbalance']) else False):
                self._create_bars(date_time, price,
                                  self.high_price, self.low_price, list_bars)

                self.imbalance_tick_statistics['num_ticks_bar'].append(self.cum_statistics['cum_ticks'])
                # Expected number of ticks based on formed bars
                self.thresholds['exp_num_ticks'] = self._get_exp_num_ticks()
                # Get expected imbalance
                self.thresholds['expected_imbalance'] = self._get_expected_imbalance(
                    self.expected_imbalance_window)
                # Reset counters
                self._reset_cache()

        return list_bars

    def _get_expected_imbalance(self, window: int):
        """
        Calculate the expected imbalance: 2P[b_t=1]-1, using a EWMA, pg 29
        :param window: (int) EWMA window for calculation
        :return: expected_imbalance: (np.ndarray) 2P[b_t=1]-1, approximated using a EWMA
        """
        if len(self.imbalance_tick_statistics['imbalance_array']) < self.thresholds['exp_num_ticks']:
            # Waiting for array to fill for ewma
            ewma_window = np.nan
        else:
            # ewma window can be either the window specified in a function call
            # or it is len of imbalance_array if window > len(imbalance_array)
            ewma_window = int(min(len(self.imbalance_tick_statistics['imbalance_array']), window))

        if np.isnan(ewma_window):
            # return nan, wait until len(self.imbalance_array) >= self.exp_num_ticks_init
            expected_imbalance = np.nan
        else:
            expected_imbalance = ewma(
                np.array(self.imbalance_tick_statistics['imbalance_array'][-ewma_window:], dtype=float),
                window=ewma_window)[-1]

        return expected_imbalance

    @abstractmethod
    def _get_exp_num_ticks(self):
        """
        Abstract method which updates expected number of ticks when new run bar is formed
        """


# pylint: disable=too-many-instance-attributes
class BaseRunBars(BaseBars):
    """
    Base class for Run Bars (EMA and Const) which implements run bars calculation logic
    """

    def __init__(self, metric: str, batch_size: int, num_prev_bars: int,
                 expected_imbalance_window: int,
                 exp_num_ticks_init: int, analyse_thresholds: bool):
        """
        Constructor

        :param metric: (str) Type of imbalance bar to create. Example: dollar_imbalance.
        :param batch_size: (int) Number of rows to read in from the csv, per batch.
        :param expected_imbalance_window: (int) Window used to estimate expected imbalance from previous trades
        :param exp_num_ticks_init: (int) Initial estimate for expected number of ticks in bar.
                                         For Const Imbalance Bars expected number of ticks equals expected number of ticks init
        :param analyse_thresholds: (bool) Flag to return thresholds values (thetas, exp_num_ticks, exp_runs) in Pandas DataFrame
        """
        BaseBars.__init__(self, metric, batch_size)

        self.num_prev_bars = num_prev_bars
        self.expected_imbalance_window = expected_imbalance_window

        self.thresholds = {'cum_theta_buy': 0, 'cum_theta_sell': 0, 'exp_imbalance_buy': np.nan,
                           'exp_imbalance_sell': np.nan, 'exp_num_ticks': exp_num_ticks_init,
                           'exp_buy_ticks_proportion': np.nan, 'buy_ticks_num': 0}

        # Previous bars number of ticks and previous tick imbalances
        self.imbalance_tick_statistics = {'num_ticks_bar': [], 'imbalance_array_buy': [], 'imbalance_array_sell': [],
                                          'buy_ticks_proportion': []}

        if analyse_thresholds:
            # Array of dicts: {'timestamp': value, 'cum_theta': value, 'exp_num_ticks': value, 'exp_imbalance': value}
            self.bars_thresholds = []
        else:
            self.bars_thresholds = None

        self.warm_up_flag = False

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for imbalance bars
        """
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}
        self.thresholds['cum_theta_buy'], self.thresholds['cum_theta_sell'], self.thresholds['buy_ticks_num'] = 0, 0, 0

    def _extract_bars(self, data: Tuple[list, np.ndarray]) -> list:
        """
        For loop which compiles the various run bars: dollar, volume, or tick.

        :param data: (list or np.ndarray) Contains 3 columns - date_time, price, and volume.
        :return: (list) of bars built using the current batch.
        """

        # Iterate over rows
        list_bars = []
        for row in data:
            # Set variables
            date_time = row[0]
            self.tick_num += 1
            price = np.float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            if self.open_price is None:
                self.open_price = price

            # Update high low prices
            self.high_price, self.low_price = self._update_high_low(price)

            # Bar statistics calculations
            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_dollar_value'] += dollar_value
            self.cum_statistics['cum_volume'] += volume
            if signed_tick == 1:
                self.cum_statistics['cum_buy_volume'] += volume

            # Imbalance calculations
            imbalance = self._get_imbalance(price, signed_tick, volume)

            if imbalance > 0:
                self.imbalance_tick_statistics['imbalance_array_buy'].append(imbalance)
                self.thresholds['cum_theta_buy'] += imbalance
                self.thresholds['buy_ticks_num'] += 1
            elif imbalance < 0:
                self.imbalance_tick_statistics['imbalance_array_sell'].append(abs(imbalance))
                self.thresholds['cum_theta_sell'] += abs(imbalance)

            self.warm_up_flag = np.isnan([self.thresholds['exp_imbalance_buy'], self.thresholds[
                'exp_imbalance_sell']]).any()  # Flag indicating that one of imbalances is not counted (warm-up)

            # Get expected imbalance for the first time, when num_ticks_init passed
            if not list_bars and self.warm_up_flag:
                self.thresholds['exp_imbalance_buy'] = self._get_expected_imbalance(
                    self.imbalance_tick_statistics['imbalance_array_buy'], self.expected_imbalance_window, warm_up=True)
                self.thresholds['exp_imbalance_sell'] = self._get_expected_imbalance(
                    self.imbalance_tick_statistics['imbalance_array_sell'], self.expected_imbalance_window,
                    warm_up=True)

                if bool(np.isnan([self.thresholds['exp_imbalance_buy'],
                                  self.thresholds['exp_imbalance_sell']]).any()) is False:
                    self.thresholds['exp_buy_ticks_proportion'] = self.thresholds['buy_ticks_num'] / \
                                                                  self.cum_statistics[
                                                                      'cum_ticks']

            if self.bars_thresholds is not None:
                self.thresholds['timestamp'] = date_time
                self.bars_thresholds.append(dict(self.thresholds))

            # Check expression for possible bar generation
            max_proportion = max(
                self.thresholds['exp_imbalance_buy'] * self.thresholds['exp_buy_ticks_proportion'],
                self.thresholds['exp_imbalance_sell'] * (1 - self.thresholds['exp_buy_ticks_proportion']))

            # Check expression for possible bar generation
            max_theta = max(self.thresholds['cum_theta_buy'], self.thresholds['cum_theta_sell'])
            if max_theta > self.thresholds['exp_num_ticks'] * max_proportion and not np.isnan(max_proportion):
                self._create_bars(date_time, price, self.high_price, self.low_price, list_bars)

                self.imbalance_tick_statistics['num_ticks_bar'].append(self.cum_statistics['cum_ticks'])
                self.imbalance_tick_statistics['buy_ticks_proportion'].append(
                    self.thresholds['buy_ticks_num'] / self.cum_statistics['cum_ticks'])

                # Expected number of ticks based on formed bars
                self.thresholds['exp_num_ticks'] = self._get_exp_num_ticks()

                # Expected buy ticks proportion based on formed bars
                exp_buy_ticks_proportion = ewma(
                    np.array(self.imbalance_tick_statistics['buy_ticks_proportion'][-self.num_prev_bars:], dtype=float),
                    self.num_prev_bars)[-1]
                self.thresholds['exp_buy_ticks_proportion'] = exp_buy_ticks_proportion

                # Get expected imbalance
                self.thresholds['exp_imbalance_buy'] = self._get_expected_imbalance(
                    self.imbalance_tick_statistics['imbalance_array_buy'], self.expected_imbalance_window)
                self.thresholds['exp_imbalance_sell'] = self._get_expected_imbalance(
                    self.imbalance_tick_statistics['imbalance_array_sell'], self.expected_imbalance_window)

                # Reset counters
                self._reset_cache()

        return list_bars

    def _get_expected_imbalance(self, array: list, window: int, warm_up: bool = False):
        """
        Advances in Financial Machine Learning, page 29.

        Calculates the expected imbalance: 2P[b_t=1]-1, using a EWMA.

        :param array: (list) of imbalances
        :param window: (int) EWMA window for calculation
        :parawm warm_up: (bool) flag of whether warm up period passed
        :return: expected_imbalance: (np.ndarray) 2P[b_t=1]-1, approximated using a EWMA
        """
        if len(array) < self.thresholds['exp_num_ticks'] and warm_up is True:
            # Waiting for array to fill for ewma
            ewma_window = np.nan
        else:
            # ewma window can be either the window specified in a function call
            # or it is len of imbalance_array if window > len(imbalance_array)
            ewma_window = int(min(len(array), window))

        if np.isnan(ewma_window):
            # return nan, wait until len(self.imbalance_array) >= self.exp_num_ticks_init
            expected_imbalance = np.nan
        else:
            expected_imbalance = ewma(
                np.array(array[-ewma_window:], dtype=float),
                window=ewma_window)[-1]

        return expected_imbalance

    @abstractmethod
    def _get_exp_num_ticks(self):
        """
        Abstract method which updates expected number of ticks when new imbalance bar is formed
        """

# Snippet 20.5 (page 306), the lin_parts function
def lin_parts(num_atoms, num_threads):
    """
    Advances in Financial Machine Learning, Snippet 20.5, page 306.

    The lin_parts function

    The simplest way to form molecules is to partition a list of atoms in subsets of equal size,
    where the number of subsets is the minimum between the number of processors and the number
    of atoms. For N subsets we need to find the N+1 indices that enclose the partitions.
    This logic is demonstrated in Snippet 20.5.

    This function partitions a list of atoms in subsets (molecules) of equal size.
    An atom is a set of indivisible set of tasks.

    :param num_atoms: (int) Number of atoms
    :param num_threads: (int) Number of processors
    :return: (np.array) Partition of atoms
    """
    # Partition of atoms with a single loop
    parts = np.linspace(0, num_atoms, min(num_threads, num_atoms) + 1)
    parts = np.ceil(parts).astype(int)
    return parts


# Snippet 20.6 (page 308), The nested_parts function
def nested_parts(num_atoms, num_threads, upper_triangle=False):
    """
    Advances in Financial Machine Learning, Snippet 20.6, page 308.

    The nested_parts function

    This function enables parallelization of nested loops.
    :param num_atoms: (int) Number of atoms
    :param num_threads: (int) Number of processors
    :param upper_triangle: (bool) Flag to order atoms as an upper triangular matrix (including the main diagonal)
    :return: (np.array) Partition of atoms
    """
    # Partition of atoms with an inner loop
    parts = [0]
    num_threads_ = min(num_threads, num_atoms)

    for _ in range(num_threads_):
        part = 1 + 4 * (parts[-1] ** 2 + parts[-1] + num_atoms * (num_atoms + 1.0) / num_threads_)
        part = (-1 + part ** 0.5) / 2.0
        parts.append(part)

    parts = np.round(parts).astype(int)

    if upper_triangle:  # The first rows are heaviest
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)

    return parts


# Snippet 20.7 (page 310), The mpPandasObj, used at various points in the book
def mp_pandas_obj(func, pd_obj, num_threads=24, mp_batches=1, lin_mols=True, verbose=True, **kargs):
    """
    Advances in Financial Machine Learning, Snippet 20.7, page 310.

    The mpPandasObj, used at various points in the book

    Parallelize jobs, return a dataframe or series.
    Example: df1=mp_pandas_obj(func,('molecule',df0.index),24,**kwds)

    First, atoms are grouped into molecules, using linParts (equal number of atoms per molecule)
    or nestedParts (atoms distributed in a lower-triangular structure). When mpBatches is greater
    than 1, there will be more molecules than cores. Suppose that we divide a task into 10 molecules,
    where molecule 1 takes twice as long as the rest. If we run this process in 10 cores, 9 of the
    cores will be idle half of the runtime, waiting for the first core to process molecule 1.
    Alternatively, we could set mpBatches=10 so as to divide that task in 100 molecules. In doing so,
    every core will receive equal workload, even though the first 10 molecules take as much time as the
    next 20 molecules. In this example, the run with mpBatches=10 will take half of the time consumed by
    mpBatches=1.

    Second, we form a list of jobs. A job is a dictionary containing all the information needed to process
    a molecule, that is, the callback function, its keyword arguments, and the subset of atoms that form
    the molecule.

    Third, we will process the jobs sequentially if numThreads==1 (see Snippet 20.8), and in parallel
    otherwise (see Section 20.5.2). The reason that we want the option to run jobs sequentially is for
    debugging purposes. It is not easy to catch a bug when programs are run in multiple processors.
    Once the code is debugged, we will want to use numThreads>1.

    Fourth, we stitch together the output from every molecule into a single list, series, or dataframe.

    :param func: (function) A callback function, which will be executed in parallel
    :param pd_obj: (tuple) Element 0: The name of the argument used to pass molecules to the callback function
                    Element 1: A list of indivisible tasks (atoms), which will be grouped into molecules
    :param num_threads: (int) The number of threads that will be used in parallel (one processor per thread)
    :param mp_batches: (int) Number of parallel batches (jobs per core)
    :param lin_mols: (bool) Tells if the method should use linear or nested partitioning
    :param verbose: (bool) Flag to report progress on asynch jobs
    :param kargs: (var args) Keyword arguments needed by func
    :return: (pd.DataFrame) of results
    """

    if lin_mols:
        parts = lin_parts(len(pd_obj[1]), num_threads * mp_batches)
    else:
        parts = nested_parts(len(pd_obj[1]), num_threads * mp_batches)

    jobs = []
    for i in range(1, len(parts)):
        job = {pd_obj[0]: pd_obj[1][parts[i - 1]:parts[i]], 'func': func}
        job.update(kargs)
        jobs.append(job)

    if num_threads == 1:
        out = process_jobs_(jobs)
    else:
        out = process_jobs(jobs, num_threads=num_threads, verbose=verbose)

    if isinstance(out[0], pd.DataFrame):
        df0 = pd.DataFrame()
    elif isinstance(out[0], pd.Series):
        df0 = pd.Series(dtype='float64')
    else:
        return out

    for i in out:
        df0 = df0.append(i)

    df0 = df0.sort_index()
    return df0


# Snippet 20.8, pg 311, Single thread execution, for debugging
def process_jobs_(jobs):
    """
    Advances in Financial Machine Learning, Snippet 20.8, page 311.

    Single thread execution, for debugging

    Run jobs sequentially, for debugging

    :param jobs: (list) Jobs (molecules)
    :return: (list) Results of jobs
    """
    out = []
    for job in jobs:
        out_ = expand_call(job)
        out.append(out_)

    return out


# Snippet 20.10 Passing the job (molecule) to the callback function
def expand_call(kargs):
    """
    Advances in Financial Machine Learning, Snippet 20.10.

    Passing the job (molecule) to the callback function

    Expand the arguments of a callback function, kargs['func']

    :param kargs: Job (molecule)
    :return: Result of a job
    """
    func = kargs['func']
    del kargs['func']
    out = func(**kargs)
    return out


# Snippet 20.9.1, pg 312, Example of Asynchronous call to pythons multiprocessing library
def report_progress(job_num, num_jobs, time0, task):
    """
    Advances in Financial Machine Learning, Snippet 20.9.1, pg 312.

    Example of Asynchronous call to pythons multiprocessing library

    :param job_num: (int) Number of current job
    :param num_jobs: (int) Total number of jobs
    :param time0: (time) Start time
    :param task: (str) Task description
    :return: (None)
    """
    # Report progress as asynch jobs are completed
    msg = [float(job_num) / num_jobs, (time.time() - time0) / 60.0]
    msg.append(msg[1] * (1 / msg[0] - 1))
    time_stamp = str(dt.datetime.fromtimestamp(time.time()))

    msg = time_stamp + ' ' + str(round(msg[0] * 100, 2)) + '% ' + task + ' done after ' + \
          str(round(msg[1], 2)) + ' minutes. Remaining ' + str(round(msg[2], 2)) + ' minutes.'

    if job_num < num_jobs:
        sys.stderr.write(msg + '\r')
    else:
        sys.stderr.write(msg + '\n')


# Snippet 20.9.2, pg 312, Example of Asynchronous call to pythons multiprocessing library
def process_jobs(jobs, task=None, num_threads=24, verbose=True):
    """
    Advances in Financial Machine Learning, Snippet 20.9.2, page 312.

    Example of Asynchronous call to pythons multiprocessing library

    Run in parallel. jobs must contain a 'func' callback, for expand_call

    :param jobs: (list) Jobs (molecules)
    :param task: (str) Task description
    :param num_threads: (int) The number of threads that will be used in parallel (one processor per thread)
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (None)
    """

    if task is None:
        task = jobs[0]['func'].__name__

    pool = mp.Pool(processes=num_threads)
    outputs = pool.imap_unordered(expand_call, jobs)
    out = []
    time0 = time.time()

    # Process asynchronous output, report progress
    for i, out_ in enumerate(outputs, 1):
        out.append(out_)
        if verbose:
            report_progress(i, len(jobs), time0, task)

    pool.close()
    pool.join()  # This is needed to prevent memory leaks
    return out

def get_roll_measure(close_prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, page 282.

    Get Roll Measure

    Roll Measure gives the estimate of effective bid-ask spread
    without using quote-data.

    :param close_prices: (pd.Series) Close prices
    :param window: (int) Estimation window
    :return: (pd.Series) Roll measure
    """
    price_diff = close_prices.diff()
    price_diff_lag = price_diff.shift(1)
    return 2 * np.sqrt(abs(price_diff.rolling(window=window).cov(price_diff_lag)))


def get_roll_impact(close_prices: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Get Roll Impact.

    Derivate from Roll Measure which takes into account dollar volume traded.

    :param close_prices: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volume series
    :param window: (int) Estimation window
    :return: (pd.Series) Roll impact
    """
    roll_measure = get_roll_measure(close_prices, window)
    return roll_measure / dollar_volume


# Corwin-Schultz algorithm
def _get_beta(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get beta estimate from Corwin-Schultz algorithm

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) Estimation window
    :return: (pd.Series) Beta estimates
    """
    ret = np.log(high / low)
    high_low_ret = ret ** 2
    beta = high_low_ret.rolling(window=2).sum()
    beta = beta.rolling(window=window).mean()
    return beta


def _get_gamma(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get gamma estimate from Corwin-Schultz algorithm.

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :return: (pd.Series) Gamma estimates
    """
    high_max = high.rolling(window=2).max()
    low_min = low.rolling(window=2).min()
    gamma = np.log(high_max / low_min) ** 2
    return gamma


def _get_alpha(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get alpha from Corwin-Schultz algorithm.

    :param beta: (pd.Series) Beta estimates
    :param gamma: (pd.Series) Gamma estimates
    :return: (pd.Series) Alphas
    """
    den = 3 - 2 * 2 ** .5
    alpha = (2 ** .5 - 1) * (beta ** .5) / den
    alpha -= (gamma / den) ** .5
    alpha[alpha < 0] = 0  # Set negative alphas to 0 (see p.727 of paper)
    return alpha


def get_corwin_schultz_estimator(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get Corwin-Schultz spread estimator using high-low prices

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) Estimation window
    :return: (pd.Series) Corwin-Schultz spread estimators
    """
    # Note: S<0 iif alpha<0
    beta = _get_beta(high, low, window)
    gamma = _get_gamma(high, low)
    alpha = _get_alpha(beta, gamma)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    start_time = pd.Series(high.index[0:spread.shape[0]], index=spread.index)
    spread = pd.concat([spread, start_time], axis=1)
    spread.columns = ['Spread', 'Start_Time']  # 1st loc used to compute beta
    return spread.Spread


def get_bekker_parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.2, page 286.

    Get Bekker-Parkinson volatility from gamma and beta in Corwin-Schultz algorithm.

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) Estimation window
    :return: (pd.Series) Bekker-Parkinson volatility estimates
    """
    # pylint: disable=invalid-name
    beta = _get_beta(high, low, window)
    gamma = _get_gamma(high, low)

    k2 = (8 / np.pi) ** 0.5
    den = 3 - 2 * 2 ** .5
    sigma = (2 ** -0.5 - 1) * beta ** 0.5 / (k2 * den)
    sigma += (gamma / (k2 ** 2 * den)) ** 0.5
    sigma[sigma < 0] = 0
    return sigma

def encode_tick_rule_array(tick_rule_array: list) -> str:
    """
    Encode array of tick signs (-1, 1, 0)

    :param tick_rule_array: (list) Tick rules
    :return: (str) Encoded message
    """
    message = ''
    for element in tick_rule_array:
        if element == 1:
            message += 'a'
        elif element == -1:
            message += 'b'
        elif element == 0:
            message += 'c'
        else:
            raise ValueError('Unknown value for tick rule: {}'.format(element))
    return message


def _get_ascii_table() -> list:
    """
    Get all ASCII symbols

    :return: (list) ASCII symbols
    """
    # ASCII table consists of 256 characters
    table = []
    for i in range(256):
        table.append(chr(i))
    return table


def quantile_mapping(array: list, num_letters: int = 26) -> dict:
    """
    Generate dictionary of quantile-letters based on values from array and dictionary length (num_letters).

    :param array: (list) Values to split on quantiles
    :param num_letters: (int) Number of letters(quantiles) to encode
    :return: (dict) Dict of quantile-symbol
    """
    encoding_dict = {}
    ascii_table = _get_ascii_table()
    alphabet = ascii_table[:num_letters]
    for quant, letter in zip(np.linspace(0.01, 1, len(alphabet)), alphabet):
        encoding_dict[np.quantile(array, quant)] = letter
    return encoding_dict


def sigma_mapping(array: list, step: float = 0.01) -> dict:
    """
    Generate dictionary of sigma encoded letters based on values from array and discretization step.

    :param array: (list) Values to split on quantiles
    :param step: (float) Discretization step (sigma)
    :return: (dict) Dict of value-symbol
    """
    i = 0
    ascii_table = _get_ascii_table()
    encoding_dict = {}
    encoding_steps = np.arange(min(array), max(array), step)
    for element in encoding_steps:
        try:
            encoding_dict[element] = ascii_table[i]
        except IndexError:
            raise ValueError(
                'Length of dictionary ceil((max(arr) - min(arr)) / step = {} is more than ASCII table lenght)'.format(
                    len(encoding_steps)))
        i += 1
    return encoding_dict


def _find_nearest(array: list, value: float) -> float:
    """
    Find the nearest element from array to value.

    :param array: (list) Values
    :param value: (float) Value for which the nearest element needs to be found
    :return: (float) The nearest to the value element in array
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def _get_letter_from_encoding(value: float, encoding_dict: dict) -> str:
    """
    Get letter for float/int value from encoding dict.

    :param value: (float/int) Value to use
    :param encoding_dict: (dict) Used dictionary
    :return: (str) Letter from encoding dict
    """
    return encoding_dict[_find_nearest(list(encoding_dict.keys()), value)]


def encode_array(array: list, encoding_dict: dict) -> str:
    """
    Encode array with strings using encoding dict, in case of multiple occurrences of the minimum values,
    the indices corresponding to the first occurrence are returned

    :param array: (list) Values to encode
    :param encoding_dict: (dict) Dict of quantile-symbol
    :return: (str) Encoded message
    """
    message = ''
    for element in array:
        message += _get_letter_from_encoding(element, encoding_dict)
    return message


# pylint: disable=invalid-name
def get_bar_based_kyle_lambda(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, p. 286-288.

    Get Kyle lambda from bars data

    :param close: (pd.Series) Close prices
    :param volume: (pd.Series) Bar volume
    :param window: (int) Rolling window used for estimation
    :return: (pd.Series) Kyle lambdas
    """
    close_diff = close.diff()
    close_diff_sign = close_diff.apply(np.sign)
    close_diff_sign.replace(0, method='pad', inplace=True)  # Replace 0 values with previous
    volume_mult_trade_signs = volume * close_diff_sign  # bt * Vt
    return (close_diff / volume_mult_trade_signs).rolling(window=window).mean()


def get_bar_based_amihud_lambda(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, p.288-289.

    Get Amihud lambda from bars data

    :param close: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volumes
    :param window: (int) rolling window used for estimation
    :return: (pd.Series) of Amihud lambda
    """
    returns_abs = np.log(close / close.shift(1)).abs()
    return (returns_abs / dollar_volume).rolling(window=window).mean()


def get_bar_based_hasbrouck_lambda(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, p.289-290.

    Get Hasbrouck lambda from bars data

    :param close: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volumes
    :param window: (int) Rolling window used for estimation
    :return: (pd.Series) Hasbrouck lambda
    """
    log_ret = np.log(close / close.shift(1))
    log_ret_sign = log_ret.apply(np.sign).replace(0, method='pad')

    signed_dollar_volume_sqrt = log_ret_sign * np.sqrt(dollar_volume)
    return (log_ret / signed_dollar_volume_sqrt).rolling(window=window).mean()


def get_trades_based_kyle_lambda(price_diff: list, volume: list, aggressor_flags: list) -> List[float]:
    """
    Advances in Financial Machine Learning, p.286-288.

    Get Kyle lambda from trades data

    :param price_diff: (list) Price diffs
    :param volume: (list) Trades sizes
    :param aggressor_flags: (list) Trade directions [-1, 1]  (tick rule or aggressor side can be used to define)
    :return: (list) Kyle lambda for a bar and t-value
    """
    signed_volume = np.array(volume) * np.array(aggressor_flags)
    X = np.array(signed_volume).reshape(-1, 1)
    y = np.array(price_diff)
    coef, std = get_betas(X, y)
    t_value = coef[0] / std[0] if std[0] > 0 else np.array([0])
    return [coef[0], t_value[0]]


def get_trades_based_amihud_lambda(log_ret: list, dollar_volume: list) -> List[float]:
    """
    Advances in Financial Machine Learning, p.288-289.

    Get Amihud lambda from trades data

    :param log_ret: (list) Log returns
    :param dollar_volume: (list) Dollar volumes (price * size)
    :return: (float) Amihud lambda for a bar
    """
    X = np.array(dollar_volume).reshape(-1, 1)
    y = np.abs(np.array(log_ret))
    coef, std = get_betas(X, y)
    t_value = coef[0] / std[0] if std[0] > 0 else np.array([0])
    return [coef[0], t_value[0]]


def get_trades_based_hasbrouck_lambda(log_ret: list, dollar_volume: list, aggressor_flags: list) -> List[float]:
    """
    Advances in Financial Machine Learning, p.289-290.

    Get Hasbrouck lambda from trades data

    :param log_ret: (list) Log returns
    :param dollar_volume: (list) Dollar volumes (price * size)
    :param aggressor_flags: (list) Trade directions [-1, 1]  (tick rule or aggressor side can be used to define)
    :return: (list) Hasbrouck lambda for a bar and t value
    """
    X = (np.sqrt(np.array(dollar_volume)) * np.array(aggressor_flags)).reshape(-1, 1)
    y = np.abs(np.array(log_ret))
    coef, std = get_betas(X, y)
    t_value = coef[0] / std[0] if std[0] > 0 else np.array([0])
    return [coef[0], t_value[0]]


# pylint: disable=invalid-name

def _get_sadf_at_t(X: pd.DataFrame, y: pd.DataFrame, min_length: int, model: str, phi: float) -> float:
    """
    Advances in Financial Machine Learning, Snippet 17.2, page 258.

    SADF's Inner Loop (get SADF value at t)

    :param X: (pd.DataFrame) Lagged values, constants, trend coefficients
    :param y: (pd.DataFrame) Y values (either y or y.diff())
    :param min_length: (int) Minimum number of samples needed for estimation
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param phi: (float) Coefficient to penalize large sample lengths when computing SMT, in [0, 1]
    :return: (float) SADF statistics for y.index[-1]
    """
    start_points, bsadf = range(0, y.shape[0] - min_length + 1), -np.inf
    for start in start_points:
        y_, X_ = y[start:], X[start:]
        b_mean_, b_std_ = get_betas(X_, y_)
        if not np.isnan(b_mean_[0]):
            b_mean_, b_std_ = b_mean_[0, 0], b_std_[0, 0] ** 0.5
            # TODO: Rewrite logic of this module to avoid division by zero
            with np.errstate(invalid='ignore'):
                all_adf = b_mean_ / b_std_
            if model[:2] == 'sm':
                all_adf = np.abs(all_adf) / (y.shape[0]**phi)
            if all_adf > bsadf:
                bsadf = all_adf
    return bsadf


def _get_y_x(series: pd.Series, model: str, lags: Union[int, list],
             add_const: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Advances in Financial Machine Learning, Snippet 17.2, page 258-259.

    Preparing The Datasets

    :param series: (pd.Series) Series to prepare for test statistics generation (for example log prices)
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param lags: (int or list) Either number of lags to use or array of specified lags
    :param add_const: (bool) Flag to add constant
    :return: (pd.DataFrame, pd.DataFrame) Prepared y and X for SADF generation
    """
    series = pd.DataFrame(series)
    series_diff = series.diff().dropna()
    x = _lag_df(series_diff, lags).dropna()
    x['y_lagged'] = series.shift(1).loc[x.index]  # add y_(t-1) column
    y = series_diff.loc[x.index]

    if add_const is True:
        x['const'] = 1

    if model == 'linear':
        x['trend'] = np.arange(x.shape[0])  # Add t to the model (0, 1, 2, 3, 4, 5, .... t)
        beta_column = 'y_lagged'  # Column which is used to estimate test beta statistics
    elif model == 'quadratic':
        x['trend'] = np.arange(x.shape[0]) # Add t to the model (0, 1, 2, 3, 4, 5, .... t)
        x['quad_trend'] = np.arange(x.shape[0]) ** 2 # Add t^2 to the model (0, 1, 4, 9, ....)
        beta_column = 'y_lagged'  # Column which is used to estimate test beta statistics
    elif model == 'sm_poly_1':
        y = series.loc[y.index]
        x = pd.DataFrame(index=y.index)
        x['const'] = 1
        x['trend'] = np.arange(x.shape[0])
        x['quad_trend'] = np.arange(x.shape[0]) ** 2
        beta_column = 'quad_trend'
    elif model == 'sm_poly_2':
        y = np.log(series.loc[y.index])
        x = pd.DataFrame(index=y.index)
        x['const'] = 1
        x['trend'] = np.arange(x.shape[0])
        x['quad_trend'] = np.arange(x.shape[0]) ** 2
        beta_column = 'quad_trend'
    elif model == 'sm_exp':
        y = np.log(series.loc[y.index])
        x = pd.DataFrame(index=y.index)
        x['const'] = 1
        x['trend'] = np.arange(x.shape[0])
        beta_column = 'trend'
    elif model == 'sm_power':
        y = np.log(series.loc[y.index])
        x = pd.DataFrame(index=y.index)
        x['const'] = 1
        # TODO: Rewrite logic of this module to avoid division by zero
        with np.errstate(divide='ignore'):
            x['log_trend'] = np.log(np.arange(x.shape[0]))
        beta_column = 'log_trend'
    else:
        raise ValueError('Unknown model')

    # Move y_lagged column to the front for further extraction
    columns = list(x.columns)
    columns.insert(0, columns.pop(columns.index(beta_column)))
    x = x[columns]
    return x, y


def _lag_df(df: pd.DataFrame, lags: Union[int, list]) -> pd.DataFrame:
    """
    Advances in Financial Machine Learning, Snipet 17.3, page 259.

    Apply Lags to DataFrame

    :param df: (int or list) Either number of lags to use or array of specified lags
    :param lags: (int or list) Lag(s) to use
    :return: (pd.DataFrame) Dataframe with lags
    """
    df_lagged = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(1, lags + 1)
    else:
        lags = [int(lag) for lag in lags]

    for lag in lags:
        temp_df = df.shift(lag).copy(deep=True)
        temp_df.columns = [str(i) + '_' + str(lag) for i in temp_df.columns]
        df_lagged = df_lagged.join(temp_df, how='outer')
    return df_lagged


def get_betas(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[np.array, np.array]:
    """
    Advances in Financial Machine Learning, Snippet 17.4, page 259.

    Fitting The ADF Specification (get beta estimate and estimate variance)

    :param X: (pd.DataFrame) Features(factors)
    :param y: (pd.DataFrame) Outcomes
    :return: (np.array, np.array) Betas and variances of estimates
    """
    xy = np.dot(X.T, y)
    xx = np.dot(X.T, X)

    try:
        xx_inv = np.linalg.inv(xx)
    except np.linalg.LinAlgError:
        return [np.nan], [[np.nan, np.nan]]

    b_mean = np.dot(xx_inv, xy)
    err = y - np.dot(X, b_mean)
    b_var = np.dot(err.T, err) / (X.shape[0] - X.shape[1]) * xx_inv

    return b_mean, b_var


def _sadf_outer_loop(X: pd.DataFrame, y: pd.DataFrame, min_length: int, model: str, phi: float,
                     molecule: list) -> pd.Series:
    """
    This function gets SADF for t times from molecule

    :param X: (pd.DataFrame) Features(factors)
    :param y: (pd.DataFrame) Outcomes
    :param min_length: (int) Minimum number of observations
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param phi: (float) Coefficient to penalize large sample lengths when computing SMT, in [0, 1]
    :param molecule: (list) Indices to get SADF
    :return: (pd.Series) SADF statistics
    """
    sadf_series = pd.Series(index=molecule, dtype='float64')
    for index in molecule:
        X_subset = X.loc[:index].values
        y_subset = y.loc[:index].values.reshape(-1, 1)
        value = _get_sadf_at_t(X_subset, y_subset, min_length, model, phi)
        sadf_series[index] = value
    return sadf_series


def get_sadf(series: pd.Series, model: str, lags: Union[int, list], min_length: int, add_const: bool = False,
             phi: float = 0, num_threads: int = 8, verbose: bool = True) -> pd.Series:
    """
    Advances in Financial Machine Learning, p. 258-259.

    Multithread implementation of SADF

    SADF fits the ADF regression at each end point t with backwards expanding start points. For the estimation
    of SADF(t), the right side of the window is fixed at t. SADF recursively expands the beginning of the sample
    up to t - min_length, and returns the sup of this set.

    When doing with sub- or super-martingale test, the variance of beta of a weak long-run bubble may be smaller than
    one of a strong short-run bubble, hence biasing the method towards long-run bubbles. To correct for this bias,
    ADF statistic in samples with large lengths can be penalized with the coefficient phi in [0, 1] such that:

    ADF_penalized = ADF / (sample_length ^ phi)

    :param series: (pd.Series) Series for which SADF statistics are generated
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param lags: (int or list) Either number of lags to use or array of specified lags
    :param min_length: (int) Minimum number of observations needed for estimation
    :param add_const: (bool) Flag to add constant
    :param phi: (float) Coefficient to penalize large sample lengths when computing SMT, in [0, 1]
    :param num_threads: (int) Number of cores to use
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.Series) SADF statistics
    """
    X, y = _get_y_x(series, model, lags, add_const)
    molecule = y.index[min_length:y.shape[0]]

    sadf_series = mp_pandas_obj(func=_sadf_outer_loop,
                                pd_obj=('molecule', molecule),
                                X=X,
                                y=y,
                                min_length=min_length,
                                model=model,
                                phi=phi,
                                num_threads=num_threads,
                                verbose=verbose,
                                )
    return sadf_series


def get_vpin(volume: pd.Series, buy_volume: pd.Series, window: int = 1) -> pd.Series:
    """
    Advances in Financial Machine Learning, p. 292-293.

    Get Volume-Synchronized Probability of Informed Trading (VPIN) from bars

    :param volume: (pd.Series) Bar volume
    :param buy_volume: (pd.Series) Bar volume classified as buy (either tick rule, BVC or aggressor side methods applied)
    :param window: (int) Estimation window
    :return: (pd.Series) VPIN series
    """
    sell_volume = volume - buy_volume
    volume_imbalance = abs(buy_volume - sell_volume)
    return volume_imbalance.rolling(window=window).mean() / volume


def vwap(dollar_volume: list, volume: list) -> float:
    """
    Get Volume Weighted Average Price (VWAP).

    :param dollar_volume: (list) Dollar volumes
    :param volume: (list) Trades sizes
    :return: (float) VWAP value
    """
    return sum(dollar_volume) / sum(volume)


def get_avg_tick_size(tick_size_arr: list) -> float:
    """
    Get average tick size in a bar.

    :param tick_size_arr: (list) Trade sizes
    :return: (float) Average trade size
    """
    return np.mean(tick_size_arr)


def get_roll_measure(close_prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, page 282.

    Get Roll Measure

    Roll Measure gives the estimate of effective bid-ask spread
    without using quote-data.

    :param close_prices: (pd.Series) Close prices
    :param window: (int) Estimation window
    :return: (pd.Series) Roll measure
    """
    price_diff = close_prices.diff()
    price_diff_lag = price_diff.shift(1)
    return 2 * np.sqrt(abs(price_diff.rolling(window=window).cov(price_diff_lag)))


def get_roll_impact(close_prices: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Get Roll Impact.

    Derivate from Roll Measure which takes into account dollar volume traded.

    :param close_prices: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volume series
    :param window: (int) Estimation window
    :return: (pd.Series) Roll impact
    """
    roll_measure = get_roll_measure(close_prices, window)
    return roll_measure / dollar_volume


# Corwin-Schultz algorithm
def _get_beta(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get beta estimate from Corwin-Schultz algorithm

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) Estimation window
    :return: (pd.Series) Beta estimates
    """
    ret = np.log(high / low)
    high_low_ret = ret ** 2
    beta = high_low_ret.rolling(window=2).sum()
    beta = beta.rolling(window=window).mean()
    return beta


def _get_gamma(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get gamma estimate from Corwin-Schultz algorithm.

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :return: (pd.Series) Gamma estimates
    """
    high_max = high.rolling(window=2).max()
    low_min = low.rolling(window=2).min()
    gamma = np.log(high_max / low_min) ** 2
    return gamma


def _get_alpha(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get alpha from Corwin-Schultz algorithm.

    :param beta: (pd.Series) Beta estimates
    :param gamma: (pd.Series) Gamma estimates
    :return: (pd.Series) Alphas
    """
    den = 3 - 2 * 2 ** .5
    alpha = (2 ** .5 - 1) * (beta ** .5) / den
    alpha -= (gamma / den) ** .5
    alpha[alpha < 0] = 0  # Set negative alphas to 0 (see p.727 of paper)
    return alpha


def get_corwin_schultz_estimator(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get Corwin-Schultz spread estimator using high-low prices

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) Estimation window
    :return: (pd.Series) Corwin-Schultz spread estimators
    """
    # Note: S<0 iif alpha<0
    beta = _get_beta(high, low, window)
    gamma = _get_gamma(high, low)
    alpha = _get_alpha(beta, gamma)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    start_time = pd.Series(high.index[0:spread.shape[0]], index=spread.index)
    spread = pd.concat([spread, start_time], axis=1)
    spread.columns = ['Spread', 'Start_Time']  # 1st loc used to compute beta
    return spread.Spread


def get_bekker_parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.2, page 286.

    Get Bekker-Parkinson volatility from gamma and beta in Corwin-Schultz algorithm.

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) Estimation window
    :return: (pd.Series) Bekker-Parkinson volatility estimates
    """
    # pylint: disable=invalid-name
    beta = _get_beta(high, low, window)
    gamma = _get_gamma(high, low)

    k2 = (8 / np.pi) ** 0.5
    den = 3 - 2 * 2 ** .5
    sigma = (2 ** -0.5 - 1) * beta ** 0.5 / (k2 * den)
    sigma += (gamma / (k2 ** 2 * den)) ** 0.5
    sigma[sigma < 0] = 0
    return sigma



def get_lempel_ziv_entropy(message: str) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.2, page 266.

    Get Lempel-Ziv entropy estimate

    :param message: (str) Encoded message
    :return: (float) Lempel-Ziv entropy
    """
    i, lib = 1, [message[0]]
    while i < len(message):
        for j in range(i, len(message)):
            message_ = message[i:j + 1]
            if message_ not in lib:
                lib.append(message_)
                break
        i = j + 1
    return len(lib) / len(message)


def _prob_mass_function(message: str, word_length: int) -> dict:
    """
    Advances in Financial Machine Learning, Snippet 18.1, page 266.

    Compute probability mass function for a one-dim discete rv

    :param message: (str or array) Encoded message
    :param word_length: (int) Approximate word length
    :return: (dict) Dict of pmf for each word from message
    """
    lib = {}
    if not isinstance(message, str):
        message = ''.join(map(str, message))
    for i in range(word_length, len(message)):
        message_ = message[i - word_length:i]
        if message_ not in lib:
            lib[message_] = [i - word_length]
        else:
            lib[message_] = lib[message_] + [i - word_length]
    pmf = float(len(message) - word_length)
    pmf = {i: len(lib[i]) / pmf for i in lib}
    return pmf


def get_plug_in_entropy(message: str, word_length: int = None) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.1, page 265.

    Get Plug-in entropy estimator

    :param message: (str or array) Encoded message
    :param word_length: (int) Approximate word length
    :return: (float) Plug-in entropy
    """
    if word_length is None:
        word_length = 1
    pmf = _prob_mass_function(message, word_length)
    out = -sum([pmf[i] * np.log2(pmf[i]) for i in pmf]) / word_length
    return out


@njit()
def _match_length(message: str, start_index: int, window: int) -> Union[int, str]:    # pragma: no cover
    """
    Advances in Financial Machine Learning, Snippet 18.3, page 267.

    Function That Computes the Length of the Longest Match

    :param message: (str or array) Encoded message
    :param start_index: (int) Start index for search
    :param window: (int) Window length
    :return: (int, str) Match length and matched string
    """
    # Maximum matched length+1, with overlap.
    sub_str = ''
    for length in range(window):
        msg1 = message[start_index: start_index + length + 1]
        for j in range(start_index - window, start_index):
            msg0 = message[j: j + length + 1]
            if len(msg1) != len(msg0):
                continue
            if msg1 == msg0:
                sub_str = msg1
                break  # Search for higher l.
    return len(sub_str) + 1, sub_str  # Matched length + 1


def get_konto_entropy(message: str, window: int = 0) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.4, page 268.

    Implementations of Algorithms Discussed in Gao et al.[2008]

    Get Kontoyiannis entropy

    :param message: (str or array) Encoded message
    :param window: (int) Expanding window length, can be negative
    :return: (float) Kontoyiannis entropy
    """
    out = {
        'h': 0,
        'r': 0,
        'num': 0,
        'sum': 0,
        'sub_str': []
    }
    if window <= 0:
        points = range(1, len(message) // 2 + 1)
    else:
        window = min(window, len(message) // 2)
        points = range(window, len(message) - window + 1)
    for i in points:
        if window <= 0:
            length, msg_ = _match_length(message, i, i)
            out['sum'] += np.log2(i + 1) / length  # To avoid Doeblin condition
        else:
            length, msg_ = _match_length(message, i, window)
            out['sum'] += np.log2(window + 1) / length  # To avoid Doeblin condition
        out['sub_str'].append(msg_)
        out['num'] += 1
    try:
        out['h'] = out['sum'] / out['num']
    except ZeroDivisionError:
        out['h'] = 0
    out['r'] = 1 - out['h'] / (np.log2(len(message)) if np.log2(len(message)) > 0 else 1)  # Redundancy, 0<=r<=1
    return out['h']



class StandardBars(BaseBars):
    """
    Contains all of the logic to construct the standard bars from chapter 2. This class shouldn't be used directly.
    We have added functions to the package such as get_dollar_bars which will create an instance of this
    class and then construct the standard bars, to return to the user.

    This is because we wanted to simplify the logic as much as possible, for the end user.
    """

    def __init__(self, metric: str, threshold: int = 50000, batch_size: int = 20000000):
        """
        Constructor

        :param metric: (str) Type of run bar to create. Example: "dollar_run"
        :param threshold: (int) Threshold at which to sample
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        """
        BaseBars.__init__(self, metric, batch_size)

        # Threshold at which to sample
        self.threshold = threshold

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for standard bars
        """
        self.open_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}

    def _extract_bars(self, data: Union[list, tuple, np.ndarray]) -> list:
        """
        For loop which compiles the various bars: dollar, volume, or tick.
        We did investigate the use of trying to solve this in a vectorised manner but found that a For loop worked well.

        :param data: (tuple) Contains 3 columns - date_time, price, and volume.
        :return: (list) Extracted bars
        """

        # Iterate over rows
        list_bars = []

        for row in data:
            # Set variables
            date_time = row[0]
            self.tick_num += 1
            price = np.float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            if isinstance(self.threshold, (int, float)):
                # If the threshold is fixed, it's used for every sampling
                threshold = self.threshold
            else:
                # If the threshold is changing, then the threshold defined just before
                # sampling time is used
                threshold = self.threshold.iloc[self.threshold.index.get_loc(date_time, method='pad')]

            if self.open_price is None:
                self.open_price = price

            # Update high low prices
            self.high_price, self.low_price = self._update_high_low(price)

            # Calculations
            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_dollar_value'] += dollar_value
            self.cum_statistics['cum_volume'] += volume
            if signed_tick == 1:
                self.cum_statistics['cum_buy_volume'] += volume

            # If threshold reached then take a sample
            if self.cum_statistics[self.metric] >= threshold:  # pylint: disable=eval-used
                self._create_bars(date_time, price,
                                  self.high_price, self.low_price, list_bars)

                # Reset cache
                self._reset_cache()
        return list_bars


def get_dollar_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: Union[float, pd.Series] = 70000000,
                    batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the dollar bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al,
    it is suggested that using 1/50 of the average daily dollar value, would result in more desirable statistical
    properties.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                            in the format[date_time, price, volume]
    :param threshold: (float, or pd.Series) A cumulative value above this threshold triggers a sample to be taken.
                      If a series is given, then at each sampling time the closest previous threshold is used.
                      (Values in the series can only be at times when the threshold is changed, not for every observation)
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) Dataframe of dollar bars
    """

    bars = StandardBars(metric='cum_dollar_value', threshold=threshold, batch_size=batch_size)
    dollar_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return dollar_bars


def get_volume_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: Union[float, pd.Series] = 70000000,
                    batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the volume bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    Following the paper "The Volume Clock: Insights into the high frequency paradigm" by Lopez de Prado, et al,
    it is suggested that using 1/50 of the average daily volume, would result in more desirable statistical properties.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                            in the format[date_time, price, volume]
    :param threshold: (float, or pd.Series) A cumulative value above this threshold triggers a sample to be taken.
                      If a series is given, then at each sampling time the closest previous threshold is used.
                      (Values in the series can only be at times when the threshold is changed, not for every observation)
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) Dataframe of volume bars
    """
    bars = StandardBars(metric='cum_volume', threshold=threshold, batch_size=batch_size)
    volume_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return volume_bars


def get_tick_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], threshold: Union[float, pd.Series] = 70000000,
                  batch_size: int = 20000000, verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates the tick bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                             in the format[date_time, price, volume]
    :param threshold: (float, or pd.Series) A cumulative value above this threshold triggers a sample to be taken.
                      If a series is given, then at each sampling time the closest previous threshold is used.
                      (Values in the series can only be at times when the threshold is changed, not for every observation)
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (bool) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) Dataframe of volume bars
    """
    bars = StandardBars(metric='cum_ticks',
                        threshold=threshold, batch_size=batch_size)
    tick_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return tick_bars


# pylint: disable=too-many-instance-attributes
class TimeBars(BaseBars):
    """
    Contains all of the logic to construct the time bars. This class shouldn't be used directly.
    Use get_time_bars instead
    """

    def __init__(self, resolution: str, num_units: int, batch_size: int = 20000000):
        """
        Constructor

        :param resolution: (str) Type of bar resolution: ['D', 'H', 'MIN', 'S']
        :param num_units: (int) Number of days, minutes, etc.
        :param batch_size: (int) Number of rows to read in from the csv, per batch
        """
        BaseBars.__init__(self, metric=None, batch_size=batch_size)

        # Threshold at which to sample (in seconds)
        self.time_bar_thresh_mapping = {'D': 86400, 'H': 3600, 'MIN': 60, 'S': 1}  # Number of seconds
        assert resolution in self.time_bar_thresh_mapping, "{} resolution is not implemented".format(resolution)
        self.resolution = resolution  # Type of bar resolution: 'D', 'H', 'MIN', 'S'
        self.num_units = num_units  # Number of days/minutes/...
        self.threshold = self.num_units * self.time_bar_thresh_mapping[self.resolution]
        self.timestamp = None  # Current bar timestamp

    def _reset_cache(self):
        """
        Implementation of abstract method _reset_cache for time bars
        """
        self.open_price = None
        self.close_price = None
        self.high_price, self.low_price = -np.inf, np.inf
        self.cum_statistics = {'cum_ticks': 0, 'cum_dollar_value': 0, 'cum_volume': 0, 'cum_buy_volume': 0}

    def _extract_bars(self, data: Union[list, tuple, np.ndarray]) -> list:
        """
        For loop which compiles time bars.
        We did investigate the use of trying to solve this in a vectorised manner but found that a For loop worked well.

        :param data: (tuple) Contains 3 columns - date_time, price, and volume.
        :return: (list) Extracted bars
        """

        # Iterate over rows
        list_bars = []

        for row in data:
            # Set variables
            date_time = row[0].timestamp()  # Convert to UTC timestamp
            self.tick_num += 1
            price = np.float(row[1])
            volume = row[2]
            dollar_value = price * volume
            signed_tick = self._apply_tick_rule(price)

            timestamp_threshold = (int(
                float(date_time)) // self.threshold + 1) * self.threshold  # Current tick boundary timestamp

            # Init current bar timestamp with first ticks boundary timestamp
            if self.timestamp is None:
                self.timestamp = timestamp_threshold
            # Bar generation condition
            # Current ticks bar timestamp differs from current bars timestamp
            elif self.timestamp < timestamp_threshold:
                self._create_bars(self.timestamp, self.close_price,
                                  self.high_price, self.low_price, list_bars)

                # Reset cache
                self._reset_cache()
                self.timestamp = timestamp_threshold  # Current bar timestamp update

            # Update counters
            if self.open_price is None:
                self.open_price = price

            # Update high low prices
            self.high_price, self.low_price = self._update_high_low(price)

            # Update close price
            self.close_price = price

            # Calculations
            self.cum_statistics['cum_ticks'] += 1
            self.cum_statistics['cum_dollar_value'] += dollar_value
            self.cum_statistics['cum_volume'] += volume
            if signed_tick == 1:
                self.cum_statistics['cum_buy_volume'] += volume

        return list_bars


def get_time_bars(file_path_or_df: Union[str, Iterable[str], pd.DataFrame], resolution: str = 'D', num_units: int = 1, batch_size: int = 20000000,
                  verbose: bool = True, to_csv: bool = False, output_path: Optional[str] = None):
    """
    Creates Time Bars: date_time, open, high, low, close, volume, cum_buy_volume, cum_ticks, cum_dollar_value.

    :param file_path_or_df: (str, iterable of str, or pd.DataFrame) Path to the csv file(s) or Pandas Data Frame containing raw tick data
                            in the format[date_time, price, volume]
    :param resolution: (str) Resolution type ('D', 'H', 'MIN', 'S')
    :param num_units: (int) Number of resolution units (3 days for example, 2 hours)
    :param batch_size: (int) The number of rows per batch. Less RAM = smaller batch size.
    :param verbose: (int) Print out batch numbers (True or False)
    :param to_csv: (bool) Save bars to csv after every batch run (True or False)
    :param output_path: (str) Path to csv file, if to_csv is True
    :return: (pd.DataFrame) Dataframe of time bars, if to_csv=True return None
    """

    bars = TimeBars(resolution=resolution, num_units=num_units, batch_size=batch_size)
    time_bars = bars.batch_run(file_path_or_df=file_path_or_df, verbose=verbose, to_csv=to_csv, output_path=output_path)
    return time_bars

####################################################################################################################
# load data
####################################################################################################################
data = pd.read_csv('/home/pisati/Desktop/Backtesting_Night/%FTMIB 1!-IDM_1T.csv')
n_contracts=1

##################################################################################################################
# 1) Roll Model
##################################################################################################################
data['aggressor'] = tick_rule(data['Close'])
spread, noise = roll_model(data['Close'])
print(" Roll Model : Spread", round(spread, 2), "Noise", round(noise, 2))

##################################################################################################################
# 2) Lambdas
##################################################################################################################
kyles=kyles_lambda(data['Close'], data['Vol'], tick_rule(data['Close']))
hasb_flow = np.sqrt(data['Close'] * data['Vol']) * tick_rule(data['Close'])
hasbroucks=hasbroucks_lambda(data['Close'], hasb_flow)
amihuds=amihuds_lambda(data['Close'], data['Close'] * data['Vol'])

print('Kyles Lambda :', round(kyles, 4))
print('Hasbroucks Lambda :', round(hasbroucks, 10))
print('Amihuds Lambda :', round(amihuds,10))

##################################################################################################################
# 3) Price Impact
##################################################################################################################
for n_contracts in range(1, 21, 2):

    price_impact_K=kyles/data['Close'].iloc[-1]*n_contracts
    price_impact_A=amihuds*data['Close'].iloc[-1]*n_contracts
    hasb_flow = np.sqrt(data['Close'] * data['Vol']) * tick_rule(data['Close'])
    hasbroucks=hasbroucks_lambda(data['Close'], hasb_flow)
    price_impact_H=hasbroucks*np.sqrt( data['Close'].iloc[-1] *n_contracts)
    PRICE_IMPACT=[price_impact_K, price_impact_H, price_impact_A]
    average_price_impact = statistics.mean(PRICE_IMPACT)

    print('price_impact_K', round(price_impact_K, 5))
    print('price_impact_A', round(price_impact_A, 5))
    print('price_impact_H', round(price_impact_H, 5))
    print("For N contracts =", n_contracts,  "The Average Price Impact is ", round(average_price_impact, 5))