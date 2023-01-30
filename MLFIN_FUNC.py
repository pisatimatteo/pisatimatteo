
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
from scipy import  linalg

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


def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

class CampbellBacktesting:
    """
    This class implements the Haircut Sharpe Ratios and Profit Hurdles algorithms described in the following paper:
    `Campbell R. Harvey and Yan Liu, Backtesting, (Fall 2015). Journal of Portfolio Management,
    2015 <https://papers.ssrn.com/abstract_id=2345489>`_; The code is based on the code provided by the authors of the paper.

    The Haircut Sharpe Ratios algorithm lets the user adjust the observed Sharpe Ratios to take multiple testing into account
    and calculate the corresponding haircuts. The haircut is the percentage difference between the original Sharpe ratio
    and the new Sharpe ratio.

    The Profit Hurdle algorithm lets the user calculate the required mean return for a strategy at a given level of
    significance, taking multiple testing into account.
    """

    def __init__(self, simulations=2000):
        """
        Set the desired number of simulations to make in Haircut Sharpe Ratios or Profit Hurdle algorithms.

        :param simulations: (int) Number of simulations
        """

        self.simulations = simulations

    @staticmethod
    def _sample_random_multest(rho, n_trails, prob_zero_mean, lambd, n_simulations, annual_vol=0.15, n_obs=240):
        """
        Generates empirical p-value distributions.

        The algorithm is described in the paper and is based on the model estimated by `Harvey, C.R., Y. Liu,
        and H. Zhu., … and the Cross-section of Expected Returns. Review of Financial Studies, forthcoming 2015`,
        referred to as the HLZ model.

        It provides a set of simulated t-statistics based on the parameters recieved from the _parameter_calculation
        method.

        Researchers propose a structural model to capture trading strategies’ underlying distribution.
        With probability p0 (prob_zero_mean), a strategy has a mean return of zero and therefore comes
        from the null distribution. With probability 1 – p0, a strategy has a nonzero mean and therefore
        comes from the alternative distribution - exponential.

        :param rho: (float) Average correlation among returns
        :param n_trails: (int) Total number of trials inside a simulation
        :param prob_zero_mean: (float) Probability for a random factor to have a zero mean
        :param lambd: (float) Average of monthly mean returns for true strategies
        :param n_simulations: (int) Number of rows (simulations)
        :param annual_vol: (float) HLZ assume that the innovations in returns follow a normal distribution with a mean
                                   of zero and a standard deviation of ma = 15%
        :param n_obs: (int) Number of observations of used for volatility estimation from HLZ
        :return: (np.ndarray) Array with distributions calculated
        """

        # Assumed level of monthly volatility = adjusted yearly volatility
        monthly_volatility = annual_vol / 12 ** (1 / 2)

        # Creating a correlation matrix of simulated returns. All correlations are assumed to be the same as average
        # correlation among returns
        # The first row of the correlation matrix: [1, rho, rho, .., rho]
        correlation_vector = np.insert(rho * np.ones((1, n_trails - 1)), 0, 1)

        # Correlation matrix created from the vector by expanding it
        correlation_matrix = linalg.toeplitz(correlation_vector)

        # Vector with mean of simulated returns - zeros
        mean = np.zeros(n_trails)

        # Creating a sample from a multivariate normal distribution as returns simulations
        # Covariance matrix - Created from correlation matrix multiplied by monthly volatility and adjusted
        covariance_matrix = correlation_matrix * (monthly_volatility ** 2 / n_obs)

        # Result - n_simulations rows with n_trails inside
        shock_mat = np.random.multivariate_normal(mean, covariance_matrix, n_simulations)

        # Sample of uniform distribution with the same dimensions as shock_mat
        prob_vec = np.random.uniform(0, 1, (n_simulations, n_trails))

        # Sample of exponential distribution with same dimensions ad shock_mat
        mean_vec = np.random.exponential(lambd, (n_simulations, n_trails))

        # Taking the factors that have non-zero mean
        nonzero_mean = prob_vec > prob_zero_mean

        # Generating the null hypothesis - either zero mean or from an exponential distribution
        mu_null = np.multiply(nonzero_mean, mean_vec)

        # Matrix of p-value distributions
        tstat_matrix = abs(mu_null + shock_mat) / (monthly_volatility / n_obs ** (1 / 2))

        return tstat_matrix

    @staticmethod
    def _parameter_calculation(rho):
        """
        Estimates the parameters used to generate the distributions in _sample_random_multest - the HLZ model.

        Based on the work of HLZ, the pairwise correlation of returns is used to estimate the probability (prob_zero_mean),
        total number of trials (n_simulations) and (lambd) - parameter of the exponential distribution. Levels and
        parameters taken from the HLZ research.

        :param rho: (float) Average correlation coefficient between strategy returns
        :return: (np.array) Array of parameters
        """

        # Levels of parameters based on rho. [rho, n_simulations, prob_zero_mean, lambd]
        parameter_levels = np.array([[0, 1295, 3.9660 * 0.1, 5.4995 * 0.001],
                                     [0.2, 1377, 4.4589 * 0.1, 5.5508 * 0.001],
                                     [0.4, 1476, 4.8604 * 0.1, 5.5413 * 0.001],
                                     [0.6, 1773, 5.9902 * 0.1, 5.5512 * 0.001],
                                     [0.8, 3109, 8.3901 * 0.1, 5.5956 * 0.001]])

        # Linear interpolation for parameter estimates
        if (rho < 0):
            parameters = parameter_levels[1]  # Set at the preferred level if rho is misspecified
        elif (rho < 0.2):
            parameters = ((0.2 - rho) / 0.2) * parameter_levels[0] + ((rho - 0) / 0.2) * parameter_levels[1]
        elif (rho < 0.4):
            parameters = ((0.4 - rho) / 0.2) * parameter_levels[1] + ((rho - 0.2) / 0.2) * parameter_levels[2]
        elif (rho < 0.6):
            parameters = ((0.6 - rho) / 0.2) * parameter_levels[2] + ((rho - 0.4) / 0.2) * parameter_levels[3]
        elif (rho < 0.8):
            parameters = ((0.8 - rho) / 0.2) * parameter_levels[3] + ((rho - 0.6) / 0.2) * parameter_levels[4]
        elif (rho < 1.0):  # Interpolation based on the previous level here
            parameters = ((0.8 - rho) / 0.2) * parameter_levels[3] + ((rho - 0.6) / 0.2) * parameter_levels[4]
        else:
            parameters = parameter_levels[1]  # Set at the preferred level if rho is misspecified

        return parameters

    @staticmethod
    def _annualized_sharpe_ratio(sharpe_ratio, sampling_frequency='A', rho=0, annualized=False,
                                 autocorr_adjusted=False):
        """
        Calculate the equivalent annualized Sharpe ratio after taking the autocorrelation of returns into account.

        Adjustments are based on the work of `Lo, A., The Statistics of Sharpe Ratios. Financial Analysts Journal,
        58 (2002), pp. 36-52` and are described there in more detail.

        :param sharpe_ratio: (float) Sharpe ratio of the strategy
        :param sampling_frequency: (str) Sampling frequency of returns
                                   ['D','W','M','Q','A'] = [Daily, Weekly, Monthly, Quarterly, Annual]
        :param rho: (float) Autocorrelation coefficient of returns at specified frequency
        :param annualized: (bool) Flag if annualized, 'ind_an' = 1, otherwise = 0
        :param autocorr_adjusted: (bool) Flag if Sharpe ratio was adjusted for returns autocorrelation
        :return: (float) Adjusted annualized Sharpe ratio
        """

        # If not annualized, calculating the appropriate multiplier for the Sharpe ratio
        if sampling_frequency == 'D':
            times_per_year = 360
        elif sampling_frequency == 'W':
            times_per_year = 52
        elif sampling_frequency == 'M':
            times_per_year = 12
        elif sampling_frequency == 'Q':
            times_per_year = 4
        elif sampling_frequency == 'A':
            times_per_year = 1
        else:
            times_per_year = 1  # Misspecified

        if not annualized:
            annual_multiplier = times_per_year ** (1 / 2)
        else:
            annual_multiplier = 1

        # If not adjusted for returns autocorrelation, another multiplier
        if not autocorr_adjusted:
            autocorr_multiplier = (1 + (2 * rho / (1 - rho)) * (1 - ((1 - rho ** (times_per_year)) /
                                                                     (times_per_year * (1 - rho))))) ** (-0.5)
        else:
            autocorr_multiplier = 1

        # And calculating the adjusted Sharpe ratio
        adjusted_sr = sharpe_ratio * annual_multiplier * autocorr_multiplier

        return adjusted_sr

    @staticmethod
    def _monthly_observations(num_obs, sampling_frequency):
        """
        Calculates the number of monthly observations based on sampling frequency and number of observations.

        :param num_obs: (int) Number of observations used for modelling
        :param sampling_frequency: (str) Sampling frequency of returns
                                   ['D','W','M','Q','A'] = [Daily, Weekly, Monthly, Quarterly, Annual]
        :return: (np.float64) Number of monthly observations
        """

        # N - Number of monthly observations
        if sampling_frequency == 'D':
            monthly_obs = np.floor(num_obs * 12 / 360)
        elif sampling_frequency == 'W':
            monthly_obs = np.floor(num_obs * 12 / 52)
        elif sampling_frequency == 'M':
            monthly_obs = np.floor(num_obs * 12 / 12)
        elif sampling_frequency == 'Q':
            monthly_obs = np.floor(num_obs * 12 / 4)
        elif sampling_frequency == 'A':
            monthly_obs = np.floor(num_obs * 12 / 1)
        else:  # If the frequency is misspecified
            monthly_obs = np.floor(num_obs)

        return monthly_obs

    @staticmethod
    def _holm_method_sharpe(all_p_values, num_mult_test, p_val):
        """
        Runs one cycle of the Holm method for the Haircut Shape ratio algorithm.

        :param all_p_values: (np.array) Sorted p-values to adjust
        :param num_mult_test: (int) Number of multiple tests allowed
        :param p_val: (float) Significance level p-value
        :return: (np.float64) P-value adjusted at a significant level
        """

        # Array for final p-values of the Holm method
        p_holm_values = np.array([])

        # Iterating through multiple tests
        for i in range(1, (num_mult_test + 2)):
            # Creating array for Holm adjusted p-values (M-j+1)*p(j) in the paper
            p_adjusted_holm = np.array([])

            # Iterating through the available subsets of Holm adjusted p-values
            for j in range(1, i + 1):
                # Holm adjusted p-values
                p_adjusted_holm = np.append(p_adjusted_holm, (num_mult_test + 1 - j + 1) * all_p_values[j - 1])

            # Calculating the final p-values of the Holm method and adding to an array
            p_holm_values = np.append(p_holm_values, min(max(p_adjusted_holm), 1))

        # Getting the Holm adjusted p-value that is significant at our p_val level
        p_holm_significant = p_holm_values[all_p_values == p_val]
        p_holm_result = p_holm_significant[0]

        return p_holm_result

    @staticmethod
    def _bhy_method_sharpe(all_p_values, num_mult_test, p_val):
        """
        Runs one cycle of the BHY method for the Haircut Shape ratio algorithm.

        :param all_p_values: (np.array) Sorted p-values to adjust
        :param num_mult_test: (int) Number of multiple tests allowed
        :param p_val: (float) Significance level p-value
        :param c_constant: (float) Constant used in BHY method
        :return: (np.float64) P-value adjusted at a significant level
        """

        # Array for final p-values of the BHY method
        p_bhy_values = np.array([])

        # BHY constant
        index_vector = np.arange(1, num_mult_test + 1)
        c_constant = sum(1 / index_vector)

        # Iterating through multiple tests backwards
        for i in range(num_mult_test + 1, 0, -1):
            if i == (num_mult_test + 1):  # If it's the last observation
                # The p-value stays the same
                p_adjusted_holm = all_p_values[-1]

            else:  # If it's the previous observations
                # The p-value is adjusted according to the BHY method
                p_adjusted_holm = min(((num_mult_test + 1) * c_constant / i) * all_p_values[i - 1], p_previous)

            # Adding the final BHY method p-values to an array
            p_bhy_values = np.append(p_adjusted_holm, p_bhy_values)
            p_previous = p_adjusted_holm

        # Getting the BHY adjusted p-value that is significant at our p_val level
        p_bhy_significant = p_bhy_values[all_p_values == p_val]
        p_bhy_result = p_bhy_significant

        return p_bhy_result

    @staticmethod
    def _sharpe_ratio_haircut(p_val, monthly_obs, sr_annual):
        """
        Calculates the adjusted Sharpe ratio and the haircut based on the final p-value of the method.

        :param p_val: (float) Adjusted p-value of the method
        :param monthly_obs: (int) Number of monthly observations
        :param sr_annual: (float) Annualized Sharpe ratio to compare to
        :return: (np.array) Elements (Adjusted annual Sharpe ratio, Haircut percentage)
        """

        # Inverting to get z-score of the method
        z_score = ss.t.ppf(1 - p_val / 2, monthly_obs - 1)

        # Adjusted annualized Sharpe ratio of the method
        sr_adjusted = (z_score / monthly_obs ** (1 / 2)) * 12 ** (1 / 2)

        # Haircut of the Sharpe ratio of the method
        haircut = (sr_annual - sr_adjusted) / sr_annual * 100

        return (sr_adjusted, haircut)

    @staticmethod
    def _holm_method_returns(p_values_simulation, num_mult_test, alpha_sig):
        """
        Runs one cycle of the Holm method for the Profit Hurdle algorithm.

        :param p_values_simulation: (np.array) Sorted p-values to adjust
        :param num_mult_test: (int) Number of multiple tests allowed
        :param alpha_sig: (float) Significance level (e.g., 5%)
        :return: (np.float64) P-value adjusted at a significant level
        """

        # Array for adjusted significance levels
        sign_levels = np.zeros(num_mult_test)

        # Creating adjusted levels of significance
        for trail_number in range(1, num_mult_test + 1):
            sign_levels[trail_number - 1] = alpha_sig / (num_mult_test + 1 - trail_number)

        # Where the simulations have higher p-values
        exceeding_pval = (p_values_simulation > sign_levels)

        # Used to find the first exceeding p-value
        exceeding_cumsum = np.cumsum(exceeding_pval)

        if sum(exceeding_cumsum) == 0:  # If no exceeding p-values
            tstat_h = 1.96
        else:
            # Getting the first exceeding p-value
            p_val = p_values_simulation[exceeding_cumsum == 1]

            # And the corresponding t-statistic
            tstat_h = ss.norm.ppf((1 - p_val / 2), 0, 1)

        return tstat_h

    @staticmethod
    def _bhy_method_returns(p_values_simulation, num_mult_test, alpha_sig):
        """
        Runs one cycle of the BHY method for the Profit Hurdle algorithm.

        :param p_values_simulation: (np.array) Sorted p-values to adjust
        :param num_mult_test: (int) Number of multiple tests allowed
        :param alpha_sig: (float) Significance level (e.g., 5%)
        :return: (np.float64) P-value adjusted at a significant level
        """

        if num_mult_test <= 1:  # If only one multiple test
            tstat_b = 1.96
        else:
            # Sort in descending order
            p_desc = np.sort(p_values_simulation)[::-1]

            # Calculating BHY constant
            index_vector = np.arange(1, num_mult_test + 1)
            c_constant = sum(1 / index_vector)

            # Array for adjusted significance levels
            sign_levels = np.zeros(num_mult_test)

            # Creating adjusted levels of significance
            for trail_number in range(1, num_mult_test + 1):
                sign_levels[trail_number - 1] = (alpha_sig * trail_number) / (num_mult_test * c_constant)

            # Finding the first exceeding value
            sign_levels_desc = np.sort(sign_levels)[::-1]
            exceeding_pval = (p_desc <= sign_levels_desc)

            if sum(exceeding_pval) == 0:  # If no exceeding p-values
                tstat_b = 1.96
            else:
                # Getting the first exceeding p-value
                p_val = p_desc[exceeding_pval == 1]
                p_val_pos = np.argmin(abs(p_desc - p_val[0]))

                if p_val_pos == 0:  # If exceeding value is first
                    p_chosen = p_val[0]
                else:  # If not first
                    p_chosen = p_desc[p_val_pos - 1]

                # And the corresponding t-statistic from p-value
                tstat_b = ss.norm.ppf((1 - (p_val[0] + p_chosen) / 4), 0, 1)

        return tstat_b

    def haircut_sharpe_ratios(self, sampling_frequency, num_obs, sharpe_ratio, annualized,
                              autocorr_adjusted, rho_a, num_mult_test, rho):
        # pylint: disable=too-many-locals
        """
        Calculates the adjusted Sharpe ratio due to testing multiplicity.

        This algorithm lets the user calculate Sharpe ratio adjustments and the corresponding haircuts based on
        the key parameters of returns from the strategy. The adjustment methods are Bonferroni, Holm,
        BHY (Benjamini, Hochberg and Yekutieli) and the Average of them. The algorithm calculates adjusted p-value,
        adjusted Sharpe ratio and the haircut.

        The haircut is the percentage difference between the original Sharpe ratio and the new Sharpe ratio.

        :param sampling_frequency: (str) Sampling frequency ['D','W','M','Q','A'] of returns
        :param num_obs: (int) Number of returns in the frequency specified in the previous step
        :param sharpe_ratio: (float) Sharpe ratio of the strategy. Either annualized or in the frequency specified in the previous step
        :param annualized: (bool) Flag if Sharpe ratio is annualized
        :param autocorr_adjusted: (bool) Flag if Sharpe ratio was adjusted for returns autocorrelation
        :param rho_a: (float) Autocorrelation coefficient of returns at the specified frequency (if the Sharpe ratio
                              wasn't corrected)
        :param num_mult_test: (int) Number of other strategies tested (multiple tests)
        :param rho: (float) Average correlation among returns of strategies tested
        :return: (np.ndarray) Array with adjuted p-value, adjusted Sharpe ratio, and haircut as rows
                              for Bonferroni, Holm, BHY and average adjustment as columns
        """

        # Calculating the annual Sharpe ratio adjusted for the autocorrelation of returns
        sr_annual = self._annualized_sharpe_ratio(sharpe_ratio, sampling_frequency, rho_a, annualized,
                                                  autocorr_adjusted)

        # Estimating the parameters used for distributions based on HLZ model
        # Result is [rho, n_simulations, prob_zero_mean, lambd]
        parameters = self._parameter_calculation(rho)

        # Getting the number of monthly observations in a sample
        monthly_obs = self._monthly_observations(num_obs, sampling_frequency)

        # Needed number of trails inside a simulation with the check of (num_simulations >= num_mul_tests)
        num_trails = int((np.floor(num_mult_test / parameters[1]) + 1) * np.floor(parameters[1] + 1))

        # Generating a panel of t-ratios (of size self.simulations * num_simulations)
        t_sample = self._sample_random_multest(parameters[0], num_trails, parameters[2], parameters[3],
                                               self.simulations)

        # Annual Sharpe ratio, adjusted to monthly
        sr_monthly = sr_annual / 12 ** (1 / 2)

        # Calculating t-ratio based on the Sharpe ratio and the number of observations
        t_ratio = sr_monthly * monthly_obs ** (1 / 2)

        # Calculating adjusted p-value from the given t-ratio
        p_val = 2 * (1 - ss.t.cdf(t_ratio, monthly_obs - 1))

        # Creating arrays for p-values from simulations of Holm and BHY methods.
        p_holm = np.ones(self.simulations)
        p_bhy = np.ones(self.simulations)

        # Iterating through the simulations
        for simulation_number in range(1, self.simulations + 1):

            # Get one sample of previously generated simulation of t-values
            t_values_simulation = t_sample[simulation_number - 1, 1:(num_mult_test + 1)]

            # Calculating adjusted p-values from the simulated t-ratios
            p_values_simulation = 2 * (1 - ss.norm.cdf(t_values_simulation, 0, 1))

            # To the N (num_mult_test) other strategies tried (from the simulation),
            # we add the adjusted p_value of the real strategy.
            all_p_values = np.append(p_values_simulation, p_val)

            # Ordering p-values
            all_p_values = np.sort(all_p_values)

            # Holm method
            p_holm[simulation_number - 1] = self._holm_method_sharpe(all_p_values, num_mult_test, p_val)

            # BHY method
            p_bhy[simulation_number - 1] = self._bhy_method_sharpe(all_p_values, num_mult_test, p_val)

        # Calculating the resulting p-values of methods from simulations
        # Array with adjusted p-values
        # [Bonferroni, Holm, BHY, Average]
        p_val_adj = np.array([np.minimum(num_mult_test * p_val, 1), np.median(p_holm), np.median(p_bhy)])
        p_val_adj = np.append(p_val_adj, (p_val_adj[0] + p_val_adj[1] + p_val_adj[2]) / 3)

        # Arrays with adjusted Sharpe ratios and haircuts
        sr_adj = np.zeros(4)
        haircut = np.zeros(4)

        # Adjusted Sharpe ratios and haircut percentages
        sr_adj[0], haircut[0] = self._sharpe_ratio_haircut(p_val_adj[0], monthly_obs, sr_annual)
        sr_adj[1], haircut[1] = self._sharpe_ratio_haircut(p_val_adj[1], monthly_obs, sr_annual)
        sr_adj[2], haircut[2] = self._sharpe_ratio_haircut(p_val_adj[2], monthly_obs, sr_annual)
        sr_adj[3], haircut[3] = self._sharpe_ratio_haircut(p_val_adj[3], monthly_obs, sr_annual)

        results = np.array([p_val_adj,
                            sr_adj,
                            haircut])

        return results

    def profit_hurdle(self, num_mult_test, num_obs, alpha_sig, vol_anu, rho):
        # pylint: disable=too-many-locals
        """
        Calculates the required mean monthly return for a strategy at a given level of significance.

        This algorithm uses four adjustment methods - Bonferroni, Holm, BHY (Benjamini, Hochberg and Yekutieli)
        and the Average of them. The result is the Minimum Average Monthly Return for the strategy to be significant
        at a given significance level, taking into account multiple testing.

        This function doesn't allow for any autocorrelation in the strategy returns.

        :param num_mult_test: (int) Number of tests in multiple testing allowed (number of other strategies tested)
        :param num_obs: (int) Number of monthly observations for a strategy
        :param alpha_sig: (float) Significance level (e.g., 5%)
        :param vol_anu: (float) Annual volatility of returns(e.g., 0.05 or 5%)
        :param rho: (float) Average correlation among returns of strategies tested
        :return: (np.ndarray) Minimum Average Monthly Returns for
                              [Independent tests, Bonferroni, Holm, BHY and Average for Multiple tests]
        """

        # Independent test t-statistic
        tstat_independent = ss.norm.ppf((1 - alpha_sig / 2), 0, 1)

        # Bonferroni t-statistic
        p_value_bonferroni = np.divide(alpha_sig, num_mult_test)
        tstat_bonderroni = ss.norm.ppf((1 - p_value_bonferroni / 2), 0, 1)

        # Estimating the parameters used for distributions based on HLZ model
        # Result is [rho, n_simulations, prob_zero_mean, lambd]
        parameters = self._parameter_calculation(rho)

        # Needed number of trails inside a simulation with the check of (num_simulations >= num_mul_tests)
        num_trails = int((np.floor(num_mult_test / parameters[1]) + 1) * np.floor(parameters[1] + 1))

        # Generating a panel of t-ratios (of size self.simulations * num_simulations)
        t_sample = self._sample_random_multest(parameters[0], num_trails, parameters[2], parameters[3],
                                               self.simulations)


        # Arrays for final t-statistics for every simulation for Holm and BHY methods
        tstats_holm = np.array([])
        tstats_bhy = np.array([])

        # Iterating through the simulations
        for simulation_number in range(1, self.simulations + 1):
            # Holm method

            # Get one sample of previously generated simulation of t-values
            t_values_simulation = t_sample[simulation_number - 1, 1:(num_mult_test + 1)]

            # Calculating p-values from the simulated t-ratios
            p_values_simulation = 2 * (1 - ss.norm.cdf(t_values_simulation))
            p_values_simulation = np.sort(p_values_simulation)

            # Holm method itself
            tstat_h = self._holm_method_returns(p_values_simulation, num_mult_test, alpha_sig)

            # Adding to array of t-statistics
            tstats_holm = np.append(tstats_holm, tstat_h)

            # BHY method

            # Get one sample of previously generated simulation of t-values
            t_values_simulation = t_sample[simulation_number - 1, 1:(num_mult_test + 1)]

            # Calculating p-values from the simulated t-ratios
            p_values_simulation = 2 * (1 - ss.norm.cdf(t_values_simulation))

            # BHY method itself
            tstat_b = self._bhy_method_returns(p_values_simulation, num_mult_test, alpha_sig)

            # Adding to array of t-statistics
            tstats_bhy = np.append(tstats_bhy, tstat_b)

        # Array of t-values for every method
        tcut_vec = np.array([tstat_independent, tstat_bonderroni, np.median(tstats_holm), np.median(tstats_bhy)])

        # Array of minimum average monthly returns for every method
        ret_hur = ((vol_anu / 12 ** (1 / 2)) / num_obs ** (1 / 2)) * tcut_vec

        # Preparing array of results
        results = np.array([ret_hur[0], ret_hur[1], ret_hur[2], ret_hur[3], np.mean(ret_hur[1:-1])]) * 100

        return results


"""
Implements statistics related to:
- flattening and flips
- average period of position holding
- concentration of bets
- drawdowns
- various Sharpe ratios
- minimum track record length
"""


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

