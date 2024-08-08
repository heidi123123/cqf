# Ignore warnings
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
from TS_plots import plot_assets_and_residuals


def download_data(ticker, start_date="2014-01-01"):
    """Download ticker data from yfinance library."""
    df = yf.download(ticker, start=start_date)
    df.dropna(inplace=True)
    return df


def prepare_time_series(data1, data2, ticker1, ticker2, index_ticker):
    """Prepare the time series data of historical prices to use later for cointegration analysis."""
    index_data = download_data(index_ticker)  # download index data

    # among the 3 dataframes: determine latest start date and trim all dataframes to start there
    latest_start_date = max(data1.index.min(), data2.index.min(), index_data.index.min())
    data1 = data1[data1.index >= latest_start_date]
    data2 = data2[data2.index >= latest_start_date]
    index_data = index_data[index_data.index >= latest_start_date]

    # combine the 3 trimmed dataframes into 1
    data = pd.concat([data1['Close'], data2['Close'], index_data['Close']], axis=1).dropna()
    data.columns = [ticker1, ticker2, index_ticker]
    return data


def least_squares_regression(y, X):
    """Perform least squares regression to obtain beta coefficients and residuals."""
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # add y-intercept to X
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)  # least sqaures regression into coefficient vector beta
    residuals = y - X @ beta
    return beta, residuals


def perform_adf_test(residuals, significance_level):
    """Perform the Augmented Dickey-Fuller (ADF) test to check for the presence of unit root in a time series.
    H0: time series has a unit root (i.e. non-stationary)"""
    adf_test = adfuller(residuals)
    adf_statistic, p_value = adf_test[0], adf_test[1]

    print(f"ADF Statistic: {adf_statistic:.4f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < significance_level:
        print(f"The residuals are stationary (reject null hypothesis) "
              f"at the {significance_level * 100}% significance level.")
    else:
        print(f"The residuals are not stationary (accept null hypothesis) "
              f"at the {significance_level * 100}% significance level.")
    return adf_test


def perform_engle_granger_step1(ticker1, ticker2, index_ticker, data, plotting, significance_level):
    """Step1 of the Engle-Granger procedure."""

    # OLS regression to obtain regression coefficients beta & residuals
    y = data[ticker1].values
    X = data[ticker2].values.reshape(-1, 1)
    beta, residuals = least_squares_regression(y, X)
    data['residuals'] = residuals

    if plotting:  # plot normalized asset prices and residuals
        plot_assets_and_residuals(data, ticker1, ticker2, index_ticker)

    # perform ADF test
    adf_test_result = perform_adf_test(data['residuals'], significance_level)
    return data, beta, adf_test_result


def get_differences(data, columns):
    """Calculate the returns (differences) Delta y_t = y_t-y_{t-1} for the specified columns in the dataframe."""
    return data[columns].diff().dropna()


def fit_ecm(data, residuals_column, target_column, independent_column):
    """Step2 of the Engle-Granger procedure: fit the Equilibrium Correction Model (ECM)."""
    data_delta = get_differences(data, [target_column, independent_column])
    data_delta['lagged_residuals'] = data[residuals_column].shift(1)  # lag the residuals
    data_delta = data_delta.dropna()

    # OLS to obtain ECM coefficients & residuals
    y = data_delta[target_column].values
    X = data_delta[[independent_column, "lagged_residuals"]].values
    ecm_coefficients, ecm_residuals = least_squares_regression(y, X)

    ecm_residuals = pd.DataFrame(ecm_residuals, index=data_delta.index, columns=["ECM_residuals"])  # convert to pd.df
    return {'coefficients': ecm_coefficients, 'residuals': ecm_residuals}


def ou_likelihood(params, residuals, dt=1):
    """Calculates the negative log-likelihood of an Ornstein-Uhlenbeck process"""
    theta, mu_e, sigma_ou = params
    likelihood = 0
    for t in range(1, len(residuals)):
        mean = residuals[t-1] + theta * (mu_e - residuals[t-1]) * dt
        variance = sigma_ou**2 * dt
        # increment the log likelihood (=log probability density) with mean and variance of the next residual
        likelihood += norm.logpdf(residuals[t], loc=mean, scale=np.sqrt(variance))
    return -likelihood


def estimate_ou_params(residuals):
    """Estimate Ornstein-Uhlenbeck process parameters using maximum likelihood estimation.
    The OU process is given as: d(residuals)_t = -theta (residuals_t-mu_e) dt + sigma_ou dX_t"""
    dt = 1  # we are using daily prices, so time increment dt = 1
    residuals = np.array(residuals)

    initial_params = [0.1, np.mean(residuals), np.std(residuals)]  # [theta0, mu_ou0, sigma_ou0]
    # we minimize negative log-likelihood, which is equivalent to using maximum likelihood estimator (MLE)
    result = minimize(ou_likelihood, initial_params, args=(residuals, dt), method="L-BFGS-B")
    theta, mu_e, sigma_ou = result.x
    return theta, mu_e, sigma_ou


def analyze_cointegration(ticker1, ticker2, index_ticker="SPY",
                          plotting=False, start_date="2014-01-01", significance_level=0.05):
    """Analyze cointegration between two assets ticker1 & ticker2 after start_date <YYYY-MM-DD>."""
    print(f"-" * 100)
    print(f"Analyzing cointegration between {ticker1} and {ticker2}...")

    df1 = download_data(ticker1, start_date)
    df2 = download_data(ticker2, start_date)
    data = prepare_time_series(df1, df2, ticker1, ticker2, index_ticker)

    # Engle-Granger procedure - Step 1
    data, beta, adf_test_result = perform_engle_granger_step1(ticker1, ticker2, index_ticker,
                                                              data, plotting, significance_level)
    # Engle-Granger procedure - Step 2: ECM
    ecm_results = fit_ecm(data, "residuals", ticker1, ticker2)
    print(f"Equilibrium mean-reversion coefficient: {ecm_results['coefficients'][-1]:2f}")

    # Engle-Granger procedure - Step 3 (inofficial): fit OU process to mean-reverting residuals
    theta, mu_e, sigma_ou = estimate_ou_params(data['residuals'])

    print(f"Estimated OU parameters:")
    print(f"Speed of mean reversion (theta): {theta:f}")
    print(f"Long-term mean (mu_e): {mu_e:.4f}")
    print(f"Volatility (sigma_ou): {sigma_ou:.4f}")

    return data, beta, adf_test_result, ecm_results, {'theta': theta, 'mu_e': mu_e, 'sigma_ou': sigma_ou}


def backtest_pairs_trading(data, ticker1, ticker2, z):
    mean_residual = data['residuals'].mean()
    std_residual = data['residuals'].std()
    upper_bound = mean_residual + z * std_residual
    lower_bound = mean_residual - z * std_residual

    # Position Management explanation:
    # position = 1: long, position = -1: short, position = 0: flat
    position_ticker1 = 0
    position_ticker2 = 0
    entry_price_ticker1 = 0
    entry_price_ticker2 = 0
    pnl = []

    for index, row in data.iterrows():
        residual = row['residuals']

        if position_ticker1 == 0 and position_ticker2 == 0:
            if residual > upper_bound:  # spread between both tickers is very positive
                position_ticker1 = -1  # short ticker1 since over-valued from equilibrium
                position_ticker2 = 1  # long ticker2 since under-valued from equilibrium
                entry_price_ticker1 = row[ticker1]
                entry_price_ticker2 = row[ticker2]
            elif residual < lower_bound:  # spread between both tickers is very negative
                position_ticker1 = 1
                position_ticker2 = -1
                entry_price_ticker1 = row[ticker1]
                entry_price_ticker2 = row[ticker2]

        elif position_ticker1 == 1 and position_ticker2 == -1:  # Long ticker1, Short ticker2
            if residual >= mean_residual:
                pnl.append(
                    (row[ticker1] - entry_price_ticker1) + (entry_price_ticker2 - row[ticker2]))  # Close positions
                position_ticker1 = 0
                position_ticker2 = 0

        elif position_ticker1 == -1 and position_ticker2 == 1:  # Short ticker1, Long ticker2
            if residual <= mean_residual:
                pnl.append(
                    (entry_price_ticker1 - row[ticker1]) + (row[ticker2] - entry_price_ticker2))  # Close positions
                position_ticker1 = 0
                position_ticker2 = 0

    return np.sum(pnl)


def evaluate_pairs_trading_strategy(data, ticker1, ticker2):
    # Test different Z values and select the best one
    results = []
    for z in np.arange(0.5, 2.5, 0.1):
        pnl = backtest_pairs_trading(data, ticker1, ticker2, z)
        results.append({'Z': z, 'PnL': pnl})
    return pd.DataFrame(results)


# Example usages
# Coca-Cola and Pepsi
ticker1 = "KO"
ticker2 = "PEP"
data, beta, adf_test_result, ecm_results, ou_params = analyze_cointegration(ticker1, ticker2, significance_level=0.01)
pnl_table = evaluate_pairs_trading_strategy(data, ticker1, ticker2)

# Roche and Novartis
ticker1 = "ROG.SW"
ticker2 = "NOVN.SW"
data, beta, adf_test_result, ecm_results, ou_params = analyze_cointegration(ticker1, ticker2, index_ticker="^SSMI",
                                                                            plotting=True, start_date="2022-01-01")

# Marriott and InterContinental Hotels Group
ticker1 = "MAR"
ticker2 = "IHG"
data, beta, adf_test_result, ecm_results, ou_params = analyze_cointegration(ticker1, ticker2)

# Exxon Mobil and Chevron
ticker1 = "XOM"
ticker2 = "CVX"
data, beta, adf_test_result, ecm_results, ou_params = analyze_cointegration(ticker1, ticker2)

# Gold commodity and Gold futures
ticker1 = "GLD"
ticker2 = "GC=F"
data, beta, adf_test_result, ecm_results, ou_params = analyze_cointegration(ticker1, ticker2)

# Apple and Microsoft - starting analysis 2014
ticker1 = "AAPL"
ticker2 = "MSFT"
data, beta, adf_test_result, ecm_results, ou_params = analyze_cointegration(ticker1, ticker2, plotting=True)

# Apple and Microsoft - starting analysis 2022
data, beta, adf_test_result, ecm_results, ou_params = analyze_cointegration(ticker1, ticker2, plotting=True,
                                                                            start_date="2022-01-01")
