# Ignore warnings
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller
from TS_backtesting import Portfolio, backtest_strategy_for_z_values
from TS_plots import plot_assets_and_residuals, plot_positions, plot_asset_prices_and_residuals, plot_pnl_against_index


def download_data(ticker, start_date="2019-01-01"):
    """Download ticker data from yfinance library."""
    df = yf.download(ticker, start=start_date, end="2024-08-19")
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


def split_data(data, split_ratio=0.8):
    """Splits the input data into training and testing subsets based on the provided split ratio."""
    split_index = int(len(data) * split_ratio)
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]
    return train_data, test_data


def calculate_test_residuals(data, beta, ticker1, ticker2):
    """Calculate the residuals for a given dataset using the provided beta vector."""
    y = data[ticker1].values
    X = data[ticker2].values
    X_with_intercept = np.hstack([np.ones((X.shape[0], 1)), X.reshape(-1, 1)])
    residuals = y - X_with_intercept @ beta
    return pd.Series(residuals, index=data.index)


def least_squares_regression(y, X):
    """Perform least squares regression to obtain beta coefficients and residuals."""
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # add y-intercept to X
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)  # least squares regression for beta
    residuals = y - X @ beta
    return beta, residuals


def perform_adf_test(residuals, significance_level, maxlag=None):
    """Perform the Augmented Dickey-Fuller (ADF) test to check for the presence of unit root in a time series.
    H0: time series has a unit root (i.e. non-stationary)"""
    adf_test = adfuller(residuals, maxlag=maxlag, autolag=None)
    # autolag=None will set lag nbr to maxlag (no lag optimization) if maxlag is not None
    # if maxlag and autolag are both None, then lag nbr will be optimized using AIC
    adf_statistic, p_value, lags = adf_test[0], adf_test[1], adf_test[2]

    print(f"ADF Statistic: {adf_statistic:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Number of lags: {lags}")

    if p_value < significance_level:
        print(f"The residuals are stationary (reject null hypothesis) "
              f"at the {significance_level * 100}% significance level.")
    else:
        print(f"The residuals are not stationary (fail to reject null hypothesis) "
              f"at the {significance_level * 100}% significance level.")
    return adf_test


def perform_engle_granger_step1(ticker1, ticker2, index_ticker, train_data, test_data,
                                plotting, significance_level, maxlag):
    """Step1 of the Engle-Granger procedure."""

    # OLS regression to obtain regression coefficients beta & residuals
    y = train_data[ticker1].values
    X = train_data[ticker2].values.reshape(-1, 1)
    beta, residuals = least_squares_regression(y, X)
    train_data['residuals'] = residuals
    test_data['residuals'] = calculate_test_residuals(test_data, beta, ticker1, ticker2)

    if plotting:  # plot normalized asset prices and residuals
        plot_assets_and_residuals(train_data, test_data, ticker1, ticker2, index_ticker)

    # perform ADF test
    adf_test_result = perform_adf_test(train_data['residuals'], significance_level, maxlag)
    return train_data, test_data, beta, adf_test_result


def get_differences(data, columns):
    """Calculate the returns (differences) Delta y_t = y_t-y_{t-1} for the specified columns in the dataframe."""
    return data[columns].diff().dropna()


def fit_ecm(data, target_column, independent_column, lag_size=1):
    """Step2 of the Engle-Granger procedure: fit the Equilibrium Correction Model (ECM)."""
    data_delta = get_differences(data, [target_column, independent_column])
    data_delta['lagged_residuals'] = data['residuals'].shift(lag_size)  # lag the residuals
    data_delta = data_delta.dropna()

    # OLS to obtain ECM coefficients & residuals
    y = data_delta[target_column].values
    X = data_delta[[independent_column, "lagged_residuals"]].values
    ecm_coefficients, ecm_residuals = least_squares_regression(y, X)

    ecm_residuals = pd.DataFrame(ecm_residuals, index=data_delta.index, columns=["ECM_residuals"])  # convert to pd.df
    return {'coefficients': ecm_coefficients, 'residuals': ecm_residuals}


def ou_likelihood(params, residuals, dt):
    """Calculates the negative log-likelihood of an Ornstein-Uhlenbeck process"""
    theta, mu_e, sigma_ou = params
    likelihood = 0
    for t in range(1, len(residuals)):
        mean = residuals[t-1] + theta * (mu_e - residuals[t-1]) * dt
        variance = sigma_ou**2 * dt
        # increment the log-likelihood by normal log-pdf of the next residual using mean and variance
        likelihood += norm.logpdf(residuals[t], loc=mean, scale=np.sqrt(variance))
    return -likelihood


def estimate_ou_params(residuals, dt=1):  # dt = 1: daily prices, so usually time increment dt = 1
    """Estimate Ornstein-Uhlenbeck process parameters using maximum likelihood estimation.
    The OU process is given as: d(residuals)_t = -theta (residuals_t-mu_e) dt + sigma_ou dW_t"""
    residuals = np.array(residuals)
    initial_params = [0.1, np.mean(residuals), np.std(residuals)]  # [theta0, mu_ou0, sigma_ou0]
    # we minimize negative log-likelihood, which is equivalent to using maximum likelihood estimator (MLE)
    result = minimize(ou_likelihood, initial_params, args=(residuals, dt), method="L-BFGS-B")
    theta, mu_e, sigma_ou = result.x
    return theta, mu_e, sigma_ou


def get_half_life(theta, dt=1):
    """Calculate the half-life of an Ornstein-Uhlenbeck process."""
    half_life = np.log(2) / (theta * dt)
    return half_life


def analyze_cointegration(ticker1, ticker2, index_ticker="SPY",
                          plotting=False, start_date="2019-01-01",
                          significance_level=0.05, maxlag=None):
    """Analyze cointegration between two assets ticker1 & ticker2 after start_date <YYYY-MM-DD>."""
    print(f"-" * 100)
    print(f"Analyzing cointegration between {ticker1} and {ticker2}...")

    df1 = download_data(ticker1, start_date)
    df2 = download_data(ticker2, start_date)
    data = prepare_time_series(df1, df2, ticker1, ticker2, index_ticker)

    # test / train split:
    train_data, test_data = split_data(data, split_ratio=0.7)

    # Engle-Granger procedure - Step 1
    train_data, test_data, beta, adf_test_result = perform_engle_granger_step1(ticker1, ticker2, index_ticker,
                                                                               train_data, test_data, plotting,
                                                                               significance_level, maxlag)
    # Engle-Granger procedure - Step 2: ECM
    lag_size = adf_test_result[2]
    ecm_results = fit_ecm(train_data, ticker1, ticker2, lag_size)
    print(f"Equilibrium mean-reversion coefficient: {ecm_results['coefficients'][-1]:2f}")

    # Engle-Granger procedure - Step 3 (inofficial): fit OU process to mean-reverting residuals
    theta, mu_e, sigma_ou = estimate_ou_params(train_data['residuals'])
    print(f"Estimated OU parameters: theta={theta:.4f}, mu_e={mu_e:.4f}, sigma_ou={sigma_ou:.4f}")
    print(f"Half-life of OU process: {get_half_life(theta):.2f} days")
    ou_params = {'theta': theta, 'mu_e': mu_e, 'sigma_ou': sigma_ou}
    return train_data, test_data, beta, adf_test_result, ecm_results, ou_params


def analyze_trading_strategy(train_data, test_data, ticker1, ticker2,
                             ou_params, hedge_ratio, index_ticker="SPY",
                             test_z_values=np.arange(0.3, 1.5, 0.1)):
    # Backtesting: in-sample performance evaluation on train_data to find best z-value
    train_results, z_best = backtest_strategy_for_z_values(train_data, ticker1, ticker2, ou_params, hedge_ratio, test_z_values)

    # Implement + analyze strategy with optimized z=z_best on test_data
    test_portfolio = Portfolio(test_data, ticker1, ticker2, ou_params, hedge_ratio, z=z_best)
    plot_positions(test_portfolio)
    plot_asset_prices_and_residuals(test_portfolio)
    plot_pnl_against_index(test_portfolio, index_ticker)


if __name__ == "__main__":
    # Example usages
    # Coca-Cola and Pepsi
    ticker1 = "KO"
    ticker2 = "PEP"
    train_data, test_data, beta, adf_test_result, ecm_results, ou_params = analyze_cointegration(ticker1, ticker2,
                                                                                                 significance_level=0.05,
                                                                                                 start_date="2020-01-01",
                                                                                                 maxlag=1)
    analyze_trading_strategy(train_data, test_data, ticker1, ticker2, ou_params, beta[1])

    """Marriott and InterContinental Hotels Group
    ticker# 1 = "MAR"
    ticker2 = "IHG"
    train_data, test_data, beta, adf_test_result, ecm_results, ou_params = analyze_cointegration(ticker1, ticker2)
    
    # Exxon Mobil and Chevron
    ticker1 = "XOM"
    ticker2 = "CVX"
    train_data, test_data, beta, adf_test_result, ecm_results, ou_params = analyze_cointegration(ticker1, ticker2)
    
    # Gold commodity and Gold futures
    ticker1 = "GLD"
    ticker2 = "GC=F"
    train_data, test_data, beta, adf_test_result, ecm_results, ou_params = analyze_cointegration(ticker1, ticker2)
    
    
    # Apple and Microsoft - starting analysis 2019
    ticker1 = "AAPL"
    ticker2 = "MSFT"
    train_data, test_data, beta, adf_test_result, ecm_results, ou_params = analyze_cointegration(ticker1, ticker2,
                                                                                                 plotting=True)
    
    # Apple and Microsoft - starting analysis 2022
    train_data, test_data, beta, adf_test_result, ecm_results, ou_params = analyze_cointegration(ticker1, ticker2,
                                                                                                 start_date="2021-01-01",
                                                                                                 plotting=True)
    analyze_trading_strategy(train_data, test_data, ticker1, ticker2, ou_params, beta[1])
    
    # Lonza and Givaudan
    ticker1 = "LONN.SW"
    ticker2 = "GIVN.SW"
    train_data, test_data, beta, adf_test_result, ecm_results, ou_params = analyze_cointegration(ticker1, ticker2,
                                                                                                 index_ticker="^SSMI",
                                                                                                 plotting=True,
                                                                                                 start_date="2017-01-01")
    """
    # Nestlé and Roche
    ticker1 = "NESN.SW"
    ticker2 = "ROG.SW"
    train_data, test_data, beta, adf_test_result, ecm_results, ou_params = analyze_cointegration(ticker1, ticker2,
                                                                                                 index_ticker="^SSMI",
                                                                                                 plotting=True,
                                                                                                 start_date="2017-01-01")
    analyze_trading_strategy(train_data, test_data, ticker1, ticker2, ou_params, beta[1],  index_ticker="^SSMI")
