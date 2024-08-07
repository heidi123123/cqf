# Ignore warnings
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def download_data(ticker, start_date="2014-01-01"):
    # download ticker data from yfinance library
    df = yf.download(ticker, start=start_date)
    df.dropna(inplace=True)
    return df


def prepare_time_series(data1, data2, label1, label2, normalize):
    data = pd.concat([data1['Close'], data2['Close']], axis=1).dropna()
    data.columns = [label1, label2]
    if normalize:
        data = data / data.iloc[0]  # normalize prices
    return data


def least_squares_regression(y, X):
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # add y-intercept to X
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)  # least sqaures regression into coefficient vector beta

    # calculate residuals
    y_pred = X @ beta
    residuals = y - y_pred
    return beta, residuals


def plot_assets_and_residuals(data, asset1_name, asset2_name):
    plt.figure(figsize=(12, 8))

    # price time series plot
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data[asset1_name], label=f"{asset1_name}", color="blue")
    plt.plot(data.index, data[asset2_name], label=f"{asset2_name}", color="orange")
    plt.title(f"Historical Prices of {asset1_name} and {asset2_name}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    # residuals plot
    plt.subplot(2, 1, 2)
    plt.plot(data.index, data['residuals'], label="Residuals", color="green")
    mean = data['residuals'].mean()
    stdev = data['residuals'].std()
    plt.axhline(mean, color="red", linestyle='--', label=f"Mean $\mu$")
    plt.axhline(mean + 1.1 * stdev, color="purple", linestyle="--", label="$\pm1.1*\sigma$")
    plt.axhline(mean - 1.1 * stdev, color="purple", linestyle="--")
    plt.title(f"Residuals of {asset1_name} and {asset2_name}")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def perform_adf_test(residuals, significance_level):
    # Augmented Dickey-Fuller (ADF) test to check for the presence of unit root in a time series
    # H0: time series has a unit root (i.e. non-stationary)

    adf_test = adfuller(residuals)
    adf_statistic, p_value = adf_test[0], adf_test[1]

    print(f"ADF Statistic: {adf_statistic}")
    print(f"p-value: {p_value}")

    if p_value < significance_level:
        print(f"The residuals are stationary (reject null hypothesis) "
              f"at the {significance_level * 100}% significance level.")
    else:
        print(f"The residuals are not stationary (accept null hypothesis) "
              f"at the {significance_level * 100}% significance level.")

    return adf_test


def analyze_cointegration(ticker1, ticker2, start_date="2014-01-01", significance_level=0.05):
    print(f"-" * 100)
    print(f"Analyzing cointegration between {ticker1} and {ticker2}...")

    # Download data
    df1 = download_data(ticker1, start_date)
    df2 = download_data(ticker2, start_date)

    # Prepare prices
    data = prepare_time_series(df1, df2, ticker1, ticker2, normalize=False)

    # Perform ordinary least square (OLS) regression
    y = data[ticker1].values
    X = data[ticker2].values.reshape(-1, 1)
    beta, residuals = least_squares_regression(y, X)
    data['residuals'] = residuals

    # Plot residuals
    plot_assets_and_residuals(data, ticker1, ticker2)

    # Perform ADF test
    adf_test_result = perform_adf_test(data['residuals'], significance_level)

    return data, beta, adf_test_result


# Example usages
# Coca-Cola and Pepsi
ticker1 = "KO"
ticker2 = "PEP"
data, beta, adf_test_result = analyze_cointegration(ticker1, ticker2, significance_level=0.01)

# Roche and Novartis
ticker1 = "ROG.SW"
ticker2 = "NOVN.SW"
data, beta, adf_test_result = analyze_cointegration(ticker1, ticker2, start_date="2022-01-01")

# Marriott and InterContinental Hotels Group
ticker1 = "MAR"
ticker2 = "IHG"
data, beta, adf_test_result = analyze_cointegration(ticker1, ticker2)

# Exxon Mobil and Chevron
ticker1 = "XOM"
ticker2 = "CVX"
data, beta, adf_test_result = analyze_cointegration(ticker1, ticker2)

# Gold commodity and Gold futures
ticker1 = "GLD"
ticker2 = "GC=F"
data, beta, adf_test_result = analyze_cointegration(ticker1, ticker2)
