import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def prepare_dataframe(ticker, start_date="2014-01-01"):
    df = yf.download(ticker, start=start_date)
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df


def normalize_prices(data1, data2, label1, label2):
    data = pd.concat([data1['Close'], data2['Close']], axis=1).dropna()
    data.columns = [label1, label2]
    data = data / data.iloc[0]  # normalize prices
    return data


def least_sqrs_regression(data, label1, label2):
    # Extracting data and adding a constant term
    y = data[label1].values
    X = data[label2].values
    X = np.vstack([np.ones(len(X)), X]).T  # add a constant term

    # Perform the least squares regression using the normal equation
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)

    # Calculate residuals
    data['residuals'] = y - (beta[0] + beta[1] * data[label2])
    return data, beta


def plot_assets_and_residuals(data, asset1_name, asset2_name):
    plt.figure(figsize=(12, 8))

    # Plot normalized prices
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data[asset1_name], label=f'{asset1_name} Normalized', color='blue')
    plt.plot(data.index, data[asset2_name], label=f'{asset2_name} Normalized', color='orange')
    plt.title(f"Normalized Prices of {asset1_name} and {asset2_name}")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid(True)

    # Plot residuals
    plt.subplot(2, 1, 2)
    plt.plot(data.index, data['residuals'], label='Residuals', color='green')
    plt.axhline(data['residuals'].mean(), color='red', linestyle='--', label='Mean $\mu$')
    plt.axhline(data['residuals'].mean() + 1.1 * data['residuals'].std(), color='purple', linestyle='--',
                label='$1.1*\sigma$')
    plt.axhline(data['residuals'].mean() - 1.1 * data['residuals'].std(), color='purple', linestyle='--')
    plt.title(f"Residuals of {asset1_name} and {asset2_name}")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def perform_adf_test(residuals, significance_level=0.01):
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


def analyze_cointegration(ticker1, ticker2, start_date="2014-01-01"):
    # Prepare data
    df1 = prepare_dataframe(ticker1, start_date)
    df2 = prepare_dataframe(ticker2, start_date)

    # Normalize prices
    data = normalize_prices(df1, df2, ticker1, ticker2)

    # Perform ordinary least square (OLS) regression
    data, beta = least_sqrs_regression(data, ticker1, ticker2)

    # Plot residuals
    plot_assets_and_residuals(data, ticker1, ticker2)

    # Perform ADF test
    adf_test_result = perform_adf_test(data['residuals'])
    return data, beta, adf_test_result


# Example usage
# Coca-Cola and Pepsi
analyze_cointegration("KO", "PEP")

# Roche and Novartis
analyze_cointegration("ROG.SW", "NOVN.SW", start_date="2022-01-01")

# Marriott and InterContinental Hotels Group
analyze_cointegration("MAR", "IHG")

# Exxon Mobil and Chevron
analyze_cointegration("XOM", "CVX")

# Gold commodity and Gold futures (example tickers)
analyze_cointegration("GC=F", "GLD")
