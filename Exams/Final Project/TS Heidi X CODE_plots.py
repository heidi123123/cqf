import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt


def analyze_cointegration(ticker1, ticker2, start_date="2014-01-01", end_date=""):
    # Download data
    tickers = [ticker1, ticker2]
    data = yf.download(tickers, start=start_date, end=end_date)["Adj Close"]

    # Normalize the prices
    data = data / data.iloc[0]

    # Regress ticker1 on ticker2 to find the cointegration coefficient
    X = sm.add_constant(data[ticker2])
    y = data[ticker1]
    model = sm.OLS(y, X).fit()
    beta = model.params[ticker2]

    # Calculate residuals
    data['residuals'] = data[ticker1] - beta * data[ticker2]

    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data['residuals'], label='Residuals')
    plt.axhline(data['residuals'].mean(), color='red', linestyle='--', label='Mean $\mu$')
    plt.axhline(data['residuals'].mean() + 1.1 * data['residuals'].std(), color='green', linestyle='--',
                label=f'1.1$\sigma$')
    plt.axhline(data['residuals'].mean() - 1.1 * data['residuals'].std(), color='green', linestyle='--')
    plt.title(f"Residuals of {ticker1} and {ticker2}")
    plt.legend()
    plt.show()

    # ADF test for stationarity of residuals
    adf_test = sm.tsa.adfuller(data['residuals'])
    print(f"ADF Statistic: {adf_test[0]}")
    print(f"p-value: {adf_test[1]}")


# Example usage
analyze_cointegration("KO", "PEP")
analyze_cointegration("ROG.SW", "NOVN.SW", start_date="2010-01-01", end_date="")
