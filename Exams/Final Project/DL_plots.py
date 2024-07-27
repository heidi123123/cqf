from scipy import stats
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np


# Prepare the data
def prepare_dataframe(ticker):
    df = yf.download(ticker, start="2018-01-01")
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df


def visualize_return_data(df, ticker):
    # Visualize the return data
    plt.figure(figsize=(11, 4))
    plt.plot(df['Return'], label='Return')
    plt.title(f'Relative Return of Close Prices of {ticker} stock')
    plt.xlabel('Time')
    plt.ylabel('Return')
    plt.legend()
    plt.show()


def compare_returns_and_norm_dist(df):
    # Compare to normal distribution
    mu = df['Return'].mean()
    sigma = df['Return'].std()
    kurtosis = stats.kurtosis(df['Return'].dropna())
    plt.figure(figsize=(9, 4))
    plt.hist(df['Return'], density=True, bins=80, label=f"Returns\nKurtosis={kurtosis:.4f}", alpha=0.5)

    # following 68–95–99.7 rule, 3 stdevs from mean contain 99.7% of normally distributed values
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x, stats.norm(mu, sigma).pdf(x), label=f"Normal PDF for\n$N(\mu=${mu:.4f}$, \sigma^2=${sigma**2:.4f})")

    plt.xlabel("Returns")
    plt.ylabel("Frequency")
    plt.title("Return Histogram vs. Normal Distribution")
    plt.legend()

    xticks = np.linspace(mu - 3*sigma, mu + 3*sigma, 7)
    plt.xticks(xticks, labels=[f"{i}$\sigma$" for i in range(-3, 4)])
    plt.grid()
    plt.show()


if __name__ == "__main__":
    ticker = "TSLA"
    df = prepare_dataframe(ticker)
    # visualize_return_data(df, ticker)
    compare_returns_and_norm_dist(df)
