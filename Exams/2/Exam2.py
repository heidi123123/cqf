import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


S0 = 100
K = 100
t = 1
sigma = 0.2
r = 0.05
dt = 0.01
number_of_mc_paths = 50000


def simulate_stock_price(S0, r, sigma, t, dt, number_of_mc_paths=10000):
    """Simulate stock price using Euler-Maruyama method."""
    N = int(t / dt)  # number of time steps
    S = np.zeros((number_of_mc_paths, N))
    S[:, 0] = S0
    for i in range(1, N):
        dW = np.random.standard_normal(number_of_mc_paths)  # Brownian increment
        S[:, i] = S[:, i-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * dW)  # Euler-Maruyama step
    return S


def plot_stock_price_simulations(S, NN=1000):
    """Plot the first NN stock price simulations."""
    plt.figure(figsize=(10, 6))
    for i in range(NN):  # plot the first NN paths
        plt.plot(S[i, :])
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Movements Simulated Using Euler-Maruyama Method')
    plt.show()


def asian_option(S, K, r, t):
    avg = np.mean(S, axis=1)  # mean stock price over the path
    asian_call = np.exp(-r * t) * np.maximum(avg - K, 0)
    asian_put = np.exp(-r * t) * np.maximum(K - avg, 0)
    return np.mean(asian_call), np.mean(asian_put)


def lookback_option(S, K, r, t, fixed_strike=1):
    min_S = np.min(S, axis=1)  # minimum stock price over the path
    max_S = np.max(S, axis=1)  # maximum stock price over the path
    if fixed_strike:
        lookback_call = np.exp(-r * t) * np.maximum(max_S - K, 0)
        lookback_put = np.exp(-r * t) * np.maximum(K - min_S, 0)
    else:  # calculate the option payoffs for floating strike now
        lookback_call = np.exp(-r * t) * np.maximum(S[:, -1] - min_S, 0)
        lookback_put = np.exp(-r * t) * np.maximum(max_S - S[:, -1], 0)
    return np.mean(lookback_call), np.mean(lookback_put)


def plot_option_prices_vs_stock_prices(S, option_prices, option_names):
    """Plot scatter plots of option prices vs. underlying stock prices"""
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.ravel()
    for i in range(len(option_names)):
        # Scatter plot of option prices vs. underlying stock prices
        axs[i].scatter(S[:, -1], option_prices[i], color='blue', alpha=0.5, label='Option Prices')

        # Calculate the mean option price for each bin of stock prices
        df = pd.DataFrame({'StockPrice': S[:, -1], 'OptionPrice': option_prices[i]})
        df['StockPriceBin'] = pd.cut(df['StockPrice'],
                                     bins=np.linspace(df['StockPrice'].min(), df['StockPrice'].max(), 100))
        df_grouped = df.groupby('StockPriceBin').mean()

        # Plot the mean option price for each bin
        axs[i].plot(df_grouped['StockPrice'], df_grouped['OptionPrice'], color='red', linewidth=2,
                    label='Mean Option Price')

        axs[i].set_xlabel('Underlying Stock Price')
        axs[i].set_ylabel('Option Price')
        axs[i].set_title(f'{option_names[i]} Prices vs. Underlying Stock Prices (Monte Carlo Simulation)')
        axs[i].legend()
    plt.show()


def summarize_option_prices(option_prices, option_names):
    """Calculate prices for Asian and Lookback call & put options."""
    avg_option_prices = [np.mean(price) for price in option_prices]
    for i in range(len(option_names)):
        print(f"The average price for the {option_names[i]} is {avg_option_prices[i]:.2f}")
    plt.figure(figsize=(10, 6))
    plt.bar(option_names, avg_option_prices, color='blue')
    plt.xlabel('Option Type')
    plt.ylabel('Average Option Price')
    plt.title('Average Option Prices')
    plt.show()


def plot_stock_simulations(S):
    """Plot the first 1000 stock price simulations along with the average, minimum, and maximum."""
    plt.figure(figsize=(10, 6))
    for i in range(min(1000, S.shape[0])):  # plot the first 1000 paths or less
        plt.plot(S[i, :], color='gray', linewidth=0.5, linestyle='dashed')

    # Plot significant functions
    avg_S = np.mean(S, axis=0)
    min_S = np.min(S, axis=0)
    max_S = np.max(S, axis=0)
    plt.plot(avg_S, label='Average Stock Price')
    plt.plot(min_S, label='Minimum Stock Price')
    plt.plot(max_S, label='Maximum Stock Price')

    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.title('Stock Price Simulations and Significant Functions')
    plt.legend()
    plt.show()


def plot_option_prices(S, K, r, t):
    """Plot the prices of different options."""
    asian_call, asian_put = asian_option(S, K, r, t)
    lookback_fixed_call, lookback_fixed_put = lookback_option(S, K, r, t, fixed_strike=1)
    lookback_float_call, lookback_float_put = lookback_option(S, r, t, t, fixed_strike=0)

    plt.figure(figsize=(10, 6))
    labels = ['Asian Call', 'Asian Put',
              'Lookback Fixed\nStrike Call', 'Lookback Fixed\nStrike Put',
              'Lookback Floating\nStrike Call', 'Lookback Floating\nStrike Put']
    prices = [asian_call, asian_put, lookback_fixed_call, lookback_fixed_put, lookback_float_call, lookback_float_put]
    plt.bar(labels, prices)
    plt.ylabel('Option Price')
    plt.title(f'Option Price\nS0={S0}, K={K}, t={t}, sigma={sigma}, r={r}')
    plt.xticks(rotation=45)  # Rotate x-axis labels
    plt.tight_layout()  # Adjust layout to prevent labels from being cut off
    plt.show()


def main():
    S = simulate_stock_price(S0, r, sigma, t, dt, number_of_mc_paths)
    plot_stock_simulations(S)
    plot_option_prices(S, K, r, t)


if __name__ == "__main__":
    main()
