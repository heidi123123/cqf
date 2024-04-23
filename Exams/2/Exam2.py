import numpy as np
import matplotlib.pyplot as plt


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


def asian_option(S, K, r, t, fixed_strike=1):
    """Calculate Asian call + put option prices by discounting their expected payoffs."""
    avg = np.mean(S, axis=1)  # mean stock price over all simulations
    if fixed_strike:
        asian_call = np.exp(-r * t) * np.maximum(avg - K, 0)
        asian_put = np.exp(-r * t) * np.maximum(K - avg, 0)
    else:  # calculate the option payoffs for floating strike now
        asian_call = np.exp(-r * t) * np.maximum(S[:, -1] - avg, 0)
        asian_put = np.exp(-r * t) * np.maximum(avg - S[:, -1], 0)
    return np.mean(asian_call), np.mean(asian_put)


def lookback_option(S, K, r, t, fixed_strike=1):
    """Calculate Lookback call + put option prices by discounting their expected payoffs."""
    min_S = np.min(S, axis=1)  # minimum stock price over all simulations
    max_S = np.max(S, axis=1)  # maximum stock price over all simulations
    if fixed_strike:
        lookback_call = np.exp(-r * t) * np.maximum(max_S - K, 0)
        lookback_put = np.exp(-r * t) * np.maximum(K - min_S, 0)
    else:  # calculate the option payoffs for floating strike now
        lookback_call = np.exp(-r * t) * np.maximum(S[:, -1] - min_S, 0)
        lookback_put = np.exp(-r * t) * np.maximum(max_S - S[:, -1], 0)
    return np.mean(lookback_call), np.mean(lookback_put)


def plot_stock_simulations(S, plot_avg_min_max=0, NN=1000):
    """Plot the first NN stock price simulations along with the average, minimum and maximum."""
    plt.figure(figsize=(10, 6))
    for i in range(NN):  # plot the first NN paths
        plt.plot(S[i, :], color='gray', linewidth=0.5, linestyle='dashed')

    if plot_avg_min_max:  # plot other functions of the stock price simulations, like average, min + max
        avg_S = np.mean(S, axis=0)
        min_S = np.min(S, axis=0)
        max_S = np.max(S, axis=0)
        plt.plot(avg_S, label='Average Stock Price')
        plt.plot(min_S, label='Minimum Stock Price')
        plt.plot(max_S, label='Maximum Stock Price')

    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.title(f'First {NN} Stock Price Simulations')
    plt.legend()
    plt.show()


def plot_option_prices(S, K, r, t):
    """Calculate + plot the prices of different options types."""
    # Calculate option prices + plot them in a bar diagram
    asian_fixed_call, asian_fixed_put = asian_option(S, K, r, t, fixed_strike=1)
    asian_float_call, asian_float_put = asian_option(S, K, r, t, fixed_strike=0)
    lookback_fixed_call, lookback_fixed_put = lookback_option(S, K, r, t, fixed_strike=1)
    lookback_float_call, lookback_float_put = lookback_option(S, K, r, t, fixed_strike=0)

    plt.figure(figsize=(10, 6))
    labels = ['Asian Call\nFixed Strike', 'Asian Put\nFixed Strike',
              'Asian Call\nFloating Strike', 'Asian Put\nFloating Strike',
              'Lookback Call\nFixed Strike', 'Lookback Put\nFixed Strike',
              'Lookback Call\nFloating Strike', 'Lookback Put\nFloating Strike']
    prices = [asian_fixed_call, asian_fixed_put, asian_float_call, asian_float_put,
              lookback_fixed_call, lookback_fixed_put, lookback_float_call, lookback_float_put]
    bars = plt.bar(labels, prices)

    # Annotate bars with prices
    for bar in bars:
        option_price = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, option_price + 0.01, round(option_price, 2),
                 ha='center', va='bottom', color='blue', weight='bold')

    plt.ylabel('Option Price')
    plt.title(f'Option Price\nS0={S0}, K={K}, t={t}, sigma={sigma}, r={r}')
    plt.xticks()
    plt.tight_layout()  # Adjust layout to prevent labels from being cut off
    plt.show()


def main():
    S = simulate_stock_price(S0, r, sigma, t, dt, number_of_mc_paths)
    # plot_stock_simulations(S, plot_avg_min_max=1, NN=1000)
    plot_option_prices(S, K, r, t)


if __name__ == "__main__":
    main()
