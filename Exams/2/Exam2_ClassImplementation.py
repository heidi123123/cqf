import numpy as np
import matplotlib.pyplot as plt


class OptionPricer:
    def __init__(self, S0, K, r, t, dt, sigma, number_of_mc_paths):
        self.S0 = S0
        self.K = K
        self.r = r
        self.t = t
        self.dt = dt
        self.sigma = sigma
        self.number_of_mc_paths = number_of_mc_paths
        self.S = self.simulate_stock_price()

    def simulate_stock_price(self):
        """Simulate stock price using Euler-Maruyama method."""
        N = int(self.t / self.dt)  # number of time steps
        S = np.zeros((self.number_of_mc_paths, N))
        S[:, 0] = self.S0
        for i in range(1, N):
            dW = np.random.standard_normal(self.number_of_mc_paths)  # Brownian increment
            S[:, i] = S[:, i-1] * np.exp((self.r - 0.5 * self.sigma ** 2) * self.dt + self.sigma * np.sqrt(self.dt) * dW)  # Euler-Maruyama step
        return S

    def asian_option(self):
        """Calculate Asian call + put option prices by discounting their expected payoffs."""
        avg = np.mean(self.S, axis=1)  # mean stock price over the path
        asian_call = np.exp(-self.r * self.t) * np.maximum(avg - self.K, 0)
        asian_put = np.exp(-self.r * self.t) * np.maximum(self.K - avg, 0)
        return np.mean(asian_call), np.mean(asian_put)

    def lookback_option(self, fixed_strike=1):
        """Calculate Lookback call + put option prices by discounting their expected payoffs."""
        min_S = np.min(self.S, axis=1)  # minimum stock price over all simulations
        max_S = np.max(self.S, axis=1)  # maximum stock price over all simulations
        if fixed_strike:
            lookback_call = np.exp(-self.r * self.t) * np.maximum(max_S - self.K, 0)
            lookback_put = np.exp(-self.r * self.t) * np.maximum(self.K - min_S, 0)
        else:  # calculate the option payoffs for floating strike now
            lookback_call = np.exp(-self.r * self.t) * np.maximum(self.S[:, -1] - min_S, 0)
            lookback_put = np.exp(-self.r * self.t) * np.maximum(max_S - self.S[:, -1], 0)
        return np.mean(lookback_call), np.mean(lookback_put)

    def plot_stock_simulations(self, plot_avg_min_max=0, NN=1000):
        """Plot the first NN stock price simulations along with the average, minimum and maximum."""
        plt.figure(figsize=(10, 6))
        for i in range(NN):  # plot the first NN paths
            plt.plot(self.S[i, :], color='gray', linewidth=0.5, linestyle='dashed')

        if plot_avg_min_max:  # plot other functions of the stock price simulations, like average, min + max
            avg_S = np.mean(self.S, axis=0)
            min_S = np.min(self.S, axis=0)
            max_S = np.max(self.S, axis=0)
            plt.plot(avg_S, label='Average Stock Price')
            plt.plot(min_S, label='Minimum Stock Price')
            plt.plot(max_S, label='Maximum Stock Price')

        plt.xlabel('Time Steps')
        plt.ylabel('Stock Price')
        plt.title(f'First {NN} Stock Price Simulations')
        plt.legend()
        plt.show()

    def calculate_option_prices(self):
        """Calculate the prices of different options types."""
        # Calculate option prices
        asian_call, asian_put = self.asian_option()
        lookback_fixed_call, lookback_fixed_put = self.lookback_option(fixed_strike=1)
        lookback_float_call, lookback_float_put = self.lookback_option(fixed_strike=0)
        return asian_call, asian_put, lookback_fixed_call, lookback_fixed_put, lookback_float_call, lookback_float_put

    def plot_option_prices(self):
        """Plot the prices of different options types."""
        # Plot option prices in a bar diagram
        asian_call, asian_put, lookback_fixed_call, lookback_fixed_put, lookback_float_call, lookback_float_put = self.calculate_option_prices()

        plt.figure(figsize=(10, 6))
        labels = ['Asian Call', 'Asian Put',
                  'Lookback Call\nFixed Strike', 'Lookback Put\nFixed Strike',
                  'Lookback Call\nFloating Strike', 'Lookback Put\nFloating Strike']
        prices = [asian_call, asian_put, lookback_fixed_call, lookback_fixed_put, lookback_float_call, lookback_float_put]
        bars = plt.bar(labels, prices)

        # Annotate bars with prices
        for bar in bars:
            option_price = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, option_price + 0.01, round(option_price, 2),
                     ha='center', va='bottom', color='blue', weight='bold')

        plt.ylabel('Option Price')
        plt.title(f'Option Price\nS0={self.S0}, K={self.K}, t={self.t}, sigma={self.sigma}, r={self.r}')
        plt.xticks()
        plt.tight_layout()  # Adjust layout to prevent labels from being cut off
        plt.show()


def plot_option_prices_vs_param(param_name, param_values, option_prices, log_x_axis=False):
    """Plot the option prices for different values of a parameter."""
    plt.figure(figsize=(10, 6))

    colors = ['b', 'g', 'r', 'c', 'm', 'y']  # define order of colors for the plots, so option price scatter dot
    # has same color as the connecting line

    for i, prices in enumerate(zip(*option_prices)):  # transpose option_prices list
        for value, price in zip(param_values, prices):
            plt.scatter(value, price, marker='o', color=colors[i])  # plot option price as scatter point
            plt.text(value, price, f"{param_name}={str(value)}", fontsize=8)  # add text annotation for each point

        # plot connecting lines for the scatter points of each option type, color=... ensures same color is used
        label_list = ['Asian Call', 'Asian Put',
                      'Lookback Fixed Call', 'Lookback Fixed Put',
                      'Lookback Float Call', 'Lookback Float Put']
        plt.plot(param_values, prices, label=label_list[i], color=colors[i])

    plt.xlabel(param_name)
    plt.ylabel('Option Price')
    plt.title(f'Option Prices vs {param_name}')
    if log_x_axis:
        plt.xscale('log')  # Set x-axis to logarithmic scale
    plt.legend()
    plt.show()


def main():
    param_name = "S0"
    param_values = np.linspace(50, 150, 10 + 1)  # an equidistant list of 10 values for S0 from 50 to 150
    option_prices = []
    for param_value in param_values:
        op = OptionPricer(S0=param_value, K=100, r=0.05, t=1, dt=0.01, sigma=0.2, number_of_mc_paths=50000)
        option_prices.append(op.calculate_option_prices())
    plot_option_prices_vs_param(param_name, param_values, option_prices, log_x_axis=False)


if __name__ == "__main__":
    main()
