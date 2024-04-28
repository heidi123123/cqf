import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean


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

    def asian_option(self, arithmetic_avg=1, fixed_strike=1):
        """Calculate Asian call + put option prices by discounting their expected payoffs."""
        if arithmetic_avg:  # calculate arithmetic mean stock price over all simulations
            avg = np.mean(self.S, axis=1)
        else:  # calculate geometric average
            avg = gmean(self.S, axis=1)
        if fixed_strike:
            asian_call = np.exp(-self.r * self.t) * np.maximum(avg - self.K, 0)
            asian_put = np.exp(-self.r * self.t) * np.maximum(self.K - avg, 0)
        else:  # calculate the option payoffs for floating strike now
            asian_call = np.exp(-self.r * self.t) * np.maximum(self.S[:, -1] - avg, 0)
            asian_put = np.exp(-self.r * self.t) * np.maximum(avg - self.S[:, -1], 0)
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

    def calculate_option_prices(self):
        """Calculate the prices of different options types."""
        # Calculate option prices
        asian_fixed_call, asian_fixed_put = self.asian_option(arithmetic_avg=1, fixed_strike=1)
        asian_float_call, asian_float_put = self.asian_option(arithmetic_avg=1, fixed_strike=0)
        asian_g_fixed_call, asian_g_fixed_put = self.asian_option(arithmetic_avg=0, fixed_strike=1)
        asian_g_float_call, asian_g_float_put = self.asian_option(arithmetic_avg=0, fixed_strike=0)
        lookback_fixed_call, lookback_fixed_put = self.lookback_option(fixed_strike=1)
        lookback_float_call, lookback_float_put = self.lookback_option(fixed_strike=0)
        return asian_fixed_call, asian_fixed_put, asian_float_call, asian_float_put,\
            asian_g_fixed_call, asian_g_fixed_put, asian_g_float_call, asian_g_float_put,\
            lookback_fixed_call, lookback_fixed_put, lookback_float_call, lookback_float_put


def get_labels_for_plot(plot_all_asian_options):
    arithmetic_labels = ['Asian Arithmetic Avg.\nFixed Call', 'Asian Arithmetic Avg.\n Fixed Put',
                         'Asian Arithmetic Avg.\n Float Call', 'Asian Arithmetic Avg.\n Float Put']
    geometric_labels = ['Asian Geometric Avg.\n Fixed Call', 'Asian Geometric Avg.\n Fixed Put',
                        'Asian Geometric Avg.\n Float Call', 'Asian Geometric Avg.\n Float Put']
    lookback_labels = ['Lookback Fixed Call', 'Lookback Fixed Put', 'Lookback Float Call', 'Lookback Float Put']
    return arithmetic_labels + (geometric_labels if plot_all_asian_options else []) + lookback_labels


def plot_option_prices_vs_param(param_name, param_values, option_prices, plot_all_asian_options,
                                annotation_precision=2, log_x_axis=False):
    """Plot the option prices for different values of a parameter."""
    plt.figure(figsize=(12, 6))

    # define order of colors for the plots, so option price scatter dot has same color as the connecting line
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'rosybrown', 'darkorange', 'grey', 'lime', 'teal']
    for i, prices in enumerate(zip(*option_prices)):  # transpose option_prices list
        for value, price in zip(param_values, prices):
            plt.scatter(value, price, marker='o', color=colors[i])  # plot option price as scatter point
            # add text annotation for each point
            plt.text(value, price, f"{param_name}={str(round(value, annotation_precision))}", fontsize=8)

        # plot connecting lines for the scatter points of each option type, color=... ensures same color is used
        label_list = get_labels_for_plot(plot_all_asian_options)
        plt.plot(param_values, prices, label=label_list[i], color=colors[i])

    plt.xlabel(param_name)
    plt.ylabel('Option Price')
    plt.title(f'Option Prices vs {param_name}')
    if log_x_axis:
        plt.xscale('log')  # set x-axis to logarithmic scale
    plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))  # legend outside of graphs
    plt.tight_layout()
    plt.show()


def get_option_prices_to_plot(option_pricer_object, plot_all_asian_options):
    all_option_prices = list(option_pricer_object.calculate_option_prices())
    if not plot_all_asian_options:
        option_prices = all_option_prices[:4] + all_option_prices[8:]
    else:
        option_prices = all_option_prices
    return option_prices


def analyze_number_of_mc_paths(plot_all_asian_options=0):
    param_name = "N"
    param_values = [10, 30, 60, 100, 300, 500, 1000, 5000, 10000, 20000, 50000, 100000, 1000000]
    option_prices = []
    for param_value in param_values:
        op = OptionPricer(S0=100, K=100, r=0.05, t=1, dt=0.01, sigma=0.2, number_of_mc_paths=param_value)
        option_prices.append(get_option_prices_to_plot(op, plot_all_asian_options))
    plot_option_prices_vs_param(param_name, param_values, option_prices, plot_all_asian_options, log_x_axis=True)


def analyze_initial_value(plot_all_asian_options=0):
    param_name = "S0"
    param_values = np.linspace(50, 150, 10 + 1)  # equidistant list of 10 values for S0 from 50 to 150
    option_prices = []
    for param_value in param_values:
        op = OptionPricer(S0=param_value, K=100, r=0.05, t=1, dt=0.01, sigma=0.2, number_of_mc_paths=NUMBER_OF_MC_PATHS)
        option_prices.append(get_option_prices_to_plot(op, plot_all_asian_options))
    plot_option_prices_vs_param(param_name, param_values, option_prices, plot_all_asian_options, log_x_axis=False)


def analyze_strike(plot_all_asian_options=0):
    param_name = "K"
    param_values = np.linspace(50, 150, 10 + 1)  # equidistant list of 10 values for K from 50 to 150
    option_prices = []
    for param_value in param_values:
        op = OptionPricer(S0=100, K=param_value, r=0.05, t=1, dt=0.01, sigma=0.2, number_of_mc_paths=NUMBER_OF_MC_PATHS)
        option_prices.append(get_option_prices_to_plot(op, plot_all_asian_options))
    plot_option_prices_vs_param(param_name, param_values, option_prices, plot_all_asian_options)


def analyze_time_to_expiry(plot_all_asian_options=0):
    param_name = "t"
    param_values = np.linspace(0.01, 5, 10 + 1)  # equidistant list of 10 values for t (time to expiry) from 0 to 1 year
    option_prices = []
    for param_value in param_values:
        op = OptionPricer(S0=100, K=100, r=0.05, t=param_value, dt=0.01, sigma=0.2, number_of_mc_paths=NUMBER_OF_MC_PATHS)
        option_prices.append(get_option_prices_to_plot(op, plot_all_asian_options))
    plot_option_prices_vs_param(param_name, param_values, option_prices, plot_all_asian_options, annotation_precision=1)


def analyze_time_step(plot_all_asian_options=0):
    param_name = "dt"
    param_values = [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1]
    option_prices = []
    for param_value in param_values:
        op = OptionPricer(S0=100, K=100, r=0.05, t=1, dt=param_value, sigma=0.2, number_of_mc_paths=NUMBER_OF_MC_PATHS)
        option_prices.append(get_option_prices_to_plot(op, plot_all_asian_options))
    plot_option_prices_vs_param(param_name, param_values, option_prices, plot_all_asian_options, annotation_precision=4, log_x_axis=True)


def analyze_volatility(plot_all_asian_options=0):
    param_name = "sigma"
    param_values = np.linspace(0.01, 1, 10 + 1)  # an equidistant list of 10 values for sigma from 1% to 100%
    option_prices = []
    for param_value in param_values:
        op = OptionPricer(S0=100, K=100, r=0.05, t=1, dt=0.01, sigma=param_value, number_of_mc_paths=NUMBER_OF_MC_PATHS)
        if not plot_all_asian_options:
            all_option_prices = op.calculate_option_prices()
            option_prices.append(get_option_prices_to_plot(op, plot_all_asian_options))
    plot_option_prices_vs_param(param_name, param_values, option_prices, plot_all_asian_options)


if __name__ == "__main__":
    NUMBER_OF_MC_PATHS = 10000
    analyze_strike()
