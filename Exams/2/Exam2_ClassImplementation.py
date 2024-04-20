import numpy as np
import matplotlib.pyplot as plt


class OptionPricing:
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
        min_S = np.min(self.S, axis=1)  # minimum stock price over the path
        max_S = np.max(self.S, axis=1)  # maximum stock price over the path
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

    def plot_option_prices(self):
        """Plot the prices of different options types."""
        # Calculate option prices + plot them in a bar diagram
        asian_call, asian_put = self.asian_option()
        lookback_fixed_call, lookback_fixed_put = self.lookback_option(fixed_strike=1)
        lookback_float_call, lookback_float_put = self.lookback_option(fixed_strike=0)

        plt.figure(figsize=(10, 6))
        labels = ['Asian Call', 'Asian Put',
                  'Lookback Call\nFixed Strike', 'Lookback Put\nFixed Strike',
                  'Lookback Call\nFloating Strike', 'Lookback Put\nFloating Strike']
        prices = [asian_call, asian_put, lookback_fixed_call, lookback_fixed_put, lookback_float_call, lookback_float_put]
        bars = plt.bar(labels, prices)

        # Annotate bars with prices
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, round(yval, 2), ha='center', va='bottom')

        plt.ylabel('Option Price')
        plt.title(f'Option Price\nS0={self.S0}, K={self.K}, t={self.t}, sigma={self.sigma}, r={self.r}')
        plt.xticks()
        plt.tight_layout()  # Adjust layout to prevent labels from being cut off
        plt.show()


def main():
    op = OptionPricing(S0=100, K=100, r=0.05, t=1, dt=0.01, sigma=0.2, number_of_mc_paths=50000)
    op.plot_stock_simulations(plot_avg_min_max=0, NN=1000)
    op.plot_option_prices()


if __name__ == "__main__":
    main()
