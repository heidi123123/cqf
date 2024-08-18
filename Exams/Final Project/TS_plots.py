import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_assets_and_residuals(train_data, test_data, ticker1, ticker2, index_ticker="SPY", split_ratio=0.7):
    plt.figure(figsize=(10, 10))

    # normalize historical prices for train data
    normalized_train_ticker1 = train_data[ticker1] / train_data[ticker1].iloc[0]
    normalized_train_ticker2 = train_data[ticker2] / train_data[ticker2].iloc[0]
    normalized_train_index = train_data[index_ticker] / train_data[index_ticker].iloc[0]

    # normalize historical prices for test data
    normalized_test_ticker1 = test_data[ticker1] / train_data[ticker1].iloc[0]
    normalized_test_ticker2 = test_data[ticker2] / train_data[ticker2].iloc[0]
    normalized_test_index = test_data[index_ticker] / train_data[index_ticker].iloc[0]

    # determine the split date
    split_date = train_data.index[-1]

    # plot normalized prices
    plt.subplot(2, 1, 1)
    plt.plot(train_data.index, normalized_train_ticker1, label=f"{ticker1} (normalized)", color="blue")
    plt.plot(train_data.index, normalized_train_ticker2, label=f"{ticker2} (normalized)", color="green")
    plt.plot(train_data.index, normalized_train_index, label=f"{index_ticker} (normalized)", color="orange")

    plt.plot(test_data.index, normalized_test_ticker1, color="blue", linestyle="--")
    plt.plot(test_data.index, normalized_test_ticker2, color="green", linestyle="--")
    plt.plot(test_data.index, normalized_test_index, color="orange", linestyle="--")

    plt.axvline(split_date, color="black", linestyle="--", label="Train/Test Data Split")
    plt.title(f"Normalized historical prices of {ticker1}, {ticker2}, and {index_ticker} index")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid(True)

    # plot residuals for train and test data
    plt.subplot(2, 1, 2)
    plt.plot(train_data.index, train_data['residuals'], label="Train Residuals", color="blue")
    plt.plot(test_data.index, test_data['residuals'], label="Test Residuals", color="blue", linestyle="--")

    plt.axvline(split_date, color="black", linestyle="--", label="Train/Test Data Split")
    mean = train_data['residuals'].mean()
    stdev = train_data['residuals'].std()
    plt.axhline(mean, color="r", linestyle='--', label=f"Mean $\mu$")
    plt.axhline(mean + 1.1 * stdev, color="purple", linestyle="--", label="$\pm1.1*\sigma$")
    plt.axhline(mean - 1.1 * stdev, color="purple", linestyle="--")

    plt.title(f"Residuals of {ticker1} and {ticker2}")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def compare_ecm_residuals(data, ecm_results):
    plt.figure(figsize=(15, 8))
    # original residuals
    plt.plot(data.index, data['residuals'], label="Equilibrium residuals $u_t$", color="blue")
    # align indices for lagged ECM residuals
    plt.plot(data.index[1:], ecm_results['residuals'], label="ECM residuals $\epsilon_t$", color="orange")
    plt.axhline(0, color="black", linestyle="--")
    plt.title("Comparison of Residuals from Engle-Granger Method")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.legend()
    plt.show()


def simulate_ou_process(theta, mu_e, sigma_ou, initial_value, num_steps, dt=1):
    """Simulate an Ornstein-Uhlenbeck process."""
    ou_process = np.zeros(num_steps)
    ou_process[0] = initial_value
    for t in range(1, num_steps):
        ou_process[t] = ou_process[t-1] + theta * (mu_e - ou_process[t-1]) * dt + sigma_ou * np.sqrt(dt) * np.random.normal()
    return ou_process


def plot_ou_process_and_residuals(data, theta, mu_e, sigma_ou):
    """Plot simulated OU process against actual residuals."""
    # Simulate an OU process with the estimated parameters
    num_steps = len(data['residuals'])
    initial_value = data['residuals'].iloc[0]
    simulated_residuals = simulate_ou_process(theta, mu_e, sigma_ou, initial_value, num_steps)

    # Plot the actual residuals and the simulated OU process
    plt.figure(figsize=(14, 8))
    plt.plot(data.index, data['residuals'], label="Actual Residuals $e_t$")
    plt.plot(data.index, simulated_residuals, label="Simulated OU Process", linestyle="-.")
    plt.xlabel("Date")
    plt.ylabel("Residuals")
    plt.title("Actual Residuals vs. Simulated Ornstein-Uhlenbeck Process")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_pnl_table(pnl_table, best_z, best_pnl, pnl_key):
    """Plot the PnL values achieved for different Z-values. Mark the best Z-value with a star."""
    plt.figure(figsize=(10, 6))
    plt.plot(pnl_table['Z'], pnl_table[pnl_key], marker="o", linestyle="-", color="b", label="PnL vs Z")

    # Highlight the best PnL point with a star
    plt.scatter(best_z, best_pnl, color='r', marker="*", s=100, zorder=5,
                label=f"Best PnL: {best_pnl:.2f} at Z={best_z:.2f}")

    # Add text annotation to best PnL
    plt.text(best_z, best_pnl, f"  Z={best_z:.1f}\n  PnL={best_pnl:.1f}", color="r", bbox=dict(facecolor="white"))

    plt.title(f"{pnl_key} vs Z")
    plt.xlabel("Z")
    plt.ylabel(f"{pnl_key}")
    plt.grid(True)
    plt.show()


def plot_positions(portfolio):
    """Plot the positions of ticker1 and ticker2 over time."""
    dates = portfolio.data.index

    plt.figure(figsize=(14, 6))
    plt.axhline(0, color="r", linestyle="--", label="no position")
    plt.plot(dates, portfolio.positions[portfolio.ticker1], label=f"Position {portfolio.ticker1}", color="blue")
    plt.plot(dates, portfolio.positions[portfolio.ticker2], label=f"Position {portfolio.ticker2}", color="orange")
    plt.xlabel("Date")
    plt.ylabel("Position")
    plt.title(f"Positions for {portfolio.ticker1} and {portfolio.ticker2} Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_asset_prices_and_residuals(portfolio):
    """Plot the asset prices and residuals with sigma_eq-bands using subplots."""
    dates = portfolio.data.index

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # primary y-axis: asset prices
    ax1.plot(dates, portfolio.data[portfolio.ticker1], label=f"{portfolio.ticker1} Price", color="blue")
    ax1.plot(dates, portfolio.data[portfolio.ticker2], label=f"{portfolio.ticker2} Price", color="orange")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Asset Price")
    ax1.legend(loc="upper left")
    ax1.grid(True)

    # secondary y-axis: residuals
    residuals = pd.Series(portfolio.data['residuals'], index=dates)
    ax2 = ax1.twinx()
    ax2.plot(dates, residuals, label="Residuals", color="green")
    ax2.axhline(portfolio.mu_e, color="black", linestyle="--", label=r"$\mu_e$")
    upper_bound, lower_bound = portfolio.calculate_optimal_bounds()
    sigma_band_label = r"$\mu_e \pm " + str(round(portfolio.z, 1)) + r" \times \sigma_{eq}$"
    ax2.axhline(upper_bound, color="grey", linestyle="--", label=sigma_band_label)
    ax2.axhline(lower_bound, color="grey", linestyle="--")
    ax2.set_ylabel("Residuals")

    # mark entry and exit signals based on changes in positions
    positions_ticker1 = portfolio.positions[portfolio.ticker1]
    position_changes = positions_ticker1.diff().fillna(0)
    signals = position_changes != 0  # track where a position is changed
    ax2.plot(dates[signals], residuals[signals], "ro", label="Position Change (Entry/Exit)")
    ax2.legend(loc="upper right")

    plt.title(f"Asset Prices and Residuals with Thresholds for {portfolio.ticker1} and {portfolio.ticker2}")
    plt.show()


def get_portfolio_investment(portfolio):
    """Calculate the initial dollar investment in the portfolio."""
    ticker1 = portfolio.ticker1
    ticker2 = portfolio.ticker2

    # boolean series for non-zero positions in ticker1 and ticker2
    condition_ticker1 = portfolio.positions[ticker1] != 0
    condition_ticker2 = portfolio.positions[ticker2] != 0
    combined_condition = np.logical_or(condition_ticker1, condition_ticker2)

    # find index of the first non-zero position
    first_trade_date_idx = portfolio.positions[combined_condition].index[0]

    # entry prices at first trade date
    entry_price_ticker1 = portfolio.data.loc[first_trade_date_idx, ticker1]
    entry_price_ticker2 = portfolio.data.loc[first_trade_date_idx, ticker2]

    # positions of first trade date
    position_ticker1 = portfolio.positions.loc[first_trade_date_idx, ticker1]
    position_ticker2 = portfolio.positions.loc[first_trade_date_idx, ticker2]
    return entry_price_ticker1 * position_ticker1 + entry_price_ticker2 * position_ticker2


def plot_pnl_against_index(portfolio, index_ticker):
    """Plot the portfolio PnL compared to index performance."""
    realized_pnl = portfolio.realized_daily_pnl
    total_pnl = portfolio.unrealized_daily_pnl + portfolio.realized_daily_pnl
    dates = portfolio.data.index

    # calculate cumulative index return and scale to portfolio investment
    index_prices = portfolio.data[index_ticker]
    initial_index_price = index_prices.iloc[0]
    normalized_index_return = (index_prices - initial_index_price) / initial_index_price
    index_return_scaled = normalized_index_return * get_portfolio_investment(portfolio)

    plt.figure(figsize=(14, 6))
    plt.plot(dates, realized_pnl, label="Realized PnL", color="blue")
    plt.plot(dates, total_pnl, label="Total PnL incl. Unrealized Gains", color="red")
    plt.plot(dates, index_return_scaled, label=f"{index_ticker} Performance", color="green", linestyle="--")
    plt.title(f"PnL Evolution over Test Data Period - compared against {index_ticker} index")
    plt.xlabel("Date")
    plt.ylabel("Performance [$]")
    plt.legend()
    plt.grid(True)
    plt.show()
