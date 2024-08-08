import matplotlib.pyplot as plt


def plot_assets_and_residuals(data, ticker1, ticker2, index_ticker):
    plt.figure(figsize=(10, 10))

    # normalize historical prices
    normalized_ticker1 = data[ticker1] / data[ticker1].iloc[0]
    normalized_ticker2 = data[ticker2] / data[ticker2].iloc[0]
    normalized_index = data[index_ticker] / data[index_ticker].iloc[0]

    # plot normalized prices
    plt.subplot(2, 1, 1)
    plt.plot(data.index, normalized_ticker1, label=f"{ticker1} (normalized)", color="b")
    plt.plot(data.index, normalized_ticker2, label=f"{ticker2} (normalized)", color="g")
    plt.plot(data.index, normalized_index, label=f"{index_ticker} (normalized)", color="r")
    plt.title(f"Normalized historical prices of {ticker1}, {ticker2}, and {index_ticker} index")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.grid(True)

    # plot residuals
    plt.subplot(2, 1, 2)
    plt.plot(data.index, data['residuals'], label="Residuals")
    mean = data['residuals'].mean()
    stdev = data['residuals'].std()
    plt.axhline(mean, color="r", linestyle="--", label=f"Mean $\mu$")
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


def find_best_pnl(pnl_df):
    # find the best PnL value and its corresponding Z
    best_row = pnl_df.loc[pnl_df['PnL'].idxmax()]
    return best_row['Z'], best_row['PnL']


def plot_pnl_table(pnl_df):
    plt.figure(figsize=(10, 6))
    plt.plot(pnl_df['Z'], pnl_df['PnL'], marker="o", linestyle="-", color="b", label="PnL vs Z")

    # Highlight the best PnL point with a star
    best_z, best_pnl = find_best_pnl(pnl_df)
    plt.scatter(best_z, best_pnl, color='r', marker="*", s=100, zorder=5,
                label=f"Best PnL: {best_pnl:.2f} at Z={best_z:.2f}")

    # Add text annotation to best PnL
    plt.text(best_z, best_pnl, f"  Z={best_z:.2f}\n  PnL={best_pnl:.2f}",
             color="r", verticalalignment="bottom", horizontalalignment="left",
             fontsize=10, bbox=dict(facecolor="white"))

    plt.title("PnL vs Z")
    plt.xlabel("Z")
    plt.ylabel("PnL")
    plt.grid(True)
    plt.show()
