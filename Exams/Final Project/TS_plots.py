import matplotlib.pyplot as plt


def plot_assets_and_residuals(data, ticker1, ticker2):
    plt.figure(figsize=(12, 8))

    # price time series plot
    plt.subplot(2, 1, 1)
    plt.plot(data.index, data[ticker1], label=f"{ticker1}", color="blue")
    plt.plot(data.index, data[ticker2], label=f"{ticker2}", color="orange")
    plt.title(f"Historical Prices of {ticker1} and {ticker2}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)

    # residuals plot
    plt.subplot(2, 1, 2)
    plt.plot(data.index, data['residuals'], label="Residuals", color="green")
    mean = data['residuals'].mean()
    stdev = data['residuals'].std()
    plt.axhline(mean, color="red", linestyle='--', label=f"Mean $\mu$")
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
    plt.axhline(0, color='black', linestyle='--')
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
