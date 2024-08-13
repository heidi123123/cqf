import numpy as np
import pandas as pd
from TS_plots import plot_pnl_table


class Portfolio:
    def __init__(self, data, ticker1, ticker2, ou_params, hedge_ratio, z):
        self.data = data
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.theta = ou_params['theta']
        self.sigma_ou = ou_params['sigma_ou']
        self.mu_e = ou_params['mu_e']
        self.hedge_ratio = hedge_ratio
        self.z = z
        self.realized_daily_pnl = pd.Series(index=data.index, dtype=float).fillna(0)
        self.unrealized_daily_pnl = pd.Series(index=data.index, dtype=float).fillna(0)
        self.positions = pd.DataFrame(index=data.index, columns=[ticker1, ticker2]).fillna(0)
        self.returns = []
        self.manage_positions()

    def calculate_optimal_bounds(self):
        """Calculate the upper and lower bounds for trading, CQF FP Workshop 2, sl. 15."""
        sigma_eq = self.sigma_ou / np.sqrt(2 * self.theta)
        upper_bound = self.mu_e + self.z * sigma_eq
        lower_bound = self.mu_e - self.z * sigma_eq
        return upper_bound, lower_bound

    def enter_position(self, row, position_ticker1, position_ticker2):
        """Enter a position based on the current market conditions."""
        entry_price_ticker1 = row[self.ticker1]
        entry_price_ticker2 = row[self.ticker2]
        return position_ticker1, position_ticker2, entry_price_ticker1, entry_price_ticker2

    def calculate_trade_pnl(self, row, position_ticker1, position_ticker2, entry_price_ticker1, entry_price_ticker2):
        """Calculate the PnL of the trade."""
        if position_ticker1 == 1 and position_ticker2 == -self.hedge_ratio:
            trade_pnl = (row[self.ticker1] - entry_price_ticker1) + (entry_price_ticker2 - row[self.ticker2]) * self.hedge_ratio
        elif position_ticker1 == -1 and position_ticker2 == self.hedge_ratio:
            trade_pnl = (entry_price_ticker1 - row[self.ticker1]) + (row[self.ticker2] - entry_price_ticker2) * self.hedge_ratio
        else:
            trade_pnl = 0
        return trade_pnl

    def calculate_unrealized_pnl(self, row, position_ticker1, position_ticker2, entry_price_ticker1, entry_price_ticker2):
        """Calculate the unrealized PnL for the day based on the movement of ticker1 and ticker2."""
        if position_ticker1 != 0 and position_ticker2 != 0:  # If a position is open
            unrealized_pnl = self.calculate_trade_pnl(row, position_ticker1, position_ticker2, entry_price_ticker1, entry_price_ticker2)
        else:
            unrealized_pnl = 0  # No unrealized PnL if no position is open
        return unrealized_pnl

    def append_return(self, trade_pnl, entry_price_ticker1, entry_price_ticker2):
        """Append the return (either simple or log) to the self.returns list."""
        simple_return = trade_pnl / (entry_price_ticker1 + self.hedge_ratio * entry_price_ticker2)
        self.returns.append(simple_return)

    def close_position(self, row, position_ticker1, position_ticker2, entry_price_ticker1, entry_price_ticker2):
        """Close the position and calculate realized PnL."""
        trade_pnl = self.calculate_trade_pnl(row, position_ticker1, position_ticker2, entry_price_ticker1, entry_price_ticker2)
        if trade_pnl != 0:
            self.append_return(trade_pnl, entry_price_ticker1, entry_price_ticker2)
            self.realized_daily_pnl.at[row.name] += trade_pnl  # add realized PnL
        return 0, 0, 0, 0  # reset positions and entry prices

    def manage_positions(self):
        """Manage positions for the trading strategy exploiting mean-reversion of 2 cointegrated assets using
        hedge ratio (beta1) previously obtained in Engle-Granger step 1 and track daily PnL."""
        position_ticker1, position_ticker2, entry_price_ticker1, entry_price_ticker2 = 0, 0, 0, 0
        upper_bound, lower_bound = self.calculate_optimal_bounds()

        # initialize the first value of realized PnL to 0 - will be needed later in the loop to be carried forward
        previous_realized_pnl = 0

        for index, row in self.data.iterrows():
            residual = row['residuals']

            # entry conditions
            if position_ticker1 == 0 and position_ticker2 == 0:
                if residual > upper_bound:  # very positive spread
                    # short ticker1 (over-valued from equilibrium), long ticker2
                    position_ticker1, position_ticker2 = -1, self.hedge_ratio
                elif residual < lower_bound:  # very negative spread
                    # long ticker1 (under-valued from equilibrium), short ticker2
                    position_ticker1, position_ticker2 = 1, -self.hedge_ratio
                position_ticker1, position_ticker2, entry_price_ticker1, entry_price_ticker2 = \
                    self.enter_position(row, position_ticker1, position_ticker2)

            # exit conditions -> close positions
            elif (position_ticker1 == 1 and position_ticker2 == -self.hedge_ratio and residual >= self.mu_e) or \
                    (position_ticker1 == -1 and position_ticker2 == self.hedge_ratio and residual <= self.mu_e):
                position_ticker1, position_ticker2, entry_price_ticker1, entry_price_ticker2 = \
                    self.close_position(row, position_ticker1, position_ticker2, entry_price_ticker1, entry_price_ticker2)

            # calculate unrealized PnL for the day
            unrealized_pnl = self.calculate_unrealized_pnl(row, position_ticker1, position_ticker2, entry_price_ticker1, entry_price_ticker2)
            self.unrealized_daily_pnl.at[index] = unrealized_pnl

            # carry forward realized PnL
            self.realized_daily_pnl.at[index] += previous_realized_pnl
            previous_realized_pnl = self.realized_daily_pnl.at[index]

            # store the positions for this date
            self.positions.at[index, self.ticker1] = position_ticker1
            self.positions.at[index, self.ticker2] = position_ticker2

    def get_cumulative_pnl(self):
        """Return cumulative realized PnL."""
        return self.realized_daily_pnl.iloc[-1]

    def get_total_pnl(self):
        """Return total PnL, combining realized and unrealized PnL."""
        return self.realized_daily_pnl.add(self.unrealized_daily_pnl, fill_value=0).iloc[-1]


class RiskMetrics:
    def __init__(self, returns):
        self.returns = returns

    def calculate_var(self, confidence_level=0.95):
        """Calculate Value at Risk (VaR) at the given confidence level."""
        if len(self.returns) > 0:
            var = np.percentile(self.returns, (1 - confidence_level) * 100)
        else:
            var = 0
        return var

    def calculate_expected_shortfall(self, confidence_level=0.95):
        """Calculate Expected Shortfall (ES) at the given confidence level."""
        var = self.calculate_var(confidence_level)
        if len(self.returns) > 0:
            expected_shortfall = np.mean([r for r in self.returns if r < var])
        else:
            expected_shortfall = 0
        return expected_shortfall

    def run_full_analysis(self):
        return {'VaR': self.calculate_var(),
                'ES': self.calculate_expected_shortfall()}


def find_best_pnl(pnl_table):
    """In a pnl_table dataframe, find the highest PnL value and its corresponding Z"""
    best_row = pnl_table.loc[pnl_table['PnL'].idxmax()]
    return best_row['Z'], best_row['PnL']


def backtest_strategy_for_z_values(data, ticker1, ticker2, ou_params, hedge_ratio, z_values, plotting=False):
    """Test Z values in range of z_values (iterable) and calculate Pnl for every Z value."""
    results = []
    for z in z_values:
        portfolio = Portfolio(data, ticker1, ticker2, ou_params, hedge_ratio, z)
        pnl = portfolio.get_cumulative_pnl()
        risk_metrics = RiskMetrics(portfolio.returns)
        metrics = risk_metrics.run_full_analysis()
        results.append({'Z': z, 'PnL': pnl, **metrics})
    results_df = pd.DataFrame(results)
    best_z, best_pnl = find_best_pnl(results_df)
    if plotting:
        plot_pnl_table(results_df, best_z, best_pnl)
    return results_df, best_z
