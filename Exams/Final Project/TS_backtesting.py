import numpy as np
import warnings
warnings.filterwarnings('ignore')


class PairsTradingBacktest:
    def __init__(self, data, ticker1, ticker2, ou_params, hedge_ratio, z):
        self.data = data
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.theta = ou_params['theta']
        self.sigma_ou = ou_params['sigma_ou']
        self.mu_e = ou_params['mu_e']
        self.hedge_ratio = hedge_ratio
        self.z = z
        self.pnl = []
        self.positions = []
        self.returns = []

    def calculate_optimal_bounds(self):
        """Calculate the upper and lower bounds for trading, CQF FP Workshop 2, sl. 15."""
        sigma_eq = self.sigma_ou / np.sqrt(2 * self.theta)
        upper_bound = self.mu_e + self.z * sigma_eq
        lower_bound = self.mu_e - self.z * sigma_eq
        return upper_bound, lower_bound

    def manage_positions(self, row, residual, mean, upper_bound, lower_bound, position_ticker1, position_ticker2,
                         entry_price_ticker1, entry_price_ticker2):
        """Manage positions for the trading strategy exploiting mean-reversion of 2 cointegrated assets using
        hedge ratio (beta1) previously obtained in Engle-Granger step 1"""
        # entry conditions
        if position_ticker1 == 0 and position_ticker2 == 0:
            if residual > upper_bound:
                # if spread between both tickers is very positive:
                # short ticker1 (over-valued from equilibrium), long ticker2
                return -1, self.hedge_ratio, row[self.ticker1], row[self.ticker2], 0
            elif residual < lower_bound:
                # if spread between both tickers is very negative:
                # long ticker1 (under-valued from equilibrium), short ticker2
                return 1, -self.hedge_ratio, row[self.ticker1], row[self.ticker2], 0

        # exit conditions
        elif position_ticker1 == 1 and position_ticker2 == -self.hedge_ratio:
            if residual >= mean:
                pnl = (row[self.ticker1] - entry_price_ticker1) + (
                            entry_price_ticker2 - row[self.ticker2]) * self.hedge_ratio
                return 0, 0, 0, 0, pnl  # positions closed

        elif position_ticker1 == -1 and position_ticker2 == self.hedge_ratio:
            if residual <= mean:
                pnl = (entry_price_ticker1 - row[self.ticker1]) + (
                            row[self.ticker2] - entry_price_ticker2) * self.hedge_ratio
                return 0, 0, 0, 0, pnl  # positions closed

        return position_ticker1, position_ticker2, entry_price_ticker1, entry_price_ticker2, 0

    def get_backtesting_pnl(self):
        position_ticker1 = 0
        position_ticker2 = 0
        entry_price_ticker1 = 0
        entry_price_ticker2 = 0
        upper_bound, lower_bound = self.calculate_optimal_bounds()

        for index, row in self.data.iterrows():
            residual = row['residuals']
            position_ticker1, position_ticker2, entry_price_ticker1, entry_price_ticker2, trade_pnl = \
                self.manage_positions(row, residual, self.mu_e, upper_bound, lower_bound, position_ticker1,
                                      position_ticker2, entry_price_ticker1, entry_price_ticker2)

            if trade_pnl != 0:
                self.pnl.append(trade_pnl)
                self.returns.append(trade_pnl / (entry_price_ticker1 + self.hedge_ratio * entry_price_ticker2))
                self.positions.append((position_ticker1, position_ticker2))

        return np.sum(self.pnl)

    def run_full_analysis(self):
        return {'PnL': self.get_backtesting_pnl()}
