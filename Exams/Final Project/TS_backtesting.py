import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


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
        self.pnl = []
        self.positions = []
        self.returns = []

    def calculate_optimal_bounds(self):
        """Calculate the upper and lower bounds for trading, CQF FP Workshop 2, sl. 15."""
        sigma_eq = self.sigma_ou / np.sqrt(2 * self.theta)
        upper_bound = self.mu_e + self.z * sigma_eq
        lower_bound = self.mu_e - self.z * sigma_eq
        return upper_bound, lower_bound

    def manage_positions(self):
        """Manage positions for the trading strategy exploiting mean-reversion of 2 cointegrated assets using
        hedge ratio (beta1) previously obtained in Engle-Granger step 1 and track PnL."""
        position_ticker1, position_ticker2, entry_price_ticker1, entry_price_ticker2 = (0, ) * 4
        upper_bound, lower_bound = self.calculate_optimal_bounds()

        for index, row in self.data.iterrows():
            residual = row['residuals']

            # entry conditions
            if position_ticker1 == 0 and position_ticker2 == 0:
                if residual > upper_bound:
                    # if spread between both tickers is very positive:
                    # short ticker1 (over-valued from equilibrium), long ticker2
                    position_ticker1, position_ticker2 = -1, self.hedge_ratio
                    entry_price_ticker1, entry_price_ticker2 = row[self.ticker1], row[self.ticker2]
                elif residual < lower_bound:
                    # if spread between both tickers is very negative:
                    # long ticker1 (under-valued from equilibrium), short ticker2
                    position_ticker1, position_ticker2 = 1, -self.hedge_ratio
                    entry_price_ticker1, entry_price_ticker2 = row[self.ticker1], row[self.ticker2]

            # exit conditions --> close positions
            elif position_ticker1 == 1 and position_ticker2 == -self.hedge_ratio:
                if residual >= self.mu_e:
                    trade_pnl = (row[self.ticker1] - entry_price_ticker1) + (entry_price_ticker2 - row[self.ticker2]) * self.hedge_ratio
                    self.pnl.append(trade_pnl)
                    if entry_price_ticker1 != 0 and entry_price_ticker2 != 0:
                        self.returns.append(trade_pnl / (entry_price_ticker1 + self.hedge_ratio * entry_price_ticker2))
                    position_ticker1, position_ticker2, entry_price_ticker1, entry_price_ticker2 = 0, 0, 0, 0

            elif position_ticker1 == -1 and position_ticker2 == self.hedge_ratio:
                if residual <= self.mu_e:
                    trade_pnl = (entry_price_ticker1 - row[self.ticker1]) + (row[self.ticker2] - entry_price_ticker2) * self.hedge_ratio
                    self.pnl.append(trade_pnl)
                    if entry_price_ticker1 != 0 and entry_price_ticker2 != 0:
                        self.returns.append(trade_pnl / (entry_price_ticker1 + self.hedge_ratio * entry_price_ticker2))
                    position_ticker1, position_ticker2, entry_price_ticker1, entry_price_ticker2 = 0, 0, 0, 0

        return np.sum(self.pnl)


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


def evaluate_pairs_trading_strategy(data, ticker1, ticker2, ou_params, hedge_ratio):
    """Test Z values in range [0.3, ..., 1.4] and select the best one."""
    results = []
    for z in np.arange(0.3, 1.5, 0.1):
        portfolio = Portfolio(data, ticker1, ticker2, ou_params, hedge_ratio, z)
        pnl = portfolio.manage_positions()
        risk_metrics = RiskMetrics(portfolio.returns)
        metrics = risk_metrics.run_full_analysis()
        results.append({'Z': z, 'PnL': pnl, **metrics})
    return pd.DataFrame(results)
