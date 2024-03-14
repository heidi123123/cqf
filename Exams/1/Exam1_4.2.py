import pandas as pd
from scipy.stats import norm

percentiles = [99.95, 99.75, 99.5, 99.25, 99, 98.5, 98, 97.5]
es_normal = []
es_std_normal = []


def get_factor_for_es(alpha):
    return norm.pdf(norm.ppf(1 - alpha)) / (1 - alpha)


def get_es(alpha, mu, sigma):
    # calculates Expected Shortfall for confidence level alpha for a r.v. X~N(mu, sigma^2)
    return mu - sigma * get_factor_for_es(alpha)


for percentile in percentiles:
    es_normal.append(f"mu - sigma * {get_factor_for_es(percentile / 100)}")
    es_std_normal.append(get_es(percentile / 100, 0, 1))

df = pd.DataFrame({
    'Percentile': percentiles,
    'Expected Shortfall for X~N(mu, sigma^2)': es_normal,
    'Expected Shortfall for X~N(0, 1)': es_std_normal
})

print(df)
