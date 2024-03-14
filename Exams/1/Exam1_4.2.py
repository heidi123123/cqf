import pandas as pd
from scipy.stats import norm

percentiles = [99.95, 99.75, 99.5, 99.25, 99, 98.5, 98, 97.5]
es = []


def get_es(alpha, mu, sigma):
    # calculates Expected Shortfall for confidence level alpha for a r.v. X~N(mu, sigma^2)
    return mu - sigma * norm.pdf(norm.ppf(1 - alpha)) / (1 - alpha)


for percentile in percentiles:
    es.append(get_es(percentile / 100, 0, 1))

df = pd.DataFrame({
    'Percentile': percentiles,
    'Expected Shortfall for X~N(0, 1)': es
})

print(df)
