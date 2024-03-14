import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm


def extend_data_set_with_return_data(data_set, time_period):
    # data_set: pandas dataframe with dates + prices in column in 0
    # time_period used to calculate VaR (required to calculate return information for this time period)

    # Calculate daily returns + log returns
    data_set['1D_Return'] = data_set.iloc[:, 0].pct_change()
    data_set['1D_LogReturn'] = data_set['1D_Return'].apply(lambda x: math.log(x + 1))
    # Log returns are summable, i.e. the N-day-log-return is the sum of N consecutive 1-day-log-returns
    data_set['ND_LogReturn'] = data_set['1D_LogReturn'].rolling(window=time_period).sum()
    return data_set


def calculate_var(extended_data_set, confidence_level):
    # extended_data_set: extended pandas dataframe with return information
    # confidence_level: confidence interval, e.g. 0.99

    mu = extended_data_set['ND_LogReturn'].mean()
    sigma = extended_data_set['ND_LogReturn'].std()
    return norm.ppf(1 - confidence_level, mu, sigma)  # scipy.stats way to do mu - sigma * norm.ppf(alpha)


time_period = 10
confidence_level = 0.99

sp500_df = pd.read_csv("Jan 24 Exam 1 Data.csv", sep=",", parse_dates=['Date'], index_col='Date')
sp500_df = extend_data_set_with_return_data(sp500_df, time_period)

# Calculating 10-D-VaR
var_99_10d = calculate_var(sp500_df, confidence_level)
print(f"The {time_period}D-VaR at confidence level {confidence_level} for the given dataset is {round(var_99_10d, 5)}")

# ******************** question 5a ********************

var_breaches = sp500_df['ND_LogReturn'] < var_99_10d
print(f"Number of VaR breaches: {var_breaches.sum()}")
print(f"Percentage of VaR breaches: {var_breaches.sum() / var_breaches.count() * 100}%")

# ******************** question 5b ********************

# add auxiliary column to dataframe to indexify the var breaches
sp500_df['Counter'] = range(sp500_df['SP500'].count())
# create new dataframe that contains only the var breaches now
sp500_var_breaches_df = sp500_df[var_breaches]


def count_consecutive(llist):
    count = 0
    for i in range(len(llist) - 1):
        if llist[i] + 1 == llist[i + 1]:
            count += 1
    return count


number_of_consecutive_var_breaches = count_consecutive(sp500_var_breaches_df['Counter'].to_list())

print(f"Number of consecutive VaR breaches: {number_of_consecutive_var_breaches}")
