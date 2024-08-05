import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta

# Prepare the data
ticker = "MSFT"
df = yf.download(ticker, start="2018-01-01")
df['Return'] = df['Close'].pct_change()
df.dropna(inplace=True)

# Basic price features
df['O-C'] = df['Open'] - df['Close']
df['H-L'] = df['High'] - df['Low']
df['Sign'] = np.sign(np.log(df['Close'] / df['Close'].shift(1)))

# Print columns before strategy
print("Columns before applying pandas_ta strategy:", df.columns)

# Adding all technical indicators using pandas_ta strategy
df.ta.study('All')

# Print columns after strategy
print("Columns after applying pandas_ta strategy:", df.columns)
