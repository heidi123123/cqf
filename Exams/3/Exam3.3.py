import warnings

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC

warnings.filterwarnings('ignore')  # ignore warnings


def prepare_dataframe():
    df = yf.download("^GSPC", start="2023-01-01")
    df['Return'] = df['Close'].pct_change()
    df.head()
    df = df.dropna()  # drop NaN values s.t. df.isnull().sum() only has 0's
    return df


def plot_returns(df):
    # Visualize the return data
    plt.figure(figsize=(11, 4))
    plt.plot(df['Return'], label="Return")
    plt.title("Relative Return of Close Prices of S&P 500")
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.legend()
    plt.show()


def get_features(df):
    df['O-C'] = df['Open'] - df['Close']  # difference between open and close prices, measuring intraday movement range
    df['H-L'] = df['High'] - df['Low']  # difference between high and low prices, measuring intraday range
    df['Sign'] = np.sign(np.log(df['Close'] / df['Close'].shift(1)))  # sign of return

    for lag in [1, 2, 3, 5, 10, 30]:  # lagged returns for different lag periods
        df[f'Past Return_{lag}'] = df['Return'].shift(lag)

    for momentum_period in [1, 2, 3, 5, 10, 30]:  # price change over different momentum periods
        df['Momentum'] = df['Close'] - df['Close'].shift(momentum_period)

    for sma_period in [1, 2, 3, 5, 10, 30]:  # simple moving average over various time periods
        df['SMA'] = df['Close'].rolling(window=sma_period).mean()

    for ema_period in [1, 2, 3, 5, 10, 30]:  # exponential moving average over various time periods
        df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

    df.dropna(inplace=True)
    features = df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
    return features


def get_target(df):
    # for trend prediction, contain days with small decrease (< 0.5%) in asset close price also within positive class
    return np.where(df['Close'].shift(-1) > 0.995 * df['Close'], 1, 0)


if __name__ == "__main__":
    df = prepare_dataframe()
    X = get_features(df).values
    y = get_target(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
