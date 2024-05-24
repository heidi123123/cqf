import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, RFE, RFECV
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap


# Prepare the data
def prepare_dataframe(ticker="TSLA"):
    df = yf.download(ticker, start="2018-01-01")
    df['Return'] = df['Close'].pct_change()
    df.dropna(inplace=True)
    return df


def get_features(df):
    df['O-C'] = df['Open'] - df['Close']
    df['H-L'] = df['High'] - df['Low']
    df['Sign'] = np.sign(np.log(df['Close'] / df['Close'].shift(1)))

    for lag in [1, 3, 5, 10, 21]:
        df[f'Past Return_{lag}'] = df['Return'].shift(lag)

    for momentum_period in [1, 3, 5, 10, 21]:
        df[f'Momentum_{momentum_period}'] = df['Close'] - df['Close'].shift(momentum_period)

    for sma_period in [1, 3, 5, 10, 21]:
        df[f'SMA_{sma_period}'] = df['Close'].rolling(window=sma_period).mean()

    for ema_period in [1, 3, 5, 10, 21]:
        df[f'EMA_{ema_period}'] = df['Close'].ewm(span=ema_period, adjust=False).mean()

    df.dropna(inplace=True)
    features = df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], axis=1)
    return features


def get_target(df):
    return np.where(df['Close'].shift(-1) > 0.995 * df['Close'], 1, 0)


def variance_inflation_factor_analysis(X, threshold=5):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data[vif_data["VIF"] < threshold]


def correlation_matrix_analysis(X):
    corr_matrix = X.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap="viridis")
    plt.title("Correlation Matrix of all Features")
    plt.show()


def select_k_best(X, y, k=10):
    selector = SelectKBest(k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return selected_features, X_selected


if __name__ == "__main__":
    df = prepare_dataframe()
    features_df = get_features(df)
    target = get_target(df)

    X_train, X_test, y_train, y_test = train_test_split(features_df, target, test_size=0.2)

    # 1. Variance Inflation Factor
    vif_data = variance_inflation_factor_analysis(X_train)
    selected_features_vif = vif_data['feature']
    X_train_vif = X_train[selected_features_vif]
    print("VIF selected features:", vif_data)

    # 2. Correlation Matrix
    correlation_matrix_analysis(X_train)

    # 3. SelectKBest
    selected_features_kbest, X_train_kbest = select_k_best(X_train, y_train, k=10)
    print("SelectKBest selected features:\n", selected_features_kbest)
