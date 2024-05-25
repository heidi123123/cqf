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
def prepare_dataframe(ticker):
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


def select_k_best(X_train, X_test, y_train, k=10):
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_features = X_train.columns[selector.get_support()]
    return X_train_selected, X_test_selected, selected_features


def rfecv_method(X_train, X_test, y_train, display_dataframe=0):
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    rfecv = RFECV(estimator=model, cv=5, scoring='accuracy')
    rfecv.fit(X_train, y_train)
    X_train_selected = rfecv.transform(X_train)
    X_test_selected = rfecv.transform(X_test)
    selected_features = X_train.columns[rfecv.get_support()]
    if display_dataframe:
        rankings_df = pd.DataFrame({'Feature': X_train.columns,
                                    'Ranking': rfecv.ranking_})
        print(rankings_df.sort_values(by='Ranking'))
    return X_train_selected, X_test_selected, selected_features


def correlation_matrix_analysis(X):
    corr_matrix = X.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap="viridis")
    plt.title("Correlation Matrix of all Features")
    plt.show()


def variance_inflation_factor_analysis(X, threshold=5):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data[vif_data["VIF"] < threshold]


if __name__ == "__main__":
    ticker = "TSLA"
    df = prepare_dataframe(ticker)
    features = get_features(df)
    target = get_target(df)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Feature selection with SelectKBest
    X_train_kbest, X_test_kbest, selected_features_kbest = select_k_best(X_train, X_test, y_train, k=16)
    print("SelectKBest selected features:", selected_features_kbest)

    # Further feature selection with RFECV
    X_train_rfecv, X_test_rfecv, selected_features_rfecv = rfecv_method(
        pd.DataFrame(X_train_kbest, columns=selected_features_kbest),
        pd.DataFrame(X_test_kbest, columns=selected_features_kbest), y_train)
    print("RFECV selected features:", selected_features_rfecv)

    # Correlation matrix
    correlation_matrix_analysis(pd.DataFrame(X_train_rfecv, columns=selected_features_rfecv))

    # Variance Inflation Factor
    vif_data = variance_inflation_factor_analysis(X_train)
    selected_features_vif = vif_data['feature']
    X_train_vif = X_train[selected_features_vif]
    print("VIF selected features:", vif_data)
