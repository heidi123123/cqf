import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.metrics import accuracy_score, classification_report, log_loss, ConfusionMatrixDisplay, \
    RocCurveDisplay
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from statsmodels.stats.outliers_influence import variance_inflation_factor


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
    features = df.drop(["Open", "High", "Low", "Close", "Volume"], axis=1)
    return features


def get_target(df):
    return np.where(df['Close'].shift(-1) > 0.995 * df['Close'], 1, 0)


def select_k_best(X_train, X_test, y_train, k=10):
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    selected_features = X_train.columns[selector.get_support()]
    return X_train_selected, X_test_selected, selected_features


def rfecv_method(X_train, X_test, y_train, step=0.1, n_estimators=50, scoring="accuracy", display_dataframe=0):
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rfecv = RFECV(estimator=model, step=step, cv=5, scoring=scoring)
    rfecv.fit(X_train, y_train)
    X_train_selected = rfecv.transform(X_train)
    X_test_selected = rfecv.transform(X_test)
    selected_features = X_train.columns[rfecv.get_support()]
    if display_dataframe:
        rankings_df = pd.DataFrame({'Feature': X_train.columns,
                                    'Ranking': rfecv.ranking_})
        print(rankings_df.sort_values(by="Ranking"))
    return X_train_selected, X_test_selected, selected_features


def correlation_matrix_analysis(X):
    corr_matrix = X.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Correlation Matrix of all Features")
    plt.show()


def variance_inflation_factor_analysis(X, threshold=5):
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    return vif_data[vif_data['VIF'] < threshold]


def analyze_model(model, X_train, X_test, y_train, y_test, generate_plots=0):
    # predict y on the test data
    y_pred = model.predict(X_test)

    # evaluate model accuracy
    accuracy_train = accuracy_score(y_train, model.predict(X_train))
    accuarcy_test = accuracy_score(y_test, y_pred)
    print(f"Train Accuracy: {accuracy_train:0.4}, Test Accuracy: {accuarcy_test:0.4}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    if generate_plots:
        print("Confusion Matrix:")
        ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.show()

        # Display ROC Curve
        RocCurveDisplay.from_estimator(model, X_test, y_test, name="Baseline Model")
        plt.title("AUC-ROC Curve")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random 50:50")
        plt.legend()
        plt.show()


def build_base_model(X, y):
    # Define a baseline model to benchmark against
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel="rbf", C=1.0, gamma=0.1, random_state=42))
    ])
    model.fit(X, y)
    return model


# Define a mapping of class_weight options
class_weight_options = [
    None,
    "balanced",
    {0: 2, 1: 1},
    {0: 1, 1: 2},
    {0: 3, 1: 1},
    {0: 1, 1: 3}
]
class_weight_mapping = {str(option): option for option in class_weight_options}


def define_hyperparameters(trial):
    # Define hyper-parameter space
    scaler_name = trial.suggest_categorical('scaler', ["standard", "minmax"])
    C = trial.suggest_loguniform("C", 0.0001, 100.0)
    gamma = trial.suggest_loguniform("gamma", 0.0001, 1.0)
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "sigmoid"])
    class_weight_str = trial.suggest_categorical("class_weight",
                                                 [str(option) for option in class_weight_options])
    return {"scaler_name": scaler_name,
            "C": C,
            "gamma": gamma,
            "kernel": kernel,
            "class_weight_str": class_weight_str}


def objective(trial, X, y):
    # Get tunable hyperparameters + pipeline
    hps = define_hyperparameters(trial)

    # Create model as Pipeline
    scaler = StandardScaler() if hps["scaler_name"] == "standard" else MinMaxScaler()
    model = Pipeline([
        ("scaler", scaler),
        ("svm", SVC(C=hps["C"],
                    gamma=hps["gamma"],
                    kernel=hps["kernel"],
                    class_weight=class_weight_mapping[hps["class_weight_str"]],
                    probability=True, random_state=42))
    ])

    # Define TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)

    # Use log-loss function for every TimeSeriesSplit to measure performance of hyper-parameters
    log_losses = []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_test)
        log_losses.append(log_loss(y_test, preds))

    # Return the negative mean log-loss for optimization
    return -1.0 * np.mean(log_losses)


def optimize_hyperparameters(X_train, y_train, n_trials=100):
    # Convert pandas objects to numpy arrays
    X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.to_numpy() if isinstance(y_train, pd.Series) else y_train

    # Study setup
    study = optuna.create_study(direction='maximize')

    # Optimization
    study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=n_trials)
    best_params = study.best_trial.params

    # Convert the best class_weight parameter back to dictionary
    best_params['class_weight'] = class_weight_mapping[best_params['class_weight']]

    # Create best pipeline
    scaler = StandardScaler() if best_params['scaler'] == 'standard' else MinMaxScaler()
    best_model = Pipeline([
        ('scaler', scaler),
        ('svm', SVC(C=best_params['C'],
                    gamma=best_params['gamma'],
                    kernel=best_params['kernel'],
                    class_weight=best_params['class_weight'],
                    probability=True,
                    random_state=42))
    ])
    return best_model, best_params, study


def visualize_optimization(study):
    optuna.visualization.matplotlib.plot_optimization_history(study)\
        .update_layout(title="Optimization History of Hyperparameter Tuning for SVM")
    optuna.visualization.matplotlib.plot_param_importances(study)\
        .update_layout(title="Hyperparameter Importances for SVM")
    optuna.visualization.matplotlib.plot_slice(study)\
        .update_layout(title="Slice Plot of Hyperparameter Optimization for SVM")
    optuna.visualization.matplotlib.plot_contour(study)\
        .update_layout(title="Contour Plot of Hyperparameter Optimization for SVM")
    plt.show()


if __name__ == "__main__":
    ticker = "MSFT"
    df = prepare_dataframe(ticker)
    features = get_features(df)
    target = get_target(df)

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    # Feature selection with SelectKBest
    X_train_kbest, X_test_kbest, selected_features_kbest = select_k_best(X_train, X_test, y_train, k=16)
    print("SelectKBest selected features:", selected_features_kbest)

    # Correlation matrix
    # correlation_matrix_analysis(pd.DataFrame(X_train_kbest, columns=selected_features_kbest))

    # Variance Inflation Factor
    vif_data = variance_inflation_factor_analysis(pd.DataFrame(X_train_kbest, columns=selected_features_kbest))
    selected_features_vif = vif_data['feature']
    X_train_vif = pd.DataFrame(X_train_kbest, columns=selected_features_kbest)[selected_features_vif]
    X_test_vif = pd.DataFrame(X_test_kbest, columns=selected_features_kbest)[selected_features_vif]
    print("VIF selected features:", vif_data)

    # Further feature selection with RFECV
    X_train_rfecv, X_test_rfecv, selected_features_rfecv = rfecv_method(
        pd.DataFrame(X_train_vif, columns=selected_features_vif),
        pd.DataFrame(X_test_vif, columns=selected_features_vif), y_train)
    print("RFECV selected features:", selected_features_rfecv)

    X_train_selected, X_test_selected = X_train_rfecv, X_test_rfecv

    base_model = build_base_model(X_train_selected, y_train)
    analyze_model(base_model, X_train_selected, X_test_selected, y_train, y_test, generate_plots=0)

    # Optimize hyperparameters
    best_model, best_params, study = optimize_hyperparameters(X_train_selected, y_train, n_trials=10)
    print("Best hyperparameters found by Optuna:", best_params)
    analyze_model(best_model, X_train_selected, X_test_selected, y_train, y_test, generate_plots=1)

    # Visualization of Optuna study results
    # visualize_optimization(study)
