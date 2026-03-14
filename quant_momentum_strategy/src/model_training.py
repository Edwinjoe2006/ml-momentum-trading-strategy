from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import pandas as pd


def train_models(data):

    # Train / test split
    train = data.loc["2017":"2022"]
    test = data.loc["2023":"2025"]

    features = [
        "ret5","ret10",
        "momentum_20","momentum_60","momentum_120",
        "volatility","volatility_60",
        "ma_ratio","trend_strength",
        "rsi",
        "macd","macd_signal",
        "volume_ratio",
        "price_position"
    ]

    X_train = train[features]
    y_train = train["target"]

    X_test = test[features]
    y_test = test["target"]

    # Feature scaling
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -------------------
    # Logistic Regression
    # -------------------

    print("Training Logistic Regression")

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    log_pred = log_model.predict(X_test)
    log_acc = accuracy_score(y_test, log_pred)

    # -------------------
    # Random Forest
    # -------------------

    print("Training Random Forest")

    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)

    # -------------------
    # XGBoost
    # -------------------

    print("Training XGBoost")

    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5
    )

    xgb_model.fit(X_train, y_train)

    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)

    # -------------------
    # Model Performance
    # -------------------

    print("\nModel Accuracy")
    print("Logistic Regression:", log_acc)
    print("Random Forest:", rf_acc)
    print("XGBoost:", xgb_acc)

    # Confusion Matrix (Random Forest)
    cm = confusion_matrix(y_test, rf_pred)

    print("\nConfusion Matrix (Random Forest)")
    print(cm)

    # -------------------
    # Feature Importance
    # -------------------

    importance = rf_model.feature_importances_

    plt.figure(figsize=(8,5))
    plt.barh(features, importance)

    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")

    plt.tight_layout()

    plt.savefig("results/feature_importance.png")

    plt.show()

    # -------------------
    # Ensemble Prediction
    # -------------------

    p1 = log_model.predict_proba(X_test)[:,1]
    p2 = rf_model.predict_proba(X_test)[:,1]
    p3 = xgb_model.predict_proba(X_test)[:,1]

    prob = (p1 + p2 + p3) / 3

    test = test.copy()
    test["probability"] = prob

    return prob, test