"""Train baseline + ML models and save the best."""
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


TARGET = "sales"
DROP_COLS = ["date", TARGET]


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    y = df[TARGET]
    return X, y


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def main():
    df = pd.read_csv("data/processed/features.csv", parse_dates=["date"])
    X, y = split_xy(df)

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    models = {
        "linear": LinearRegression(),
        "rf": RandomForestRegressor(n_estimators=200, random_state=42),
        "xgb": XGBRegressor(n_estimators=300, learning_rate=0.05,
                            max_depth=5, random_state=42),
    }

    results = {}
    best_name, best_model, best_rmse = None, None, float("inf")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        m = evaluate(y_test, preds)
        results[name] = m
        print(f"{name:8s} -> {m}")
        if m["RMSE"] < best_rmse:
            best_rmse, best_name, best_model = m["RMSE"], name, model

    joblib.dump(best_model, "models/best_model.pkl")
    pd.DataFrame(results).T.to_csv("reports/metrics/model_comparison.csv")
    print(f"\nBest model: {best_name} (RMSE={best_rmse:.2f})")


if __name__ == "__main__":
    main()