"""Load best model and evaluate on test split."""
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_test_data(path="data/processed/features.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    split = int(len(df) * 0.8)
    test = df.iloc[split:]
    X = test.drop(columns=["date", "sales"])
    y = test["sales"]
    return X, y


def report(X, y):
    model = joblib.load("models/best_model.pkl")
    preds = model.predict(X)
    out = {
        "MAE": mean_absolute_error(y, preds),
        "RMSE": np.sqrt(mean_squared_error(y, preds)),
        "MAPE": float(np.mean(np.abs((y - preds) / y)) * 100),
        "R2": r2_score(y, preds),
    }
    print("Evaluation metrics:")
    for k, v in out.items():
        print(f"  {k}: {v:.3f}")
    return out


if __name__ == "__main__":
    X, y = load_test_data()
    report(X, y)