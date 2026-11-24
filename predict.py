"""Generate forecast from the saved best model."""
import joblib
import pandas as pd


def predict(features_csv: str, out_csv: str):
    model = joblib.load("models/best_model.pkl")
    df = pd.read_csv(features_csv, parse_dates=["date"])
    X = df.drop(columns=[c for c in ["date", "sales"] if c in df.columns])
    df["predicted_sales"] = model.predict(X)
    df[["date", "predicted_sales"]].to_csv(out_csv, index=False)
    print(f"Predictions saved to {out_csv}")
    return df[["date", "predicted_sales"]]


if __name__ == "__main__":
    predict("data/processed/features.csv", "reports/metrics/predictions.csv")