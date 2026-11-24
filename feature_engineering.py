"""Feature engineering: lag, rolling, calendar features."""
import pandas as pd


def add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df["day"] = df[date_col].dt.day
    df["month"] = df[date_col].dt.month
    df["dayofweek"] = df[date_col].dt.dayofweek
    df["weekofyear"] = df[date_col].dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target_col: str,
                     lags=(1, 7, 14, 30)) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, target_col: str,
                         windows=(7, 14, 30)) -> pd.DataFrame:
    df = df.copy()
    for w in windows:
        df[f"{target_col}_roll_mean_{w}"] = df[target_col].rolling(w).mean()
        df[f"{target_col}_roll_std_{w}"] = df[target_col].rolling(w).std()
    return df


def build_features(df: pd.DataFrame, date_col: str,
                   target_col: str) -> pd.DataFrame:
    """Full feature pipeline."""
    df = add_calendar_features(df, date_col)
    df = add_lag_features(df, target_col)
    df = add_rolling_features(df, target_col)
    return df.dropna().reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/clean_sales.csv", parse_dates=["date"])
    df = build_features(df, "date", "sales")
    df.to_csv("data/processed/features.csv", index=False)
    print(f"Feature set: {df.shape}")