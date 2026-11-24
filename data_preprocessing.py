"""Data loading and cleaning utilities."""
import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV dataset."""
    df = pd.read_csv(path, parse_dates=True)
    return df


def handle_missing(df: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    """Fill missing values using forward-fill or interpolation."""
    if method == "ffill":
        return df.fillna(method="ffill").fillna(method="bfill")
    elif method == "interpolate":
        return df.interpolate()
    return df.dropna()


def remove_outliers(df: pd.DataFrame, col: str, z_thresh: float = 3.0) -> pd.DataFrame:
    """Remove rows where |z-score| > threshold for a given column."""
    z = np.abs((df[col] - df[col].mean()) / df[col].std())
    return df[z < z_thresh]


def basic_clean(df: pd.DataFrame, date_col: str, target_col: str) -> pd.DataFrame:
    """Standard cleaning pipeline."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).drop_duplicates(subset=[date_col])
    df = handle_missing(df, "ffill")
    df = remove_outliers(df, target_col)
    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_data("data/raw/sales.csv")
    df = basic_clean(df, "date", "sales")
    df.to_csv("data/processed/clean_sales.csv", index=False)
    print(f"Cleaned dataset saved: {df.shape}")