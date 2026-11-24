"""Simple Streamlit dashboard for the sales forecast demo."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"

st.set_page_config(page_title="Sales Forecast", layout="wide")
st.title("Sales Forecasting Dashboard")

st.caption("Upload a feature CSV containing a date column and model feature columns.")

uploaded = st.file_uploader("Upload feature CSV", type=["csv"])
if uploaded is None:
    st.info("Upload a features CSV to generate forecasts.")
elif not MODEL_PATH.is_file() or MODEL_PATH.stat().st_size == 0:
    st.warning("The trained model is unavailable. Add a valid models/best_model.pkl file first.")
else:
    try:
        df = pd.read_csv(uploaded)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="raise")

        model = joblib.load(MODEL_PATH)
        feature_columns = df.drop(
            columns=[column for column in ("date", "sales") if column in df.columns]
        )
        df["predicted_sales"] = model.predict(feature_columns)

        st.subheader("Forecast Preview")
        preview = df[["date", "predicted_sales"]] if "date" in df.columns else df[["predicted_sales"]]
        st.dataframe(preview.head(20), use_container_width=True)
        if "date" in df.columns:
            st.line_chart(preview.set_index("date"))
        else:
            st.line_chart(preview)
    except Exception as exc:
        st.error(f"Forecast could not be generated: {exc}")