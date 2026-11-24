# Project Proposal: Sales Forecasting Using Machine Learning

## 1. Problem Statement
Traditional sales forecasting methods (moving average, ARIMA) struggle with
non-linear demand patterns, external shocks, and multi-source data. This
project builds an ML-based framework to produce accurate, scalable forecasts.

## 2. Objectives
1. Analyze limitations of traditional forecasting.
2. Integrate internal + external features.
3. Develop regression/tree/ensemble models.
4. Compare models via MAE, RMSE, MAPE, R².
5. Propose a deployable framework.

## 3. Methodology
- **Data:** historical sales, promotions, holidays, economic indicators, sentiment.
- **Preprocessing:** cleaning, outliers, missing values.
- **Features:** calendar, lag, rolling statistics, external factors.
- **Models:** Linear Regression, Decision Tree, Random Forest, XGBoost, LightGBM + ARIMA baseline.
- **Validation:** time-series split + walk-forward CV.
- **Metrics:** MAE, RMSE, MAPE, R².

## 4. Timeline
| Week | Activity |
|------|----------|
| 1 | Setup & proposal |
| 2–3 | Literature review |
| 4–5 | Data & EDA |
| 6 | Feature engineering |
| 7–9 | Modeling & tuning |
| 10–11 | Evaluation |
| 12 | Deployment |
| 13–14 | Documentation & defense |