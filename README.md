# Uber Eats: From Data to Decisions

An end-to-end machine learning project covering customer segmentation, review sentiment analysis, delivery-time prediction and demand forecasting.

## 1. Business Problem

Uber Eats holds large volumes of order, review and delivery data. This project turns that data into decisions by answering three questions:

- **Who are our customers?** Segment customers by spend, frequency and habits.
- **What are they saying?** Classify review sentiment to find unhappy customers.
- **Can we deliver on time?** Predict delivery times and forecast demand.

## 2. Dataset

| Dataset     | Rows   | Columns | Missing Values |
|-------------|-------:|--------:|---------------:|
| Customers   | 8,000  | 32      | 899            |
| Restaurants | 500    | 7       | 0              |
| Sentiment   | 19,082 | 10      | 0              |
| Delivery    | 9,082  | 4       | 0              |

- **Customers:** demographics, membership tier, spend, order behaviour, segment labels
- **Restaurants:** delivery and preparation time, rating, cancellation rate, revenue, total orders
- **Sentiment:** review text, rating, sentiment label, would-order-again flag
- **Delivery:** actual vs predicted delivery time and prediction error

## 3. Data Preparation

Issues identified:

- 899 missing values in the Customers data
- Duplicate `_x` / `_y` columns after merging (orders, spend, average order value)
- Delivery data has no IDs, so it cannot be joined to orders
- Delivery prediction error is exactly 0.0 (possible leakage, needs validation)

Actions taken:

- Handled missing values column by column
- Consolidated duplicates into `total_spending` and `average_order_value`
- Cleaned review text into `cleaned_review`

## 4. Customer Segmentation

- **K-Means:** k chosen using elbow and silhouette scores
- **DBSCAN:** density-based clustering that flags outliers as noise (-1)
- **PCA:** reduces features to 2 components for visualisation
- **Output columns:** `kmeans_segment`, `dbscan_segment`, `customer_segment`, `business_segment`
- **Interpretation:** clusters are profiled on `total_spending`, `average_order_value`, `ordering_frequency`, `weekend_orders`, `late_night_orders`, `membership_tier` and `preferred_cuisine`

## 5. NLP: Sentiment Analysis

Pipeline: text preprocessing → TF-IDF → classifier → evaluation.

| Sentiment | Reviews | Share  |
|-----------|--------:|-------:|
| Positive  | 14,132  | 74.06% |
| Neutral   | 3,809   | 19.96% |
| Negative  | 1,141   | 5.98%  |

Negative reviews are rare, so the model is judged on per-class recall and macro-F1, not accuracy alone.

**Model results:** _add model name, accuracy, precision, recall, F1 here_

## 6. Regression: Delivery Prediction

- **Features:** restaurant prep time, rating, cancellation rate, order volume, time-of-day signals
- **Models:** Linear Regression (baseline), Random Forest, Gradient Boosting
- **Metrics:** MAE, RMSE, R²
- **Best model and feature importance:** _add results here_

> Note: the delivery export currently reports MAE 0.0 and RMSE 0.0. This is a red flag for leakage and must be verified.

## 7. Forecasting

1. Decomposition (trend, seasonality, residual)
2. Stationarity testing (ADF) and differencing
3. ARIMA
4. Prophet
5. Forecast comparison on the same hold-out window (MAE, RMSE, MAPE)

**Results:** _add forecast comparison here_

## 8. Business Impact

- **Targeted marketing:** the right offer for each segment
- **Service recovery:** flag negative reviews quickly
- **Honest ETAs:** more realistic delivery promises
- **Capacity planning:** schedule riders and restaurants ahead of demand

## 9. Limitations

- Delivery MAE/RMSE of 0.0 suggests leakage or copied columns
- Missing-value handling depends on imputation choices
- Class imbalance (only ~6% negative reviews)
- Delivery data cannot be linked to orders or restaurants
- Cluster names require human judgement

## 10. Future Scope

- Fix and validate: resolve delivery leakage, add IDs to delivery data, finish forecast results
- Improve models: transformer-based sentiment (BERT), hyperparameter tuning, weather and event features
- Operationalise: live dashboard, automated retraining, A/B testing of offers

## Project Structure(brief)

```
├── data/              # raw and processed datasets
├── notebooks/         # segmentation, NLP, regression, forecasting
├── presentation/      # final slide deck
├── requirements.txt
└── README.md
```

## Getting Started

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```