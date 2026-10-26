"""Analytical and model feature tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eats_intel.config import FEATURES_DIR
from eats_intel.data.load import load_all

LATE_NIGHT_HOURS = {22, 23, 0, 1, 2, 3, 4}
CANCELLED_PREFIX = "Cancelled"


def _tables(tables: dict[str, pd.DataFrame] | None) -> dict[str, pd.DataFrame]:
    return tables if tables is not None else load_all()


def _is_cancelled(status: pd.Series) -> pd.Series:
    return status.astype(str).str.startswith(CANCELLED_PREFIX)


def build_order_fact(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Customer + restaurant + order + delivery grain for prediction and segments."""
    tables = _tables(tables)
    orders = tables["orders"]
    deliveries = tables["deliveries"]
    customers = tables["customers"]
    restaurants = tables["restaurants"]
    fact = orders.merge(deliveries, on=["order_id", "customer_id", "restaurant_id"], how="left")
    fact = fact.merge(
        customers[
            [
                "customer_id",
                "age",
                "gender",
                "neighborhood",
                "preferred_cuisine",
                "device_type",
                "account_status",
                "lifetime_orders_prior",
            ]
        ],
        on="customer_id",
        how="left",
    )
    fact = fact.merge(
        restaurants[
            [
                "restaurant_id",
                "category",
                "price_tier",
                "restaurant_rating",
                "avg_prep_time_min",
                "partner_status",
                "is_open_late",
            ]
        ],
        on="restaurant_id",
        how="left",
    )
    return fact


def build_customer_features(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """One row per customer with order, spend, frequency, rating, and time-of-week features."""
    tables = _tables(tables)
    customers = tables["customers"].copy()
    orders = tables["orders"].copy()
    reviews = tables["reviews"].copy()

    orders["order_timestamp"] = pd.to_datetime(orders["order_timestamp"])
    cancelled = _is_cancelled(orders["order_status"])
    delivered = orders["order_status"].eq("Delivered")
    paid = orders.loc[delivered, ["customer_id", "order_amount"]]
    weekend = orders["is_weekend"].astype(bool)
    late_night = orders["order_hour"].isin(LATE_NIGHT_HOURS)

    window_days = max(
        (orders["order_timestamp"].max() - orders["order_timestamp"].min()).days,
        1,
    )
    window_weeks = window_days / 7.0

    order_stats = (
        orders.assign(
            _cancelled=cancelled,
            _weekend=weekend,
            _late_night=late_night,
        )
        .groupby("customer_id", as_index=False)
        .agg(
            total_orders=("order_id", "size"),
            weekend_orders=("_weekend", "sum"),
            late_night_orders=("_late_night", "sum"),
            cancelled_orders=("_cancelled", "sum"),
            first_order_at=("order_timestamp", "min"),
            last_order_at=("order_timestamp", "max"),
        )
    )

    spend_stats = (
        paid.groupby("customer_id", as_index=False)
        .agg(
            total_spending=("order_amount", "sum"),
            average_order_value=("order_amount", "mean"),
            delivered_orders=("order_amount", "size"),
        )
    )

    rating_stats = reviews.groupby("customer_id", as_index=False).agg(
        average_rating_given=("rating", "mean"),
        ratings_given=("rating", "size"),
    )

    features = customers.merge(order_stats, on="customer_id", how="left")
    features = features.merge(spend_stats, on="customer_id", how="left")
    features = features.merge(rating_stats, on="customer_id", how="left")

    count_cols = [
        "total_orders",
        "weekend_orders",
        "late_night_orders",
        "cancelled_orders",
        "delivered_orders",
        "ratings_given",
    ]
    features[count_cols] = features[count_cols].fillna(0).astype(int)
    features["total_spending"] = features["total_spending"].fillna(0.0).round(2)
    features["average_order_value"] = features["average_order_value"].round(2)
    features["average_rating_given"] = features["average_rating_given"].round(2)
    features["ordering_frequency"] = (features["total_orders"] / window_weeks).round(3)
    features["weekend_order_share"] = (
        features["weekend_orders"] / features["total_orders"].clip(lower=1)
    ).round(3)
    features["late_night_order_share"] = (
        features["late_night_orders"] / features["total_orders"].clip(lower=1)
    ).round(3)

    return features


def build_restaurant_features(tables: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """One row per restaurant with volume, revenue, rating, prep, cancel, and delivery features."""
    tables = _tables(tables)
    restaurants = tables["restaurants"].copy()
    orders = tables["orders"].copy()
    deliveries = tables["deliveries"].copy()
    reviews = tables["reviews"].copy()

    cancelled = _is_cancelled(orders["order_status"])
    delivered = orders["order_status"].eq("Delivered")

    order_stats = (
        orders.assign(_cancelled=cancelled)
        .groupby("restaurant_id", as_index=False)
        .agg(
            total_orders=("order_id", "size"),
            cancelled_orders=("_cancelled", "sum"),
        )
    )
    revenue_stats = (
        orders.loc[delivered]
        .groupby("restaurant_id", as_index=False)
        .agg(
            revenue=("order_amount", "sum"),
            delivered_orders=("order_id", "size"),
        )
    )
    ops_stats = deliveries.groupby("restaurant_id", as_index=False).agg(
        average_preparation_time=("preparation_time_min", "mean"),
        average_delivery_time=("delivery_time_min", "mean"),
    )
    rating_stats = reviews.groupby("restaurant_id", as_index=False).agg(
        average_rating=("rating", "mean"),
        review_count=("rating", "size"),
    )

    features = restaurants.merge(order_stats, on="restaurant_id", how="left")
    features = features.merge(revenue_stats, on="restaurant_id", how="left")
    features = features.merge(ops_stats, on="restaurant_id", how="left")
    features = features.merge(rating_stats, on="restaurant_id", how="left")

    features[["total_orders", "cancelled_orders", "delivered_orders", "review_count"]] = (
        features[["total_orders", "cancelled_orders", "delivered_orders", "review_count"]]
        .fillna(0)
        .astype(int)
    )
    features["revenue"] = features["revenue"].fillna(0.0).round(2)
    features["cancellation_rate"] = (
        features["cancelled_orders"] / features["total_orders"].clip(lower=1)
    ).round(4)
    features["average_rating"] = features["average_rating"].round(2)
    features["average_preparation_time"] = features["average_preparation_time"].round(2)
    features["average_delivery_time"] = features["average_delivery_time"].round(2)

    return features


def build_analytical_features(
    tables: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    tables = _tables(tables)
    return {
        "customer_features": build_customer_features(tables),
        "restaurant_features": build_restaurant_features(tables),
    }


def save_analytical_features(
    features: dict[str, pd.DataFrame] | None = None,
    out_dir: Path | None = None,
) -> dict[str, Path]:
    out_dir = Path(out_dir) if out_dir is not None else FEATURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    features = features or build_analytical_features()
    written: dict[str, Path] = {}
    for name, frame in features.items():
        path = out_dir / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        frame.to_csv(out_dir / f"{name}.csv", index=False)
        written[name] = path
    return written
