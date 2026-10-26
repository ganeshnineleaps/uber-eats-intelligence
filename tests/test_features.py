import pandas as pd

from eats_intel.data.features import (
    build_customer_features,
    build_restaurant_features,
)


def _mini_tables() -> dict[str, pd.DataFrame]:
    customers = pd.DataFrame(
        {
            "customer_id": ["C1", "C2"],
            "first_name": ["A", "B"],
            "last_name": ["X", "Y"],
            "email": ["a@x.com", "b@y.com"],
            "age": [30, 40],
            "gender": ["Female", "Male"],
            "city": ["Austin", "Austin"],
            "neighborhood": ["Downtown", "Zilker"],
            "latitude": [30.0, 30.1],
            "longitude": [-97.0, -97.1],
            "signup_date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "membership_tier": ["Uber One", "Standard"],
            "preferred_cuisine": ["Thai", "Pizza"],
            "device_type": ["iOS", "Android"],
            "account_status": ["Active", "Active"],
            "lifetime_orders_prior": [3, 1],
        }
    )
    restaurants = pd.DataFrame(
        {
            "restaurant_id": ["R1"],
            "restaurant_name": ["Test Kitchen"],
            "category": ["Thai"],
            "price_tier": ["$$"],
            "restaurant_rating": [4.2],
            "city": ["Austin"],
            "neighborhood": ["Downtown"],
            "latitude": [30.0],
            "longitude": [-97.0],
            "avg_prep_time_min": [15],
            "delivery_radius_km": [5.0],
            "is_open_late": [True],
            "years_on_platform": [2.0],
            "partner_status": ["Plus"],
        }
    )
    orders = pd.DataFrame(
        {
            "order_id": ["O1", "O2", "O3"],
            "customer_id": ["C1", "C1", "C2"],
            "restaurant_id": ["R1", "R1", "R1"],
            "order_timestamp": pd.to_datetime(
                ["2026-10-02 23:10:00", "2026-10-04 12:00:00", "2026-10-03 19:00:00"]
            ),
            "order_hour": [23, 12, 19],
            "is_weekend": [False, True, True],
            "order_amount": [20.0, 40.0, 30.0],
            "order_status": ["Delivered", "Cancelled by customer", "Delivered"],
        }
    )
    deliveries = pd.DataFrame(
        {
            "delivery_id": ["D1", "D3"],
            "order_id": ["O1", "O3"],
            "restaurant_id": ["R1", "R1"],
            "preparation_time_min": [10, 20],
            "delivery_time_min": [15.0, 25.0],
        }
    )
    reviews = pd.DataFrame(
        {
            "review_id": ["V1"],
            "order_id": ["O1"],
            "customer_id": ["C1"],
            "restaurant_id": ["R1"],
            "rating": [5],
        }
    )
    return {
        "customers": customers,
        "restaurants": restaurants,
        "orders": orders,
        "deliveries": deliveries,
        "reviews": reviews,
    }


def test_customer_features_mini():
    features = build_customer_features(_mini_tables()).set_index("customer_id")
    c1 = features.loc["C1"]
    assert c1["total_orders"] == 2
    assert c1["total_spending"] == 20.0
    assert c1["average_order_value"] == 20.0
    assert c1["weekend_orders"] == 1
    assert c1["late_night_orders"] == 1
    assert c1["average_rating_given"] == 5.0
    assert c1["ordering_frequency"] > 0
    assert features.loc["C2", "total_orders"] == 1
    assert features.loc["C2", "total_spending"] == 30.0


def test_restaurant_features_mini():
    features = build_restaurant_features(_mini_tables()).iloc[0]
    assert features["total_orders"] == 3
    assert features["revenue"] == 50.0
    assert features["average_rating"] == 5.0
    assert features["average_preparation_time"] == 15.0
    assert features["average_delivery_time"] == 20.0
    assert features["cancellation_rate"] == 0.3333
