#!/usr/bin/env python3
"""Generate synthetic Uber Eats operational datasets for ML and analytics.

Tables: customers, restaurants, drivers, orders, deliveries, payments, reviews.

Designed to support:
  - customer / restaurant segmentation
  - review sentiment analysis
  - delivery-time and tip prediction
  - hourly order-demand forecasting
  - actionable ops and growth recommendations
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
CITY_SHORT = {
    "San Francisco": "SF",
    "New York": "NYC",
    "Chicago": "CHI",
    "Austin": "ATX",
    "Los Angeles": "LA",
}

CITIES = {
    "San Francisco": {
        "lat": 37.7749,
        "lon": -122.4194,
        "tz_offset_hours": -7,
        "neighborhoods": [
            "Mission",
            "SOMA",
            "Hayes Valley",
            "Marina",
            "Sunset",
            "Nob Hill",
        ],
    },
    "New York": {
        "lat": 40.7128,
        "lon": -74.0060,
        "tz_offset_hours": -4,
        "neighborhoods": [
            "East Village",
            "Williamsburg",
            "Harlem",
            "Chelsea",
            "Astoria",
            "Upper West Side",
        ],
    },
    "Chicago": {
        "lat": 41.8781,
        "lon": -87.6298,
        "tz_offset_hours": -5,
        "neighborhoods": [
            "Loop",
            "Lincoln Park",
            "Wicker Park",
            "Hyde Park",
            "Logan Square",
            "River North",
        ],
    },
    "Austin": {
        "lat": 30.2672,
        "lon": -97.7431,
        "tz_offset_hours": -5,
        "neighborhoods": [
            "Downtown",
            "East Austin",
            "South Congress",
            "Domain",
            "Zilker",
            "Mueller",
        ],
    },
    "Los Angeles": {
        "lat": 34.0522,
        "lon": -118.2437,
        "tz_offset_hours": -7,
        "neighborhoods": [
            "Silver Lake",
            "Koreatown",
            "Santa Monica",
            "Downtown",
            "Venice",
            "Hollywood",
        ],
    },
}

CUISINES = [
    "American",
    "Mexican",
    "Italian",
    "Chinese",
    "Japanese",
    "Indian",
    "Thai",
    "Mediterranean",
    "Korean",
    "Pizza",
    "Burgers",
    "Coffee & Bakery",
    "Healthy / Bowls",
    "Dessert",
    "Seafood",
]

RESTAURANT_PREFIXES = [
    "Golden",
    "Urban",
    "Pacific",
    "Sunset",
    "Harbor",
    "Maple",
    "Fire",
    "Blue",
    "Lucky",
    "Royal",
    "Green",
    "Neon",
    "Cedar",
    "Iron",
    "Lotus",
]
RESTAURANT_SUFFIXES = [
    "Kitchen",
    "Grill",
    "House",
    "Bistro",
    "Eatery",
    "Bowl",
    "Table",
    "Express",
    "Cantina",
    "Diner",
    "Garden",
    "Bar",
    "Spot",
    "Works",
    "Room",
]

REVIEW_TEMPLATES = {
    "positive": [
        "Food arrived hot and packing was solid. {driver} was professional.",
        "Really tasty {cuisine}. Will order again from {restaurant}.",
        "Fast delivery and the portion size was generous for the price.",
        "Uber One perks made this a great value. Highly recommend.",
        "Everything was fresh. Prep time felt accurate in the app.",
        "Loved the meal. Tip well earned — smooth handoff at the door.",
    ],
    "neutral": [
        "Food was fine, nothing special. Delivery took about as long as estimated.",
        "Order was complete but a bit lukewarm. {cuisine} is hit or miss here.",
        "Okay experience. App estimated {minutes} minutes and that was roughly right.",
        "Packaging could be better but the meal itself was acceptable.",
        "Average {cuisine}. Would try another restaurant next time.",
    ],
    "negative": [
        "Delivery was late and the food was cold. Frustrating experience.",
        "Missing items in the bag. Customer support had to refund part of the order.",
        "Took far longer than the {minutes} minute estimate. Traffic or kitchen backup?",
        "Poor quality for the price. Would not order from {restaurant} again.",
        "Driver seemed lost and the meal arrived soggy. Not coming back.",
        "Wrong order showed up. Had to reorder dinner after waiting.",
    ],
}


def jitter_coord(base: float, scale: float, rng: np.random.Generator) -> float:
    return float(base + rng.normal(0, scale))


def sample_hour(rng: np.random.Generator) -> int:
    """Lunch and dinner peaks, lighter overnight demand."""
    hours = np.arange(24)
    weights = np.array(
        [
            0.4, 0.2, 0.1, 0.1, 0.2, 0.5, 0.8, 1.2,
            1.8, 2.2, 2.8, 4.5, 5.5, 4.2, 2.4, 2.0,
            2.6, 3.8, 5.8, 6.2, 4.4, 2.8, 1.6, 0.8,
        ],
        dtype=float,
    )
    weights /= weights.sum()
    return int(rng.choice(hours, p=weights))


def weather_for_hour(hour: int, month: int, rng: np.random.Generator) -> str:
    rain_p = 0.18 if month in (11, 12, 1, 2, 3) else 0.10
    if rng.random() < rain_p:
        return rng.choice(["Rain", "Heavy Rain"], p=[0.75, 0.25])
    if hour >= 22 or hour <= 5:
        return rng.choice(["Clear", "Cloudy"], p=[0.7, 0.3])
    return rng.choice(["Clear", "Cloudy", "Windy"], p=[0.62, 0.28, 0.10])


def traffic_for_hour(hour: int, weather: str, rng: np.random.Generator) -> str:
    rush = hour in {7, 8, 9, 16, 17, 18, 19}
    lunch = hour in {11, 12, 13}
    if weather in {"Rain", "Heavy Rain"}:
        base = np.array([0.08, 0.32, 0.40, 0.20])
    elif rush:
        base = np.array([0.10, 0.28, 0.42, 0.20])
    elif lunch:
        base = np.array([0.18, 0.42, 0.30, 0.10])
    elif 0 <= hour <= 5:
        base = np.array([0.70, 0.22, 0.07, 0.01])
    else:
        base = np.array([0.35, 0.40, 0.20, 0.05])
    return str(rng.choice(["Light", "Moderate", "Heavy", "Gridlock"], p=base))


def generate_customers(n: int, rng: np.random.Generator, fake: Faker) -> pd.DataFrame:
    cities = list(CITIES)
    city_p = np.array([0.22, 0.28, 0.18, 0.12, 0.20])
    rows = []
    for i in range(n):
        city = str(rng.choice(cities, p=city_p))
        meta = CITIES[city]
        neighborhood = str(rng.choice(meta["neighborhoods"]))
        gender = str(rng.choice(["Female", "Male", "Non-binary"], p=[0.49, 0.48, 0.03]))
        first = fake.first_name_female() if gender == "Female" else fake.first_name()
        last = fake.last_name()
        age = int(np.clip(rng.normal(34, 11), 18, 75))
        signup = fake.date_between(start_date="-4y", end_date="-14d")
        uber_one = bool(rng.random() < 0.32)
        rows.append(
            {
                "customer_id": f"C{i+1:06d}",
                "first_name": first,
                "last_name": last,
                "email": f"{first.lower()}.{last.lower()}{i}@example.com",
                "age": age,
                "gender": gender,
                "city": city,
                "neighborhood": neighborhood,
                "latitude": jitter_coord(meta["lat"], 0.045, rng),
                "longitude": jitter_coord(meta["lon"], 0.055, rng),
                "signup_date": pd.Timestamp(signup),
                "membership_tier": "Uber One" if uber_one else "Standard",
                "preferred_cuisine": str(rng.choice(CUISINES)),
                "device_type": str(rng.choice(["iOS", "Android", "Web"], p=[0.48, 0.46, 0.06])),
                "account_status": str(rng.choice(["Active", "Churned"], p=[0.93, 0.07])),
                "lifetime_orders_prior": int(np.clip(rng.lognormal(2.4, 0.9), 0, 400)),
            }
        )
    return pd.DataFrame(rows)


def generate_restaurants(n: int, rng: np.random.Generator) -> pd.DataFrame:
    cities = list(CITIES)
    rows = []
    used_names: set[str] = set()
    for i in range(n):
        city = str(rng.choice(cities))
        meta = CITIES[city]
        cuisine = str(rng.choice(CUISINES))
        name = f"{rng.choice(RESTAURANT_PREFIXES)} {rng.choice(RESTAURANT_SUFFIXES)}"
        if cuisine in {"Pizza", "Burgers", "Coffee & Bakery"}:
            name = f"{name} {cuisine.split()[0]}"
        name = f"{name} ({CITY_SHORT[city]})"
        if name in used_names:
            name = f"{name} #{i+1}"
        used_names.add(name)
        price_tier = str(rng.choice(["$", "$$", "$$$"], p=[0.38, 0.47, 0.15]))
        rating = float(np.clip(rng.normal(4.15, 0.45), 2.6, 4.95))
        avg_prep = {
            "Coffee & Bakery": 8,
            "Dessert": 10,
            "Pizza": 18,
            "Burgers": 14,
            "Japanese": 22,
            "Seafood": 24,
            "Indian": 22,
            "Chinese": 16,
        }.get(cuisine, 16)
        avg_prep = int(np.clip(rng.normal(avg_prep, 3), 6, 40))
        rows.append(
            {
                "restaurant_id": f"R{i+1:05d}",
                "restaurant_name": name,
                "category": cuisine,
                "price_tier": price_tier,
                "restaurant_rating": round(rating, 2),
                "city": city,
                "neighborhood": str(rng.choice(meta["neighborhoods"])),
                "latitude": jitter_coord(meta["lat"], 0.04, rng),
                "longitude": jitter_coord(meta["lon"], 0.05, rng),
                "avg_prep_time_min": avg_prep,
                "delivery_radius_km": float(np.clip(rng.normal(6.5, 1.4), 3.0, 12.0)),
                "is_open_late": bool(rng.random() < 0.28),
                "years_on_platform": float(np.clip(rng.uniform(0.3, 8.0), 0.3, 8.0)),
                "partner_status": str(rng.choice(["Standard", "Plus", "Premier"], p=[0.55, 0.32, 0.13])),
            }
        )
    return pd.DataFrame(rows)


def generate_drivers(n: int, rng: np.random.Generator, fake: Faker) -> pd.DataFrame:
    cities = list(CITIES)
    rows = []
    for i in range(n):
        city = str(rng.choice(cities))
        vehicle = str(rng.choice(["Bicycle", "Scooter", "Car"], p=[0.18, 0.27, 0.55]))
        rows.append(
            {
                "driver_id": f"D{i+1:05d}",
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "age": int(np.clip(rng.normal(31, 8), 19, 64)),
                "city": city,
                "vehicle_type": vehicle,
                "driver_rating": round(float(np.clip(rng.normal(4.72, 0.22), 3.6, 5.0)), 2),
                "years_experience": round(float(np.clip(rng.exponential(1.6), 0.1, 10.0)), 1),
                "is_active": bool(rng.random() < 0.91),
            }
        )
    return pd.DataFrame(rows)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1 = np.radians(lat1)
    p2 = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def sample_hours(n: int, rng: np.random.Generator) -> np.ndarray:
    hours = np.arange(24)
    weights = np.array(
        [
            0.4, 0.2, 0.1, 0.1, 0.2, 0.5, 0.8, 1.2,
            1.8, 2.2, 2.8, 4.5, 5.5, 4.2, 2.4, 2.0,
            2.6, 3.8, 5.8, 6.2, 4.4, 2.8, 1.6, 0.8,
        ],
        dtype=float,
    )
    weights /= weights.sum()
    return rng.choice(hours, size=n, p=weights)


def sample_weather(hour: np.ndarray, month: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(hour)
    rain_p = np.where(np.isin(month, [11, 12, 1, 2, 3]), 0.18, 0.10)
    is_rain = rng.random(n) < rain_p
    rain = rng.choice(["Rain", "Heavy Rain"], size=n, p=[0.75, 0.25])
    night = (hour >= 22) | (hour <= 5)
    night_w = rng.choice(["Clear", "Cloudy"], size=n, p=[0.7, 0.3])
    day_w = rng.choice(["Clear", "Cloudy", "Windy"], size=n, p=[0.62, 0.28, 0.10])
    return np.where(is_rain, rain, np.where(night, night_w, day_w))


def sample_traffic(hour: np.ndarray, weather: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(hour)
    labels = np.array(["Light", "Moderate", "Heavy", "Gridlock"])
    p = np.tile(np.array([0.35, 0.40, 0.20, 0.05]), (n, 1))
    rain = np.isin(weather, ["Rain", "Heavy Rain"])
    rush = np.isin(hour, [7, 8, 9, 16, 17, 18, 19])
    lunch = np.isin(hour, [11, 12, 13])
    overnight = (hour >= 0) & (hour <= 5)
    p[overnight] = [0.70, 0.22, 0.07, 0.01]
    p[lunch] = [0.18, 0.42, 0.30, 0.10]
    p[rush] = [0.10, 0.28, 0.42, 0.20]
    p[rain] = [0.08, 0.32, 0.40, 0.20]
    cdf = np.cumsum(p, axis=1)
    idx = (rng.random(n)[:, None] > cdf).sum(axis=1)
    return labels[np.clip(idx, 0, 3)]


def generate_orders_and_related(
    n_orders: int,
    customers: pd.DataFrame,
    restaurants: pd.DataFrame,
    drivers: pd.DataFrame,
    start: datetime,
    end: datetime,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    city_names = np.array(list(CITIES))
    city_p = np.array([0.22, 0.28, 0.18, 0.12, 0.20])
    cities = rng.choice(city_names, size=n_orders, p=city_p)

    cust_pos = np.empty(n_orders, dtype=np.int64)
    rest_pos = np.empty(n_orders, dtype=np.int64)
    drv_pos = np.empty(n_orders, dtype=np.int64)

    customers = customers.reset_index(drop=True)
    restaurants = restaurants.reset_index(drop=True)
    drivers = drivers.reset_index(drop=True)

    for city in city_names:
        mask = cities == city
        n_city = int(mask.sum())
        if n_city == 0:
            continue
        c_idx = customers.index[customers["city"] == city].to_numpy()
        r_idx = restaurants.index[restaurants["city"] == city].to_numpy()
        d_idx = drivers.index[drivers["city"] == city].to_numpy()
        if len(c_idx) == 0 or len(r_idx) == 0 or len(d_idx) == 0:
            raise ValueError(f"Missing dimension rows for city {city}")
        cust_pos[mask] = rng.choice(c_idx, size=n_city)
        drv_pos[mask] = rng.choice(d_idx, size=n_city)

        rest_draw = rng.choice(r_idx, size=n_city)
        prefer = rng.random(n_city) < 0.4
        pref_cuisines = customers.loc[cust_pos[mask], "preferred_cuisine"].to_numpy()
        rest_city = restaurants.loc[r_idx]
        preferred = pref_cuisines[prefer]
        if preferred.size:
            for cuisine in np.unique(preferred):
                cuisine_mask = prefer & (pref_cuisines == cuisine)
                cuisine_rest = rest_city.index[rest_city["category"] == cuisine].to_numpy()
                if len(cuisine_rest) == 0 or cuisine_mask.sum() == 0:
                    continue
                rest_draw[cuisine_mask] = rng.choice(cuisine_rest, size=int(cuisine_mask.sum()))
        rest_pos[mask] = rest_draw

    cust = customers.iloc[cust_pos]
    rest = restaurants.iloc[rest_pos]
    drv = drivers.iloc[drv_pos]

    window_days = (end - start).days
    hour = sample_hours(n_orders, rng)
    day_offset = rng.integers(0, window_days, size=n_orders)
    minute = rng.integers(0, 60, size=n_orders)
    second = rng.integers(0, 60, size=n_orders)
    ts = (
        pd.Timestamp(start)
        + pd.to_timedelta(day_offset, unit="D")
        + pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="m")
        + pd.to_timedelta(second, unit="s")
    )
    month = ts.month.to_numpy()
    weather = sample_weather(hour, month, rng)
    traffic = sample_traffic(hour, weather, rng)

    price_map = {"$": 0.72, "$$": 1.0, "$$$": 1.55}
    price_mult = rest["price_tier"].map(price_map).to_numpy()
    uber_one = cust["membership_tier"].to_numpy() == "Uber One"
    item_count = np.clip(rng.poisson(2.4, n_orders) + 1 + ((rng.random(n_orders) < 0.25) & uber_one), 1, 8)
    subtotal = np.round(rng.lognormal(2.55, 0.35, n_orders) * price_mult * item_count, 2)
    delivery_fee = np.where(uber_one, 0.0, rng.choice([2.99, 3.99, 4.99, 5.99], size=n_orders))
    service_fee = np.round(subtotal * 0.12, 2)
    tax = np.round(subtotal * 0.0875, 2)
    promo = np.where(
        rng.random(n_orders) < 0.16,
        np.round(np.minimum(subtotal * rng.uniform(0.1, 0.35, n_orders), 15.0), 2),
        0.0,
    )
    order_amount = np.round(np.maximum(subtotal + delivery_fee + service_fee + tax - promo, 4.99), 2)

    distance = haversine_km(
        cust["latitude"].to_numpy(),
        cust["longitude"].to_numpy(),
        rest["latitude"].to_numpy(),
        rest["longitude"].to_numpy(),
    )
    distance = np.clip(distance, 0.4, rest["delivery_radius_km"].to_numpy() + 1.5)

    prep = np.clip(
        rng.normal(rest["avg_prep_time_min"].to_numpy() + item_count * 1.4, 4.0),
        5,
        55,
    ).astype(int)
    traffic_mult = pd.Series(traffic).map(
        {"Light": 1.0, "Moderate": 1.18, "Heavy": 1.42, "Gridlock": 1.75}
    ).to_numpy()
    weather_mult = pd.Series(weather).map(
        {"Clear": 1.0, "Cloudy": 1.04, "Windy": 1.08, "Rain": 1.22, "Heavy Rain": 1.38}
    ).to_numpy()
    vehicle_speed = drv["vehicle_type"].map({"Bicycle": 14.0, "Scooter": 22.0, "Car": 28.0}).to_numpy()
    speed = np.maximum(vehicle_speed / traffic_mult / weather_mult, 6.0)
    wait_at_store = np.clip(rng.normal(4.0, 2.5, n_orders), 0, 18)
    delivery_time = np.clip((distance / speed) * 60.0 + wait_at_store + rng.normal(2, 1.5, n_orders), 8, 95)

    rain = np.isin(weather, ["Rain", "Heavy Rain"])
    churned = cust["account_status"].to_numpy() == "Churned"
    cancel_p = 0.035 + 0.04 * rain + 0.03 * (delivery_time > 50) + 0.04 * churned
    u = rng.random(n_orders)
    cancel_kind = rng.choice(
        ["Cancelled by customer", "Cancelled by restaurant"], size=n_orders, p=[0.7, 0.3]
    )
    status = np.full(n_orders, "Delivered", dtype=object)
    cancelled = u < cancel_p
    status[cancelled] = cancel_kind[cancelled]
    status[(u >= cancel_p) & (u < cancel_p + 0.03)] = "Out for delivery"
    status[(u >= cancel_p + 0.03) & (u < cancel_p + 0.05)] = "Preparing"

    order_ids = np.array([f"O{i+1:07d}" for i in range(n_orders)])
    payment_ids = np.array([f"P{i+1:07d}" for i in range(n_orders)])
    weekday = ts.day_name()
    pay_status = np.where(
        pd.Series(status).str.startswith("Cancelled"),
        "Refunded",
        np.where(status == "Delivered", "Captured", "Pending"),
    )
    pay_method = rng.choice(
        ["Visa", "Mastercard", "Amex", "Apple Pay", "Google Pay", "Uber Cash"],
        size=n_orders,
        p=[0.34, 0.26, 0.08, 0.14, 0.10, 0.08],
    )

    eta_error = delivery_time - (12 + distance * 2.2)
    tip_base = 1.25 + 0.08 * order_amount
    tip_base = np.where(uber_one, tip_base * 1.12, tip_base)
    tip_base = np.where(eta_error > 15, tip_base * 0.45, np.where(eta_error < 0, tip_base * 1.15, tip_base))
    tip_base *= 0.85 + 0.06 * (rest["restaurant_rating"].to_numpy() - 3.5)
    tip_amount = np.round(np.clip(rng.normal(tip_base, 1.1), 0, 25), 2)
    tip_amount = np.where(rng.random(n_orders) < 0.11, 0.0, tip_amount)
    delivered = status == "Delivered"
    in_progress = ~pd.Series(status).str.startswith("Cancelled").to_numpy()
    tip_for_pay = np.where(delivered, tip_amount, 0.0)

    pickup = ts + pd.to_timedelta(prep + wait_at_store * 0.4, unit="m")
    dropoff = pickup + pd.to_timedelta(np.round(delivery_time), unit="m")

    orders_df = pd.DataFrame(
        {
            "order_id": order_ids,
            "customer_id": cust["customer_id"].to_numpy(),
            "restaurant_id": rest["restaurant_id"].to_numpy(),
            "order_timestamp": ts,
            "order_date": ts.date,
            "order_hour": hour,
            "day_of_week": weekday,
            "is_weekend": ts.day_of_week >= 5,
            "city": cities,
            "weather": weather,
            "traffic_condition": traffic,
            "item_count": item_count,
            "subtotal": subtotal,
            "delivery_fee": np.round(delivery_fee, 2),
            "service_fee": service_fee,
            "tax": tax,
            "promo_discount": promo,
            "order_amount": order_amount,
            "order_status": status,
            "membership_tier": cust["membership_tier"].to_numpy(),
        }
    )
    payments_df = pd.DataFrame(
        {
            "payment_id": payment_ids,
            "order_id": order_ids,
            "customer_id": cust["customer_id"].to_numpy(),
            "payment_method": pay_method,
            "amount": order_amount,
            "currency": "USD",
            "payment_status": pay_status,
            "paid_at": ts + pd.to_timedelta(rng.integers(2, 40, size=n_orders), unit="s"),
            "tip_amount": tip_for_pay,
            "total_charged": np.round(order_amount + tip_for_pay, 2),
        }
    )

    deliv_idx = np.where(in_progress)[0]
    deliveries_df = pd.DataFrame(
        {
            "delivery_id": [f"DL{i+1:07d}" for i in deliv_idx],
            "order_id": order_ids[deliv_idx],
            "driver_id": drv["driver_id"].to_numpy()[deliv_idx],
            "restaurant_id": rest["restaurant_id"].to_numpy()[deliv_idx],
            "customer_id": cust["customer_id"].to_numpy()[deliv_idx],
            "pickup_timestamp": pickup.to_numpy()[deliv_idx],
            "dropoff_timestamp": np.where(
                delivered[deliv_idx],
                dropoff.to_numpy()[deliv_idx],
                np.datetime64("NaT", "ns"),
            ),
            "delivery_distance_km": np.round(distance[deliv_idx], 2),
            "preparation_time_min": prep[deliv_idx],
            "delivery_time_min": np.where(delivered[deliv_idx], np.round(delivery_time[deliv_idx], 1), np.nan),
            "total_time_min": np.where(delivered[deliv_idx], np.round(prep[deliv_idx] + delivery_time[deliv_idx], 1), np.nan),
            "tip_amount": tip_for_pay[deliv_idx],
            "vehicle_type": drv["vehicle_type"].to_numpy()[deliv_idx],
            "on_time_flag": np.where(delivered[deliv_idx], eta_error[deliv_idx] < 8, False),
        }
    )

    review_flag = delivered & (rng.random(n_orders) < 0.42)
    review_idx = np.where(review_flag)[0]
    late = eta_error[review_idx] > 12
    rating_mu = (
        4.3
        - 1.4 * late
        - 0.8 * (tip_amount[review_idx] == 0)
        + 0.25 * (rest["restaurant_rating"].to_numpy()[review_idx] - 4.0)
    )
    stars = np.clip(np.rint(rng.normal(rating_mu, 0.7)), 1, 5).astype(int)
    sentiment = np.where(stars >= 4, "positive", np.where(stars == 3, "neutral", "negative"))
    driver_first = drv["first_name"].to_numpy()[review_idx]
    cuisine = rest["category"].to_numpy()[review_idx]
    rest_name = rest["restaurant_name"].to_numpy()[review_idx]
    minutes = (prep[review_idx] + delivery_time[review_idx]).astype(int)
    texts = []
    for i, label in enumerate(sentiment):
        template = str(rng.choice(REVIEW_TEMPLATES[str(label)]))
        texts.append(
            template.format(
                driver=driver_first[i],
                cuisine=cuisine[i],
                restaurant=rest_name[i],
                minutes=int(minutes[i]),
            )
        )
    reviews_df = pd.DataFrame(
        {
            "review_id": [f"RV{i+1:07d}" for i in range(len(review_idx))],
            "order_id": order_ids[review_idx],
            "customer_id": cust["customer_id"].to_numpy()[review_idx],
            "restaurant_id": rest["restaurant_id"].to_numpy()[review_idx],
            "rating": stars,
            "review_text": texts,
            "sentiment_label": sentiment,
            "review_timestamp": dropoff.to_numpy()[review_idx]
            + pd.to_timedelta(rng.integers(8, 180, size=len(review_idx)), unit="m").to_numpy(),
            "would_order_again": stars >= 4,
        }
    )
    return orders_df, deliveries_df, payments_df, reviews_df


def write_outputs(tables: dict[str, pd.DataFrame], out_dir: Path) -> None:
    csv_dir = out_dir / "csv"
    parquet_dir = out_dir / "parquet"
    csv_dir.mkdir(parents=True, exist_ok=True)
    parquet_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for name, df in tables.items():
        df.to_csv(csv_dir / f"{name}.csv", index=False)
        df.to_parquet(parquet_dir / f"{name}.parquet", index=False)
        summary_rows.append({"table": name, "rows": len(df), "columns": len(df.columns)})
        print(f"  {name:12s}  {len(df):7,d} rows  {len(df.columns):2d} cols")
    pd.DataFrame(summary_rows).to_csv(out_dir / "manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Uber Eats synthetic datasets")
    parser.add_argument("--customers", type=int, default=8000)
    parser.add_argument("--restaurants", type=int, default=500)
    parser.add_argument("--drivers", type=int, default=400)
    parser.add_argument("--orders", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    fake = Faker()
    Faker.seed(args.seed)

    end = datetime(2026, 10, 26, 23, 59, 59)
    start = end - timedelta(days=90)

    print("Generating dimension tables...")
    customers = generate_customers(args.customers, rng, fake)
    restaurants = generate_restaurants(args.restaurants, rng)
    drivers = generate_drivers(args.drivers, rng, fake)

    print("Generating orders, deliveries, payments, reviews...")
    orders, deliveries, payments, reviews = generate_orders_and_related(
        args.orders, customers, restaurants, drivers, start, end, rng
    )

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing files under {out_dir.resolve()}")
    write_outputs(
        {
            "customers": customers,
            "restaurants": restaurants,
            "drivers": drivers,
            "orders": orders,
            "deliveries": deliveries,
            "payments": payments,
            "reviews": reviews,
        },
        out_dir,
    )

    delivered = orders[orders["order_status"] == "Delivered"]
    print("\nQuick sanity checks")
    print(f"  date range:           {orders['order_timestamp'].min()} → {orders['order_timestamp'].max()}")
    print(f"  delivered share:      {len(delivered) / len(orders):.1%}")
    print(f"  mean order amount:    ${orders['order_amount'].mean():.2f}")
    print(f"  mean delivery km:     {deliveries['delivery_distance_km'].mean():.2f}")
    print(f"  mean delivery min:    {deliveries['delivery_time_min'].mean():.1f}")
    print(f"  mean tip (delivered): ${deliveries.loc[deliveries['tip_amount'] > 0, 'tip_amount'].mean():.2f}")
    print(f"  review sentiment:\n{reviews['sentiment_label'].value_counts().to_string()}" if len(reviews) else "  no reviews")
    print("Done.")


if __name__ == "__main__":
    main()
