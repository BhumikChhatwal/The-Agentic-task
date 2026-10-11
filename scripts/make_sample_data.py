"""Generate the bundled sample datasets.

Run with:  python scripts/make_sample_data.py

Everything is seeded, so the files are reproducible and the tests can rely on
them. Five deliberately different tables are produced:

* ``ecommerce_orders.csv``  — two years of orders; trend, weekly seasonality,
  a demand shock, data-entry anomalies, missing values. Joins to customers.
* ``customers.csv``         — a dimension table (no measures), for join demos.
* ``employees.csv``         — HR data shaped for t-tests, ANOVA and chi-square.
* ``weather_daily.csv``     — a clean, strongly seasonal multi-city time series
  with a planted heatwave.
* ``support_tickets.csv``   — a messy table: a constant column, a 40%-null column,
  free text, and a heavy-tailed duration, for profiling and anomaly demos.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

CATEGORIES = ["Electronics", "Home", "Apparel", "Grocery", "Sports"]
REGIONS = ["North", "South", "East", "West"]
CHANNELS = ["Web", "Mobile", "Store"]
SEGMENTS = ["Consumer", "SMB", "Enterprise"]
PLANS = ["Basic", "Plus", "Pro"]
COUNTRIES = ["US", "CA", "DE", "IN", "BR", "GB"]
DEPARTMENTS = ["Engineering", "Sales", "Support", "Marketing", "Finance"]
LEVELS = ["L1", "L2", "L3", "L4"]


def _customer_ids(count: int) -> list[str]:
    return [f"C{i:04d}" for i in range(1, count + 1)]


def make_customers(rng: np.random.Generator, n: int = 400) -> pd.DataFrame:
    segment = rng.choice(SEGMENTS, n, p=[0.55, 0.30, 0.15])
    plan = np.where(
        segment == "Enterprise",
        rng.choice(["Plus", "Pro"], n, p=[0.3, 0.7]),
        np.where(segment == "SMB", rng.choice(PLANS, n, p=[0.3, 0.5, 0.2]),
                 rng.choice(PLANS, n, p=[0.7, 0.25, 0.05])),
    )
    signup = pd.to_datetime("2021-01-01") + pd.to_timedelta(
        rng.integers(0, 1400, n), unit="D"
    )
    plan_base = pd.Series(plan).map({"Basic": 120.0, "Plus": 480.0, "Pro": 2400.0}).to_numpy()
    # Churn is more likely on cheap plans and early in the relationship.
    churn_p = np.where(plan == "Basic", 0.34, np.where(plan == "Plus", 0.18, 0.07))
    churned = rng.random(n) < churn_p
    ltv = plan_base * rng.uniform(0.6, 1.9, n) * np.where(churned, rng.uniform(0.15, 0.6, n), 1.0)
    return pd.DataFrame(
        {
            "customer_id": _customer_ids(n),
            "signup_date": signup,
            "segment": segment,
            "plan": plan,
            "country": rng.choice(COUNTRIES, n),
            "is_churned": np.where(churned, "yes", "no"),
            "lifetime_value": ltv.round(2),
        }
    )


def make_orders(rng: np.random.Generator, customers: pd.DataFrame, n: int = 2600) -> pd.DataFrame:
    days = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    # Weekly seasonality plus a slow upward trend and a two-week demand shock.
    weekday_weight = np.array([0.80, 0.90, 0.95, 1.00, 1.15, 1.35, 1.10])  # Mon..Sun
    trend = np.linspace(1.0, 1.28, len(days))
    shock = np.ones(len(days))
    shock[430:444] = 0.45  # a slump worth discovering as a change point
    weights = weekday_weight[days.dayofweek] * trend * shock
    weights = weights / weights.sum()
    order_days = rng.choice(days, n, p=weights)

    customer_ids = customers["customer_id"].to_numpy()
    customers_by_id = customers.set_index("customer_id")

    quantity = rng.integers(1, 7, n)
    category = rng.choice(CATEGORIES, n, p=[0.22, 0.24, 0.20, 0.24, 0.10])
    base_price = pd.Series(category).map(
        {"Electronics": 320.0, "Home": 120.0, "Apparel": 65.0, "Grocery": 18.0, "Sports": 145.0}
    ).to_numpy()
    unit_price = (base_price * rng.uniform(0.75, 1.3, n)).round(2)
    discount = rng.choice([0, 0, 0, 5, 10, 15, 20], n) / 100.0
    revenue = (quantity * unit_price * (1 - discount)).round(2)

    status = rng.choice(["completed", "returned", "cancelled"], n, p=[0.88, 0.07, 0.05])
    delivery_days = np.where(
        status == "cancelled",
        np.nan,
        np.clip(rng.gamma(3.0, 1.4, n), 1, 30).round(0),
    )
    satisfaction = np.where(rng.random(n) < 0.22, np.nan, rng.choice([1, 2, 3, 4, 5], n, p=[0.06, 0.09, 0.18, 0.36, 0.31]))

    frame = pd.DataFrame(
        {
            "order_id": [f"ORD-{i:06d}" for i in range(1, n + 1)],
            "order_date": pd.to_datetime(order_days),
            "customer_id": rng.choice(customer_ids, n),
            "product_category": category,
            "region": rng.choice(REGIONS, n, p=[0.33, 0.27, 0.24, 0.16]),
            "channel": rng.choice(CHANNELS, n, p=[0.45, 0.40, 0.15]),
            "quantity": quantity,
            "unit_price": unit_price,
            "discount_pct": (discount * 100).astype(int),
            "revenue": revenue,
            "delivery_days": delivery_days,
            "satisfaction": satisfaction,
            "status": status,
        }
    )
    # Attach the customer's segment so single-table questions still work,
    # while the join to customers.csv stays meaningful.
    frame["segment"] = customers_by_id.loc[frame["customer_id"], "segment"].to_numpy()

    # Planted data-entry anomalies: quantity typos that inflate revenue ~100x.
    anomaly_index = rng.choice(frame.index, 18, replace=False)
    frame.loc[anomaly_index, "quantity"] = rng.choice([250, 400, 750, 999], len(anomaly_index))
    frame.loc[anomaly_index, "revenue"] = (
        frame.loc[anomaly_index, "quantity"]
        * frame.loc[anomaly_index, "unit_price"]
        * (1 - frame.loc[anomaly_index, "discount_pct"] / 100)
    ).round(2)

    return frame.sort_values("order_date").reset_index(drop=True)


def make_employees(rng: np.random.Generator, n: int = 320) -> pd.DataFrame:
    department = rng.choice(DEPARTMENTS, n, p=[0.30, 0.24, 0.18, 0.16, 0.12])
    level = rng.choice(LEVELS, n, p=[0.38, 0.30, 0.22, 0.10])
    tenure = np.clip(rng.gamma(2.2, 1.9, n), 0.2, 22).round(1)
    level_base = pd.Series(level).map({"L1": 62000, "L2": 88000, "L3": 121000, "L4": 168000}).to_numpy()
    salary = (level_base * (1 + 0.021 * tenure) * rng.normal(1.0, 0.07, n)).round(0)
    performance = np.clip(rng.normal(3.4, 0.65, n), 1.0, 5.0).round(1)
    overtime = np.clip(rng.gamma(2.0, 4.2, n), 0, 60).round(1)
    # Attrition is driven by overtime and by low performance.
    logit = -2.4 + 0.052 * overtime - 0.85 * (performance - 3.4)
    attrition = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame(
        {
            "employee_id": [f"E{i:04d}" for i in range(1, n + 1)],
            "department": department,
            "level": level,
            "hire_date": pd.to_datetime("2010-01-01") + pd.to_timedelta(
                rng.integers(0, 5400, n), unit="D"
            ),
            "tenure_years": tenure,
            "salary": salary,
            "performance_score": performance,
            "overtime_hours": overtime,
            "attrition": np.where(attrition, "Yes", "No"),
        }
    )


def make_weather(rng: np.random.Generator) -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", "2024-12-31", freq="D")
    rows = []
    for city, offset, amplitude in [("Berlin", 10.0, 11.0), ("Lisbon", 17.0, 6.5), ("Oslo", 6.0, 9.5)]:
        day_of_year = dates.dayofyear.to_numpy()
        seasonal = amplitude * np.sin(2 * np.pi * (day_of_year - 105) / 365.25)
        trend = np.linspace(0, 1.1, len(dates))  # gentle warming
        temp = offset + seasonal + trend + rng.normal(0, 2.4, len(dates))
        rain = np.clip(rng.gamma(1.4, 1.8, len(dates)) - 1.4, 0, None)
        humidity = np.clip(72 - 12 * np.sin(2 * np.pi * (day_of_year - 105) / 365.25) + rng.normal(0, 7, len(dates)), 20, 100)
        rows.append(pd.DataFrame({"date": dates, "city": city, "temp_c": temp.round(2),
                                  "precipitation_mm": rain.round(2), "humidity_pct": humidity.round(1)}))
    frame = pd.concat(rows, ignore_index=True)
    # Planted heatwave: an extreme, sustained temperature excursion in Berlin.
    mask = (frame["city"] == "Berlin") & frame["date"].between("2022-07-10", "2022-07-20")
    frame.loc[mask, "temp_c"] = frame.loc[mask, "temp_c"] + 14.0
    return frame.sort_values(["city", "date"]).reset_index(drop=True)


def make_tickets(rng: np.random.Generator, n: int = 1800) -> pd.DataFrame:
    created = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365 * 24 * 60, n), unit="m")
    priority = rng.choice(["low", "medium", "high", "urgent"], n, p=[0.34, 0.40, 0.19, 0.07])
    channel = rng.choice(["email", "chat", "phone", "portal"], n, p=[0.42, 0.28, 0.18, 0.12])
    base_hours = pd.Series(priority).map({"low": 20.0, "medium": 9.0, "high": 4.0, "urgent": 1.5}).to_numpy()
    resolution = (base_hours * rng.lognormal(0.0, 0.75, n)).round(2)
    resolution = np.where(rng.random(n) < 0.02, resolution * rng.uniform(20, 60, n), resolution)
    first_response = np.clip(resolution * 60 * rng.uniform(0.02, 0.5, n), 1, None).round(0)
    csat = np.where(rng.random(n) < 0.40, np.nan, rng.choice([1, 2, 3, 4, 5], n, p=[0.12, 0.13, 0.20, 0.30, 0.25]))
    return pd.DataFrame(
        {
            "ticket_id": [f"T{i:06d}" for i in range(1, n + 1)],
            "created_at": created,
            "channel": channel,
            "priority": priority,
            "region": rng.choice(REGIONS, n),
            "resolution_hours": resolution,
            "first_response_minutes": first_response,
            "csat_score": csat,
            "agent_notes": rng.choice(
                ["customer reported a login error", "billing question about invoice",
                 "requested a refund for a damaged item", "asked how to export data",
                 "password reset could not be completed"],
                n,
            ),
            # A column with one value: profiling should flag it as uninformative.
            "source_system": "ticketing_v2",
        }
    )


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20241011)

    customers = make_customers(rng)
    employees = make_employees(rng)
    orders = make_orders(rng, customers)
    weather = make_weather(rng)
    tickets = make_tickets(rng)

    outputs = {
        "ecommerce_orders.csv": orders,
        "customers.csv": customers,
        "employees.csv": employees,
        "weather_daily.csv": weather,
        "support_tickets.csv": tickets,
    }
    for name, frame in outputs.items():
        path = DATA / name
        frame.to_csv(path, index=False)
        print(f"wrote {path.relative_to(ROOT)}  ({len(frame):,} rows x {len(frame.columns)} cols)")


if __name__ == "__main__":
    main()
