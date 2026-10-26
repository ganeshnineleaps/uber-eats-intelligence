"""Load raw synthetic tables from parquet (fallback to CSV)."""

from __future__ import annotations

import pandas as pd

from eats_intel.config import RAW_CSV, RAW_PARQUET, TABLES


def load_table(name: str) -> pd.DataFrame:
    if name not in TABLES:
        raise ValueError(f"Unknown table '{name}'. Expected one of {TABLES}")
    parquet_path = RAW_PARQUET / f"{name}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    csv_path = RAW_CSV / f"{name}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No parquet or csv found for table '{name}'")


def load_all() -> dict[str, pd.DataFrame]:
    return {name: load_table(name) for name in TABLES}
