"""Project paths and shared constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_PARQUET = DATA_DIR / "parquet"
RAW_CSV = DATA_DIR / "csv"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

TABLES = (
    "customers",
    "restaurants",
    "drivers",
    "orders",
    "deliveries",
    "payments",
    "reviews",
)

RANDOM_STATE = 42
