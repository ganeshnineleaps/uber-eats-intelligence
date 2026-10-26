from eats_intel.config import TABLES
from eats_intel.data import load_all, load_table


def test_load_customers():
    customers = load_table("customers")
    assert "customer_id" in customers.columns
    assert len(customers) > 0


def test_load_all_tables():
    tables = load_all()
    assert set(tables) == set(TABLES)
    assert all(len(df) > 0 for df in tables.values())
