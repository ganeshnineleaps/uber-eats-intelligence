# Uber Eats Operational & Customer Intelligence

Synthetic-data ML system for customer/restaurant segmentation, review sentiment, delivery-time and tip prediction, hourly demand forecasting, and business recommendations.

## Layout

```
uber-ml/
├── data/
│   ├── csv/              # raw tables (generated)
│   ├── parquet/          # raw tables (generated)
│   ├── processed/        # cleaned / joined extracts
│   └── features/         # model-ready matrices
├── notebooks/            # exploration and storyboards
├── scripts/              # one-off jobs (data generation, training CLIs)
├── src/eats_intel/       # importable package
│   ├── data/             # loaders and feature joins
│   ├── segmentation/     # customer + restaurant segments
│   ├── sentiment/        # review scoring
│   ├── models/           # delivery time, tip, demand
│   └── recommend/        # actions from model outputs
├── models/               # serialized artifacts per use case
├── reports/figures/      # charts for recs
└── tests/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Regenerate raw data:

```bash
python scripts/generate_uber_eats_data.py --out data
```
