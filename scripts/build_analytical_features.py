#!/usr/bin/env python3
"""Build and persist customer and restaurant analytical features."""

from __future__ import annotations

import argparse
from pathlib import Path

from eats_intel.config import FEATURES_DIR
from eats_intel.data.features import build_analytical_features, save_analytical_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build customer and restaurant features")
    parser.add_argument("--out", type=Path, default=FEATURES_DIR)
    args = parser.parse_args()

    features = build_analytical_features()
    written = save_analytical_features(features, args.out)
    for name, path in written.items():
        frame = features[name]
        print(f"{name:22s} {len(frame):7,d} rows  {len(frame.columns):2d} cols  → {path}")


if __name__ == "__main__":
    main()
