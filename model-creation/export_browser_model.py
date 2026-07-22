#!/usr/bin/env python3
"""Export a slim RandomForest JSON model for browser inference."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np


def export_forest(model, n_trees: int) -> dict:
    trees = []
    for est in model.estimators_[:n_trees]:
        tr = est.tree_
        # Store winning class index per node (used at leaves)
        leaf_class = [int(np.argmax(v[0])) for v in tr.value]
        trees.append({
            "l": tr.children_left.astype(np.int32).tolist(),
            "r": tr.children_right.astype(np.int32).tolist(),
            "f": tr.feature.astype(np.int16).tolist(),
            "t": [round(float(x), 6) for x in tr.threshold],
            "c": leaf_class,
        })

    classes = [int(c) for c in model.classes_]
    return {
        "n_features": int(model.n_features_in_),
        "classes": classes,
        "trees": trees,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.path.join(os.path.dirname(__file__), "..", "model", "model.p"),
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__), "..", "UI", "static", "model", "sign_rf.json"
        ),
    )
    parser.add_argument("--trees", type=int, default=30, help="Number of trees to export")
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)["model"]

    payload = export_forest(model, args.trees)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    size = os.path.getsize(args.out)
    print(f"Wrote {args.out} ({size:,} bytes, {len(payload['trees'])} trees)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
