#!/usr/bin/env python3
"""Re-evaluate archived Paper 1 embeddings without modifying historical scripts.

This script does NOT regenerate embeddings, GPT-4 summaries, or API-based
similarity scores. It calculates four clustering metrics after fitting K-Means
on the coordinates stored in the archived embedding CSV files.

Example:
    python evaluate_archived_embeddings.py \
      --dataset S1_TRUE_L_patient_dataset_with_reports_2_x.xlsx

Requires: numpy, pandas, scipy, scikit-learn, openpyxl (for .xlsx).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

# Table I, IEEE ACDSA 2025: Accuracy, NMI, silhouette, Davies–Bouldin.
# PCA is supplementary and does not appear in the published table.
PUBLISHED = {
    ("t-SNE", 2): (0.899, 0.701, 0.736, 0.355),
    ("t-SNE", 3): (0.738, 0.450, 0.349, 1.060),
    ("UMAP", 2): (0.906, 0.717, 0.795, 0.274),
    ("UMAP", 3): (0.906, 0.717, 0.792, 0.272),
    ("t-SNE-PSO", 2): (0.906, 0.717, 0.820, 0.239),
    ("t-SNE-PSO", 3): (0.906, 0.717, 0.837, 0.222),
}
ARCHIVES = [
    ("t-SNE", 2, "tSNE-embedding-NC2.csv"),
    ("t-SNE", 3, "tSNE-embedding-NC3.csv"),
    ("UMAP", 2, "UMAP-embedding-NC2.csv"),
    ("UMAP", 3, "UMAP-embedding-NC3.csv"),
    ("t-SNE-PSO", 2, "t-SNE-PSO-embedding-NC2.csv"),
    ("t-SNE-PSO", 3, "t-SNE-PSO-embedding-NC3.csv"),
    ("PCA", 2, "PCA-embedding-NC2.csv"),
    ("PCA", 3, "PCA-embedding-NC3.csv"),
]


def load_labels(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".xlsx":
        frame = pd.read_excel(path, sheet_name="patient_dataset_with_reports_2")
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError("Dataset must be an .xlsx or .csv file")
    if "Hypertension_Stage" not in frame.columns:
        raise ValueError("Missing Hypertension_Stage column")
    labels = frame["Hypertension_Stage"].to_numpy(dtype=int)
    if len(labels) != 149:
        print(f"WARNING: Found {len(labels)} records; archived snapshot had 149.")
    return labels


def matched_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    actual = np.unique(y_true)
    predicted = np.unique(y_pred)
    confusion = np.array(
        [[np.count_nonzero((y_pred == p) & (y_true == a)) for a in actual]
         for p in predicted],
        dtype=np.int64,
    )
    row, col = linear_sum_assignment(-confusion)
    return float(confusion[row, col].sum() / len(y_true))


def evaluate(embedding: np.ndarray, labels: np.ndarray) -> dict:
    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains non-finite coordinates")
    if embedding.ndim != 2 or embedding.shape[0] != len(labels):
        raise ValueError("Embedding shape does not match dataset records")
    clusters = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(embedding)
    return {
        "accuracy": matched_accuracy(labels, clusters),
        "nmi": float(normalized_mutual_info_score(labels, clusters)),
        "silhouette": float(silhouette_score(embedding, clusters)),
        "dbi": float(davies_bouldin_score(embedding, clusters)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("S1_TRUE_L_patient_dataset_with_reports_2_x.xlsx"),
        help="Canonical S1 workbook (default) or equivalent .csv dataset",
    )
    parser.add_argument("--embeddings-dir", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/recomputed_metrics.csv"))
    args = parser.parse_args()

    labels = load_labels(args.dataset)
    print(f"Loaded {len(labels)} patient records: " +
          ", ".join(f"stage {k}={int((labels == k).sum())}" for k in sorted(set(labels))))
    fields = ["method", "filename", "nominal_dimensions", "actual_dimensions", "patients",
              "accuracy", "nmi", "silhouette", "dbi",
              "paper_accuracy", "paper_nmi", "paper_silhouette", "paper_dbi", "status"]
    rows = []
    for method, dimensions, filename in ARCHIVES:
        path = args.embeddings_dir / filename
        if not path.is_file():
            print(f"SKIP: {filename} (not present)")
            continue
        embedding = pd.read_csv(path).to_numpy(dtype=float)
        values = evaluate(embedding, labels)
        published = PUBLISHED.get((method, dimensions), (None,) * 4)
        nominal_ok = (embedding.shape[1] == dimensions)
        if not nominal_ok:
            status = "DIMENSION MISMATCH — not a valid nominal experiment"
        elif published[0] is None:
            status = "Supplementary (no published Table I row)"
        elif all(round(values[key], 3) == pub for key, pub in zip(
                ("accuracy", "nmi", "silhouette", "dbi"), published)):
            status = "All four match published precision"
        else:
            status = "Differs from published Table I"
        row = {
            "method": method, "filename": filename,
            "nominal_dimensions": dimensions, "actual_dimensions": embedding.shape[1],
            "patients": embedding.shape[0], **values,
            "paper_accuracy": published[0], "paper_nmi": published[1],
            "paper_silhouette": published[2], "paper_dbi": published[3],
            "status": status,
        }
        rows.append(row)
        print(f"{method:10s} {dimensions}D: ACC={values['accuracy']:.3f}, "
              f"NMI={values['nmi']:.3f}, SC={values['silhouette']:.3f}, "
              f"DBI={values['dbi']:.3f} | {status}")

    if not rows:
        raise FileNotFoundError("No archived embedding CSVs found in --embeddings-dir")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} comparison rows to: {args.output}")
    print("Note: Scores are from stored coordinates with a new, seeded K-Means fit;")
    print("      these are NOT complete reproductions of the original embedding runs.")


if __name__ == "__main__":
    main()
