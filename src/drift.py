from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def make_baseline_stats(X: pd.DataFrame, bins: int = 10) -> dict:
    stats = {}
    for col in X.columns:
        s = pd.to_numeric(X[col], errors="coerce").dropna()
        if s.empty:
            continue

        quantiles = np.unique(np.quantile(s, np.linspace(0, 1, bins + 1)))
        if len(quantiles) < 3:
            lo, hi = float(s.min()), float(s.max())
            if lo == hi:
                lo -= 1
                hi += 1
            quantiles = np.linspace(lo, hi, bins + 1)

        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
        counts, _ = np.histogram(s, bins=quantiles)
        proportions = (counts / max(counts.sum(), 1)).tolist()

        stats[col] = {
            "edges": [None if np.isneginf(x) or np.isposinf(x) else float(x) for x in quantiles],
            "proportions": proportions,
        }
    return stats


def _edges_from_json(edges):
    arr = []
    for i, x in enumerate(edges):
        if x is None:
            arr.append(-np.inf if i == 0 else np.inf)
        else:
            arr.append(float(x))
    return np.array(arr, dtype=float)


def psi_for_feature(series: pd.Series, baseline: dict) -> float:
    edges = _edges_from_json(baseline["edges"])
    expected = np.array(baseline["proportions"], dtype=float)

    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0

    actual_counts, _ = np.histogram(s, bins=edges)
    actual = actual_counts / max(actual_counts.sum(), 1)

    eps = 1e-6
    expected = np.clip(expected, eps, None)
    actual = np.clip(actual, eps, None)

    return float(np.sum((actual - expected) * np.log(actual / expected)))


def check_drift(X: pd.DataFrame, baseline_stats: dict) -> dict:
    feature_psi = {}
    for col, baseline in baseline_stats.items():
        if col in X.columns:
            feature_psi[col] = round(psi_for_feature(X[col], baseline), 6)

    max_psi = max(feature_psi.values(), default=0.0)
    mean_psi = float(np.mean(list(feature_psi.values()))) if feature_psi else 0.0

    if max_psi >= 0.25:
        status = "HIGH_DRIFT"
    elif max_psi >= 0.10:
        status = "MODERATE_DRIFT"
    else:
        status = "STABLE"

    return {
        "status": status,
        "max_psi": round(max_psi, 6),
        "mean_psi": round(mean_psi, 6),
        "feature_psi": feature_psi,
        "retrain_recommended": bool(max_psi >= 0.25),
    }


def load_baseline():
    return json.loads((ROOT / "models" / "baseline_stats.json").read_text(encoding="utf-8"))
