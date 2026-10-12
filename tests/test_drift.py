import pandas as pd
from src.drift import make_baseline_stats, check_drift


def test_psi_stable_for_same_distribution():
    X = pd.DataFrame({"x": list(range(100))})
    baseline = make_baseline_stats(X)
    result = check_drift(X.copy(), baseline)
    assert result["status"] == "STABLE"
    assert result["max_psi"] < 0.1
