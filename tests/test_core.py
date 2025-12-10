import pandas as pd
import numpy as np

from transfer_compat.core import DatasetComparator


def _make_test_data():
    # dataset A (source)
    df_a = pd.DataFrame({
        "age": np.random.normal(40, 10, 200),
        "income": np.random.normal(5000, 1000, 200),
        "gender": np.random.choice([0, 1], 200)
    })

    # dataset B (target) — drifted
    df_b = pd.DataFrame({
        "age": np.random.normal(50, 12, 200),
        "income": np.random.normal(5200, 1200, 200),
        "gender": np.random.choice([0, 1], 200)
    })

    return df_a, df_b


# ---------------------------------------------------------
# 1) SHAPE + BASIC STRUCTURE
# ---------------------------------------------------------

def test_shape_comparison():
    df_a, df_b = _make_test_data()
    comp = DatasetComparator(df_a, df_b)

    result = comp.shape_comparison()

    assert "dataset_a_rows" in result
    assert "dataset_b_rows" in result
    assert result["same_number_of_columns"] is True
    assert len(result["shared_columns"]) == 3


# ---------------------------------------------------------
# 2) FEATURE ALIGNMENT
# ---------------------------------------------------------

def test_feature_alignment():
    df_a, df_b = _make_test_data()
    comp = DatasetComparator(df_a, df_b)

    aligned = comp.feature_alignment()

    # output keys
    assert "df_a_aligned" in aligned
    assert "df_b_aligned" in aligned
    assert "shared_features" in aligned

    # shapes should match
    assert aligned["df_a_aligned"].shape == aligned["df_b_aligned"].shape

    # check standardization (mean ≈ 0)
    a_age = aligned["df_a_aligned"]["age"].mean()
    assert abs(a_age) < 1e-6


# ---------------------------------------------------------
# 3) DOMAIN ADAPTATION SCORE
# ---------------------------------------------------------

def test_domain_adaptation_score():
    df_a, df_b = _make_test_data()
    comp = DatasetComparator(df_a, df_b)

    scores = comp.domain_adaptation_score()

    assert "mean_ks" in scores
    assert "mean_js" in scores
    assert "mean_psi" in scores

    # scores must be non-negative
    assert scores["mean_ks"] >= 0
    assert scores["mean_js"] >= 0
    assert scores["mean_psi"] >= 0

    # datasets are drifted → scores should NOT be zero
    assert scores["mean_ks"] > 0
    assert scores["mean_js"] > 0


# ---------------------------------------------------------
# 4) TRANSFERABILITY INDEX (0–1)
# ---------------------------------------------------------

def test_transferability_index():
    df_a, df_b = _make_test_data()
    comp = DatasetComparator(df_a, df_b)

    score = comp.transferability_index()

    assert 0.0 <= score <= 1.0

    # Since datasets differ → transferability shouldn’t be 1
    assert score < 1.0
