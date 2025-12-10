import pandas as pd
import numpy as np

from transfer_compat.preprocessing import Preprocessor


def _make_test_data():
    df_a = pd.DataFrame({
        "age": [20, 25, 30, None, 200],       # outlier + missing
        "income": [3000, None, 7000, 8000, 100000],  # missing + big outlier
        "gender": ["M", "F", None, "M", "F"]         # missing categorical
    })

    df_b = pd.DataFrame({
        "age": [22, None, 35, 40, 300],       # missing + outlier
        "income": [2500, 4000, None, 9000, 50000],  # missing + outlier
        "gender": ["F", None, "F", "M", "M"]        # missing categorical
    })

    return df_a, df_b


# ---------------------------------------------------------
# 1) TYPE HARMONIZATION
# ---------------------------------------------------------
def test_type_harmonization():
    df_a, df_b = _make_test_data()
    prep = Preprocessor(df_a, df_b)

    result = prep.harmonize_types()

    for col in prep.shared_features:
        assert result["df_a_types"][col] == result["df_b_types"][col]


# ---------------------------------------------------------
# 2) MISSING VALUE ALIGNMENT
# ---------------------------------------------------------
def test_missing_value_alignment():
    df_a, df_b = _make_test_data()
    prep = Preprocessor(df_a, df_b)

    prep.harmonize_types()
    result = prep.align_missing_values()

    assert result["missing_after_a"].sum() == 0
    assert result["missing_after_b"].sum() == 0


# ---------------------------------------------------------
# 3) OUTLIER HARMONIZATION
# ---------------------------------------------------------
def test_outlier_harmonization():
    df_a, df_b = _make_test_data()
    prep = Preprocessor(df_a, df_b)

    prep.harmonize_types()
    prep.align_missing_values()
    result = prep.harmonize_outliers()

    assert result["status"] == "outliers_clipped_iqr"

    assert prep.df_a["income"].max() < 100000
    assert prep.df_b["income"].max() < 50000


# ---------------------------------------------------------
# 4) NORMALIZATION
# ---------------------------------------------------------
def test_normalization():
    df_a, df_b = _make_test_data()
    prep = Preprocessor(df_a, df_b)

    prep.harmonize_types()
    prep.align_missing_values()
    prep.harmonize_outliers()

    result = prep.normalize()

    df_a_norm = result["df_a_normalized"]
    df_b_norm = result["df_b_normalized"]

    assert df_a_norm.shape == df_b_norm.shape

    combined = pd.concat([df_a_norm, df_b_norm]).mean()
    assert all(abs(combined[col]) < 1e-6 for col in combined.index)


# ---------------------------------------------------------
# 5) FULL PIPELINE
# ---------------------------------------------------------
def test_full_preprocess_pipeline():
    df_a, df_b = _make_test_data()
    prep = Preprocessor(df_a, df_b)

    result = prep.full_preprocess()

    assert "df_a_normalized" in result
    assert "df_b_normalized" in result

    assert result["df_a_normalized"].shape == result["df_b_normalized"].shape
