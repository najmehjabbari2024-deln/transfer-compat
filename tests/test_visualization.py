import pandas as pd
from transfer_compat.visualization import Visualizer
from .test_preprocessing import _make_test_data
import matplotlib

# FOR HEADLESS TESTING
matplotlib.use("Agg")


def _get_visualizer():
    df_a, df_b = _make_test_data()
    shared = [c for c in df_a.columns if c in df_b.columns]
    return Visualizer(df_a, df_b, shared)


# -------------------------------------------------------
# 1. test distribution comparison
# -------------------------------------------------------
def test_distribution_comparison():
    vis = _get_visualizer()
    result = vis.compare_distributions()

    assert result["status"] == "ok"
    assert "figures" in result
    assert isinstance(result["figures"], dict)

    for col in vis.shared_features:
        assert col in result["figures"]
        fig = result["figures"][col]
        assert hasattr(fig, "axes")


# -------------------------------------------------------
# 2. test wasserstein trend
# -------------------------------------------------------
def test_wasserstein_trend():
    vis = _get_visualizer()
    result = vis.wasserstein_trend()

    assert result["status"] == "ok"
    assert "distances" in result
    assert "figure" in result

    for col, d in result["distances"].items():
        assert isinstance(d, float)
        assert d >= 0

    fig = result["figure"]
    assert hasattr(fig, "axes")


# -------------------------------------------------------
# 3. test drift radar chart
# -------------------------------------------------------
def test_drift_radar():
    vis = _get_visualizer()
    result = vis.drift_radar()

    assert result["status"] == "ok"
    assert "radar_values" in result
    assert "figure" in result

    vals = result["radar_values"]

    assert len(vals) == len(vis.shared_features) + 1
    assert all(0 <= v <= 1 for v in vals)

    fig = result["figure"]
    assert hasattr(fig, "axes")


# -------------------------------------------------------
# 4. test domain overlap heatmap
# -------------------------------------------------------
def test_domain_overlap():
    vis = _get_visualizer()
    result = vis.domain_overlap()

    assert result["status"] == "ok"
    assert "overlap_matrix" in result
    assert "figure" in result

    matrix = result["overlap_matrix"]
    assert isinstance(matrix, pd.DataFrame)

    assert matrix.shape[0] == matrix.shape[1] == len(vis.shared_features)

    for v in matrix.values.flatten():
        assert float(v) >= 0

    fig = result["figure"]
    assert hasattr(fig, "axes")
