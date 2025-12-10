from typing import Sequence, Optional, Dict
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

NumericArray = Sequence[float]


def _to_numpy(x: NumericArray) -> np.ndarray:
    """Convert list/Series/DataFrame to clean 1D numpy array."""
    if isinstance(x, (pd.Series, pd.DataFrame)):
        x = x.squeeze()
    return np.asarray(x, dtype=float)


# ------------------------------------------------------------------------------
#                             POPULATION STABILITY INDEX
# ------------------------------------------------------------------------------

def population_stability_index(
    a: NumericArray,
    b: NumericArray,
    bins: Optional[int] = 10,
    eps: float = 1e-8
) -> float:
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)

    values = np.concatenate([a_np, b_np])

    # --- FIX: handle low unique → discrete values ---
    unique_vals = np.unique(values)
    if len(unique_vals) <= 5:  
        # treat as categorical: one bin per unique value
        bins_edges = np.sort(unique_vals)
        # add small edge so that histogram has correct bins
        bins_edges = np.append(bins_edges, bins_edges[-1] + 1e-6)

        pa, _ = np.histogram(a_np, bins=bins_edges)
        pb, _ = np.histogram(b_np, bins=bins_edges)

    else:
        # normal quantile-based binning
        quantiles = np.linspace(0, 1, bins + 1)
        bins_edges = np.unique(np.quantile(values, quantiles))
        if len(bins_edges) <= 1:
            bins_edges = np.linspace(values.min(), values.max(), bins + 1)

        pa, _ = np.histogram(a_np, bins=bins_edges)
        pb, _ = np.histogram(b_np, bins=bins_edges)

    pa = pa.astype(float) / (pa.sum() + eps)
    pb = pb.astype(float) / (pb.sum() + eps)

    pa = np.clip(pa, eps, 1.0)
    pb = np.clip(pb, eps, 1.0)

    psi = np.sum((pa - pb) * np.log(pa / pb))
    return float(psi)


# ------------------------------------------------------------------------------
#                         JENSEN-SHANNON DIVERGENCE
# ------------------------------------------------------------------------------

def jensen_shannon_divergence(
    a: NumericArray,
    b: NumericArray,
    bins: int = 100,
    eps: float = 1e-12
) -> float:
    """
    Compute Jensen-Shannon divergence between a and b.
    Discretized via histogram → robust for arbitrary numeric data.

    scipy.spatial.distance.jensenshannon returns sqrt(JS), so we square it.
    """
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)

    if len(a_np) == 0 or len(b_np) == 0:
        return 0.0

    # Shared support bins
    values = np.concatenate([a_np, b_np])
    hist_bins = np.linspace(values.min(), values.max(), bins + 1)

    pa, _ = np.histogram(a_np, bins=hist_bins, density=True)
    pb, _ = np.histogram(b_np, bins=hist_bins, density=True)

    # Numerical stability
    pa = pa + eps
    pb = pb + eps

    pa = pa / pa.sum()
    pb = pb / pb.sum()

    js_sqrt = jensenshannon(pa, pb)
    return float(js_sqrt ** 2)


# ------------------------------------------------------------------------------
#                              KS STATISTIC
# ------------------------------------------------------------------------------

def ks_statistic(a: NumericArray, b: NumericArray) -> Dict[str, float]:
    """
    2-sample Kolmogorov–Smirnov test.
    Returns dict {"statistic": float, "pvalue": float}
    """
    a_np = _to_numpy(a)
    b_np = _to_numpy(b)

    if len(a_np) == 0 or len(b_np) == 0:
        return {"statistic": 0.0, "pvalue": 1.0}

    res = ks_2samp(a_np, b_np)
    return {"statistic": float(res.statistic), "pvalue": float(res.pvalue)}
