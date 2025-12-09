import numpy as np
from transfer_compat.metrics.distribution import mmd, kl_divergence, wasserstein_dist

def test_mmd_basic():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.1, 1.9, 3.2])

    result = mmd(x, y)
    assert result >= 0


def test_kl_divergence_basic():
    x = np.random.normal(0, 1, 1000)
    y = np.random.normal(0, 1, 1000)

    result = kl_divergence(x, y)
    assert result >= 0


def test_wasserstein_distance_basic():
    x = np.array([0, 1, 2])
    y = np.array([1, 2, 3])

    result = wasserstein_dist(x, y)
    assert result >= 0
 
