import numpy as np
from scipy.stats import entropy, wasserstein_distance
from typing import Tuple

def mmd(x: np.ndarray, y: np.ndarray, kernel='rbf', gamma=1.0) -> float:
    """Compute Maximum Mean Discrepancy between two arrays."""
    
    def rbf_kernel(a, b, gamma):
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)
        sq_dist = (a - b.T) ** 2
        return np.exp(-gamma * sq_dist)

    K_XX = rbf_kernel(x, x, gamma).mean()
    K_YY = rbf_kernel(y, y, gamma).mean()
    K_XY = rbf_kernel(x, y, gamma).mean()

    return K_XX + K_YY - 2 * K_XY


def kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 20) -> float:
    """Compute KL divergence between two distributions."""
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)

    p_hist += 1e-8
    q_hist += 1e-8

    return entropy(p_hist, q_hist)


def wasserstein_dist(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Wasserstein distance."""
    return wasserstein_distance(x, y)
 
