# tests/test_shift.py
import numpy as np
from transfer_compat.metrics.shift import (
    population_stability_index,
    jensen_shannon_divergence,
    ks_statistic,
)

def test_psi_identical():
    a = np.array([1,2,3,4,5,6,7,8,9,10])
    b = a.copy()
    psi = population_stability_index(a, b, bins=5)
    assert psi == 0.0 or psi < 1e-8

def test_psi_shift():
    a = np.array([1]*50 + [10]*50)
    b = np.array([1]*10 + [10]*90)
    psi = population_stability_index(a, b, bins=5)
    assert psi > 0.0

def test_js_identical():
    a = np.random.normal(0,1,1000)
    b = a.copy()
    js = jensen_shannon_divergence(a, b, bins=50)
    assert js == 0.0 or js < 1e-9

def test_js_shift():
    a = np.random.normal(0,1,1000)
    b = np.random.normal(1.0,1,1000)  # shifted mean
    js = jensen_shannon_divergence(a, b, bins=50)
    assert js > 0.0

def test_ks_identical():
    a = np.array([0,1,2,3,4])
    b = a.copy()
    res = ks_statistic(a, b)
    assert res["statistic"] == 0.0 or res["statistic"] < 1e-8
    assert res["pvalue"] > 0.99

def test_ks_shift():
    a = np.random.normal(0,1,500)
    b = np.random.normal(2.0,1,500)
    res = ks_statistic(a, b)
    assert res["statistic"] > 0.0
    assert res["pvalue"] < 0.05
