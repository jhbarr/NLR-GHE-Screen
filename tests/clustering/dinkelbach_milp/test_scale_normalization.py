import pytest
import numpy as np
import itertools
import cplex

from nlr_ghe_screen.clustering.dinkelbach_milp import solve_dinkelbach_milp
from load_scenarios import (
    solve_scenario,
    scenario_random,
    scenario_large_beats_small,
    scenario_intensity_not_size,
    scenario_imbalance_scale,
    scenario_raw_peak_tie_broken_by_area
)

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

TOL = dict(rel=1e-3, abs=1e-3)

DETERMINISTIC = [
    scenario_large_beats_small(),
    scenario_intensity_not_size(),
    scenario_imbalance_scale(),
    scenario_raw_peak_tie_broken_by_area(),
]

RANDOM = [
    scenario_random(seed=s, seed_idx=s, k_min=2 + (s % 3), heating_fraction=hf)
    for s, hf in zip(range(6), [0.5, 0.5, 0.3, 0.7, 0.5, 0.5])
]

ALL_SCENARIOS = DETERMINISTIC
KNOWN = [s for s in ALL_SCENARIOS if s.expected_peak is not None]

by_name = lambda s: s.name


# ---------------------------------------------------------------------------
# Test - Peak Reduction
# ---------------------------------------------------------------------------

# ---- Helper Functions ----

def cluster_metrics(loads, areas, sel):
    """Raw and per-area peak/imbalance of a cluster."""
    L = np.asarray(loads, float)
    sel = list(sel)
    net = L[sel].sum(axis=0)
    N = float(np.asarray(areas, float)[sel].sum())
    peak, imb = float(np.abs(net).max()), float(abs(net.sum()))

    return dict(peak=peak, imb=imb, area=N, p=peak / N, q=imb / N)

def all_clusters(n, seed_idx, k_min):
    others = [i for i in range(n) if i != seed_idx]
    for r in range(max(k_min - 1, 0), len(others) + 1):
        for combo in itertools.combinations(others, r):
            yield tuple(sorted((seed_idx, *combo)))

def brute_force_combined(loads, areas, seed_idx, k_min, w_p, w_q):
    """Minimize (w_p*peak + w_q*imbalance) / area."""
    best = (np.inf, None)
    for sel in all_clusters(len(loads), seed_idx, k_min):
        m = cluster_metrics(loads, areas, sel)
        val = w_p * m["p"] + w_q * m["q"]
        if val < best[0]:
            best = (val, sel)

    return best

# --------------------------

class TestScaleNormalization:
    """
    Test cases to investigate how the MILP approaches problems where buildings have different scales
    and whether it will select the best normalized option
    """

# ---- core correctness ----
    # 1. 
    @pytest.mark.parametrize('sc', RANDOM, ids=by_name)
    def test_milp_matches_brute_force(self, sc):
        # Check that the MILP has the same result as a brute force method for random methodology
        weight_peak = 1.0
        weight_net = 0.0

        res = solve_scenario(sc, weight_peak=weight_peak, weight_net=weight_net, distance_cap=cplex.infinity)
        expected_ratio, expected_selected = brute_force_combined(sc.loads, sc.areas, sc.seed_idx, sc.k_min, w_p=weight_peak, w_q=weight_net)

        assert sorted(res['selected']) == sorted(expected_selected)
        assert res['cost_ratio'] == pytest.approx(expected_ratio, rel=1e-6)
    

# ---- known optima ----
    # 2. 
    def test_chooses_large_profiles_peak(self):
        # Check that the MILP selects the larger but more complementary loads
        # when peak is the main cost component 
        sc = scenario_large_beats_small()
        weight_peak = 1.0
        weight_net = 0.0
        res = solve_scenario(sc, weight_peak=weight_peak, weight_net=weight_net, distance_cap=cplex.infinity)

        assert res['cost_ratio'] == sc.expected_cost_ratio

    # 3. 
    def test_not_biggest_area(self):
        # Check that the MILP doesn't always try to just create a cluster with the largest area possible
        sc = scenario_intensity_not_size()
        weight_peak = 1.0
        weight_net = 0.0
        res = solve_scenario(sc, weight_peak=weight_peak, weight_net=weight_net, distance_cap=cplex.infinity)

        assert res['cost_ratio'] == sc.expected_cost_ratio

    # 4. 
    def test_chooses_large_profiles_imbalance(self):
        # Check that the MILP selects the larger but more complementary loads
        # when imbalance is the main cost component 
        sc = scenario_imbalance_scale()
        weight_peak = 0.0
        weight_net = 1.0
        res = solve_scenario(sc, weight_peak=weight_peak, weight_net=weight_net, distance_cap=cplex.infinity)

        assert res['cost_ratio'] == sc.expected_cost_ratio

    # 5. 
    def test_tie_broken_by_area(self):
        # Check that if two clusters have the same raw peak value
        # the tie is broken by the area normalization
        sc = scenario_raw_peak_tie_broken_by_area()
        weight_peak = 1.0
        weight_net = 0.0
        res = solve_scenario(sc, weight_peak=weight_peak, weight_net=weight_net, distance_cap=cplex.infinity)

        assert res['cost_ratio'] == sc.expected_cost_ratio

# ---- invariance ----
    # 6.
    @pytest.mark.parametrize("sc, weights", [
         (scenario_large_beats_small(), (1.0, 0.0)),
         (scenario_imbalance_scale(), (0.0, 1.0))
    ])
    def test_uniform_load_scaling_invariance(self, sc, weights):
        # Check that multiplying each area by a uniform factor does not affect the cluster selection
        sc = scenario_large_beats_small()
        weight_peak = weights[0]
        weight_net = weights[1]
        
        res1 = solve_scenario(sc, weight_peak=weight_peak, weight_net=weight_net, distance_cap=cplex.infinity)
        res2 = solve_scenario(sc, weight_peak=weight_peak, weight_net=weight_net, distance_cap=cplex.infinity, loads=sc.loads * 3.0)

        assert res1['selected'] == res2['selected']
        assert res2['cost_ratio'] == pytest.approx(res1['cost_ratio'] * 3.0, rel=1e-6)

    # 7. 
    @pytest.mark.parametrize("sc, weights", [
            (scenario_large_beats_small(), (1.0, 0.0)),
            (scenario_imbalance_scale(), (0.0, 1.0))
    ])
    def test_uniform_area_scaling_invariance(self, sc, weights):
        # Check that multiplying each area by a uniform factor does not affect the cluster selection
        sc = scenario_large_beats_small()
        weight_peak = weights[0]
        weight_net = weights[1]
        
        res1 = solve_scenario(sc, weight_peak=weight_peak, weight_net=weight_net, distance_cap=cplex.infinity)
        res2 = solve_scenario(sc, weight_peak=weight_peak, weight_net=weight_net, distance_cap=cplex.infinity, areas=sc.areas * 7.0)

        assert res1['selected'] == res2['selected']
        assert res2['cost_ratio'] == pytest.approx(res1['cost_ratio'] / 7.0, rel=1e-6)
    