import itertools
import numpy as np
import pytest
  
from nlr_ghe_screen.clustering.milp import solve_cluster 
from load_scenarios import (
    solve_scenario,
    scenario_all_cooling,
    scenario_all_heating,
    scenario_peak_vs_net_conflict,
    scenario_perfect_cancellation,
    scenario_phase_shifted_sinusoids,
    scenario_random,
    scenario_staggered_spikes,
    scenario_subset_sum,
    scenario_three_way_tradeoff,
    scenario_zero_energy_seed,
)

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

OBJ_TOL = dict(rel=1e-3, abs=1e-4)
SLACK = 1e-2 

CONFLICT = scenario_peak_vs_net_conflict()
THREE_WAY = scenario_three_way_tradeoff()

DETERMINISTIC = [
    scenario_perfect_cancellation(),
    scenario_all_heating(),
    scenario_all_cooling(),
    scenario_phase_shifted_sinusoids(),
    scenario_staggered_spikes(),
    scenario_subset_sum(),
    scenario_zero_energy_seed(),
    CONFLICT,
    THREE_WAY,
]

RANDOM = [
    scenario_random(seed=s, seed_idx=s, k_min=2 + (s % 3), heating_fraction=hf)
    for s, hf in zip(range(7), [0.5, 0.5, 0.3, 0.7, 0.5, 0.5, 0.4])
]

ALL_SCENARIOS = DETERMINISTIC + RANDOM
KNOWN = [s for s in ALL_SCENARIOS if s.expected_peak is not None]

WEIGHT_PAIRS = [(1.0, 0.0), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.0, 1.0)]

by_name = lambda s: s.name



# ---------------------------------------------------------------------------
# Test - Weighted Objective Functions
# ---------------------------------------------------------------------------

# ------------- Helper Functions ------------- 
def weights_from_ratio(lam):
    """(w_peak, w_net) with w_net / w_peak = lam and w_peak + w_net = 1.
    lam is the ratio between the peak weight and the imbalance weight 
    """
    return 1.0 / (1.0 + lam), lam / (1.0 + lam)
 
 
def normalisers(loads):
    """
    Get the normalizers for the peak and imbalance terms
    """
    L = np.asarray(loads, dtype=float)

    z_ref = max(np.abs(L).sum(axis=0).max(), 1e-9)
    q_ref = max(np.abs(L.sum(axis=1)).sum(), 1e-9)

    return z_ref, q_ref
 
 
def cluster_terms(loads, sel):
    """
    Normalised (peak, imbalance) of a cluster: (peak / Z_ref, imbalance / Q_ref).
    """
    L = np.asarray(loads, dtype=float)

    z_ref, q_ref = normalisers(L)
    net = L[list(sel)].sum(axis=0)

    return np.abs(net).max() / z_ref, abs(net.sum()) / q_ref


def weighted_objective(loads, sel, w_peak, w_net):
    """
    Get the weighted objective given the weights, normalizer values and the 
    actual current peak and imbalance values 
    """
    fp, fn = cluster_terms(loads, sel)
    return w_peak * fp + w_net * fn


def valid_clusters(n, seed_idx, k_min):
    """
    Aggregate all valid combinations of load profiles 
    """
    others = [i for i in range(n) if i != seed_idx]
    for r in range(max(k_min - 1, 0), len(others) + 1):
        for combo in itertools.combinations(others, r):
            yield sorted([seed_idx, *combo])


def brute_force_weighted(loads, seed_idx, k_min, w_peak, w_net):
    """
    best weighted objective, one cluster achieving it
    """
    best, best_sel = np.inf, None

    for sel in valid_clusters(len(loads), seed_idx, k_min):
        val = weighted_objective(loads, sel, w_peak, w_net)
        if val < best:
            best, best_sel = val, sel

    return best, best_sel


def expected_optima(sc):
    """(min achievable peak, min achievable imbalance) for a scenario: the values stored on the
    Scenario if present, otherwise found by brute force (e.g. for random scenarios with many
    buildings, where load_scenarios skips the enumeration and leaves the fields as None)."""
    L = np.asarray(sc.loads, dtype=float)
    peak, imbalance = sc.expected_peak, sc.expected_imbalance

    if peak is None:
        _, sel = brute_force_weighted(L, sc.seed_idx, sc.k_min, 1.0, 0.0)
        peak = float(np.abs(L[sel].sum(axis=0)).max())
    if imbalance is None:
        _, sel = brute_force_weighted(L, sc.seed_idx, sc.k_min, 0.0, 1.0)
        imbalance = float(abs(L[sel].sum(axis=0).sum()))

    return peak, imbalance


def crossover_ratio(loads, sel_a, sel_b):
    """lam at which clusters a and b have equal weighted objective, where a has the lower peak
    and b the lower imbalance:  fp_a + lam*fn_a = fp_b + lam*fn_b."""
    fp_a, fn_a = cluster_terms(loads, sel_a)
    fp_b, fn_b = cluster_terms(loads, sel_b)

    assert fp_a < fp_b and fn_b < fn_a, "clusters are not a peak-vs-imbalance trade-off pair"
    return (fp_b - fp_a) / (fn_a - fn_b)

# -----------------------------------------------

class TestWeightedObjective:
    """
    Tests to analyze the performance of the MILP when there are different weights given to 
    the different portions of the objective function
    """

# ---- core correctness ----
    # 1. 
    @pytest.mark.parametrize("sc", ALL_SCENARIOS, ids=by_name)
    @pytest.mark.parametrize("w_peak, w_net", WEIGHT_PAIRS)
    def test_objective_matches_brute_force(self, sc, w_peak, w_net):
        # Check that for all scenarios and for all weight combinations within those scenarios
        # the MILP result matches the brute force results
        # The solver returns the best 'weighted' objective result 
        res = solve_scenario(sc, w_peak, w_net, 0.0)
        best, _ = brute_force_weighted(sc.loads, sc.seed_idx, sc.k_min, w_peak, w_net)
        returned = weighted_objective(sc.loads, res["selected"], w_peak, w_net)
    
        assert res["selected"] is not None
        assert res["objective"] == pytest.approx(best, **OBJ_TOL)
        assert returned == pytest.approx(best, **OBJ_TOL)


    # 2. 
    @pytest.mark.parametrize("sc", ALL_SCENARIOS, ids=by_name)
    def test_extreme_weights_recover_single_objective_optima(self, sc):
        # Check that when the weights are skewed one way or the other, the minimum peak / imbalance
        # is returned by the solver
        z_ref, q_ref = normalisers(sc.loads)
        expected_peak, expected_imbalance = expected_optima(sc)
    
        peak_only = solve_scenario(sc, 1.0, 0.0, 0.0)
        assert peak_only["peak"] == pytest.approx(expected_peak, rel=1e-3, abs=1e-3)
    
        net_only = solve_scenario(sc, 0.0, 1.0, 0.0)
        assert net_only["imbalance"] == pytest.approx(
            expected_imbalance, rel=1e-3, abs=max(1e-4 * q_ref, 1e-6)
        )


# ---- analytic crossover points 
    # 3. 
    """*** Look at this test more ***"""
    def test_conflict_scenario_switches_at_the_analytic_crossover(self):
        # Check the point at which the peak / imbalance take each other over for priority 
        # in the objective function
        lam_star = crossover_ratio(CONFLICT.loads, [0, 2], [0, 1])
        print(lam_star)
 
        below = solve_scenario(CONFLICT, *weights_from_ratio(lam_star / 2), 0.0)
        above = solve_scenario(CONFLICT, *weights_from_ratio(lam_star * 2), 0.0)
    
        assert sorted(below["selected"]) == [0, 2]   # peak still wins
        assert sorted(above["selected"]) == [0, 1]   # imbalance now wins


    """*** Look at this test more ***"""
    # 4. 
    def test_three_way_tradeoff(self):
        # Check how the MILP solutions change as the weights change 
        # Show that they cannot improve on one metric without sacrificing another 
        # Pareto Frontier
        L = THREE_WAY.loads
        lam_12 = crossover_ratio(L, [0, 1], [0, 2])
        lam_23 = crossover_ratio(L, [0, 2], [0, 3])
        assert lam_12 < lam_23   # precondition: the middle point is genuinely on the frontier
    
        lams = [lam_12 / 2, np.sqrt(lam_12 * lam_23), lam_23 * 2]
        expected = [[0, 1], [0, 2], [0, 3]]
        for lam, exp in zip(lams, expected):
            res = solve_scenario(THREE_WAY, *weights_from_ratio(lam), 0.0)
            assert sorted(res["selected"]) == exp, f"lam={lam:.3f}"


    """*** Look at this test more ***"""
    # 5. 
    def test_equal_weights_in_three_way(self):
        # Analyze the clusters chosen when there are equal weights in the objective funciton
        res = solve_scenario(THREE_WAY, 0.5, 0.5, 0.0)

        assert sorted(res["selected"]) == [0, 1]



# ---- properties of trade off ----
    # 5. 
    @pytest.mark.parametrize("sc", RANDOM + [THREE_WAY, CONFLICT], ids=by_name)
    def test_sweep_is_monotone(self, sc):
        # Analyze how the peak and imbalance shrink / grow as their weight ratio changes
        # the peak objective term should shrink while the imbalance term should grow
        lams = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
        points = []
        for lam in lams:
            res = solve_scenario(sc, *weights_from_ratio(lam), 0.0)
            points.append(cluster_terms(sc.loads, res["selected"]))   # (peak, imbalance), normalised
    
        for (p_a, q_a), (p_b, q_b) in zip(points, points[1:]):
            assert q_b <= q_a + SLACK, "imbalance rose as its weight increased"
            assert p_b >= p_a - SLACK, "peak fell as its weight decreased"


    # 6.    
    @pytest.mark.parametrize("sc", RANDOM, ids=by_name)
    @pytest.mark.parametrize("w_peak, w_net", [(0.5, 0.5), (0.9, 0.1), (0.1, 0.9)])
    def test_chosen_cluster_is_not_dominated(self, sc, w_peak, w_net):
        # Check that as the weights are not totally dominant one way or the other
        # the relative objective terms of the peak and imbalance are not dominated either
        # The optimal brute force peak and imbalance should be better than the MILP's results
        res = solve_scenario(sc, w_peak, w_net, 0.0)
        p, q = cluster_terms(sc.loads, res["selected"])

        for sel in valid_clusters(len(sc.loads), sc.seed_idx, sc.k_min):
            p2, q2 = cluster_terms(sc.loads, sel)
            dominates = (p2 <= p + 1e-6 and q2 <= q + 1e-6) and (p2 < p - SLACK or q2 < q - SLACK)
            
            assert not dominates, f"cluster {sel} dominates the solver's choice {res['selected']}"


    # 7 
    @pytest.mark.parametrize("sc", RANDOM, ids=by_name)
    @pytest.mark.parametrize("factor", [0.01, 100.0])
    def test_tradeoff_is_unit_independent(self, sc, factor):
        # Check that the scale of the loads is independent of how the weights affect the cluster selection
        base = solve_scenario(sc, 0.5, 0.5, 0.0)
        scaled = solve_scenario(sc, 0.5, 0.5, 0.0, loads=sc.loads * factor)
        
        assert scaled["objective"] == pytest.approx(base["objective"], **OBJ_TOL)