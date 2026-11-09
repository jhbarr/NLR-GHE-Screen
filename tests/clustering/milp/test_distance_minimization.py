import itertools
import numpy as np
import pytest
import time
  
from nlr_ghe_screen.clustering.milp import solve_cluster 
from load_scenarios import (
    Scenario,
    solve_scenario,
    random_loads,
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
    scenario_shuffled_distances,
    scenario_near_decoy_far_complement,
    scenario_ladder
)

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

DETERMINISTIC = [
    scenario_perfect_cancellation(),
    scenario_all_heating(),
    scenario_all_cooling(),
    scenario_phase_shifted_sinusoids(),
    scenario_staggered_spikes(),
    scenario_subset_sum(),
    scenario_zero_energy_seed(),
]

RANDOM = [
    scenario_random(seed=s, seed_idx=s, k_min=2 + (s % 3), heating_fraction=hf)
    for s, hf in zip(range(7), [0.5, 0.5, 0.3, 0.7, 0.5, 0.5, 0.4])
]

ALL_SCENARIOS = DETERMINISTIC + RANDOM
 
by_name = lambda s: s.name

LOAD_VARIANTS = {
    "random": lambda n, T: random_loads(n, T, seed=1),
    "zeros": lambda n, T: np.zeros((n, T)),
    "huge": lambda n, T: random_loads(n, T, seed=2) * 1e6,
    "all_cooling": lambda n, T: -np.abs(random_loads(n, T, seed=3)),
    "identical": lambda n, T: np.tile(random_loads(1, T, seed=4), (n, 1)),
}

# ---------------------------------------------------------------------------
# Test - Distance Minimization 
# ---------------------------------------------------------------------------

# ------------- Helper Functions ------------- 
def d_ref(distances):
    return max(float(np.sum(distances)), 1e-9)
 
 
def nearest_k_cluster(distances, seed_idx, k_min):
    """Closed-form optimum: the seed plus the (k_min - 1) nearest other buildings."""
    d = np.asarray(distances, dtype=float)
    others = [int(i) for i in np.argsort(d, kind="stable") if i != seed_idx]
    return sorted([seed_idx, *others[: max(k_min - 1, 0)]])
 
 
def min_total_distance(distances, seed_idx, k_min):
    d = np.asarray(distances, dtype=float)
    return float(d[nearest_k_cluster(d, seed_idx, k_min)].sum())
 
 
def approx_total(expected, distances):
    return pytest.approx(expected, rel=1e-3, abs=max(1e-4 * d_ref(distances), 1e-6))
 
 
def normalisers(loads, distances):
    L = np.asarray(loads, dtype=float)
    z_ref = max(np.abs(L).sum(axis=0).max(), 1e-9)
    q_ref = max(np.abs(L.sum(axis=1)).sum(), 1e-9)
    return z_ref, q_ref, d_ref(distances)
 
 
def objective_terms(loads, distances, sel):
    """Normalised (peak, imbalance, distance) of a cluster."""
    L = np.asarray(loads, dtype=float)
    d = np.asarray(distances, dtype=float)
    z_ref, q_ref, dr = normalisers(L, d)
    net = L[list(sel)].sum(axis=0)
    return np.abs(net).max() / z_ref, abs(net.sum()) / q_ref, d[list(sel)].sum() / dr
 
 
def weighted_objective(loads, distances, sel, w_peak, w_net, w_dist):
    fp, fn, fd = objective_terms(loads, distances, sel)
    return w_peak * fp + w_net * fn + w_dist * fd
 
 
def valid_clusters(n, seed_idx, k_min):
    others = [i for i in range(n) if i != seed_idx]
    for r in range(max(k_min - 1, 0), len(others) + 1):
        for combo in itertools.combinations(others, r):
            yield sorted([seed_idx, *combo])
 
 
def brute_force_weighted(loads, distances, seed_idx, k_min, w_peak, w_net, w_dist):
    best, best_sel = np.inf, None
    for sel in valid_clusters(len(loads), seed_idx, k_min):
        val = weighted_objective(loads, distances, sel, w_peak, w_net, w_dist)
        if val < best:
            best, best_sel = val, sel
    return best, best_sel

# -----------------------------------------------

class TestDistanceMinimization:
    """
    Tests to analyze the performance of the MILP at minimizing the distance between buildings 
    in a possible cluster
    """

# ---- core correctness ----
    @pytest.mark.parametrize("seed_rank", ["nearest", "middle", "farthest"])
    @pytest.mark.parametrize("k_min", [1, 3, 6, 12])
    def test_selects_seed_plus_nearest_buildings(self, seed_rank, k_min):
        # Check that despite whatever the seed is the MILP selects the set of buildings with the least total distance
        sc = scenario_ladder(k_min=k_min, seed_rank=seed_rank)
        res = solve_scenario(sc, 0.0, 0.0, 1.0)

        assert sorted(res['selected']) == nearest_k_cluster(sc.distances, sc.seed_idx, k_min)
        assert res['total_distance'] == approx_total(
            min_total_distance(sc.distances, sc.seed_idx, k_min), sc.distances
        )


    @pytest.mark.parametrize("sc", ALL_SCENARIOS, ids=by_name)
    @pytest.mark.parametrize("w_dist", [1.0, 0.4])
    def test_total_distances_matches_brute_force(self, sc, w_dist):
        # Check that the MILP's decision matches the brute force optimum
        res = solve_scenario(sc, 0.0, 0.0, weight_distance=w_dist)
        best, _ = brute_force_weighted(sc.loads, sc.distances, sc.seed_idx, sc.k_min, 0.0, 0.0, w_dist)
        returned = weighted_objective(sc.loads, sc.distances, res["selected"], 0.0, 0.0, w_dist)

        assert res['objective'] == pytest.approx(best, rel=1e-3, abs=1e-5)
        assert returned == pytest.approx(best, rel=1e-3, abs=1e-5)


    @pytest.mark.parametrize("sc", ALL_SCENARIOS, ids=by_name)
    def test_reported_metrics_are_consistent(self, sc):
        # Check that all of the metrics from the MILP are correct
        res = solve_scenario(sc, 0.0, 0.0, 1.0)
        sel = res["selected"]
        d = np.asarray(sc.distances, dtype=float)
        L = np.asarray(sc.loads, dtype=float)

        net = L[sel].sum(axis=0)
        np.testing.assert_allclose(res["net_curve"], net)
        
        assert res["total_distance"] == pytest.approx(d[sel].sum(), abs=1e-9)
        assert res["objective"] == pytest.approx(d[sel].sum() / d_ref(d), rel=1e-3, abs=1e-5)
        assert res["peak"] == pytest.approx(np.abs(net).max())


# ---- distance weight ----
    @pytest.mark.parametrize("w_dist", [1.0, 0.5, 0.1, 0.01])
    def test_objective_scales_linearly_with_distance_weight(self, w_dist):
        # Check that the objective result scales with the weight of the distance factor
        # in the absence of other weights
        sc = scenario_ladder(k_min=4)
        res = solve_scenario(sc, 0.0, 0.0, weight_distance=w_dist)
        expected = w_dist * min_total_distance(sc.distances, sc.seed_idx, 4) / d_ref(sc.distances)

        assert res['objective'] == pytest.approx(expected, rel=1e-3, abs=1e-5)


    @pytest.mark.parametrize("w_dist", [1.0, 0.1, 0.01])
    def test_selection_does_not_depend_on_distance_weight(self, w_dist):
        # Check that in the absence of other objective weights, the distance weight is negligible, 
        # so long that it is > 0.0
        sc = scenario_ladder(k_min=5)
        res = solve_scenario(sc, 0.0, 0.0, weight_distance=w_dist)
        assert sorted(res["selected"]) == nearest_k_cluster(sc.distances, sc.seed_idx, 5)


# ---- loads must not influence cluster ----
    @pytest.mark.parametrize("variant", list(LOAD_VARIANTS))
    def test_loads_are_irrelevant_when_only_distance_counts(self, variant):
        # Check that the MILP decision is not dependent on the profiles 
        # when the non distance weights are zero
        n, T = 10, 24
        d = scenario_shuffled_distances(n, seed=5)
        seed_idx = int(np.argsort(d)[n // 2])
        sc = Scenario(f"loads_{variant}", LOAD_VARIANTS[variant](n, T), d, seed_idx, k_min=4)
        res = solve_scenario(sc, 0.0, 0.0, 1.0)

        assert sorted(res["selected"]) == nearest_k_cluster(d, seed_idx, 4)


    def test_ignores_a_far_away_perfect_partner(self):
        # Check that the perfect complement is ignored for the much closer building
        sc = scenario_perfect_cancellation()
        res = solve_scenario(sc, 0.0, 0.0, 1.0)

        assert sorted(res['selected']) == [0, 2]
        assert res["total_distance"] == approx_total(100.0, sc.distances)
        assert res["peak"] > 1.0


# ---- distance patterns ----
    def test_optimal_peak_is_non_decreasing_in_k_min(self):
        # Check that raising k_min only shrinks the feasible set, so the best peak can never improve
        loads = random_loads(7, T=24, seed=11)
        distances = [i for i in range(7)]
        total_distances = [
            solve_cluster(loads, distances, seed_idx=0, k_min=k, weight_peak=0.0, weight_net=0.0, weight_distance=1.0)['total_distance']
            for k in range(1, 8)
        ]

        for earlier, later in zip(total_distances, total_distances[1:]):
            assert later >= earlier - 1e3