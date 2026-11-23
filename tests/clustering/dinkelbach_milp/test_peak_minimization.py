import pytest
import numpy as np
import itertools
import cplex

from nlr_ghe_screen.clustering.dinkelbach_milp import solve_dinkelbach_milp
from load_scenarios import (
    solve_scenario,
    constant_profile,
    random_loads,
    scenario_perfect_cancellation,
    scenario_all_cooling,
    scenario_all_heating,
    scenario_phase_shifted_sinusoids,
    scenario_staggered_spikes,
    scenario_random
)

# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

TOL = dict(rel=1e-3, abs=1e-3)

DETERMINISTIC = [
    scenario_perfect_cancellation(),
    scenario_all_heating(),
    scenario_all_cooling(),
    scenario_phase_shifted_sinusoids(),
    scenario_staggered_spikes(),
]

RANDOM = [
    scenario_random(seed=s, seed_idx=s, k_min=2 + (s % 3), heating_fraction=hf)
    for s, hf in zip(range(6), [0.5, 0.5, 0.3, 0.7, 0.5, 0.5])
]

ALL_SCENARIOS = DETERMINISTIC + RANDOM
KNOWN = [s for s in ALL_SCENARIOS if s.expected_peak is not None]

by_name = lambda s: s.name

# ---------------------------------------------------------------------------
# Test - Peak Reduction
# ---------------------------------------------------------------------------

# ------------- Helper Functions ------------- 
def brute_force_min_peak_ratio(loads, areas, seed_idx, k_min,
                               distances=None, D_max=None):
    """Enumerate every subset containing the seed with size >= k_min and
    return the one minimizing peak / total_area (peak = max |net load|)."""

    L = np.asarray(loads, dtype=float)
    area = np.asarray(areas, dtype=float)
    d = None if distances is None else np.asarray(distances, dtype=float)
    others = [i for i in range(len(L)) if i != seed_idx]
    best_ratio, best_peak, best_set = np.inf, np.inf, None

    for r in range(max(k_min - 1, 0), len(others) + 1):
        for combo in itertools.combinations(others, r):
            sel = [seed_idx, *combo]

            # Skip clusters that violate the distance cap (if one is used)
            if D_max is not None and d[sel].sum() > D_max:
                continue

            peak = float(np.abs(L[sel].sum(axis=0)).max())
            ratio = peak / area[sel].sum()

            if ratio < best_ratio:
                best_ratio, best_peak, best_set = ratio, peak / area[sel].sum(), sorted(sel)

    return best_ratio, best_peak, best_set

# -----------------------------------------------

class TestPeakReduction:
    """
    Tests to analyze the performance of the MILP at strictly reducing the maximum
    net load curve peak when distance and imbalance are non-factors

    Hold the distance and imbalance weights to zero
    """

# ---- core correctness ----
    # 1. 
    @pytest.mark.parametrize('sc', ALL_SCENARIOS, ids=by_name)
    def test_peak_matches_brute_force(self, sc):
        # Check that for all scenarios, the MILP peak matches the brute force peak
        res = solve_scenario(sc, weight_peak=1, weight_net=0.0, distance_cap=cplex.infinity)
        expected_ratio, expected_peak, expected_set = brute_force_min_peak_ratio(sc.loads, sc.areas, seed_idx=sc.seed_idx, k_min=sc.k_min, distances=sc.distances)

        assert res['cost_ratio'] == pytest.approx(expected_ratio, **TOL)
        assert sorted(res["selected"]) == sorted(expected_set)
        assert res["norm_peak"] == pytest.approx(expected_peak, **TOL)


    # 2.
    @pytest.mark.parametrize('sc', ALL_SCENARIOS, ids=by_name)
    def test_selection_is_valid(self, sc):
        # Check that for all scenarios, all of the constraints are satisfied
        res = solve_scenario(sc, weight_peak=1, weight_net=0.0, distance_cap=cplex.infinity)
        sel = res['selected']

        assert sc.seed_idx in sel
        assert len(sel) >= sc.k_min
        assert len(set(sel)) == len(sel)

# ---- known optima ----
    # 3. 
    def test_perfect_cancellation(self):
        # Check that two pefectly opposite loads are selected out of a batch of randoms
        sc = scenario_perfect_cancellation()
        res = solve_scenario(sc, weight_peak=1, weight_net=0.0, distance_cap=cplex.infinity)

        assert sorted(res['selected']) == [0, 1] 
        assert res['norm_peak'] == pytest.approx(0.0, abs=1e-6)


    # 4. 
    def test_all_heating_selects_smallest(self):
        # Check that the smallest load profile is chosen amongst increasing load profiles
        sc = scenario_all_heating(n=6, seed_idx=2, k_min=3)
        res = solve_scenario(sc, weight_peak=1, weight_net=0.0, distance_cap=cplex.infinity)

        assert sorted(res['selected']) == [0, 1, 2]

    # 5. 
    def test_staggered_spikes_avoids_coincident_building(self):
        # Check that a building that increases the peak is not selected
        sc = scenario_staggered_spikes()
        res = solve_scenario(sc, weight_peak=1, weight_net=0.0, distance_cap=cplex.infinity)

        assert 1 not in res['selected']
        assert sorted(res['selected']) == [0, 2, 3, 4]


    # 6. 
    def test_peak_counts_cooling_as_well_as_heating(self):
        # Check that the MILP is minimizing the absolute value of the peak
        loads = np.vstack([constant_profile(5.0, 6), constant_profile(-4.0, 6), constant_profile(-30.0, 6)])
        res = solve_dinkelbach_milp(loads, distances=[0] * 3, areas=np.ones(3), seed_idx=0, k_min=2, weight_peak=1, weight_net=0.0, distance_cap=cplex.infinity)

        assert sorted(res["selected"]) == [0, 1]
        assert res["norm_peak"] == pytest.approx(1.0 / 2, **TOL)


# ---- invariance / symmetry ----
    # 7. 
    @pytest.mark.parametrize('sc', RANDOM, ids=by_name)
    def test_sign_flip_invariance(self, sc):
        # Check that flipping the sign (+ or -) does not affect the overall peak
        expected_ratio, expected_peak, expected_set = brute_force_min_peak_ratio(sc.loads, sc.areas, seed_idx=sc.seed_idx, k_min=sc.k_min, distances=sc.distances)
        res = solve_scenario(sc, weight_peak=1, weight_net=0.0, distance_cap=cplex.infinity, loads=-sc.loads)

        assert res['cost_ratio'] == pytest.approx(expected_ratio, **TOL)
        assert res['norm_peak'] == pytest.approx(expected_peak, **TOL)
        assert sorted(res["selected"]) == sorted(expected_set)


    # 8. 
    @pytest.mark.parametrize('sc', RANDOM, ids=by_name)
    def test_building_order_invariance(self, sc):
        # Check that shuffling the ordering of the building loads does not affect overall peak
        _, expected_peak, expected_set = brute_force_min_peak_ratio(sc.loads, sc.areas, seed_idx=sc.seed_idx, k_min=sc.k_min, distances=sc.distances)
        perm = np.random.default_rng(1).permutation(len(sc.loads))

        new_seed = int(np.where(perm == sc.seed_idx)[0][0])
        res = solve_scenario(sc, weight_peak=1, weight_net=0.0, distance_cap=cplex.infinity, loads=sc.loads[perm], distances=sc.distances[perm], seed_idx=new_seed)
        
        assert res["norm_peak"] == pytest.approx(expected_peak, **TOL)


# ---- k_min behavior ----
    # 9. 
    def test_optimal_peak_is_non_decreasing_in_k_min(self):
        # Check that raising k_min only shrinks the feasible set, so the best peak can never improve
        loads = random_loads(7, T=24, seed=11)
        distances = [0] * 7
        areas = np.ones(7)
        peaks = [
            solve_dinkelbach_milp(loads, areas, distances, seed_idx=0, k_min=k, weight_peak=1, weight_net=0.0, distance_cap=cplex.infinity)['norm_peak']
            for k in range(1, 8)
        ]

        for earlier, later in zip(peaks, peaks[1:]):
            assert later >= earlier - 1e3

    # 10. 
    def test_k_min_equals_n(self):
        # Check that if k_min equals the total number of loads, the max peak is the sum of all the loads
        loads = random_loads(6, T=24, seed=5)
        distances = [0] * 6
        areas = np.ones(6)
        res = solve_dinkelbach_milp(loads, areas, distances, seed_idx=0, k_min=6, weight_peak=1, weight_net=0.0, distance_cap=cplex.infinity)

        assert sorted(res['selected']) == list(range(6))
        assert res["norm_peak"] == pytest.approx(np.abs(loads.sum(axis=0)).max() / 6, **TOL)
