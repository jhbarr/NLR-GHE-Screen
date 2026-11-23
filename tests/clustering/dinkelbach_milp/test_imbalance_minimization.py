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
    scenario_random,
    scenario_subset_sum,
    scenario_zero_energy_seed
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
    scenario_subset_sum(),
    scenario_zero_energy_seed()
]

RANDOM = [
    scenario_random(seed=s, seed_idx=s, k_min=2 + (s % 3), heating_fraction=hf)
    for s, hf in zip(range(7), [0.5, 0.5, 0.3, 0.7, 0.5, 0.5, 0.4])
]

ALL_SCENARIOS = DETERMINISTIC + RANDOM
KNOWN = [s for s in ALL_SCENARIOS if s.expected_peak is not None]

by_name = lambda s: s.name


# ---------------------------------------------------------------------------
# Test - Net Imbalance Reduction
# ---------------------------------------------------------------------------

# ------------- Helper Functions ------------- 
def approx_imbalance(expected, loads):
    """Tolerance scaled to Q_ref = sum|A_i|, since the solver's objective is q / Q_ref and
    CPLEX's absolute MIP gap applies to that scaled value."""
    q_ref = max(np.abs(annual_energy(loads)).sum(), 1e-9)

    return pytest.approx(expected, rel=1e-3, abs=max(1e-4 * q_ref, 1e-6))

def annual_energy(loads):
    L = np.asarray(loads, dtype=float)
    return L.sum(axis=1)

def brute_force_min_imbalance_ratio(loads, areas, seed_idx, k_min,
                                    distances=None, D_max=None):
    """Enumerate every subset containing the seed with size >= k_min and
    return the one minimizing |net annual energy| / total_area."""

    A = annual_energy(loads)
    area = np.asarray(areas, dtype=float)
    d = None if distances is None else np.asarray(distances, dtype=float)
    others = [i for i in range(len(A)) if i != seed_idx]
    best_ratio, best_q, best_set = np.inf, np.inf, None

    for r in range(max(k_min - 1, 0), len(others) + 1):
        for combo in itertools.combinations(others, r):
            sel = [seed_idx, *combo]

            # Skip clusters that violate the distance cap (if one is used)
            if D_max is not None and d[sel].sum() > D_max:
                continue

            q = float(abs(A[sel].sum()))
            ratio = q / area[sel].sum()

            if ratio < best_ratio:
                best_ratio, best_q, best_set = ratio, q / area[sel].sum(), sorted(sel)

    return best_ratio, best_q, best_set

# -----------------------------------------------


class TestImbalanceMinimization:
    """
    Tests to analyze the performance of the MILP at strictly reducing the maximum
    net load when distance and peak are non-factors

    Hold the distance and peak weights to zero
    """

# ---- core correctness ----
    # 1. 
    @pytest.mark.parametrize('sc', ALL_SCENARIOS, ids=by_name)
    def test_peak_matches_brute_force(self, sc):
        # Check that for all scenarios, the MILP peak matches the brute force peak
        res = solve_scenario(sc, weight_peak=0.0, weight_net=1.0, distance_cap=cplex.infinity)
        expected_ratio, expected_imbalance, expected_set = brute_force_min_imbalance_ratio(sc.loads, sc.areas, seed_idx=sc.seed_idx, k_min=sc.k_min, distances=sc.distances)

        assert res['cost_ratio'] == pytest.approx(expected_ratio, **TOL)
        assert res["norm_imbalance"] == approx_imbalance(expected_imbalance, sc.loads)


    # # 2. 
    # @pytest.mark.parametrize("sc", KNOWN, ids=by_name)
    # def test_imbalance_matches_analytic_answer(self, sc):
    #     # Check that the imbalances match the known imbalances for the given scenarios
    #     res = solve_scenario(sc, weight_peak=0.0, weight_net=1, distance_cap=cplex.infinity)
    #     assert res["imbalance"] == approx_imbalance(sc.expected_imbalance, sc.loads)


    # 3. 
    @pytest.mark.parametrize('sc', ALL_SCENARIOS, ids=by_name)
    def test_selection_is_valid(self, sc):
        # Check that for all scenarios, all of the constraints are satisfied
        res = solve_scenario(sc, weight_peak=0.0, weight_net=1.0, distance_cap=cplex.infinity)
        sel = res['selected']

        assert sc.seed_idx in sel
        assert len(sel) >= sc.k_min
        assert len(set(sel)) == len(sel)


# ---- known optima ----
    # 4. 
    def test_subset_sum_picks_best_combination(self):
        T = 4
        sc = scenario_subset_sum(T=T)
        res = solve_scenario(sc, weight_peak=0.0, weight_net=1.0, distance_cap=cplex.infinity)

        for key, val in res.items():
            print(key, val)

        assert sorted(res['selected']) == [0, 2, 3]
        assert res['norm_imbalance'] == approx_imbalance((1.0 * T) / 3, sc.loads)

    # 5. 
    def test_zero_energy_seed_is_optimal(self):
        # Check that the optimal selection is simply the seed 
        sc = scenario_zero_energy_seed()
        res = solve_scenario(sc, weight_peak=0.0, weight_net=1.0, distance_cap=cplex.infinity)

        assert res['selected'] == [0]
        assert res['norm_imbalance'] == pytest.approx(0.0, abs=1e-6)


    # 6. 
    def test_all_heating_selects_smallest(self):
        # Check that the smallest load profile is chosen amongst increasing load profiles
        sc = scenario_all_heating(n=6, seed_idx=2, k_min=3)
        res = solve_scenario(sc, weight_peak=0.0, weight_net=1.0, distance_cap=cplex.infinity)

        assert sorted(res['selected']) == [0, 1, 2]


    # 7. 
    def test_perfect_cancellation_picks_the_complement(self):
        sc = scenario_perfect_cancellation()
        res = solve_scenario(sc, weight_peak=0.0, weight_net=1.0, distance_cap=cplex.infinity)

        assert sorted(res["selected"]) == [0, 1]
        assert res["norm_imbalance"] == pytest.approx(0.0, abs=1e-6)


    # 8. 
    def test_imbalance_uses_absolute_value(self):
        # Check that the MILP uses absolute value and doesn't try to find the absolute minimum
        T = 6
        loads = np.vstack([constant_profile(5.0, T), constant_profile(-4.0, T), constant_profile(-30.0, T)])
        areas = np.ones(len(loads))
        res = solve_dinkelbach_milp(loads, areas, distances=[0] * 3, distance_cap=cplex.infinity, seed_idx=0, k_min=2)

        assert sorted(res["selected"]) == [0, 1]
        assert res["norm_imbalance"] == approx_imbalance(6.0 / 2, loads)


# ---- invariance / symmetry ----
    # 9. 
    @pytest.mark.parametrize('sc', RANDOM, ids=by_name)
    def test_sign_flip_invariance(self, sc):
        # Check that having a positive or negative load does not affect the overall net load
        _, expected_imbalance, _ = brute_force_min_imbalance_ratio(sc.loads, sc.areas, seed_idx=sc.seed_idx, k_min=sc.k_min, distances=sc.distances)
        res = solve_scenario(sc, weight_peak=0.0, weight_net=1.0, distance_cap=cplex.infinity, loads=-sc.loads)

        assert res["norm_imbalance"] == approx_imbalance(expected_imbalance, sc.loads)


    # 10.
    @pytest.mark.parametrize("sc", RANDOM, ids=by_name)
    @pytest.mark.parametrize("factor", [0.01, 100.0])
    def test_scale_invariance(self, sc, factor):
        # Check that scaling all loads scales the optimal imbalance by the same factor (tests Q_ref)
        _, expected_imbalance, _ = brute_force_min_imbalance_ratio(sc.loads, sc.areas, seed_idx=sc.seed_idx, k_min=sc.k_min, distances=sc.distances)
        scaled = sc.loads * factor
        res = solve_scenario(sc, weight_peak=0.0, weight_net=1.0, distance_cap=cplex.infinity, loads=scaled)

        assert res["norm_imbalance"] == approx_imbalance(expected_imbalance * factor, scaled)


# ---- k_min behavior ----
    # 11. 
    def test_optimal_peak_is_non_decreasing_in_k_min(self):
        # Check that raising k_min only shrinks the feasible set, so the best peak can never improve
        loads = random_loads(7, T=24, seed=11)
        distances = [0] * 7
        areas = np.ones(len(distances))
        peaks = [
            solve_dinkelbach_milp(loads, areas, distances, seed_idx=0, k_min=k, weight_peak=0.0, weight_net=1.0, distance_cap=cplex.infinity)['norm_imbalance']
            for k in range(1, 8)
        ]

        for earlier, later in zip(peaks, peaks[1:]):
            assert later >= earlier - 1e3


    # 12. 
    def test_k_min_equals_n_imbalance_is_whole_set_total(self):
        # Check that all profiles are included when k_min is maxed out
        sc = scenario_random(n=6, seed=5, seed_idx=3)
        res = solve_scenario(sc, weight_peak=0.0, weight_net=1.0, distance_cap=cplex.infinity, k_min=6)
        expected = abs(annual_energy(sc.loads).sum())

        assert sorted(res["selected"]) == list(range(6))
        assert res["norm_imbalance"] == approx_imbalance(expected / 6, sc.loads)