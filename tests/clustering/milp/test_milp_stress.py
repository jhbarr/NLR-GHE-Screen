import itertools
import numpy as np
import pytest
import time
  
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

NUM_PROFILES = [5, 50, 500, 999]

by_name = lambda s: s.name


# ---------------------------------------------------------------------------
# Test - MILP Performance
# ---------------------------------------------------------------------------

class TestMILPStress:
    """
    Test cases to analyze the performance of the MILP with significant data inputs
    to determine its computational limits
    """

    # @pytest.mark.parametrize('num', NUM_PROFILES)
    # @pytest.mark.timeout(5)
    # def test_cluster_size_limits(self, num):
    #     # Analyze how many building load profiles can be in a cluster before it 
    #     # begins to degrade 
    #     sc = scenario_random(n=num, k_min=2 + (num % 3))
    #     res = solve_scenario(sc, 0.5, 0.5, 0.0)



    # def test_load_profile_time_resolution_limits(self):
    #     # Analyze the time seires resolution at which the MILP performance 
    #     # begins to degrade
    #     assert False