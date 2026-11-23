import pytest

from nlr_ghe_screen.clustering.milp import solve_cluster

# ---------------------------------------------------------------------------
# Test - Input Validation
# ---------------------------------------------------------------------------

# ------------- Fixture Functions -------------

@pytest.fixture
def small_problem():
    """5 buildings x 4 timesteps. Heating +, cooling -."""
    loads = [
        [5.0, 4.0, 3.0, 2.0],
        [-4.0, -3.0, -2.0, -1.0],
        [2.0, -2.0, 2.0, -2.0],
        [1.0, 1.0, 1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0],
    ]
    distances = [0.0, 100.0, 200.0, 300.0, 400.0]
    return loads, distances

def assert_valid_solution(result, seed_idx, k_min, n):
    assert result["selected"] is not None
    sel = result["selected"]
    assert seed_idx in sel
    assert len(sel) >= k_min
    assert len(set(sel)) == len(sel)
    assert all(0 <= i < n for i in sel)

# ----------------------------------------------

class TestMILPInputs:
    """
    Test cases for the handling of the inputs to the MILP solver
    """

# ---- General Case ----

    def test_valid_arguments(self, small_problem):
        loads, dist = small_problem
        res = solve_cluster(loads, dist, seed_idx=0)
        assert_valid_solution(res, 0, 3, 5)

# ---- Seed Index ----

    def test_seed_index_out_of_range(self, small_problem):
        loads, dist = small_problem

        with pytest.raises(ValueError):
            res = solve_cluster(loads, dist, seed_idx=20)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_seed_always_in_selection(self, small_problem, seed):
        loads, dist = small_problem
        res = solve_cluster(loads, dist, seed_idx=seed, k_min=3)
        assert_valid_solution(res, seed, 3, 5)

# ---- k_min ----

    @pytest.mark.parametrize("k_min", [1, 2, 3, 4, 5])
    def test_selection_size_respects_k_min(self, small_problem, k_min):
        loads, dist = small_problem
        res = solve_cluster(loads, dist, seed_idx=0, k_min=k_min)
        assert_valid_solution(res, 0, k_min, 5) 

    def test_too_large_k_min(self, small_problem):
        loads, dist = small_problem

        with pytest.raises(ValueError):
            res = solve_cluster(loads, dist, seed_idx=0, k_min=20)

# ---- Shape / Dimension Handling

    def test_ragged_loads_raise(self):
        with pytest.raises(ValueError):
            solve_cluster([[1.0, 2.0, 3.0], [1.0, 2.0]], [0.0, 1.0], seed_idx=0, k_min=1)
    
    def test_empty_loads_raise(self):
        with pytest.raises(ValueError):
            solve_cluster([], [], seed_idx=0, k_min=1)
    
    def test_distances_too_short_raise(self, small_problem):
        loads, dist = small_problem
        with pytest.raises((ValueError, IndexError)):
            solve_cluster(loads, dist[:-1], seed_idx=0)