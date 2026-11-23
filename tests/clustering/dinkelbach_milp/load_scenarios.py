import numpy as np
from typing import NamedTuple, Optional

from nlr_ghe_screen.clustering.dinkelbach_milp import solve_dinkelbach_milp

# ---------------------------------------------------------------------
# Load Profile Class
# ---------------------------------------------------------------------

class Scenario(NamedTuple):
    name: str
    loads: np.ndarray               # (n, T), heating +, cooling -
    distances: np.ndarray           # (n,), metres from the GHX
    areas: np.ndarray
    seed_idx: int
    k_min: int
    expected_peak: Optional[float] = None 
    expected_imbalance: Optional[float] = None
    expected_cost_ratio: Optional[float] = None

    def solver_kwargs(self) -> dict:
        """Keyword arguments for solve_cluster; add weights on top, e.g.
        solve_cluster(**sc.solver_kwargs(), weight_peak=1, weight_net=0, weight_distance=0)"""
        return dict(loads=self.loads, distances=self.distances, areas=self.areas,
                    seed_idx=self.seed_idx, k_min=self.k_min)

def solve_scenario(sc, weight_peak, weight_net, distance_cap, **overrides):
    """Peak-only solve of a Scenario using its own loads/distances/seed/k_min.
    Any of loads, distances, seed_idx, k_min can be overridden."""

    args = {**sc.solver_kwargs(), **overrides}
    return solve_dinkelbach_milp(
        **args,
        weight_peak=weight_peak,
        weight_net=weight_net,
        distance_cap=distance_cap
    )

# ---------------------------------------------------------------------
# Single-building profiles
# ---------------------------------------------------------------------

def constant_profile(value: float, T: int = 24) -> np.ndarray:
    return np.full(T, float(value))


def sinusoid_profile(T: int = 24, amplitude: float = 1.0, phase: float = 0.0,
                     offset: float = 0.0) -> np.ndarray:
    """offset + amplitude * sin(2*pi*(t - phase)/T). Phase is in timesteps."""
    t = np.arange(T)
    return offset + amplitude * np.sin(2 * np.pi * (t - phase) / T)


def step_profile(T: int = 24, start: int = 0, stop: Optional[int] = None,
                 value: float = 1.0) -> np.ndarray:
    """`value` between [start, stop), zero elsewhere."""
    p = np.zeros(T)
    p[start:stop] = value
    return p


def spike_profile(T: int = 24, at: int = 0, value: float = 1.0) -> np.ndarray:
    """Zero everywhere except a single timestep."""
    p = np.zeros(T)
    p[at] = value
    return p


def random_building(T: int, rng: np.random.Generator, heating: bool = True,
                    scale: float = 10.0, noise: float = 0.05) -> np.ndarray:
    """A smooth, mostly single-signed daily-shaped profile with random amplitude/phase."""
    
    amp = scale * rng.uniform(0.5, 1.5)
    phase = rng.uniform(0, T)
    shape = 0.6 + 0.4 * np.sin(2 * np.pi * (np.arange(T) - phase) / T)  # in [0.2, 1.0]
    profile = amp * shape + rng.normal(0.0, noise * amp, size=T)
    return profile if heating else -profile


def random_loads(n: int, T: int = 24, seed: int = 0, heating_fraction: float = 0.5,
                 scale: float = 10.0, noise: float = 0.05) -> np.ndarray:
    """(n, T) array of random buildings; each is heating-dominated with prob. heating_fraction."""

    rng = np.random.default_rng(seed)
    heating = rng.random(n) < heating_fraction
    
    return np.vstack([random_building(T, rng, bool(h), scale, noise) for h in heating])


# ---------------------------------------------------------------------
# Distance profiles
# ---------------------------------------------------------------------

def uniform_distances(n: int, value: float = 500.0) -> np.ndarray:
    """Every building the same distance away (distance term can't discriminate)."""
    return np.full(n, float(value))
 
 
def linear_distances(n: int, seed_idx: int = 0, spacing: float = 100.0) -> np.ndarray:
    """Buildings in a line along one street, `spacing` m apart, seed at distance 0.
    distance_i = |i - seed_idx| * spacing."""
    return np.abs(np.arange(n) - seed_idx) * float(spacing)
 
 
def random_distances(n: int, seed: int = 0, low: float = 50.0, high: float = 2000.0,
                     seed_idx: Optional[int] = None) -> np.ndarray:
    """Uniform random distances in [low, high]; the seed (if given) is set to 0."""
    d = np.random.default_rng(seed).uniform(low, high, size=n)
    if seed_idx is not None:
        d[seed_idx] = 0.0
    return d


# ---------------------------------------------------------------------
# Building Scenarios - General 
# ---------------------------------------------------------------------

def scenario_perfect_cancellation(T: int = 24, n_decoys: int = 4, seed: int = 0, 
                                  complement_distance: float = 2000.0) -> Scenario:
    """Seed + its exact negative gives a zero net curve. Decoys are large heating loads.
    With k_min=2 the optimum is {seed, complement} with peak 0."""
    
    rng = np.random.default_rng(seed)
    seed_profile = sinusoid_profile(T, amplitude=3.0, offset=8.0)
    decoys = [random_building(T, rng, heating=True, scale=15.0) for _ in range(n_decoys)]
    loads = np.vstack([seed_profile, -seed_profile, *decoys])

    # distances = uniform_distances(len(decoys) + 2, seed)
    distances = np.concatenate([[0.0, complement_distance], 100.0 * (1 + np.arange(n_decoys))])
    areas = np.ones(len(distances))

    return Scenario("perfect_cancellation", loads, distances, areas, seed_idx=0, k_min=2, expected_peak=0.0, expected_imbalance=0.0)


def scenario_all_heating(n: int = 6, T: int = 12, seed_idx: int = 2, k_min: int = 3) -> Scenario:
    """Building i has constant load (i+1). Nothing cancels, so the best cluster is the seed
    plus the (k_min-1) smallest other buildings."""

    vals = np.arange(1, n + 1, dtype=float)
    loads = np.vstack([constant_profile(v, T) for v in vals])
    others = np.sort(np.delete(vals, seed_idx))
    distances = uniform_distances(n, seed_idx)
    areas = np.ones(len(distances))

    expected_peak = float(vals[seed_idx] + others[: k_min - 1].sum())
    expected_imbalance = expected_peak * T

    return Scenario("all_heating", loads, distances, areas, seed_idx, k_min, expected_peak, expected_imbalance)


def scenario_all_cooling(n: int = 6, T: int = 12, seed_idx: int = 2, k_min: int = 3) -> Scenario:
    """Mirror of all_heating (loads negated). Peak is about |N(t)|, so same expected peak."""

    s = scenario_all_heating(n, T, seed_idx, k_min)

    return Scenario("all_cooling", -s.loads, s.distances, s.areas, s.seed_idx, s.k_min, s.expected_peak, s.expected_imbalance)


def scenario_phase_shifted_sinusoids(T: int = 24, amplitude: float = 5.0) -> Scenario:
    """Four zero-mean sinusoids with phases 0, T/2, T/4, 3T/4. Seed (phase 0) cancels exactly
    with the T/2 building, so the k_min=2 optimum has peak ~0."""

    phases = [0, T / 2, T / 4]
    loads = np.vstack([sinusoid_profile(T, amplitude, phase=p) for p in phases])
    distances = uniform_distances(3, 0) 
    areas = np.ones(len(distances))

    return Scenario("phase_shifted_sinusoids", loads, distances, areas, seed_idx=0, k_min=2, expected_peak=0.0, expected_imbalance=0.0)


def scenario_staggered_spikes(T: int = 8) -> Scenario:
    """Spikes of 10. Seed and building 1 coincide (adding both -> 20). Buildings 2-4 spike at
    different times. With k_min=4 the peak-optimal cluster is {0, 2, 3, 4} with peak 10.
    The coincident building is the *closest* one (10 m), so distance would tempt you to take it."""
    loads = np.vstack([
        spike_profile(T, at=0, value=10.0),  # seed
        spike_profile(T, at=0, value=10.0),  # coincident decoy
        spike_profile(T, at=2, value=10.0),
        spike_profile(T, at=4, value=10.0),
        spike_profile(T, at=6, value=10.0),
    ])
    distances = [0.0, 10.0, 200.0, 300.0, 400.0]
    areas = np.ones(len(distances))

    return Scenario("staggered_spikes", loads, distances, areas, 
                         seed_idx=0, k_min=4, expected_peak=10.0, expected_imbalance=40.0)


def scenario_random(n: int = 8, T: int = 24, seed: int = 0, seed_idx: int = 0,
                    k_min: int = 3, heating_fraction: float = 0.5) -> Scenario:
    """Random mixed heating/cooling buildings at random distances (seed at 0 m).
    No analytic answer (expected_peak=None)."""

    loads = random_loads(n, T, seed, heating_fraction)
    distances = random_distances(n, seed + 1000, seed_idx=seed_idx)
    areas = np.ones(len(distances))

    return Scenario(f"random_s{seed}", loads, distances, areas, seed_idx, k_min, None)



# ---------------------------------------------------------------------
# Building Scenarios - Imbalance Aimed 
# ---------------------------------------------------------------------

def scenario_subset_sum(T: int = 4) -> Scenario:
    """Seed +10 and cooling buildings -3, -4, -5 (constant profiles, k_min=1).
    Annual-energy sums over the options: {s}:10, {s,-3}:7, {s,-4}:6, {s,-5}:5, {s,-3,-4}:3,
    {s,-3,-5}:2, {s,-4,-5}:1, all:-2 (times T). Unique net optimum is {0,2,3} with |sum| = 1*T."""
    vals = [10.0, -3.0, -4.0, -5.0]
    loads = np.vstack([constant_profile(v, T) for v in vals])
    areas = np.ones(len(loads))

    return Scenario("subset_sum", loads, linear_distances(4, 0), areas,
                         seed_idx=0, k_min=1, expected_peak=1.0, expected_imbalance=1.0 * T)


def scenario_zero_energy_seed(T: int = 4) -> Scenario:
    """Seed alternates +1/-1 (annual energy exactly 0); the others are heating buildings with
    constant loads 2, 3, 4. With k_min=1 the seed alone is the unique optimum for both
    imbalance (0) and peak (1)."""
    seed_profile = np.tile([1.0, -1.0], T // 2)
    loads = np.vstack([seed_profile, *[constant_profile(v, T) for v in (2.0, 3.0, 4.0)]])
    areas = np.ones(len(loads))
    
    return Scenario("zero_energy_seed", loads, linear_distances(4, 0), areas, 
                         seed_idx=0, k_min=1, expected_peak=1.0, expected_imbalance=0.0)



# ---------------------------------------------------------------------
# Building Scenarios - Scale Preferences
# ---------------------------------------------------------------------

def _stack(*rows):
    return np.vstack(rows)

def scenario_large_beats_small(T=8):
    """Peak mode. B1 has the lower raw peak but is small; B2 has the higher raw peak but is much larger.
    {0,1}: peak 3,  area 11 -> 0.2727
    {0,2}: peak 6,  area 50 -> 0.1200   <- optimum
    {0,1,2}: peak 7, area 51 -> 0.1373
    Raw-peak minimization picks {0,1}."""
    loads = _stack(constant_profile(2.0, T), constant_profile(1.0, T), constant_profile(4.0, T))
    
    return Scenario("large_beats_small", loads=loads, distances=np.ones(len(loads)), areas=np.array([10., 1., 40.]), 
                    seed_idx=0, k_min=2, expected_cost_ratio=6/50)


def scenario_intensity_not_size(T=8):
    """Peak mode. Guards against 'just maximize area'. B2 is bigger but its load per sq ft is far higher.
    {0,1}: peak 1.2,  area 12 -> 0.1000   <- optimum
    {0,2}: peak 11,   area 30 -> 0.3667
    {0,1,2}: peak 11.2, area 32 -> 0.3500
    An area-greedy model would pick {0,2}."""
    loads = _stack(constant_profile(1.0, T), constant_profile(0.2, T), constant_profile(10.0, T))
    
    return Scenario("intensity_not_size", loads=loads, distances=np.ones(len(loads)), areas=np.array([10., 2., 20.]),
                     seed_idx=0, k_min=2, expected_cost_ratio=1.2/12)


def scenario_imbalance_scale(T=8):
    """Imbalance mode. Energies: seed +8, B1 -10 (small), B2 -4 (large).
    {0,1}: |net| 2, area 5  -> 0.400   (lowest RAW imbalance)
    {0,2}: |net| 4, area 24 -> 0.1667  <- optimum
    {0,1,2}: |net| 6, area 25 -> 0.240
    Raw imbalance minimization picks {0,1}."""
    loads = _stack(constant_profile(1.0, T), constant_profile(-1.25, T), constant_profile(-0.5, T))
    
    return Scenario("imbalance_scale", loads=loads, distances=np.ones(len(loads)), areas=np.array([4., 1., 20.]),
                     seed_idx=0, k_min=2, expected_cost_ratio=4/24)


def scenario_raw_peak_tie_broken_by_area(T=8):
    """Peak mode. {0,1} and {0,2} both have raw peak 2, so only area sepdarates them.
    {0,1}: peak 2, area 4  -> 0.5
    {0,2}: peak 2, area 12 -> 0.1667  <- optimum (B2 is spiky but large)
    {0,1,2}: peak 3, area 14 -> 0.2143"""
    loads = _stack(constant_profile(1.0, T), constant_profile(1.0, T), spike_profile(T, at=0, value=1.0))
    
    return Scenario("peak_tie_area_break", loads=loads, distances=np.ones(len(loads)), areas=np.array([2., 2., 10.]),
                     seed_idx=0, k_min=2, expected_cost_ratio=2/12)
