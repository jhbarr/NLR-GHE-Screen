import numpy as np
import cplex

# ---------------------------------------------------------------------------
# Base MILP 
# ---------------------------------------------------------------------------

def solve_dinkelbach_milp(
        # lam: float,
        loads: list[list[float]],
        areas: list[float],
        distances: list[float],
        distance_cap: float,
        seed_idx: int,
        k_min: int = 3,
        weight_peak: float = 0.5,
        weight_net: float = 0.5,
):
    """Generates a cluster of buildings based on a MILP optimization function

    This function takes in load profiles of various buildings and their respective distances to a GHE or waste heat source location,
    which acts as the seed for the cluster. The MILP attempts to minimize both the total summed net load of all of the buildings in
    the cluster as well as the maximum peak height of that net load. 

    The function returns which buildings should be included in the cluster in order to create the minimum net load profile curve.

    Args:
        lam (float): The dinkelback ratio value
        loads (list[list[float]]): A matrix - (n_buildings, n_timesteps) array, heating +, cooling -
        areas (list[float]): A list of how large each building is in square footage
        distances (list[float]): (n_buildings,) road distance from the GHX, meters
        k_min (int): The minimum number of buildings that should be in the cluster - including the seed 
        weight_peak (float): The relative weight that the peak component is given in the minimization variable
        weight_net (float): The relative weight that the total net load component is given
        weight_distance (float): The relative weight that the distance metric is given
    
    Returns:
        dict: A dictionary containing the results of the MILP solution
    """
    # Check that the inputs are valid
    if seed_idx > len(loads):
        raise ValueError('seed_idx too large')
    elif k_min > len(loads):
        raise ValueError('k_min too large')

    # --------- Set up the variables to be used in the MILP ---------
    L = np.asarray(loads, dtype=float) # Load profiles of each building
    d = np.asarray(distances, dtype=float) # Distances from each building to centroid
    area = np.asarray(areas, dtype=float) # Squre footage per building
    n, T = L.shape
    A = L.sum(axis=1) # Annual net energy per building

    # --------- Variable Layout ---------
    z_idx, q_idx = n, n + 1
    x_idx = list(range(n))

    # Force the building values to be in {0, 1}
        # ** Setting these not to floats may impact something down the line
    x_lb = [0] * n  # Lower bound for all buildings - 0
    x_ub = [1] * n  # Upper bound for all buildings - 1
    x_lb[seed_idx] = 1 # Force include the seed building in the cluster
    

    # --------- Problem Instantiation ---------
    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.minimize)
    problem.set_problem_name("ghe_cluster")
    problem.set_results_stream(None)

    problem.variables.add(
        obj=[0.0] * (n + 2), # Place holder coefficient values 
        lb=x_lb + [0.0, 0.0],
        ub=x_ub + [cplex.infinity, cplex.infinity],
        types=['B'] * n + ['C', 'C'],
        names=[f'x_{i}' for i in range(n)] + ['peak_z', 'imbalance_q']
    )


    # --------- Problem Constraint Definition ---------
    rows, senses, rhs, names = [], [], [], []

    # --- Peak   z -/+ N(t) >= 0 ---
    for t in range(T):
        col = L[:, t].tolist()  # Get all values from each timeseries at time t

        # z - sum_i (x_i * load_i(t)) >= 0  ->  z >= N(t)
        rows.append(cplex.SparsePair(
            ind=x_idx + [z_idx],
            val=[-v for v in col] + [1.0]
        ))
        senses.append("G"); rhs.append(0.0); names.append(f"peak_pos_{t}")

        # z + sum_i x_i L_i(t) >= 0     (z >= -N(t))
        rows.append(cplex.SparsePair(
            ind=x_idx + [z_idx],
            val=col + [1.0]
        ))
        senses.append("G"); rhs.append(0.0); names.append(f"peak_neg_{t}")


    # --- Imbalance   q -/+ sum_i x_i A_i >= 0 ---
    a = A.tolist()
    rows.append(cplex.SparsePair(
        ind=x_idx + [q_idx],
        val=[-v for v in a] + [1.0]
    ))
    senses.append("G"); rhs.append(0.0); names.append("imb_pos")

    rows.append(cplex.SparsePair(
        ind=x_idx + [q_idx],
        val=a + [1.0]
    ))
    senses.append("G"); rhs.append(0.0); names.append("imb_neg")


    # --- Cluster Size ---
    ones = [1.0] * n
    rows.append(cplex.SparsePair(ind=x_idx, val=ones))
    senses.append("G"); rhs.append(float(k_min)); names.append("size_min")


    # ---- Maximum distance cap ----
    rows.append(cplex.SparsePair(
        ind=x_idx,
        val=list(d)
    ))
    senses.append("L"); rhs.append(float(distance_cap)), names.append("dist_cap")


    # --------- Calculate the lambda ratio of a cluster ---------   
    def lambda_ratio(x):
        """
        Calculate the lambda ratio of a particular cluster
        """
        selected = np.asarray(x) > 0.5 # Mask - loads that are in the cluster
        net_profile = L[selected].sum(axis=0)
        
        numerator = (
            # weight_distance * d[selected].sum()
            weight_peak * float(np.abs(net_profile).max())
            + weight_net * float(abs(net_profile.sum())) # ** MIGHT CAUSE PROBLEMS **
        )  
        denominator = area[selected].sum()

        return numerator / denominator


    # --------- Dinkelbach loop ---------
    problem.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs, names=names)
    problem.parameters.mip.tolerances.mipgap.set(1e-9)
    problem.parameters.mip.tolerances.absmipgap.set(1e-9)
    
    x = np.concatenate([np.ones(k_min), np.zeros(n - k_min)]) # All buildings should be considered in the initial cluster group 
    lam = lambda_ratio(x) # Get the initial lambda ratio value

    MAX_ITER = 10
    TOL = 1e-10
    for _ in range(MAX_ITER):
        # coeffs = [(i, float(weight_distance * d[i] - lam * area[i])) for i in range(n)]
        coeffs = [(i, float(-lam * area[i])) for i in range(n)]
        coeffs += [(z_idx, float(weight_peak)), (q_idx, float(weight_net))]
        problem.objective.set_linear(coeffs)

        problem.solve()
        vals = problem.solution.get_values()
        x = np.round(vals[:n])
        F = problem.solution.get_objective_value()

        if 0 < F < TOL:
            break
        lam = lambda_ratio(x)

    selected = [i for i in range(n) if x[i] > 0.5]
    NET_CURVE = L[selected].sum(axis=0)
    AREA_NORM = area[selected].sum()
    NORM_CURVE = NET_CURVE / AREA_NORM
    return {
        "total variables": problem.variables.get_num(),
        "total constraints": problem.linear_constraints.get_num(),
        "objective": problem.solution.get_objective_value(),
        "best_bound": problem.solution.MIP.get_best_objective(),
        "cost_ratio": lambda_ratio(x),
        "selected": selected,
        "norm_curve": NORM_CURVE,
        "norm_peak": float(np.abs(NET_CURVE).max() / AREA_NORM),
        "norm_imbalance": float(abs(NET_CURVE.sum()) / AREA_NORM),
        "total_distance": float(d[selected].sum()),
    }