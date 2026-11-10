import numpy as np
import cplex


def solve_cluster(
        loads: list[list[float]],
        distances: list[float],
        seed_idx: int,
        k_min: int = 3,
        weight_peak: float = 0.5,
        weight_net: float = 0.5,
        weight_distance: float = 0.1,
):
    """Generates a cluster of buildings based on a MILP optimization function

    This function takes in load profiles of various buildings and their respective distances to a GHE or waste heat source location,
    which acts as the seed for the cluster. The MILP attempts to minimize both the total summed net load of all of the buildings in
    the cluster as well as the maximum peak height of that net load. 

    The function returns which buildings should be included in the cluster in order to create the minimum net load profile curve.

    Args:
        loads (list[list[float]]): A matrix - (n_buildings, n_timesteps) array, heating +, cooling -
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
    L = np.asarray(loads, dtype=float)
    d = np.asarray(distances, dtype=float)
    n, T = L.shape


    # --------- Constants ---------
    A = L.sum(axis=1)   # Annual net energy per building 

    # Normalizers 
    Z_ref = max(np.abs(L).sum(axis=0).max(), 1e-9)
    Q_ref = max(np.abs(A).sum(), 1e-9)
    D_ref = max(d.sum(), 1e-9)


    # --------- Variable Layout ---------
    z_idx, q_idx = n, n + 1
    x_idx = list(range(n))

    # Force the building values to be in {0, 1}
        # ** Setting these not to floats may impact something down the line
    x_lb = [0] * n  # Lower bound for all buildings - 0
    x_ub = [1] * n  # Upper bound for all buildings - 1
    x_lb[seed_idx] = 1 # Force include the seed building in the cluster

    # Create the objective - to be minimized
    obj = [weight_distance * (d[i] / D_ref) for i in range(n)] + [weight_peak / Z_ref, weight_net / Q_ref]


    # --------- Problem Instantiation ---------
    problem = cplex.Cplex()
    # problem.set_problem_type(cplex.Cplex.problem_type.LP)
    problem.objective.set_sense(problem.objective.sense.minimize)
    problem.set_problem_name("ghe_cluster")
    problem.set_results_stream(None)

    problem.variables.add(
        obj=obj,
        lb=x_lb + [0.0, 0.0],
        ub=x_ub + [cplex.infinity, cplex.infinity],
        types=['B'] * n + ['C', 'C'],    # Binary for the buildings, continuous for the objective variables
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


    # --------- Problem Solver Settings ---------
    problem.linear_constraints.add(lin_expr=rows, senses=senses, rhs=rhs, names=names)
    problem.solve()

    status = problem.solution.get_status_string()
    if not problem.solution.is_primal_feasible():
        return {"status": status, "selected": None}
 
    vals = np.array(problem.solution.get_values())
    selected = [i for i in range(n) if vals[i] > 0.5]
 
    # recompute metrics directly from the selection (independent check)
    NET = L[selected].sum(axis=0)
    return {
        "total variables": problem.variables.get_num(),
        "total constraints": problem.linear_constraints.get_num(),
        "status": status,
        "objective": problem.solution.get_objective_value(),
        "best_bound": problem.solution.MIP.get_best_objective(),
        "selected": selected,
        "net_curve": NET,
        "peak": float(np.abs(NET).max()),
        "imbalance": float(abs(NET.sum())),
        "total_distance": float(d[selected].sum()),
    }
