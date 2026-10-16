import geopandas as gpd
import numpy as np

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from scipy.sparse import csr_matrix
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# Aggregate GHE / Waste heat sources as seeds for building clusters
# ---------------------------------------------------------------------------

def seed_clusters(
        dist_matrix: csr_matrix,
        is_dense: bool = False,
        max_distance: int = 500
) -> list[int]:
    """Groups ghe locations and waste heat sources together based on their proximity

    This function is used to create groups consisting of both ghe locations and waste heat sources
    to be used as 'seeds' for clustering of buildings in a TEN. These locations will be combined into one
    seed group if they are within a certain distance of each other. 

    If the provided distance matrix is dense, then we can run simple complete clustering on the elements of that matrix

    Args:
        dist_matrix (scipy.sparse.csr_matrix): A distance matrix describing the distances between geometries that are up 
            to max_distance meters away from each other
        is_dense (bool): Whether the provided dist_matrix is dense. 
        max_distance (int): The maximum distance any element in a cluster can be from any other element
            
    Returns:
        list[int]: Cluster labels for each of the ghe / waste heat source locations
    """
    if not is_dense:
        n = dist_matrix.shape[0]
        indptr, indices = dist_matrix.indptr, dist_matrix.indices # indptr maps the rows of the matrix to their corresponding values

        # Indices - The column position for each corresponding value in the data array
        # indptr - pointers mapping rows to the slices of data and indices

        # Neighbor sets
        # Where neighbors[i] is the set of all geometries within a radius of geometry i
        neighbors = [set(indices[indptr[i] : indptr[i + 1]]) - {i} for i in range(n)]

        unassigned = set(range(n))
        clusters = []

        # Process high-degree nodes first
        order = sorted(range(n), key=lambda i: -len(neighbors[i]))

        for seed in order:
            # Do not process the node if it has already been assigned a cluster
            if seed not in unassigned:
                continue

            cluster = [seed]
            unassigned.discard(seed)

            # Candidate neighbors must be unassigned and adjacent to the seed
            candidates = neighbors[seed] & unassigned

            while candidates:
                # Pick the candidate with the most remaining neighbors
                best = max(candidates, key=lambda c: len(neighbors[c] & candidates))
                cluster.append(best)
                unassigned.discard(best)

                # Keep only candidates adjacent to everyone in the cluster so far
                candidates = (candidates & neighbors[best]) - {best} # Set intersection 

            clusters.append(cluster)

        labels = np.empty(dist_matrix.shape[0], dtype=int)
        for k, c in enumerate(clusters):
            labels[c] = k

        return labels

    else:
        Z = linkage(squareform(dist_matrix, checks=False), method="complete")
        labels = fcluster(Z, t=max_distance, criterion="distance")

        return labels


def geometric_median(
        points: np.ndarray,
        tol: float = 1e6,
        max_iter: int = 500
) -> np.ndarray:
    """Finds the geometric median point from a set of provided points

    This function is used to find the geometric median point of a cluster. Each of the provided points
    should be the centroids of the different geometries in a cluster. This can be used to find the point 
    closest to the densest part of the cluster, and so is unaffected by geometries that may be irregularly shaped
    compared to other geometries and thus does not throw off the center. 

    * Note * - This funciton expects the provided points to be projected in a UTM CRS

    Args:
        points (np.ndarray): The centroid points of the geometries
        tol (float): A small value to prevent dividing by zero
        max_iter (int): The max number of iterations to converge on the median point
    
    Returns:
        np.ndarray: The geometric median point
    """
    # Intialize the median point to be the mean of all centroid points
    center_point = points.mean(axis=0)

    for _ in range(max_iter):
        # Get the distance betwen the center point and all other points
        distance = np.linalg.norm(points - center_point, axis=1)
        distance = np.where(distance < tol, tol, distance) # Clamp the value of distance to a non-zero number

        # Get the weight of the distance value
        # further away points are weighted less
        weight = 1 / distance

        # Set the new center point
        # Turn the weights into a column and multiply each point by those weights
        new_center = (points * weight[:, None]).sum(axis=0) / weight.sum()

        if np.linalg.norm(center_point - new_center) < tol: # If the new point moved less than 1e6, we've converged
            break
        center_point = new_center

    return center_point



def find_seed_centroids(
        ghe_locations: gpd.GeoDataFrame,
        cluster_col: str = 'cluster'
) -> gpd.GeoDataFrame:
    """Aggregates the geometric median centroids of each seed cluster in the provided dataframe

    This function is used to gather Point geometries that represent the heart of each seed cluster 
    so that a buffer of buildings can be aggregated around the heart of each seed

    *Note* - The function expects the GHE locations to be in a projected UTM CRS

    Args:s
        ghe_locations (geopandas.GeoDataframe): A GeoDataframe of GHE locations pre-labeled with cluster assignments
    
    Returns:
        geopandas.GeoDataframe: Dataframe containing point objects and the label associated with their cluster
    """
    centroids = ghe_locations.geometry.centroid
    coords = np.column_stack([centroids.x, centroids.y])

    rows = []
    for label, idx in ghe_locations.groupby(cluster_col).indices.items():
        points = coords[idx]
        if len(points) == 1:
            center = points[0]
        else:
            center = geometric_median(points)

        rows.append({cluster_col: label, 'geometry': Point(center)})

    return gpd.GeoDataFrame(rows, geometry='geometry', crs=ghe_locations.crs)