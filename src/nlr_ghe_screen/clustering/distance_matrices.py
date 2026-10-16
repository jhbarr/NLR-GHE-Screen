import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import osmnx as ox
import networkx as nx

from scipy.sparse import coo_matrix, csr_matrix


# ---------------------------------------------------------------------------
# Construct euclidian dist matrix
# ---------------------------------------------------------------------------

def construct_euclidian_dist_matrix(
        buildings_df: gpd.GeoDataFrame,
        max_distance: int = 250
) -> csr_matrix:
    """Constructs a distance matrix of the buildings to be clustered based on their euclidian distance
    to each other.

    This matrix can be used in clustering to limit the maximum distance members of a cluster can be
    apart from each other. 

    * Note * - This function expects the buildings_df to be in a CRS that allows for meter 
    based calculations, such as UTM.
    
    Args:
        buildings_df (geopandas.GeoDataframe): The dataframe with building geometry information
        max_distance (int): The maximum distance any two buildings in a cluster can be from each other
        
    Returns;
        np.ndarray: The numpy array adjacency matrix describing the distances between buildings
    """
    print("Constructing distance matrix - euclidian")

    geoms = buildings_df.geometry.to_numpy()
    tree = shapely.STRtree(geoms)

    # 'dwithin' finds all geometries that are located within a specific max distance
    # of a given target geometry
    left, right = tree.query(geoms, predicate='dwithin', distance=max_distance) # 2xK array, where each pair represents a pair of close buildings
    mask = left < right # keep each pair once
    left, right = left[mask], right[mask]

    # returns a 1D array with distances for each pair
    d = shapely.distance(geoms[left], geoms[right])

    # Create a sparse coordinate 'coo' matrix 
    n = len(geoms)
    rows = np.concatenate([left, right])
    cols = np.concatenate([right, left])
    vals = np.concatenate([d, d])

    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    # centroids = projected.geometry.centroid 
    # coords = np.column_stack((
    #     centroids.x.to_numpy(),
    #     centroids.y.to_numpy()
    # ))

    # return squareform(pdist(coords, metric='euclidean'))



# ---------------------------------------------------------------------------
# Construct road network dist matrix
# ---------------------------------------------------------------------------

def construct_road_network_dist_matrix(
        gdf: gpd.GeoDataFrame,
        G: nx.MultiDiGraph,
        max_distance: int = 250,
) -> csr_matrix:
    """Constructs a distance adjacency matrix based on the road network distance between geometries

    This function utilizes the Networkx road network graph to construct a more realistic distance adjacency
    matrix between geometries in an area. This function will produce a sparse adjacency matrix, since geometries
    whose distance is greater than the max_distance threshold will not have a distance value 
    
    Args:
        gdf (geopandas.DataFrame): A dataframe whose geometries shoul dbe included in the distance matrix
        G (nx.MultiDiGraph): A Networkx road network graph projected to a UTM CRS
        max_distance (int): The maximum distance two geometries scan be apart
    
    Returns:
        scipy.sparse.csr_matrix: The sparse adjacency matrix
    """
    gdf_projected = gdf.to_crs(G.graph['crs'])

    n = len(gdf_projected)

    # --- Snap each building to the nearest graph node ---
    points = gdf_projected.geometry.representative_point()
    node_ids, snap = ox.distance.nearest_nodes(
        G,
        X=points.x.values,
        Y=points.y.values,
        return_dist=True
    )
    node_ids = np.asarray(node_ids)
    snap = np.asarray(snap)

    # --- Group nodes based on their uniqueness ---
    df = pd.DataFrame({'node': node_ids, 'i': np.arange(n)})
    groups = {k: v.to_numpy() for k, v in df.groupby("node")['i']}


    # --- Run dijsktra per unique node ---
    rows, cols, vals = [], [], []
    for src, src_idx in groups.items():
        reachable = nx.single_source_dijkstra_path_length(
            G, 
            src,
            cutoff=max_distance,
            weight='length'
        )

        for target, distance in reachable.items():
            target_idx = groups.get(target)
            if target_idx is None:
                continue

            ii = np.repeat(src_idx, len(target_idx))
            jj = np.tile(target_idx, len(src_idx))
            dd = distance + snap[ii] + snap[jj]

            keep_mask = (dd <= max_distance) & (ii != jj)
            rows.append(ii[keep_mask]); cols.append(jj[keep_mask]); vals.append(dd[keep_mask])

    rows, cols, vals = map(np.concatenate, (rows, cols, vals))
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()