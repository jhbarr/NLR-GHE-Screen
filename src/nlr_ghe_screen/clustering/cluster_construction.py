import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import osmnx as ox
import networkx as nx

# ---------------------------------------------------------------------------
# Gather and Rank Nearby Buildings
# ---------------------------------------------------------------------------

def find_nearby_buildings_euclidian(
    seed_cluster_centroids: gpd.GeoDataFrame,
    building_data: gpd.GeoDataFrame,
    max_distance: int = 500
) -> gpd.GeoDataFrame:
    """Find all buildings within a certain distance of the seed cluster centroid of each seed cluster
    using Euclidian distance

    This function gathers all building footprints that are reasonably close to each seed cluster.
    This information can inform which buildings should be included and excluded from consideration for
    inclusion in certain TEN clusters

    *Note* - This function expects each GeoDataframe to be in a projected UTM CRS

    Args:
         seed_cluster_centroids (geopandas.GeoDataframe): A dataframe consisting of points representing the geometric
            median points of each seed cluster
        building_data (geopandas.GeoDataframe): A dataframe with all relevant building data 
        max_distance (int): How far (in meters) a building can be located from a seed cluster centroid 
            to be considered 
    
    Returns:    
        geopandas.GeoDataframe: A dataframe of cluster assignments for each of the buildings
            A building can appear multiple times under different cluster assignments if it is close to multiple
            seed clusters
    """
    building_geoms = building_data.geometry.to_numpy()
    buildings_tree = shapely.STRtree(building_geoms)

    centroid_geoms = seed_cluster_centroids.geometry.to_numpy()

    # Returns a (2, n_matches) array
    # row 0 = positions in centroid_geoms
    # row 1 = positions in teh buildings tree
    center_idx, building_idx = buildings_tree.query(
        centroid_geoms,
        predicate='dwithin',
        distance=max_distance
    )

    labels = seed_cluster_centroids['cluster'].to_numpy()

    matches = building_data.iloc[building_idx].copy()
    matches['cluster'] = labels[center_idx]

    return matches


def find_nearby_buildings_network(
        seed_cluster_centroids: gpd.GeoDataFrame,
        building_data: gpd.GeoDataFrame,
        G: nx.MultiDiGraph,
        max_distance: int = 500
):
    """Find all buildings within a certain distance of the seed cluster centroid of each seed cluster
    using road network distance

    This function gathers all building footprints that are reasonably close to each seed cluster.
    This information can inform which buildings should be included and excluded from consideration for
    inclusion in certain TEN clusters

    Args:
        seed_cluster_centroids (geopandas.GeoDataframe): A dataframe consisting of points representing the geometric
            median points of each seed cluster
        building_data (geopandas.GeoDataframe): A dataframe with all relevant building data 
        G (networkx.MultiDiGraph): 
        max_distance (int): How far (in meters) a building can be located from a seed cluster centroid 
            to be considered 
    
    Returns:    
        geopandas.GeoDataframe: A dataframe of cluster assignments for each of the buildings
            A building can appear multiple times under different cluster assignments if it is close to multiple
            seed clusters
    """
    building_data_projected = building_data.to_crs(G.graph['crs'])
    centroids_projected = seed_cluster_centroids.to_crs(G.graph['crs'])

    # --- Snap each building to the nearest graph node ---
    # --- And group each building by its corresponding graph node ---
    def snap_to_nodes(geoms):
        pts = geoms.geometry.representative_point()   # returns the point itself for Points
        nodes, d = ox.distance.nearest_nodes(
            G, X=pts.x.values, Y=pts.y.values, return_dist=True
        )
        d = np.asarray(d)
        return np.asarray(nodes), d

    def group_by_node(nodes):
        return {k: v.to_numpy() for k, v in
                pd.DataFrame({"node": nodes, "i": np.arange(len(nodes))}).groupby("node")["i"]}

    c_nodes, c_snap = snap_to_nodes(centroids_projected)
    b_nodes, b_snap = snap_to_nodes(building_data_projected)
    c_groups = group_by_node(c_nodes)    # node -> centroid positions
    b_groups = group_by_node(b_nodes)    # node -> building positions

    # --- Find the distance between each centroid and every other node ---
    centroids, buildings, distances = [], [], []
    for centroid_node, centroid_indices in c_groups.items():
        # Get all nodes that are reachable from the current centroid
        reachable = nx.single_source_dijkstra_path_length(
            G,
            centroid_node,
            cutoff=max_distance,
            weight='length'
        )

        for building_node, distance in reachable.items():
            building_indices = b_groups.get(building_node)
            if building_indices is None:
                continue

            pair_centroids = np.repeat(centroid_indices, len(building_indices))
            pair_buildings = np.tile(building_indices, len(centroid_indices))
            pair_distances = distance + c_snap[pair_centroids] + b_snap[pair_buildings]

            # A mask array of bool values based on if the distance between nodes was less than max_distance
            keep = pair_distances <= max_distance
            centroids.append(pair_centroids[keep]); buildings.append(pair_buildings[keep]); distances.append(pair_distances[keep])

    # Concatenate each of the arrays together
    centroids, buildings, distances = map(np.concatenate, (centroids, buildings, distances))  

    # Establish the pairs of buildings and centroids
    pairs = pd.DataFrame({
        'centroid_idx': centroids,
        'building_idx': buildings,
        'cluster': seed_cluster_centroids['cluster'].to_numpy()[centroids]
    })        

    pairs = pairs.sort_values(['building_idx'])
    
    result = building_data.iloc[pairs['building_idx'].to_numpy()].copy()
    result['cluster'] = pairs['cluster'].to_numpy()

    return result


# ---------------------------------------------------------------------------
# Grow Cluster From Seed
# ---------------------------------------------------------------------------

OCC_CLS_RENAME = {
    'Industrial': 'Warehouse' # Warehouse
}

PRIM_OCC_RENAME = {
    'Institutional Dormitory': 'SmallHotel',
    'Medical Office/Clinic': 'Outpatient',
    'Temporary Lodging': 'Hotel',
    'Multi - Family Dwelling': "Multi-Family with 5+ Units",
    'apartments': "Multi-Family with 5+ Units"
}

def _norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()

def find_and_filter_nearby_buildings(
    seed_centroids: gpd.GeoDataFrame,
    building_data: gpd.GeoDataFrame,
    G: nx.MultiDiGraph = None,
    max_distance: int = 500
) -> gpd.GeoDataFrame:
    """Construct a cluster based sequentially

    The cluster is constructed based on multiple factors including how close a building is to the 
    seed cluster centroids and how complementary its thermal load profile is to those locations. 

    *Note* - This function expects each GeoDataframe to be in a projected UTM CRS

    Args:
        seed_centroids (geopandas.GeoDataframe): A dataframe containing each of the centroids for the seed clusters
        building_data (geopandas.GeoDataframe): A dataframe containing building information and their preliminary
            cluster assignments
        G (networkx.MultiDiGraph): An undirected networkx graph that describes the road network connecting the buildings 
            and centroids
        max_distance (int): How far (in meters) a building can be located from a seed cluster centroid 
                    to be considered 
            
    Returns:
        geopandas.GeoDataframe: A dataframe of all cluster buildings with cluster assignment labels
    """
    # Normalize once so the mask and the mapping use identical keys
    prim = _norm(building_data["PRIM_OCC"]).map(PRIM_OCC_RENAME)
    occ = _norm(building_data["OCC_CLS"]).map(OCC_CLS_RENAME)

    # Keep buildings whose type appears in either mapping (even if mapped to None)
    mask = (
        _norm(building_data["PRIM_OCC"]).isin(PRIM_OCC_RENAME.keys())
        | _norm(building_data["OCC_CLS"]).isin(OCC_CLS_RENAME.keys())
    )

    viable_buildings = building_data[mask].copy()

    # comstock type: PRIM_OCC is more specific, so it takes priority over OCC_CLS
    viable_buildings["comstock_type"] = prim[mask].where(prim[mask].notna(), occ[mask])

    # Convert floor area from square meters to square feet
    SQM_TO_SQFT = 10.7639  # 1 m² = 10.7639 ft²
    viable_buildings["sqfeet"] = (viable_buildings['SQMETERS'] * SQM_TO_SQFT).round(-1).astype("Int64")

    # Find all nearby buildings for all cluster centroids
    if G is not None:
        nearby_buildings = find_nearby_buildings_network(
            seed_cluster_centroids=seed_centroids,
            building_data=viable_buildings,
            G=G,
            max_distance=max_distance,
        )
    else:
        nearby_buildings = find_nearby_buildings_euclidian(
            seed_cluster_centroids=seed_centroids,
            building_data=viable_buildings,
            max_distance=max_distance,
        )

    # Fallback: if the search functions dropped our columns, re-attach by index
    missing = [c for c in ("comstock_type", "sqfeet") if c not in nearby_buildings.columns]
    if missing:
        nearby_buildings = nearby_buildings.join(viable_buildings[missing], how="left")

    # Keep only the first instance of each building within a cluster
    dupes = nearby_buildings.assign(_bldg_id=nearby_buildings.index).duplicated(
        subset=['cluster', "_bldg_id"],
        keep="first",
    )
    nearby_buildings = nearby_buildings[~dupes]

    return nearby_buildings


# ---------------------------------------------------------------------------
# Calculate cluster total pipe length
# ---------------------------------------------------------------------------

def create_cluster_subgraph(
        cluster_buildings: gpd.GeoDataFrame,
        cluster_ghes: gpd.GeoDataFrame,
        G: nx.MultiDiGraph,
) -> tuple[nx.MultiDiGraph, float]:
    """Creates a subgraph of a cluster comprised of buildings and GHE / waste heat source locations

    This function is used to create an approximation of an MST for a cluster using road network data. In this way,
    it can be used to calculate the minimum length of pipe necessary to connect all of the components of a proposed
    TEN cluster. This function utilizes Prim's algorithm to find a greedy approximation of the MST.

    Args:
        cluster_buildings (geopandas.GeoDataFrame): Buildings belonging to a single cluster
        cluster_ghes (geopandas.GeoDataFrame): GHE / waste heat source locations belonging to the same cluster
        G (networkx.MultiDiGraph): The road network connecting elements of a cluster

    Returns:
        tuple[nx.MultiDiGraph, float]: The subgraph connecting all elements of the cluster
            and its total length
    """
    def extract_nodes(points):
        return ox.distance.nearest_nodes(
            G,
            X=points.x.values,
            Y=points.y.values,
            return_dist=False
        )

    # Get all of the Networkx nodes associated with those locations
    building_nodes = extract_nodes(cluster_buildings.geometry.representative_point())
    ghe_nodes = extract_nodes(cluster_ghes.geometry.representative_point())

    # Create a concatenated list of all those nodes
    terminals = list(set(np.concatenate((building_nodes, ghe_nodes))))
    if len(terminals) < 2:
        return nx.MultiDiGraph(), 0.0

    # --- Prim's algorithm over shortest-path distances ---
    key = dict.fromkeys(terminals, float('inf'))
    parent = dict.fromkeys(terminals, None)
    key[terminals[0]] = 0

    mst_edges = []  # (u, v, road_distance)
    while key:
        cur = min(key, key=key.get)
        dist_to_cur = key.pop(cur)

        if dist_to_cur == float('inf'):
            raise nx.NetworkXNoPath("Cluster has disconnected nodes")

        if parent[cur] is not None:
            mst_edges.append((parent[cur], cur, dist_to_cur))

        lengths = nx.single_source_dijkstra_path_length(G, cur, weight='length')

        for neighbor in key:
            distance = lengths.get(neighbor, float('inf'))
            if distance < key[neighbor]:
                key[neighbor] = distance
                parent[neighbor] = cur

    # Build the actual road-network subgraph from the MST edges
    sub_edges = set()
    for u, v, _ in mst_edges:
        path = nx.shortest_path(G, u, v, weight='length')
        for a, b in zip(path[:-1], path[1:]):
            k = min(G[a][b], key=lambda k: G[a][b][k].get('length', 0))
            sub_edges.add((a, b, k))

    mst_graph = G.edge_subgraph(sub_edges).copy()
    total_length = mst_graph.size(weight='length')

    return mst_graph, total_length