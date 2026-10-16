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

    c_nodes, c_snap = snap_to_nodes(building_data_projected)
    b_nodes, b_snap = snap_to_nodes(centroids_projected)
    c_groups = group_by_node(c_nodes)    # node -> centroid positions
    b_groups = group_by_node(b_nodes)    # node -> building positions

    ci_all, bi_all, d_all = [], [], []
    for src, centroid_idx in c_groups.items():
        reachable = nx.single_source_dijkstra_path_length(
            G,
            src,
            cutoff=max_distance,
            weight='length'
        )

        for target, distance in reachable.items():
            building_idx = b_groups.get(target)
            if building_idx is None:
                continue

            ii = np.repeat(centroid_idx, len(building_idx))     # every centroid x building pair
            jj = np.tile(building_idx, len(centroid_idx))       # at these two nodes
            dd = distance + c_snap[ii] + b_snap[jj]
            keep = dd <= max_distance
            ci_all.append(ii[keep]); bi_all.append(jj[keep]); d_all.append(dd[keep])

    ci, bi, distance = map(np.concatenate, (ci_all, bi_all, d_all))

    pairs = pd.DataFrame({
        "building_idx": bi,
        "cluster": seed_cluster_centroids['cluster'].to_numpy()[ci], # pulls out cluster column as a plain array
        # "centroid": seed_cluster_centroids.index.to_numpy()[ci], # pulls out indices for each centroid
        # "network_dist": distance,
    })

    pairs = pairs.sort_values(['building_idx'])

    result = building_data.iloc[pairs['building_idx'].to_numpy()].copy()
    result['cluster'] = pairs['cluster'].to_numpy()

    return result


# ---------------------------------------------------------------------------
# Grow Cluster From Seed
# ---------------------------------------------------------------------------

def build_clusters(
    seed_centroids: gpd.GeoDataFrame,
    building_data: gpd.GeoDataFrame,
    max_distance: int = 250
) -> gpd.GeoDataFrame:
    """Construct a cluster based sequentially

    The cluster is constructed based on multiple factors including how close a building is to the 
    seed cluster centroids and how complementary its thermal load profile is to those locations. 

    *Note* - This function expects each GeoDataframe to be in a projected UTM CRS

    Args:
        seed_centroids (geopandas.GeoDataframe): A dataframe containing each of the centroids for the seed clusters
        building_data (geopandas.GeoDataframe): A dataframe containing building information and their preliminary
            cluster assignments
        max_distance (int): How far (in meters) a building can be located from a seed cluster centroid 
                    to be considered 
            
    Returns:
        geopandas.GeoDataframe: A dataframe of all cluster buildings with cluster assignment labels
    """
    # Isolate the building types with the most theoretically the most promising 
    # thermal load profiles
    occ_cls_buildings = [
        'Industrial'
    ]

    prim_occ_buildings = [
        'Multi - Family Dwelling',
        'Institutional Dormitory',
        'apartments',
        'Medical Office/Clinic',
    ]
    mask = (
        building_data['OCC_CLS'].isin(occ_cls_buildings)
        | building_data['PRIM_OCC'].isin(prim_occ_buildings)
    )
    viable_buildings = building_data[mask].copy()


    # Find all nearby buildings for all cluster centroids
    nearby_buildings = find_nearby_buildings_euclidian(
        seed_cluster_centroids=seed_centroids,
        building_data=viable_buildings,
        max_distance=max_distance
    )

    return nearby_buildings
