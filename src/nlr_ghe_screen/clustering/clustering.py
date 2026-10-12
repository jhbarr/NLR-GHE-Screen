import geopandas as gpd
import numpy as np

import osmnx as ox
import networkx as nx

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform, pdist
from scipy.sparse.csgraph import connected_components
from scipy.sparse import coo_matrix
import shapely

# ---------------------------------------------------------------------------
# Construct dist matrix + cluster buildings
# ---------------------------------------------------------------------------

def construct_euclidian_dist_matrix(
        buildings_df: gpd.GeoDataFrame,
        max_distance: int
) -> np.ndarray:
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


def cluster_buildings(
        dist_matrix: np.ndarray,
        distance_cutoff: int,
        cluster_method: str
) -> list[int]:
    """
    This function uses hierarchical clustering to cluster building nodes on a networkx graph

    Args:
        dist_matrix (numpy.ndarray): An adjacency matrix for all of the building nodes that are to be clustered
        distance_cutoff (int): The maximum distance a building can be from a cluster centroid
        linkage (str): 'single', 'complete', or 'average'
    
    Returns:
        list[int]: A list of labels corresponding to each building in the graph
    """
    print(f"Clustering buildings - method: {cluster_method}")

    # Split into connected components first
    # Finds islands of buildings that are within distance_cutoff of each other
    n = dist_matrix.shape[0]
    n_comp, comp_labels = connected_components(dist_matrix, directed=False)

    # Setup array that will hold final cluster labels
    labels = np.zeros(n, dtype=int)
    next_label = 1

    # Loop over all 'islands'
    for c in range(n_comp):
        idx = np.where(comp_labels == c)[0] # idx is the array of building indices in the island

        # Isolated building -> its own cluster
        if len(idx) == 1:
            labels[idx] = next_label
            next_label += 1
            continue

        # Rebuild a dense matrix for this component only.
        # Pairs with no stored entry are farther than the cutoff, so fill them
        # with a value above the cutoff instead of 0.
        sub = dist_matrix[idx][:, idx].tocoo() # gets the sub sparse matrix of the island and .tocoo() exposes the data of that matrix
        m = len(idx)
        dense = np.full((m, m), 2 * distance_cutoff, dtype=float) # Fill the matrix with values above the cutoff as default
        dense[sub.row, sub.col] = sub.data # Fill in the entries of the buildings in the island
        np.fill_diagonal(dense, 0) # Each building is next to itself

        # Cluster the buildings in the island
        Z = linkage(squareform(dense, checks=False), method=cluster_method)
        sub_labels = fcluster(Z, distance_cutoff, criterion="distance")

        # Guarantees that each cluster has a unique id across whole dataset
        # since clustering sub_groups may reuse the same cluster ids
        labels[idx] = sub_labels + (next_label - 1)
        next_label += sub_labels.max()

    return labels

    # # Convert the square matrix into condensed form
    # # and perform clustering linkage 
    # condensed = squareform(dist_matrix)
    # Z = linkage(condensed, method=cluster_method)

    # # Create clusters from the linkage using the distance cutoff
    # labels = fcluster(Z, distance_cutoff, criterion='distance')
    
    # return labels



# ---------------------------------------------------------------------------
# Adjust cluster sizes
# ---------------------------------------------------------------------------

def split_clusters(
        buildings_df: gpd.GeoDataFrame,
        max_cluster_size: int
) -> None:
    """
    This function splits all clusters above a max size threshold into smaller clusters
    It splits them based on how close buildings are to each other along the North / South direction

    * Note * - This function expects the buildings_df to be in a CRS that allows for meter 
    based calculations, such as UTM.

    Args:
        buildings_df (geopandas.GeoDataframe): A Dataframe of buildings that has cluster assignments already
        max_cluster_size (int): How large a cluster must be for the process to split it

    Returns
        geopandas.GeoDataframe: A Dataframe of the same buildings with new cluster assignments based on splitting process
    """
    print("Splitting clusters")

    # Iteratively split large clusters until no cluster is above the max threshold
    next_label = buildings_df['cluster'].max() + 1
    while True:
        cluster_counts = buildings_df['cluster'].value_counts()
        large_clusters = cluster_counts[cluster_counts > max_cluster_size].index.to_list()

        # If there are not clusters above the max threhold then the process is complete
        if not large_clusters:
            print('All clusters meet the max size threhold')
            break

        # Iterate through each of the large clusters and split them
        split_any = False
        for large_id in large_clusters:
            # Check if the current cluster has already been split beyond the threhold
            if buildings_df['cluster'].value_counts().get(large_id, 0) <= max_cluster_size:
                continue

            # Filter the dataframe to get the buildings assigned to the current cluster
            cluster_buildings = buildings_df[buildings_df['cluster'] == large_id].copy()
            cluster_buildings['_y'] = cluster_buildings.geometry.centroid.y
            cluster_buildings = cluster_buildings.sort_values('_y')
            cluster_size = len(cluster_buildings)

            # Split the cluster in half and get the indices of the buildings beyond that split point
            split_point = cluster_size // 2
            second_half_indices = cluster_buildings.iloc[split_point:].index

            # Adjust the cluster labels of the buildings past the split point
            cluster_buildings.loc[second_half_indices, 'cluster'] = next_label
            result = cluster_buildings.drop(columns=['_y'], errors='ignore')
            buildings_df.loc[result.index, 'cluster'] = result['cluster']

            print(f'Split cluster {large_id} from {cluster_size} to {split_point}')
            next_label += 1


def get_cluster_centroids(gdf, cluster_col='cluster'):
    """Get the centroid x, y for each cluster

    * Note * - This function expects the buildings_df matrix to be in a CRS that allows for meter 
    based calculations, such as UTM.

    Args:
        gdf (geopandas.GeoDataframe): A geodataframe with cluster assignments for each row
    
    Returns:
        np.ndarray: An array containing the x, y center points for each cluster 
    """
    # Compute each building's centroid once, up front
    building_centroids = gdf.geometry.centroid
    xs = building_centroids.x
    ys = building_centroids.y

    centroids = {}
    for cluster_id in gdf[cluster_col].unique():
        mask = (gdf[cluster_col] == cluster_id).to_numpy()
        centroids[cluster_id] = np.array([xs[mask].mean(), ys[mask].mean()])

    return centroids


def join_clusters(
        buildings_df: gpd.GeoDataFrame,
        min_cluster_size: int
) -> None:
    """
    This function joins all clusters below a min size threshold into larger clusters
    It joins them based on how close two cluster centroids are.

    This function does not return any values. Instead, it adjusts the cluster assignments for each geometry
    in place in the provided geodataframe.  

    * Note * - This function expects the buildings_df to be in a CRS that allows for meter 
    based calculations, such as UTM.

    Args:
        buildings_df (geopandas.Geodataframe): A Dataframe of buildings that has cluster assignments already
        min_cluster_size (int): How small a cluster must be for the process to join it
    """
    print("Joining clusters")

    # Iteratively merge all small clusters until all are larger than min threhold
    while True:
        cluster_counts = buildings_df['cluster'].value_counts()
        small_clusters = cluster_counts[cluster_counts < min_cluster_size].index.to_list()

        # If there are not clusters under the min threhold then the process is complete
        if not small_clusters:
            print('All clusters meet the min size threhold')
            break

        # Get the centroids of all clusters
        centroids = get_cluster_centroids(buildings_df)
        all_cluster_ids = list(centroids.keys())

        # Iterate through each of the small clusters and merge them with their closest neighbor
        merged_any = False
        for small_id in small_clusters:
            # Check if the cluster still exists and hasn't been merged already
            # or if a previous merge process has increased the size past the threshold
            if small_id not in buildings_df['cluster'].values:
                continue
            if buildings_df['cluster'].value_counts().get(small_id, 0) >= min_cluster_size:
                continue
            
            # Get the centroid of the small cluster
            # check each of the other nearby clusters and merge it with the closest one
            small_centroid = centroids[small_id]
            best_neighbor = None
            best_dist = np.inf

            for other_id in all_cluster_ids:
                if other_id == small_id:
                    continue
                if other_id not in buildings_df['cluster'].values:
                    continue

                # Calculate the distance to the other centroid
                # and check if its the smallest distance yet
                other_centroid = centroids[other_id]
                dist = np.linalg.norm(small_centroid - other_centroid)
                if dist < best_dist:
                    best_dist = dist
                    best_neighbor = other_id
            
            # If there is a neighbor that the cluster should be merged with, do so
            if best_neighbor is not None:
                print(f"Merging cluster {small_id} (size {cluster_counts.get(small_id, 0)}) "
                f"into cluster {best_neighbor} (size {cluster_counts.get(best_neighbor, 0)})")

                # Reassign the labels of the dataframe to match this merge
                buildings_df.loc[buildings_df['cluster'] == small_id, 'cluster'] = best_neighbor
                merged_any = True
        
        # Stop the process once no more merges should take place 
        if not merged_any:
            print("Warning: no merges possible, breaking to avoid infinite loop")
            break



# ---------------------------------------------------------------------------
# Gather nearby GHE locations
# ---------------------------------------------------------------------------

def nearby_ghe_locations(
        buildings_df: gpd.GeoDataFrame, 
        ghe_locations: gpd.GeoDataFrame,
        distance_radius: int = 200
    ) -> gpd.GeoDataFrame:
    """Determines which clusters each of the ghe locations is closest to

    This function can be used as the backbone of modelling the kinds of TENs that can be implemented
    around a given cluster. One ghe location can be associated with multiple clusters

    * Note * - This function expects the buildings_df and gdf dataframes to be in a CRS that allows for meter 
    based calculations, such as UTM.

    Args:
        buildings_df (geopandas.Geodataframe): A Dataframe of buildings that has cluster assignments already
        gdf (geopandas.Geodataframe): The dataframe containing the GHE location information
        distance_radius (int): How far away from any one member of the cluster a possible GHE location can be located

    Returns:
        geopandas.GeoDataframe: A dataframe containing a subset of the 
    """
    cluster_geoms = buildings_df.dissolve(by='cluster').geometry
    cluster_buffers = gpd.GeoDataFrame(
        {"cluster": cluster_geoms.index},
        geometry=cluster_geoms.buffer(distance_radius).values,
        crs=ghe_locations.crs,
    )

    # centroids = clustered.dissolve(by='cluster').centroid
    # centroid_buffers = gpd.GeoDataFrame(
    #     {'cluster': centroids.index},
    #     geometry=centroids.buffer(radius).values,
    #     crs=ghe_locations.crs,
    # )

    # how='inner': keeps only rows where the predicate is true
    # predicate='within': evaluates to true when the left geometry is totally within the right geometry
    nearby = gpd.sjoin(ghe_locations, cluster_buffers, how='inner', predicate='intersects')

    return nearby



# ---------------------------------------------------------------------------
# Main clustering execution
# ---------------------------------------------------------------------------

def get_clusters(
        ghe_locations: gpd.GeoDataFrame,
        buildings_df: gpd.GeoDataFrame,
        distance_cutoff: int = 1000,
        cluster_method: str = 'complete'
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Clusters buildings together based on their distance and gathers all possible GHE
    locations that are within a certain distance of those clusters

    This function first creates a distance matrix based on either euclidian or road network 
    distance between buildings. From those distances, clusters of max size 150 and min size 50 are
    created. Based on the locations of those clusters, all GHE locations are aggregated and assigned to the
    clusters. 
    
    Args:
        ghe_locations (geopandas.GeoDataframe): A geodataframe with cluster assignments for each row
        buildings_df (geopandas.Geodataframe): A Dataframe of buildings that has cluster assignments already
        distance_cutoff (int): How far (in meters) buildings in a single cluster can be from each other
        cluster_method (str): Either ``'complete'`` ``'average'`` or ``'single'``

    Returns:
        geopandas.GeoDataframe: The buildings_df dataframe with each row assigned to a cluster label
        geopandas.GeoDataframe: The gdf dataframe with each GHE location assigned a cluster label
            A single GHE location can be close to multiple clusters. Therefore, locations may appear
            more than once with different cluster labels. 
    """
    # Save the original CRS of both dataframes
    buildings_df_crs = buildings_df.crs
    ghe_crs = ghe_locations.crs

    projected_buildings = buildings_df.to_crs(buildings_df.estimate_utm_crs())
    projected_ghe = ghe_locations.to_crs(projected_buildings.crs)

    # Create distance matrix for the buildings
    dist_matrix = construct_euclidian_dist_matrix(
        buildings_df=projected_buildings,
        max_distance=distance_cutoff
    )

    # Cluster the buildings based on the distance matrix
    # and gather each buildings' label
    labels = cluster_buildings(
        dist_matrix=dist_matrix,
        distance_cutoff=distance_cutoff,
        cluster_method=cluster_method
    )

    projected_buildings['cluster'] = labels

    # Adjust the sizes of the clusters so there is no mega cluster
    # or any extremely small clusters
    split_clusters(buildings_df=projected_buildings, max_cluster_size=150)
    join_clusters(buildings_df=projected_buildings, min_cluster_size=50)

    # Aggregate all GHE locations near each cluster
    nearby_ghes = nearby_ghe_locations(
        buildings_df=projected_buildings,
        ghe_locations=projected_ghe
    ).drop(columns='index_right')

    # Convert dataframes back to original CRS
    projected_buildings = projected_buildings.to_crs(buildings_df_crs)
    nearby_ghes = nearby_ghes.to_crs(ghe_crs)

    return projected_buildings, nearby_ghes