import geopandas as gpd
import networkx as nx
import cplex

from nlr_ghe_screen.clustering.seed_clustering import find_seed_centroids_single
from nlr_ghe_screen.clustering.cluster_construction import find_and_filter_nearby_buildings
from nlr_ghe_screen.clustering.distance_metrics import road_network_distance_to_centroid, euclidian_distance_to_centroid
from nlr_ghe_screen.clustering.load_profile_processing import get_building_load_profiles
from nlr_ghe_screen.clustering.dinkelbach_milp import solve_dinkelbach_milp

from nlr_ghe_screen.api.building_footprint_api import get_overture_footprints, get_ornl_footprints, join_ornl_overture_buildings, OvertureQueryError, ORNLQueryError
from nlr_ghe_screen.api.osmnx_api import get_road_network, OXError

# ---------------------------------------------------------------------------
# State Abbreviation 
# ---------------------------------------------------------------------------
STATE_ABBREVIATIONS = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "district of columbia": "DC", "florida": "FL", "georgia": "GA", "hawaii": "HI",
    "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME",
    "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE",
    "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH",
    "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI",
    "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}

def state_to_abbr(name: str) -> str | None:
    return STATE_ABBREVIATIONS.get(name.strip().lower())

# ---------------------------------------------------------------------------
# Aggregate Building and Load Profile Data
# ---------------------------------------------------------------------------

def aggregate_building_data(
        bbox: tuple[float],
        crs: gpd.GeoDataFrame.crs
) -> gpd.GeoDataFrame:
    """Gather all of the building data necessary for the clustering procedure

    This function is used to aggregate data from both ORNL USA Structures and Overture Maps databases
    using their respective API calls to get building data. This data is then parsed and cleaned and joined 
    together.

    Args:
        bbox (Tuple[float]): A four float tuple of the form (lat_min, long_min, lat_max, long_max)
        crs (geopandas.GeoDataFrame.crs): The CRS that the dataframes should be in

    Returns:
        (geopandas.GeoDataFrame): the dataframe containing the building data

    Raises:
        ORNLQueryError: General error for ORNL building query procedures
        OvertureQueryError: General error for Overture buildings query procedures
    """
    # Get building data from Overture Maps and ORNL USA Structures
    overture_buildings = get_overture_footprints(bbox=bbox)
    ornl_buildings = get_ornl_footprints(bbox=bbox)

    # Set the CRS of each dataframe
    overture_buildings = overture_buildings.to_crs(crs=crs)
    ornl_buildings = ornl_buildings.to_crs(crs=crs)

    # Join the building data together
    building_data = join_ornl_overture_buildings(
        overture_data=overture_buildings,
        ornl_data=ornl_buildings
    )

    return building_data


def aggregate_load_data(
        building_data: gpd.GeoDataFrame
) -> list[gpd.GeoDataFrame]:
    """Gather all of the building thermal load profile data necessary for the clustering procedure

    This function is used to aggregate load profile data from NLR ComStock and ResStock databases using the 
    NLR building-energy-profiles Python package. For each building in the provided building_data frame, this 
    function will gather a scaled time series profile.

    Args:
        building_data (gpd.GeoDataFrame): A dataframe containing information about the buildings to be clustered
    
    Returns:
        list[gpd.GeoDataFrame]: A list of dataframes containing load profiles for each building 
    
    Raises:
        ValueError: If there are no representative buidlings that can be found 
    """
    # For each of the buildings, gather necessary data
    building_ids = building_data.index.tolist()
    building_types = building_data['comstock_type'].tolist()
    building_sizes = building_data['sqfeet'].tolist()
    building_sources = [
        'restock' if building_type == "Multi-Family with 5+ Units" else 'comstock'
        for building_type in building_types
    ]
    state = state_to_abbr(building_data['PROP_ST'].tolist()[0])

    # Gather the load profiles for each of the buildings in the selected cluster
    load_profiles = get_building_load_profiles(
        building_ids=building_ids,
        building_types=building_types,
        building_sizes=building_sizes,
        building_sources=building_sources,
        state=state,
        downsample=True # downsample to daily time frequency
    )

    return load_profiles




# ---------------------------------------------------------------------------
# Execute MILP
# ---------------------------------------------------------------------------

def execute_milp(
        building_data: gpd.GeoDataFrame,
        building_loads: gpd.GeoDataFrame,
        seed_idx: int,
        k_min: int,
        weight_peak: float = 0.5,
        weight_net: float = 0.5
):
    """Execute the MILP on the proposed buildings using their aggregated building loads

    This function is used to run the Dinkelbach MILP procedure.

    Args:
        building_data (gpd.GeoDataFrame): A dataframe containing information about the buildings to be clustered
        building_loads (gpd.GeoDataFrame): A dataframe with the load profiles of the provided buildings
        seed_idx (int): The index of the load that must be included in the final MILP result
        weight_peak (float): The cost function weight that should be applied to the peak term
        weight_net (float): The cost function weight that should be applied to the imbalance term
        
    Returns:
        dict: A dictionary containing the results of the MILP 
    """
    # Exract the necessary information from the parameters
    building_loads = [b['net_load'].tolist() for b in building_loads]
    building_sizes = building_data['sqfeet'].tolist()
    building_distances = building_data['centroid_distance'].tolist()

    # Execute MILP clustering for the selected cluster
    milp_result = solve_dinkelbach_milp(
        loads=building_loads,
        areas=building_sizes,
        distances=building_distances,
        distance_cap=cplex.infinity,
        seed_idx=seed_idx,
        k_min=k_min,
        weight_peak=weight_peak,
        weight_net=weight_net
    )

    return milp_result




# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def execute_clustering(
        bbox: tuple[float],
        ghe_locations: gpd.GeoDataFrame,
        use_network: bool = True
):
    """Execute the full building clustering procedure
    
    This function is used to peform the entire workflow of building thermal network clustering.

    Args:
        bbox (Tuple[float]): A four float tuple of the form (lat_min, long_min, lat_max, long_max)
        ghe_locations (gpd.GeoDataFrame): Pre-aggregated GHE / waste heat source locations
        use_network (bool): Whether the clustering procedure should use road network or euclidian distances 
        
    Returns:
        dict: A dictionary with the details of the MILP result
        gpd.GeoDataFrame: A dataframe of the buildings to be included in a thermal network cluster 
        gpd.GeoDataFrame: A dataframe with the GHE / waste heat source locations with cluster label assignments
    """
    try:
        # Get building data from Overture Maps and ORNL USA Structures
        building_data = aggregate_building_data(
            bbox=bbox, 
            crs=ghe_locations.crs
        )

        # Get road network data
        # Ensure that it is a non-directed graph
        if use_network:
            network = get_road_network(bbox=bbox)
            network = network.to_undirected()
        else:
            network = None

        # Create seed centroids from each GHE / waste heat source location
        labeled_ghes, seed_centroids = find_seed_centroids_single(
            ghe_locations=ghe_locations,
        )

        # For each of the centroid
        # gather all buildings within a certain road network radiuss
        nearby_buildings = find_and_filter_nearby_buildings(
            seed_centroids=seed_centroids,
            building_data=building_data,
            G=network,
            max_distance=500,
        )

        # Get all buildings near the selected cluster
        # and the centroid associated with that cluster
        seed_idx = 250 # *** This needs to be dynamic somehow *** 

        building_set = nearby_buildings[nearby_buildings['cluster'] == seed_idx].copy()
        cluster_seed_centroid = seed_centroids.loc[seed_centroids['cluster'] == seed_idx, 'geometry']

        # Find the distance between each of the buildings and the seed centroid
        if use_network:
            building_set['centroid_distance'] = road_network_distance_to_centroid(
                buildings_df=building_set,
                centroid=cluster_seed_centroid,
                G=network,
                max_distance=500
            )
        else:
            building_set['centroid_distance'] = euclidian_distance_to_centroid(
                buildings_df=building_set,
                centroid=cluster_seed_centroid,
                max_distance=500
            )

        # Gather load profiles for each building in the selected cluster
        building_loads = aggregate_load_data(
            building_data=building_set
        )

        # Isolate the buildings whose profiles were able to be retrieved 
        building_ids = [b.attrs['bldg_id'] for b in building_loads]
        building_set = building_set.loc[building_ids]

        # Execute the MILP 
        milp_result = execute_milp(
            building_data=building_set,
            building_loads=building_loads
        )
        
        return milp_result, building_set, labeled_ghes

    except(
        OXError,
        OvertureQueryError,
        ORNLQueryError,
        ValueError
    ) as exc:
        print(f"Error - {exc}")