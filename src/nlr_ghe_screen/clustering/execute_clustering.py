import geopandas as gpd

from nlr_ghe_screen.clustering.seed_clustering import find_seed_centroids_single
from nlr_ghe_screen.clustering.cluster_construction import find_and_filter_nearby_buildings, create_cluster_subgraph
from nlr_ghe_screen.clustering.distance_metrics import road_network_distance_to_centroid
from nlr_ghe_screen.clustering.load_profile_processing import get_building_load_profiles
from nlr_ghe_screen.clustering.dinkelbach_milp import solve_dinkelbach_milp

from nlr_ghe_screen.api.building_footprint_api import get_overture_footprints, get_ornl_footprints, join_ornl_overture_buildings
from nlr_ghe_screen.api.osmnx_api import get_road_network, OXError


# ---------------------------------------------------------------------------
# Main Cluster Execution
# ---------------------------------------------------------------------------

def execute_clustering(
        bbox: tuple[float]
):
    """
    
    """