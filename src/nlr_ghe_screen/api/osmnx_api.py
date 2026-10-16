import osmnx as ox
import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
import folium

# ---------------------------------------------------------------------------
# osmnx settings
# ---------------------------------------------------------------------------

ox.settings.use_cache = False          # don't write/read the local cache
# ox.settings.requests_timeout = 30      # used for both the [timeout:N] and the HTTP timeout
ox.settings.overpass_rate_limit = True # osmnx checks the server's slot status and waits if busy
# ox.settings.max_query_area_size = 1000 * 1000 * 1000  # osmnx can automatically split queries if the query area is too large
ox.settings.http_user_agent = "ghe-screen/1.0 (contact: jbarrows@nrel.gov)"
ox.settings.http_referer = "ghe-screen"

# ox.settings.overpass_url = "https://overpass-api.de/api"  # default; change to use another instance

# ---------------------------------------------------------------------------
# Tag filters 
# ---------------------------------------------------------------------------

# A list of values is turned into a regex match: ["key"~"a|b|c"]
TAGS = {
    "leisure": ["park", "dog_park"],
    "landuse": ["recreation_ground", "meadow", "grass", "farmland"],
    "natural": ["wood", "grassland", "scrub", "heath", "wetland"],
    "amenity": ["parking"],
}

# The original query only asked for ways and relations (no nodes).
# osmnx also returns nodes, so we filter them out afterwards.
WANTED_ELEMENTS = ["way", "relation"]

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OXError(Exception):
    """Base exception for OSMNX API"""

class OXNoDataError(OXError):
    """Exception thrown when no data is returned from the OSMNX query"""

# ---------------------------------------------------------------------------
# Main query execution
# ---------------------------------------------------------------------------

def get_osmnx(bbox):
    """
    Fetch OSM features matching TAGS inside a bounding box.

    Args:
        bbox (Tuple[float]): (lat_min, lon_min, lat_max, lon_max),
            i.e. (south, west, north, east) - same convention as the old code.

    Returns:
        GeoDataFrame: one row per way/relation, with a geometry
        column and a column for each tag key found on the returned features.
        Empty GeoDataFrame if nothing matched.
    
    Raises:
        OXNoDataError: If the query to OSMNX does not return any data
        OXError: If the OSMNX query results in any sort of general error
    """
    print("\n---------------------------")
    print("Post - OSMNX API Request")

    south, west, north, east = bbox

    try:
        # osmnx 2.x expects (left, bottom, right, top) = (west, south, east, north)
        gdf = ox.features_from_bbox(bbox=(west, south, east, north), tags=TAGS)
    except ox._errors.InsufficientResponseError:
        # osmnx raises this when the query returns no elements at all
        raise OXNoDataError("No data returned from OSMNX query")
    except Exception as e:
        raise OXError(f"OSMNX exited with error: {e}")


    print("Success - Query Received")
    print("---------------------------")

    # Index is a MultiIndex of (element, id); drop nodes.
    mask = gdf.index.get_level_values("element").isin(WANTED_ELEMENTS)
    return gdf[mask]


def get_road_network(bbox):
    """Extracts all of the road networks and intersections from OSMNX within the provided bounding box

    This function utilizes the ``osmnx.graph_from_bbox()`` feature to extract all of the different
    roads and intersections from an area. This information is useful when creating clusters of buildings 
    as it can be used to define a distance metric.

    Args:
        bbox (Tuple[float]): (lat_min, lon_min, lat_max, lon_max),
            i.e. (south, west, north, east) - same convention as the old code.
    
    Returns:
        networkx.MultiDiGraph: A networkx graph of the roads and intersections

    Raises:
        OXError: In case of any error while retrieving the road network
    """
    print("\n---------------------------")
    print("Post - Networkx road network request")

    south, west, north, east = bbox

    try:
        graph = ox.graph_from_bbox(
            bbox=(west, south, east, north),
            network_type='drive'
        )
    except Exception as e:
        raise OXError(f"OSMnx exited with error: {e}")

    # Convert the graph to undirected
    graph = graph.to_undirected()

    print("Success - Query Received")
    print("---------------------------")

    return graph


def plot_network_graph(
    network, 
    base_gdf,
):
    edge_lines = []
    for u, v, data in network.edges(data=True):
        if 'geometry' in data:
            geom = data['geometry']
        else:
            # Build a LineString from node coordinates
            x1, y1 = network.nodes[u]['x'], network.nodes[u]['y']
            x2, y2 = network.nodes[v]['x'], network.nodes[v]['y']
            geom = LineString([(x1, y1), (x2, y2)])
        edge_lines.append({'u': u, 'v': v, 'geometry': geom})

    gdf_edges = gpd.GeoDataFrame(edge_lines, crs='EPSG:4326')

    # Convert NetworkX nodes to a GeoDataFrame of Points
    node_points = []
    for n, data in network.nodes(data=True):
        node_points.append(
            {'node': n, 'geometry': Point(data['x'], data['y']), **data}
        )

    gdf_nodes = gpd.GeoDataFrame(node_points, crs='EPSG:4326')

    # 3. Create the base interactive map using GeoPandas .explore()
    m = base_gdf.explore(color='green', name='Base Map')

    # 4. Overlay the NetworkX edges and nodes onto the Folium map
    gdf_edges.explore(m=m, color='blue', weight=2, name='Network Edges')
    gdf_nodes.explore(m=m, color='red', marker_kwds={'radius': 4}, name='Network Nodes')

    # Add layer control and display
    folium.LayerControl().add_to(m)

    return m