import osmnx as ox

# ---------------------------------------------------------------------------
# osmnx settings
# ---------------------------------------------------------------------------

ox.settings.use_cache = False          # don't write/read the local cache
# ox.settings.requests_timeout = 30      # used for both the [timeout:N] and the HTTP timeout
ox.settings.overpass_rate_limit = True # osmnx checks the server's slot status and waits if busy
# ox.settings.max_query_area_size = 1000 * 1000 * 1000  # osmnx can automatically split queries if the query area is too large

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
    """
    print("\n---------------------------")
    print("Post - Networkx road network request")

    south, west, east, north = bbox
    graph = ox.graph_from_bbox(
        bbox=(west, south, east, north),
        network_type='drive'
    )

    # Convert the graph to undirected
    graph = graph.to_undirected()

    print("Success - Query Received")
    print("---------------------------")

    return graph