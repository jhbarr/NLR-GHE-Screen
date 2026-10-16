import requests
import time
import geopandas as gpd
from overturemaps import geodataframe

# ---------------------------------------------------------------------------
# API Query Filters + URL Information
# ---------------------------------------------------------------------------

OUTPUT_FIELDS = [
    "OCC_CLS",
    "PRIM_OCC",
    # "SEC_OCC",
    # "PROP_CITY",
    "PROP_ST",
    "PROP_CNTY",
    "HEIGHT",
    "SQMETERS",
    "POP_MEDIAN",
    "SHAPE__AREA",
    "SHAPE__LENGTH"
]

LAYER_URL = "https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/USA_Structures_View/FeatureServer/0"

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ORNLQueryError(Exception):
    """General error for ORNL Buildings query procedures"""

class OvertureQueryError(Exception):
    """General error for Overture buildings query procedure"""

# ---------------------------------------------------------------------------
# Building footprint data aggregation - Via ORNL USA Structures
# ---------------------------------------------------------------------------

def _execute_ornl_query(
        params: dict,
        timeout: int = 60
) -> dict:
    """Send an ORNL query and handle any errors

    This function is responsible for sending a query to the ORNL USA Structures database URL endpoint. 
    Additionally, it handles the parsing of the returned information and catches any possible errors related 
    to the query

    Args:
        params (dict): A dictionary containing the parameters for the ORNL query call
        timeout (int): The API timeout
    
    Returns:
        dict: The parsed API response

    Raises:
        ORNLQueryError: If there is any error while executing the query call
    """
    # Execute the query call to the ORNL databse
    try:
        response = requests.get(f"{LAYER_URL}/query", params=params, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise ORNLQueryError(f"ORNL Request failed: {e}") from e

    try:
        parsed_data = response.json()
    except ValueError as e:
        raise ORNLQueryError(f"Response was not valid JSON: {response.text[:200]!r}") from e
 
    # ArcGIS servers often return HTTP 200 with an error payload
    if "error" in parsed_data:
        raise ORNLQueryError(f"Server returned an error: {parsed_data['error']}")
    
    return parsed_data


def get_ornl_footprints(
        bbox: tuple[float],
        max_features: int = 10000,
        sleep: float = 0.2
) -> gpd.GeoDataFrame:
    """Generate and send a query for all building footprints in a bounding box area from ORNL USA Structures
    
    This function gathers building footprint data from the ORNL USA Structures database via an API query.
    The queries limit responses to 2000 elements, therefore, the function repeatedly sends queries until
    all footprints from the provided bounding box area have been retrieved. 

    Args:
        bbox (Tuple[float]): A coordinate bounding box in the form (lat_min, lon_min, lat_max, lon_max)
        max_features (int): The maximum number of buildings that will be retrieved by the function
        sleep (float): How long the function should pause in between API calls
         
    Returns:
        geopandas.GeoDataframe: A dataframe containing all of the building footprint information 
    """
    print("\n---------------------------")
    print("Retrieving ORNL Structure Data")

    # --- Initialize the query parameters ---
    south, east, north, west = bbox
    params = {
        "where": "1=1",
        "geometry": f"{west}, {south}, {east}, {north}",  # xmin,ymin,xmax,ymax in lon/lat
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": ",".join(OUTPUT_FIELDS),
        "outSR": 4326,
        "returnGeometry": "true",
        "f": "geojson",
    }

    # --- Retrieve the total number of building matches ---
    count_info = _execute_ornl_query(
        {**params, "returnCountOnly": "true", "f": "json"}, 
        timeout=60
    )
    if "count" not in count_info:
        raise ORNLQueryError(f"Count response missing 'count' field: {count_info}")
    
    total = count_info["count"]
    print("total features:", total)


    # --- Issue warning if total exceeds max_features ---
    target = min(total, max_features)
    if max_features is not None and total > max_features:
        print(
            f"Bounding box contains {total} features; only the first {max_features} will be retrieved. "
            "Use a smaller bbox or raise max_features."
        )


    # --- Retrieve the buildings from the ORNL database ---
    features = []
    offset = 0
    try:
        while offset < target:
            p = {**params, "resultOffset": offset, "orderByFields": "OBJECTID"}
            parsed_data = _execute_ornl_query(p, timeout=120)
 
            batch = parsed_data.get("features", [])
            if not batch:  # safety: avoid an infinite loop
                print(f"Empty batch at offset {offset} (expected {target} total); stopping early.")
                break

            features.extend(batch)
            offset += len(batch)  # advance by what the server actually returned
            print(f"{offset}/{total}")
            time.sleep(sleep)

    except ORNLQueryError as e:
        raise ORNLQueryError(
            f"Failed after retrieving {len(features)}/{target} features: {e}"
        ) from e


    # Return an empty dataset if nothing was returned from the API
    if not features:
        print("No features found in bounding box")
        return gpd.GeoDataFrame(columns=[*OUTPUT_FIELDS, "geometry"], geometry="geometry", crs="EPSG:4326")


    print("Success - Data retrieved")
    print("---------------------------")

    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")


def join_ornl_overture_buildings(
        ornl_data: gpd.GeoDataFrame,
        overture_data: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Enrich the ornl structures data using the information from overturemaps

    This function spatially joins the two datasets and utilizes infomration from teh overturemaps
    to fill in any gaps from the ORNL database. The two provided geodataframes must hold info
    that comes from these two databases, otherwise some of the post-processing will throw errors.

    The two dataframes must be in the same CRS

    Args:
        ornl_data (geopandas.GeoDataframe): Dataframe holding ORNL USA Structares building data
        overture_data (geopandas.GeoDataframe): Dataframe holding Overture Maps building data
    
    Returns:
        geopandas.GeoDataframe: The spatially joined dataframe
    """
    joined = gpd.sjoin(
        ornl_data,
        overture_data,
        how='left',
        predicate='intersects'
    ).drop(columns='index_right')

    # Set all primary occupancy fields to apartment 
    # if the data from Overture Maps says so
    target_vals = ['apartments']
    mask = joined['class'].isin(target_vals)
    joined.loc[mask, 'PRIM_OCC'] = joined.loc[mask, 'class']

    # Use the building height from overture maps in the case where
    # ORNL Structures doesn't have any data
    mask = joined['HEIGHT'].isna()
    joined.loc[mask, 'HEIGHT'] = joined.loc[mask, 'height']

    return joined



# ---------------------------------------------------------------------------
# Building footprint data aggregation - Via Overturemaps
# ---------------------------------------------------------------------------

def get_overture_footprints(
        bbox: tuple[float]
    ) -> gpd.GeoDataFrame:
    """Retrieves all of the Overturemap data from the bounding box pertaining to 'buildings'
    
    This function utilizes the :func:`geodataframe` function from the overturemaps Python package

    Args:
        bbox (Tuple[float]): A coordinate bounding box in the form (lat_min, lon_min, lat_max, lon_max)
    
    Returns:
        geopandas.GeoDataframe: A geodataframe with all of the building data from the provided bounding box
    """
    print("\n---------------------------")
    print("Retrieving Overture Maps Data")

    south, west, north, east = bbox
    gdf = geodataframe("building", bbox=(west, south, east, north))

    desired_attributes = [
        'height',
        'num_floors',
        'subtype',
        'class',
        'geometry'
    ]
    gdf = gdf[desired_attributes]
    gdf = gdf.set_crs(epsg=4326)

    print("Success - Data retrieved")
    print("---------------------------")

    return gdf