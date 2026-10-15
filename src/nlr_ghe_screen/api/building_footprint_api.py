import requests
import time
import geopandas as gpd
from overturemaps import geodataframe

# ---------------------------------------------------------------------------
# Building footprint data aggregation - Via ORNL USA Structures
# ---------------------------------------------------------------------------

OUTPUT_FIELDS = [
    "OCC_CLS",
    "PRIM_OCC",
    "SEC_OCC",
    "PROP_CITY",
    "PROP_ST",
    "PROP_CNTY",
    "HEIGHT",
    "SQMETERS",
    "POP_MEDIAN",
    "SHAPE__AREA",
    "SHAPE__LENGTH"
]

LAYER_URL = "https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/USA_Structures_View/FeatureServer/0"


def get_ornl_footprints(
        bbox: tuple[float],
        sleep: float = 0.2
) -> gpd.GeoDataFrame:
    """Generate and send a query for all building footprints in a bounding box area from ORNL USA Structures
    
    This function gathers building footprint data from the ORNL USA Structures database via an API query.
    The queries limit responses to 2000 elements, therefore, the function repeatedly sends queries until
    all footprints from the provided bounding box area have been retrieved. 

    Args:
        bbox (Tuple[float]): A coordinate bounding box in the form (lat_min, lon_min, lat_max, lon_max)
        sleep (float): How long the function should pause in between API calls
         
    Returns:
        geopandas.GeoDataframe: A dataframe containing all of the building footprint information 
    """
    print("\n---------------------------")
    print("Retrieving ORNL Structure Data")

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

    # total number of matches
    total = requests.get(f"{LAYER_URL}/query",
                            params={**params, "returnCountOnly": "true", "f": "json"},
                            timeout=60).json()["count"]
    print("total features:", total)

    features, offset = [], 0
    while offset < min(total, 6000):
        p = {**params, "resultOffset": offset, "orderByFields": "OBJECTID"}
        # omit resultRecordCount so the server uses its own limit
        r = requests.get(f"{LAYER_URL}/query", params=p, timeout=120)
        r.raise_for_status()
        js = r.json()
        if "error" in js:
            raise RuntimeError(js["error"])
        batch = js.get("features", [])
        if not batch:          # safety: avoid an infinite loop
            break
        features.extend(batch)
        offset += len(batch)   # advance by what the server actually returned
        print(f"{offset}/{total}")
        time.sleep(sleep)

    print("Success - Data retrieved")
    print("---------------------------")

    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")



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