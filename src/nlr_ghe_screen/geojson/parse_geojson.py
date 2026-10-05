import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import unary_union, polygonize
import osm2geojson
import numpy as np


# ---------------------------------------------------------------------------
# Parse API Results
# ---------------------------------------------------------------------------

def create_dataframe(data):
    """Build a GeoDataframe from a raw Overpass API JSON response

    Converts the Overpass API JSON resposne into GeoJSON format using osm2geojson library and packages that
    GeoJSON data into a geopandas.Dataframe. In the case that the JSON response is empty, a ValueError is thrown

    Args:
        data (dict): The parsed JSON body of an Overpass API response, containing an
            ``elements`` list.
 
    Returns:
        geopandas.GeoDataFrame: One row per element with a valid polygon geometry,
            with the element's tags as columns.
 
    Raises:
        ValueError: If the response contains no elements, or if none of the elements
            produce a valid geometry.
    """
    geojson_dict = osm2geojson.json2geojson(data)

    if len(geojson_dict['features']) == 0:
        raise ValueError("Overpass API response is empty")

    processed_features = []
    for feature in geojson_dict["features"]:
        properties = dict(feature.get("properties", {}))
        tags = properties.pop("tags", {})

        processed_features.append({
            "type": "Feature",
            "geometry": feature.get("geometry"),
            "properties": {**properties, **tags},
        })

    gdf = gpd.GeoDataFrame.from_features(processed_features, crs="EPSG:4326")

    return gdf


def normalize_osmnx_gdf(gdf):
    """Clean an OSMNX GeoDataframe so it matches the format used downstream

    Applies the following siteps to the output of ``ox.features_from_bbox``:

        1. Drops nodes, keeping only ways and relations. This only happens if the
           index has an ``element`` level, so frames without the OSMnx MultiIndex
           still work.
        2. Keeps only ``Polygon`` and ``MultiPolygon`` geometries, discarding points
           and lines such as open ways.
        3. Flattens the index to a fresh ``RangeIndex``. The OSM element type and id
           are discarded rather than kept as columns.
        4. Ensures the CRS is ``EPSG:4326``, setting it if missing or reprojecting
           if it differs.
    
    Args:
        gdf (geopandas.GeoDataFrame): Output of ``ox.features_from_bbox``, normally
            indexed by a MultiIndex of ``(element, id)``.
 
    Returns:
        geopandas.GeoDataFrame: Polygon and MultiPolygon features only, with a fresh
            ``RangeIndex`` and CRS ``EPSG:4326``.
 
    Raises:
        ValueError: If the input is ``None`` or empty, or if no polygon geometries
            remain after filtering.
    """
    POLYGON_TYPES = ("Polygon", "MultiPolygon", "Point")

    if gdf is None or gdf.empty:
        raise ValueError("The osmnx response is empty")
 
    # Drop nodes (the old Overpass query only asked for ways and relations).
    # Guarded so a dataframe without the osmnx MultiIndex still works.
    if "element" in (gdf.index.names or []):
        gdf = gdf[gdf.index.get_level_values("element").isin(["way", "relation"])]
 
    # osmnx can return LineStrings/Points (e.g. open ways); keep areas only.
    gdf = gdf[gdf.geometry.geom_type.isin(POLYGON_TYPES)]
 
    if gdf.empty:
        raise ValueError("No valid polygon geometries found in the osmnx response")
 
    # Flatten the (element, id) MultiIndex. Use reset_index() without drop=True
    # instead if you want to keep the OSM ids as columns.
    gdf = gdf.reset_index(drop=True)
 
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")
 
    return gdf



# ---------------------------------------------------------------------------
# Crop Geometries
# ---------------------------------------------------------------------------

def crop_to_bbox(gdf, bbox):
    """Crop a GeoDataframe and its geometries to a bounding box
    
    Accepts a geopandas.GeoDataframe and a coordinate bounding box and crops the geometries within that dataframe to the boundaries
    of that bounding box. Additionally, the cropping only keeps the geometry types that were originally in the dataframe. Therefore,
    if any new geometry type is created by the cropping, it will be dropped. 

    Args:
        bbox (tuple[float, float, float, float]): The bounding box as
            ``(south, west, north, east)``, i.e. ``(lat_min, lon_min, lat_max,
            lon_max)``
        gdf (geopandas.GeoDataframe): The dataframe with the geometries to be cropped by the bounding box
    
    Returns:
        geopandas.GeoDataframe: The new dataframe with the cropped geometries
    """
    south, west, north, east = bbox
    clipped = gdf.clip((west, south, east, north), keep_geom_type=True)
    clipped = clipped[clipped.geom_type.isin(["Polygon", "MultiPolygon", "Point"])]

    return clipped



# ---------------------------------------------------------------------------
# Process Waste Heat Source Columns
# ---------------------------------------------------------------------------

def add_waste_heat_source(gdf):
    """Takes various tags associated with waste heat sources and compacts them into a single 'waste heat source' column

    Searches for the presence of different tags associated with waste heat sources. If they exist for some building,
    that building is given a waste heat source attribute with that tag as its value. This is to simplify the data in the geodataframe
    so that the program does not need to search multiple attribute fields to check whether a building is a possible waste heat source
    
    Args:
        gdf (geopandas.GeoDataframe): A Geodataframe containing the results of an OpenStreetMap query
    
    Returns:
        geopandas.GeoDataframe: The original dataframe with a waste_heat_source column added
    """
    WASTE_HEAT_RULES = {
        'man_made': {'wastewater_plant', 'water_works'},
        'power': {'plant'},
        'telecom': {'data_center'},
        'building': {'industrial', 'data_center', 'cold_storage'},
        'amenity': {'hospital'},
        'shop': {'supermarket'},
    }

    result = pd.Series(np.nan, index=gdf.index, dtype="object")

    for tag, values in WASTE_HEAT_RULES.items():
        if tag not in gdf.columns:
            continue # tag never appeared in the Overpass query

        # Builds a boolean series from the two conditions
        # Element wise AND, so row is in the mask if it matches the rule and hasn't already been labeled
        mask = gdf[tag].isin(values) & result.isna()
        result[mask] = gdf.loc[mask, tag].astype(str)

    gdf['waste_heat_source'] = result
    return gdf



# ---------------------------------------------------------------------------
# Main Parse Execution
# ---------------------------------------------------------------------------

def parse_api_response(data, bbox, is_df=False):
    """Parse OSM data into a GeoDataframe in the URBANopt GeoJSON format

    Accepts either a raw Overpass API JSON response or an OSMNX GeoDataframe, converts it to a GeoDataframe
    of polygon features, and then:

        - Keeps only the columns ``name``, ``landuse``, ``leisure``, ``natural``,
          ``boundary`` (used to detect protected areas), ``amenity`` (used to detect
          parking), and ``geometry``. All other tags are dropped, and any of these
          columns missing from the source are added as NaN.
        - Adds the fields required by the URBANopt schema: ``type`` is set to
          ``"District System"`` and ``district_system_type`` to ``"Central Hot
          Water"``.
        - Replaces missing ``name`` values with an empty string, since the URBANopt
          schema does not allow null names.
 
    The result uses ``EPSG:4326``.

    Args:
        data (dict | geopandas.GeoDataFrame): An Overpass API JSON response if
            ``is_df`` is ``False``, or an OSMnx GeoDataFrame if ``is_df`` is ``True``.
        is_df (bool, optional): Whether ``data`` is already a GeoDataFrame from OSMnx
            rather than raw Overpass JSON. Defaults to ``False``.
 
    Returns:
        geopandas.GeoDataFrame: Polygon features with the columns above plus
            ``type`` and ``district_system_type``.
 
    Raises:
        ValueError: If the input data is empty, or if no valid geometries are found
            while parsing.
        TypeError: If an Overpass element has a type that cannot be converted to a
            geometry (raised by :func:`create_geometry_object`).
    """
    print("\n---------------------------")
    print("Parsing API Results")

    if not is_df:
        gdf = create_dataframe(data=data)
    else:
        gdf = normalize_osmnx_gdf(gdf=data)

    gdf = add_waste_heat_source(gdf)

    desired_attributes = [
        'name', 
        'landuse', 
        'leisure', 
        'natural', 
        'water',
        'boundary', # For checks on protected areas
        'amenity', # For information on parking areas 
        'waste_heat_source', # For waste heat source info
        'geometry'
    ]
    gdf = gdf.reindex(columns=desired_attributes)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

    # Add required fields for URBANopt GeoJSON schema
    gdf['type'] = "District System"
    gdf['district_system_type'] = "Central Hot Water"
    gdf['name'] = gdf['name'].fillna('') # URBANopt schema cannot have empty name fields

    # Crop the GeoDataframe to the original bounding box
    gdf = crop_to_bbox(gdf, bbox)

    print("Parsing Successful - All necessary data")
    print("---------------------------") 

    return gdf