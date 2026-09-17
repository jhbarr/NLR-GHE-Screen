import geopandas as gpd
from shapely.geometry import shape
import math
import pandas as pd



# ---------------------------------------------------------------------------
# Coordinates to MS Footprint Tiles / Quadkey
# ---------------------------------------------------------------------------

def latlon_to_tile(lat, lon, level=9):
    """
    Convert WGS84 latitude/longitude coordinates into a tile used by MS footprint
    to store data

    Parameters:
        lat (float): Latitude coordiante in the correct CRS
        lon (float): Longitude coordiante in the correct CRS

    Returns:
        tile_x (int): The x value of the MS footprint tile
        tile_y (int): The y value of the MS footprint tile
    """
    lat = max(-85.05112878, min(85.05112878, lat))
    
    n = 2 ** level

    x = (lon + 180.0) / 360.0
    y = (
        1.0
        - math.asinh(math.tan(math.radians(lat))) / math.pi
    ) / 2.0

    tile_x = int(x * n)
    tile_y = int(y * n)

    # Guard against the maximum boundary
    tile_x = min(tile_x, n - 1)
    tile_y = min(tile_y, n - 1)

    return tile_x, tile_y


def tile_to_quadkey(tile_x, tile_y, level=9):
    """
    Converts a MS Footprint tile to a quadkey value to access the building values
    via the MS API

    Parameters:
        tile_x (int): The x value of the MS footprint tile
        tile_y (int): The y value of the MS footprint tile
    
    Returns:
        quadkey (str): The quadkey value used to query the MS API for building data
    """

    quadkey = ""

    for i in range(level, 0, -1):
        digit = 0
        mask = 1 << (i - 1)

        if tile_x & mask:
            digit += 1

        if tile_y & mask:
            digit += 2

        quadkey += str(digit)

    return quadkey


def extract_quadkeys(bbox):
    """
    Using bounding box coordinates, get all quadkey values associated with the MS footprint tiles
    that fall within the area defined by that bounding box

    Parameters:
        bbox (Tuple[float]): A coordinate bounding box in the form (lat_min, lon_min, lat_max, lon_max)

    Returns:
        quadkeys (list[str]): A list of the quadkeys associated with MS footprint tiles
    """
    min_lat, min_lon, max_lat, max_lon = bbox

    min_x, min_y = latlon_to_tile(min_lat, min_lon, level=9)
    max_x, max_y = latlon_to_tile(max_lat, max_lon, level=9)

    quadkeys = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            quadkey = tile_to_quadkey(x, y, level=9)
            quadkeys.append(quadkey[1:]) # ** may cause error later on **

    print("Extracting quadkeys:", quadkeys)

    return quadkeys



# ---------------------------------------------------------------------------
# MS Footprint data aggregation
# ---------------------------------------------------------------------------

def get_ms_building_data(quadkeys):
    """
    Using a set of MS quadkeys, extract all building data from the MS footprint database

    Paramters:
        quadkeys (list[str]): A list of the quadkeys associated with MS footprint tiles
    
    Returns:
        gdf (GeoDataframe): A dataframe containing all extracted building data
    """
    links = pd.read_csv(
        "https://bfppub.blob.core.windows.net/$web/2026-08-13/dataset-links.csv"
    )
    links['QuadKey'] = links['QuadKey'].astype(str)

    selected_regions = links[
        links['QuadKey'].isin(quadkeys)
    ]

    gdfs = []
    for _, row in selected_regions.iterrows():
        print("Loading:", row["QuadKey"])

        df = pd.read_json(row["Url"], lines=True)
        df["geometry"] = df["geometry"].apply(shape)

        gdf = gpd.GeoDataFrame(
            df,
            geometry="geometry",
            crs="EPSG:4326"
        )

        gdfs.append(gdf)

    ms_buildings = pd.concat(gdfs, ignore_index=True) 
    ms_buildings = gpd.GeoDataFrame(
        ms_buildings,
        geometry="geometry",
        crs="EPSG:4326"
    )

    return ms_buildings



# ---------------------------------------------------------------------------
# MS Footprint + OSM space filtering / cross-validation
# ---------------------------------------------------------------------------

def cross_validate_osm_spaces(bbox, osm_spaces):
    """
    Eliminates all parking spaces from the data queried from OSM by cross validating it
    with building data from MS Footprint

    Paramters:
        osm_spaces (GeoDataframe): A dataframe containing all extracted OSM space data
        bbox (Tuple[float]): A coordinate bounding box in the form (lat_min, lon_min, lat_max, lon_max)
    """
    print("\n---------------------------")
    print("Cross validating with MS Footprint")

    osm_spaces_crs = osm_spaces.crs
    osm_parking = osm_spaces[osm_spaces['amenity'] == 'parking'].copy()
    osm_parking = osm_parking.to_crs(osm_spaces.estimate_utm_crs())

    quadkeys = extract_quadkeys(bbox=bbox)
    ms_buildings = get_ms_building_data(quadkeys=quadkeys)
    ms_buildings = ms_buildings.to_crs(osm_parking.crs)

    matches = gpd.sjoin(
        osm_parking,
        ms_buildings,
        how='inner',
        predicate='intersects'
    )

    matches['osm_area'] = matches.geometry.area
    matches['ms_area'] = matches['index_right'].map(
        ms_buildings.geometry.area
    )
    matches['height'] = matches['properties'].apply(
        lambda x: x.get('height', -1)
    )

    matches['intersection_area'] = matches.apply(
        lambda row: row.geometry.intersection(
            ms_buildings.loc[row['index_right']].geometry
        ).area,
        axis=1
    )

    matches["osm_overlap_pct"] = (
        matches["intersection_area"] /
        matches["osm_area"] * 100
    )

    matches["ms_overlap_pct"] = (
        matches["intersection_area"] /
        matches["ms_area"] * 100
    )

    best_matches = matches.loc[
        matches.groupby(matches.index)["osm_overlap_pct"].idxmax()
    ]

    best_matches = best_matches[
        (
            (best_matches["osm_overlap_pct"] >= 50) |
            (best_matches['ms_overlap_pct'] >= 80) &
            (best_matches['height'] > -1.0)
        ) 
    ]

    osm_spaces = osm_spaces.drop(best_matches.index.unique().to_list())

    print("Cross Validation Successful")
    print("---------------------------")

    return osm_spaces.to_crs(osm_spaces_crs)