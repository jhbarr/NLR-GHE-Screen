import gzip
import json
import math

import geopandas as gpd
import pandas as pd
import shapely
from shapely.geometry import shape, box
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor


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

    x1, y1 = latlon_to_tile(min_lat, min_lon, level=9)
    x2, y2 = latlon_to_tile(max_lat, max_lon, level=9)

    min_x, max_x = sorted((x1, x2))
    min_y, max_y = sorted((y1, y2))

    quadkeys = []
    for x in range(min_x, max_x + 1):
        for y in range(min_y, max_y + 1):
            quadkey = tile_to_quadkey(x, y, level=9)
            quadkeys.append(str(int(quadkey)))  # drops leading zeros, matches the CSV

    return quadkeys



# ---------------------------------------------------------------------------
# MS Footprint data aggregation
# ---------------------------------------------------------------------------

def load_tile(url, quadkey, bbox_geom):
    """Loads all of the MS footprint buildings from a tile if they overlap with the bounding box

    This loads all of the MS footprint builings from a tile associated with the provided URL.
    Additinally, it crops those buildings to the bounding box prior to loading those buildings 
    into a dataframe, to prevent unnecessary processing. 

    Args:
        url (str): The MS Footprint URL associated with a tile
        bbox_geom (shapely.geometry): A shapely geometry associated with the bounding box
    
    Returns:
        geopandas.GeoDataframe: A dataframe containing the MS building footprints in the tile

    Raises:
        HTTPError: Raised if there is a problem fetching the data from the URL
    """

    response = requests.get(url, timeout=600)
    response.raise_for_status()
    lines = np.array(
        gzip.decompress(response.content).decode('utf-8').splitlines(), dtype=object
    )

    # Convert GeoJSON strings or bytes into shapely geometry objects
    geoms = shapely.from_geojson(lines)

    # Crop the buildings to within the bounding box
    # Check that geometries still exist once mask has been applied
    mask = shapely.intersects(geoms, bbox_geom)
    if not mask.any():
        return None # No buildings to return within the bbox

    # Parse properties that are associated with the buildings being kept
    props = [json.loads(line).get('properties') or {} for line in lines[mask]]

    print(f"Loaded Quadkey: {quadkey}")
    return gpd.GeoDataFrame(
        pd.DataFrame(props), geometry=geoms[mask], crs="EPSG:4326"
    )


def get_ms_building_data(quadkeys, bbox):
    """Extract all MS building footprints within a given bounding box

    Using all of the quadkeys associated with the MS footprint tiles that overlap with the provided bounding box,
    the process extracts all of the building footprints within that bounding box and returns them in a geodataframe. 
    The return data is clipped so that nothing outside the boundaries of the bbox are returned. 

    This process utilizes a maximum of four concurrent threads to speed up the time it takes to load and process
    each MS footprint tile

    Paramters:
        quadkeys (list[str]): A list of the quadkeys associated with MS footprint tiles
        bbox (Tuple[float]): A coordinate bounding box in the form (lat_min, lon_min, lat_max, lon_max)

    Returns:
        geopandas.GeoDataframe: A dataframe containing all extracted building data
    """
    links = pd.read_csv(
        "https://bfppub.blob.core.windows.net/$web/2026-08-13/dataset-links.csv"
    )
    links['QuadKey'] = links['QuadKey'].astype(str)

    selected_regions = links[
        (links['QuadKey'].isin(quadkeys)) & (links['Location'] == 'UnitedStates')
    ]
    if selected_regions.empty:
        return # Return nothing if there are no quadkeys found in the region

    south, west, north, east = bbox
    bbox_geom = box(west, south, east, north)

    # Open a pool of threads to concurrently load the tiles
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda url, quadkey: load_tile(url, quadkey, bbox_geom), selected_regions['Url'], selected_regions['QuadKey']))

    gdfs = [g for g in results if g is not None]
    if not gdfs:
        return # Return nothing if no data was extracted

    return gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True), geometry='geometry', crs="EPSG:4326"
    )



# ---------------------------------------------------------------------------
# MS Footprint Use Functions
# ---------------------------------------------------------------------------

def footprints_for_points(gdf, ms_buildings):
    """Assigns MS building footprints to corresponding Point geometries in a GeoDataframe

    Point geometries in a GeoDataframe are a set of coordinates. This function takes those coordinates and checks
    whether they overlap with a building footprint from MS Footprint. If so, the Point geometries in the original 
    dataframe are overwritten with the new Polygon shape geometries of the bulding that Point corresponds to.

    Args:
        gdf (geopandas.GeoDataframe): A GeoDataframe containing OSM space information
        ms_buildings (geopandas.GeoDataframe): A GeoDataframe containing MS Footprint building footprtint shapes

    Returns:
        geopandas.GeoDataframe: A new dataframe with Point geometries converted to corresponding Polygon geometries
    """
    gdf = gdf.to_crs(ms_buildings.crs)

    # Join each point to the footprint it falls in
    fp = ms_buildings[['geometry']] # Returns a dataframe with the geometry
    joined = gpd.sjoin(gdf, fp, how='left', predicate='within') # Use left keys and retain left geometries. 

    # Swap the point geometry for the footprtint geometry
    matched = joined[joined['index_right'].notna()].copy()
    matched['geometry'] = ms_buildings.geometry.loc[matched['index_right'].astype(int)].values # Get the geometry values associated with the geometries in the sjoin
    matched = matched.drop(columns='index_right').set_geometry('geometry')

    # Ensure that there is only one polygon for each OSM row
    # Two points that had overlapped the same polygon would create duplicated indicies
    matched = matched[~matched.index.duplicated(keep='first')] # The ~ means NOT

    # Overwrite the geometry for matched rows only
    new_geom = gdf.geometry.copy()
    new_geom.loc[matched.index] = matched.geometry
    result = gdf.set_geometry(new_geom)

    return result



# ---------------------------------------------------------------------------
# MS Footprint + OSM space filtering / cross-validation
# ---------------------------------------------------------------------------

def cross_validate_osm_spaces(bbox, osm_spaces):
    """
    Eliminates all parking spaces from the data queried from OSM by cross validating it
    with building data from MS Footprint

    Executes the entire MS Footprint cross validation workflow. First it isolates only the geometries
    associated with parking spaces in the provided osm_spaces GeoDataframe. 

        1. Extracts the MS quadkeys within the bounds of the GeoDataframe. Does not continue the procedure
            if no quadkeys can be found
        2. Retrieves all of the MS Footprint building data from those quadkey tiles
        3. Overlays those building footprints on the ones from ``osm_spaces`` and isolates the instances
            where there is more than an 80% area overlap and that have an elevation greater than 1.0
        4. Eliminates those geometries from osm_spaces
    
    This modifies the original GeoDataframe ``osm_spaces`` and returns a new one without the parking structures

    Args:
        osm_spaces (Geopandas Dataframe): A dataframe containing all extracted OSM space data
        bbox (Tuple[float]): A coordinate bounding box in the form (lat_min, lon_min, lat_max, lon_max)
    
    Returns:
        geopandas.GeoDataframe: The geometries that were not found to be parking spaces that had a height elevation
            indicating the presence of a parking structure 
    """
    print("\n---------------------------")
    print("Cross validating with MS Footprint")

    osm_spaces_crs = osm_spaces.crs
    osm_parking = osm_spaces[osm_spaces['amenity'] == 'parking'].copy()
    osm_parking = osm_parking.to_crs(osm_spaces.estimate_utm_crs())

    # Quit the process if no MS footprint quadkeys can be found
    quadkeys = extract_quadkeys(bbox=bbox)
    if not quadkeys:
        print("Unable to extract quadkeys for desired area")
        print("---------------------------")
        return osm_spaces

    # If there are no MS buildings in the area of interest
    # return nothing
    ms_buildings = get_ms_building_data(quadkeys=quadkeys, bbox=bbox)
    if len(ms_buildings) > 0:
        ms_buildings = ms_buildings.to_crs(osm_parking.crs)
    else:
        print("No available MS footprint data for desired area")
        print("---------------------------")
        return osm_spaces

    matches = gpd.sjoin(
        osm_parking,
        ms_buildings,
        how='inner',
        predicate='intersects'
    )

    # Check if there is any overlap between the MS and OSM spaces
    if len(matches) > 0:
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
            (
                (best_matches["osm_overlap_pct"] >= 50) |
                (best_matches["ms_overlap_pct"] >= 80)
            ) &
            (best_matches["height"] > -1.0)
        )

        ]

        print("Cross Validation Successful")
        print("---------------------------")

        osm_spaces = osm_spaces.drop(best_matches.index.unique().to_list())
        return osm_spaces.to_crs(osm_spaces_crs)

    print("No overlap - skipping cross validation")
    print("---------------------------")

    return osm_spaces