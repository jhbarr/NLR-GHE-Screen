import rasterio
import math
import numpy as np

from rasterio.merge import merge
from rasterio.windows import from_bounds
from rasterio.io import MemoryFile # Lets you read / write files in RAM without having to touch disk


YEAR = 2021          # 2020 or 2021
VERSION = "v200"      # "v100" for 2020, "v200" for 2021
OUTPUT_TIF = "../src/imports/worldcover_clip.tif"
 
BASE_URL = f"https://esa-worldcover.s3.amazonaws.com/{VERSION}/{YEAR}/map"



# ---------------------------------------------------------------------------
# Retrieve + Merge WorldCover Tiles
# ---------------------------------------------------------------------------

def tile_id(lat_ll, lon_ll):
    """
    This function returns the World Cover tile associated with a gicen coordinate location

    Paramters:
        lat_ll (float): The latitude coordinate
        lon_ll (float): The longitude coordinate
    
    Returns:
        str: The lookup string for the World Cover tile
    """
    ns = "N" if lat_ll >= 0 else "S"
    ew = "E" if lon_ll >= 0 else "W"

    return f"{ns}{abs(lat_ll):02d}{ew}{abs(lon_ll):03d}"


def tiles_for_bbox(west, south, east, north, step=3):
    """
    This function gathers all of the World Cover tiles within and touching the provided bounding box

    Paramters:
        west, south, east, north (float): The four coordinates describing the bounding box
    
    Returns:
        list[str]: A list of the world cover tile ids within the provided bounding box
    """
    lat_start = math.floor(south / step) * step
    lat_end = math.floor(north / step) * step
    lon_start = math.floor(west / step) * step
    lon_end = math.floor(east / step) * step
 
    ids = []
    lat = lat_start
    while lat <= lat_end:
        lon = lon_start
        while lon <= lon_end:
            ids.append(tile_id(lat, lon))
            lon += step
        lat += step

    return ids


def get_tile_data(urls, bbox):
    """
    This function retrieves all of the world cover tiles associated with the URL links in the given parameter

    Parameters:
        urls (list[str]): A list of AWS bucket links associated with world cover tiles
        bbox (tuple[float]): A bounding box described by coordinates
    
    Returns:
        list[tuple]: A dictionary containing the raster data from each of the world cover tiles. 
    """
    clips = []
    for url in urls:
        # Open a URL stream with rasterio
        # does not require making an API request 
        with rasterio.open(url) as src:
            window = from_bounds(*bbox, transform=src.transform)
            window = window.round_offsets().round_lengths()

            # skip tiles that don't actually intersect (can happen at edges)
            if window.width <= 0 or window.height <= 0:
                continue

            # Retrieve the necessary data and transform info from the file
            transform = src.window_transform(window)
            data = src.read(1, window=window)
            clips.append((data, transform, src.crs))
    
    if not clips:
        raise SystemExit("No data found for this bbox — check coordinates/order (west, south, east, north).")

    return clips


def merge_and_save_tiles(clips):
    """
    This function merges each of the individual world cover tiles together from clips and exports them locally to a .tif file

    Parameters:
        clips (list[tuple]): A list containing the data and transform information from World cover tiles
    """
    if len(clips) == 1:
        data, transform, crs = clips[0]
        data = data[np.newaxis, :, :]

    else:
        # merge() wants rasterio dataset-like objects; easiest is to write
        # each clip to an in-memory dataset first
        memfiles = []
        for arr, transform, crs in clips:
            # Create a memoryfile from the clip
            mem = MemoryFile()
            with mem.open(
                driver="GTiff", height=arr.shape[0], width=arr.shape[1],
                count=1, dtype=arr.dtype, crs=crs, transform=transform,
            ) as dst:
                dst.write(arr, 1)

            memfiles.append(mem)

        data, transform = merge([mf.open() for mf in memfiles]) # combines multiple adjacent / overlapping raster datasets into one
        crs = clips[0][2]

    # Export the merged file locally 
        # Export the merged file locally 
    with rasterio.open(
        OUTPUT_TIF, "w", driver="GTiff",
        height=data.shape[-2], width=data.shape[-1],
        count=1, dtype=data.dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(data)  # data is already (bands, height, width) — no index needed
    
    print(f"Saved clipped raster to {OUTPUT_TIF}")



# ---------------------------------------------------------------------------
# Cross Validate Green Spaces
# ---------------------------------------------------------------------------

def cross_validate_greenspace(clips):
    """
    Go through each of the geometries in clips and cross validate that a majority of their space is indeed green space

    Parameters:
        clips (list[tuple]): A list containing the data and transform information from World cover tiles
    """
    GREEN_CLASSES = [10, 20, 30, 90, 95]

    non_green = []
    for idx, (arr, transform) in clips.items():
        arr = arr.compressed()

        green_mask = np.isin(arr, GREEN_CLASSES)
        pct_green = 100 * green_mask.sum() / green_mask.size

        if pct_green <= 20:
            non_green.append(idx)

    return non_green