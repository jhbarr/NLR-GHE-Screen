import rasterio
import numpy as np
import pandas as pd

from rasterio.mask import mask
from rasterio.warp import Resampling, calculate_default_transform, reproject


IMPORT_DIR = 'imports/'

# ---------------------------------------------------------------------------
# Load and Reproject Data
# ---------------------------------------------------------------------------

def load_elevation_data(gdf):
    """
    Loads elevation raster data from local imports folder

    Parameters
        gdf (Geopandas Dataframe): A GeoDataframe containing geometries within the bounds of the elevation raster data
    
    Returns
        rasterio.io.DatasetReader: The elevation raster data

    Raises:
        FileNotFoundError: If the elevation raster data cannot be found
    """
    # Attempt to open the raster data that should be downloaded in the local imports/ folder
    try:
        file_path = IMPORT_DIR + 'elevation_data.tif'
        elevation_src = rasterio.open(file_path)
    except rasterio.errors.RasterioIOError as e:
        raise FileNotFoundError(f"No elevation data found: {e}")

    # Ensure that the raster and vector data are in the same CRS
    if gdf.crs != elevation_src.crs:
        gdf = gdf.to_crs(elevation_src.crs)

    return elevation_src


def reproject_elevation_data(elevation_src, dst_crs, resampling=Resampling.bilinear):
    """
    Reproject a raster to a new crs and write that information to a local file
    
    Parameters
        elevation_src (rasterio.io.DatasetReader): The elevation raster data
        dst_crs (pyproj.crs.CRS): The new CRS that the raster data is to be projected to
        resampling : bilinear is a good default for continuous data like elevation.
                        Never use nearest for slope work if you can avoid it; it
                        produces blocky, noisy slopes.
    """
    transform, width, height = calculate_default_transform(
        elevation_src.crs,
        dst_crs,
        elevation_src.width,
        elevation_src.height,
        *elevation_src.bounds,
        resolution=None
    )

    nodata = elevation_src.nodata if elevation_src.nodata is not None else -9999.0
     
    profile = elevation_src.profile.copy()
    profile.update(
        crs=dst_crs,
        transform=transform,
        width=width,
        height=height,
        dtype="float32",   # bilinear resampling produces fractional values
        nodata=nodata,
    )

    # Write the new reprojected data to the imports folder
    dst_path = IMPORT_DIR + 'reprojected_elevation_data.tif'
    with rasterio.open(dst_path, "w", **profile) as dst:
        for band in range(1, elevation_src.count + 1):
            reproject(
                source=rasterio.band(elevation_src, band),
                destination=rasterio.band(dst, band),
                src_transform=elevation_src.transform,
                src_crs=elevation_src.crs,
                src_nodata=elevation_src.nodata,
                dst_transform=transform,
                dst_crs=dst_crs,
                dst_nodata=nodata,
                resampling=resampling,
            )




# ---------------------------------------------------------------------------
# Organize Raster Data
# ---------------------------------------------------------------------------

def clip_raster_to_shapes(elevation_src, gdf):
    """
    Iterates through each geometry in the provided GeoDataframe and attaches the raster data that the
    geometry contains / touches

    Parameters:
            gdf (Geopandas Dataframe): A GeoDataframe containing geometries within the bounds of the elevation raster data
            elevation_src (rasterio.io.DatasetReader): The elevation raster data
    
    Returns:
        gdf (Geopandas Dataframe): A GeoDataframe containing geometries within the bounds of the elevation raster data
        clips (dict): A dictionary of the form
            {
                geometry_index: (raster_data, raster_transform)
            }
    """
    clips = {}
    for idx, geom in gdf.geometry.items():
        # Get the elevation raster data tiles that the geometry touches
        # filled=True means all other data will be masked
        try:
            out_image, out_transform = mask(
                elevation_src, [geom], crop=True, filled=False, all_touched=True
            )
        except ValueError as e:
            print(f"Skipping geometry {idx}: {e} | bounds={geom.bounds}")
            continue

        arr = out_image[0] # First band
        arr = np.ma.masked_equal(arr, elevation_src.nodata)
        clips[idx] = (arr, out_transform)

    return gdf, clips



# ---------------------------------------------------------------------------
# Summarize + Calculate Raster information
# ---------------------------------------------------------------------------

def summarize_stats(clips):
    """
    This funciton iterates through all of the geometries in the provided clips dictionary and 
    aggregates informational stats about the raster data associated with those geometries

    Parameters:
        clips (dict): A dictionary of the form
            {
                geometry_index: (raster_data, raster_transform)
            }
    
    Returns:
        stats (dict): A dictionary containing the min, max, mean, median, num_pixels values of the raster data
            associated with each of the individual geometries
    """
    stats = {}
    for idx, (arr, transform) in clips.items():
        vals = arr.compressed() # Extracts and returns all non-masked data as a standard 1d array

        if vals.size == 0:
            continue

        stats[idx] = {
            "min": vals.min(),
            "max": vals.max(),
            "mean": vals.mean(),
            "median": np.median(vals),
            "num_pixels": vals.size
        }

    return stats


def calculate_slopes(elevation_src):
    """
    Calculate the slope angle of each of the individual raster tiles and writes that information to a new
    raster file in the local imports folder

    Parameters:
        elevation_src (rasterio.io.DatasetReader): The elevation raster data
    """

    dem = elevation_src.read(1, masked=True).astype(float).filled(np.nan) # Pixels matching the nodata value are flagged as masked so as to not skew math functions
    dx, dy = elevation_src.res # Get the real world dimensions of a single raster pixel
    profile = elevation_src.profile

    gy, gx = np.gradient(dem, dy, dx)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype('float32')

    profile.update(dtype="float32", nodata=np.nan)
    file_path = IMPORT_DIR + 'slope.tif'
    with rasterio.open(file_path, "w", **profile) as dst:
        dst.write(slope, 1)



# ---------------------------------------------------------------------------
# Categorize geometries by steepness
# ---------------------------------------------------------------------------

def categorize_geometries(gdf, clips):
    """
    Assign a category to each of the geometries in the provided GeoDataframe based on how steep they are

    Parameters:
        gdf (Geopandas Dataframe): A GeoDataframe containing geometries within the bounds of the elevation raster data
        clips (dict): A dictionary of the form
            {
                geometry_index: (raster_data, raster_transform)
            }
    """

    stats = pd.DataFrame(summarize_stats(clips)).T

    stats['category'] = pd.cut(
        x=stats['median'], # The 1d array or series to be cut
        bins=[0, 4, 8, 15, np.inf], # The list of bin edges
        labels=['flat', 'mild', 'medium', 'steep'],
        include_lowest=True
    )

    gdf['slope_category'] = stats['category']



# ---------------------------------------------------------------------------
# Execute slope calculations and categorization
# ---------------------------------------------------------------------------

def categorize_steepness(gdf):
    """
    This function will categorize all of the geometries in the provided gdf based on how steep they are
    
    Parameters:
        gdf (Geopandas Dataframe): A GeoDataframe containing geometries within the bounds of the elevation raster data
    """
    print("\n---------------------------")
    print("Categorizing geometry steepness")

    elevation_src = load_elevation_data(gdf=gdf)

    # Reproject the elevation data so that steepness calculations are possible
    reproject_elevation_data(
        elevation_src=elevation_src,
        dst_crs=gdf.estimate_utm_crs(),
    )
    reprojected_file_path = IMPORT_DIR + 'reprojected_elevation_data.tif'
    reprojected_elevation = rasterio.open(reprojected_file_path)

    # Calculate the slopes of each of the reprojected raster tiles
    # and download the crated slope raster data
    calculate_slopes(reprojected_elevation)
    slope_file_path = IMPORT_DIR + 'slope.tif'
    slope_src = rasterio.open(slope_file_path)

    # Clip the slope data to the gdf's geometries
    reprojected_gdf = gdf.to_crs(gdf.estimate_utm_crs()).copy()
    reprojected_gdf, slope_clips = clip_raster_to_shapes(slope_src, reprojected_gdf)

    # Categorize the steepness of each geometry
    categorize_geometries(reprojected_gdf, slope_clips)

    print("Categorization Successful")
    print("---------------------------")

    gdf['slope_category'] = reprojected_gdf['slope_category']
    return gdf