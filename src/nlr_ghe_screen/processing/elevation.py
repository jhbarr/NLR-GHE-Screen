import rasterio
import numpy as np
import pandas as pd
from pathlib import Path

from rasterio.mask import mask
from rasterio.warp import Resampling, calculate_default_transform, reproject


IMPORT_DIR = Path(__file__).parent.parent / '..' / 'imports'

# ---------------------------------------------------------------------------
# Load and Reproject Data
# ---------------------------------------------------------------------------

def load_elevation_data(gdf):
    """Open the local elevation raster file and check if against the vector data's CRS
    
    Looks for a 'elevation_data.tif' file located in the src/imports folder within the project repository and opens
    it with rasterio. The GeoDataframe is only used to compare coordinate reference systems (CRS) to ensure that they are the same. 

    Args:
        gdf (geopandas.GeoDataFrame): Geometries that fall within the bounds of the
            elevation raster. Only its CRS is inspected.
    
    Returns:
        rasterio.io.DatasetReader: An open, read-mode handle to the elevation raster.
            The caller is responsible for closing it.

     Raises:
        FileNotFoundError: If ``elevation_data.tif`` cannot be found or opened in the
            imports folder.
    """
    # Attempt to open the raster data that should be downloaded in the local imports/ folder
    try:
        file_path = IMPORT_DIR / 'elevation_data.tif'
        elevation_src = rasterio.open(file_path)
    except rasterio.errors.RasterioIOError as e:
        raise FileNotFoundError(f"No elevation data found: {e}")

    # Ensure that the raster and vector data are in the same CRS
    if gdf.crs != elevation_src.crs:
        gdf = gdf.to_crs(elevation_src.crs)

    return elevation_src


def reproject_elevation_data(elevation_src, dst_crs, resampling=Resampling.bilinear):
    """Reproject an elevation raster to a new crs and write that information to disk

    Every band of the source raster is warped into the new 'dst_crs' and written to a file named 
    'reprojected_elevation_data.tif' in the src/imports folder. This overwrites any preexisting file of the same
    name in that location. 

    This reprojection is typically run before calculations regarding slope, since slope requires a projected CRS with
    linear units rather than coordinate degrees
    
    Args:
        elevation_src (rasterio.io.DatasetReader): The source elevation raster.
        dst_crs (pyproj.crs.CRS): The CRS the raster should be reprojected to.
        resampling (rasterio.enums.Resampling, optional): Resampling method used when
            warping. Defaults to ``Resampling.bilinear``, which suits continuous data
            like elevation. Avoid ``nearest`` for slope work, as it produces blocky,
            noisy slopes.
 
    Returns:
        None: The result is written to ``imports/reprojected_elevation_data.tif``.
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
    dst_path = IMPORT_DIR / 'reprojected_elevation_data.tif'
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
    """Clip raster pixels to each geometry in the GeoDataframe

    For every geometry, it extracts the raster cells that the geometry contains or touches and returns them as a masked numpy array 
    cropped to the geometry's bounding box. Geometries that do not overlap the raster completely are skipped with a warning message. 

    The raster and the geometries must share the same CRS. 
    

    Args:
        elevation_src (rasterio.io.DatasetReader): The raster to clip (band 1 is used).
        gdf (geopandas.GeoDataFrame): Geometries to clip the raster with.
    
    Returns:
        (geopandas.GeoDataFrame): The input GeoDataFrame, unchanged
        clips (dict): (dict): Maps each geometry's index to a tuple of
              ``(masked_array, affine_transform)``, i.e.
              ``{geometry_index: (raster_data, raster_transform)}``.
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
    """Summarize the slope distribution within each clipped geometry

    For each geometry, takes the unmasked slope values (percent gradient) and computes the share of pixels that fall into each
    slope bin. Plus the mean, median, and pixel count. 

    The bins are 0-5%, 5-10%, 10-20%, and >20%. Bin values are the percentage of
    valid pixels in that bin, rounded to the nearest whole number (0-100).

    Args:
    clips (dict): Clipped slope data in the form
        ``{geometry_index: (raster_data, raster_transform)}``, as produced by
        :func:`clip_raster_to_shapes`.
    
    Returns:
        dict: Maps each geometry index to a dictionary with the keys:
 
            - ``"0-5%"``, ``"5-10%"``, ``"10-20%"``, ``">20%"`` (float): Percentage of
              the geometry's valid pixels in each slope bin.
            - ``"mean"`` (float): Mean slope value.
            - ``"median"`` (float): Median slope value.
            - ``"num_pixels"`` (int): Number of valid pixels used.
    """
    stats = {}
    for idx, (arr, transform) in clips.items():
        vals = arr.compressed() # Extracts and returns all non-masked data as a standard 1d array

        if vals.size == 0:
            continue

        bins = [0, 5, 10, 20, np.inf]
        labels = ["0-5%", "5-10%", "10-20%", ">20%"]

        counts, _ = np.histogram(vals, bins=bins)
        num_pixels = vals.size

        stats[idx] = {
            **dict(zip(labels, np.round(counts / num_pixels, 2) * 100)),
            "mean": vals.mean(),
            "median": np.median(vals),
            "num_pixels": num_pixels,
        }

    return stats


def calculate_slopes(elevation_src):
    """Calculate the percent-gradient slope for an elevation raster and save it

    Computes the elevation gradient percentage for all cells in the provided raster and the resulting data
    is written to 'slope.tif' in the src/imports folder. This overwrites an preexisting files of the same name.
    A slope angle in degrees is also calculated internally. 

    The raster must be in a projected CRS with linear units (see :func:`reproject_elevation_data`), otherwise the cell sizes
    and elevation values will be in different units and slope values will be nonsensical. 

    Args:
        elevation_src (rasterio.io.DatasetReader): Elevation raster in a projected
            CRS. Band 1 is used, and masked cells become NaN.
    
    Returns:
        numpy.ndarray: A float32 2D array of percent gradient values with the same
            shape as the input raster. Nodata cells are NaN.
    """
    dem = elevation_src.read(1, masked=True).astype(float).filled(np.nan)
    dx, dy = elevation_src.res
    profile = elevation_src.profile

    gy, gx = np.gradient(dem, dy, dx)
    rise_run = np.hypot(gx, gy)  # tangent of the slope angle, i.e. rise/run

    slope_deg = np.degrees(np.arctan(rise_run)).astype("float32")
    percent_gradient = (rise_run * 100).astype("float32")

    profile.update(dtype="float32", nodata=np.nan)
    dst_path = IMPORT_DIR / 'slope.tif'
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(percent_gradient, 1)

    return percent_gradient



# ---------------------------------------------------------------------------
# Categorize geometries by steepness
# ---------------------------------------------------------------------------

def categorize_geometries(gdf, clips):
    """Assign each geometry a steepness category based on its dominant slope bin.

    Summariizes the sloep data for each geometry (see :func:`summarize_stats) and picks the slop bin that contains the largest share
    of that geometry's pixels. The bin is then mapped to this category. 

        - 0-5%   -> ``"flat"``
        - 5-10%  -> ``"mild"``
        - 10-20% -> ``"medium"``
        - >20%   -> ``"steep"``
    
    The result is then stored in a new 'slope_category' column the gdf. Geometries missing from clips get NaN. This function
    modifies gdf in place and does not return anything.

    Args:
        gdf (geopandas.GeoDataFrame): Geometries to categorize. Modified in place.
        clips (dict): Clipped slope data in the form
            ``{geometry_index: (raster_data, raster_transform)}``, as produced by
            :func:`clip_raster_to_shapes`.
 
    Returns:
        None: ``gdf`` is updated in place with a ``slope_category`` column.
    """
    stats = pd.DataFrame(summarize_stats(clips)).T

    bin_to_category = {
        "0-5%": "flat",
        "5-10%": "mild",
        "10-20%": "medium",
        ">20%": "steep",
    }

    bin_cols = list(bin_to_category)

    stats["category"] = (
        stats[bin_cols]
        .astype(float)
        .idxmax(axis=1)
        .map(bin_to_category)
    )

    gdf["slope_category"] = stats["category"]



# ---------------------------------------------------------------------------
# Execute slope calculations and categorization
# ---------------------------------------------------------------------------

def categorize_steepness(gdf):
    """Categorize geometries by steepness and drop the steep ones

    This function runs the full slope workflow:

        1. Load the local elevation raster.
        2. Reproject it to the UTM zone estimated from ``gdf`` so slopes can be
           calculated in linear units.
        3. Calculate percent-gradient slopes and save them to ``slope.tif``.
        4. Clip the slope raster to each geometry (reprojected to the same UTM CRS).
        5. Assign each geometry a category of flat, mild, medium, or steep.
        6. Add the category to ``gdf`` and filter out steep geometries.
 
    Intermediate rasters (``reprojected_elevation_data.tif`` and ``slope.tif``) are
    written to the imports folder as a side effect. The returned GeoDataFrame keeps
    its original CRS.
    
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
    reprojected_file_path = IMPORT_DIR / 'reprojected_elevation_data.tif'
    reprojected_elevation = rasterio.open(reprojected_file_path)

    # Calculate the slopes of each of the reprojected raster tiles
    # and download the crated slope raster data
    calculate_slopes(reprojected_elevation)
    slope_file_path = IMPORT_DIR / 'slope.tif'
    slope_src = rasterio.open(slope_file_path)

    # Clip the slope data to the gdf's geometries
    reprojected_gdf = gdf.to_crs(gdf.estimate_utm_crs()).copy()
    reprojected_gdf, slope_clips = clip_raster_to_shapes(slope_src, reprojected_gdf)

    # Categorize the steepness of each geometry
    categorize_geometries(reprojected_gdf, slope_clips)

    print("Categorization Successful")
    print("---------------------------")

    gdf['slope_category'] = reprojected_gdf['slope_category']
    gdf = gdf[gdf['slope_category'] != 'steep'] # Exclude all of the geometries that were categorized as steep
    
    return gdf