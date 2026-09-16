import argparse
import geopandas as gpd
import pyproj
from pathlib import Path

from pyproj import CRS
from shapely.geometry import box
from shapely.ops import transform

# ---------------------------------------------------------------------------
# Verification 
# ---------------------------------------------------------------------------

def verify_bbox_coordinates(bbox):
    """
    Ensures that the coordinates in the provided bounding box are in the correct CRS
    And raises a ValueError if they are not
    
    Parameters:
        bbox (Tuple[float]): A four float tuple of the form (lat_min, long_min, lat_max, long_max)
    
    Raises:
        ValueError: If the bbox coordinates are not in the correct CRS (EPSG:4326)
    """
    # Check that the coordinates are in the correct CRS
    target_crs = CRS.from_epsg(4326)  # or "EPSG:4326"
    x_min, y_min, x_max, y_max = target_crs.area_of_use.bounds

    is_valid = (
        (bbox[0] >= y_min and bbox[0] <= y_max)
        and (bbox[1] >= x_min and bbox[1] <= x_max)
        and (bbox[2] <= y_max and bbox[2] >= y_min)
        and (bbox[3] <= x_max and bbox[3] >= x_min)
        and (bbox[0] <= bbox[2])
        and (bbox[1] <= bbox[3])
    )

    # If the coordinates do not follow the preceding rules, then they are in 
    # an incorrect form
    if not is_valid:
        raise ValueError("Invalid coordinates for EPSG:4326")



# ---------------------------------------------------------------------------
# Argument Parsing 
# ---------------------------------------------------------------------------  
        
def parse_arguments(args):
    """
    Parses the user inputted command line arguments and runs checks on them

    Parameters:
        args (Str): The user inputted command line arguments
    
    Returns:
        bbox (Tuple[float]): A tuple containing the coordinates of the bounding box 
    
    Raises:
        ValueError: If the bbox coordinates are not in the correct CRS (EPSG:4326)
        SystemExit: If the arguments are in incorrect form or if there are arguments missing
        FileNotFound: If the user's input file cannot be found
        DataSourceError: If the user's input file is is not a proper spatial file
    """
    parser = argparse.ArgumentParser(description="Program CLI")
    group = parser.add_mutually_exclusive_group(required=True)

    # Describe the format in which the user can input a series of coords
    # using the --bbox flag
    group.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("south", "west", "north", "east"),
        help="Bounding box coordinates: south west north east"
    )

    # Describe the format in which the user can input an URBANopt GeoJSON file
    # using the --file flag
    group.add_argument(
        "--file",
        type=str,
        help="Path to a file containing geometries"
    )

    # The parsed arguments that are cast to the correct types
    # They are accessed via dot notation
    parsed = parser.parse_args(args)

    if parsed.bbox is not None:
        print("\n---------------------------")
        print("Checking inputted coordinate values")

        # Retrieve the bounding box coordinates from the parsed arguments
        bbox = tuple(parsed.bbox)

        verify_bbox_coordinates(bbox=bbox)

        print("Success - Valid coordinates")
        print("---------------------------")

        return bbox

    elif parsed.file is not None:
        print("\n---------------------------")
        print("Checking inputted file")

        # Ensure that file exists
        file_path = Path(parsed.file)
        if not file_path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {parsed.file}")

        # Open the proviided file 
        # And check that it meets the URBANopt GeoJSON standards
        gdf = gpd.read_file(parsed.file) # Will raise DataSourceError if file is not proper spatial file
        bounding_polygon = gdf.geometry.union_all().convex_hull

        # Retrieve the bounding box coordinates that encompass all buildings
        west, south, east, north = bounding_polygon.bounds
        bbox = (south, west, north, east)

        verify_bbox_coordinates(bbox=bbox)

        print("Success - Valid file")
        print("---------------------------")

        return bbox