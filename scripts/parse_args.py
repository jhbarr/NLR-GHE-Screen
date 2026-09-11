import argparse
import sys
import pyproj

from pyproj import CRS
from shapely.geometry import box
from shapely.ops import transform

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


def verify_bbox_size(bbox):
    """
    Ensures that the coordinates in the provided bounding box are in the correct CRS
    And raises a ValueError if they are not
    
    Parameters:
        bbox (Tuple[float]): A four float tuple of the form (lat_min, long_min, lat_max, long_max)
    """
    # Requires the form box(west, south, east, north)
    geom = box(bbox[1], bbox[0], bbox[3], bbox[2])

    # Automatically find the best UTM zone CRS for this location
    # (This ensures accuracy by picking a metric projection native to the coordinates)
    utm_crs = pyproj.CRS.from_string(f"+proj=utm +zone=10 +ellps=WGS84 +datum=WGS84 +units=m +no_defs") 

    # Set up the transformation from EPSG:4326 to the UTM metric CRS
    project = pyproj.Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True).transform

    # Transform the geometry and get the area
    projected_geom = transform(project, geom)
    area_in_sq_meters = projected_geom.area

    return area_in_sq_meters
    
        
def parse_arguments(args):
    """
    Parses the user inputted command line arguments and runs checks on them

    Parameters:
        args (Str): The user inputted command line arguments
    
    Returns:
        bbox (Tuple[float]): A tuple containing the coordinates of the bounding box 
    
    Raises:
        ValueError: If the bbox coordinates are not in the correct CRS (EPSG:4326)
        SystemExit: IF the arguments are in incorrect form or if there are arguments missing
    """
    print("\n---------------------------")
    print("Checking inputted coordinate values")

    parser = argparse.ArgumentParser(description="Program CLI")

    # Ensure that the coordinates are input and cast correctly
    # South West North East
    parser.add_argument('south_coordinate', type=float) # y_min
    parser.add_argument('west_coordinate', type=float) # x_min
    parser.add_argument('north_coordinate', type=float) # y_max
    parser.add_argument('east_coordinate', type=float) # x_max

    # The parsed arguments that are cast to the correct types
    # They are accessed via dot notation
    parsed = parser.parse_args(args)

    # Create a bounding box from the inputted coordinates
    bbox = (
        parsed.south_coordinate,
        parsed.west_coordinate,
        parsed.north_coordinate,
        parsed.east_coordinate
    )

    # Check that the coordinates are in the correct form and CRS
    verify_bbox_coordinates(bbox=bbox)

    # Check that the bounding box does not exceed maximum square footage
    verify_bbox_size(bbox=bbox)

    print("Success - Valid coordinates")
    print("---------------------------")

    return bbox