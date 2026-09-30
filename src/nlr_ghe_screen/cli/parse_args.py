import argparse
import geopandas as gpd
from pathlib import Path
from pyproj import CRS

# ---------------------------------------------------------------------------
# Verification 
# ---------------------------------------------------------------------------

def verify_bbox_coordinates(bbox):
    """Ensures that the coorinates in the ``bbox`` are in the correect CRS: ``EPSG:4326``

    Unpacks the coordinates from ``bbox`` and checks that they are in the correct ranges for proper coordinate
    degrees. The function raises a ValueError if the coordinates do not pass all of these checks. Otherwise, 
    nothing is returned.
    
    Args:
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
    """Parse and validate the command line arguments, returning a bounding box

    The user musts provide exactly one of two mututally exclusive options:

        - ``--bbox SOUTH WEST NORTH EAST``: Four coordinates, in EPSG:4326 (decimal
          degrees), describing the corners of the area's bounding box.
        - ``--file PATH``: The path to an URBANopt GeoJSON file. The file must
          contain exactly one feature whose ``type`` property is ``"bounding box"``,
          for example::
 
              {
                  "type": "Feature",
                  "properties": {"type": "bounding box"},
                  "geometry": {"type": "Polygon", "coordinates": []}
              }
 
          The bounding box is taken from the extent of that feature's geometry.
    
    In both cases, the resulting coordinates are checked with :func:`verify_bbox_coordinates`` before being returned. 

    Args:
        args (list[str]): The raw command line arguments, without the program name
            (e.g. ``sys.argv[1:]``).
 
    Returns:
        tuple[float, float, float, float]: The bounding box as
            ``(south, west, north, east)``.
 
    Raises:
        SystemExit: If the arguments are malformed, if neither or both of ``--bbox``
            and ``--file`` are given, or if ``--help`` is requested (raised by
            ``argparse``).
        FileNotFoundError: If the path given with ``--file`` does not point to an
            existing file.
        pyogrio.errors.DataSourceError: If the file given with ``--file`` cannot be
            read as a spatial file.
        ValueError: If the input file does not contain exactly one ``bounding box``
            feature, or if the coordinates fail validation in
            :func:`verify_bbox_coordinates` (e.g. they are not valid EPSG:4326
            values).
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

        # Open the provided file 
        # And check that it meets the URBANopt GeoJSON standards
        gdf = gpd.read_file(parsed.file) # Will raise DataSourceError if file is not proper spatial file

        # Check that there is a type column
        if 'type' not in gdf.columns:
            raise ValueError("No such column 'type' in gdf")

        # Retrieve the coordinates from the GeoJSON file
        matches = gdf.loc[gdf['type'] == 'bounding box']
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one bounding box, found: {len(matches)}")
        west, south, east, north = matches.bounds.iloc[0]

        # Retrieve the bounding box coordinates that encompass all buildings
        bbox = (south, west, north, east)

        verify_bbox_coordinates(bbox=bbox)

        print("Success - Valid file")
        print("---------------------------")

        return bbox