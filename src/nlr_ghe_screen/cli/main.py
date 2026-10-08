import sys
import json
from pathlib import Path
import os
from dotenv import load_dotenv
from jsonschema import ValidationError
from pyogrio.errors import DataSourceError

from nlr_ghe_screen.cli.parse_args import parse_arguments

from nlr_ghe_screen.api.otp_api import get_opentop, OpentopError
from nlr_ghe_screen.api.osmnx_api import get_osmnx, OXError
from nlr_ghe_screen.api.osm_api import get_overpass_with_splitting, OverpassError

from nlr_ghe_screen.geojson.parse_geojson import parse_api_response
from nlr_ghe_screen.geojson.validate_geojson import validate_geojson

from nlr_ghe_screen.processing.geometry import combine_geometries, classify_geometries
from nlr_ghe_screen.processing.ms_cross_validation import footprints_for_points
from nlr_ghe_screen.processing.elevation import categorize_steepness

# Load the environment variables
load_dotenv()


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def aggregate_metadata(df):
    """Aggregates metadata about the total area covered by greenspace and parking lots in ``df``

    Initially categorizes the geometries in the provided ``df`` based on their atrributes into either the
    ``greenspace`` category or the ``parking`` category. It then calculates the total area covered by all
    geometries in each category in km^2. 

    It returns this information in a dictionary

    Args:
        df (geopandas.GeoDataframe): The Dataframe that holds all of the information about queried OSM data
    
    Returns:
        dict: Dictionary containing all descriptive metadata
    """
    categories = ['waste_heat', 'water', 'parking', 'green_space']

    # Add the categories to the rows to which they apply
    df = df.copy()
    df['_category'] = classify_geometries(df)
    crs = df.estimate_utm_crs()

    metadata = {}

    for category in categories:
        # Get all of the row entries that were assigned this category 
        subset = df[df['_category'] == category].drop(columns='_category')
        if subset.empty:
            continue

        subset = subset.to_crs(crs)
        metadata["total " + category] = f"{round(subset.area.sum() / 1_000_000, 2)} km^2"

    return metadata


def export_results(df, response):
    """Export all information and descriptive metadata compiled by the entire program workflow

    Three main files are exported to the local ``src/exports`` folder. The three files are:

        1. ``ghe_locations.geojson``: Includes the geometries of each of the compiled greenspaces and parking lots
            that are considered locations for possible GHE installation sites. Additionally, it contains all of the 
            OSM attributes of each geometry
        2. ``ghe_location_metadata.json``: Contains the total area covered by greenspaces and parking lots in km^2
        3. ``overpass_api_metadata``: Metadata embedded in the Overpass API response. Record needs to be kept for proper
            credit attribution in research papers. 

    Args:
        df (geopandas.GeoDataframe): The GeoDataframe that includes the final compiled GHE location data
        response (dict): The Overpass API JSON response body
    """
    export_dir = Path(__file__).parent.parent / ".." / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Export geometry data to the necessary file
    file_path = export_dir / 'ghe_locations.geojson'
    df.to_file(file_path, driver="GeoJSON")

    # Export geometry metadata to the necessary file
    file_path = export_dir / 'ghe_location_metadata.json'
    with open(file_path, "w", encoding="utf-8") as file:
        metadata = aggregate_metadata(df=df)
        json.dump(metadata, file, indent=4)

    # Isolate and export the metadata regarding the Overpass API query 
    file_path = export_dir / 'overpass_api_metadata.json'
    with open(file_path, "w", encoding="utf-8") as file:
        keys = {'version', 'generator', 'osm3s'}
        query_metadata = {key: response[key] for key in keys if key in response}
        json.dump(query_metadata, file, indent=4)


# ---------------------------------------------------------------------------
# Execution Functions
# ---------------------------------------------------------------------------

def run(args):
    """
    This function is responsible for executing the entire end to end flow of the program and all of 
    the internal functions that handle each step of the API calling, parsing, exportation process

    Args:
        args (list[Str]): The arguments to the program
    """
    bbox = parse_arguments(args)

    result = get_overpass_with_splitting(bbox=bbox)

    gdf = parse_api_response(data=result, bbox=bbox, is_df=False)

    validate_geojson(data=gdf)

    gdf = footprints_for_points(gdf=gdf, bbox=bbox)

    gdf['category'] = classify_geometries(gdf=gdf) # ** Optional Step **

    get_opentop(bbox=bbox, api_key=os.getenv("API_KEY"))

    categorize_steepness(gdf=gdf)

    export_results(df=gdf, response=result)


def main(args):
    """
    Executes the main functionality of the program

    Parameters:
        args (list[Str]): The arguments to the program
    """

    try:
        run(args)

    except SystemExit:
        print(f"\nError: Incorrect arguments supplied")

    except (
        ValueError,
        ValidationError,
        TypeError,
        FileNotFoundError,
        DataSourceError,
        OverpassError,
        OpentopError,
        OXError
    ) as exc:
         print(f"Error - {exc}")



# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    main(args=sys.argv[1:])