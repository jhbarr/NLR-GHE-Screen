import sys
import json
from pathlib import Path

from jsonschema import ValidationError
from pyogrio.errors import DataSourceError
from osm_api import OverpassError

from parse_args import parse_arguments
from osm_api import get_overpass
from parse_geojson import parse_api_response
from validate_geojson import validate_geojson
from geometry_manipulation import combine_geometries, classify_geometry


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def aggregate_metadata(df):
    """
    This function aggregates descriptive metadata about the queried areas 

    Parameters:
        df (GeoDataframe): The Dataframe that holds all of the information about queried OSM data
    
    Returns:
        metadata (dict): Dictionary containing all descriptive metadata
    """
    categories = ['green_space', 'parking']

    # Add the categories to the rows to which they apply
    df = df.copy()
    df['_category'] = df.apply(classify_geometry, axis=1)

    metadata = {}

    for category in categories:
        # Get all of the row entries that were assigned this category 
        subset = df[df['_category'] == category].drop(columns='_category')
        if subset.empty:
            continue

        subset = subset.to_crs(epsg=3857)
        metadata["total " + category] = subset.area.sum()

    return metadata


def export_results(df, response):
    """
    Exports all of the necessary information to the exports folder 

    Parameters:
        df (Geopandas Dataframe)
        response (dict): The Overpass API JSON response body
    """
    # Export all files to the export folder 
    df.to_file("../Exports/ghe_locations.geojson", driver="GeoJSON")
    with open("../Exports/ghe_loction_metadata.json", "w", encoding="utf-8") as file:
        metadata = aggregate_metadata(df=df)
        json.dump(metadata, file, indent=4)

    # Isolate and export the metadata regarding the Overpass API query 
    with open("../Exports/overpass_api_metadata.json", "w", encoding="utf-8") as file:
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

    Parameters:
        args (list[Str]): The arguments to the program
    """
    bbox = parse_arguments(args)

    file_path = Path(__file__).parent.parent / "tests" / "data" / "golden_query.json"
    with open(file_path, 'r', encoding='utf-8') as file:
        result = json.load(file)

    # result, status_code = get_overpass(bbox=bbox)

    gdf = parse_api_response(data=result)

    validate_geojson(data=gdf, is_dataframe=True)

    gdf = combine_geometries(df=gdf)

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
        OverpassError
    ) as exc:
         print(f"Error - {exc}")



# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = sys.argv[1:]
    main(args=args)