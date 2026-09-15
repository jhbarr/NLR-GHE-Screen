import sys
import json
from pathlib import Path
from jsonschema import ValidationError
from pyogrio.errors import DataSourceError

from parse_args import parse_arguments
from osm_api import get_overpass
from parse_geojson import parse_api_response
from validate_geojson import validate_geojson
from geometry_manipulation import combine_geometries, classify_geometry

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

def main(args):
    """
    This function is responsible for executing the entire end to end flow of the program and all of 
    the internal functions that handle each step of the API calling, parsing, exportation process

    Parameters:
        args (list[Str]): The arguments to the program
    """

    try:
        bbox = parse_arguments(args)
        # ** Raises ValueError - coordinates not in correct CRS **
        # ** Raises SystemExit - if there are missing arguments **
        # ** Raises FileNotFoundError - if the input file cannot be found ** 
        # ** Raises DataSourceError - if input file is not valid spatial file ** 

        file_path = Path(__file__).parent.parent / "Tests" / "Data" / "golden_query.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            result = json.load(file)

        # result, status_code = get_overpass(bbox=bbox)
        # # ** Raises RuntimeError - If the API does not return a 200 status code or any other error **

        gdf = parse_api_response(data=result)
        # ** Raises ValueError - If the API response is empty or NUll ** 

        validate_geojson(
            data=gdf,
            is_dataframe=True
        )
        # ** Raises jsonschema.ValidationError - If the given API response does not fit the schema **

        gdf = combine_geometries(df=gdf)

        # Export all files to the export folder 
        gdf.to_file("../Exports/ghe_locations.geojson", driver="GeoJSON")
        with open("../Exports/ghe_loction_metadata.json", "w", encoding="utf-8") as file:
            metadata = aggregate_metadata(df=gdf)
            json.dump(metadata, file, indent=4)

        # Isolate and export the metadata regarding the Overpass API query 
        with open("../Exports/overpass_api_metadata.json", "w", encoding="utf-8") as file:
            keys = {'version', 'generator', 'osm3s'}
            query_metadata = {key: result[key] for key in keys if key in result}
            json.dump(query_metadata, file, indent=4)


    except SystemExit as exec:
        print(f"\nError: Incorrect arguments supplied")

    except ValueError as exec:
        print(f"\nError - {exec}")

    except RuntimeError as exec:
        print(f"\nError - {exec}")

    except ValidationError as exec:
        print(f"Error - {exec}")

    except TypeError as exec:
        print(f"Error - {exec}")

    except FileNotFoundError as exec:
            print(f"Error - {exec}")

    except DataSourceError as exec:
                print(f"Error - {exec}")

# Execute the program
if __name__ == '__main__':
    args = sys.argv[1:]
    main(args=args)