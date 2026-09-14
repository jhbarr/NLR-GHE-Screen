import json
import geopandas as gpd
from jsonschema import validate
from pathlib import Path

def validate_geojson(data, is_dataframe=False):
    """
    Takes in data in the form of a JSON object (dict) and validates that it correctly matches a preset
    GeoJSON schema (from URBANopt requirements)
    
    Parameters:
        data (dict): The data that is to be parsed into the GeoJSON schema
    
    Raises:
        ValidationError: If the resultant data does not fit the GeoJSON format
    """
    print("\n---------------------------")
    print("Validating GeoJSON")

    # Open the URBANopt GeoJSON schema
    file_path = Path(__file__).parent / "urbanopt_schema.json"
    with open(file_path, "r", encoding="utf-8") as file:
        geojson_schema = json.load(file)

    # Convert the dataframe into GeoJSON format
    geojson_dict = json.loads(data.to_json(na="null"))

    # Validate that each of the required features are there 
    for feature in geojson_dict["features"]:
        validate(
            instance=feature["properties"],
            schema=geojson_schema
        )
    
    print("Schema enforcement passed - Data is valid")
    print("---------------------------") 