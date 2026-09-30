import json
from jsonschema import validate
from pathlib import Path


# ---------------------------------------------------------------------------
# GeoJSON Validation 
# ---------------------------------------------------------------------------

def validate_geojson(data):
    """Validates that all of the fields required by URBANopt GeoJSON standards are present

    Loads in URBANopt GeoJSON schema ``urbanopt_schema.json`` from the local ``data/`` folder. Then 
    goes through each row in the provided GeoDataframe ``data`` and ensures that they contain the fields that 
    are labeled as required by the schema. 

    The function raises a jsonschema.ValidationError if the ``data`` does not fit the schema. Otherwise, there
    is no return value
    
    Args:
        data (geopandas.GeoDataframe): Contains geometry vectors and their corresponding OSM tags. 
            To be cross checked against the URBNANopt GeoJSON schema
    
    Raises:
        ValidationError: If the resultant data does not fit the GeoJSON format
    """
    print("\n---------------------------")
    print("Validating GeoJSON")

    # Open the URBANopt GeoJSON schema
    file_path = Path(__file__).parent.parent / "data" / "urbanopt_schema.json"
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