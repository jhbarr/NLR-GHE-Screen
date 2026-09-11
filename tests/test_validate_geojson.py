from jsonschema import ValidationError
import pytest
from shapely.geometry import Polygon
import geopandas as gpd

from scripts.validate_geojson import validate_geojson

class TestGeoJSONValidation:
    """
    This tests the functionality of validating whether data queried and constructed by the program
    correctly fits the predetermined URBANopt GeoJSON schema standard
    """

    def test_schema_valid_data(self):
        """
        Verify the functionality of the geojson parse handling
        """
        # Create example data
        valid_data = {
            "type": ["District System"],
            "district_system_type": ["Central Hot Water"],
            "geometry": [
                Polygon([
                    (0, 0),
                    (1, 0),
                    (1, 1),
                    (0, 0)
                ])
            ]
        }

        # Verify that the data passes the validation 
        gdf = gpd.GeoDataFrame(valid_data, crs="EPSG:4326")
        validate_geojson(gdf)

    def test_schema_valid_data_additional(self):
            """
            Verify the functionality of the geojson parse handling when given data with all required properties 
            and some extra ones
            """
            # Create example data
            valid_data = {
                "type": ["District System"],
                "district_system_type": ["Central Hot Water"],
                "landuse": ["park"], # EXTRA FIELD
                "geometry": [
                    Polygon([
                        (0, 0),
                        (1, 0),
                        (1, 1),
                        (0, 0)
                    ])
                ]
            }
    
            # Verify that the data passes the validation
            gdf = gpd.GeoDataFrame(valid_data, crs="EPSG:4326")
            validate_geojson(gdf)

    def test_schema_missing_requirement(self):
        """
        Verifies that when the data does not fit the schema, the correcet error is raised
        """
        # Create example data
        # Missing required field - district_system_type
        invalid_data = {
        "type": ["District System"],
        "geometry": [
            Polygon([
                (0, 0),
                (1, 0),
                (1, 1),
                (0, 0)
            ])
        ]
    }

        with pytest.raises(ValidationError):
            gdf = gpd.GeoDataFrame(invalid_data, crs="EPSG:4326")
            validate_geojson(gdf)


    def test_schema_invalid_type(self):
        """
        Verifies that when an invalid type is given to the district_system_type is provided,
        that the function correctly throws an error
        """
        # Create example data
        invalid_data = {
            "type": ["District System"],
            "district_system_type": ["Nothing"],
            "geometry": [
                Polygon([
                    (0, 0),
                    (1, 0),
                    (1, 1),
                    (0, 0)
                ])
            ]
        }

        with pytest.raises(ValidationError):
            gdf = gpd.GeoDataFrame(invalid_data, crs="EPSG:4326")
            validate_geojson(gdf)