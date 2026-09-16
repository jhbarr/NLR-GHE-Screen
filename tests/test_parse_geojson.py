from pathlib import Path
import pytest
import json

from scripts.parse_geojson import (
    parse_api_response, 
    create_geometry_object
)



# ---------------------------------------------------------------------------
# Test - Create geometry from JSON
# ---------------------------------------------------------------------------

class TestGeometryParse:
    """
    This class tests the functionality of process of parsing different kinds of OSM geometries inclusing ways and relations
    """

    def test_way_geometry(self):
        """
        Test that the function correctly parses out the way geometry from the test data
        """
        # Load in test data
        file_path = Path(__file__).parent / "data" / "parse_test_data.json"      
        with open(file_path, 'r', encoding='utf-8') as file:
            test_data = json.load(file)

        # Run the parsing function on a known way geometry
        way_geometry = test_data['elements'][0]
        geometry = create_geometry_object(way_geometry)

        assert geometry != None

    def test_relation_geometry(self):
            """
            Test that the function correctly parses out the relation geometry from the test data
            """
            # Load in test data
            file_path = Path(__file__).parent / "data" / "parse_test_data.json"
            with open(file_path, 'r', encoding='utf-8') as file:
                test_data = json.load(file)

            # Run the parsing function on a known relation geometry
            relation_geometry = test_data['elements'][1]
            geometry = create_geometry_object(relation_geometry)

            assert geometry != None

    def test_unrecognized_geometry(self):
        """
        Test the handling of the case where the function encounters a geometry it is not designed to handle
        """
        test_data = {
             "type": "node",
             "geometry": [
                  { "lat": 39.7513176, "lon": -105.2222512 }
             ]
        }

        with pytest.raises(TypeError):
             create_geometry_object(test_data)



# ---------------------------------------------------------------------------
# Test - API Parse
# ---------------------------------------------------------------------------

class TestAPIParse:
    """
    This class tests whether the method to parse the Overpass API response data into a GeoDataframe
    does so correctly and includes all of the necessary fields 
    """

    def test_parse_api_response(self):
        """
        Verify that all desired fields are a part of the parsed Overpass API response
        """
        file_path = Path(__file__).parent / "data" / "mock_overpass_query.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            test_data = json.load(file)

        gdf = parse_api_response(test_data)

        assert len(gdf) > 0

    def test_parse_empty_api_response(self):
        """
        Verify correct error is thrown if an API response does not return any data
        """
        empty_response = {
            "version": 0.6,
            "generator": "Overpass API 0.7.62 ...",
            "osm3s": {
                "timestamp_osm_base": "2026-09-11T14:30:00Z",
                "copyright": "The data included in this document is from www.openstreetmap.org. The data is made available under ODbL."
            },
            "elements": []
        }

        with pytest.raises(ValueError):
            gdf = parse_api_response(empty_response)