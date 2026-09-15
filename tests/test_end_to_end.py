import pytest
import json
from pathlib import Path

from scripts.parse_args import parse_arguments
from scripts.osm_api import get_overpass
from scripts.parse_geojson import parse_api_response
from scripts.validate_geojson import validate_geojson
from scripts.geometry_manipulation import combine_geometries

class TestEnd2End:
    """
    This class tests different scenarios of the data flow through the entire program structure
    """

    def test_mock(self, mocker):
        """
        Test the end to end flow using a mock API call
        """
        # 1. 
        # Create test argument bounding box
        # Coordinates for Golden, CO
        test_args = [
            '--bbox',
            "39.710000",
            "-105.250000",
            "39.790000",
            "-105.125000"
        ]

        # Parse arguments into bounding box tuple
        bbox = parse_arguments(test_args)
        # ** Raises ValueError - coordinates not in correct CRS **
        # ** Raises SystemExit - if there are missing arguments **

        # 2. 
        # Pass bounding box argument to mock API call
        file_path = Path(__file__).parent / "Data" / "mock_overpass_query.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Create fake API response
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = data

        # Mock requests.post()
        mock_post = mocker.patch('scripts.osm_api.requests.post')
        mock_post.return_value = mock_response

        result, status_code = get_overpass(bbox=bbox)
        # ** Raises RuntimeError - If the API does not return a 200 status code or any other error **

        # 3. 
        # Retrieve API response data and parse it into the correct form
        gdf = parse_api_response(result)
        # ** Raises ValueError - If the API response is empty or NUll ** 

        # 4. 
        # Validate that the parsed API response correctly fits the URBANopt GeoJSON schema
        validate_geojson(data=gdf, is_dataframe=True)
        # ** Raises ValidationError - If the given API response does not fit the schema ** 

        # 5. 
        # Combine geometries into new GeoDataframe
        gdf = combine_geometries(df=gdf)


    def test_live(self):
        """
        Test the end to end flow using a live API call
        """
        test_args = [
            '--bbox',
            "39.710000",
            "-105.250000",
            "39.790000",
            "-105.125000"
        ]

        bbox = parse_arguments(test_args)

        result, status_code = get_overpass(bbox=bbox)

        gdf = parse_api_response(result)

        validate_geojson(data=gdf, is_dataframe=True)

        gdf = combine_geometries(df=gdf)
       