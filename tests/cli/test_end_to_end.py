import pytest
import json
from pathlib import Path

from nlr_ghe_screen.cli.main import (
    run
)

class TestEnd2End:
    """
    This class tests different scenarios of the data flow through the entire program structure
    """

    # def test_mock(self, mocker):
    #     """
    #     Test the end to end flow using a mock API call
    #     """
    #     # 1. 
    #     # Create test argument bounding box
    #     # Coordinates for Golden, CO
    #     test_args = [
    #         '--bbox',
    #         "39.710000",
    #         "-105.250000",
    #         "39.790000",
    #         "-105.125000"
    #     ]

    #     # Pass bounding box argument to mock API call
    #     file_path = Path(__file__).parent / "cli_data" / "mock_overpass_query.json"
    #     with open(file_path, 'r', encoding='utf-8') as file:
    #         data = json.load(file)

    #     # Create fake API response
    #     mock_response = mocker.MagicMock()
    #     mock_response.status_code = 200
    #     mock_response.json.return_value = data

    #     # Mock requests.post()
    #     mock_post = mocker.patch('scripts.osm_api.requests.post')
    #     mock_post.return_value = mock_response

    #     run(test_args)


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

        run(test_args)