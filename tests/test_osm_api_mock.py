from pathlib import Path
import json
import pytest
import requests

from scripts.osm_api import get_overpass

class TestMockOverpassAPI:
    """
    Uses simulated API responses to test Overpass API response handling.
    """

    def test_overpass_post(self, mocker):
        """
        Tests if the program correctly handles a successful
        Overpass API response.
        """

        file_path = Path(__file__).parent / "data" / "mock_overpass_query.json"

        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Create fake API response
        mock_response = mocker.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = data

        # Mock requests.post()
        mock_post = mocker.patch('scripts.osm_api.requests.post')
        mock_post.return_value = mock_response

        test_bbox = (
            '47.7341',
            '47.4810',
            '-122.2244',
            '-122.4597'
        )

        # Run the function
        result, status_code = get_overpass(bbox=test_bbox)

        # Check that the response was parsed correctly
        elements = result.get('elements', [])
        park = elements[0]

        assert park.get('tags')['name'] == "Stratton Commons"

        # Check that requests.post() was called correctly
        mock_post.assert_called_once()

        call_args = mock_post.call_args
        assert call_args.args[0] == "https://overpass-api.de/api/interpreter"
        assert 'data' in call_args.kwargs
        assert 'data' in call_args.kwargs['data']
        assert status_code == 200


    def test_bad_overpass_post(self, mocker):
        """
        Tests the case where there was an error response from the API.
        """

        mock_response = mocker.MagicMock()
        mock_response.status_code = 404
        mock_response.json.return_value = {'response': None}

        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "404 Client Error"
        )

        mock_post = mocker.patch('scripts.osm_api.requests.post')
        mock_post.return_value = mock_response

        test_bbox = (
            '47.7341',
            '47.4810',
            '-122.2244',
            '-122.4597'
        )

        with pytest.raises(RuntimeError):
            get_overpass(bbox=test_bbox)
