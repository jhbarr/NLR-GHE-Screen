import pytest
from dotenv import load_dotenv
import os
from pathlib import Path

from scripts.otp_api import get_opentop



TEST_BBOX = (47.4810, -122.4597, 47.7341, -122.2244)  # south, west, north, east

# Load the API key from the .env file
load_dotenv()
api_key = os.getenv("API_KEY")

@pytest.mark.live_api
class TestLiveOpenTopographyAPI:
    """
    Uses actual API reaponses to test whether the API funcitonality is working properly
    and integtates with external OpenTopography API
    """

    def test_opentop_query(self):
        get_opentop(bbox=TEST_BBOX, api_key=api_key)

        path_str = '../imports/elevation_data.tif'
        assert Path(path_str).is_file()