import pytest

from nlr_ghe_screen.api.osm_api import (
    get_overpass_no_splitting
)

@pytest.mark.live_api
class TestLiveOverpassAPI:
    """
    Uses actual API responses to test whether the API post function
    integrates with the external Overpass API and database.
    """

    def test_overpass_query(self):
        """
        Verifies that simple API call is returned and parsed correctly.
        """
        # Create a test bounding box whose query results are known
        # Coordinates: City of Golden, CO
        test_bbox = (
            '39.710000', # South
            '-105.250000', # West
            '39.790000', # North
            '-105.125000' # East
        )

        # Make live API call and assert that it responded successfully
        result = get_overpass_no_splitting(bbox=test_bbox)
        elements = result.get('elements', [])
        
        assert len(elements) > 0

    def test_empty_overpass_query(self):
        """
        Tests the scenario where a query does not return any results
        """
        # Create a test bounding box whose query results are known
        # Coordinates: Middle of the Pacific Ocean
        test_bbox = ('20.000', '154.875', '20.080', '155.000')

        # Make live API call and assert that it responded successfully
        result = get_overpass_no_splitting(bbox=test_bbox)
        elements = result.get('elements', [])

        assert len(elements) == 0