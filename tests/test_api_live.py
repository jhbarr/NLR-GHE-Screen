import pytest

from scripts.osm_api import get_overpass

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
        result, status_code = get_overpass(bbox=test_bbox)
        elements = result.get('elements', [])
        park = elements[0]
        
        assert status_code == 200
        assert elements
        assert park.get('tags')['name'] == "Stratton Commons"

    def test_empty_overpass_query(self):
        """
        Tests the scenario where a query does not return any results
        """
        # Create a test bounding box whose query results are known
        # Coordinates: Middle of the Pacific Ocean
        test_bbox = (
            '20.000', # South
            '155.000', # West
            '20.080', # North
            '154.875' # East
        )

        # Make live API call and assert that it responded successfully
        result, status_code = get_overpass(bbox=test_bbox)
        elements = result.get('elements', [])

        assert not elements
        assert status_code == 200