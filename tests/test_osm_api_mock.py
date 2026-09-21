from pathlib import Path
import json
import pytest
import requests

# Import API functions
from scripts.osm_api import (
    build_overpass_query,
    run_overpass_query,
    get_overpass,
    get_count,
    split_bbox,
)

# Import special API errors
from scripts.osm_api import (
    OverpassError,
    OverpassTooLargeError,
    OverpassRateLimitedError,
    OverpassServerBusyError,
    OverpassBadQueryError,
    OverpassConnectionError,
    OverpassTimeoutError,
)

TEST_BBOX = (47.4810, -122.4597, 47.7341, -122.2244)  # south, west, north, east


# ---------------------------------------------------------------------------
# Helper functions - Loading mock data
# ---------------------------------------------------------------------------

def make_response(mocker, status_code=200, json_data=None, text="", json_error=False):
    """
    Build a MagicMock standing in for a requests.Response
    """
    response = mocker.MagicMock()
    response.status_code = status_code
    response.text = text
    if json_error:
        response.json.side_effect = requests.exceptions.JSONDecodeError("bad json", "", 0)
    else:
        response.json.return_value = json_data if json_data is not None else {}
    return response

def load_mock_data():
    file_path = Path(__file__).parent / "data" / "mock_overpass_query.json"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_mock_count_data():
    count_data = {
        "elements": [
            {"type": "count", "id": 0,
                "tags": {"nodes": "12", "ways": "34", "relations": "1", "total": "47"}}
        ]
    }
    return count_data


@pytest.fixture(autouse=True)
def no_real_sleep(mocker):
    """Prevent tests from actually waiting through backoff delays."""
    return mocker.patch("scripts.osm_api.time.sleep")



# ---------------------------------------------------------------------------
# Helper functions - Splitting bbox information
# ---------------------------------------------------------------------------

def make_element(el_id, el_type="way"):
    """Minimal Overpass element."""
    return {"type": el_type, "id": el_id, "tags": {"name": f"element-{el_id}"}}
 
 
def make_data(*elements, **meta):
    """Minimal Overpass data response body."""
    return {"generator": "mock", "elements": list(elements), **meta}

def bbox_str(bbox):
    """The bbox exactly as build_overpass_query interpolates it."""
    south, west, north, east = bbox
    return f"{south},{west},{north},{east}"


# ---------------------------------------------------------------------------
# Test - Query construction
# ---------------------------------------------------------------------------

class TestBuildOverpassQuery:
    """
    Test the different cases for the construction of the Overpass QL text
    """
 
    def test_data_mode_uses_out_geom(self):
        query = build_overpass_query(TEST_BBOX, mode="data")
        assert "out geom;" in query
        assert "out count;" not in query
 
    def test_count_mode_uses_out_count(self):
        query = build_overpass_query(TEST_BBOX, mode="count")
        assert "out count;" in query
        assert "out geom;" not in query
 
    def test_bbox_values_interpolated(self):
        query = build_overpass_query(TEST_BBOX, mode="data")
        south, west, north, east = TEST_BBOX
        assert f"{south},{west},{north},{east}" in query
 
    def test_timeout_and_maxsize_interpolated(self):
        query = build_overpass_query(TEST_BBOX, mode="data", query_timeout=42)
        assert "[timeout:42]" in query



# ---------------------------------------------------------------------------
# Test - Successful requests
# ---------------------------------------------------------------------------

class TestSuccessfulRequests:
    """
    Verify correct behavior upon successful response from Overpass API
    """
 
    def test_get_overpass_success(self, mocker):
        count_data = load_mock_count_data()
        response_data = load_mock_data()

        response_1 = make_response(mocker, status_code=200, json_data=count_data)
        response_2 = make_response(mocker, status_code=200, json_data=response_data)

        mock_post = mocker.patch(
            "scripts.osm_api.requests.post",
            side_effect = [response_1, response_2]
        )

        result = get_overpass(bbox=TEST_BBOX)

        elements = result.get("elements", [])
        assert elements[0]["tags"]["name"] == "Stratton Commons"

        assert mock_post.call_count == 2

        # First API call
        first_call = mock_post.call_args_list[0]
        assert first_call.args[0] == "https://overpass-api.de/api/interpreter"
        assert "data" in first_call.kwargs["data"]

        # Second API call
        first_call = mock_post.call_args_list[0]
        assert first_call.args[0] == "https://overpass-api.de/api/interpreter"

    def test_get_count_parses_total(self, mocker):
        count_data = load_mock_count_data()
        response = make_response(mocker, status_code=200, json_data=count_data)
        mocker.patch("scripts.osm_api.requests.post", return_value=response)
 
        count = get_count(bbox=TEST_BBOX)
 
        assert count == 47
        assert isinstance(count, int)
 
    def test_get_count_handles_empty_elements(self, mocker):
        response = make_response(mocker, status_code=200, json_data={"elements": []})
        mocker.patch("scripts.osm_api.requests.post", return_value=response)
 
        count = get_count(bbox=TEST_BBOX)
 
        assert count == 0



# ---------------------------------------------------------------------------
# Test - HTTP-level error classification
# ---------------------------------------------------------------------------

class TestHTTPErrorClassification:
    """
    Test the various possible HTTP response error codes and their respective OverpassError throws
    """

    def test_400_raises_bad_query_error(self, mocker):
        response = make_response(mocker, status_code=400, text="Query parse error near line 3")
        mocker.patch("scripts.osm_api.requests.post", return_value=response)
 
        with pytest.raises(OverpassBadQueryError):
            get_overpass(bbox=TEST_BBOX)

    def test_400_does_not_retry(self, mocker):
        response = make_response(mocker, status_code=400, text="bad query")
        mock_post = mocker.patch("scripts.osm_api.requests.post", return_value=response)
 
        with pytest.raises(OverpassBadQueryError):
            get_overpass(bbox=TEST_BBOX)
 
        mock_post.assert_called_once()

    @pytest.mark.parametrize("status", [502, 503, 504])
    def test_5xx_raises_server_busy_after_retries(self, mocker, status):
        response = make_response(mocker, status_code=status)
        mock_post = mocker.patch("scripts.osm_api.requests.post", return_value=response)
 
        with pytest.raises(OverpassServerBusyError):
            run_overpass_query("dummy query", max_retries=3)
 
        assert mock_post.call_count == 3
 
    def test_429_raises_rate_limited_after_retries(self, mocker):
        response = make_response(mocker, status_code=429)
        mock_post = mocker.patch("scripts.osm_api.requests.post", return_value=response)
 
        with pytest.raises(OverpassRateLimitedError):
            run_overpass_query("dummy query", max_retries=3)
 
        assert mock_post.call_count == 3
 
    def test_unexpected_status_raises_generic_overpass_error(self, mocker):
        response = make_response(mocker, status_code=418)
        mocker.patch("scripts.osm_api.requests.post", return_value=response)
 
        with pytest.raises(OverpassError):
            run_overpass_query("dummy query", max_retries=1)



# ---------------------------------------------------------------------------
# Overpass-embedded errors on HTTP 200
# ---------------------------------------------------------------------------

class TestEmbeddedRemarkErrors:
    """
    Tests to verify that the program correctly identifies cases where an error is embedded in a response
    that was marked as 200-success
    """

    def test_timeout_remark_raises_too_large(self, mocker):
        body = {"remark": "runtime error: Query timed out in \"query\" at line 5"}
        response = make_response(mocker, status_code=200, json_data=body)
        mocker.patch("scripts.osm_api.requests.post", return_value=response)
 
        with pytest.raises(OverpassTooLargeError):
            get_overpass(bbox=TEST_BBOX)

    def test_no_remark_present_parses_normally(self, mocker):
        data = load_mock_data()
        assert "remark" not in data  # sanity check on the fixture
        response = make_response(mocker, status_code=200, json_data=data)
        mocker.patch("scripts.osm_api.requests.post", return_value=response)
 
        result = get_overpass(bbox=TEST_BBOX)
        assert "elements" in result



# ---------------------------------------------------------------------------
# Client-side network errors
# ---------------------------------------------------------------------------
 
class TestClientSideErrors:
 
    def test_requests_timeout_raises_too_large_error(self, mocker):
        count_data = load_mock_count_data()
        response = make_response(mocker, status_code=200, json_data=count_data)

        mock_post = mocker.patch(
            "scripts.osm_api.requests.post",
            side_effect= [
                response,
                requests.exceptions.Timeout("timed out"),
            ]
        )

        with pytest.raises(OverpassTimeoutError):
            get_overpass(bbox=TEST_BBOX)

        assert mock_post.call_count == 2


    def test_connection_error_retries_then_raises(self, mocker):
        mock_post = mocker.patch(
            "scripts.osm_api.requests.post",
            side_effect=requests.exceptions.ConnectionError("refused"),
        )
 
        with pytest.raises(OverpassConnectionError):
            run_overpass_query("dummy query", max_retries=3)
 
        assert mock_post.call_count == 3



# ---------------------------------------------------------------------------
# Retry mechanics
# ---------------------------------------------------------------------------
 
class TestRetryMechanics:
    """
    Test that the program correctly retries and can get a valid response when the server is initially 
    too busy to handle the request
    """
 
    def test_succeeds_after_transient_failure(self, mocker):
        """First attempt hits 503, second attempt succeeds - should
        return normally without raising, and should have retried once."""
        data = load_mock_data()
        bad_response = make_response(mocker, status_code=503)
        good_response = make_response(mocker, status_code=200, json_data=data)
 
        mock_post = mocker.patch(
            "scripts.osm_api.requests.post",
            side_effect=[bad_response, good_response],
        )
 
        result = run_overpass_query("dummy query", max_retries=3)
 
        assert result["elements"][0]["tags"]["name"] == "Stratton Commons"
        assert mock_post.call_count == 2



# ---------------------------------------------------------------------------
# Test - Query Splitting Functionality
# ---------------------------------------------------------------------------

class TestQuerySplit:
    """
    Tests the mechanics of probing the API for query count and splitting a large query 
    """

    # ----- Split bbox functionality ---------------

    def test_split_bbox_halves_cover_original(self):
        """
        The two halves must tile the parent: no gap, no overlap, nothing outside
        """
        south, west, north, east = TEST_BBOX
        a, b = split_bbox(TEST_BBOX)
 
        # Together they span the whole parent
        assert min(a[0], b[0]) == south
        assert min(a[1], b[1]) == west
        assert max(a[2], b[2]) == north
        assert max(a[3], b[3]) == east
 
        # They share exactly one edge (lat split or lon split)
        lat_split = a[1] == b[1] and a[3] == b[3]
        lon_split = a[0] == b[0] and a[2] == b[2]
        assert lat_split != lon_split  # exactly one axis was split
        if lat_split:
            assert a[2] == b[0]
        else:
            assert a[3] == b[1]


    # ----- No split necessary ---------------
 
    def test_small_count_runs_single_data_query(self, mocker):
        mock_count = mocker.patch("scripts.osm_api.get_count", return_value=10)
        mock_run = mocker.patch(
            "scripts.osm_api.run_overpass_query",
            return_value=make_data(make_element(1), make_element(2)),
        )
 
        result = get_overpass(TEST_BBOX, max_elements=100)
 
        mock_count.assert_called_once_with(TEST_BBOX)
        mock_run.assert_called_once()
        assert bbox_str(TEST_BBOX) in mock_run.call_args.args[0]
        assert [e["id"] for e in result["elements"]] == [1, 2]


    # ----- Splitting ---------------

    def test_recursive_split_multiple_levels(self, mocker):
        """
        Full box too big, first half too big again, everything else fits
        """
        half_a, half_b = split_bbox(TEST_BBOX)
        quarter_aa, quarter_ab = split_bbox(half_a)
 
        counts = {
            TEST_BBOX: 5000,
            half_a: 800,
            quarter_aa: 40,
            quarter_ab: 40,
            half_b: 60,
        }
        mocker.patch("scripts.osm_api.get_count", side_effect=lambda bbox: counts[bbox])
 
        def fake_run(query, **kwargs):
            for bbox, el_id in ((quarter_aa, 1), (quarter_ab, 2), (half_b, 3)):
                if bbox_str(bbox) in query:
                    return make_data(make_element(el_id))
            raise AssertionError(f"Unexpected data query: {query}")
 
        mock_run = mocker.patch("scripts.osm_api.run_overpass_query", side_effect=fake_run)
 
        result = get_overpass(TEST_BBOX, max_elements=100)
 
        assert mock_run.call_count == 3
        assert sorted(e["id"] for e in result["elements"]) == [1, 2, 3]


    def test_data_query_too_large_error_triggers_split(self, mocker):
        """Count looked fine, but the real query still exceeded the budget."""
        mocker.patch("scripts.osm_api.get_count", return_value=10)
        mock_run = mocker.patch(
            "scripts.osm_api.run_overpass_query",
            side_effect=[
                OverpassTooLargeError("data query timed out"),  # full bbox
                make_data(make_element(1)),                     # half A
                make_data(make_element(2)),                     # half B
            ],
        )
 
        result = get_overpass(TEST_BBOX, max_elements=100)
 
        assert mock_run.call_count == 3
        assert {e["id"] for e in result["elements"]} == {1, 2}


    def test_duplicate_elements_across_halves_are_deduplicated(self, mocker):
        """A way crossing the split line is returned by both halves."""
        shared = make_element(99)
        mocker.patch("scripts.osm_api.get_count", side_effect=[5000, 50, 50])
        mocker.patch(
            "scripts.osm_api.run_overpass_query",
            side_effect=[
                make_data(make_element(1), shared),
                make_data(shared, make_element(2)),
            ],
        )
 
        result = get_overpass(TEST_BBOX, max_elements=100)
 
        ids = [e["id"] for e in result["elements"]]
        assert sorted(ids) == [1, 2, 99]
        assert ids.count(99) == 1


    # -- Depth limit and error propagation ----------------------------------
 
    def test_max_depth_exceeded_raises_too_large(self, mocker):
        mock_count = mocker.patch("scripts.osm_api.get_count", return_value=10**6)
        mock_run = mocker.patch("scripts.osm_api.run_overpass_query")
 
        with pytest.raises(OverpassTooLargeError):
            get_overpass(TEST_BBOX, max_elements=100, max_depth=2)
 
        # Depth 0 -> depth 1 -> depth 2, then the depth-2 box refuses to split
        assert mock_count.call_count == 3
        mock_run.assert_not_called()