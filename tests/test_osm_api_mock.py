from pathlib import Path
import json
 
import pytest
import requests
 
from scripts.osm_api import (
    build_overpass_query,
    run_overpass_query,
    get_overpass,
    get_count,
    OverpassError,
    OverpassTooLargeError,
    OverpassRateLimitedError,
    OverpassServerBusyError,
    OverpassBadQueryError,
    OverpassConnectionError,
)

TEST_BBOX = (47.4810, -122.4597, 47.7341, -122.2244)  # south, west, north, east



# ---------------------------------------------------------------------------
# Helpers
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


@pytest.fixture(autouse=True)
def no_real_sleep(mocker):
    """Prevent tests from actually waiting through backoff delays."""
    return mocker.patch("scripts.osm_api.time.sleep")



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
        data = load_mock_data()
        response = make_response(mocker, status_code=200, json_data=data)
        mock_post = mocker.patch("scripts.osm_api.requests.post", return_value=response)
 
        result = get_overpass(bbox=TEST_BBOX)
 
        elements = result.get("elements", [])
        assert elements[0]["tags"]["name"] == "Stratton Commons"
 
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.args[0] == "https://overpass-api.de/api/interpreter"
        assert "data" in call_args.kwargs["data"]

    def test_get_count_parses_total(self, mocker):
        count_data = {
            "elements": [
                {"type": "count", "id": 0,
                 "tags": {"nodes": "12", "ways": "34", "relations": "1", "total": "47"}}
            ]
        }
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
        mock_post = mocker.patch(
            "scripts.osm_api.requests.post",
            side_effect=requests.exceptions.Timeout("timed out"),
        )
 
        with pytest.raises(OverpassTooLargeError):
            get_overpass(bbox=TEST_BBOX)

        mock_post.assert_called_once()

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