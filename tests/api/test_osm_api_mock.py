from pathlib import Path
import json
 
import pytest
import requests
 
from nlr_ghe_screen.api.osm_api import (
    build_overpass_query,
    run_overpass_query,
    get_overpass_with_splitting,
    get_overpass_no_splitting,
    split_bbox,
    OverpassError,
    OverpassTooLargeError,
    OverpassRateLimitedError,
    OverpassServerBusyError,
    OverpassBadQueryError,
    OverpassConnectionError,
    OverpassTimeoutError
)

TEST_BBOX = (47.4810, -122.4597, 47.7341, -122.2244)  # south, west, north, east

MODULE = "nlr_ghe_screen.api.osm_api"

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
    file_path = Path(__file__).parent / "osm_data" / "mock_overpass_query.json"
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
    return mocker.patch("nlr_ghe_screen.api.osm_api.time.sleep")



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

@pytest.fixture(autouse=True)
def fast_and_isolated(mocker):
    """
    No real sleeping between split requests, and no dependency on
    overpassql_query.txt: build_overpass_query returns str(bbox), so each
    data query can be identified by the exact bbox it was built for.
    """
    mocker.patch(f"nlr_ghe_screen.api.osm_api.time.sleep")
    mocker.patch(
        f"nlr_ghe_screen.api.osm_api.build_overpass_query",
        side_effect=lambda bbox, query_timeout=None: str(bbox),
    )



# ---------------------------------------------------------------------------
# Test - Query construction
# ---------------------------------------------------------------------------

class TestBuildOverpassQuery:
    """
    Test the different cases for the construction of the Overpass QL text
    """
 
    def test_data_mode_uses_out_geom(self):
        query = build_overpass_query(TEST_BBOX)
        assert "out geom;" in query
        assert "out count;" not in query
 
    def test_bbox_values_interpolated(self):
        query = build_overpass_query(TEST_BBOX)
        south, west, north, east = TEST_BBOX
        assert f"{south},{west},{north},{east}" in query
 
    def test_timeout_and_maxsize_interpolated(self):
        query = build_overpass_query(TEST_BBOX, query_timeout=42)
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
        mock_post = mocker.patch("nlr_ghe_screen.api.osm_api.requests.post", return_value=response)
 
        result = get_overpass_no_splitting(bbox=TEST_BBOX)
 
        elements = result.get("elements", [])
        assert elements[0]["tags"]["name"] == "Stratton Commons"
 
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args.args[0] == "https://overpass-api.de/api/interpreter"
        assert "data" in call_args.kwargs["data"]



# ---------------------------------------------------------------------------
# Test - HTTP-level error classification
# ---------------------------------------------------------------------------

class TestHTTPErrorClassification:
    """
    Test the various possible HTTP response error codes and their respective OverpassError throws
    """

    def test_400_raises_bad_query_error(self, mocker):
        response = make_response(mocker, status_code=400, text="Query parse error near line 3")
        mocker.patch("nlr_ghe_screen.api.osm_api.requests.post", return_value=response)
 
        with pytest.raises(OverpassBadQueryError):
            get_overpass_no_splitting(bbox=TEST_BBOX)

    def test_400_does_not_retry(self, mocker):
        response = make_response(mocker, status_code=400, text="bad query")
        mock_post = mocker.patch("nlr_ghe_screen.api.osm_api.requests.post", return_value=response)
 
        with pytest.raises(OverpassBadQueryError):
            get_overpass_no_splitting(bbox=TEST_BBOX)
 
        mock_post.assert_called_once()

    @pytest.mark.parametrize("status", [502, 503, 504])
    def test_5xx_raises_server_busy(self, mocker, status):
        response = make_response(mocker, status_code=status)
        mock_post = mocker.patch("nlr_ghe_screen.api.osm_api.requests.post", return_value=response)
 
        with pytest.raises(OverpassServerBusyError):
            run_overpass_query("dummy query", max_retries=3)
 
        assert mock_post.call_count == 3
 
    def test_429_raises_rate_limited(self, mocker):
        response = make_response(mocker, status_code=429)
        mock_post = mocker.patch("nlr_ghe_screen.api.osm_api.requests.post", return_value=response)
 
        with pytest.raises(OverpassRateLimitedError):
            run_overpass_query("dummy query", max_retries=3)
 
        assert mock_post.call_count == 3
 
    def test_unexpected_status_raises_generic_overpass_error(self, mocker):
        response = make_response(mocker, status_code=418)
        mocker.patch("nlr_ghe_screen.api.osm_api.requests.post", return_value=response)
 
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
        mocker.patch("nlr_ghe_screen.api.osm_api.requests.post", return_value=response)
 
        with pytest.raises(OverpassTooLargeError):
            get_overpass_no_splitting(bbox=TEST_BBOX)

    def test_no_remark_present_parses_normally(self, mocker):
        data = load_mock_data()
        assert "remark" not in data  # sanity check on the fixture
        response = make_response(mocker, status_code=200, json_data=data)
        mocker.patch("nlr_ghe_screen.api.osm_api.requests.post", return_value=response)
 
        result = get_overpass_no_splitting(bbox=TEST_BBOX)
        assert "elements" in result



# ---------------------------------------------------------------------------
# Client-side network errors
# ---------------------------------------------------------------------------
 
class TestClientSideErrors:
 
    def test_requests_timeout_raises_too_large_error(self, mocker):
        mock_post = mocker.patch(
            "nlr_ghe_screen.api.osm_api.requests.post",
            side_effect=requests.exceptions.Timeout("timed out"),
        )
 
        with pytest.raises(OverpassTimeoutError):
            get_overpass_no_splitting(bbox=TEST_BBOX)

        mock_post.assert_called_once()

    def test_connection_error(self, mocker):
        mock_post = mocker.patch(
            "nlr_ghe_screen.api.osm_api.requests.post",
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
            "nlr_ghe_screen.api.osm_api.requests.post",
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
    Tests the mechanics of splitting a large query: when the API reports a
    bbox is too large, it is halved and each half is fetched, with results
    merged and de-duplicated.
    """

    # ----- Split bbox functionality ---------------

    def test_split_bbox_halves_cover_original(self):
        """The two halves must tile the parent: no gap, no overlap, nothing outside."""
        south, west, north, east = TEST_BBOX
        a, b = split_bbox(TEST_BBOX)

        assert min(a[0], b[0]) == south
        assert min(a[1], b[1]) == west
        assert max(a[2], b[2]) == north
        assert max(a[3], b[3]) == east

        lat_split = a[1] == b[1] and a[3] == b[3]
        lon_split = a[0] == b[0] and a[2] == b[2]
        assert lat_split != lon_split  # exactly one axis was split
        if lat_split:
            assert a[2] == b[0]
        else:
            assert a[3] == b[1]

    def test_split_bbox_splits_longer_side(self):
        """Tall box splits north/south; wide box splits west/east."""
        tall = (0.0, 0.0, 10.0, 1.0)
        a, b = split_bbox(tall)
        assert a == (0.0, 0.0, 5.0, 1.0)
        assert b == (5.0, 0.0, 10.0, 1.0)

        wide = (0.0, 0.0, 1.0, 10.0)
        a, b = split_bbox(wide)
        assert a == (0.0, 0.0, 1.0, 5.0)
        assert b == (0.0, 5.0, 1.0, 10.0)

    def test_split_bbox_accounts_for_longitude_shrinkage(self):
        """At high latitude, equal degree spans mean longitude is physically shorter -> split lat."""
        box = (60.0, 0.0, 70.0, 10.0)  # 10 deg lat vs ~10*cos(65) = 4.2 deg lon
        a, b = split_bbox(box)
        assert a[0] == b[0] - 5.0 or a[2] == b[0]  # north/south split
        assert a[1] == b[1] and a[3] == b[3]

    # ----- No split necessary ---------------

    def test_no_splitting_runs_single_data_query(self, mocker):
        mock_run = mocker.patch(
            f"{MODULE}.run_overpass_query",
            return_value=make_data(make_element(1), make_element(2)),
        )

        result = get_overpass_no_splitting(TEST_BBOX)

        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["query"] == str(TEST_BBOX)
        assert [e["id"] for e in result["elements"]] == [1, 2]

    def test_no_splitting_propagates_too_large(self, mocker):
        mock_run = mocker.patch(
            f"{MODULE}.run_overpass_query",
            side_effect=OverpassTooLargeError("too big"),
        )

        with pytest.raises(OverpassTooLargeError):
            get_overpass_no_splitting(TEST_BBOX)

        mock_run.assert_called_once()  # never splits

    def test_splitting_fits_first_try_makes_one_query(self, mocker):
        mock_run = mocker.patch(
            f"{MODULE}.run_overpass_query",
            return_value=make_data(make_element(1)),
        )

        result = get_overpass_with_splitting(TEST_BBOX)

        mock_run.assert_called_once()
        assert [e["id"] for e in result["elements"]] == [1]

    # ----- Splitting ---------------

    def test_too_large_error_triggers_split_into_halves(self, mocker):
        half_a, half_b = split_bbox(TEST_BBOX)
        mock_run = mocker.patch(
            f"{MODULE}.run_overpass_query",
            side_effect=[
                OverpassTooLargeError("timed out"),  # full bbox
                make_data(make_element(1)),          # half A
                make_data(make_element(2)),          # half B
            ],
        )

        result = get_overpass_with_splitting(TEST_BBOX)

        queries = [c.args[0] for c in mock_run.call_args_list]
        assert queries == [str(TEST_BBOX), str(half_a), str(half_b)]
        assert {e["id"] for e in result["elements"]} == {1, 2}

    def test_recursive_split_multiple_levels(self, mocker):
        """Full box too big, first half too big again, everything else fits."""
        half_a, half_b = split_bbox(TEST_BBOX)
        quarter_aa, quarter_ab = split_bbox(half_a)

        outcomes = {
            str(TEST_BBOX): OverpassTooLargeError("full"),
            str(half_a): OverpassTooLargeError("half a"),
            str(quarter_aa): make_data(make_element(1)),
            str(quarter_ab): make_data(make_element(2)),
            str(half_b): make_data(make_element(3)),
        }

        def fake_run(query, **kwargs):
            outcome = outcomes[query]  # KeyError => unexpected bbox queried
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        mock_run = mocker.patch(f"{MODULE}.run_overpass_query", side_effect=fake_run)

        result = get_overpass_with_splitting(TEST_BBOX)

        assert mock_run.call_count == 5
        assert sorted(e["id"] for e in result["elements"]) == [1, 2, 3]

    def test_split_pauses_between_requests(self, mocker):
        mocker.patch(
            f"{MODULE}.run_overpass_query",
            side_effect=[
                OverpassTooLargeError("full"),
                make_data(make_element(1)),
                make_data(make_element(2)),
            ],
        )
        mock_sleep = mocker.patch(f"{MODULE}.time.sleep")

        get_overpass_with_splitting(TEST_BBOX)

        assert mock_sleep.call_count == 2  # one before each half

    def test_duplicate_elements_across_halves_are_deduplicated(self, mocker):
        """A way crossing the split line is returned by both halves."""
        shared = make_element(99)
        mocker.patch(
            f"{MODULE}.run_overpass_query",
            side_effect=[
                OverpassTooLargeError("full"),
                make_data(make_element(1), shared),
                make_data(shared, make_element(2)),
            ],
        )

        result = get_overpass_with_splitting(TEST_BBOX)

        ids = [e["id"] for e in result["elements"]]
        assert sorted(ids) == [1, 2, 99]
        assert ids.count(99) == 1

    def test_same_id_different_type_is_not_deduplicated(self, mocker):
        """Dedup key is (type, id), so a way and relation sharing an id are both kept."""
        way = {**make_element(5), "type": "way"}
        rel = {**make_element(5), "type": "relation"}
        mocker.patch(
            f"{MODULE}.run_overpass_query",
            side_effect=[
                OverpassTooLargeError("full"),
                make_data(way),
                make_data(rel),
            ],
        )

        result = get_overpass_with_splitting(TEST_BBOX)

        assert len(result["elements"]) == 2

    def test_meta_taken_from_first_response(self, mocker):
        first = {"version": 0.6, "generator": "first", "elements": [make_element(1)]}
        second = {"version": 0.6, "generator": "second", "elements": [make_element(2)]}
        mocker.patch(
            f"{MODULE}.run_overpass_query",
            side_effect=[OverpassTooLargeError("full"), first, second],
        )

        result = get_overpass_with_splitting(TEST_BBOX)

        assert result["generator"] == "first"
        assert {e["id"] for e in result["elements"]} == {1, 2}

    def test_empty_results_return_empty_elements(self, mocker):
        mocker.patch(f"{MODULE}.run_overpass_query", return_value=make_data())

        result = get_overpass_with_splitting(TEST_BBOX)

        assert result["elements"] == []

    # ----- Depth limit and error propagation ----------------------------------

    def test_max_depth_exceeded_raises_too_large(self, mocker):
        mock_run = mocker.patch(
            f"{MODULE}.run_overpass_query",
            side_effect=OverpassTooLargeError("always too big"),
        )

        with pytest.raises(OverpassTooLargeError):
            get_overpass_with_splitting(TEST_BBOX, max_depth=2)

        # Depth 0 -> 1 -> 2 (always taking the first half), then depth 2 refuses to split
        assert mock_run.call_count == 3

    def test_max_depth_zero_never_splits(self, mocker):
        mock_run = mocker.patch(
            f"{MODULE}.run_overpass_query",
            side_effect=OverpassTooLargeError("too big"),
        )

        with pytest.raises(OverpassTooLargeError):
            get_overpass_with_splitting(TEST_BBOX, max_depth=0)

        mock_run.assert_called_once()

    @pytest.mark.parametrize("error", [OverpassBadQueryError("bad"), OverpassError("boom")])
    def test_other_errors_propagate_without_splitting(self, mocker, error):
        mock_run = mocker.patch(f"{MODULE}.run_overpass_query", side_effect=error)

        with pytest.raises(type(error)):
            get_overpass_with_splitting(TEST_BBOX)

        mock_run.assert_called_once()

    def test_error_in_one_half_aborts_whole_fetch(self, mocker):
        mocker.patch(
            f"{MODULE}.run_overpass_query",
            side_effect=[
                OverpassTooLargeError("full"),
                make_data(make_element(1)),
                OverpassBadQueryError("bad"),  # half B fails
            ],
        )

        with pytest.raises(OverpassBadQueryError):
            get_overpass_with_splitting(TEST_BBOX)