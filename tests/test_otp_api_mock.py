import pytest
import requests
from pathlib import Path

from scripts.otp_api import (
    build_opentop_query,
    run_opentop_query,
    get_opentop,
    OpentopError,
    OpentopConnectionError,
    OpentopUnauthorizedRequestError,
    OpentopBadRequestError,
    OpentopInternalError,
    OpentopNoDataError
)

TEST_BBOX = (47.4810, -122.4597, 47.7341, -122.2244)  # south, west, north, east
TEST_API_KEY = 'example key'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_response(
        mocker, 
        status_code=200, 
        content=b"", 
        content_type="image/tiff", 
        text=""
    ):
    """
    Build a MagicMock standing in for a requests.Response
    """
    response = mocker.MagicMock()
    response.status_code = status_code
    response.content = content
    response.headers = {"Content-Type": content_type}
    
    return response


def load_mock_data_bytes():
    """
    Load in the test elevation_raster_data.tif file and return its contents in byte format
    """
    data_dir = Path(__file__).parent / 'data_otp'
    return (data_dir / 'elevation_raster_data.tif').read_bytes()



# ---------------------------------------------------------------------------
# Test - API Parameter Construction
# --------------------------------------------------------------------------

class TestQueryConstruction:
    """
    Test the different cases for the construction of the OpenTopography API parameters
    """

    def test_valid_api_key(self):
        api_key = "example key"
        params = build_opentop_query(TEST_BBOX, api_key=TEST_API_KEY)
        assert api_key in params.values()


    def test_invalid_api_key(self):
        with pytest.raises(ValueError):
            params = build_opentop_query(TEST_BBOX, "")



# ---------------------------------------------------------------------------
# Test - Successful requests
# ---------------------------------------------------------------------------
    
class TestSuccessfulRequests:
    """
    Verify correct behavior upon successful response from OpenTopography API
    """

    def test_get_otp_success(self, mocker):
        data = load_mock_data_bytes()
        response = make_response(mocker, status_code=200, content=data)
        mock_get = mocker.patch("scripts.otp_api.requests.get", return_value=response)

        get_opentop(bbox=TEST_BBOX, api_key=TEST_API_KEY)

        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args.args[0] == 'https://portal.opentopography.org/API/globaldem'

        path_str = '../imports/elevation_data.tif'
        assert Path(path_str).is_file()



# ---------------------------------------------------------------------------
# Test - HTTP-level error classification
# ---------------------------------------------------------------------------

class TestHTTPErrorClassification:
    """
    Test the various possible HTTP errors and that their respective OpentopError are being raised
    """

    def test_400_raises_bad_request_error(self, mocker):
        response = make_response(mocker, status_code=400, text="400 Bad Request")
        mocker.patch("scripts.otp_api.requests.get", return_value=response)

        with pytest.raises(OpentopBadRequestError):
            run_opentop_query(params={})


    def test_401_raises_unauthorized_error(self, mocker):
        response = make_response(mocker, status_code=401, text="401 Unauthorized Access")
        mocker.patch("scripts.otp_api.requests.get", return_value=response)

        with pytest.raises(OpentopUnauthorizedRequestError):
            run_opentop_query(params={})


    def test_204_raises_nodata_error(self, mocker):
        response = make_response(mocker, status_code=204, text="204 No Data")
        mocker.patch("scripts.otp_api.requests.get", return_value=response)

        with pytest.raises(OpentopNoDataError):
            run_opentop_query(params={})

    def test_500_raises_internal_error(self, mocker):
        response = make_response(mocker, status_code=500, text="500 Internal Error")
        mocker.patch("scripts.otp_api.requests.get", return_value=response)

        with pytest.raises(OpentopInternalError):
            run_opentop_query(params={})


    def test_unexpected_raises_generic_error(self, mocker):
        response = make_response(mocker, status_code=406, text="400 Bad Request")
        mocker.patch("scripts.otp_api.requests.get", return_value=response)

        with pytest.raises(OpentopError):
            run_opentop_query(params={})



# ---------------------------------------------------------------------------
# Client-side network errors
# ---------------------------------------------------------------------------

class TestClientSideError:
    """
    Test that the program handles client side errors correctly
    """

    def test_request_timeout(self, mocker):
        mock_get = mocker.patch(
            "scripts.otp_api.requests.get", 
            side_effect=requests.exceptions.Timeout("Timed Out")
        )

        with pytest.raises(OpentopError):
            get_opentop(bbox=TEST_BBOX, api_key=TEST_API_KEY)

        mock_get.assert_called_once()


    def test_request_timeout(self, mocker):
        mock_get = mocker.patch(
            "scripts.otp_api.requests.get", 
            side_effect=requests.exceptions.ConnectionError("Refused")
        )

        with pytest.raises(OpentopConnectionError):
            get_opentop(bbox=TEST_BBOX, api_key=TEST_API_KEY)

        mock_get.assert_called_once()