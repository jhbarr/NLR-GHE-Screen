import requests
import os
from pathlib import Path

OPENTOP_URL = 'https://portal.opentopography.org/API/globaldem'
HTTP_TIMEOUT_BUFFER = 8


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class OpentopError(Exception):
    """Base class for all Overpass-related failures"""


class OpentopNoDataError(OpentopError):
    """HTTP - The API call returned with no data"""


class OpentopBadRequestError(OpentopError):
    """HTTP 400 - The query resulted in a bad or incorrect response from the API"""


class OpentopUnauthorizedRequestError(OpentopError):
    """HTTP 401 - Unathorized query was executed"""


class OpentopInternalError(OpentopError):
    """HTTP 500 - Internal Open Topography error. Do not retry"""


class OpentopConnectionError(OpentopError):
    """A general HTTP connection error"""
    

# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def classify_and_raise(response):
    """Check an OpenTopography API response and raise the matching error if it failed

    The following checks are executed:

        - 204 -> :class:`OpentopNoDataError`
        - 500 -> :class:`OpentopInternalError`
        - 400 -> :class:`OpentopBadRequestError` (includes the first 500 characters
          of the response text)
        - 401 -> :class:`OpentopUnauthorizedRequestError`
        - any other non-200 status -> :class:`OpentopError`

    Args:
        response (requests.Response): The HTTP response returned by the OpenTopography API.
 
    Returns:
        dict: The parsed JSON body of the response, if no error was detected.
 
    Raises:
        OpentopNoDataError: If status code is 204
        OpentopInternalError: If status code is 500
        OpentopBadRequestError: If status caoade is 400
        OpentopUnauthorizedRequestError: If status code is 401
        OpentopError: For any unexpected, non-200 status code
    """
    status = response.status_code

    if status == 204:
        # If the query did not return any valid data 
        raise OpentopNoDataError("Opentop API no data returned (204)")

    if status == 500:
        raise OpentopInternalError("Opentop database service internal error (500)")

    if status == 400:
        # If there is a bad request, then overpass api will return an HTMl body
        # describing what went wrong with the request
        raise OpentopBadRequestError(
            f"Overpass API rejected the query as malformed (400): "
            f"{response.text[:500]}"
        )

    if status == 401:
        # If the query or API request was sent with invalid authentication credentials 
        # For example - an invalid API key 
        raise OpentopUnauthorizedRequestError("Opentop API call did not have valid authorization (401)")

    if status != 200:
        # Unknown or unexpected error code
        raise OpentopError(f"Opentop API returned unexpected error: {status}")


    # Parse the raster content out of the API response body and return it
    return response.content



# ---------------------------------------------------------------------------
# Query building + Execution
# ---------------------------------------------------------------------------

def build_opentop_query(bbox, api_key):
    """Constructs the parameters required by an OpenTopography API call

    The required parameters are: 

        south, north, east, west: Four coordinates describing the area where the elevation data should be 
            retrieved from
        api_key: Each request to the OpenTopography API requires a key

    Args:
        bbox (tuple[float, float, float, float]): The bounding box as
            ``(south, west, north, east)``, i.e. ``(lat_min, lon_min, lat_max,
            lon_max)``
        api_key (str): An OpenTopography API key
        
    Returns:
        params (dict): The parameters for the Opentop API call
    
    Raises:
        ValueError: If an API key is not provided
    """
    south, west, north, east = bbox

    if not api_key:
        raise ValueError("Opentop API key not found in environment variables")

    params = {
        "demtype": "SRTMGL1",
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": api_key
    }

    return params


def run_opentop_query(params):
    """
    Execute Opentop API query and raise any specific OpentopError conditions 

    Sends constructed parameters from `params` to the OpenTopography API and checks the response with :func:`classify_and_raise`.
    No retry mechanics are currently implemented. Therefore, any error is propogated upwards.

    Args:
        params (dict): A dictionary containing the necessary Opentop API parameters, including an API key

    Returns:
        response (str): The content of the API response body (must be accessed in GTiff format)
    
    Raises:
        OpentopConnectionError: If there was an error en countered connecting to the API
        OpentopError (and subclasses): If any error encountered while querying the API
    """
    http_timeout =  HTTP_TIMEOUT_BUFFER

    try:
        response = requests.get(
            OPENTOP_URL,
            params=params,
            # timeout=HTTP_TIMEOUT_BUFFER
        )

        return classify_and_raise(response)

    except requests.exceptions.ConnectionError as e:
        raise OpentopConnectionError(f"Connection faild: {e}")

    except requests.exceptions.RequestException as e:
        raise OpentopError(f"Unexpected request error: {e}")

    except OpentopError:
        raise



# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def get_opentop(bbox, api_key):
    """Fetch the elevation raster data from OpenTopography within the area described by bbox

    Executes an API query using ``bbox`` and the ''api_key``. Once all error checks are passed,
    then the data is exported to ``elevation_data.tif' in the local ``src/imports`` folder, for later
    use in elevation processing functions.

    Args:
        bbox (tuple[float, float, float, float]): The bounding box as
            ``(south, west, north, east)``, i.e. ``(lat_min, lon_min, lat_max,
            lon_max)``
        api_key (str): An OpenTopography API key
    
    Raises:
        OpentopError (and subclasses): Any of the errors described in the :func:`classify_and_raise` function
    """
    print("\n---------------------------")
    print("Get - OpenTopography API Request")

    # Create imports folder if it does not yet exist
    dir_path = Path(__file__).parent.parent / '..' / 'imports'
    dir_path.mkdir(parents=True, exist_ok=True)

    # Build and execute the API call to OpenTopography
    params = build_opentop_query(bbox=bbox, api_key=api_key)
    data = run_opentop_query(params=params)

    # Export the data to a GTiff file 
    file_path = dir_path / 'elevation_data.tif'
    with open(file_path, "wb") as f:
        f.write(data)

    print("Success - Raster data imported")
    print("---------------------------")