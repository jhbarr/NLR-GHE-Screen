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
    """
    Inspect a resopnse from the Overpass API and raise appropriate errors

    Parameters:
        response (dict): The response body from the Overpass API
    
    Returns:
        data (str): The response body of the Opentop API call - in the event that no errors were raised

    Raises:
        OpentopError: Various kinds of Opentop API errors based on the status of the API response
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
    """
    This function creates the parameters for the Opentop API call

    Parameters:
        bbox (Tuple[float]): Coordinate bounding box in the form (lat_min, lon_min, lat_max, lon_max)
        api_key (str): An OpenTopography API key
        
    Returns:
        params (dict): The parameters for the Opentop API call
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
    Execute Opentop API query. Raises any specific OpentopError conditions 

    Parameters:
        params (dict): A dictionary containing the necessary Opentop API parameters, including an API key

    Returns:
        response (str): The content of the API response body (must be accessed in GTiff format)
    
    Raises:
        OpentopError (and subclasses): other failures
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



# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------

def get_opentop(bbox, api_key):
    """
    Create and execute query on OpenTopography API, fetch the response body, analyze any errors
    and export the response to a GTiff file 

    Parameters:
        bbox (Tuple[float]): A coordinate bounding box in the form (lat_min, lon_min, lat_max, lon_max)
    
    Returns:
        None
    
    Raises:
        OpentopError (and subclasses): other failures
    """
    print("\n---------------------------")
    print("Get - OpenTopography API Request")

    # Create imports folder if it does not yet exist
    dir_path = Path('../imports')
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