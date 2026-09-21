import time
import requests
import math
 
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

DEFAULT_QUERY_TIMEOUT = 30
HTTP_TIMEOUT_BUFFER = 5

MAX_ELEMENTS_PER_QUERY = 3500 # Subject to change based on live results
MAX_SPLIT_DEPTH = 4
REQUEST_PAUSE = 1.0

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
 
class OverpassError(Exception):
    """Base class for all Overpass-related failures."""
 
 
class OverpassTooLargeError(OverpassError):
    """
    The query exceeded the server's time or memory budget for the
    requested area. This is the signal to split the bbox and retry
    the halves, not to retry the same query.
    """
 
 
class OverpassRateLimitedError(OverpassError):
    """HTTP 429 - too many requests. Retry the same query after backoff."""
 
 
class OverpassServerBusyError(OverpassError):
    """HTTP 502/503/504 - server overloaded. Retry the same query after backoff."""
 
 
class OverpassBadQueryError(OverpassError):
    """
    HTTP 400 or a non-timeout/non-memory remark in the response body.
    Indicates a malformed query - a bug, not a transient condition.
    Don't retry, don't split.
    """
 
 
class OverpassConnectionError(OverpassError):
    """Network-level failure (DNS, connection refused, etc). Retry a few times."""


class OverpassTimeoutError(OverpassError):
    """If there is a network-level failure with the HTTP timing out before reaching Overpass API"""



# ---------------------------------------------------------------------------
# Error classification
# ---------------------------------------------------------------------------

def classify_and_raise(response):
    """
    Inspect a resopnse from the Overpass API and raise appropriate errors

    Parameters:
        response (dict): The response body from the Overpass API
    
    Returns:
        data (dict): The response body of the Overpass API call - in the event that no errors were raised
    """
    status = response.status_code

    if status == 429:
        # If the server rate limit is hit
        raise OverpassRateLimitedError("Overpass API rate limit hit (429)")

    if status in (502, 503, 504):
        raise OverpassServerBusyError(f"Overpass API is overloaded: {status}")

    if status == 400:
        # If there is a bad request, then overpass api will return an HTMl body
        # describing what went wrong with the request
        raise OverpassBadQueryError(
            f"Overpass API rejected the query as malformed (HTTP 400): "
            f"{response.text[:500]}"
        )

    if status != 200:
        # Unknown or unexpected error code
        raise OverpassError(f"Overpass API returned unexpected error: {status}")

    # There can be instances where a 200 status code is returned, but there is an error in the message body
    # Must discard that response
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError as e:
        raise OverpassError(
            "Overpass API returned a 200 response that was not valid JSON"
        ) from e

    remark = data.get("remark", "")
    if remark:
        lowered = remark.lower()
        if "timed out" in lowered or "timeout" in lowered:
            raise OverpassTooLargeError(f"Query exceeded time budget: {remark}")
        if "out of memory" in lowered or "memory" in lowered:
            raise OverpassTooLargeError(f"Query exceeded memory budget: {remark}")
        # Some other runtime remark we don't specifically recognize.
        raise OverpassBadQueryError(f"Overpass API returned an error: {remark}")
 
    return data



# ---------------------------------------------------------------------------
# Query building + Query execution
# ---------------------------------------------------------------------------

def build_overpass_query(bbox, mode="data", query_timeout = DEFAULT_QUERY_TIMEOUT):
    """
    Build an Overpass QL query for the given bounding box

    Parameters:
        bbox (Tuple[float]): Coordinate bounding box in the form (lat_min, lon_min, lat_max, lon_max)
        mode (str): "data" for the full output of all the geometries from the query
                    "count" for the 'probe' to see how many geometries will be returned
        query_timeout (int): The value for the Overpass timeout setting
    
    Returns:
        str: The Overpass QL query string
    """

    south, west, north, east = bbox
    bbox_str = f"{south},{west},{north},{east}"

    filters = f"""
    way["leisure"~"park|dog_park"]({bbox_str});
    relation["leisure"~"park|dog_park"]({bbox_str});
 
    way["landuse"~"recreation_ground|meadow|grass|farmland"]({bbox_str});
    relation["landuse"~"recreation_ground|meadow|grass|farmland"]({bbox_str});
 
    way["natural"~"wood|grassland|scrub|heath|wetland"]({bbox_str});
    relation["natural"~"wood|grassland|scrub|heath|wetland"]({bbox_str});
 
    way["amenity"="parking"]({bbox_str});
    relation["amenity"="parking"]({bbox_str});
    """

    out_clause = "out count;" if mode == "count" else "out geom;"

    return f"""
    [out:json][timeout:{query_timeout}];
    (
    {filters}
    );
    {out_clause}
    """

def run_overpass_query(query, query_timeout=DEFAULT_QUERY_TIMEOUT, max_retries=3, backoff_base=2.0):
    """
    Exeecure an Overpass QL query with retries for certain failures. Additionally,
    raise any specific Overpass errors for certain conditions

    Parameters:
        query (str): A constructed Overpass QL query text
        query_timeout (int): The value for the Overpass timeout setting
        max_retries (int): retry attempts for rate limit/server busy cases 
        backoff_base (float): exponential backoff base - how long the program will wait to retry API call

    Returns:
        response (dict): The API JSON response body

    Raises:
        OverpassTooLargeError: query exceeded time/memory budget - split
            the bbox and retry the halves.
        OverpassBadQueryError: malformed query - fix the query, don't retry.
        OverpassRateLimitedError / OverpassServerBusyError /
        OverpassConnectionError: transient - raised only after retries
            are exhausted.
    """
    http_timeout = query_timeout + HTTP_TIMEOUT_BUFFER
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                timeout=http_timeout
            )

            return classify_and_raise(response)
        
        except (OverpassTooLargeError, OverpassBadQueryError):
            # Nothing to be retried
            raise 

        except (OverpassRateLimitedError, OverpassServerBusyError) as e:
            last_error = e

        except requests.exceptions.Timeout:
            # The HTTP client gave up before overpass responded at all
            raise OverpassTimeoutError(
                f"Request timed out client-side before overpass responded"
            )

        except requests.exceptions.ConnectionError as e:
            last_error = OverpassConnectionError(f"Connection failed: {e}")

        except requests.exceptions.RequestException as e:
            last_error = OverpassError(f"Unexpected request error: {e}")
 
        if attempt < max_retries:
            wait = backoff_base * (2 ** (attempt - 1))
            print(f"  Overpass request failed ({last_error}); "
                  f"retrying in {wait:.1f}s (attempt {attempt}/{max_retries})")
            time.sleep(wait)

    raise last_error



# ---------------------------------------------------------------------------
# Query splitting
# ---------------------------------------------------------------------------

def split_bbox(bbox):
    """
    Split a Overpass QL bounding box in half along its longer side

    Parameters:
        bbox (Tuple[float]): A coordinate bounding box in the form (lat_min, lon_min, lat_max, lon_max)
    
    Returns:
        Tuple[Tuple[float]]: Two coordinate bounding boxes in the form (lat_min, lon_min, lat_max, lon_max)
    """
    south, west, north, east = bbox

    lat_span = north - south
    lon_span = (east - west) * math.cos(math.radians((south + north) / 2))
 
    if lat_span >= lon_span:
        mid = (south + north) / 2
        return (south, west, mid, east), (mid, west, north, east)
 
    mid = (west + east) / 2
    return (south, west, north, mid), (south, mid, north, east)


def fetch_bbox(bbox, collected, depth, query_timeout=DEFAULT_QUERY_TIMEOUT, max_elements=MAX_ELEMENTS_PER_QUERY, max_depth=MAX_SPLIT_DEPTH):
    """
    Recursively fetch a bounding box, splitting whenever it is too large

    Parameters:
        bbox (Tuple[float]): A coordinate bounding box in the form (lat_min, lon_min, lat_max, lon_max)
        collected (dict): Mimics the structure of the 'elements' section of Overpass API JSON body. Is used to collect the responses of the different
            Overpass sub-queries
    """
    indent = " " * depth
    too_large = False

    # Probe the API to see how large the query response would be and whether
    # that count would exceed the max count limit
    try:
        count = get_count(bbox)
        print(f"{indent}bbox {bbox}: {count} elements")
        if count == 0:
            return

        too_large = count > max_elements
    except OverpassTooLargeError:
        print(f"{indent}bbox {bbox}: count probe exceeded limit")


    # Run the API query if the count probe was under the max elements limit
    # However, because the probe is only a rudimentary check, the query still may exceed the API limits
    if not too_large:
        time.sleep(REQUEST_PAUSE)
        try:
            overpass_query = build_overpass_query(bbox, mode='data', query_timeout=query_timeout)
            data = run_overpass_query(overpass_query, query_timeout=query_timeout)

        except OverpassTooLargeError:
            print(f"{indent}bbox {bbox}: query exceeded limit")
            too_large = True

        else:
            if collected['meta'] is None:
                collected['meta'] = {k: v for k, v in data.items() if k != 'elements'}
            for element in data.get('elements', []):
                collected['elements'][(element['type'], element['id'])] = element
            return


    # Split the bounding box and recurse 
    if depth >= max_depth:
        raise OverpassTooLargeError(
            f"{bbox} bbox still too large after max recursive splits"
        )

    print(f"{indent}splitting bbox {bbox}")
    for half in split_bbox(bbox):
        time.sleep(REQUEST_PAUSE)
        fetch_bbox(half, collected, depth + 1, query_timeout, max_elements, max_depth)


 
# ---------------------------------------------------------------------------
# Main API Execution
# ---------------------------------------------------------------------------

def get_count(bbox, query_timeout=10):
    """
    Function to probe the database to see how many geometry objects will be returned by query call

    Parameters:
        bbox (Tuple[float]): A coordinate bounding box in the form (lat_min, lon_min, lat_max, lon_max)
        query_timeout (int): The Overpass database timeout
    
    Returns:
        int: The number of geometry elements that would be returned
    """
    query = build_overpass_query(bbox, mode="count", query_timeout=query_timeout)
    data = run_overpass_query(query, query_timeout=query_timeout)

    elements = data.get("elements", [])
    if not elements:
        return 0

    return int(elements[0].get("tags", {}).get("total", 0))


def get_overpass(bbox, query_timeout=DEFAULT_QUERY_TIMEOUT, max_elements=MAX_ELEMENTS_PER_QUERY, max_depth=MAX_SPLIT_DEPTH):
    """
    Fetch full geometry and tags from the Overpass API given the bounding box area. Employs procedures to 
    split bounding box queries if original returns an exceeded memory or time limit error or if it predicted to
    
    Parameters:
        bbox (Tuple[float]): A coordinate bounding box in the form (lat_min, lon_min, lat_max, lon_max)
    
    Returns:
        dict: The Overpass API JSON response body

    Raises:
        OverpassTooLargeError: split the bbox and retry the halves.
        OverpassBadQueryError: fix the query - this is a bug, not a
            transient condition.
        OverpassError (and subclasses): other failures after retries
            are exhausted.
    """
    print("\n---------------------------")
    print("Post - Overpass API Request")

    collected = {'elements': {}, 'meta': None}
    fetch_bbox(bbox, collected, depth=0, query_timeout=query_timeout, max_elements=max_elements, max_depth=max_depth)

    print(collected)

    data = dict(collected['meta'] or {})
    data['elements'] = list(collected['elements'].values())
 
    print("Success - Query Received")
    print("---------------------------")
 
    return data


def get_overpass_no_splitting(bbox, query_timeout=DEFAULT_QUERY_TIMEOUT):
    """
    Fetch full geometry and tags from the Overpass API given the bounding box area
        
    Parameters:
        bbox (Tuple[float]): A coordinate bounding box in the form (lat_min, lon_min, lat_max, lon_max)
    
    Returns:
        dict: The Overpass API JSON response body

    Raises:
        OverpassTooLargeError: split the bbox and retry the halves.
        OverpassBadQueryError: fix the query - this is a bug, not a
            transient condition.
        OverpassError (and subclasses): other failures after retries
            are exhausted.
    """
    print("\n---------------------------")
    print("Post - Overpass API Request")

    query = build_overpass_query(bbox=bbox, mode='data', query_timeout=query_timeout)
    data = run_overpass_query(query=query, query_timeout=query_timeout)
    
    print("Success - Query Received")
    print("---------------------------")
    
    return data