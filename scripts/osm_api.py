import time
import requests
 
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

DEFAULT_QUERY_TIMEOUT = 30
HTTP_TIMEOUT_BUFFER = 5

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
# Query building + Execution
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
            raise OverpassTooLargeError(
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
# Convenience wrappers
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


def get_overpass(bbox, query_timeout=DEFAULT_QUERY_TIMEOUT):
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
 
    query = build_overpass_query(bbox, mode="data", query_timeout=query_timeout)
    data = run_overpass_query(query, query_timeout=query_timeout)
 
    print("Success - Query Received")
    print("---------------------------")
 
    return data
