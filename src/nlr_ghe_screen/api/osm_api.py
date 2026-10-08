import time
import requests
import math
from pathlib import Path
 
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DIR = Path(__file__).parent

DEFAULT_QUERY_TIMEOUT = 60
HTTP_TIMEOUT_BUFFER = 5

MAX_SPLIT_DEPTH = 4
REQUEST_PAUSE = 1.0

API_TIME_THRESHOLD = 10.0

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
    """Check an Overpass API response and raise the matching error if it failed

    Inspects the HTTP status code first, then for a 200 response, the JSON body is checked because overpass can return
    a 200 status with an error described in a ``remark`` field in the JSON response body.

    The following checks are executed:

        - 429 -> :class:`OverpassRateLimitedError`
        - 502, 503, 504 -> :class:`OverpassServerBusyError`
        - 400 -> :class:`OverpassBadQueryError` (includes the first 500 characters
          of the response text)
        - any other non-200 status -> :class:`OverpassError`
        - 200 with a body that is not valid JSON -> :class:`OverpassError`
        - 200 with a timeout or memory remark -> :class:`OverpassTooLargeError`
        - 200 with any other remark -> :class:`OverpassBadQueryError`

    Args:
        response (requests.Response): The HTTP response returned by the Overpass API.
 
    Returns:
        dict: The parsed JSON body of the response, if no error was detected.
 
    Raises:
        OverpassRateLimitedError: If the status code is 429.
        OverpassServerBusyError: If the status code is 502, 503, or 504.
        OverpassBadQueryError: If the status code is 400, or the body contains an
            unrecognized remark.
        OverpassTooLargeError: If the body contains a timeout or memory remark.
        OverpassError: If the status code is unexpected, or a 200 response is not
            valid JSON.
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
# Query building + Bounding box splitting
# ---------------------------------------------------------------------------

def build_overpass_query(bbox, query_timeout = DEFAULT_QUERY_TIMEOUT):
    """Build an Overpass QL query for the given bounding box

    The query selects ways and relations within the bounding box that match any of the following
    tag filters:

        - ``leisure`` matching ``park|dog_park``
        - ``landuse`` matching ``recreation_ground|meadow|grass|farmland``
        - ``natural`` matching ``wood|grassland|scrub|heath|wetland``
        - ``amenity`` equal to ``parking``
    
    The output clause depends on ``mode``: ``count`` returns only a summary element with the total number of matches.
    While any other value returns the full geometry and tags of each space. 

    Args:
        bbox (tuple[float, float, float, float]): The bounding box as
            ``(south, west, north, east)``, i.e. ``(lat_min, lon_min, lat_max,
            lon_max)``.
        query_timeout (int, optional): The value, in seconds, for the Overpass
            ``timeout`` setting. Defaults to ``DEFAULT_QUERY_TIMEOUT``.
 
    Returns:
        str: The Overpass QL query string.
    """

    south, west, north, east = bbox
    bbox_str = f"{south},{west},{north},{east}"

    # Open the file containing the Ovperass QL Query
    with open(DIR / 'overpassql_query.txt') as file:
        query_content = file.read()
    filters = query_content.format(bbox_str=bbox_str)

    out_clause = "out geom;"

    return f"""
    [out:json][timeout:{query_timeout}];
    (
    {filters}
    );
    {out_clause}
    """


def split_bbox(bbox):
    """Split a Overpass QL bounding box in half along its longer side

    Args:
        bbox (tuple[float, float, float, float]): The bounding box as
            ``(south, west, north, east)``, i.e. ``(lat_min, lon_min, lat_max,
            lon_max)``.
 
    Returns:
        tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
            Two bounding boxes in the same ``(south, west, north, east)`` form,
            ordered south then north, or west then east.
    """
    south, west, north, east = bbox

    # Use the midpoint latitude when converting longitude degrees to
    # approximate physical distance. This accounts for longitude
    # degrees getting shorter toward the poles.
    mid_lat = (south + north) / 2
    lat_span = north - south
    lon_span = (east - west) * math.cos(math.radians(mid_lat))

    if lat_span >= lon_span:
        # North/south split
        mid = (south + north) / 2

        return (
            (south, west, mid, east),
            (mid, west, north, east),
        )

    else:
        # West/east split
        mid = (west + east) / 2

        return (
            (south, west, north, mid),
            (south, mid, north, east),
        )



# ---------------------------------------------------------------------------
# Query Execution
# ---------------------------------------------------------------------------

def run_overpass_query(query, query_timeout=DEFAULT_QUERY_TIMEOUT, max_retries=3, backoff_base=2.0):
    """Execute an Overpass QL query, retrying on transient failures

    Sends a query to the Overpass API and checks the response with :func:`classify_and_raise`. Rate limiting, 
    server overload, and connection failures are retried with exponential backoff. If all retries fail, the last 
    error raised is propogated

    Queries that are too large or malformed are not retried. 

    Note:
        The query returns any spaces that overlap the bounding box, not only the spaces that lie 
        strictly within it.

    Args:
        query (str): A constructed Overpass QL query string.
        query_timeout (int, optional): The Overpass ``timeout`` setting, in seconds,
            which is also used to derive the HTTP timeout. Defaults to
            ``DEFAULT_QUERY_TIMEOUT``.
        max_retries (int, optional): The maximum number of attempts for transient
            failures (rate limiting, server overload, connection errors). Defaults
            to 3.
        backoff_base (float, optional): The base of the exponential backoff, in
            seconds, that controls how long to wait before the next attempt.
            Defaults to 2.0.
 
    Returns:
        dict: The parsed JSON body of the API response.
 
    Raises:
        OverpassTooLargeError: If the query exceeded the server's time or memory
            budget. Split the bounding box and retry the halves.
        OverpassBadQueryError: If the query is malformed. Fix the query; do not retry.
        OverpassTimeoutError: If the HTTP request timed out client-side before
            Overpass responded.
        OverpassRateLimitedError: If the rate limit is still being hit after all
            retries.
        OverpassServerBusyError: If the server is still overloaded after all
            retries.
        OverpassConnectionError: If the connection still fails after all retries.
        OverpassError: For unexpected status codes, invalid JSON, or other request
            errors.
    """
    http_timeout = query_timeout + HTTP_TIMEOUT_BUFFER
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.monotonic() # Get the current time
            
            response = requests.post(
                OVERPASS_URL,
                data={"data": query},
                headers = {
                    "User-Agent": "ghe-finder-script/0.1 (contact: joseph.barrows@nlr.gov)",
                    "Accept": "application/json",
                },
                timeout=http_timeout
            )

            elapsed = time.monotonic() - t0
            return classify_and_raise(response)

        # ** No Retry Errors - Should force program exit ** 
        except (OverpassBadQueryError):
            # Bad query was executed - Nothing to be retried
            raise

        except requests.exceptions.Timeout:
            # The HTTP client gave up before overpass responded at all
            # Not a problem with Overpass 
            raise OverpassTimeoutError(
                f"Request timed out client-side before overpass responded"
            )


        # ** No retry errors - Should force query split ** 
        except (OverpassTooLargeError) as e:
            # Query was too large - raise error to force split
            print(f"Query was too large - Forcing query split")
            raise


        # ** Retry errors - Should force query retry ** 
        except (OverpassRateLimitedError) as e:
            # Rate limited error - backoff and wait
            print(f"Query was rate limited - Retrying: {e}")
            last_error = e

        except requests.exceptions.ConnectionError as e:
            last_error = e

        except requests.exceptions.RequestException as e:
            last_error = e
        
        except (OverpassServerBusyError, OverpassTimeoutError) as e:
            # Depending on the elapsed time either retry or force split 
            if elapsed < API_TIME_THRESHOLD:
                print(f"Server was busy - Retrying: {e}")
                last_error = e
            else:
                # If above the time threshold, assume query was too large
                # force query split
                print(f"Server busy - responded after {elapsed:.2f} seconds - forcing query split: {e}")
                raise OverpassTooLargeError


        # Execute the retry mechanics - with exponential backoff
        if attempt < max_retries:
            wait = backoff_base * (2 ** (attempt - 1))
            print(f"  Overpass request failed ({last_error}); "
                    f"retrying in {wait:.1f}s (attempt {attempt}/{max_retries})")
            time.sleep(wait)

    # If there was an error after the max number of retries raise it
    if isinstance(last_error, OverpassTooLargeError):
        # Raise a general OverpassError so as to not force a query split 
        raise OverpassError(f"Query was too large - After max retries")
    else:
        raise last_error
    

def fetch_bbox(bbox, collected, depth, query_timeout=DEFAULT_QUERY_TIMEOUT, max_depth=MAX_SPLIT_DEPTH):
    """Recursively fetch a bounding box, splitting whenever it is too large

    The process for each bounding box is:

        1. Run a count probe (see :func:`get_count`). If the count is 0, stop.
        2. If the count is within ``max_elements``, run the full data query.
        3. If the count exceeds ``max_elements``, or the full query exceeds the
           server's limits, split the box with :func:`split_bbox` and repeat for each
           half at ``depth + 1``.

    Args:
        bbox (tuple[float, float, float, float]): The bounding box as
            ``(south, west, north, east)``, i.e. ``(lat_min, lon_min, lat_max,
            lon_max)``.
        collected (dict): Accumulator for results, modified in place. It has the
            form ``{"elements": {(type, id): element}, "meta": dict | None}``, where
            ``"meta"`` holds the non-``elements`` fields of the first response
            received (such as version and generator information).
        depth (int): The current recursion depth. Start at 0.
        query_timeout (int, optional): The Overpass ``timeout`` setting for the full
            data query, in seconds. Defaults to ``DEFAULT_QUERY_TIMEOUT``.
        max_depth (int, optional): The maximum number of times a box may be split.
            Defaults to ``MAX_SPLIT_DEPTH``.
 
    Returns:
        None: Results are stored in ``collected``.
 
    Raises:
        OverpassTooLargeError: If the bounding box still needs splitting once
            ``max_depth`` has been reached.
        OverpassError: Any other error from :func:`run_overpass_query` that is not
            handled by splitting.
    """
    indent = "   " * depth

    try:
        overpass_query = build_overpass_query(bbox, query_timeout=query_timeout)
        data = run_overpass_query(overpass_query, query_timeout=query_timeout)

    except OverpassTooLargeError:
        # If the query was too large - execute recursive split
        print(f"{indent}bbox {bbox}: query exceeded limit")

    else:
        # If there are no errors - collect data and stop recursion
        print(f"{indent}bbox {bbox}: responded successfully\n")

        if collected['meta'] is None:
            collected['meta'] = {k: v for k, v in data.items() if k != 'elements'}
        for element in data.get('elements', []):
            collected['elements'][(element['type'], element['id'])] = element
        return

    # Check if we have exceeded the max recursive depth 
    if depth >= max_depth:
        raise OverpassTooLargeError(f"{bbox} bbox still too large after max recursive splits")

    # Execute recursive call for each split bbox
    print(f"{indent}splitting bbox {bbox}\n")
    for half in split_bbox(bbox):
        time.sleep(REQUEST_PAUSE)
        fetch_bbox(half, collected, depth + 1, query_timeout, max_depth)



# ---------------------------------------------------------------------------
# Main API Workflow
# ---------------------------------------------------------------------------

def get_overpass_with_splitting(bbox, query_timeout=DEFAULT_QUERY_TIMEOUT, max_depth=MAX_SPLIT_DEPTH):
    """Fetch full geometry and tags from the Overpass API given the bounding box area, splitting the bbox if needed

    Uses :func:`fetch_bbox` to recursively split large areas into smaller bounding boxes so that no single request
    exceeds the Overpass limits, then merges all results into one response. 
    
    Args:
        bbox (tuple[float, float, float, float]): The bounding box as
            ``(south, west, north, east)``, i.e. ``(lat_min, lon_min, lat_max,
            lon_max)``.
        query_timeout (int, optional): The Overpass ``timeout`` setting for each
            data query, in seconds. Defaults to ``DEFAULT_QUERY_TIMEOUT``.
        max_depth (int, optional): The maximum number of times a box may be split.
            Defaults to ``MAX_SPLIT_DEPTH``.
 
    Returns:
        dict: The merged Overpass API JSON response body, with the de-duplicated
            results in the ``"elements"`` list.
 
    Raises:
        OverpassTooLargeError: If the area is still too large after ``max_depth``
            splits.
        OverpassBadQueryError: If the query is malformed. This is a bug, not a
            transient condition.
        OverpassError: Any other failure, once retries are exhausted.
    """
    print("\n---------------------------")
    print("Post - Overpass API Request")

    collected = {'elements': {}, 'meta': None}
    fetch_bbox(bbox, collected, depth=0, query_timeout=query_timeout, max_depth=max_depth)

    data = dict(collected['meta'] or {})
    data['elements'] = list(collected['elements'].values())
 
    print("Success - Query Received")
    print("---------------------------")
 
    return data


def get_overpass_no_splitting(bbox, query_timeout=DEFAULT_QUERY_TIMEOUT):
    """Fetch full geometries and tags for a bounding box in a single request.
 
    Sends one data query for the whole bounding box, with no count probe and no
    splitting. It is simpler and faster than :func:`get_overpass` for small areas,
    but fails with :class:`OverpassTooLargeError` if the area is too large for the
    server to handle in one query.
        
    Args:
        bbox (tuple[float, float, float, float]): The bounding box as
            ``(south, west, north, east)``, i.e. ``(lat_min, lon_min, lat_max,
            lon_max)``.
        query_timeout (int, optional): The Overpass ``timeout`` setting, in seconds.
            Defaults to ``DEFAULT_QUERY_TIMEOUT``.
 
    Returns:
        dict: The Overpass API JSON response body.
 
    Raises:
        OverpassTooLargeError: If the area is too large for a single query. Split
            the bounding box and retry the halves, or use :func:`get_overpass`.
        OverpassBadQueryError: If the query is malformed. This is a bug, not a
            transient condition.
        OverpassError: Any other failure, once retries are exhausted.
    """
    print("\n---------------------------")
    print("Post - Overpass API Request")

    query = build_overpass_query(bbox=bbox, query_timeout=query_timeout)
    data = run_overpass_query(query=query, query_timeout=query_timeout)
    
    print("Success - Query Received")
    print("---------------------------")
    
    return data