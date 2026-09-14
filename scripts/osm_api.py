import requests
import sys

def get_overpass(bbox):
    """
    Calls the Overpass API using a preset Overpass query with the user inputted
    bounding box coordinates

    Parameters:
        bbox (Tuple[float]): A four float tuple of the form (lat_min, long_min, lat_max, long_max)
    
    Raises:
        RuntimeError: If the API response does not return with a 200 success code
    """
    print("\n---------------------------")
    print("Post - Overpass API Request")

    # Define the Overpass API endpoint
    overpass_url = "https://overpass-api.de/api/interpreter"

    # Parse the bounding box
    south = bbox[0]
    west = bbox[1]
    north = bbox[2]
    east = bbox[3]

    # Create the Overpass QL query
    # *** Need to update to include search for parking lots as well ***
    overpass_query = f"""
    [out:json][timeout:25];

    (
    way["leisure"~"park|dog_park"]({south},{west},{north},{east})
    relation["leisure"~"park|dog_park"]({south},{west},{north},{east});

    way["landuse"~"recreation_ground|meadow|grass|farmland"]({south},{west},{north},{east});
    relation["landuse"~"recreation_ground|meadow|grass|farmland"]({south},{west},{north},{east});

    way["natural"~"wood|grassland|scrub|heath|wetland"]({south},{west},{north},{east});
    relation["natural"~"wood|grassland|scrub|heath|wetland"]({south},{west},{north},{east});

    way["amenity"="parking"]({south},{west},{north},{east});
    relation["amenity"="parking"]({south},{west},{north},{east});
    );

    out geom;
    """
    
    try:
        response = requests.post(
            overpass_url,
            data={'data': overpass_query},
            timeout=25
        )

        # HTTP-level errors (400, 404, 500, etc.)
        response.raise_for_status()

        # Response-level errors, e.g. invalid JSON
        data = response.json()

        print("Success - Query Received")
        print("---------------------------")

        return data, response.status_code

    except requests.exceptions.Timeout:
        raise RuntimeError(
            "The Overpass API request timed out after 25 seconds."
        )

    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(
            f"Could not connect to the Overpass API: {e}"
        )

    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f"Overpass API returned an HTTP error: {response.status_code}"
        ) from e

    except requests.exceptions.JSONDecodeError as e:
        raise RuntimeError(
            "The Overpass API returned a response that was not valid JSON."
        ) from e

    except requests.exceptions.RequestException as e:
        # Catch any other requests-related error
        raise RuntimeError(
            f"An error occurred while making the Overpass API request: {e}"
        ) from e