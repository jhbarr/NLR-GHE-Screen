import geopandas as gpd
from shapely.geometry import Polygon, LineString, MultiPolygon
from shapely.ops import unary_union, polygonize


# ---------------------------------------------------------------------------
# Create geometries + parse API response 
# ---------------------------------------------------------------------------

def create_geometry_object(element):
    """
    Converts an OSM way or relation into a Shapely geometry that can be input to a Geopandas Dataframe

    Parameters:
        element (dict): A single element from an Overpass API response

    Returns:
        shape (shapely.geometry): Either a polygon, multipolygon or way

    Raises:
        TypeError: If the function receives type of geometry that it does not know how to handle
    """
    # Check if the type of the element is a way geometry 
    if element["type"] == "way":
        geometry = element.get("geometry", [])

        coords = [
            (point["lon"], point["lat"])
            for point in geometry
        ]

        # A polygon requires at least 3 coordinates and a closed ring
        if len(coords) >= 3 and coords[0] == coords[-1]:
            return Polygon(coords)

        return None

    # Check if the type of the element is a relation geometry
    elif element["type"] == "relation":
        lines = []

        for member in element.get("members", []):
            geometry = member.get("geometry", [])

            coords = [
                (point["lon"], point["lat"])
                for point in geometry
            ]

            if len(coords) >= 2:
                lines.append(LineString(coords))

        if not lines:
            return None

        # Combine the member ways
        merged = unary_union(lines)

        # Build polygons from the combined lines
        polygons = list(polygonize(merged))

        if not polygons:
            return None

        if len(polygons) == 1:
            return polygons[0]

        return MultiPolygon(polygons)

    # If a non-recognized geometry type is passed, throw an error
    else:
        raise TypeError(f"Unrecognized geometry type: {element['type']}")


def parse_api_response(data):
    """
    This function will take a JSON response object from the Overpass API call and add all necessary fields for
    the URBANopt GeoJSON format while also isolating the other fields relevant to the query

    Parameters:
        data (dict): A JSON object that is the response provided by the Overpass API
    
    Returns:
        gdf (GeoDataframe): A GeoDataframe with all necessary fields

    Raises:
        ValueError: If the API data that is passed to the function is empty
            or if there are no valid geometries found while parsing
        TypeError: If the function receives type of geometry that it does not know how to handle
    """
    print("\n---------------------------")
    print("Parsing API Results")

    # Get the elements of the Overpass API response
    elements = data.get('elements', [])

    # Check that the data is populated and not empty / null
    if not elements:
        raise ValueError("API response is empty")

    # Go through each of the elements from the response 
    rows = []
    for element in elements:

        # Create a geometry object that can be recognized by Geopandas
        # from the provided coordinates in the Overpass API JSON response body
        geometry = create_geometry_object(element)

        if geometry is None:
            continue

        # Create a dataframe row entry based on the elements features 
        row = {
            **element.get("tags", {}),
            "geometry": geometry # Overpass API always has geometry descriptions of elements
        }

        rows.append(row)

    if not rows:
        raise ValueError("No valid geometries found")

    gdf = gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs="EPSG:4326"
    )

    desired_attributes = [
        'name', 
        'landuse', 
        'leisure', 
        'natural', 
        'boundary', # For checks on protected areas
        'amenity', # For information on parking areas 
        'geometry'
    ]
    gdf = gdf.reindex(columns=desired_attributes)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs="EPSG:4326")

    # Add required fields for URBANopt GeoJSON schema
    gdf['type'] = "District System"
    gdf['district_system_type'] = "Central Hot Water"
    gdf['name'] = gdf['name'].fillna('') # URBANopt schema cannot have empty name fields
    

    print("Parsing Successful - All necessary data")
    print("---------------------------") 

    return gdf