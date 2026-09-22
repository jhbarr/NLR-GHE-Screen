import geopandas as gpd
import pandas as pd
from shapely import make_valid
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union


# ---------------------------------------------------------------------------
# Create + Combine geometry groups
# ---------------------------------------------------------------------------

def create_overlapping_groups(df):
    """
    This function handles creating a new dataframe of combined geometries based on pre-determined
    groups of geometries

    Parameters:
        df (GeoDataframe): A GeoDataframe with only polygon and multipolygon geometries

    Returns:
        groups (list[str]): A list of lists, each index in each sublist corresponds with a geometry that overlaps all others in the group
    """
    # Make sure geometries are valid
    df["geometry"] = df.geometry.make_valid()

    # Create a spatial index
    sindex = df.sindex

    # Build groups of overlapping/touching polygons
    groups = []
    visited = set()

    # Go through each of the unique geometries in the dataframe 
    # and combine the geometries that are overlapping with each other
    # using DFS
    for idx in df.index:
        if idx in visited:
            continue

        group = {idx}
        stack = [idx]

        # Keep checking if there are other geometries that overlap with the group that has
        # been gathered so far
        while stack:
            current = stack.pop()

            # Find polygons whose bounding boxes intersect
            candidates = list(
                sindex.query(
                    df.loc[current, "geometry"],
                    predicate="intersects"
                )
            )

            for candidate in candidates:
                if candidate not in group:
                    group.add(candidate)
                    stack.append(candidate)

        visited.update(group)
        groups.append(group)

    return groups


def combine_overlapping_groups(df):
    """
    This function combines Geodataframe rows based on whether they contain overlapping geometries
    It returns a new and condensed dataframe based on these overlaps

    Parameters:
        df (GeoDataframe): A GeoDataframe with only polygon and multipolygon geometries

    Returns:
        result (GeoDataframe): A new df with combined row entries and geometries
    """
    # To avoid any issues with indexing the rows 
    df = df.reset_index(drop=True)

    # Create groups of overlapping geometries 
    groups = create_overlapping_groups(df=df)

    # Create list to hold all of the rows in the final combine dataframe
    combined_rows = []

    # Go through each of the groups of geographic areas that have been calculated to overlap
    for group_number, group in enumerate(groups, start=1):
        # Get all of the geometries from the group and combine them into one 
        # conglomerate geometry 
        group_df = df.loc[list(group)].copy()
        combined_geometry = group_df.geometry.union_all()

        # Create a new row for the final dataframe
        row = {
            "geometry": combined_geometry,
        }

        # Go through each of the attributes that we want to gather and apend them to 
        # the final row 
        column_list = df.columns.to_list()
        column_list.remove('geometry')
        for column in column_list:
            if column not in group_df.columns:
                continue
    
            # Go through each value in the selected attribute column (from the group)
            # and combine them into one tuple
            values = (
                group_df[column]
                .dropna()
                .astype(str)
                .unique()
            )

            # Join all non-unique values 
            row[column] = "; ".join(values)
    
        combined_rows.append(row)

    # Create the final dataframe with the new combined geometries and the 
    # aggregated attributes
    combined = gpd.GeoDataFrame(
        combined_rows,
        crs=df.crs
    )

    return combined



# ---------------------------------------------------------------------------
# Classify geometry objects
# ---------------------------------------------------------------------------

def classify_geometry(row):
    """
    Assigns a single category label to a row based on its OSM tags.
    Order matters: this determines priority when a geometry could 
    plausibly belong to multiple categories (e.g., a tagged protected 
    park should be treated as 'protected', not 'green_space').

    Parameters:
        row (GeoSeries): A single row from a Geodataframe

    Returns:
        category (Str): A string categorizing the row 
    """
    boundary = row.get('boundary')
    boundary_tags = [
        'protected_area', 
        'forest', 
        'forest_compartment', 
        'national_park', 
        'aboriginal_lands'
    ]
    if pd.notna(boundary) and boundary in boundary_tags:
        return 'protected'

    amenity = row.get('amenity')
    amenity_tags = [
        'parking'
    ]
    if pd.notna(amenity) and amenity in amenity_tags:
        return 'parking'

    if (
        pd.notna(row.get('leisure')) and str(row.get('leisure')).strip()
    ) or (
        pd.notna(row.get('landuse')) and str(row.get('landuse')).strip()
    ) or (
        pd.notna(row.get('natural')) and str(row.get('natural')).strip()
    ):
        return 'green_space'

    return 'other'



# ---------------------------------------------------------------------------
# Geodata cleaner functions
# ---------------------------------------------------------------------------

def _polygonal_only(geom):
    """
    Return only the (Multi)Polygon part of a geometry, or None if there is none
    
    Parameters:
        geom (Geometry object): A Geopandas geometry object 
    
    Returns:
        Geometry object: A unary union (combined) geometry object 
    """
    if geom is None or geom.is_empty:
        return None
    
    if not geom.is_valid:
        geom = make_valid(geom)

    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    
    if isinstance(geom, GeometryCollection) or hasattr(geom, "geoms"):
        polys = []
        for g in geom.geoms:
            g = _polygonal_only(g)
            if g is not None:
                polys.extend(g.geoms if isinstance(g, MultiPolygon) else [g])

        if polys:
            return unary_union(polys) if len(polys) > 1 else polys[0]
        
    return None  # lines / points are discarded


def _clean(gdf):
    """
    Make geometries valid, strip non-polygon parts, drop empties
    
    Parameters:
        gdf (Geopandas Dataframe): A Dataframe containing geospatial data

    Returns:
        Geopandas Dataframe: A Dataframe containing geospatial data
    """
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(_polygonal_only)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    return gdf



# ---------------------------------------------------------------------------
# Main combination execution
# ---------------------------------------------------------------------------

def combine_geometries(df):
    """
    Splits the dataframe into mutually-exclusive categories (protected areas,
    parking lots, green spaces, etc.), combines overlapping geometries within
    each category, and ensures higher-priority categories "claim" any
    spatially overlapping area so different categories never get merged
    together.

    Parameters:
        df (GeoDataframe): A GeoDataframe with only polygon and multipolygon geometries

    Returns:
        result (GeoDataframe): A new df with combined row entries and geometries
    """
    print("\n---------------------------")
    print("Combining overlapping geometries")

    # Establish the priority of space categorization
    # As well as which categories should be excluded from final results
    priority = ['protected', 'parking', 'green_space', 'other']
    excluded_categories = ['protected']

    df = _clean(df)
    df['_category'] = df.apply(classify_geometry, axis=1)

    combined_parts = []
    claimed = None  # a single shapely geometry now, not a GeoDataFrame

    for category in priority:
        subset = df[df['_category'] == category].drop(columns='_category')
        if subset.empty:
            continue

        # Subtract higher-priority area with plain shapely, not gpd.overlay
        if claimed is not None:
            subset = subset.copy()
            subset["geometry"] = subset.geometry.difference(claimed)
            subset = _clean(subset)
            if subset.empty:
                continue

        # Combine all overlapping geometries within the category 
        # if that category has not been highlighed for exclusion
        if category not in excluded_categories:
            subset = _clean(combine_overlapping_groups(subset))
            combined_parts.append(subset)

        print(f"Combining category: {category}")

        # Either create or add to the 'combined' data structure
        # which is a conglomerate of all previously examined geometries
        # as to not allow overlap between geometries of different categories
        category_union = _polygonal_only(subset.geometry.union_all())
        if category_union is not None:
            claimed = (
                category_union if claimed is None
                else _polygonal_only(claimed.union(category_union))
            )

    if not combined_parts:
        return gpd.GeoDataFrame(columns=df.columns.drop('_category'), crs=df.crs)

    result = pd.concat(combined_parts, ignore_index=True)
    result = gpd.GeoDataFrame(result, geometry='geometry', crs=df.crs)

    print("Combination Successful")
    print("---------------------------")
    return result