import geopandas as gpd
import pandas as pd
from shapely import make_valid
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union


# ---------------------------------------------------------------------------
# Create + Combine geometry groups
# ---------------------------------------------------------------------------

def create_overlapping_groups(df):
    """Group rows whose geometries overlap or touch, directly through a chain

    Use spatial index and depth first search to find connected clusters of geometries. Two geometries are
    connected if they intersect. Every row ends up in exactly one group, and a geometry that does not intersect
    anything else forms a group of its own. 

    Args:
        df (geopandas.GeoDataFrame): Rows with only Polygon and MultiPolygon
            geometries and a default ``RangeIndex``.
 
    Returns:
        list[set]: One set per group, each holding the index values of the rows in
            that group.
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
    """Merge rows with overlapping geometries into single rows. 

    Finds groups of overlapping or touching geometries (see :func:`create_overlapping_groups`)
    and collapses each group into one row and conglomerated shape. The new row is of the form:

        - **geometry**: the union of all geometries in the group.
        - **every other column**: the unique, non-null values in the group, converted
          to strings and joined with ``"; "``. A column that is entirely null within
          a group becomes an empty string.
        
    Attributes are joined as text, so all non-geometry columns will become strings regardless of their original 
    dtype. 

    Args:
        df (geopandas.GeoDataFrame): Rows with only Polygon and MultiPolygon
            geometries.
 
    Returns:
        geopandas.GeoDataFrame: One row per group of overlapping geometries.
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
    """Assign a category label to a row / geometry based on its OSM tag

    Checks the tags in priority order and returns the first match, so a geometry that could fall into several
    categories gets teh highest priority one. Missing tag columns are treated as empty. 

    The categories are:

        1. ``'protected'``
        2. ``'parking'``
        3. ``'greenspace'``
        4. ``'other'``

    Args:
        row (pandas.Series): A single row from a GeoDataFrame, expected to contain
            some of the OSM tag columns ``boundary``, ``amenity``, ``leisure``,
            ``landuse``, and ``natural``.
 
    Returns:
        str: One of ``"protected"``, ``"parking"``, ``"green_space"``, or ``"other"``.
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

    water = row.get('natural')
    water_bodies = row.get('water')
    if (pd.notna(water) and water == 'water') or (pd.notna(water_bodies)):
        return 'water'

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
    """Extracts only the polygon parts of a geometry

    Invalid geometries are repaired with``make_valid`` first, which can turn a single geometry into a collection of mixed
    types. Polygons and MultiPolygons are returned as they are. For any other
    multi-part geometry, the function recurses into each part, keeps the polygonal
    pieces, and merges them with a union
    
    Args:
        geom (shapely.geometry.base.BaseGeometry | None): The geometry to filter.
 
    Returns:
        shapely.geometry.Polygon | shapely.geometry.MultiPolygon | None: The
            polygonal part of ``geom``, or ``None`` if it has none. 
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
    """Reduce a GeoDataFrame to valid, non-empty polygon geometries.
 
    Applies :func:`_polygonal_only` to every geometry (repairing invalid ones and
    stripping any non-polygon parts), then drops rows whose geometry is missing or
    empty. The input is not modified.
 
    Args:
        gdf (geopandas.GeoDataFrame): Data with a ``geometry`` column.
 
    Returns:
        geopandas.GeoDataFrame: A copy containing only rows with valid, non-empty
            Polygon or MultiPolygon geometries.
    """
    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(_polygonal_only)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    return gdf



# ---------------------------------------------------------------------------
# Main combination execution
# ---------------------------------------------------------------------------

def combine_geometries(df):
    """Combine overlapping geometries within mutually exclusive categories. 

    Classifies each geometry with :func:`classify_geometry`, the processes the categories from highest to lowest 
    priority: ``protected``, ``parking``,
    ``green_space``, ``other``. For each category:
 
        1. The area already claimed by higher-priority categories is subtracted, so
           categories never overlap and are never merged together.
        2. Unless the category is excluded, overlapping geometries within it are
           merged with :func:`combine_overlapping_groups` and added to the output.
        3. The category's remaining area is added to the claimed area.

    ``protected`` is excluded from the output but still claims its area, so
    lower-priority categories are cut back around protected land instead of
    overlapping it. 

    Args:
        df (geopandas.GeoDataFrame): Rows with only Polygon and MultiPolygon
            geometries and OSM tag columns such as ``boundary``, ``amenity``,
            ``leisure``, ``landuse``, and ``natural``.
 
    Returns:
        geopandas.GeoDataFrame: The combined geometries from all non-excluded
            categories, with the input's CRS. If nothing remains, an empty
            GeoDataFrame with the same columns as ``df`` is returned.
    """
    print("\n---------------------------")
    print("Combining overlapping geometries")

    # Establish the priority of space categorization
    # As well as which categories should be excluded from final results
    priority = ['protected', 'parking', 'water', 'green_space', 'other']
    excluded_categories = ['protected']

    df = _clean(df)
    df['_category'] = df.apply(classify_geometry, axis=1)

    combined_parts = []
    claimed = None  # a single shapely geometry now, not a GeoDataFrame

    for category in priority:
        # subset = df[df['_category'] == category].drop(columns='_category')
        subset = df[df['_category'] == category]
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