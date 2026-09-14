import geopandas as gpd
import pandas as pd


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


def combine(df):
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
            # "source_count": len(group_df), # Number of sub-geometries to create the combined one
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

    # Calculate area of the geometries
    # calculate_area(df=combined)

    return combined


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
    boundary_tags = ['protected_area', 'forest', 'forest_compartment', 'national_park', 'aboriginal_lands']
    if pd.notna(boundary) and boundary in boundary_tags:
        return 'protected'

    amenity = row.get('amenity')
    if pd.notna(amenity) and amenity == 'parking':
        return 'parking'

    if pd.notna(row.get('leisure')) or pd.notna(row.get('landuse')) or pd.notna(row.get('natural')):
        return 'green_space'

    return 'other'


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

    # Earlier categories take priority: their geometry is subtracted from
    # later categories before those are combined, so overlapping polygons
    # of different types never get merged into one another.
    priority = ['protected', 'parking', 'green_space', 'other']
    excluded_categories = ['protected']

    # Add the categories to the rows to which they apply
    df = df.copy()
    df['_category'] = df.apply(classify_geometry, axis=1)


    combined_parts = [] # Subset geometries that have been combined
    claimed = None  # running union of geometry already assigned to a category

    for category in priority:
        # Get all of the row entries that were assigned this category 
        subset = df[df['_category'] == category].drop(columns='_category')
        if subset.empty:
            continue

        # If there is an area already 'claimed' by geometries with higher priority
        # crop those areas from the geometries of the current category
        # so that they do not overlap 
        if claimed is not None:
            subset = gpd.overlay(subset, claimed, how="difference")
            if subset.empty:
                continue

        # Within the current category subset
        # merge the geometries that overlap with each other 
        if category not in excluded_categories:
            subset = combine(subset)
            combined_parts.append(subset)

        # Create / expand the large Multipolygon that describes all of the geometry area already 
        # claimed by higher priority category geometries
        category_union = gpd.GeoDataFrame(
            geometry=[subset.geometry.union_all()], crs=df.crs
        )
        claimed = (
            category_union if claimed is None
            else gpd.overlay(claimed, category_union, how="union")
        )

    # If nothing was combined
    if not combined_parts:
        return gpd.GeoDataFrame(columns=df.columns.drop('_category'), crs=df.crs)

    # Create the new dataframe of newly merged-by-category geometries
    result = pd.concat(combined_parts, ignore_index=True)
    result = gpd.GeoDataFrame(result, geometry='geometry', crs=df.crs)

    return result