import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
from shapely import make_valid
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union

from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


# ---------------------------------------------------------------------------
# Create + Combine geometry groups
# ---------------------------------------------------------------------------

def create_group_labels(gdf):
    """Returns group id assignments for all of the geometries in a GeoDataframe

    This function takes in a GeoDataframe and then provides labels for all clusters of geometries that overlap with each 
    other. Therefore, all geometries that overlap will be assigned a cluster label, indicating that they overlap. 
    This is done by running a spatial query on the gdf geometry values and the predicate ``'intersects``. 

    A sparse adjacency matrix is then created out of these pairs using scipy.coo_matrix (coordinate matrix). Then, 
    the scipy.connected_components function is run on this graph to assign cluster labels to all different nodes in
    the graph that overlap each other. 

    The function returns an array 'labels' of length n, where n is the number of original geometries. labels[i] = group id
    of row i. i corresponds to the ith geometry in the GeoDataframe, not to the geometries true corresponding index. 

    Args:
        gdf (geopandas.GeoDataframe): A GeoDataframe containing OSM information and geometries
    
    Returns:
        list[int]: The list of cluster assignments
    """

    n = len(gdf)

    # Returns a 2 x M array of pairs
    # where each column is a pair of geometries that overlap
    # Unpacking it via left, right gives the two rows and separate arrays
    left, right = gdf.sindex.query(gdf.geometry.values, predicate="intersects")

    # Create a 'sparse' adjacency matrix that stores the query information regarding which 
    # geometries overlap with each other 
    graph = coo_matrix((np.ones(len(left), dtype=bool), (left, right)), shape=(n, n))

    # Takes in a sparse graph and returns the cluster label assignments for each geometry
    _, labels = connected_components(graph, directed=False)

    return labels  # labels[i] = group id of row i


def _join_unique(s):
    return "; ".join(pd.unique(s.dropna()))


def combine_overlapping_groups(gdf):
    """Combines geometries based on whether they overlap with each other

    This function first assigns cluster labels to all geometries. Geometries with the same cluster label
    either directly or transitively overlap pwith each other. All cluster groups are then 'dissolved' into 
    one large geometry. 

    Args:
        gdf (geopandas.GeoDataframe): A GeoDataframe containing OSM information and geometries
    
    Returns:
        geopandas.GeoDataframe: A new GeoDataframe with the combined geometries and their combined attributes
    """
    # Reset the index so that the 'create_group_label' function works properly
    df = gdf.reset_index(drop=True)

    # Assign all of the attribute columns a string type 
    attr_cols = [c for c in df.columns if c != "geometry"]
    df[attr_cols] = df[attr_cols].astype(str)  # stringify once, keeps NA

    labels = create_group_labels(df)
    sizes = np.bincount(labels)
    is_multi = sizes[labels] > 1

    # Singletons: nothing to merge
    singles = df[~is_multi].copy()
    singles[attr_cols] = singles[attr_cols].fillna("")

    # Multi-row groups: one dissolve call
    multi = df[is_multi].assign(_g=labels[is_multi])
    merged = multi.dissolve(
        by="_g", 
        aggfunc={c: _join_unique for c in attr_cols},
        as_index=False).drop(columns="_g")

    return gpd.GeoDataFrame(
        pd.concat([singles, merged], ignore_index=True),
        geometry="geometry", 
        crs=df.crs
    )




# ---------------------------------------------------------------------------
# Classify geometry objects
# ---------------------------------------------------------------------------

def classify_geometries(gdf):
    """Assign a category label to a row / geometry based on its OSM tag

    Checks the tags in priority order and returns the first match, so a geometry that could fall into several
    categories gets teh highest priority one. Missing tag columns are treated as empty. 

    The categories are:

        1. ``'protected'``
        2. ``'parking'``
        3. ``'greenspace'``
        4. ``'other'``

    Args:
        gdf (geopandas.GeoDataframe): A GeoDataframe containing OSM information and geometries
 
    Returns:
        geopandas.GeoSeries: Each entry is either ``"protected"``, ``"parking"``, ``"green_space"``, or ``"other"``.
    """
    PROTECTED = ["protected_area", "forest", "forest_compartment", "national_park", "aboriginal_lands"]

    def col(name):
        return gdf[name] if name in gdf else pd.Series(None, index=gdf.index, dtype=object)
    def nonblank(name):
        return col(name).notna() & col(name).astype(str).str.strip().ne("")

    conditions = [
        col("boundary").isin(PROTECTED),
        col("waste_heat_source").notna(),
        col("amenity").eq("parking"),
        col("natural").eq("water") | col("water").notna(),
        nonblank("leisure") | nonblank("landuse") | nonblank("natural"),
    ]
    choices = ["protected", "waste_heat", "parking", "water", "green_space"]

    return np.select(conditions, choices, default="other")



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
    invalid = ~gdf.geometry.is_valid
    if invalid.any():
        gdf.loc[invalid, "geometry"] = gdf.loc[invalid, "geometry"].make_valid()
    odd = ~gdf.geom_type.isin(["Polygon", "MultiPolygon"])
    if odd.any():
        gdf.loc[odd, "geometry"] = gdf.loc[odd, "geometry"].apply(_polygonal_only)

    keep = ~gdf.geometry.isna() & ~gdf.geometry.is_empty
    return gdf[keep]


# ---------------------------------------------------------------------------
# Main combination execution
# ---------------------------------------------------------------------------

def subtract_claimed(subset, claimed):
    """

    """
    sub_idx, claim_idx = claimed.sindex.query(subset.geometry.values, predicate="intersects")
    if len(sub_idx) == 0:
        return subset

    claimed_arr = np.asarray(claimed.values)
    cutters = (pd.Series(claim_idx).groupby(sub_idx)
               .agg(lambda c: shapely.union_all(claimed_arr[c.values])))

    arr = np.asarray(subset.geometry.values).copy()
    arr[cutters.index.values] = shapely.difference(arr[cutters.index.values], cutters.values)

    subset = subset.copy()
    subset["geometry"] = gpd.GeoSeries(arr, index=subset.index, crs=subset.crs)
    return _clean(subset)


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

    priority = ['protected', 'waste_heat', 'water', 'parking', 'green_space', 'other']
    excluded_categories = {'protected'}

    df = _clean(df).reset_index(drop=True).copy()
    df['_category'] = classify_geometries(df)

    combined_parts = []   # output pieces from non-excluded categories
    claimed_parts = []    # geometry of every processed category, used for subtraction
    claimed = None        # GeoSeries built from claimed_parts

    for category in priority:
        subset = df[df['_category'] == category]
        if subset.empty:
            continue

        # Subtract only the claimed pieces that actually touch each geometry
        if claimed is not None:
            subset = subtract_claimed(subset, claimed)
            if subset.empty:
                continue

        # Merge overlapping geometries within the category (unless excluded)
        if category not in excluded_categories:
            subset = _clean(combine_overlapping_groups(subset))
            if subset.empty:
                continue
            combined_parts.append(subset)

        print(f"Combining category: {category}")

        # Whatever is left in this category (excluded or not) is now claimed.
        # No union needed: subtract_claimed unions only the nearby pieces on demand.
        claimed_parts.append(subset.geometry.reset_index(drop=True))
        claimed = gpd.GeoSeries(
            pd.concat(claimed_parts, ignore_index=True), crs=df.crs
        )

    if not combined_parts:
        return gpd.GeoDataFrame(columns=df.columns, crs=df.crs)

    result = pd.concat(combined_parts, ignore_index=True)
    result = gpd.GeoDataFrame(result, geometry='geometry', crs=df.crs)

    print("Combination Successful")
    print("---------------------------")

    return result