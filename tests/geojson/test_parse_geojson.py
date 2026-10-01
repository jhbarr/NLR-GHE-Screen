from pathlib import Path
import pytest
import json
 
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon

from nlr_ghe_screen.geojson.parse_geojson import (
    parse_api_response
)


# Columns that parse_api_response must always return
EXPECTED_COLUMNS = {
    "name", "landuse", "leisure", "natural", "boundary", "amenity",
    "geometry", "type", "district_system_type",
}

TEST_BBOX = (39.69, -105.27, 39.79, -105.16)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def make_polygon(x0=-105.20, y0=39.70, size=0.01):
    """A small square polygon"""
    return Polygon([
        (x0, y0), (x0 + size, y0), (x0 + size, y0 + size), (x0, y0 + size), (x0, y0)
    ])
 
 
def make_osmnx_gdf(rows, columns=None, crs="EPSG:4326"):
    """
    Build a GeoDataFrame shaped like the output of ox.features_from_bbox:
    a MultiIndex of (element, id), a geometry column, and one column per tag key.
 
    Parameters:
        rows (list[tuple]): (element, osm_id, geometry, tags_dict)
        columns (list[str]): tag columns to include. Defaults to the union of the tag keys in rows.
        crs (str | None): CRS for the GeoDataFrame
    """
    if columns is None:
        columns = sorted({key for _, _, _, tags in rows for key in tags})
 
    index = pd.MultiIndex.from_tuples(
        [(element, osm_id) for element, osm_id, _, _ in rows],
        names=["element", "id"],
    )
    data = {col: [tags.get(col) for _, _, _, tags in rows] for col in columns}
    data["geometry"] = [geom for _, _, geom, _ in rows]
 
    return gpd.GeoDataFrame(data, index=index, geometry="geometry", crs=crs)
 
 
@pytest.fixture
def osmnx_gdf():
    """
    A representative osmnx response: a park way, a parking way, and a meadow relation
    """
    multipolygon = MultiPolygon([make_polygon(-105.10, 39.70), make_polygon(-105.05, 39.70)])
    return make_osmnx_gdf([
        ("way", 1001, make_polygon(-105.20, 39.70), {"leisure": "park", "name": "Bear Creek Park"}),
        ("way", 1002, make_polygon(-105.18, 39.70), {"amenity": "parking"}),
        ("relation", 2001, multipolygon, {"landuse": "meadow", "name": "Some Meadow"}),
    ])



# ---------------------------------------------------------------------------
# Test - Overpass API Parse
# ---------------------------------------------------------------------------

class TestAPIParse:
    """
    This class tests whether the method to parse the Overpass API response data into a GeoDataframe
    does so correctly and includes all of the necessary fields 
    """

    def test_parse_api_response(self):
        """
        Verify that all desired fields are a part of the parsed Overpass API response
        """
        file_path = Path(__file__).parent / "geojson_data" / "parse_test_data.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            test_data = json.load(file)

        gdf = parse_api_response(test_data, TEST_BBOX, is_df=False)

        assert len(gdf) > 0

    def test_parse_empty_api_response(self):
        """
        Verify correct error is thrown if an API response does not return any data
        """
        empty_response = {
            "version": 0.6,
            "generator": "Overpass API 0.7.62 ...",
            "osm3s": {
                "timestamp_osm_base": "2026-09-11T14:30:00Z",
                "copyright": "The data included in this document is from www.openstreetmap.org. The data is made available under ODbL."
            },
            "elements": []
        }

        with pytest.raises(ValueError):
            gdf = parse_api_response(empty_response, TEST_BBOX, is_df=False)



# ---------------------------------------------------------------------------
# Test - OSMNX API Parse
# ---------------------------------------------------------------------------

class TestOSMNXParse:
    """
    This class tests that an osmnx features GeoDataFrame (is_df=True) is normalized and
    parsed into the same output structure as the Overpass JSON path
    """
 
    def test_basic_parse(self, osmnx_gdf):
        """
        A valid osmnx response returns a non-empty GeoDataFrame with every expected column
        """
        gdf = parse_api_response(osmnx_gdf, TEST_BBOX, is_df=True)
 
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) == 2
        assert EXPECTED_COLUMNS.issubset(gdf.columns)

    def test_output_columns_match_exactly(self, osmnx_gdf):
        """
        No extra osmnx columns (e.g. 'nodes', 'surface') should leak into the output
        """
        osmnx_gdf["surface"] = "grass"
        osmnx_gdf["nodes"] = [[1, 2, 3]] * len(osmnx_gdf)
 
        gdf = parse_api_response(osmnx_gdf, TEST_BBOX, is_df=True)
 
        assert set(gdf.columns) == EXPECTED_COLUMNS

    def test_urbanopt_fields_added(self, osmnx_gdf):
        """
        The required URBANopt schema fields are set on every row
        """
        gdf = parse_api_response(osmnx_gdf, TEST_BBOX, is_df=True)
 
        assert (gdf["type"] == "District System").all()
        assert (gdf["district_system_type"] == "Central Hot Water").all()

    def test_nodes_dropped(self):
        """
        Nodes are removed, matching the old Overpass query that only requested ways and relations
        """
        gdf_in = make_osmnx_gdf([
            ("way", 1, make_polygon(), {"leisure": "park"}),
            ("node", 99, Point(-105.2, 39.7), {"amenity": "parking"}),
        ])
 
        gdf = parse_api_response(gdf_in, TEST_BBOX, is_df=True)
 
        assert len(gdf) == 1
        assert gdf["leisure"].iloc[0] == "park"


# ---------------------------------------------------------------------------
# Test - Overpass and osmnx paths produce the same output structure
# ---------------------------------------------------------------------------
 
class TestParsePathConsistency:
    """
    Downstream code should not care whether the data came from Overpass JSON or osmnx
    """
 
    def test_same_columns_from_both_sources(self, osmnx_gdf):
        """
        Both input types produce the same set of output columns
        """
        file_path = Path(__file__).parent / "geojson_data" / "parse_test_data.json"
        with open(file_path, 'r', encoding='utf-8') as file:
            overpass_data = json.load(file)
 
        from_overpass = parse_api_response(overpass_data, TEST_BBOX, is_df=False)
        from_osmnx = parse_api_response(osmnx_gdf, TEST_BBOX, is_df=True)
 
        assert set(from_overpass.columns) == set(from_osmnx.columns)
        assert from_overpass.crs == from_osmnx.crs