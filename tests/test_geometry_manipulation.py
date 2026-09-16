from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
import pandas as pd
from geopandas.testing import assert_geodataframe_equal

from scripts.geometry_manipulation import (
    create_overlapping_groups, 
    combine_geometries, 
    classify_geometry
)



# ---------------------------------------------------------------------------
# Test - Combine Overlapping Geometries 
# ---------------------------------------------------------------------------

class TestGeometryOverlap:
    """
    This class will handle testing if the functionality for combining overlapping geometries 
    into one large geometry works properly
    """

    def test_overlap_polygon(self):
        """
        Test that rows in a Geopandas Dataframe with overlapping polygons will be combined 
        into one row that shares attributes
        """
        # Create test data
        test_data = {
            "type": ["District System", "District System"],
            "district_system_type": ["Central Hot Water", "Central Hot Water"],
            "geometry": [
                Polygon([
                    (0, 0),
                    (4, 0),
                    (4, 4),
                    (0, 4),
                    (0, 0)
                ]),
                Polygon([
                    (2, 2),
                    (6, 2),
                    (6, 6),
                    (2, 6),
                    (2, 2)
                ])
            ]
        }

        df = gpd.GeoDataFrame(test_data, crs="EPSG:4326")
        groups = create_overlapping_groups(df)

        assert len(groups) == 1
        

    def test_overlap_multipolygon_and_polygon(self):
        """
        Test that rows in a Geopandas Dataframe with overlapping polygons and multipolygons will be combined 
        into one row that shares attributes
        """
        poly2_a = Polygon([
            (2, 2),
            (6, 2),
            (6, 6),
            (2, 6),
            (2, 2)
        ])

        poly2_b = Polygon([
            (20, 20),
            (22, 20),
            (22, 22),
            (20, 22),
            (20, 20)
        ])

        # Create test data
        test_data = {
            "type": ["District System", "District System"],
            "district_system_type": ["Central Hot Water", "Central Hot Water"],
            "geometry": [
                Polygon([
                    (0, 0),
                    (4, 0),
                    (4, 4),
                    (0, 4),
                    (0, 0)
                ]),
                MultiPolygon([
                    poly2_a,
                    poly2_b
                ])
            ]
        }

        df = gpd.GeoDataFrame(test_data, crs="EPSG:4326")
        groups = create_overlapping_groups(df)

        assert len(groups) == 1


    def test_non_overlapping_geometries(self):
        """
        Verifies that if two geometries do not overlap, they will not be grouped together as overlapping
        """
        # Create test data
        test_data = {
            "type": ["District System", "District System"],
            "district_system_type": ["Central Hot Water", "Central Hot Water"],
            "geometry": [
                Polygon([
                    (0, 0),
                    (2, 0),
                    (2, 2),
                    (0, 2),
                    (0, 0)
                ]),
                Polygon([
                    (4, 4),
                    (6, 4),
                    (6, 6),
                    (4, 6),
                    (4, 4)
                ])
            ]
        }

        df = gpd.GeoDataFrame(test_data, crs="EPSG:4326")
        groups = create_overlapping_groups(df)

        assert len(groups) == 2



# ---------------------------------------------------------------------------
# Test - Dataframe consolidation of overlapping geometries
# ---------------------------------------------------------------------------

class TestGeometryCombination:
    """
    This class verifies that the behavior of the procedure handling combining geometries into condensed
    dataframes performs correctly
    """

    def test_overlapping_geometries(self):
        """
        Tests the dataframe created when the original includes overlapping geometries
        """
        # Create test data
        test_data = {
            "type": ["District System", "District System"],
            "district_system_type": ["Central Hot Water", "Central Hot Water"],
            "name": ["Park1", "Park2"],
            "landuse": ["recreation", "recreation"],
            "geometry": [
                Polygon([
                    (0, 0),
                    (4, 0),
                    (4, 4),
                    (0, 4),
                    (0, 0)
                ]),
                Polygon([
                    (2, 2),
                    (6, 2),
                    (6, 6),
                    (2, 6),
                    (2, 2)
                ])
            ]
        }

        df = gpd.GeoDataFrame(test_data, crs="EPSG:4326")
        result_df = combine_geometries(df)

        final_data = {
            "type": ["District System"],
            "district_system_type": ["Central Hot Water"],
            "name": ["Park1; Park2"],
            "landuse": ["recreation"],
            "geometry": [
                Polygon([
                    (4, 0),
                    (0, 0),
                    (0, 4),
                    (2, 4), 
                    (2, 6),
                    (6, 6),
                    (6, 2),
                    (4, 2),
                    (4, 0)
                ])
            ],
        }
        exppected_df = gpd.GeoDataFrame(final_data, crs="EPSG:4326")

        assert_geodataframe_equal(
            result_df,
            exppected_df,
            check_like=True
        )

    def test_non_overlapping_geometries(self):
        """
        Tests the dataframe created when the original does not include overlapping geometries
        """
        test_data = {
            "type": ["District System", "District System"],
            "district_system_type": ["Central Hot Water", "Central Hot Water"],
            "geometry": [
                Polygon([
                    (0, 0),
                    (2, 0),
                    (2, 2),
                    (0, 2),
                    (0, 0)
                ]),
                Polygon([
                    (4, 4),
                    (6, 4),
                    (6, 6),
                    (4, 6),
                    (4, 4)
                ])
            ]
        }

        expected_df = gpd.GeoDataFrame(test_data, crs="EPSG:4326")
        result_df = combine_geometries(expected_df)

        assert_geodataframe_equal(
            expected_df,
            result_df,
            check_like=True
        )



# ---------------------------------------------------------------------------
# Test - Attribute classification
# ---------------------------------------------------------------------------

class TestClassifyGeometry:
    """
    These test cases verify that geometries are being correctly attributed to the correct category 
    for exclusion and geometry merging purposes
    """

    def test_category_assignment(self):
        """
        Check that green space, parking and protected areas are all being assigned correctly to a category
        """
        test_data = {
            "name": ["Park1", "", "Protected Park 1", "Space"],
            "landuse": ["recreation", "", "recreation_ground", ""],
            "boundary": ["", "", "national_park", ""],
            "amenity": ["", "parking", "", ""],
        }

        expected_data = {
            "name": ["Park1", "", "Protected Park 1", "Space"],
            "landuse": ["recreation", "", "recreation_ground", ""],
            "boundary": ["", "", "national_park", ""],
            "amenity": ["", "parking", "", ""],
            "_category": ["green_space", "parking", "protected", "other"]
        }

        test_df = pd.DataFrame(test_data)
        expected_df = pd.DataFrame(expected_data)
        test_df['_category'] = test_df.apply(classify_geometry, axis=1)

        print(test_df)
        print()
        print(expected_df)

        assert test_df.equals(expected_df)