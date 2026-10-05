import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import nlr_ghe_screen.processing.elevation as elv
from nlr_ghe_screen.processing.elevation import (
    load_elevation_data,
    reproject_elevation_data,
    calculate_slopes,
    clip_raster_to_shapes,
    categorize_geometries
)
 
 
CRS = "EPSG:32618"  # UTM 18N, so units are metres
RES = 10.0          # 10 m pixels
SIZE = 100          # 100 x 100 pixels -> 1 km x 1 km
X0, Y0 = 500_000.0, 4_300_000.0  # top-left corner
NODATA = -9999.0

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------
 
def make_dem(percent_by_col):
    """Build a DEM that rises eastward with the given % slope for each column."""
    rise_per_col = np.asarray(percent_by_col, dtype=float) / 100 * RES
    row = np.cumsum(rise_per_col)
    return np.tile(row, (SIZE, 1)).astype("float32")
 
 
def write_dem(path, dem):
    with rasterio.open(
        path, "w", driver="GTiff",
        height=dem.shape[0], width=dem.shape[1], count=1,
        dtype="float32", crs=CRS,
        transform=from_origin(X0, Y0, RES, RES),
        nodata=NODATA,
    ) as dst:
        dst.write(dem, 1)
 
 
def make_gdf(*boxes):
    return gpd.GeoDataFrame(geometry=list(boxes), crs=CRS)
 
 
@pytest.fixture
def imports_dir(tmp_path, monkeypatch):
    """Point the module's IMPORT_DIR at a temp folder."""
    monkeypatch.setattr(elv, "IMPORT_DIR", tmp_path)
    return tmp_path
 
 
@pytest.fixture
def uniform_10pct(imports_dir):
    """DEM with a uniform 10% slope, saved as elevation_data.tif."""
    write_dem(imports_dir / "elevation_data.tif", make_dem([10] * SIZE))
    return imports_dir
 
 
@pytest.fixture
def inner_gdf():
    """One polygon well inside the raster, away from edge effects."""
    return make_gdf(box(X0 + 200, Y0 - 800, X0 + 800, Y0 - 200))



# ---------------------------------------------------------------------------
# Test - load_elevation_data
# ---------------------------------------------------------------------------
 
class TestLoadElevationData:
    """Test that the elevation data can be loaded properly"""

    def test_returns_open_dataset(self, uniform_10pct, inner_gdf):
        src = load_elevation_data(inner_gdf)
        try:
            assert not src.closed
            assert src.count == 1
            assert src.crs == rasterio.crs.CRS.from_string(CRS)
        finally:
            src.close()
 
    def test_missing_file_raises(self, imports_dir, inner_gdf):
        with pytest.raises(FileNotFoundError):
            load_elevation_data(inner_gdf)



# ---------------------------------------------------------------------------
# Test - reproject_elevation_data
# ---------------------------------------------------------------------------
 
class TestReprojectElevationData:
    """Test that the data can be reprojected to the correct CRS for metric calculations"""

    def test_writes_float32_file_in_new_crs(self, uniform_10pct):
        with rasterio.open(uniform_10pct / "elevation_data.tif") as src:
            reproject_elevation_data(src, dst_crs="EPSG:32617")
 
        out_path = uniform_10pct / "reprojected_elevation_data.tif"
        assert out_path.exists()
        with rasterio.open(out_path) as out:
            assert out.crs.to_epsg() == 32617
            assert out.dtypes[0] == "float32"
            assert out.nodata == NODATA



# ---------------------------------------------------------------------------
# Raster organization and calculations
# ---------------------------------------------------------------------------
 
class TestCalculateSlopes:
    """Test that the slope gradients are calculated properly"""

    def test_uniform_plane(self, uniform_10pct):
        with rasterio.open(uniform_10pct / "elevation_data.tif") as src:
            slopes = calculate_slopes(src)
 
        assert slopes.shape == (SIZE, SIZE)
        assert slopes.dtype == np.float32
        # A plane rising 10 m per 100 m should give 10% everywhere
        assert np.allclose(slopes, 10.0, atol=0.1)
        assert (uniform_10pct / "slope.tif").exists()
 
    def test_flat_surface_is_zero(self, imports_dir):
        write_dem(imports_dir / "elevation_data.tif", make_dem([0] * SIZE))
        with rasterio.open(imports_dir / "elevation_data.tif") as src:
            slopes = calculate_slopes(src)

        assert np.allclose(slopes, 0.0)


class TestClipRasterToShapes:
    """Test that raster data is properly attributed to vector shapes"""

    def test_returns_masked_array_per_geometry(self, uniform_10pct, inner_gdf):
        with rasterio.open(uniform_10pct / "elevation_data.tif") as src:
            gdf_out, clips = clip_raster_to_shapes(src, inner_gdf)
 
        assert gdf_out is inner_gdf
        assert list(clips) == [0]
        arr, transform = clips[0]
        assert isinstance(arr, np.ma.MaskedArray)
        assert arr.compressed().size > 0
        assert transform is not None
 
    def test_skips_non_overlapping_geometry(self, uniform_10pct, capsys):
        far_away = make_gdf(box(X0 + 50_000, Y0 + 50_000, X0 + 51_000, Y0 + 51_000))
        with rasterio.open(uniform_10pct / "elevation_data.tif") as src:
            _, clips = clip_raster_to_shapes(src, far_away)
 
        assert clips == {}
        assert "Skipping geometry" in capsys.readouterr().out



# ---------------------------------------------------------------------------
# Categorization
# ---------------------------------------------------------------------------

class TestCategorizeGeometries:
    """Test that geometries are correctly assigned a gradient category"""

    def test_assigns_dominant_bin(self):
        gdf = make_gdf(*[box(i, 0, i + 1, 1) for i in range(4)])
        clips = {
            0: (np.ma.masked_array([1.0, 2.0, 3.0]), None),     # flat
            1: (np.ma.masked_array([6.0, 7.0, 8.0]), None),     # mild
            2: (np.ma.masked_array([12.0, 15.0, 18.0]), None),  # medium
            3: (np.ma.masked_array([30.0, 40.0, 50.0]), None),  # steep
        }
 
        categorize_geometries(gdf, clips)
 
        assert gdf["slope_category"].tolist() == ["flat", "mild", "medium", "steep"]
 
    def test_missing_clip_gets_nan(self):
        gdf = make_gdf(box(0, 0, 1, 1), box(1, 0, 2, 1))
        clips = {0: (np.ma.masked_array([1.0, 2.0]), None)}
 
        categorize_geometries(gdf, clips)
 
        assert gdf.loc[0, "slope_category"] == "flat"
        assert gdf["slope_category"].isna().iloc[1]