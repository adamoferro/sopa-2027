#!/usr/bin/python3

#   Copyright 2016-2019 Adamo Ferro
#
#   This file is part of SOPA.
#
#   SOPA is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   SOPA is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with SOPA. If not, see <http://www.gnu.org/licenses/>.
#
#   The use of SOPA or part of it for the creation of any sub-product
#   (e.g., scientific papers, posters, images, other softwares)
#   must be acknowledged.

"""Unit tests of the DEM readers (sopa.dem.dem_base, sopa.dem.mola.mola).

The base class is tested directly (it is directly instantiable and defines
the interface contract), the MOLA/MOLA128 readers on a small synthetic ENVI
DEM written through sopa.io_utils.envi, covering pixel/radian conversions,
radius lookups with the dummy-value mask, sub-image extraction with the MOLA
offset, the smoothed/upscaled sub-DEM extraction, and the read/reset states.
"""

import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import scipy.ndimage as snd

from sopa.dem.dem_base import DEMBase
from sopa.dem.mola.mola import MOLA, MOLA128
from sopa.io_utils import envi

DEM_ROWS = 256
DEM_COLS = 128
MOLA_OFFSET = 3396000


@pytest.fixture()
def dem_file(tmp_path: Path) -> tuple[Path, npt.NDArray[np.int16]]:
    """Writes a small synthetic int16 DEM as ENVI and returns its base path
    and the values it stores."""
    rng = np.random.default_rng(11)
    values = rng.integers(-500, 500, size=(DEM_ROWS, DEM_COLS)).astype("int16")
    base = tmp_path / "synth_dem"
    envi.write(str(base), values)
    return base, values


@pytest.fixture()
def dem(
    dem_file: tuple[Path, npt.NDArray[np.int16]],
) -> tuple[MOLA128, npt.NDArray[np.int16]]:
    """Reads the synthetic DEM file through MOLA128."""
    base, values = dem_file
    m = MOLA128(filename_base=str(base))
    assert m.read() is True
    return m, values


def test_dem_base_defaults() -> None:
    """Checks the base-class constructor defaults and the string form."""
    b = DEMBase()
    assert b.dem_name == "no name"
    assert b.filename_base == ""
    assert b.dummy_value == 0
    assert b.dem is None
    assert b.deg_sampling == 0
    assert b.offset_to_be_added == 0
    assert str(b) == "no name, base filename: "


def test_dem_base_interface_raises_not_implemented() -> None:
    """Checks that the three interface methods of the base class raise
    NotImplementedError when not overridden."""
    b = DEMBase()
    with pytest.raises(NotImplementedError):
        b.read()
    with pytest.raises(NotImplementedError):
        b.get_dem_radius_from_lat_lon(0.0, 0.0)
    with pytest.raises(NotImplementedError):
        b.reset()


def test_mola128_defaults() -> None:
    """Checks the MOLA128 subclass parameters: name, 1/128 deg sampling and
    the MOLA radius offset of 3396000 m."""
    m = MOLA128()
    assert isinstance(m, MOLA)
    assert m.dem_name == "MOLA128"
    assert m.deg_sampling == pytest.approx(1.0 / 128.0)
    assert m.offset_to_be_added == MOLA_OFFSET


def test_read_from_envi(dem: tuple[MOLA128, npt.NDArray[np.int16]]) -> None:
    """Checks the ENVI read: shape, int16 dtype and the geographic extents
    derived from the image size (max_lat 88, 1/128 deg sampling)."""
    m, _ = dem
    assert m.dem is not None
    assert m.dem.shape == (DEM_ROWS, DEM_COLS)
    assert m.dem.dtype == np.dtype("int16")
    assert m._MOLA_MAX_LAT == 88
    assert m._MOLA_MIN_LAT == pytest.approx(88 - DEM_ROWS / 128.0)
    assert m._MOLA_MIN_LONG == 0
    assert m._MOLA_MAX_LONG == pytest.approx(DEM_COLS / 128.0)


def test_read_twice_warns_and_skips(
    dem_file: tuple[Path, npt.NDArray[np.int16]], capsys: pytest.CaptureFixture[str]
) -> None:
    """Checks that a second read is refused with a warning and returns False."""
    base, _ = dem_file
    m = MOLA128(filename_base=str(base))
    assert m.read() is True
    capsys.readouterr()
    assert m.read() is False
    assert "WARNING" in capsys.readouterr().out


def test_read_without_filename(capsys: pytest.CaptureFixture[str]) -> None:
    """Checks that read without a base filename reports an error and returns False."""
    m = MOLA128()
    assert m.read() is False
    assert "ERROR" in capsys.readouterr().out


def test_read_missing_file_reports_error(
    dem_file: tuple[Path, npt.NDArray[np.int16]], capsys: pytest.CaptureFixture[str]
) -> None:
    """Checks the missing-file contract of read(): the ENVI reader reports
    the error itself and returns None, the reader's error message is printed
    and False is returned instead of failing on the None image."""
    base, _ = dem_file
    m = MOLA128(filename_base=str(base) + "_missing")
    assert m.read() is False
    assert m.dem is None
    assert "ERROR" in capsys.readouterr().out


def test_constructor_image_is_copied() -> None:
    """Checks that a DEM image provided to the constructor is copied, so
    later edits of the internal image do not touch the source array."""
    values = np.zeros((8, 8), dtype="int16")
    m = MOLA128(image=values)
    assert m.dem is not None
    m.dem[0, 0] = 12345
    assert values[0, 0] == 0


def test_to_xy_from_ll(dem: tuple[MOLA128, npt.NDArray[np.int16]]) -> None:
    """Checks lat/lon to pixel conversions: the upper-left corner maps to
    (0, 0) and a half-degree point maps to pixel 64 on both axes."""
    m, _ = dem
    x0, y0 = m.to_XY_from_LL(88.0, 0.0)
    assert np.array_equal(x0, np.array([0])) and np.array_equal(y0, np.array([0]))
    x, y = m.to_XY_from_LL(87.5, 0.5)
    assert np.array_equal(x, np.array([64])) and np.array_equal(y, np.array([64]))


def test_to_xy_from_ll_as_float(dem: tuple[MOLA128, npt.NDArray[np.int16]]) -> None:
    """Checks the float variant of the lat/lon to pixel conversion with array
    inputs (the scalar variant is covered by a separate behavior test)."""
    m, _ = dem
    x, y = m.to_XY_from_LL(np.array([87.5]), np.array([0.5]), return_as_float=True)
    assert x.dtype.kind == "f" and y.dtype.kind == "f"
    assert x[0] == pytest.approx(64.0) and y[0] == pytest.approx(64.0)


def test_to_xy_from_ll_as_float_scalar(
    dem: tuple[MOLA128, npt.NDArray[np.int16]],
) -> None:
    """Checks the float variant with scalar float inputs: scalar floats in,
    scalar floats out (the scalar continuation of the array path), with the
    internal mask set for the single point."""
    m, _ = dem
    x, y = m.to_XY_from_LL(87.5, 0.5, return_as_float=True)
    assert isinstance(x, float) and isinstance(y, float)
    assert x == pytest.approx(64.0) and y == pytest.approx(64.0)
    assert np.array_equal(m._mask, np.array([False]))


def test_to_xy_clamps_and_masks_out_of_range(
    dem: tuple[MOLA128, npt.NDArray[np.int16]],
) -> None:
    """Checks that out-of-coverage points are masked and clamped to the
    upper-left pixel, a longitude exactly on the right edge (x == n_cols)
    included: it is flagged out-of-range like any beyond-edge value instead
    of being wrapped to column 0."""
    m, _ = dem
    x, y = m.to_XY_from_LL(np.array([100.0, 80.0, 88.0]), np.array([0.5, 0.5, 1.0]))
    assert np.array_equal(x, np.array([64, 64, 0]))
    assert np.array_equal(y, np.array([0, 0, 0]))
    assert np.array_equal(m._mask, np.array([True, True, True]))


def test_to_xy_masks_right_edge_longitude(
    dem: tuple[MOLA128, npt.NDArray[np.int16]],
) -> None:
    """Checks the right-edge mask on the coordinate level: the last in-range
    column (x == n_cols - 1) stays unmasked, while a longitude mapping to
    x == n_cols and one beyond it are both masked and clamped to the
    upper-left pixel."""
    m, _ = dem
    lats = np.array([88.0, 88.0, 88.0])
    lons = np.array([127.0 / 128.0, 1.0, 1.0 + 1e-6])
    x, y = m.to_XY_from_LL(lats, lons)
    assert np.array_equal(x, np.array([127, 0, 0]))
    assert np.array_equal(y, np.array([0, 0, 0]))
    assert np.array_equal(m._mask, np.array([False, True, True]))


def test_radius_lookup_right_edge_receives_dummy() -> None:
    """Checks that a longitude exactly on the right edge receives the dummy
    value: x == n_cols is flagged out-of-range, so the lookup no longer
    silently reads column 0."""
    values = np.ones((8, 8), dtype="int16")
    m = MOLA128(image=values, dummy_value=77)
    radii = m.get_dem_radius_from_lat_lon(88.0, 8.0 / 128.0)
    assert radii is not None
    assert radii[0] == 77


def test_to_xy_int_variant_masks_non_finite(
    dem: tuple[MOLA128, npt.NDArray[np.int16]],
) -> None:
    """Checks the cast contract of the integer variant for non-finite
    coordinates (NaN, +inf, -inf): all of them are mapped to the
    out-of-coverage sentinel on the float array before the int cast and
    flagged out-of-range (ADR 0037). Casting a non-finite float to int is
    platform-dependent (INT64_MIN on x86-64, 0 on aarch64), so no non-finite
    value may reach the cast: the conversion must not warn."""
    m, _ = dem
    coords = np.array([np.nan, np.inf, -np.inf])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        x, y = m.to_XY_from_LL(coords, coords)
    assert np.array_equal(x, np.array([0, 0, 0]))
    assert np.array_equal(y, np.array([0, 0, 0]))
    assert np.array_equal(m._mask, np.array([True, True, True]))


def test_radius_lookup_non_finite_receives_dummy() -> None:
    """Checks that a non-finite latitude receives the dummy value: the
    coordinate is masked to the out-of-coverage sentinel before the int cast
    (no cast warning on any platform, ADR 0037) and the point is treated as
    out of coverage."""
    values = np.ones((8, 8), dtype="int16")
    m = MOLA128(image=values, dummy_value=77)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        radii = m.get_dem_radius_from_lat_lon(np.nan, 0.03)
    assert radii is not None
    assert radii[0] == 77


def test_to_ll_from_xy(dem: tuple[MOLA128, npt.NDArray[np.int16]]) -> None:
    """Checks the pixel to lat/lon conversion against the same geometry as
    the forward conversion."""
    m, _ = dem
    lat, lon = m.to_LL_from_XY(64, 64)
    assert lat == pytest.approx(87.5) and lon == pytest.approx(0.5)
    lat, lon = m.to_LL_from_XY(0, 0)
    assert lat == pytest.approx(88.0) and lon == pytest.approx(0.0)


def test_radius_lookup(dem: tuple[MOLA128, npt.NDArray[np.int16]]) -> None:
    """Checks per-point radius lookup: in-coverage points return the stored
    int16 value plus the MOLA offset as int32."""
    m, values = dem
    radii = m.get_dem_radius_from_lat_lon(88.0, 0.0)
    assert radii is not None
    assert radii.dtype == np.dtype("int32")
    assert radii[0] == int(values[0, 0]) + MOLA_OFFSET


def test_radius_lookup_dummy_outside_coverage() -> None:
    """Checks that out-of-coverage points receive the dummy value given at
    construction."""
    values = np.ones((8, 8), dtype="int16")
    m = MOLA128(image=values, dummy_value=77)
    radii = m.get_dem_radius_from_lat_lon(50.0, 0.0)  # below the DEM southern limit
    assert radii is not None
    assert radii[0] == 77


def test_xy_ranges_to_radius(dem: tuple[MOLA128, npt.NDArray[np.int16]]) -> None:
    """Checks the sub-image extraction with and without the MOLA offset."""
    m, values = dem
    sub = m.XY_ranges_to_radius(0, 3, 0, 3)
    assert sub.dtype == np.dtype("int32")
    assert np.array_equal(sub, values[0:4, 0:4].astype("int32") + MOLA_OFFSET)
    sub_raw = m.XY_ranges_to_radius(0, 3, 0, 3, add_offset=False)
    assert np.array_equal(sub_raw, values[0:4, 0:4].astype("int32"))


def test_radius_lookup_before_load(capsys: pytest.CaptureFixture[str]) -> None:
    """Checks that a radius lookup on a not-loaded DEM reports an error and
    returns None."""
    m = MOLA128()
    assert m.get_dem_radius_from_lat_lon(0.0, 0.0) is None
    assert "ERROR" in capsys.readouterr().out


def test_reset(
    dem: tuple[MOLA128, npt.NDArray[np.int16]], capsys: pytest.CaptureFixture[str]
) -> None:
    """Checks that reset frees the DEM image and later lookups report the
    not-loaded state."""
    m, _ = dem
    m.reset()
    assert m.dem is None
    assert m.get_dem_radius_from_lat_lon(88.0, 0.0) is None
    assert "ERROR" in capsys.readouterr().out


def test_scaled_subdem(dem: tuple[MOLA128, npt.NDArray[np.int16]]) -> None:
    """Checks the smoothed/upscaled sub-DEM extraction inside the DEM coverage
    (lat 86.75..86.5, lon 0.5..0.0): since phase-6 P7 it returns a lazy radius
    sampler instead of a materialized MOLA instance (the sampler keeps the
    sampling bookkeeping and the radius-lookup/dummy-value interface of the
    MOLA view it replaces), covering the requested limits at 4x sampling, and
    its radius lookups return finite int32 values inside the extracted area.
    Bit-level equality with the materialized zoom path is pinned by the
    full-lattice probe test. The snapped pixel ranges are rows 160..192 and
    columns 0..64, i.e. a 33x65 sub-image upscaled to 132x260."""
    m, _ = dem
    sub = m.get_scaled_subdem_from_latlon_limits(86.75, 86.5, 0.5, 0.0, scale_factor=4)
    assert sub is not None
    assert sub.deg_sampling == pytest.approx(1.0 / 512.0)
    assert sub._MOLA_MAX_LAT == pytest.approx(86.75)
    assert sub._MOLA_MIN_LONG == pytest.approx(0.0)
    assert sub.dem_shape == (132, 260)
    radii = sub.get_dem_radius_from_lat_lon(86.625, 0.25)
    assert radii is not None
    assert radii.dtype == np.dtype("int32")
    assert np.all(np.isfinite(radii))


def test_scaled_subdem_full_lattice_bitwise_equal_to_zoom(
    dem: tuple[MOLA128, npt.NDArray[np.int16]],
) -> None:
    """Probes the lazy scaled sub-DEM sampler against the materialized
    `scipy.ndimage.zoom` path it replaces: lookups over the complete upscaled
    lattice must return values bit-identical (after the int32 truncation and
    the MOLA-offset add) to indexing the zoomed image, as the previous
    zoom-based MOLA view did. The limits lie on the native grid, so the
    method's ceil/floor snapping leaves them unchanged and the reference goes
    through the same public conversion/extraction calls."""
    m, _ = dem
    scale_factor = 4
    sub = m.get_scaled_subdem_from_latlon_limits(
        86.75, 86.5, 0.5, 0.0, scale_factor=scale_factor
    )
    assert sub is not None

    mola_x_min, mola_y_min = m.to_XY_from_LL(86.75, 0.0)
    mola_x_max, mola_y_max = m.to_XY_from_LL(86.5, 0.5)
    subdem_image = m.XY_ranges_to_radius(
        mola_x_min[0], mola_x_max[0], mola_y_min[0], mola_y_max[0], add_offset=False
    )
    zoomed = snd.zoom(
        snd.gaussian_filter(subdem_image.astype("float"), 1), scale_factor, order=2
    )
    ref_radii = zoomed.astype("int32") + MOLA_OFFSET
    n_lat, n_long = zoomed.shape
    assert sub.dem_shape == (n_lat, n_long)

    ky, kx = np.meshgrid(np.arange(n_lat), np.arange(n_long), indexing="ij")
    lat_lattice = sub._MOLA_MAX_LAT - ky * sub.deg_sampling
    lon_lattice = sub._MOLA_MIN_LONG + kx * sub.deg_sampling
    radii = sub.get_dem_radius_from_lat_lon(lat_lattice, lon_lattice)
    assert radii is not None
    assert radii.dtype == np.dtype("int32")
    assert np.array_equal(radii, ref_radii)

    # boundary semantics carried over from to_XY_from_LL: a longitude mapping
    # to mola_x == n_long is masked out-of-range like any beyond-edge value
    # (it used to be wrapped to column 0 and read silently), and a latitude
    # beyond the southern limit is masked; both receive the dummy value
    q = n_lat // 2
    lat_probe = np.array([float(lat_lattice[q, 0]), 86.0])
    lon_probe = np.array(
        [float(sub._MOLA_MIN_LONG[0] + n_long * sub.deg_sampling), 0.25]
    )
    radii_edge = sub.get_dem_radius_from_lat_lon(lat_probe, lon_probe)
    assert radii_edge is not None
    assert radii_edge[0] == sub.dummy_value
    assert radii_edge[1] == sub.dummy_value


def test_scaled_subdem_non_finite_coordinates_receive_dummy(
    dem: tuple[MOLA128, npt.NDArray[np.int16]],
) -> None:
    """Checks the sampler cast contract for non-finite coordinates: a NaN
    latitude is mapped to the out-of-coverage sentinel before the int cast
    (no cast warning on any platform, ADR 0037) and receives the dummy value
    like any masked point."""
    m, _ = dem
    sub = m.get_scaled_subdem_from_latlon_limits(86.75, 86.5, 0.5, 0.0, scale_factor=4)
    assert sub is not None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        radii = sub.get_dem_radius_from_lat_lon(np.nan, 0.25)
    assert radii is not None
    assert radii[0] == sub.dummy_value


def test_str(dem: tuple[MOLA128, npt.NDArray[np.int16]]) -> None:
    """Checks the string form with the geographic extension (the max
    latitude and min longitude render without decimals, as stored)."""
    m, _ = dem
    text = str(m)
    assert text.startswith("MOLA128, base filename: ")
    assert "extension: lat 86.0, 88; long 0, 1.0" in text
