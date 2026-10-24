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

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

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


def test_read_missing_file_raises(dem_file: tuple[Path, npt.NDArray[np.int16]]) -> None:
    """Checks the current behavior on a missing DEM file: envi.read returns
    None and the subsequent _set_info raises AttributeError (documented as-is;
    flagged for the technical-debt page)."""
    base, _ = dem_file
    m = MOLA128(filename_base=str(base) + "_missing")
    with pytest.raises(AttributeError):
        m.read()


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


def test_to_xy_from_ll_as_float_scalar_crashes(
    dem: tuple[MOLA128, npt.NDArray[np.int16]],
) -> None:
    """Documents the current behavior of the float variant with scalar float
    inputs: the conversion produces plain Python floats, so the internal
    shape check fails with AttributeError (inherited behavior; flagged for
    the technical-debt page)."""
    m, _ = dem
    with pytest.raises(AttributeError):
        m.to_XY_from_LL(87.5, 0.5, return_as_float=True)


def test_to_xy_clamps_and_masks_out_of_range(
    dem: tuple[MOLA128, npt.NDArray[np.int16]],
) -> None:
    """Checks that out-of-coverage points are masked and clamped to the
    upper-left pixel. The longitude at the right edge (x == n_cols) is wrapped
    to column 0 before the mask is computed, so it is NOT masked and silently
    reads column 0 (documented as-is; flagged for the technical-debt page)."""
    m, _ = dem
    x, y = m.to_XY_from_LL(np.array([100.0, 80.0, 88.0]), np.array([0.5, 0.5, 1.0]))
    assert np.array_equal(x, np.array([64, 64, 0]))
    assert np.array_equal(y, np.array([0, 0, 0]))
    assert np.array_equal(m._mask, np.array([True, True, False]))


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
    (lat 86.75..86.5, lon 0.5..0.0): it returns a new MOLA instance covering
    the requested limits at 4x sampling, and its radius lookups return finite
    values inside the extracted area. The snapped pixel ranges are rows 160..192
    and columns 0..64, i.e. a 33x65 sub-image upscaled to 132x260."""
    m, _ = dem
    sub = m.get_scaled_subdem_from_latlon_limits(86.75, 86.5, 0.5, 0.0, scale_factor=4)
    assert isinstance(sub, MOLA)
    assert sub.deg_sampling == pytest.approx(1.0 / 512.0)
    assert sub._MOLA_MAX_LAT == pytest.approx(86.75)
    assert sub._MOLA_MIN_LONG == pytest.approx(0.0)
    assert sub.dem is not None and sub.dem.shape == (132, 260)
    radii = sub.get_dem_radius_from_lat_lon(86.625, 0.25)
    assert radii is not None and np.all(np.isfinite(radii))


def test_str(dem: tuple[MOLA128, npt.NDArray[np.int16]]) -> None:
    """Checks the string form with the geographic extension (the max
    latitude and min longitude render without decimals, as stored)."""
    m, _ = dem
    text = str(m)
    assert text.startswith("MOLA128, base filename: ")
    assert "extension: lat 86.0, 88; long 0, 1.0" in text
