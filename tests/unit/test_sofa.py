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

"""Unit tests of the SOFA focuser (sopa.processing.focusers.sofa): the raw-data
type validation, the derived focusing parameters (coherence time, Doppler
window, Gaussian-filter multilook decomposition, weighting windows), the
frame-range validation of focus(), and a small end-to-end focus run on a
tiny synthetic raw dataset.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from sopa.geometry.coordinate_converter import MARS_IAU2000
from sopa.input.data_base import RSData
from sopa.input.sharad.instrument import SHARAD
from sopa.processing.focusers.sofa import SOFA
from sopa.processing.parameters import FocuserParameters

N_FRAMES = 20
N_INPUT_SAMPLES = 900
PRI = 1428e-6
PRESUMMING = 16
N_HALF_APERTURE = 3


class _StubRawData(RSData):
    """Minimal in-memory raw-data holder satisfying the SOFA constructor and
    the focus-time data access."""

    def __init__(self, data: npt.NDArray[np.float64], orbit_data: dict[str, npt.NDArray[np.float64] | float]) -> None:
        super().__init__(dataset_name="stub", pri=PRI, presumming=PRESUMMING)
        self._data_type = "raw"
        self._data = data
        self.set_orbit_data_from_ext(orbit_data)

    def _load_from_file(self, mode: str | None = None, force_reload: bool = False) -> bool:
        """Never used by the focuser in these tests."""
        return True

    def generate_orbit_data(self, skip: int = 1) -> None:
        """Never used by the focuser in these tests."""
        raise NotImplementedError


def _orbit_data() -> dict[str, npt.NDArray[np.float64] | float]:
    """Builds a synthetic orbit-data dictionary on an equatorial track with
    constant radius and receive-window time."""
    cc = MARS_IAU2000()
    lat = np.zeros(N_FRAMES)
    lon = np.linspace(0.0, 0.38, N_FRAMES)
    radius = np.full(N_FRAMES, 3646190.0)
    x, y, z = cc.to_xyz_from_latlon_and_radius(lat, lon, radius)
    return {"Lat": lat, "Lon": lon, "X": x, "Y": y, "Z": z, "Radius": radius,
            "rxwin_time_s": np.full(N_FRAMES, 5e-3), "dt": 37.5e-9}


@pytest.fixture()
def stub_data() -> _StubRawData:
    """A small synthetic raw dataset (random real echoes) with its orbit data."""
    rng = np.random.default_rng(31)
    data = rng.standard_normal((N_INPUT_SAMPLES, N_FRAMES))
    return _StubRawData(data, _orbit_data())


def test_constructor_rejects_non_raw_data() -> None:
    """Checks that SOFA refuses non-raw input data."""
    holder = _StubRawData(np.zeros((8, N_FRAMES)), _orbit_data())
    holder._data_type = "focused"
    with pytest.raises(ValueError):
        SOFA(SHARAD(), MARS_IAU2000(), np.full(N_FRAMES, 3396000, dtype="int32"), holder, FocuserParameters())


def test_constructor_derived_parameters(stub_data: _StubRawData) -> None:
    """Checks the derived focusing parameters: PRI-based coherence time and
    spacing, aperture length, and the Doppler half window clamped to zero for
    a short aperture."""
    pars = FocuserParameters(frame_start=0, frame_end=1, frame_half_aperture=N_HALF_APERTURE)
    f = SOFA(SHARAD(), MARS_IAU2000(), np.full(N_FRAMES, 3396000, dtype="int32"), stub_data, pars)
    assert f._pri_data == pytest.approx(PRI * PRESUMMING)
    assert f._n_aperture_frames == 2 * N_HALF_APERTURE + 1
    assert f._coherence_time == pytest.approx((2 * N_HALF_APERTURE) * PRI * PRESUMMING)
    assert f._du_global == pytest.approx(PRI * PRESUMMING * 3397.0)
    assert f._doppler_half_window_width == 0
    assert f._gf_power_coefficients == []
    assert f._gf_std_devs == []
    assert f._SIZE_RANGE_FFT == 4096


def test_constructor_gaussian_filter_decomposition(stub_data: _StubRawData) -> None:
    """Checks the Gaussian multilook decomposition setup for three filters:
    four power coefficients summing to one, starting with alpha, and the
    linearly increasing standard deviations up to the rounded maximum."""
    pars = FocuserParameters(frame_start=0, frame_end=1, frame_half_aperture=100, gauss_rec_n_filters=3, gauss_rec_alpha=0.66)
    f = SOFA(SHARAD(), MARS_IAU2000(), np.full(N_FRAMES, 3396000, dtype="int32"), stub_data, pars)
    coefficients = f._gf_power_coefficients
    assert len(coefficients) == 4
    assert coefficients[0] == pytest.approx(0.66)
    assert sum(coefficients) == pytest.approx(1.0, abs=1e-12)
    std_devs = f._gf_std_devs
    assert len(std_devs) == 3
    assert std_devs[0] < std_devs[1] < std_devs[2]
    assert std_devs[-1] == pytest.approx(1.8)
    assert f._doppler_half_window_width == 1


def test_constructor_weighting_windows(stub_data: _StubRawData) -> None:
    """Checks the range and azimuth weighting windows built from the named
    window functions, including the kaiser variant with its beta."""
    pars = FocuserParameters(frame_start=0, frame_end=1, frame_half_aperture=N_HALF_APERTURE,
                             range_weighting_function="kaiser", range_weighting_kaiser_beta=2.5,
                             azimuth_weighting_function="kaiser", azimuth_weighting_kaiser_beta=1.5)
    f = SOFA(SHARAD(), MARS_IAU2000(), np.full(N_FRAMES, 3396000, dtype="int32"), stub_data, pars)
    np.testing.assert_allclose(f._range_weighting_window_original, np.kaiser(4096 / 2.0, 2.5))
    np.testing.assert_allclose(f._azimuth_weighting_window_original, np.kaiser(2 * N_HALF_APERTURE + 1, 1.5))


def test_constructor_weighting_windows_none(stub_data: _StubRawData) -> None:
    """Checks the weighting-window fallbacks for the "none" functions: an
    all-ones float64 range window over half the range FFT length and the
    scalar azimuth window that focus() broadcasts over the output samples."""
    pars = FocuserParameters(frame_start=0, frame_end=1, frame_half_aperture=N_HALF_APERTURE,
                             range_weighting_function="none", azimuth_weighting_function="none")
    f = SOFA(SHARAD(), MARS_IAU2000(), np.full(N_FRAMES, 3396000, dtype="int32"), stub_data, pars)
    np.testing.assert_array_equal(f._range_weighting_window_original, np.ones(2048))
    assert f._range_weighting_window_original.dtype == np.float64
    assert f._azimuth_weighting_window_original == 1.


def test_focus_rejects_end_after_data(stub_data: _StubRawData, capsys: pytest.CaptureFixture[str]) -> None:
    """Checks that an end frame at or after the input length stops the focus
    before any processing."""
    pars = FocuserParameters(frame_start=3, frame_end=N_FRAMES, frame_half_aperture=N_HALF_APERTURE)
    f = SOFA(SHARAD(), MARS_IAU2000(), np.full(N_FRAMES, 3396000, dtype="int32"), stub_data, pars)
    assert f.focus() is None
    assert "ERROR" in capsys.readouterr().out


def test_focus_rejects_empty_range(stub_data: _StubRawData, capsys: pytest.CaptureFixture[str]) -> None:
    """Checks that a frame range ending before its (clamped) start stops the
    focus before any processing."""
    pars = FocuserParameters(frame_start=5, frame_end=4, frame_half_aperture=N_HALF_APERTURE)
    f = SOFA(SHARAD(), MARS_IAU2000(), np.full(N_FRAMES, 3396000, dtype="int32"), stub_data, pars)
    assert f.focus() is None
    assert "ERROR" in capsys.readouterr().out


def test_focus_tiny_run(stub_data: _StubRawData) -> None:
    """Checks a complete focus run on the tiny dataset: one output column per
    focused frame (arange(3, 10) = 7), all values finite and non-negative,
    and positive total power."""
    pars = FocuserParameters(frame_start=3, frame_end=10, frame_half_aperture=N_HALF_APERTURE, squint_angle=0)
    f = SOFA(SHARAD(), MARS_IAU2000(), np.full(N_FRAMES, 3396000, dtype="int32"), stub_data, pars, n_processes=1)
    output = f.focus()
    assert output is not None
    assert output.shape == (667 * 2, 7)
    assert np.all(np.isfinite(output))
    assert np.all(output >= 0)
    assert np.sum(output) > 0


def test_focus_tiny_run_unweighted(stub_data: _StubRawData) -> None:
    """Checks the same focus run as test_focus_tiny_run with both weighting
    functions set to "none": the run completes, one output column per focused
    frame (arange(3, 10) = 7), all values finite and non-negative, and
    positive total power."""
    pars = FocuserParameters(frame_start=3, frame_end=10, frame_half_aperture=N_HALF_APERTURE, squint_angle=0,
                             range_weighting_function="none", azimuth_weighting_function="none")
    f = SOFA(SHARAD(), MARS_IAU2000(), np.full(N_FRAMES, 3396000, dtype="int32"), stub_data, pars, n_processes=1)
    output = f.focus()
    assert output is not None
    assert output.shape == (667 * 2, 7)
    assert np.all(np.isfinite(output))
    assert np.all(output >= 0)
    assert np.sum(output) > 0
