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

"""Unit tests of the simulator base class (sopa.processing.simulators.sim_base),
exercised through a minimal concrete subclass with a trivial per-frame
callback: constructor validation and normalization, frame selection by frame
skip and by meters skip, the output-image sizing (set and unset receive-window
times), the chirp convolution, and the subprocess frame-range split.
"""

import numpy as np
import numpy.typing as npt
import pytest

from sopa.geometry.coordinate_converter import MARS_IAU2000
from sopa.input.sharad.instrument import SHARAD
from sopa.processing.parameters import SimulatorParameters
from sopa.processing.simulators.sim_base import _SIMBase

N_FRAMES = 12
DEFAULT_DT = 37.5e-9


class _ConstDEM:
    """Stand-in DEM returning one constant radius for every requested point."""

    def __init__(self, radius: float = 3396000.0) -> None:
        self.radius = radius

    def get_dem_radius_from_lat_lon(self, lat: npt.ArrayLike, lon: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Returns the constant DEM radius for all points."""
        return np.full(np.shape(lat), self.radius, dtype="float64")


class _RecordingCallback:
    """Trivial per-frame callback recording its arguments and returning a
    one-spike image plus empty bookkeeping."""

    def __init__(self) -> None:
        self.calls: list[dict[str, float | int]] = []

    def __call__(self, id_: int, n_processes: int, orbit_xyz: tuple[npt.NDArray[np.float64], ...], frame_range: npt.NDArray[np.int64], sim_n_samples: int, local_direction_g: npt.NDArray[np.float64], UTMNorthing_g: npt.NDArray[np.float64], UTMEasting_g: npt.NDArray[np.float64], UTMZone_g: list[str], rxwin_ref: float, shifts_alignment_t: npt.NDArray[np.float64], dt: float, dem_obj: object, geom_obj: object, alt_sim_matrix: npt.NDArray[np.float64], act_sim_matrix: npt.NDArray[np.float64], gaussian_filter_sigma: int, further_pars: list[float], roi_img: npt.NDArray | None = None) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], tuple[dict[int, npt.NDArray[np.float64]], ...]]:
        """Records the call and returns a spike image with zero ancillary data."""
        n_frames = len(frame_range)
        self.calls.append({"sim_n_samples": sim_n_samples, "dt": dt, "n_frames": n_frames})
        sim_image = np.zeros([sim_n_samples, n_frames])
        sim_image[0, :] = 1.0
        return (sim_image, np.zeros(sim_image.shape, dtype="uint8"), np.zeros(n_frames), np.zeros(n_frames),
                np.zeros(sim_image.shape), np.zeros(sim_image.shape), ({}, {}, {}, {}))


class _TrivialSim(_SIMBase):
    """Minimal concrete simulator handing the per-frame work to the recorded
    trivial callback."""

    def __init__(self, geom_obj: object, dem_obj: object, param_obj: SimulatorParameters, callback: _RecordingCallback, simulate_chirp: bool = False, instr_obj: object = None) -> None:
        super().__init__(geom_obj, dem_obj, param_obj, n_processes=1, simulate_chirp=simulate_chirp, instr_obj=instr_obj)
        self.n_frames_offset_for_dir_estimation = 1
        self.alt_sim_matrix = np.zeros((2, 2))
        self.act_sim_matrix = np.zeros((2, 2))
        self.gaussian_filter_sigma = 1
        self._callback = callback

    def simulate(self, orbit_data: dict[str, npt.NDArray[np.float64] | float]) -> tuple[npt.NDArray[np.float64], ...] | None:
        """Runs the base-class simulation over the selected frames."""
        return super().simulate(orbit_data, self._callback)


def _orbit_data(rxwin_value: float | None = 2e-3, varying_radius: bool = False) -> dict[str, npt.NDArray[np.float64] | float]:
    """Builds a synthetic orbit-data dictionary on an equatorial track. A
    None receive-window value produces the unset (all-zero) case."""
    cc = MARS_IAU2000()
    lat = np.zeros(N_FRAMES)
    lon = np.linspace(0.05, 0.60, N_FRAMES)
    radius = np.full(N_FRAMES, 3646190.0)
    if varying_radius:
        radius = radius + 300.0 * np.sin(np.linspace(0.0, np.pi, N_FRAMES))
    x, y, z = cc.to_xyz_from_latlon_and_radius(lat, lon, radius)
    rxwin = np.zeros(N_FRAMES) if rxwin_value is None else np.full(N_FRAMES, rxwin_value)
    return {"Lat": lat, "Lon": lon, "X": x, "Y": y, "Z": z, "Radius": radius, "rxwin_time_s": rxwin, "dt": DEFAULT_DT}


def test_constructor_normalizes_n_processes(capsys: pytest.CaptureFixture[str]) -> None:
    """Checks that n_processes above one falls back to one with a notice and
    values below one are raised to one silently."""
    pars = SimulatorParameters(frame_start=0, frame_end=1)
    base_high = _SIMBase(MARS_IAU2000(), _ConstDEM(), pars, n_processes=4)
    assert "NOTICE" in capsys.readouterr().out
    assert base_high.n_processes == 1
    base_low = _SIMBase(MARS_IAU2000(), _ConstDEM(), pars, n_processes=0)
    assert base_low.n_processes == 1


def test_chirp_requires_instrument() -> None:
    """Checks that requesting the chirp convolution without an instrument
    object raises ValueError."""
    with pytest.raises(ValueError):
        _SIMBase(MARS_IAU2000(), _ConstDEM(), SimulatorParameters(frame_start=0, frame_end=1), simulate_chirp=True)


def test_frame_selection_by_frame_skip() -> None:
    """Checks the frame-skip selection: frames from the (clamped) start up to
    the end are simulated, one output column per selected frame, and the
    callback receives the orbit-data fast-time step."""
    callback = _RecordingCallback()
    sim = _TrivialSim(MARS_IAU2000(), _ConstDEM(), SimulatorParameters(frame_start=2, frame_end=8, frame_half_aperture=2), callback)
    result = sim.simulate(_orbit_data())
    assert result is not None
    assert len(result) == 7
    assert result[0].shape[1] == 6
    assert callback.calls[0]["n_frames"] == 6
    assert callback.calls[0]["dt"] == DEFAULT_DT


def test_invalid_frame_range_returns_none() -> None:
    """Checks that an end frame at or after the input length is rejected."""
    callback = _RecordingCallback()
    sim = _TrivialSim(MARS_IAU2000(), _ConstDEM(), SimulatorParameters(frame_start=2, frame_end=N_FRAMES, frame_half_aperture=2), callback)
    assert sim.simulate(_orbit_data()) is None


def test_output_sizing_with_rxwin_set() -> None:
    """Checks the output size with constant receive-window times and constant
    spacecraft radius: exactly the nominal 1334 samples, with no alignment shifts."""
    callback = _RecordingCallback()
    sim = _TrivialSim(MARS_IAU2000(), _ConstDEM(), SimulatorParameters(frame_start=2, frame_end=8, frame_half_aperture=2), callback)
    result = sim.simulate(_orbit_data())
    assert result is not None
    assert result[0].shape == (667 * 2, 6)


def test_rxwin_unset_derives_reference_and_grows_output(capsys: pytest.CaptureFixture[str]) -> None:
    """Checks the unset receive-window branch: the reference is derived from
    the DEM surface (with a notice) and varying spacecraft radii extend the
    output beyond the nominal sample count."""
    callback = _RecordingCallback()
    sim = _TrivialSim(MARS_IAU2000(), _ConstDEM(), SimulatorParameters(frame_start=2, frame_end=8, frame_half_aperture=2), callback)
    result = sim.simulate(_orbit_data(rxwin_value=None, varying_radius=True))
    assert result is not None
    assert result[0].shape[0] > 667 * 2
    assert "NOTICE" in capsys.readouterr().out


def test_frame_selection_by_meters_skip() -> None:
    """Checks the meters-based selection: the ground-distance skip produces a
    strictly increasing subset of the input frames."""
    callback = _RecordingCallback()
    pars = SimulatorParameters(frame_start=2, frame_end=N_FRAMES - 1, frame_half_aperture=2, meters_skip=6000.0)
    sim = _TrivialSim(MARS_IAU2000(), _ConstDEM(), pars, callback)
    result = sim.simulate(_orbit_data())
    assert result is not None
    n_simulated = result[0].shape[1]
    assert 2 <= n_simulated < N_FRAMES - 2
    assert callback.calls[0]["n_frames"] == n_simulated


def test_chirp_convolution() -> None:
    """Checks the chirp convolution of the simulated image: finite non-negative
    output with preserved shape and positive energy."""
    callback = _RecordingCallback()
    sim = _TrivialSim(MARS_IAU2000(), _ConstDEM(), SimulatorParameters(frame_start=2, frame_end=8, frame_half_aperture=2),
                      callback, simulate_chirp=True, instr_obj=SHARAD())
    result = sim.simulate(_orbit_data())
    assert result is not None
    sim_image = result[0]
    assert sim_image.shape[0] == 667 * 2
    assert np.all(np.isfinite(sim_image))
    assert np.all(sim_image >= 0)
    assert np.sum(sim_image) > 0


def test_define_frame_ranges() -> None:
    """Checks the subprocess frame-range split for serial processing."""
    sim = _TrivialSim(MARS_IAU2000(), _ConstDEM(), SimulatorParameters(frame_start=0, frame_end=1), _RecordingCallback())
    starts, stops = sim._define_frame_ranges(10)
    np.testing.assert_array_equal(starts, [0])
    np.testing.assert_array_equal(stops, [10])
