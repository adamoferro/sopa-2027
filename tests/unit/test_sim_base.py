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
times), the chirp convolution, the subprocess frame-range split, the
n_processes clamping of the phase-6 parallel reactivation, the
parallel-dispatch machinery (result assembly and the single-worker frame-range
partition invariance), and the sample-position cast contract of the shared
`_sample_positions_to_int` helper.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from sopa.geometry.coordinate_converter import MARS_IAU2000
from sopa.input.sharad.instrument import SHARAD
from sopa.processing.parameters import SimulatorParameters
from sopa.processing.simulators import sim_base as sim_base_module
from sopa.processing.simulators.sim_base import (
    _alt_unit_vector_carry,
    _assemble_range_results,
    _sample_positions_to_int,
    _SIMBase,
)

N_FRAMES = 12
DEFAULT_DT = 37.5e-9


class _ConstDEM:
    """Stand-in DEM returning one constant radius for every requested point."""

    def __init__(self, radius: float = 3396000.0) -> None:
        self.radius = radius

    def get_dem_radius_from_lat_lon(
        self, lat: npt.ArrayLike, lon: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Returns the constant DEM radius for all points."""
        return np.full(np.shape(lat), self.radius, dtype="float64")


class _RecordingCallback:
    """Trivial per-frame callback recording its arguments and returning a
    one-spike image plus empty bookkeeping."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        id_: int,
        n_processes: int,
        orbit_xyz: tuple[npt.NDArray[np.float64], ...],
        frame_range: npt.NDArray[np.int64],
        sim_n_samples: int,
        local_direction_g: npt.NDArray[np.float64],
        UTMNorthing_g: npt.NDArray[np.float64],
        UTMEasting_g: npt.NDArray[np.float64],
        UTMZone_g: list[str],
        rxwin_ref: float,
        shifts_alignment_t: npt.NDArray[np.float64],
        dt: float,
        dem_obj: object,
        geom_obj: object,
        alt_sim_matrix: npt.NDArray[np.float64],
        act_sim_matrix: npt.NDArray[np.float64],
        gaussian_filter_sigma: int,
        further_pars: list[float],
        roi_img: npt.NDArray | None = None,
        *,
        alt_unit_vector_carry: npt.NDArray[np.float64] | None = None,
        verbose: bool = True,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        tuple[dict[int, npt.NDArray[np.float64]], ...],
    ]:
        """Records the call and returns a spike image with zero ancillary data."""
        n_frames = len(frame_range)
        self.calls.append(
            {
                "sim_n_samples": sim_n_samples,
                "dt": dt,
                "n_frames": n_frames,
                "range_start": int(frame_range[0]),
                "range_stop": int(frame_range[-1]) + 1,
                "verbose": verbose,
                "carry_shape": None
                if alt_unit_vector_carry is None
                else alt_unit_vector_carry.shape,
            }
        )
        sim_image = np.zeros([sim_n_samples, n_frames])
        sim_image[0, :] = 1.0
        return (
            sim_image,
            np.zeros(sim_image.shape, dtype="uint8"),
            np.zeros(n_frames),
            np.zeros(n_frames),
            np.zeros(sim_image.shape),
            np.zeros(sim_image.shape),
            ({}, {}, {}, {}),
        )


class _TrivialSim(_SIMBase):
    """Minimal concrete simulator handing the per-frame work to the recorded
    trivial callback."""

    def __init__(
        self,
        geom_obj: object,
        dem_obj: object,
        param_obj: SimulatorParameters,
        callback: _RecordingCallback,
        simulate_chirp: bool = False,
        instr_obj: object = None,
        n_processes: int = 1,
    ) -> None:
        super().__init__(
            geom_obj,
            dem_obj,
            param_obj,
            n_processes=n_processes,
            simulate_chirp=simulate_chirp,
            instr_obj=instr_obj,
        )
        self.n_frames_offset_for_dir_estimation = 1
        self.alt_sim_matrix = np.zeros((2, 2))
        self.act_sim_matrix = np.zeros((2, 2))
        self.gaussian_filter_sigma = 1
        self._callback = callback

    def simulate(
        self, orbit_data: dict[str, npt.NDArray[np.float64] | float]
    ) -> tuple[npt.NDArray[np.float64], ...] | None:
        """Runs the base-class simulation over the selected frames."""
        return super().simulate(orbit_data, self._callback)


def _orbit_data(
    rxwin_value: float | None = 2e-3, varying_radius: bool = False
) -> dict[str, npt.NDArray[np.float64] | float]:
    """Builds a synthetic orbit-data dictionary on an equatorial track. A
    None receive-window value produces the unset (all-zero) case."""
    cc = MARS_IAU2000()
    lat = np.zeros(N_FRAMES)
    lon = np.linspace(0.05, 0.60, N_FRAMES)
    radius = np.full(N_FRAMES, 3646190.0)
    if varying_radius:
        radius = radius + 300.0 * np.sin(np.linspace(0.0, np.pi, N_FRAMES))
    x, y, z = cc.to_xyz_from_latlon_and_radius(lat, lon, radius)
    rxwin = (
        np.zeros(N_FRAMES) if rxwin_value is None else np.full(N_FRAMES, rxwin_value)
    )
    return {
        "Lat": lat,
        "Lon": lon,
        "X": x,
        "Y": y,
        "Z": z,
        "Radius": radius,
        "rxwin_time_s": rxwin,
        "dt": DEFAULT_DT,
    }


def test_constructor_n_processes_clamping(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Checks the n_processes interface since the parallel reactivation:
    values below one fall back to one, one is kept, values above one are
    preserved, and the constructor prints no notice in any case (the former
    parallelization-is-not-implemented notice is gone)."""
    pars = SimulatorParameters(frame_start=0, frame_end=1)
    for requested, expected in ((-3, 1), (0, 1), (1, 1), (4, 4)):
        base = _SIMBase(MARS_IAU2000(), _ConstDEM(), pars, n_processes=requested)
        assert base.n_processes == expected
        assert capsys.readouterr().out == ""


def test_chirp_requires_instrument() -> None:
    """Checks that requesting the chirp convolution without an instrument
    object raises ValueError."""
    with pytest.raises(ValueError):
        _SIMBase(
            MARS_IAU2000(),
            _ConstDEM(),
            SimulatorParameters(frame_start=0, frame_end=1),
            simulate_chirp=True,
        )


def test_frame_selection_by_frame_skip() -> None:
    """Checks the frame-skip selection: frames from the (clamped) start up to
    the end are simulated, one output column per selected frame, and the
    callback receives the orbit-data fast-time step."""
    callback = _RecordingCallback()
    sim = _TrivialSim(
        MARS_IAU2000(),
        _ConstDEM(),
        SimulatorParameters(frame_start=2, frame_end=8, frame_half_aperture=2),
        callback,
    )
    result = sim.simulate(_orbit_data())
    assert result is not None
    assert len(result) == 7
    assert result[0].shape[1] == 6
    assert callback.calls[0]["n_frames"] == 6
    assert callback.calls[0]["dt"] == DEFAULT_DT


def test_invalid_frame_range_returns_none() -> None:
    """Checks that an end frame at or after the input length is rejected."""
    callback = _RecordingCallback()
    sim = _TrivialSim(
        MARS_IAU2000(),
        _ConstDEM(),
        SimulatorParameters(frame_start=2, frame_end=N_FRAMES, frame_half_aperture=2),
        callback,
    )
    assert sim.simulate(_orbit_data()) is None


def test_output_sizing_with_rxwin_set() -> None:
    """Checks the output size with constant receive-window times and constant
    spacecraft radius: exactly the nominal 1334 samples, with no alignment shifts."""
    callback = _RecordingCallback()
    sim = _TrivialSim(
        MARS_IAU2000(),
        _ConstDEM(),
        SimulatorParameters(frame_start=2, frame_end=8, frame_half_aperture=2),
        callback,
    )
    result = sim.simulate(_orbit_data())
    assert result is not None
    assert result[0].shape == (667 * 2, 6)


def test_rxwin_unset_derives_reference_and_grows_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Checks the unset receive-window branch: the reference is derived from
    the DEM surface (with a notice) and varying spacecraft radii extend the
    output beyond the nominal sample count."""
    callback = _RecordingCallback()
    sim = _TrivialSim(
        MARS_IAU2000(),
        _ConstDEM(),
        SimulatorParameters(frame_start=2, frame_end=8, frame_half_aperture=2),
        callback,
    )
    result = sim.simulate(_orbit_data(rxwin_value=None, varying_radius=True))
    assert result is not None
    assert result[0].shape[0] > 667 * 2
    assert "NOTICE" in capsys.readouterr().out


def test_frame_selection_by_meters_skip() -> None:
    """Checks the meters-based selection: the ground-distance skip produces a
    strictly increasing subset of the input frames."""
    callback = _RecordingCallback()
    pars = SimulatorParameters(
        frame_start=2, frame_end=N_FRAMES - 1, frame_half_aperture=2, meters_skip=6000.0
    )
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
    sim = _TrivialSim(
        MARS_IAU2000(),
        _ConstDEM(),
        SimulatorParameters(frame_start=2, frame_end=8, frame_half_aperture=2),
        callback,
        simulate_chirp=True,
        instr_obj=SHARAD(),
    )
    result = sim.simulate(_orbit_data())
    assert result is not None
    sim_image = result[0]
    assert sim_image.shape[0] == 667 * 2
    assert np.all(np.isfinite(sim_image))
    assert np.all(sim_image >= 0)
    assert np.sum(sim_image) > 0


def test_define_frame_ranges() -> None:
    """Checks the frame-range split: one full range for the serial default,
    and contiguous ceiling-sized ranges covering every frame exactly once for
    a requested process count above one (the parallel dispatch partitions
    per requested process, never per effective worker)."""
    sim = _TrivialSim(
        MARS_IAU2000(),
        _ConstDEM(),
        SimulatorParameters(frame_start=0, frame_end=1),
        _RecordingCallback(),
    )
    starts, stops = sim._define_frame_ranges(10)
    np.testing.assert_array_equal(starts, [0])
    np.testing.assert_array_equal(stops, [10])

    sim_parallel = _SIMBase(
        MARS_IAU2000(),
        _ConstDEM(),
        SimulatorParameters(frame_start=0, frame_end=1),
        n_processes=4,
    )
    starts, stops = sim_parallel._define_frame_ranges(10)
    np.testing.assert_array_equal(starts, [0, 3, 6, 9])
    np.testing.assert_array_equal(stops, [3, 6, 9, 10])


def test_alt_unit_vector_carry_shape_and_values() -> None:
    """Checks the parent-side along-track carry computation: the unit vector
    of the spacecraft-position difference between the previous simulated
    frame and its successor, in the (3, 1, 1) layout the worker loop uses."""
    cc = MARS_IAU2000()
    rng = np.random.default_rng(7)
    x, y, z = (rng.random(6) + 3000.0 for _ in range(3))
    carry = _alt_unit_vector_carry((x, y, z), 2)
    assert carry.shape == (3, 1, 1)
    expected = cc.unit_vector(
        np.array(
            [
                [[x[3] - x[2]]],
                [[y[3] - y[2]]],
                [[z[3] - z[2]]],
            ]
        )
    )
    assert carry.tobytes() == expected.tobytes()


def test_assemble_range_results_frame_order() -> None:
    """Checks the parallel-result assembly: image blocks and first-return
    values concatenate in frame order and the per-range ROI dictionaries
    merge key by key in frame order, reproducing exactly the values and the
    key order of a single-call (serial) result."""
    first = (
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.zeros((2, 2), dtype="uint8"),
        np.array([10.0, 11.0]),
        np.array([20.0, 21.0]),
        np.array([[np.nan, 1.0], [np.nan, 2.0]]),
        np.array([[np.nan, 5.0], [np.nan, 6.0]]),
        ({7: np.array([1.0, 2.0]), 3: np.array([9.0])}, {}),
    )
    second = (
        np.array([[5.0, 6.0], [7.0, 8.0]]),
        np.ones((2, 2), dtype="uint8"),
        np.array([12.0, 13.0]),
        np.array([22.0, 23.0]),
        np.array([[3.0, np.nan], [4.0, np.nan]]),
        np.array([[7.0, np.nan], [8.0, np.nan]]),
        ({7: np.array([3.0]), 3: np.array([8.0, 10.0])}, {}),
    )

    (
        sim_image,
        uncert_image,
        first_return_lats,
        first_return_lons,
        ground_distance_min_image,
        ground_distance_max_image,
        roi_pos,
    ) = _assemble_range_results([first, second])

    np.testing.assert_array_equal(
        sim_image, [[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]
    )
    assert (
        uncert_image.tobytes()
        == np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype="uint8").tobytes()
    )
    np.testing.assert_array_equal(first_return_lats, [10.0, 11.0, 12.0, 13.0])
    np.testing.assert_array_equal(first_return_lons, [20.0, 21.0, 22.0, 23.0])
    np.testing.assert_array_equal(
        ground_distance_min_image,
        [[np.nan, 1.0, 3.0, np.nan], [np.nan, 2.0, 4.0, np.nan]],
    )
    np.testing.assert_array_equal(
        ground_distance_max_image,
        [[np.nan, 5.0, 7.0, np.nan], [np.nan, 6.0, 8.0, np.nan]],
    )
    assert list(roi_pos[0].keys()) == [7, 3]  # first-appearance order
    np.testing.assert_array_equal(roi_pos[0][7], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(roi_pos[0][3], [9.0, 8.0, 10.0])
    assert roi_pos[1] == {}


def test_single_worker_partition_invariance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks the phase-6 parallel dispatch with the worker pool capped down
    to one effective worker: the requested partition is still derived from
    the requested process count (here one range per frame, which also routes
    the last frame's along-track carry through the parent-side computation),
    each range runs silently, and the assembled result equals the serial
    single-call result."""
    monkeypatch.setattr(
        sim_base_module,
        "_max_memory_limited_pool_workers",
        lambda cpu_count, available_memory_bytes: 1,
    )
    pars = SimulatorParameters(frame_start=2, frame_end=8, frame_half_aperture=2)

    serial_callback = _RecordingCallback()
    serial_sim = _TrivialSim(MARS_IAU2000(), _ConstDEM(), pars, serial_callback)
    serial_result = serial_sim.simulate(_orbit_data())
    assert serial_result is not None
    assert len(serial_callback.calls) == 1

    partitioned_callback = _RecordingCallback()
    partitioned_sim = _TrivialSim(
        MARS_IAU2000(), _ConstDEM(), pars, partitioned_callback, n_processes=6
    )
    partitioned_result = partitioned_sim.simulate(_orbit_data())
    assert partitioned_result is not None

    assert len(partitioned_callback.calls) == 6  # ceil(6/6)=1 frame per range
    assert [call["range_start"] for call in partitioned_callback.calls] == list(
        range(6)
    )
    assert all(not call["verbose"] for call in partitioned_callback.calls)
    assert partitioned_callback.calls[0]["carry_shape"] is None
    assert partitioned_callback.calls[1]["carry_shape"] == (3, 1, 1)
    assert partitioned_callback.calls[-1]["carry_shape"] == (3, 1, 1)

    for serial_part, partitioned_part in zip(serial_result, partitioned_result):
        if isinstance(serial_part, tuple):
            assert len(serial_part) == len(partitioned_part)
            for serial_dict, partitioned_dict in zip(serial_part, partitioned_part):
                assert list(serial_dict.keys()) == list(partitioned_dict.keys())
        else:
            assert serial_part.tobytes() == partitioned_part.tobytes()


def test_sample_positions_to_int_finite_in_range() -> None:
    """Checks the cast contract for finite in-range positions: the already-rounded values are cast to int64 exactly, including the range edges 0 and sim_n_samples - 1."""
    rounded_positions = np.array([[0.0, 1.0, 666.0, 1334.0]], dtype="float64")
    result = _sample_positions_to_int(rounded_positions, 1335)
    np.testing.assert_array_equal(result, np.array([[0, 1, 666, 1334]], dtype="int64"))
    assert result.dtype == np.dtype("int64")


def test_sample_positions_to_int_non_finite_become_sentinel() -> None:
    """Checks the cast contract for non-finite positions (NaN, +inf, -inf): all of them become the sim_n_samples sentinel before the cast. This pins the cross-platform reason the helper exists: casting NaN to int yields INT64_MIN on x86-64 (masked to the sentinel by the former post-cast mask) but 0 on aarch64, which survives as a valid sample bin and corrupts the first-return selection over DEM holes."""
    rounded_positions = np.array([[np.nan, np.inf, -np.inf]], dtype="float64")
    result = _sample_positions_to_int(rounded_positions, 1335)
    np.testing.assert_array_equal(result, np.full((1, 3), 1335, dtype="int64"))


def test_sample_positions_to_int_out_of_range_become_sentinel() -> None:
    """Checks the cast contract for finite out-of-range positions: negative values and values at or beyond sim_n_samples become the sentinel, the same replacement the call sites' post-cast mask applies."""
    rounded_positions = np.array([[-1.0, -100.0, 1335.0, 1.0e9]], dtype="float64")
    result = _sample_positions_to_int(rounded_positions, 1335)
    np.testing.assert_array_equal(result, np.full((1, 4), 1335, dtype="int64"))
