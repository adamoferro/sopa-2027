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

"""Unit tests of the square-facet clutter simulator
(sopa.processing.simulators.sfrssim): constructor defaults and grid matrices,
and a small end-to-end run on a synthetic DEM and track checking the nadir
peak placement, facet-weighted energy, first-return bookkeeping and the
uncertainty mask.
"""

import numpy as np
import numpy.typing as npt
import pytest

from sopa.dem.mola.mola import MOLA128
from sopa.geometry.coordinate_converter import MARS_IAU2000
from sopa.processing.parameters import SimulatorParameters
from sopa.processing.simulators import sim_base as sim_base_module
from sopa.processing.simulators.sfrssim import sfRSsim

N_FRAMES = 12
N_HALF_APERTURE = 2
FRAME_START = N_HALF_APERTURE
FRAME_END = 10
N_SIMULATED = FRAME_END - FRAME_START  # arange(start, end) with skip 1


def _dem() -> MOLA128:
    """Builds a small synthetic DEM covering lat 2..0.4375, lon 0..1 deg."""
    rng = np.random.default_rng(21)
    values = (3396000 * 0 + rng.integers(-100, 100, size=(200, 128))).astype("int16")
    return MOLA128(image=values, max_lat=2.0)


def _orbit_data() -> dict[str, npt.NDArray[np.float64] | float]:
    """Builds a synthetic orbit-data dictionary on a slightly varying track
    above the DEM, with unset receive-window times and the fast-time step."""
    cc = MARS_IAU2000()
    lat = np.linspace(1.1, 1.3, N_FRAMES)
    lon = np.linspace(0.3, 0.8, N_FRAMES)
    radius = np.full(N_FRAMES, 3646190.0) + 300.0 * np.sin(
        np.linspace(0.0, np.pi, N_FRAMES)
    )
    x, y, z = cc.to_xyz_from_latlon_and_radius(lat, lon, radius)
    return {
        "Lat": lat,
        "Lon": lon,
        "X": x,
        "Y": y,
        "Z": z,
        "Radius": radius,
        "rxwin_time_s": np.zeros(N_FRAMES),
        "dt": 37.5e-9,
    }


def _make_sim(n_processes: int = 1, roi_img: npt.NDArray | None = None) -> sfRSsim:
    """Builds an sfRSsim instance with the across/along-track grids shrunk to
    keep the unit run fast (the matrices are per-instance attributes)."""
    sim = sfRSsim(
        geom_obj=MARS_IAU2000(),
        dem_obj=_dem(),
        param_obj=SimulatorParameters(
            frame_start=FRAME_START,
            frame_end=FRAME_END,
            frame_half_aperture=N_HALF_APERTURE,
        ),
        n_processes=n_processes,
        roi_img=roi_img,
    )
    alt = np.arange(-250.0, 250.0 + 12.5, 12.5)
    act = np.arange(-2500.0, 2500.0 + 25.0, 25.0)
    sim.alt_sim_matrix = np.array([alt]).T * np.ones([1, len(act)])
    sim.act_sim_matrix = np.array([act]) * np.ones([len(alt), 1])
    return sim


def test_constructor_defaults() -> None:
    """Checks the constructor defaults: facet weighting exponent, spacings,
    smoothing sigma, grid extents and the half-sampled grid matrix shapes."""
    sim = sfRSsim(geom_obj=None, dem_obj=None, param_obj=SimulatorParameters())
    assert sim.cos_exp == 1000
    assert sim.alt_sim_spacing == 50
    assert sim.act_sim_spacing == 50
    assert sim.gaussian_filter_sigma == 8
    assert sim.alt_sim_forward_max_distance == 500
    assert sim.alt_sim_backward_max_distance == 500
    assert sim.act_sim_max_distance == 50000
    assert sim.alt_sim_matrix.shape == (41, 4001)
    assert sim.act_sim_matrix.shape == (41, 4001)
    assert sim.alt_sim_matrix[0, 0] == -500.0
    assert sim.act_sim_matrix[0, -1] == 50000.0


def test_simulation_produces_weighted_energy() -> None:
    """Checks the end-to-end facet simulation on the synthetic scene: positive
    energy with the nadir return near the reference sample, no uncertainty
    flagged (full DEM coverage), sane first returns and ground distances."""
    sim = _make_sim()
    result = sim.simulate(_orbit_data())
    assert result is not None
    (
        sim_image,
        uncert_image,
        first_return_lats,
        first_return_lons,
        gd_min,
        gd_max,
        roi_pos,
    ) = result
    assert sim_image.shape == (sim_image.shape[0], N_SIMULATED)
    assert uncert_image.shape == sim_image.shape and np.all(uncert_image == 0)
    assert np.nansum(sim_image) > 0
    peak = int(np.argmax(sim_image[:, 0]))
    assert (
        150 <= peak <= 260
    )  # the unset-rxwin branch places the surface top at sample 200
    assert np.all(np.isfinite(first_return_lats)) and np.all(
        np.isfinite(first_return_lons)
    )
    assert np.all((first_return_lats > 0.4) & (first_return_lats < 2.0))
    assert np.all((first_return_lons > 0.0) & (first_return_lons < 1.0))
    both_finite = np.isfinite(gd_min) & np.isfinite(gd_max)
    assert np.any(both_finite)
    assert np.all(gd_min[both_finite] <= gd_max[both_finite] + 1e-6)
    assert isinstance(roi_pos, tuple) and len(roi_pos) == 4
    for roi_dict in roi_pos:
        assert isinstance(roi_dict, dict) and len(roi_dict) == 0


def test_process_count_partition_invariance_with_last_frame_carry(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Checks the phase-6 frame-range partition on the synthetic scene with
    the worker pool capped down to one effective worker: the requested
    n_processes still derives the partition (one range per frame here), the
    ROI-incidence bookkeeping exercises the along-track unit vector at every
    frame including the track's last simulated frame (which reuses the
    previous frame's value, carried in by the parent for the range starting
    at it), and the assembled result is byte-equal to the serial single-call
    result."""
    monkeypatch.setattr(
        sim_base_module,
        "_max_memory_limited_pool_workers",
        lambda cpu_count, available_memory_bytes: 1,
    )
    roi_img = np.zeros((3, N_FRAMES), dtype="int32")
    roi_img[1, :] = 1
    roi_img[2, ::3] = 2

    serial_result = _make_sim(roi_img=roi_img).simulate(_orbit_data())
    assert serial_result is not None
    capsys.readouterr()  # discard the serial run's progress output

    partitioned_result = _make_sim(n_processes=N_SIMULATED, roi_img=roi_img).simulate(
        _orbit_data()
    )
    assert partitioned_result is not None
    assert "PROCESSED RANGE 0 (1/8)" in capsys.readouterr().out

    for serial_part, partitioned_part in zip(serial_result, partitioned_result):
        if isinstance(serial_part, tuple):
            for serial_dict, partitioned_dict in zip(serial_part, partitioned_part):
                assert list(serial_dict.keys()) == list(partitioned_dict.keys())
                for roi_id, serial_values in serial_dict.items():
                    assert serial_values.tobytes() == partitioned_dict[roi_id].tobytes()
        else:
            assert serial_part.tobytes() == partitioned_part.tobytes()
