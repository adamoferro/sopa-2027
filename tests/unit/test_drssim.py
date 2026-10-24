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

"""Unit tests of the discrete-point clutter simulator
(sopa.processing.simulators.drssim): constructor defaults (including the
chirp requirement), and small end-to-end runs on a synthetic DEM and track
checking the unweighted point counts, the nadir peak placement and the chirp
convolution variant.
"""

import numpy as np
import numpy.typing as npt
import pytest

from sopa.dem.mola.mola import MOLA128
from sopa.geometry.coordinate_converter import MARS_IAU2000
from sopa.input.sharad.instrument import SHARAD
from sopa.processing.parameters import SimulatorParameters
from sopa.processing.simulators.drssim import dRSsim

N_FRAMES = 12
N_HALF_APERTURE = 2
FRAME_START = N_HALF_APERTURE
FRAME_END = 10
N_SIMULATED = FRAME_END - FRAME_START  # arange(start, end) with skip 1


def _dem() -> MOLA128:
    """Builds a small synthetic DEM covering lat 2..0.4375, lon 0..1 deg."""
    rng = np.random.default_rng(22)
    values = rng.integers(-100, 100, size=(200, 128)).astype("int16")
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


def _make_sim(simulate_chirp: bool = False, instr_obj: object = None) -> dRSsim:
    """Builds a dRSsim instance with the across/along-track grids shrunk to
    keep the unit run fast (the matrices are per-instance attributes)."""
    sim = dRSsim(
        geom_obj=MARS_IAU2000(),
        dem_obj=_dem(),
        param_obj=SimulatorParameters(
            frame_start=FRAME_START,
            frame_end=FRAME_END,
            frame_half_aperture=N_HALF_APERTURE,
        ),
        n_processes=1,
        simulate_chirp=simulate_chirp,
        instr_obj=instr_obj,
    )
    alt = np.arange(-250.0, 250.0 + 25.0, 25.0)
    act = np.arange(-2500.0, 2500.0 + 50.0, 50.0)
    sim.alt_sim_matrix = np.array([alt]).T * np.ones([1, len(act)])
    sim.act_sim_matrix = np.array([act]) * np.ones([len(alt), 1])
    return sim


def test_constructor_requires_instrument_for_default_chirp() -> None:
    """Checks that the constructor's default simulate_chirp=True requires an
    instrument object."""
    with pytest.raises(ValueError):
        dRSsim(geom_obj=MARS_IAU2000(), dem_obj=_dem(), param_obj=SimulatorParameters())


def test_constructor_defaults() -> None:
    """Checks the constructor defaults: smoothing sigma (half the sfRSsim
    one), spacings, grid extents and the point-grid matrix shapes."""
    sim = dRSsim(
        geom_obj=None,
        dem_obj=None,
        param_obj=SimulatorParameters(),
        simulate_chirp=False,
    )
    assert sim.gaussian_filter_sigma == 4
    assert sim.alt_sim_spacing == 50
    assert sim.act_sim_spacing == 50
    assert sim.alt_sim_forward_max_distance == 500
    assert sim.alt_sim_backward_max_distance == 500
    assert sim.act_sim_max_distance == 50000
    assert sim.alt_sim_matrix.shape == (21, 2001)
    assert sim.act_sim_matrix.shape == (21, 2001)
    assert sim.alt_sim_matrix[0, 0] == -500.0
    assert sim.act_sim_matrix[0, -1] == 50000.0


def test_simulation_counts_points() -> None:
    """Checks the end-to-end point simulation without chirp: every simulated
    sample holds an integer point count, the nadir return is near the
    reference sample, no uncertainty is flagged and the first returns fall
    inside the DEM coverage."""
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
    assert np.all(
        sim_image == np.round(sim_image)
    )  # one count per grid point, no facet weighting
    assert np.sum(sim_image) > 0
    peak = int(np.argmax(sim_image[:, 0]))
    assert (
        150 <= peak <= 260
    )  # the unset-rxwin branch places the surface top at sample 200
    assert np.all(uncert_image == 0)
    assert np.all(np.isfinite(first_return_lats)) and np.all(
        np.isfinite(first_return_lons)
    )
    assert isinstance(roi_pos, tuple) and len(roi_pos) == 2


def test_simulation_with_chirp() -> None:
    """Checks the chirp-convolved variant: finite non-negative output with
    preserved shape and positive energy."""
    sim = _make_sim(simulate_chirp=True, instr_obj=SHARAD())
    result = sim.simulate(_orbit_data())
    assert result is not None
    sim_image = result[0]
    assert np.all(np.isfinite(sim_image))
    assert np.all(sim_image >= 0)
    assert np.sum(sim_image) > 0
