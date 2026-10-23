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

"""Reduced-parameter version of run_sopa_processing_2610.py, used to verify
the port against the REAL data (PDS EDR edr1260201 + MOLA128 r128 DEM) without
running the full-hours processing of the example script.

One session-scoped fixture runs the whole chain exactly once (MOLA128 read,
sfRSsim simulation, ENVI write of the simulated products, the plots, the DEM
radius profile, SOFA focusing in debug mode, ENVI write and read-back of the
focused product) and the tests then assert on the returned products. The
module requires the SOPA real data under /mnt/SOPA/DATA (marker `realdata`)
and writes its products under /mnt/SOPA/OUTPUT/PDS. It takes about two
minutes, dominated by the 2 GB DEM read.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from sopa.input.sharad.instrument import SHARAD
from sopa.input.sharad.data.edr.pds import EDR as EDR_PDS
from sopa.geometry.coordinate_converter import MARS_IAU2000
from sopa.dem.mola.mola import MOLA128
from sopa.processing.parameters import FocuserParameters, SimulatorParameters
from sopa.processing.simulators.sfrssim import sfRSsim
from sopa.processing.focusers.sofa import SOFA
from sopa.io_utils import envi
from sopa.log.log import logger
from sopa.plot_utils.tracks import TrackPlotOnDEM, TrackAndFirstReturnsPlotOnDEM
from sopa.plot_utils.radargrams import SimulationPlot, GroundDistancePlot, RadargramPlot


DATA_DIR = Path("/mnt/SOPA/DATA")

pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(not DATA_DIR.is_dir(), reason="SOPA real data not present"),
]

MOLA128_FILENAME_BASE = "/mnt/SOPA/DATA/dem/mola/radius/r128"
EDR_PDS_INPUT_FILENAME_BASE = "/mnt/SOPA/DATA/sharad/EDR/PDS/edr1260201/e_1260201_001_ss19_700_a"
EDR_PDS_INPUT_DATASET_NAME = "edr1260201"
EDR_PDS_OUTPUT_SIM_FILENAME_BASE = "/mnt/SOPA/OUTPUT/PDS/PDS-EDR-1260201_SIM_test"
EDR_PDS_OUTPUT_FOC_FILENAME_BASE = "/mnt/SOPA/OUTPUT/PDS/PDS-EDR-1260201_FOC_test"
OUTPUT_PDS_DIR = Path("/mnt/SOPA/OUTPUT/PDS")

# REDUCED processing load wrt the example script (verification only)
# note: the frame range must be long enough wrt METERS_SKIP, as the
# skip-distance frame selection needs at least 2 intervals
FRAME_START = 80000
FRAME_END = 81000
METERS_SKIP = 2500
FRAME_HALF_APERTURE = 64
SQUINT_ANGLE = 0


@dataclass(frozen=True)
class RealDataProducts:
    """Products of the reduced real-data processing chain."""

    edr: EDR_PDS
    sim_image: npt.NDArray[np.float64]
    focused_rdr: npt.NDArray[np.float64]
    foc_back: npt.NDArray | None
    plots: tuple[Any, ...]


@pytest.fixture(scope="session")
def real_products() -> RealDataProducts:
    """Runs the reduced real-data chain exactly once and returns its products.

    The original script guarded the focusing stage with `if d.data is not
    None` / `if focused_rdr is not None`; here a failure at any stage must
    surface as a fixture error, not a silent skip, so those guards are
    assertions."""
    OUTPUT_PDS_DIR.mkdir(parents=True, exist_ok=True)

    lg = logger(verbose=True)
    i = SHARAD()
    cc = MARS_IAU2000()

    m = MOLA128(filename_base=MOLA128_FILENAME_BASE)
    m.read()

    pars = FocuserParameters(frame_start=FRAME_START, frame_end=FRAME_END, meters_skip=METERS_SKIP, frame_half_aperture=FRAME_HALF_APERTURE, squint_angle=SQUINT_ANGLE)
    sim_pars = SimulatorParameters(frame_start=FRAME_START, frame_end=FRAME_END, meters_skip=METERS_SKIP, frame_half_aperture=FRAME_HALF_APERTURE, squint_angle=SQUINT_ANGLE)

    s = sfRSsim(geom_obj=cc, dem_obj=m, param_obj=sim_pars, n_processes=1)

    d = EDR_PDS(dataset_name=EDR_PDS_INPUT_DATASET_NAME, filename_base=EDR_PDS_INPUT_FILENAME_BASE, logger=lg)
    d.load(mode="full")
    d.generate_orbit_data()
    assert d.orbit_data is not None

    tpod = TrackPlotOnDEM(m, d.orbit_data["Lat"], d.orbit_data["Lon"], title="Spacecraft ground track on DEM (full track)")
    tpod.plot()

    sim_result = s.simulate(d.orbit_data)
    assert sim_result is not None
    sim_image, uncert_image, first_return_lats, first_return_lons, ground_distance_min_image, ground_distance_max_image, roi_pos = sim_result

    envi.write(EDR_PDS_OUTPUT_SIM_FILENAME_BASE, sim_image)
    envi.write(f"{EDR_PDS_OUTPUT_SIM_FILENAME_BASE}_uncert", uncert_image)

    top_point = int(np.min(np.argmin(np.isnan(ground_distance_min_image).astype(int), axis=0))) - 50
    bottom_point = top_point + 1000
    sp = SimulationPlot(sim_image, top_point=top_point, bottom_point=bottom_point, title="Radargram simulation with sfRSsim\n(square facets and cos^exp function)")
    sp.plot()

    tfrpod = TrackAndFirstReturnsPlotOnDEM(m, d.orbit_data["Lat"], d.orbit_data["Lon"], first_return_lats, first_return_lons)
    tfrpod.plot()

    gdp = GroundDistancePlot(ground_distance_min_image/1000., top_point=top_point, bottom_point=bottom_point, title="Distance [km] of the closest facet contributing\nto each radargram sample simulation")
    gdp.plot()

    dem_radius_profile = m.get_dem_radius_from_lat_lon(d.orbit_data["Lat"], d.orbit_data["Lon"])
    del m

    assert d.data is not None
    f = SOFA(i, cc, dem_radius_profile, d, pars, n_processes=1, debug_mode=True)
    focused_rdr = f.focus()
    assert focused_rdr is not None

    foc_base = f"{EDR_PDS_OUTPUT_FOC_FILENAME_BASE}_ha-{pars.frame_half_aperture}_s-{pars.squint_angle}"
    envi.write(foc_base, focused_rdr)
    rp = RadargramPlot(focused_rdr, top_point=top_point, bottom_point=bottom_point, title="Radargram")
    rp.plot()
    foc_back = envi.read(foc_base)

    return RealDataProducts(edr=d, sim_image=sim_image, focused_rdr=focused_rdr, foc_back=foc_back, plots=(tpod, sp, tfrpod, gdp, rp))


def test_orbit_extents(real_products: RealDataProducts) -> None:
    """Checks the orbit frame count and the sanity of the lat/lon extents."""
    orbit = real_products.edr.orbit_data
    assert orbit is not None
    lat = orbit["Lat"]
    lon = orbit["Lon"]
    assert len(lat) == real_products.edr.n_frames
    assert np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))
    assert float(np.min(lat)) >= -90.0 and float(np.max(lat)) <= 90.0
    assert float(np.min(lon)) >= 0.0 and float(np.max(lon)) <= 360.0


def test_simulation(real_products: RealDataProducts) -> None:
    """Checks the simulated radargram shape and that it carries energy."""
    sim_image = real_products.sim_image
    assert sim_image.ndim == 2 and sim_image.shape[1] >= 5
    energy = np.nansum(sim_image)
    assert energy > 0, f"nansum = {energy}"


def test_focused_radargram(real_products: RealDataProducts) -> None:
    """Checks the focused radargram shape and that all its samples are finite."""
    focused_rdr = real_products.focused_rdr
    assert focused_rdr.ndim == 2 and focused_rdr.shape[1] >= 5
    assert np.all(np.isfinite(focused_rdr))


def test_focused_envi_roundtrip(real_products: RealDataProducts) -> None:
    """Checks that the focused product reads back from ENVI equal to what was
    written."""
    foc_back = real_products.foc_back
    assert foc_back is not None and np.allclose(foc_back, real_products.focused_rdr)


def test_plots_executed(real_products: RealDataProducts) -> None:
    """Checks that all five plots of the chain exist; the plots run inside the
    session fixture, so a plot exception aborts the fixture with an error."""
    assert len(real_products.plots) == 5
