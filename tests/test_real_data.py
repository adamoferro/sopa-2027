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

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from sopa.dem.mola.mola import MOLA128
from sopa.geometry.coordinate_converter import MARS_IAU2000
from sopa.input.sharad.data.edr.pds import EDR as EDR_PDS
from sopa.input.sharad.instrument import SHARAD
from sopa.io_utils import envi
from sopa.log.log import logger
from sopa.plot_utils.radargrams import GroundDistancePlot, RadargramPlot, SimulationPlot
from sopa.plot_utils.tracks import TrackAndFirstReturnsPlotOnDEM, TrackPlotOnDEM
from sopa.processing.focusers.sofa import SOFA
from sopa.processing.parameters import FocuserParameters, SimulatorParameters
from sopa.processing.simulators.drssim import dRSsim
from sopa.processing.simulators.sfrssim import sfRSsim

DATA_DIR = Path("/mnt/SOPA/DATA")

pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(not DATA_DIR.is_dir(), reason="SOPA real data not present"),
]

MOLA128_FILENAME_BASE = "/mnt/SOPA/DATA/dem/mola/radius/r128"
EDR_PDS_INPUT_FILENAME_BASE = (
    "/mnt/SOPA/DATA/sharad/EDR/PDS/edr1260201/e_1260201_001_ss19_700_a"
)
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
    dem_radius_profile: npt.NDArray
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

    pars = FocuserParameters(
        frame_start=FRAME_START,
        frame_end=FRAME_END,
        meters_skip=METERS_SKIP,
        frame_half_aperture=FRAME_HALF_APERTURE,
        squint_angle=SQUINT_ANGLE,
    )
    sim_pars = SimulatorParameters(
        frame_start=FRAME_START,
        frame_end=FRAME_END,
        meters_skip=METERS_SKIP,
        frame_half_aperture=FRAME_HALF_APERTURE,
        squint_angle=SQUINT_ANGLE,
    )

    s = sfRSsim(geom_obj=cc, dem_obj=m, param_obj=sim_pars, n_processes=1)

    d = EDR_PDS(
        dataset_name=EDR_PDS_INPUT_DATASET_NAME,
        filename_base=EDR_PDS_INPUT_FILENAME_BASE,
        logger=lg,
    )
    d.load(mode="full")
    d.generate_orbit_data()
    assert d.orbit_data is not None

    tpod = TrackPlotOnDEM(
        m,
        d.orbit_data["Lat"],
        d.orbit_data["Lon"],
        title="Spacecraft ground track on DEM (full track)",
    )
    tpod.plot()

    sim_result = s.simulate(d.orbit_data)
    assert sim_result is not None
    (
        sim_image,
        uncert_image,
        first_return_lats,
        first_return_lons,
        ground_distance_min_image,
        ground_distance_max_image,
        roi_pos,
    ) = sim_result

    envi.write(EDR_PDS_OUTPUT_SIM_FILENAME_BASE, sim_image)
    envi.write(f"{EDR_PDS_OUTPUT_SIM_FILENAME_BASE}_uncert", uncert_image)

    top_point = (
        int(np.min(np.argmin(np.isnan(ground_distance_min_image).astype(int), axis=0)))
        - 50
    )
    bottom_point = top_point + 1000
    sp = SimulationPlot(
        sim_image,
        top_point=top_point,
        bottom_point=bottom_point,
        title="Radargram simulation with sfRSsim\n(square facets and cos^exp function)",
    )
    sp.plot()

    tfrpod = TrackAndFirstReturnsPlotOnDEM(
        m,
        d.orbit_data["Lat"],
        d.orbit_data["Lon"],
        first_return_lats,
        first_return_lons,
    )
    tfrpod.plot()

    gdp = GroundDistancePlot(
        ground_distance_min_image / 1000.0,
        top_point=top_point,
        bottom_point=bottom_point,
        title="Distance [km] of the closest facet contributing\nto each radargram sample simulation",
    )
    gdp.plot()

    dem_radius_profile = m.get_dem_radius_from_lat_lon(
        d.orbit_data["Lat"], d.orbit_data["Lon"]
    )
    del m

    assert d.data is not None
    f = SOFA(i, cc, dem_radius_profile, d, pars, n_processes=1, debug_mode=True)
    focused_rdr = f.focus()
    assert focused_rdr is not None

    foc_base = f"{EDR_PDS_OUTPUT_FOC_FILENAME_BASE}_ha-{pars.frame_half_aperture}_s-{pars.squint_angle}"
    envi.write(foc_base, focused_rdr)
    rp = RadargramPlot(
        focused_rdr, top_point=top_point, bottom_point=bottom_point, title="Radargram"
    )
    rp.plot()
    foc_back = envi.read(foc_base)

    return RealDataProducts(
        edr=d,
        dem_radius_profile=dem_radius_profile,
        sim_image=sim_image,
        focused_rdr=focused_rdr,
        foc_back=foc_back,
        plots=(tpod, sp, tfrpod, gdp, rp),
    )


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


def test_focus_block_partition_invariance(
    real_products: RealDataProducts, capsys: pytest.CaptureFixture[str]
) -> None:
    """Checks that focusing is invariant to the read-block partition: the same
    range focused once with a single block (read_block_size_max 5000, the
    constructor default) and once with several blocks (read_block_size_max
    200, which splits this range into 4 blocks of 2 focused frames each)
    produces byte-identical outputs. This pins the partition-invariance of the
    per-frame focus outputs that the n_processes reactivation of the focuser
    (phase-6 task P5) builds on."""
    pars = FocuserParameters(
        frame_start=FRAME_START,
        frame_end=FRAME_END,
        meters_skip=METERS_SKIP,
        frame_half_aperture=FRAME_HALF_APERTURE,
        squint_angle=SQUINT_ANGLE,
    )

    def focus_with(read_block_size_max: int) -> npt.NDArray:
        f = SOFA(
            SHARAD(),
            MARS_IAU2000(),
            real_products.dem_radius_profile,
            real_products.edr,
            pars,
            n_processes=1,
            read_block_size_max=read_block_size_max,
            debug_mode=True,
        )
        focused_rdr = f.focus()
        assert focused_rdr is not None
        return focused_rdr

    out_one_block = focus_with(5000)
    capsys.readouterr()  # discard the single-block run's debug output

    out_small_blocks = focus_with(200)
    n_blocks_match = re.search(r"N BLOCKS = (\d+)", capsys.readouterr().out)
    assert n_blocks_match is not None, "no N BLOCKS line in the debug output"
    assert int(n_blocks_match.group(1)) >= 2, (
        "read_block_size_max=200 did not split the range into multiple blocks"
    )

    assert out_small_blocks.tobytes() == out_one_block.tobytes()


def test_focus_process_count_invariance(
    real_products: RealDataProducts, capsys: pytest.CaptureFixture[str]
) -> None:
    """Checks that focusing with n_processes=2 (the fork-based worker pool of
    the phase-6 P5 reactivation) produces byte-identical output to the serial
    n_processes=1 path on the same real-data range with the same
    read_block_size_max. The pool workers reach the raw data through fork
    copy-on-write and run the same per-block operation sequence, and the
    output is assembled in block order, so not a single byte may differ. This
    is the end-to-end pin of the n_processes reactivation."""
    pars = FocuserParameters(
        frame_start=FRAME_START,
        frame_end=FRAME_END,
        meters_skip=METERS_SKIP,
        frame_half_aperture=FRAME_HALF_APERTURE,
        squint_angle=SQUINT_ANGLE,
    )

    def focus_with(n_processes: int) -> npt.NDArray:
        f = SOFA(
            SHARAD(),
            MARS_IAU2000(),
            real_products.dem_radius_profile,
            real_products.edr,
            pars,
            n_processes=n_processes,
            read_block_size_max=5000,
            debug_mode=True,
        )
        focused_rdr = f.focus()
        assert focused_rdr is not None
        return focused_rdr

    out_one_process = focus_with(1)
    capsys.readouterr()  # discard the serial run's debug output

    out_two_processes = focus_with(2)
    n_blocks_match = re.search(
        r"Focusing (\d+) block\(s\) on 2 worker processes",
        capsys.readouterr().out,
    )
    assert n_blocks_match is not None, "no worker-pool line in the debug output"
    assert int(n_blocks_match.group(1)) >= 2, (
        "n_processes=2 did not split the range into multiple blocks"
    )

    assert out_two_processes.tobytes() == out_one_process.tobytes()


@dataclass(frozen=True)
class SimulationInputs:
    """Shared inputs of the simulator process-count invariance tests."""

    orbit_data: dict[str, npt.NDArray[np.float64] | float]
    dem: MOLA128


@pytest.fixture(scope="session")
def simulation_inputs() -> SimulationInputs:
    """Loads the EDR orbit data (ancillary data only, no echo data needed for
    the simulation) and the MOLA128 DEM once for the simulator
    process-count invariance tests; the DEM read dominates the fixture cost."""
    lg = logger(verbose=True)
    d = EDR_PDS(
        dataset_name=EDR_PDS_INPUT_DATASET_NAME,
        filename_base=EDR_PDS_INPUT_FILENAME_BASE,
        logger=lg,
    )
    d.load(mode="ancillary")
    d.generate_orbit_data()
    assert d.orbit_data is not None

    m = MOLA128(filename_base=MOLA128_FILENAME_BASE)
    m.read()
    return SimulationInputs(orbit_data=d.orbit_data, dem=m)


def _simulate_with_engine(
    engine: type[sfRSsim] | type[dRSsim],
    n_processes: int,
    orbit_data: dict[str, npt.NDArray[np.float64] | float],
    dem: MOLA128,
) -> tuple[npt.NDArray[np.float64], ...]:
    """Runs one simulator engine over the reduced real-data range and returns
    the full simulate() result tuple."""
    sim_pars = SimulatorParameters(
        frame_start=FRAME_START,
        frame_end=FRAME_END,
        meters_skip=METERS_SKIP,
        frame_half_aperture=FRAME_HALF_APERTURE,
        squint_angle=SQUINT_ANGLE,
    )
    if engine is dRSsim:
        # mirror run_matrix.py: the matrix loads no instrument object, so the
        # dRSsim engine is constructed with the chirp convolution disabled
        simulator = engine(
            geom_obj=MARS_IAU2000(),
            dem_obj=dem,
            param_obj=sim_pars,
            n_processes=n_processes,
            simulate_chirp=False,
        )
    else:
        simulator = engine(
            geom_obj=MARS_IAU2000(),
            dem_obj=dem,
            param_obj=sim_pars,
            n_processes=n_processes,
        )
    result = simulator.simulate(orbit_data)
    assert result is not None
    return result


def _assert_simulation_results_byte_equal(
    serial_result: tuple[npt.NDArray[np.float64], ...],
    parallel_result: tuple[npt.NDArray[np.float64], ...],
    engine_name: str,
) -> None:
    """Asserts byte equality of the full simulate() results: images, first
    returns, ground distances and the ROI dictionaries (keys in first-appearance
    order, values concatenated in frame order)."""
    product_names = (
        "sim_image",
        "uncert_image",
        "first_return_lats",
        "first_return_lons",
        "ground_distance_min_image",
        "ground_distance_max_image",
    )
    for product_index, product_name in enumerate(product_names):
        assert (
            serial_result[product_index].tobytes()
            == parallel_result[product_index].tobytes()
        ), (
            f"{engine_name}: {product_name} differs between n_processes=1 and n_processes=2"
        )

    assert len(serial_result[6]) == len(parallel_result[6])
    for roi_dict_index, (serial_dicts, parallel_dicts) in enumerate(
        zip(serial_result[6], parallel_result[6])
    ):
        assert list(serial_dicts.keys()) == list(parallel_dicts.keys()), (
            f"{engine_name}: ROI dict {roi_dict_index} keys differ"
        )
        for roi_id, serial_values in serial_dicts.items():
            assert serial_values.tobytes() == parallel_dicts[roi_id].tobytes(), (
                f"{engine_name}: ROI dict {roi_dict_index} values for id {roi_id} differ"
            )


def test_sfrssim_process_count_invariance(
    simulation_inputs: SimulationInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    """Checks that simulating with n_processes=2 (the fork-based worker pool
    of the phase-6 P6 reactivation) produces byte-identical products to the
    serial n_processes=1 path on the same real-data range. The pool workers
    reach the DEM and the per-track arrays through fork copy-on-write and run
    the same per-frame operation sequence, and the outputs are assembled in
    frame order, so not a single byte may differ. This is the end-to-end pin
    of the simulator n_processes reactivation for the sfRSsim engine."""
    orbit_data = simulation_inputs.orbit_data
    dem = simulation_inputs.dem

    serial_result = _simulate_with_engine(sfRSsim, 1, orbit_data, dem)
    capsys.readouterr()  # discard the serial run's progress output

    parallel_result = _simulate_with_engine(sfRSsim, 2, orbit_data, dem)
    pool_line_match = re.search(
        r"Simulating (\d+) frame range\(s\) on 2 worker processes",
        capsys.readouterr().out,
    )
    assert pool_line_match is not None, "no worker-pool line in the output"
    assert int(pool_line_match.group(1)) >= 2, (
        "n_processes=2 did not split the range into multiple frame ranges"
    )

    _assert_simulation_results_byte_equal(serial_result, parallel_result, "sfRSsim")


def test_drssim_process_count_invariance(
    simulation_inputs: SimulationInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    """Checks that simulating with n_processes=2 (the fork-based worker pool
    of the phase-6 P6 reactivation) produces byte-identical products to the
    serial n_processes=1 path on the same real-data range, for the dRSsim
    engine (constructed with simulate_chirp=False, mirroring the fidelity
    matrix's run script, which loads no instrument object)."""
    orbit_data = simulation_inputs.orbit_data
    dem = simulation_inputs.dem

    serial_result = _simulate_with_engine(dRSsim, 1, orbit_data, dem)
    capsys.readouterr()  # discard the serial run's progress output

    parallel_result = _simulate_with_engine(dRSsim, 2, orbit_data, dem)
    pool_line_match = re.search(
        r"Simulating (\d+) frame range\(s\) on 2 worker processes",
        capsys.readouterr().out,
    )
    assert pool_line_match is not None, "no worker-pool line in the output"
    assert int(pool_line_match.group(1)) >= 2, (
        "n_processes=2 did not split the range into multiple frame ranges"
    )

    _assert_simulation_results_byte_equal(serial_result, parallel_result, "dRSsim")
