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

"""End-to-end smoke tests on a fully synthetic SHARAD-like dataset.

Synthetic PDS EDR ingredients (PDS-like header, orbit ancillary file, 4-bit
compressed science data) and a synthetic MOLA128-like DEM are built in a
temporary directory, then the whole SOPA chain runs on them exactly once per
session: EDR load, orbit data generation, sfRSsim simulation, SOFA focusing,
and ENVI/PNG output with read-back. The tests verify that every low-level
mechanism (big-endian record parsing, bit unpacking, ENVI I/O, simulators,
focuser) works on the current Python/numpy/scipy stack; they do not verify
the physical correctness of the results against real reference outputs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import bitstring as bstr
import pytest

from sopa.input.sharad.instrument import SHARAD
from sopa.input.sharad.data.edr.pds import EDR as EDR_PDS
from sopa.geometry.coordinate_converter import MARS_IAU2000
from sopa.dem.mola.mola import MOLA128
from sopa.processing.parameters import FocuserParameters, SimulatorParameters
from sopa.processing.simulators.sfrssim import sfRSsim
from sopa.processing.focusers.sofa import SOFA
from sopa.io_utils import envi, png as png_io
from sopa.log.log import logger
from sopa.plot_utils.tracks import TrackPlotOnDEM, TrackAndFirstReturnsPlotOnDEM
from sopa.plot_utils.radargrams import SimulationPlot, GroundDistancePlot, RadargramPlot


DATASET_NAME = "e_9999999"
N_FRAMES = 4000
PRI_US = 1428
PRESUMMING = 16                 # IM SS3
N_BITS_PER_BYTE = 4
N_BYTES_PER_RECORD = 1800
N_SAMPLES = 3600
DT = 3 / 80e6
DCG_DELAY = 11.98e-6
# rxwin opening time (in DT steps) chosen so that the nadir surface echo of the
# synthetic geometry (altitude ~250 km) falls inside the simulated window
RXWIN = 6674.
DEM_ROWS = 11700
DEM_COLS = 512                  # 4 degrees of longitude at 1/128 deg


@dataclass(frozen=True)
class SyntheticOrbit:
    """Objects of the synthetic orbit stage: the coordinate converter and the
    latitude/longitude arrays of the synthetic spacecraft track."""

    coord_converter: MARS_IAU2000
    lat: npt.NDArray[np.float64]
    lon: npt.NDArray[np.float64]


@dataclass(frozen=True)
class SyntheticScienceData:
    """The known integer sample values the synthetic 4-bit science data pack."""

    true_ints: npt.NDArray[np.int64]


@dataclass(frozen=True)
class SyntheticEdr:
    """The loaded synthetic EDR object plus the return value of its load call."""

    edr: EDR_PDS
    load_ok: bool


@dataclass(frozen=True)
class SimulationOutputs:
    """The seven outputs of one sfRSsim simulation run."""

    sim_image: npt.NDArray[np.float64]
    uncert_image: npt.NDArray[np.uint8]
    first_return_lats: npt.NDArray[np.float64]
    first_return_lons: npt.NDArray[np.float64]
    ground_distance_min_image: npt.NDArray[np.float64]
    ground_distance_max_image: npt.NDArray[np.float64]
    roi_pos: tuple[Any, ...]


@dataclass(frozen=True)
class EnviRoundTrip:
    """Result of writing an array with the ENVI writer and reading it back."""

    write_ok: bool
    read_back: npt.NDArray | None
    data_base: Path


# --------------------------------------------------------------------------- #
# 1) synthetic DEM (ENVI file, same conventions as the real MOLA128 product)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Working directory for all synthetic-dataset files of this session."""
    return tmp_path_factory.mktemp("sopa_synth")


@pytest.fixture(scope="session")
def synth_dem(synth_dir: Path) -> MOLA128:
    """Builds the synthetic MOLA128-like DEM, writes it as ENVI and reads it
    back through the internal ENVI reader."""
    rng = np.random.default_rng(42)
    dem_values = (500 + 100*np.sin(np.arange(DEM_COLS)/20.) + rng.normal(0, 5, (DEM_ROWS, DEM_COLS))).astype("int16")
    dem_filename_base = synth_dir / "synth_dem"
    envi.write(str(dem_filename_base), dem_values)
    m = MOLA128(filename_base=str(dem_filename_base))
    m.read()
    return m


# --------------------------------------------------------------------------- #
# 2) synthetic orbit (circular, equatorial, lon 0 -> 2 deg)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def synth_orbit(synth_dir: Path) -> SyntheticOrbit:
    """Writes the synthetic orbit ancillary file and PDS-like header and
    returns the coordinate converter plus the synthetic track arrays."""
    cc = MARS_IAU2000()
    lat = np.linspace(-0.2, 0.2, N_FRAMES)      # slightly inclined track, like a real one
    lon = np.linspace(0.1, 2.1, N_FRAMES)
    sc_radius = np.full(N_FRAMES, 3396190. + 250000.) + 500.*np.sin(np.linspace(0, np.pi, N_FRAMES))       # ~250 km altitude, slowly varying
    x, y, z = cc.to_xyz_from_latlon_and_radius(lat, lon, sc_radius)
    vtang = np.full(N_FRAMES, 3200.)
    vradial = np.zeros(N_FRAMES)

    # geom record layout (copied from input/sharad/data/edr/pds.py)
    geom_dt = np.dtype([("SCET_BLOCK_WHOLE", ">u4"), ("SCET_BLOCK_FRAC", ">u2"),
                    ("EPHEMERIS_TIME", ">f8"), ("GEOMETRY_EPOCH", np.void, 23), ("SOLAR_LONGITUDE", ">f8"),
                    ("ORBIT_NUMBER", ">i4"), ("X_MARS_SC_POSITION_VECTOR", ">f8"), ("Y_MARS_SC_POSITION_VECTOR", ">f8"),
                    ("Z_MARS_SC_POSITION_VECTOR", ">f8"), ("SPACECRAFT_ALTITUDE", ">f8"), ("SUB_SC_EAST_LONGITUDE", ">f8"),
                    ("SUB_SC_PLANETOCENTRIC_LATITUDE", ">f8"), ("SUB_SC_PLANETOGRAPHIC_LATITUDE", ">f8"),
                    ("X_MARS_SC_VELOCITY_VECTOR", ">f8"), ("Y_MARS_SC_VELOCITY_VECTOR", ">f8"), ("Z_MARS_SC_VELOCITY_VECTOR", ">f8"),
                    ("MARS_SC_RADIAL_VELOCITY", ">f8"), ("MARS_SC_TANGENTIAL_VELOCITY", ">f8"), ("LOCAL_TRUE_SOLAR_TIME", ">f8"),
                    ("SOLAR_ZENITH_ANGLE", ">f8"), ("SC_PITCH_ANGLE", ">f8"), ("SC_YAW_ANGLE", ">f8"), ("SC_ROLL_ANGLE", ">f8"),
                    ("MRO_SAMX_INNER_GIMBAL_ANGLE", ">f8"), ("MRO_SAMX_OUTER_GIMBAL_ANGLE", ">f8"),
                    ("MRO_SAPX_INNER_GIMBAL_ANGLE", ">f8"), ("MRO_SAPX_OUTER_GIMBAL_ANGLE", ">f8"),
                    ("MRO_HGA_INNER_GIMBAL_ANGLE", ">f8"), ("MRO_HGA_OUTER_GIMBAL_ANGLE", ">f8"),
                    ("DES_TEMP", ">f4"), ("DES_5V", ">f4"), ("DES_12V", ">f4"), ("DES_2V5", ">f4"),
                    ("RX_TEMP", ">f4"), ("TX_TEMP", ">f4"), ("TX_LEV", ">f4"), ("TX_CURR", ">f4"),
                    ("CORRUPTED_DATA_FLAG", ">i2")])
    geom = np.zeros(N_FRAMES, dtype=geom_dt)
    geom["SCET_BLOCK_WHOLE"] = np.arange(N_FRAMES) * 100
    geom["SCET_BLOCK_FRAC"] = 0
    geom["EPHEMERIS_TIME"] = np.arange(N_FRAMES) * (PRI_US * PRESUMMING * 1e-6)
    # note: PDS EDR ancillary files store positions in km (the reader converts to m)
    geom["X_MARS_SC_POSITION_VECTOR"] = x / 1000.
    geom["Y_MARS_SC_POSITION_VECTOR"] = y / 1000.
    geom["Z_MARS_SC_POSITION_VECTOR"] = z / 1000.
    geom["SUB_SC_EAST_LONGITUDE"] = lon
    geom["SUB_SC_PLANETOCENTRIC_LATITUDE"] = lat
    geom["MARS_SC_TANGENTIAL_VELOCITY"] = vtang
    geom["MARS_SC_RADIAL_VELOCITY"] = vradial
    geom["SOLAR_ZENITH_ANGLE"] = 45.

    with open(synth_dir / (DATASET_NAME + "_a_a.dat"), "wb") as fp:
        fp.write(geom.tobytes())

    # PDS-like header (fields must appear in the order searched by the reader)
    header_txt = f"""PDS_VERSION_ID                 = "PDS3"
START_TIME                     = 2010-01-01T00:00:00
FILE_RECORDS                   = {N_FRAMES}
INSTRUMENT_MODE_ID             = SS3
MRO:PULSE_REPETITION_INTERVAL  = {PRI_US} <MICROSECOND>
MRO:PHASE_COMPENSATION_TYPE    = "NO COMPENSATION"
MRO:COMPRESSION_SELECTION_FLAG = "STATIC"
DATA_QUALITY_ID                = "0"
END
"""
    with open(synth_dir / (DATASET_NAME + "_a.lbl"), "wt") as fp:
        fp.write(header_txt)

    return SyntheticOrbit(coord_converter=cc, lat=lat, lon=lon)


# --------------------------------------------------------------------------- #
# 3) synthetic 4-bit compressed science data (known sample values)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def synth_science_data(synth_dir: Path) -> SyntheticScienceData:
    """Writes the synthetic 4-bit compressed science data file and returns the
    known integer sample values it packs."""
    rng2 = np.random.default_rng(7)
    true_ints = rng2.integers(-8, 8, size=(N_SAMPLES, N_FRAMES)).astype("int")     # 4-bit signed values
    rxwin_values = np.full(N_FRAMES, RXWIN, dtype=">f4")

    with open(synth_dir / (DATASET_NAME + "_a_s.dat"), "wb") as fp:
        for i_f in range(N_FRAMES):
            fp.write(b"\x00" * 178)
            fp.write(rxwin_values[i_f:i_f+1].tobytes())     # slice keeps the big-endian dtype (a scalar would be converted to native order)
            fp.write(b"\x00" * 4)
            # simple, unambiguous packing: 4 bits per sample, MSB first
            ba = bstr.BitArray()
            for v in true_ints[:, i_f]:
                ba.append(bstr.BitArray(int=int(v), length=4))
            fp.write(ba.tobytes())

    return SyntheticScienceData(true_ints=true_ints)


# --------------------------------------------------------------------------- #
# 4) EDR reading (header, ancillary, 4-bit unpacking, static decompression)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def synth_edr(synth_dem: MOLA128, synth_orbit: SyntheticOrbit, synth_science_data: SyntheticScienceData, synth_dir: Path) -> SyntheticEdr:
    """Loads the synthetic EDR (full mode) and generates its orbit data.

    Depends on the DEM, orbit and science-data fixtures so that the pipeline
    stages run once, in the original order."""
    lg = logger(verbose=False)
    d = EDR_PDS(dataset_name="smoke", filename_base=str(synth_dir / (DATASET_NAME + "_a")), logger=lg)
    load_ok = d.load(mode="full")
    d.generate_orbit_data()
    return SyntheticEdr(edr=d, load_ok=load_ok)


# --------------------------------------------------------------------------- #
# 5) DEM radius profile along the track
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def synth_dem_radius_profile(synth_dem: MOLA128, synth_edr: SyntheticEdr) -> npt.NDArray[np.int32] | None:
    """DEM radius profile along the synthetic track."""
    return synth_dem.get_dem_radius_from_lat_lon(synth_edr.edr.orbit_data["Lat"], synth_edr.edr.orbit_data["Lon"])


# --------------------------------------------------------------------------- #
# 6) sfRSsim surface clutter simulation (serial)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def synth_simulation(synth_dem: MOLA128, synth_orbit: SyntheticOrbit, synth_edr: SyntheticEdr) -> SimulationOutputs:
    """Runs sfRSsim once (serial, reduced parameters) and returns its outputs."""
    sim_pars = SimulatorParameters(frame_start=20, frame_end=N_FRAMES-20, meters_skip=20000, frame_half_aperture=8)
    s = sfRSsim(geom_obj=synth_orbit.coord_converter, dem_obj=synth_dem, param_obj=sim_pars, n_processes=1)
    sim_result = s.simulate(synth_edr.edr.orbit_data)
    assert sim_result is not None
    sim_image, uncert_image, first_return_lats, first_return_lons, ground_distance_min_image, ground_distance_max_image, roi_pos = sim_result
    return SimulationOutputs(sim_image=sim_image, uncert_image=uncert_image,
                             first_return_lats=first_return_lats, first_return_lons=first_return_lons,
                             ground_distance_min_image=ground_distance_min_image,
                             ground_distance_max_image=ground_distance_max_image, roi_pos=roi_pos)


# --------------------------------------------------------------------------- #
# 7) ENVI write + read back (simulated radargram)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def synth_sim_roundtrip(synth_simulation: SimulationOutputs, synth_dir: Path) -> EnviRoundTrip:
    """Writes the simulated image as ENVI and reads it back once."""
    out_base = synth_dir / "sim_out"
    write_ok = envi.write(str(out_base), synth_simulation.sim_image)
    read_back = envi.read(str(out_base))
    return EnviRoundTrip(write_ok=write_ok, read_back=read_back, data_base=out_base)


# --------------------------------------------------------------------------- #
# 9) SOFA focusing (serial) + focused ENVI roundtrip + PNG
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def synth_focused(synth_orbit: SyntheticOrbit, synth_edr: SyntheticEdr, synth_dem_radius_profile: npt.NDArray[np.int32] | None) -> npt.NDArray[np.float64] | None:
    """Runs the SOFA focuser once (serial, reduced parameters)."""
    i = SHARAD()
    foc_pars = FocuserParameters(frame_start=20, frame_end=N_FRAMES-20, meters_skip=20000, frame_half_aperture=8, squint_angle=0)
    f = SOFA(i, synth_orbit.coord_converter, synth_dem_radius_profile, synth_edr.edr, foc_pars, n_processes=1, debug_mode=False)
    return f.focus()


@pytest.fixture(scope="session")
def synth_focused_roundtrip(synth_focused: npt.NDArray[np.float64] | None, synth_dir: Path) -> EnviRoundTrip:
    """Writes the focused radargram as ENVI and reads it back once."""
    focused_rdr = synth_focused
    assert focused_rdr is not None
    foc_base = synth_dir / "foc_out"
    write_ok = envi.write(str(foc_base), focused_rdr)
    read_back = envi.read(str(foc_base))
    return EnviRoundTrip(write_ok=write_ok, read_back=read_back, data_base=foc_base)


# --------------------------------------------------------------------------- #
# tests (one per check of the original script, in the original order)
# --------------------------------------------------------------------------- #
def test_dem_read(synth_dem: MOLA128) -> None:
    """Checks that the synthetic DEM was read through the internal ENVI reader."""
    m = synth_dem
    assert m.dem is not None and m.dem.shape == (DEM_ROWS, DEM_COLS), f"dem shape = {m.dem.shape if m.dem is not None else None}"


def test_edr_load(synth_edr: SyntheticEdr) -> None:
    """Checks that the synthetic EDR load reported success."""
    assert synth_edr.load_ok is True


def test_edr_n_frames(synth_edr: SyntheticEdr) -> None:
    """Checks the frame count parsed from the synthetic EDR."""
    assert synth_edr.edr.n_frames == N_FRAMES, f"n_frames = {synth_edr.edr.n_frames}"


def test_edr_pri(synth_edr: SyntheticEdr) -> None:
    """Checks the PRI parsed from the synthetic PDS-like header."""
    assert abs(synth_edr.edr.pri - PRI_US*1e-6) < 1e-12, f"pri = {synth_edr.edr.pri}"


def test_edr_presumming(synth_edr: SyntheticEdr) -> None:
    """Checks the presumming factor parsed from the synthetic PDS-like header."""
    assert synth_edr.edr.presumming == PRESUMMING, f"presumming = {synth_edr.edr.presumming}"


def test_edr_4bit_unpack_and_decompression(synth_edr: SyntheticEdr, synth_science_data: SyntheticScienceData) -> None:
    """Checks the 4-bit unpacking plus static decompression against the known
    sample values packed into the synthetic science data."""
    factorL = int(np.ceil(np.log2(PRESUMMING)))
    factorS = factorL - N_BITS_PER_BYTE + 8
    expected = synth_science_data.true_ints * 2**factorS / PRESUMMING
    assert np.allclose(synth_edr.edr.data[:100, :50], expected[:100, :50]), f"data shape {synth_edr.edr.data.shape}"


def test_orbit_data_generated(synth_edr: SyntheticEdr) -> None:
    """Checks that the orbit data were generated for the synthetic EDR."""
    assert synth_edr.edr.orbit_data is not None


def test_orbit_rxwin(synth_edr: SyntheticEdr) -> None:
    """Checks the rxwin opening time against its analytical expectation."""
    rxwin_expected = RXWIN*DT - DCG_DELAY + PRI_US*1e-6
    assert np.allclose(synth_edr.edr.orbit_data["rxwin_time_s"], rxwin_expected), f"rxwin_time_s[0] = {synth_edr.edr.orbit_data['rxwin_time_s'][0]:.6f}"


def test_orbit_lat_lon(synth_edr: SyntheticEdr, synth_orbit: SyntheticOrbit) -> None:
    """Checks that the decoded orbit lat/lon match the synthetic track."""
    assert np.allclose(synth_edr.edr.orbit_data["Lat"][::100], synth_orbit.lat[::100]) and np.allclose(synth_edr.edr.orbit_data["Lon"][::100], synth_orbit.lon[::100])


def test_dem_radius_profile(synth_dem_radius_profile: npt.NDArray[np.int32] | None) -> None:
    """Checks the DEM radius profile along the synthetic track."""
    dem_radius_profile = synth_dem_radius_profile
    assert dem_radius_profile is not None
    assert np.all(dem_radius_profile > 3396000), f"mean = {np.mean(dem_radius_profile):.0f} m"


def test_simulation_outputs(synth_simulation: SimulationOutputs) -> None:
    """Checks the shape of the simulated radargram."""
    assert synth_simulation.sim_image.shape[1] >= 5, f"sim shape {synth_simulation.sim_image.shape}"


def test_simulation_has_energy(synth_simulation: SimulationOutputs) -> None:
    """Checks that the simulated radargram carries energy."""
    energy = np.nansum(synth_simulation.sim_image)
    assert energy > 0, f"sum = {energy:.1f}"


def test_first_returns_set(synth_simulation: SimulationOutputs) -> None:
    """Checks that all first-return lat/lon values are finite."""
    assert np.all(np.isfinite(synth_simulation.first_return_lats)) and np.all(np.isfinite(synth_simulation.first_return_lons))


def test_envi_write(synth_sim_roundtrip: EnviRoundTrip) -> None:
    """Checks that the ENVI write of the simulated image succeeded (data and
    companion .hdr file present)."""
    data_base = synth_sim_roundtrip.data_base
    hdr_base = data_base.parent / (data_base.name + ".hdr")
    assert synth_sim_roundtrip.write_ok is True
    assert data_base.is_file() and hdr_base.is_file()


def test_envi_read_back(synth_sim_roundtrip: EnviRoundTrip, synth_simulation: SimulationOutputs) -> None:
    """Checks the ENVI read-back equality of the simulated image."""
    assert synth_sim_roundtrip.read_back is not None and np.allclose(synth_sim_roundtrip.read_back, synth_simulation.sim_image, equal_nan=True), f"read-back shape {synth_sim_roundtrip.read_back.shape if synth_sim_roundtrip.read_back is not None else None}"


def test_plot_classes(synth_simulation: SimulationOutputs, synth_edr: SyntheticEdr, synth_dem: MOLA128) -> None:
    """Runs the track, first-returns, simulation and ground-distance plot
    classes end to end (Agg backend); any exception fails the test."""
    top_point = int(np.min(np.argmin(np.isnan(synth_simulation.ground_distance_min_image).astype(int), axis=0)))
    TrackPlotOnDEM(synth_dem, synth_edr.edr.orbit_data["Lat"], synth_edr.edr.orbit_data["Lon"]).plot()
    TrackAndFirstReturnsPlotOnDEM(synth_dem, synth_edr.edr.orbit_data["Lat"], synth_edr.edr.orbit_data["Lon"],
                                  synth_simulation.first_return_lats, synth_simulation.first_return_lons).plot()
    SimulationPlot(synth_simulation.sim_image, top_point=top_point, bottom_point=top_point+500).plot()
    GroundDistancePlot(synth_simulation.ground_distance_min_image/1000., top_point=top_point, bottom_point=top_point+500).plot()


def test_focusing_output(synth_focused: npt.NDArray[np.float64] | None) -> None:
    """Checks the shape of the focused radargram."""
    focused_rdr = synth_focused
    assert focused_rdr is not None and focused_rdr.shape[1] >= 5, f"foc shape {focused_rdr.shape if focused_rdr is not None else None}"


def test_focusing_has_finite_energy(synth_focused: npt.NDArray[np.float64] | None) -> None:
    """Checks that the focused radargram is all-finite with positive sum."""
    focused_rdr = synth_focused
    assert focused_rdr is not None and np.all(np.isfinite(focused_rdr)) and np.sum(focused_rdr) > 0


def test_focused_envi_roundtrip(synth_focused: npt.NDArray[np.float64] | None, synth_focused_roundtrip: EnviRoundTrip) -> None:
    """Checks the ENVI read-back equality of the focused radargram."""
    focused_rdr = synth_focused
    assert focused_rdr is not None
    assert synth_focused_roundtrip.read_back is not None and np.allclose(synth_focused_roundtrip.read_back, focused_rdr)


def test_png_write(synth_focused_roundtrip: EnviRoundTrip, synth_dir: Path) -> None:
    """Checks the PNG write of the read-back focused radargram."""
    focused_back = synth_focused_roundtrip.read_back
    assert focused_back is not None
    png_filename = synth_dir / "foc_out.png"
    png_ok = png_io.write(str(png_filename), focused_back)
    assert png_ok is True and png_filename.is_file()


def test_radargram_plot(synth_focused: npt.NDArray[np.float64] | None, synth_simulation: SimulationOutputs) -> None:
    """Runs the radargram plot class on the focused product (Agg backend)."""
    focused_rdr = synth_focused
    assert focused_rdr is not None
    top_point = int(np.min(np.argmin(np.isnan(synth_simulation.ground_distance_min_image).astype(int), axis=0)))
    RadargramPlot(focused_rdr, top_point=top_point, bottom_point=top_point+500).plot()
