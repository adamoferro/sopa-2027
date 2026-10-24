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

"""Unit tests of the PDS EDR reader (sopa.input.sharad.data.edr.pds and its
companion log-message module), on a tiny synthetic dataset: label parsing and
validation, big-endian ancillary record decoding, orbit-data generation, 4-bit
unpacking with static decompression, receive-window timing, acquisition-mode
decoding and the rejection paths.
"""

from pathlib import Path

import bitstring as bstr
import numpy as np
import numpy.typing as npt
import pytest

from sopa.geometry.coordinate_converter import MARS_IAU2000
from sopa.input.sharad.data.edr import pds_log
from sopa.input.sharad.data.edr.pds import EDR
from sopa.log.log import logger

N_FRAMES = 5
PRI_US = 1428
PRESUMMING = 16
N_BITS_PER_BYTE = 4
N_BYTES_PER_RECORD = 1800
N_SAMPLES = 3600
DT = 3 / 80e6
DCG_DELAY = 11.98e-6
RXWIN_STEP_0 = 100000.0

GEOM_DTYPE = np.dtype(
    [
        ("SCET_BLOCK_WHOLE", ">u4"),
        ("SCET_BLOCK_FRAC", ">u2"),
        ("EPHEMERIS_TIME", ">f8"),
        ("GEOMETRY_EPOCH", np.void, 23),
        ("SOLAR_LONGITUDE", ">f8"),
        ("ORBIT_NUMBER", ">i4"),
        ("X_MARS_SC_POSITION_VECTOR", ">f8"),
        ("Y_MARS_SC_POSITION_VECTOR", ">f8"),
        ("Z_MARS_SC_POSITION_VECTOR", ">f8"),
        ("SPACECRAFT_ALTITUDE", ">f8"),
        ("SUB_SC_EAST_LONGITUDE", ">f8"),
        ("SUB_SC_PLANETOCENTRIC_LATITUDE", ">f8"),
        ("SUB_SC_PLANETOGRAPHIC_LATITUDE", ">f8"),
        ("X_MARS_SC_VELOCITY_VECTOR", ">f8"),
        ("Y_MARS_SC_VELOCITY_VECTOR", ">f8"),
        ("Z_MARS_SC_VELOCITY_VECTOR", ">f8"),
        ("MARS_SC_RADIAL_VELOCITY", ">f8"),
        ("MARS_SC_TANGENTIAL_VELOCITY", ">f8"),
        ("LOCAL_TRUE_SOLAR_TIME", ">f8"),
        ("SOLAR_ZENITH_ANGLE", ">f8"),
        ("SC_PITCH_ANGLE", ">f8"),
        ("SC_YAW_ANGLE", ">f8"),
        ("SC_ROLL_ANGLE", ">f8"),
        ("MRO_SAMX_INNER_GIMBAL_ANGLE", ">f8"),
        ("MRO_SAMX_OUTER_GIMBAL_ANGLE", ">f8"),
        ("MRO_SAPX_INNER_GIMBAL_ANGLE", ">f8"),
        ("MRO_SAPX_OUTER_GIMBAL_ANGLE", ">f8"),
        ("MRO_HGA_INNER_GIMBAL_ANGLE", ">f8"),
        ("MRO_HGA_OUTER_GIMBAL_ANGLE", ">f8"),
        ("DES_TEMP", ">f4"),
        ("DES_5V", ">f4"),
        ("DES_12V", ">f4"),
        ("DES_2V5", ">f4"),
        ("RX_TEMP", ">f4"),
        ("TX_TEMP", ">f4"),
        ("TX_LEV", ">f4"),
        ("TX_CURR", ">f4"),
        ("CORRUPTED_DATA_FLAG", ">i2"),
    ]
)

LAT = np.linspace(0.9, 1.1, N_FRAMES)
LON = np.linspace(0.5, 0.7, N_FRAMES)
RADIUS = np.full(N_FRAMES, 3646190.0)


def _write_label(
    directory: Path,
    pri_text: str = f"{PRI_US} <MICROSECOND>",
    im_text: str = "SS3",
    quality_line: str = 'DATA_QUALITY_ID                = "0"',
    n_frames: int = N_FRAMES,
) -> Path:
    """Writes a PDS-like label with the fields in the order searched by the
    reader, allowing single-field overrides for the rejection tests."""
    lines = [
        'PDS_VERSION_ID                 = "PDS3"',
        "START_TIME                     = 2010-01-01T00:00:00",
        f"FILE_RECORDS                   = {n_frames}",
        f"INSTRUMENT_MODE_ID             = {im_text}",
        f"MRO:PULSE_REPETITION_INTERVAL  = {pri_text}",
        'MRO:PHASE_COMPENSATION_TYPE    = "NO COMPENSATION"',
        'MRO:COMPRESSION_SELECTION_FLAG = "STATIC"',
    ]
    if quality_line != "":
        lines.append(quality_line)
    lines.append("END")
    header = directory / "e_test_a.lbl"
    header.write_text("\n".join(lines) + "\n")
    return header


def _write_ancillary(directory: Path, n_frames: int = N_FRAMES) -> None:
    """Writes the binary orbit ancillary file for the synthetic track
    (positions stored in km, as in the real PDS products)."""
    cc = MARS_IAU2000()
    x, y, z = cc.to_xyz_from_latlon_and_radius(LAT, LON, RADIUS)
    geom = np.zeros(n_frames, dtype=GEOM_DTYPE)
    geom["SCET_BLOCK_WHOLE"] = np.arange(n_frames) * 100
    geom["EPHEMERIS_TIME"] = np.arange(n_frames) * (PRI_US * PRESUMMING * 1e-6)
    geom["X_MARS_SC_POSITION_VECTOR"] = x / 1000.0
    geom["Y_MARS_SC_POSITION_VECTOR"] = y / 1000.0
    geom["Z_MARS_SC_POSITION_VECTOR"] = z / 1000.0
    geom["SUB_SC_EAST_LONGITUDE"] = LON
    geom["SUB_SC_PLANETOCENTRIC_LATITUDE"] = LAT
    geom["MARS_SC_TANGENTIAL_VELOCITY"] = 3200.0
    geom["SOLAR_ZENITH_ANGLE"] = 45.0
    with open(directory / "e_test_a_a.dat", "wb") as fp:
        fp.write(geom.tobytes())


def _write_science(directory: Path) -> npt.NDArray[np.int64]:
    """Writes the 4-bit compressed science file with known sample values and
    per-frame receive-window opening times; returns the packed values."""
    rng = np.random.default_rng(7)
    true_ints = rng.integers(-8, 8, size=(N_SAMPLES, N_FRAMES)).astype("int")
    with open(directory / "e_test_a_s.dat", "wb") as fp:
        for i_f in range(N_FRAMES):
            fp.write(b"\x00" * 178)
            fp.write(np.array([RXWIN_STEP_0 + i_f], dtype=">f4").tobytes())
            fp.write(b"\x00" * 4)
            ba = bstr.BitArray()
            for v in true_ints[:, i_f]:
                ba.append(bstr.BitArray(int=int(v), length=4))
            fp.write(ba.tobytes())
    return true_ints


def _make_edr(directory: Path) -> tuple[EDR, logger]:
    """Builds a PDS EDR reader with an in-memory logger on the synthetic base."""
    lg = logger(verbose=False)
    return EDR(
        dataset_name="test", filename_base=str(directory / "e_test_a"), logger=lg
    ), lg


@pytest.fixture()
def dataset(tmp_path: Path) -> tuple[Path, npt.NDArray[np.int64]]:
    """Writes the complete synthetic dataset (label, ancillary, science)."""
    _write_label(tmp_path)
    _write_ancillary(tmp_path)
    true_ints = _write_science(tmp_path)
    return tmp_path, true_ints


def test_log_message_tables_complete() -> None:
    """Checks that the reader's log-message module exposes the message keys
    used by the reader, each mapped to a non-empty string."""
    for key in [
        "header_file_generic",
        "header_file_read",
        "header_file_decode",
        "header_file_no_ss",
        "other_file_not-found",
        "discarded",
        "data_frame-limits",
        "data_read",
    ]:
        assert key in pds_log.ERR_MESSAGES
        assert (
            isinstance(pds_log.ERR_MESSAGES[key], str)
            and pds_log.ERR_MESSAGES[key] != ""
        )


def test_decode_im_table() -> None:
    """Checks the acquisition-mode decoding: presumming, bits per sample and
    record bytes for one mode of each bit depth (4-bit: 3, 6-bit: 11, 8-bit: 19)."""
    d, _ = _make_edr(Path("/tmp"))
    assert d._decode_im(3) == (16, 4, 1800)
    assert d._decode_im(11) == (8, 6, 2700)
    assert d._decode_im(19) == (4, 8, 3600)


def test_load_full(dataset: tuple[Path, npt.NDArray[np.int64]]) -> None:
    """Checks a successful full load: header fields, ancillary and data."""
    directory, _ = dataset
    d, _ = _make_edr(directory)
    assert d.load(mode="full") is True
    assert d.n_frames == N_FRAMES
    assert d.pri == pytest.approx(PRI_US * 1e-6, abs=1e-15)
    assert d.presumming == PRESUMMING
    assert d._n_bits_per_byte == N_BITS_PER_BYTE
    assert d._n_bytes_per_record == N_BYTES_PER_RECORD


def test_4bit_unpack_and_decompression(
    dataset: tuple[Path, npt.NDArray[np.int64]],
) -> None:
    """Checks the 4-bit unpacking plus static decompression against the known
    sample values (factorL = ceil(log2 16) = 4, factorS = 8)."""
    directory, true_ints = dataset
    d, _ = _make_edr(directory)
    assert d.load(mode="full") is True
    expected = true_ints * 2**8 / PRESUMMING
    assert d.data is not None
    assert d.data.shape == (N_SAMPLES, N_FRAMES)
    assert d.data.dtype == np.dtype("float64")
    np.testing.assert_allclose(d.data, expected)


def test_orbit_data_fields(dataset: tuple[Path, npt.NDArray[np.int64]]) -> None:
    """Checks the generated orbit data: decoded lat/lon, cartesian positions
    in meters and the radius as the position-vector norm."""
    directory, _ = dataset
    d, _ = _make_edr(directory)
    assert d.load(mode="full") is True
    d.generate_orbit_data()
    orbit = d.orbit_data
    assert orbit is not None
    np.testing.assert_allclose(orbit["Lat"], LAT)
    np.testing.assert_allclose(orbit["Lon"], LON)
    cc = MARS_IAU2000()
    x, y, z = cc.to_xyz_from_latlon_and_radius(LAT, LON, RADIUS)
    np.testing.assert_allclose(orbit["X"], x, rtol=1e-12)
    np.testing.assert_allclose(orbit["Y"], y, rtol=1e-12)
    np.testing.assert_allclose(orbit["Z"], z, rtol=1e-12)
    np.testing.assert_allclose(
        orbit["Radius"], np.sqrt(x * x + y * y + z * z), rtol=1e-12
    )
    assert orbit["dt"] == DT


def test_rxwin_time(dataset: tuple[Path, npt.NDArray[np.int64]]) -> None:
    """Checks the receive-window opening time conversion: steps in DT units,
    minus the DCG delay, plus the PRI (1428 us is below the 1500 us switch)."""
    directory, _ = dataset
    d, _ = _make_edr(directory)
    assert d.load(mode="full") is True
    d.generate_orbit_data()
    orbit = d.orbit_data
    assert orbit is not None
    expected = (RXWIN_STEP_0 + np.arange(N_FRAMES)) * DT - DCG_DELAY + PRI_US * 1e-6
    np.testing.assert_allclose(orbit["rxwin_time_s"], expected)


def test_ancillary_only_then_data(
    dataset: tuple[Path, npt.NDArray[np.int64]], capsys: pytest.CaptureFixture[str]
) -> None:
    """Checks the staged load: ancillary-only leaves the data unread and the
    orbit data with zero receive-window times (with a warning); a subsequent
    data load fills them in."""
    directory, _ = dataset
    d, _ = _make_edr(directory)
    assert d.load(mode="ancillary") is True
    assert d.data is None
    assert "ERROR" in capsys.readouterr().out
    d.generate_orbit_data()
    assert "WARNING" in capsys.readouterr().out
    orbit = d.orbit_data
    assert orbit is not None
    np.testing.assert_array_equal(orbit["rxwin_time_s"], np.zeros(N_FRAMES))
    assert d.load(mode="data") is True
    d.generate_orbit_data()
    orbit = d.orbit_data
    assert orbit is not None
    expected = (RXWIN_STEP_0 + np.arange(N_FRAMES)) * DT - DCG_DELAY + PRI_US * 1e-6
    np.testing.assert_allclose(orbit["rxwin_time_s"], expected)


def test_header_bad_pri_rejected(tmp_path: Path) -> None:
    """Checks that a non-compliant PRI value discards the dataset."""
    _write_label(tmp_path, pri_text="716 <MICROSECOND>")
    _write_ancillary(tmp_path)
    d, lg = _make_edr(tmp_path)
    assert d.load(mode="ancillary") is False
    assert any(
        "discarded" in m and "PULSE_REPETITION_INTERVAL" in m for m in lg.LOG_MESSAGES
    )


def test_header_bad_quality_id_rejected(tmp_path: Path) -> None:
    """Checks that a non-zero DATA_QUALITY_ID discards the dataset."""
    _write_label(tmp_path, quality_line='DATA_QUALITY_ID                = "1"')
    _write_ancillary(tmp_path)
    d, lg = _make_edr(tmp_path)
    assert d.load(mode="ancillary") is False
    assert any("discarded" in m and "DATA_QUALITY_ID" in m for m in lg.LOG_MESSAGES)


def test_header_missing_quality_id_rejected(tmp_path: Path) -> None:
    """Checks that a label lacking DATA_QUALITY_ID entirely fails the parse."""
    _write_label(tmp_path, quality_line="")
    _write_ancillary(tmp_path)
    d, lg = _make_edr(tmp_path)
    assert d.load(mode="ancillary") is False
    assert any("header file" in m for m in lg.LOG_MESSAGES)


def test_header_non_ss_mode_rejected(tmp_path: Path) -> None:
    """Checks that a non-SS instrument mode discards the dataset."""
    _write_label(tmp_path, im_text="LFO")
    _write_ancillary(tmp_path)
    d, lg = _make_edr(tmp_path)
    assert d.load(mode="ancillary") is False
    assert any("not SS" in m for m in lg.LOG_MESSAGES)


def test_missing_files_rejected(tmp_path: Path) -> None:
    """Checks that a base filename with no files behind it reports the error
    and returns False."""
    d, lg = _make_edr(tmp_path)
    assert d.load(mode="full") is False
    assert any("not present" in m for m in lg.LOG_MESSAGES)


def test_frame_limits_rejected(dataset: tuple[Path, npt.NDArray[np.int64]]) -> None:
    """Checks that invalid frame limits are reported and return None."""
    directory, _ = dataset
    d, lg = _make_edr(directory)
    assert d.load(mode="full") is True
    assert d._load_raw_data(start_frame=-1) is None
    assert any("frame" in m for m in lg.LOG_MESSAGES)
