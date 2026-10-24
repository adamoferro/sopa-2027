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

"""Unit tests of the PDS RDR reader (sopa.input.sharad.data.rdr.pds and its
companion log-message module), on a tiny synthetic dataset: little-endian
fixed-record decoding, complex echo matrix assembly, the per-frame range-shift
alignment by zero padding, orbit-data generation and the rejection paths.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from sopa.input.sharad.data.rdr import pds_log
from sopa.input.sharad.data.rdr.pds import RDR
from sopa.log.log import logger

N_FRAMES = 3
N_STORED_SAMPLES = 667
DT_ADC = 3 / 80e6
DCG_DELAY = 11.98e-6
PRI_US = 1428
RXWIN_STEP = 100000.0
RANGE_SHIFTS = np.array([10, 4, 0], dtype="int16")

RECORD_DTYPE = np.dtype([("SCET_BLOCK_WHOLE", "<u4"),
                         ("SCET_BLOCK_FRAC", "<u2"),
                         ("SKIP1", np.void, 180),
                         ("RECEIVE_WINDOW_OPENING_TIME", "<f4"),
                         ("SKIP2", np.void, 4),
                         ("ECHO_SAMPLES_REAL", "<f4", N_STORED_SAMPLES),
                         ("ECHO_SAMPLES_IMAGINARY", "<f4", N_STORED_SAMPLES),
                         ("SKIP3", np.void, 30),
                         ("RANGE_SHIFT", "<i2"),
                         ("SKIP4", np.void, 31),
                         ("SOLAR_LONGITUDE", "<f8"),
                         ("ORBIT_NUMBER", "<i4"),
                         ("X_MARS_SC_POSITION_VECTOR", "<f8"),
                         ("Y_MARS_SC_POSITION_VECTOR", "<f8"),
                         ("Z_MARS_SC_POSITION_VECTOR", "<f8"),
                         ("SPACECRAFT_ALTITUDE", "<f8"),
                         ("SUB_SC_EAST_LONGITUDE", "<f8"),
                         ("SUB_SC_PLANETOCENTRIC_LATITUDE", "<f8"),
                         ("SUB_SC_PLANETOGRAPHIC_LATITUDE", "<f8"),
                         ("X_MARS_SC_VELOCITY_VECTOR", "<f8"),
                         ("Y_MARS_SC_VELOCITY_VECTOR", "<f8"),
                         ("Z_MARS_SC_VELOCITY_VECTOR", "<f8"),
                         ("MARS_SC_RADIAL_VELOCITY", "<f8"),
                         ("MARS_SC_TANGENTIAL_VELOCITY", "<f8"),
                         ("LOCAL_TRUE_SOLAR_TIME", "<f8"),
                         ("SOLAR_ZENITH_ANGLE", "<f8"),
                         ("SC_PITCH_ANGLE", "<f8"),
                         ("SC_YAW_ANGLE", "<f8"),
                         ("SC_ROLL_ANGLE", "<f8"),
                         ("MRO_SAMX_INNER_GIMBAL_ANGLE", "<f8"),
                         ("MRO_SAMX_OUTER_GIMBAL_ANGLE", "<f8"),
                         ("MRO_SAPX_INNER_GIMBAL_ANGLE", "<f8"),
                         ("MRO_SAPX_OUTER_GIMBAL_ANGLE", "<f8"),
                         ("MRO_HGA_INNER_GIMBAL_ANGLE", "<f8"),
                         ("MRO_HGA_OUTER_GIMBAL_ANGLE", "<f8"),
                         ("DES_TEMP", "<f4"),
                         ("DES_5V", "<f4"),
                         ("DES_12V", "<f4"),
                         ("DES_2V5", "<f4"),
                         ("RX_TEMP", "<f4"),
                         ("TX_TEMP", "<f4"),
                         ("TX_LEV", "<f4"),
                         ("TX_CURR", "<f4"),
                         ("CORRUPTED_DATA_FLAG", "<i1")])


def _expected_complex_column(i_frame: int) -> npt.NDArray[np.complex128]:
    """The complex echo samples written into record i_frame."""
    base = (np.arange(N_STORED_SAMPLES, dtype="float64") - 333.0) * 0.5 + i_frame
    return base + 1j * (-base)


def _write_dataset(directory: Path, pri_text: str = f"{PRI_US} <MICROSECOND>", quality_line: str = 'DATA_QUALITY_ID = "0"') -> None:
    """Writes the synthetic RDR label and record file."""
    lines = [
        'PDS_VERSION_ID                 = "PDS3"',
        "START_TIME                     = 2010-01-01T00:00:00",
        f"FILE_RECORDS                   = {N_FRAMES}",
        "INSTRUMENT_MODE_ID             = SS3",
        f"MRO:PULSE_REPETITION_INTERVAL  = {pri_text}",
    ]
    if quality_line != "":
        lines.append(quality_line)
    lines.append("END")
    (directory / "r_test.lbl").write_text("\n".join(lines) + "\n")

    records = np.zeros(N_FRAMES, dtype=RECORD_DTYPE)
    for i in range(N_FRAMES):
        records[i]["SCET_BLOCK_WHOLE"] = 1000 + i
        records[i]["RECEIVE_WINDOW_OPENING_TIME"] = RXWIN_STEP + i
        records[i]["ECHO_SAMPLES_REAL"] = _expected_complex_column(i).real.astype("float32")
        records[i]["ECHO_SAMPLES_IMAGINARY"] = _expected_complex_column(i).imag.astype("float32")
        records[i]["RANGE_SHIFT"] = RANGE_SHIFTS[i]
        records[i]["X_MARS_SC_POSITION_VECTOR"] = 1000.0 + i
        records[i]["Y_MARS_SC_POSITION_VECTOR"] = 2000.0 + i
        records[i]["Z_MARS_SC_POSITION_VECTOR"] = 3000.0 + i
        records[i]["SUB_SC_EAST_LONGITUDE"] = 2.0 + i
        records[i]["SUB_SC_PLANETOCENTRIC_LATITUDE"] = 1.0 + i
        records[i]["MARS_SC_TANGENTIAL_VELOCITY"] = 3200.0
    (directory / "r_test.dat").write_bytes(records.tobytes())


def _make_rdr(directory: Path) -> tuple[RDR, logger]:
    """Builds a PDS RDR reader with an in-memory logger."""
    lg = logger(verbose=False)
    return RDR(dataset_name="test", filename_base=str(directory / "r_test"), logger=lg), lg


def test_log_message_tables_complete() -> None:
    """Checks that the reader's log-message module exposes the message keys
    used by the reader, each mapped to a non-empty string."""
    for key in ["header_file_generic", "header_file_read", "header_file_decode", "header_file_no_ss",
                "other_file_not-found", "discarded", "data_frame-limits", "data_read"]:
        assert key in pds_log.ERR_MESSAGES
        assert isinstance(pds_log.ERR_MESSAGES[key], str) and pds_log.ERR_MESSAGES[key] != ""


def test_load_full(tmp_path: Path) -> None:
    """Checks a successful full load: header fields and the focused data type."""
    _write_dataset(tmp_path)
    d, _ = _make_rdr(tmp_path)
    assert d.load() is True
    assert d.n_frames == N_FRAMES
    assert d.pri == pytest.approx(PRI_US * 1e-6, abs=1e-15)
    assert d.presumming == 16
    assert d._data_type == "focused"


def test_range_shift_alignment(tmp_path: Path) -> None:
    """Checks the range-shift alignment: the frame with the maximum range
    shift stays at zero offset, the others are shifted by half their deficit
    (in ADC steps), and the output is zero-padded to the tallest column."""
    _write_dataset(tmp_path)
    d, _ = _make_rdr(tmp_path)
    assert d.load() is True
    data = d.data
    assert data is not None
    frame_shifts = np.round((RANGE_SHIFTS.max() - RANGE_SHIFTS) / 2).astype("int")
    assert data.shape == (N_STORED_SAMPLES + frame_shifts.max(), N_FRAMES)
    assert data.dtype == np.dtype("complex64")     # float32 + 1j*float32 stays complex64 under NEP 50
    for i in range(N_FRAMES):
        s = frame_shifts[i]
        np.testing.assert_array_equal(data[:s, i], np.zeros(s, dtype="complex128"))
        np.testing.assert_allclose(data[s:s + N_STORED_SAMPLES, i], _expected_complex_column(i))
        np.testing.assert_array_equal(data[s + N_STORED_SAMPLES:, i], np.zeros(data.shape[0] - s - N_STORED_SAMPLES, dtype="complex128"))


def test_correct_range_shift_static() -> None:
    """Checks the alignment helper directly on a small case: shifts of (2, 0)
    move the second frame down by one sample."""
    data = np.array([[1.0 + 0j, 10.0 + 0j],
                     [2.0 + 0j, 20.0 + 0j],
                     [3.0 + 0j, 30.0 + 0j],
                     [4.0 + 0j, 40.0 + 0j]])
    aligned = RDR._correct_range_shift(data, np.array([2, 0], dtype="int16"))
    assert aligned.shape == (5, 2)
    assert aligned[0, 0] == 1.0 + 0j and aligned[3, 0] == 4.0 + 0j
    assert aligned[0, 1] == 0.0 + 0j and aligned[1, 1] == 10.0 + 0j and aligned[4, 1] == 40.0 + 0j


def test_generate_orbit_data(tmp_path: Path) -> None:
    """Checks orbit-data generation from the record ancillary fields: decoded
    lat/lon, positions and radius in meters, and the receive-window time."""
    _write_dataset(tmp_path)
    d, _ = _make_rdr(tmp_path)
    assert d.load() is True
    d.generate_orbit_data()
    orbit = d.orbit_data
    assert orbit is not None
    np.testing.assert_allclose(orbit["Lat"], np.arange(1.0, 1.0 + N_FRAMES))
    np.testing.assert_allclose(orbit["Lon"], np.arange(2.0, 2.0 + N_FRAMES))
    xyz_km = np.stack([np.arange(1000.0, 1000.0 + N_FRAMES),
                       np.arange(2000.0, 2000.0 + N_FRAMES),
                       np.arange(3000.0, 3000.0 + N_FRAMES)])
    expected_radius = np.linalg.norm(xyz_km, axis=0) * 1000.0
    np.testing.assert_allclose(orbit["Radius"], expected_radius, rtol=1e-12)
    np.testing.assert_allclose(orbit["Vtang"], np.full(N_FRAMES, 3200.0))
    expected_rxwin = (RXWIN_STEP + np.arange(N_FRAMES)) * DT_ADC - DCG_DELAY + PRI_US * 1e-6
    np.testing.assert_allclose(orbit["rxwin_time_s"], expected_rxwin)
    assert orbit["dt"] == DT_ADC * 2.0


def test_header_bad_pri_rejected(tmp_path: Path) -> None:
    """Checks that a non-compliant PRI value discards the dataset."""
    _write_dataset(tmp_path, pri_text="716 <MICROSECOND>")
    d, lg = _make_rdr(tmp_path)
    assert d.load() is False
    assert any("discarded" in m and "PULSE_REPETITION_INTERVAL" in m for m in lg.LOG_MESSAGES)


def test_header_missing_quality_id_rejected(tmp_path: Path) -> None:
    """Checks that a label lacking DATA_QUALITY_ID entirely fails the parse."""
    _write_dataset(tmp_path, quality_line="")
    d, lg = _make_rdr(tmp_path)
    assert d.load() is False
    assert any("header file" in m for m in lg.LOG_MESSAGES)


def test_missing_files_rejected(tmp_path: Path) -> None:
    """Checks that a base filename with no files behind it reports the error
    and returns False."""
    d, lg = _make_rdr(tmp_path)
    assert d.load() is False
    assert any("not present" in m for m in lg.LOG_MESSAGES)
