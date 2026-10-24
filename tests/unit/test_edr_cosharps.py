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

"""Unit tests of the CO-SHARPS EDR reader (sopa.input.sharad.data.edr.cosharps
and its companion log-message module), on a tiny synthetic dataset: tab-
separated header parsing and validation, aux-file consistency, orbit-table
reading with cubic interpolation to the frame times, 8-bit raw data loading,
housekeeping telemetry parsing, and the hole-filling helper.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sopa.input.sharad.data.edr import cosharps_log
from sopa.input.sharad.data.edr.cosharps import EDR
from sopa.log.log import logger

N_FRAMES = 5
N_SAMPLES = 3600

HEADER_FIELDS = [
    ("Number_of_Modes", "1"),
    ("Oper_Mode", "SS"),
    ("PRI_Value", "1428"),
    ("Presumming", "16"),
    ("Resolution", "150"),
    ("Phase_Comp", "None"),
    ("Compression", "Static"),
    ("Tracking", "Disabled"),
    ("Data_Blocks", str(N_FRAMES)),
    ("Lost_Packets", "0"),
    ("Wrong_Packets", "0"),
    ("Anomalies", "0"),
]

AUX_TIME_OFFSETS = np.arange(N_FRAMES, dtype="float64")
AUX_RXWIN_US = 1000.0 + 10.0 * np.arange(N_FRAMES)

GEOM_LAT = np.linspace(0.9, 1.1, N_FRAMES)
GEOM_LON = np.linspace(10.0, 10.4, N_FRAMES)
GEOM_RADIUS_KM = 3396.0 + 0.001 * np.arange(N_FRAMES)
GEOM_X_KM = 1000.0 + np.arange(N_FRAMES)
GEOM_Y_KM = 2000.0 + np.arange(N_FRAMES)
GEOM_Z_KM = 3000.0 + np.arange(N_FRAMES)

DCG_DELAY = 11.98e-6


def _write_dataset(directory: Path, header_fields: list[tuple[str, str]] | None = None, n_aux_rows: int = N_FRAMES) -> None:
    """Writes a complete CO-SHARPS dataset (header, aux, HK, orbit, mode data);
    single fields can be overridden for the rejection tests."""
    fields = header_fields if header_fields is not None else HEADER_FIELDS
    (directory / "e_test_Header.txt").write_text("\n".join(f"{f}\t{v}" for f, v in fields) + "\n")
    aux_lines = [f"{float(i):.1f},{1000.0 + 10.0 * float(i):.1f},0" for i in range(n_aux_rows)]
    (directory / "e_test_Aux_001.csv").write_text("\n".join(aux_lines) + "\n")
    (directory / "e_test_HK.dat").write_text("# no telemetry here\n")
    geom_header = "\n".join(f"# orbit header line {i}" for i in range(7))
    timestamp = np.datetime64("2026-01-01T00:00:00.000") + np.arange(N_FRAMES) * np.timedelta64(1000, "ms")
    rows = [
        f"{t}\t"
        f"{GEOM_LAT[i]}\t{GEOM_LON[i]}\t{GEOM_RADIUS_KM[i]}\t{3200.0}\t{0.0}\t"
        f"{GEOM_X_KM[i]}\t{GEOM_Y_KM[i]}\t{GEOM_Z_KM[i]}\t{10.0}\t{20.0}\t{30.0}\t"
        f"{1.0}\t{2.0}\t{3.0}\t{4.0}\t{5.0}\t{6.0}\t{7.0}\t{8.0}\t{9.0}\t{45.0}\t{10.0}\t{11.0}"
        for i, t in enumerate(timestamp)
    ]
    (directory / "e_test_Orbit_1.txt").write_text(geom_header + "\n" + "\n".join(rows) + "\n")
    data = ((np.arange(N_FRAMES * N_SAMPLES) % 128) - 64).astype("int8")
    (directory / "e_test_Mode_001.dat").write_bytes(data.tobytes())


def _make_edr(directory: Path, **kwargs: Any) -> tuple[EDR, logger]:
    """Builds a CO-SHARPS EDR reader with an in-memory logger."""
    lg = logger(verbose=False)
    return EDR(dataset_name="test", filename_base=str(directory / "e_test"), logger=lg, **kwargs), lg


def _hk_line(seconds_hex: str, frac_hex: str, payload_hex: str) -> str:
    """Builds one housekeeping telemetry line in the layout parsed by the
    reader (label, seconds, fraction, filler, engineering payload)."""
    return "ENG:7EE5" + seconds_hex + frac_hex + "0123456789ABCDEF" + payload_hex


def test_log_message_tables_complete() -> None:
    """Checks that the reader's log-message module exposes the message keys
    used by the reader, each mapped to a non-empty string."""
    for key in ["header_file_generic", "header_file_read", "header_file_decode", "aux_file_read",
                "aux_file_generic", "orbit_file_not-found", "orbit_file_read", "orbit_file_generic",
                "other_file_not-found", "discarded", "no-consistency", "data_frame-limits", "data_read",
                "hk-data_read", "hk_file_generic"]:
        assert key in cosharps_log.ERR_MESSAGES
        assert isinstance(cosharps_log.ERR_MESSAGES[key], str) and cosharps_log.ERR_MESSAGES[key] != ""


def test_load_full(tmp_path: Path) -> None:
    """Checks a successful full load: header fields, aux data and 8-bit raw
    data of the first acquisition mode."""
    _write_dataset(tmp_path)
    d, _ = _make_edr(tmp_path)
    assert d.load(mode="full") is True
    assert d.n_frames == N_FRAMES
    assert d.pri == pytest.approx(1428e-6)
    assert d.presumming == 16
    data = d.data
    assert data is not None
    assert data.shape == (N_SAMPLES, N_FRAMES)
    assert data.dtype == np.dtype("float64")
    expected = ((np.arange(N_FRAMES * N_SAMPLES) % 128) - 64).astype("float64").reshape(N_FRAMES, N_SAMPLES).T
    np.testing.assert_array_equal(data, expected)


def test_generate_orbit_data(tmp_path: Path) -> None:
    """Checks orbit-data generation: frame time offsets from the aux file,
    the receive-window conversion minus the DCG delay, and the cubic
    interpolation of the geometric fields at the frame times (exact at the
    orbit-table nodes, which coincide here)."""
    _write_dataset(tmp_path)
    d, _ = _make_edr(tmp_path)
    assert d.load(mode="full") is True
    d.generate_orbit_data()
    orbit = d.orbit_data
    assert orbit is not None
    np.testing.assert_allclose(orbit["time_offset_s"], AUX_TIME_OFFSETS)
    np.testing.assert_allclose(orbit["rxwin_time_s"], AUX_RXWIN_US / 1e6 - DCG_DELAY)
    np.testing.assert_allclose(orbit["Lat"], GEOM_LAT)
    np.testing.assert_allclose(orbit["Lon"], GEOM_LON)
    np.testing.assert_allclose(orbit["Radius"], GEOM_RADIUS_KM * 1000.0)
    np.testing.assert_allclose(orbit["X"], GEOM_X_KM * 1000.0)
    assert orbit["dt"] == 3 / 80e6


def test_aux_consistency_failure(tmp_path: Path) -> None:
    """Checks that a frame-count mismatch between header and aux file stops
    the ancillary load."""
    _write_dataset(tmp_path, n_aux_rows=N_FRAMES + 1)
    d, lg = _make_edr(tmp_path)
    assert d.load(mode="ancillary") is False
    assert any("consistency" in m for m in lg.LOG_MESSAGES)
    assert d.orbit_data is None


def test_header_constraint_failure(tmp_path: Path) -> None:
    """Checks that a non-zero Anomalies count discards the dataset."""
    fields = [(f, "1" if f == "Anomalies" else v) for f, v in HEADER_FIELDS]
    _write_dataset(tmp_path, header_fields=fields)
    d, lg = _make_edr(tmp_path)
    assert d.load(mode="full") is False
    assert any("discarded" in m and "Anomalies" in m for m in lg.LOG_MESSAGES)


def test_header_wrong_compression_rejected(tmp_path: Path) -> None:
    """Checks that a non-static compression flag discards the dataset."""
    fields = [(f, "Dynamic" if f == "Compression" else v) for f, v in HEADER_FIELDS]
    _write_dataset(tmp_path, header_fields=fields)
    d, lg = _make_edr(tmp_path)
    assert d.load(mode="full") is False
    assert any("discarded" in m and "Compression" in m for m in lg.LOG_MESSAGES)


def test_missing_files_rejected(tmp_path: Path) -> None:
    """Checks that a base filename with no files behind it reports the error
    and returns False."""
    d, lg = _make_edr(tmp_path)
    assert d.load(mode="full") is False
    assert any("not present" in m for m in lg.LOG_MESSAGES)


def test_hk_data_parsing(tmp_path: Path) -> None:
    """Checks the housekeeping telemetry parsing: lines carrying the
    engineering label are decoded into timestamps (reference epoch plus the
    hex seconds/fraction) and the eight payload bytes."""
    _write_dataset(tmp_path)
    (tmp_path / "e_test_HK.dat").write_text(
        "# comment\n" + _hk_line("0000000A", "8000", "0102030405060708") + "\n"
        + "noise line without label\n" + _hk_line("00000014", "0000", "0807060504030201") + "\n"
    )
    d, _ = _make_edr(tmp_path, read_hk_data=True)
    assert d.load(mode="ancillary") is True
    hk = d._hk_data
    assert hk is not None
    assert hk["timestamp_ms"].shape == (2,)
    assert hk["timestamp_ms"][0] == pytest.approx((315532800 + 10.5) * 1000.0)
    assert hk["timestamp_ms"][1] == pytest.approx((315532800 + 20.0) * 1000.0)
    for key, first, second in [("des_temp", 1, 8), ("des_5v", 2, 7), ("des_12v", 3, 6), ("des_2v5", 4, 5),
                               ("rx_temp", 5, 4), ("tx_temp", 6, 3), ("tx_lev", 7, 2), ("tx_curr", 8, 1)]:
        assert hk[key][0] == first and hk[key][1] == second


def test_hk_data_empty_telemetry_accepted(tmp_path: Path) -> None:
    """Checks that with housekeeping reading enabled, an HK file without any
    telemetry line yields empty arrays without failing the ancillary load."""
    _write_dataset(tmp_path)
    d, _ = _make_edr(tmp_path, read_hk_data=True)
    assert d.load(mode="ancillary") is True
    assert d._hk_data is not None
    assert d._hk_data["timestamp_ms"].shape == (0,)


def test_fill_data_holes() -> None:
    """Checks the nearest-neighbour hole filling on a one-dimensional case."""
    data = np.array([1.0, 0.0, 0.0, 4.0])
    filled = EDR._fill_data_holes(data)
    np.testing.assert_allclose(filled, [1.0, 1.0, 4.0, 4.0])


def test_fill_data_holes_keeps_valid_values() -> None:
    """Checks that filling leaves values above the hole threshold untouched."""
    data = np.array([5.0, 6.0, 0.0, 0.0, 2.0])
    filled = EDR._fill_data_holes(data)
    np.testing.assert_allclose(filled, [5.0, 6.0, 6.0, 2.0, 2.0])
