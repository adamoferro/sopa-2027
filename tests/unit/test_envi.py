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

"""Unit tests of the ENVI reader/writer (sopa.io_utils.envi): dtype
roundtrips, the complex-to-power conversion, the header text, the header
parser (including multi-line values) and the reader's error and offset/band
handling.
"""

from pathlib import Path

import numpy as np
import pytest

from sopa.io_utils import envi


@pytest.mark.parametrize(
    "dtype_name",
    ["uint8", "int16", "float32", "float64", "uint16", "uint32", "int64", "uint64"],
)
def test_roundtrip_dtypes(tmp_path: Path, dtype_name: str) -> None:
    """Checks that write/read preserves values and dtype for the real dtypes
    with a well-defined ENVI type code."""
    rng = np.random.default_rng(5)
    if dtype_name.startswith("u") or dtype_name == "int16":
        data = rng.integers(0, 1000, size=(7, 5)).astype(dtype_name)
    else:
        data = rng.standard_normal((7, 5)).astype(dtype_name)
    base = tmp_path / "rt"
    assert envi.write(str(base), data) is True
    read_back = envi.read(str(base))
    assert read_back is not None
    assert read_back.dtype == np.dtype(dtype_name)
    np.testing.assert_array_equal(read_back, data, strict=True)


def test_roundtrip_complex(tmp_path: Path) -> None:
    """Checks the complex64/complex128 roundtrip with complex_output set."""
    rng = np.random.default_rng(6)
    data = (rng.standard_normal((4, 3)) + 1j * rng.standard_normal((4, 3))).astype(
        "complex64"
    )
    base = tmp_path / "rtc"
    assert envi.write(str(base), data, complex_output=True) is True
    read_back = envi.read(str(base))
    assert read_back is not None
    assert read_back.dtype == np.dtype("complex64")
    np.testing.assert_array_equal(read_back, data, strict=True)


def test_write_complex_converts_to_power(tmp_path: Path) -> None:
    """Checks that complex input is stored as power (|z|^2, float64) by default."""
    data = np.array([[1 + 1j, 0 - 2j]], dtype="complex64")
    base = tmp_path / "pw"
    assert envi.write(str(base), data) is True
    read_back = envi.read(str(base))
    assert read_back is not None
    assert read_back.dtype == np.dtype("float64")
    np.testing.assert_allclose(read_back, np.array([[2.0, 4.0]]))


def test_header_contents(tmp_path: Path) -> None:
    """Checks the .hdr text written for a float32 image."""
    base = tmp_path / "hdr"
    envi.write(str(base), np.zeros((6, 4), dtype="float32"))
    text = (Path(str(base) + ".hdr")).read_text()
    lines = text.splitlines()
    assert lines[0] == "ENVI"
    assert "samples = 4" in lines
    assert "lines   = 6" in lines
    assert "bands   = 1" in lines
    assert "data type = 4" in lines
    assert "interleave = bsq" in lines
    assert "byte order = 0" in lines


def test_int32_written_with_float32_type_code(tmp_path: Path) -> None:
    """Documents the current int32 behavior: the writer stores int32 with the
    ENVI type code 4 (32-bit float, shared with float32), so a read returns a
    float32 array whose bit pattern equals the original int32 values
    (inherited 2026-port behavior; flagged for the technical-debt page)."""
    data = np.array([[1, 2], [3, 4]], dtype="int32")
    base = tmp_path / "i32"
    assert envi.write(str(base), data) is True
    assert "data type = 4" in (Path(str(base) + ".hdr")).read_text().splitlines()
    read_back = envi.read(str(base))
    assert read_back is not None
    assert read_back.dtype == np.dtype("float32")
    np.testing.assert_array_equal(read_back.view("int32"), data)


def test_read_missing_header(tmp_path: Path) -> None:
    """Checks that reading without a .hdr file returns None."""
    assert envi.read(str(tmp_path / "missing")) is None


def test_read_incomplete_header(tmp_path: Path) -> None:
    """Checks that a .hdr without the data-type key makes the read fail
    gracefully with None (the KeyError fires before the data file is opened,
    so a present data file isolates the header path)."""
    base = tmp_path / "bad"
    base.write_bytes(b"\x00" * 16)
    (tmp_path / "bad.hdr").write_text("ENVI\nsamples = 2\nlines = 2\n")
    assert envi.read(str(base)) is None


def test_read_multiband_bsq_returns_first_band(tmp_path: Path) -> None:
    """Checks that a two-band BSQ file returns only the first band."""
    base = tmp_path / "mb"
    band1 = np.arange(12, dtype="<i2").reshape(3, 4)
    band2 = (np.arange(12, dtype="<i2") + 100).reshape(3, 4)
    with open(base, "wb") as fp:
        band1.tofile(fp)
        band2.tofile(fp)
    (tmp_path / "mb.hdr").write_text(
        "ENVI\nsamples = 4\nlines = 3\nbands = 2\ndata type = 2\ninterleave = bsq\nbyte order = 0\n"
    )
    read_back = envi.read(str(base))
    assert read_back is not None
    assert read_back.shape == (3, 4)
    np.testing.assert_array_equal(read_back, band1)


def test_read_honors_header_offset_and_byte_order(tmp_path: Path) -> None:
    """Checks the header-offset skip and the big-endian byte-order flag."""
    base = tmp_path / "off"
    data = np.array([[1, 2], [3, 4]], dtype=">i2")
    with open(base, "wb") as fp:
        fp.write(b"JUNK")
        data.tofile(fp)
    (tmp_path / "off.hdr").write_text(
        "ENVI\nsamples = 2\nlines = 2\nbands = 1\nheader offset = 4\ndata type = 2\ninterleave = bsq\nbyte order = 1\n"
    )
    read_back = envi.read(str(base))
    assert read_back is not None
    np.testing.assert_array_equal(read_back, data.astype("<i2"))


def test_header_parser_skips_multiline_values(tmp_path: Path) -> None:
    """Checks the .hdr parser: keys are lowercased and multi-line { ... }
    values are skipped without swallowing the following keys."""
    hdr = tmp_path / "h.hdr"
    hdr.write_text(
        "ENVI\n"
        "description = {\n  a multi-line,\n  example value\n}\n"
        "Samples = 4\n"
        "Lines   = 2\n"
    )
    header = envi._read_envi_header(str(hdr))
    assert header["samples"] == "4"
    assert header["lines"] == "2"
    assert "description" not in header
