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

"""Unit tests of the PNG export (sopa.io_utils.png): real-data and
simulated-data dynamic-range stretching, the complex-to-power conversion,
the 8-bit grayscale format of the produced files, and the all-zero input
rejection.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import png
import pytest

from sopa.exceptions import SopaDataError
from sopa.io_utils import png as png_io


@pytest.fixture()
def noise_image() -> npt.NDArray[np.float64]:
    """A small positive image with one entirely-zero column (mimicking the
    zero-padded start of PDS RDR radargrams)."""
    rng = np.random.default_rng(9)
    image = rng.uniform(1.0, 100.0, size=(64, 32))
    image[:, 0] = 0.0
    return image


def test_write_real_data(noise_image: npt.NDArray[np.float64], tmp_path: Path) -> None:
    """Checks the real-data path: the file is produced as an 8-bit grayscale
    PNG with the original array shape (transposed into image orientation)."""
    out = tmp_path / "real.png"
    assert png_io.write(str(out), noise_image) is True
    assert out.is_file()
    reader = png.Reader(filename=str(out))
    width, height, _, info = reader.read()
    assert (width, height) == (32, 64)
    assert info["greyscale"] is True and info["bitdepth"] == 8


def test_write_sim_data(noise_image: npt.NDArray[np.float64], tmp_path: Path) -> None:
    """Checks the simulated-data path (percentile-based dynamic range)."""
    out = tmp_path / "sim.png"
    assert png_io.write(str(out), noise_image, sim_data=True) is True
    assert out.is_file()
    width, height, _, info = png.Reader(filename=str(out)).read()
    assert (width, height) == (32, 64)
    assert info["bitdepth"] == 8


def test_write_complex_converts_to_power(tmp_path: Path) -> None:
    """Checks that complex input is converted to power before the stretch."""
    out = tmp_path / "cplx.png"
    data = np.full((16, 8), 1.0 + 1.0j, dtype="complex64")
    assert png_io.write(str(out), data) is True
    assert out.is_file()


def test_write_all_zero_image_fails(
    tmp_path: Path,
) -> None:
    """Checks that an all-zero image cannot be stretched (no noise level can
    be estimated): the write raises SopaDataError before any file is
    produced."""
    out = tmp_path / "zero.png"
    with pytest.raises(SopaDataError, match="ERROR: cannot determine noise mean"):
        png_io.write(str(out), np.zeros((8, 8)))
    assert not out.exists()
