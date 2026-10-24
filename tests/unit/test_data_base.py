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

"""Unit tests of the dataset-reader base class (sopa.input.data_base),
exercised through a minimal concrete subclass: metadata storage, encoding
selection, the default and injected loggers, the read-only data/orbit_data
properties, the external orbit-data installation, the longitude correction,
the string form and the load dispatch.
"""

import gzip

import numpy as np
import pytest

from sopa.input.data_base import RSData
from sopa.log.log import fake_logger, logger


class _MinimalData(RSData):
    """Minimal concrete reader used to exercise the RSData base behavior."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)          # type: ignore[arg-type]
        self.load_called_with: tuple[str | None, bool] | None = None

    def _load_from_file(self, mode: str | None = None, force_reload: bool = False) -> bool:
        """Records the dispatch arguments and reports success."""
        self.load_called_with = (mode, force_reload)
        return True

    def generate_orbit_data(self, skip: int = 1) -> None:
        """Installs a minimal orbit-data dictionary with one negative longitude."""
        self._orbit_data = {"Lon": np.array([-1.0, 5.0])}


def test_constructor_defaults() -> None:
    """Checks the constructor defaults: name, counters, UTF-8/plain-open
    selection and the fake logger."""
    d = _MinimalData()
    assert d.dataset_name == "no name"
    assert d.n_frames == 0
    assert d.pri == 0
    assert d.presumming == 0
    assert d.filename_base == ""
    assert d.encoding == "UTF-8"
    assert d.open_function is open
    assert isinstance(d.logger, fake_logger)


def test_constructor_gzip_selection() -> None:
    """Checks that the gzip option selects ISO-8859-1 encoding and gzip.open."""
    d = _MinimalData(use_gzip_and_iso8859_1=True)
    assert d.encoding == "ISO-8859-1"
    assert d.open_function is gzip.open


def test_constructor_logger_injection() -> None:
    """Checks that a provided logger is used instead of the fake one."""
    lg = logger(verbose=False)
    d = _MinimalData(logger=lg)
    assert d.logger is lg


def test_data_property_before_and_after_load() -> None:
    """Checks the read-only data property: None with an error message before
    the echo matrix is loaded, the matrix afterwards."""
    d = _MinimalData()
    assert d.data is None
    d._data = np.arange(6).reshape(2, 3)
    np.testing.assert_array_equal(d.data, np.arange(6).reshape(2, 3))


def test_data_setter_rejected() -> None:
    """Checks that assigning the data property is refused and leaves the
    internal matrix untouched."""
    d = _MinimalData()
    d.data = np.ones((2, 2))            # type: ignore[assignment]
    assert d._data is None


def test_orbit_data_property_and_external_install() -> None:
    """Checks the read-only orbit_data property and the external orbit-data
    installation bypassing generate_orbit_data."""
    d = _MinimalData()
    assert d.orbit_data is None
    external = {"Lat": np.array([1.0])}
    d.set_orbit_data_from_ext(external)
    assert d.orbit_data is external
    d.orbit_data = {}                   # type: ignore[assignment]
    assert d.orbit_data is external


def test_generate_orbit_data_and_lon_correction() -> None:
    """Checks the longitude correction: negative longitudes are remapped to
    the 0-360 range, positive ones are untouched."""
    d = _MinimalData()
    d.generate_orbit_data()
    assert d.orbit_data is not None
    d._correct_lon()
    np.testing.assert_allclose(d.orbit_data["Lon"], [359.0, 5.0])


def test_str_with_and_without_filename() -> None:
    """Checks the string form with and without a base filename."""
    d = _MinimalData(dataset_name="obs", pri=1428e-6, presumming=16)
    assert str(d) == "obs, n_frames=0, PRI=1428.0 us, presum=16"
    d.filename_base = "/some/base"
    assert str(d) == "obs, n_frames=0, PRI=1428.0 us, presum=16, base filename: /some/base"


def test_load_without_filename(capsys: pytest.CaptureFixture[str]) -> None:
    """Checks that load without a base filename reports an error and returns
    False without dispatching."""
    d = _MinimalData()
    assert d.load() is False
    assert "ERROR" in capsys.readouterr().out
    assert d.load_called_with is None


def test_load_dispatches_to_subclass() -> None:
    """Checks that load dispatches mode and force_reload to the concrete
    _load_from_file implementation."""
    d = _MinimalData(filename_base="/some/base")
    assert d.load() is True
    assert d.load_called_with == ("full", False)
    assert d.load(mode="ancillary", force_reload=True) is True
    assert d.load_called_with == ("ancillary", True)
