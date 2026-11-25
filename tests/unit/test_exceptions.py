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

"""Unit tests of the library exception hierarchy (sopa.exceptions): the
derivation of every leaf class from the library root and from its matching
built-in, and the message-carrying behavior of a raised instance.
"""

import pytest

from sopa.exceptions import SopaDataError, SopaError, SopaFileError, SopaSelectionError


def test_all_leaves_derive_from_sopa_error() -> None:
    """Checks that every leaf exception derives from the library root, so a
    single `except SopaError` isolates every refusal raised by the library."""
    for exc_type in (SopaFileError, SopaDataError, SopaSelectionError):
        assert issubclass(exc_type, SopaError)


def test_sopa_file_error_is_os_error() -> None:
    """Checks the dual rooting of SopaFileError in OSError: existing
    `except OSError` handlers keep catching the library's missing/unreadable
    -file refusals, and an underlying FileNotFoundError stays reachable as
    the chained cause."""
    assert issubclass(SopaFileError, OSError)
    with pytest.raises(OSError):
        raise SopaFileError("ERROR: missing file.")
    with pytest.raises(SopaFileError) as excinfo:
        try:
            open("/nonexistent/sopa/no/such/file", "rb")
        except FileNotFoundError as err:
            raise SopaFileError("ERROR: missing file.") from err
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_sopa_data_error_is_value_error() -> None:
    """Checks the dual rooting of SopaDataError in ValueError: malformed
    data and not-yet-loaded state refusals keep being caught by existing
    `except ValueError` handlers."""
    assert issubclass(SopaDataError, ValueError)


def test_sopa_selection_error_is_value_error() -> None:
    """Checks the dual rooting of SopaSelectionError in ValueError: invalid
    frame selections and degenerate geometry requests keep being caught by
    existing `except ValueError` handlers."""
    assert issubclass(SopaSelectionError, ValueError)


def test_raised_instance_carries_the_message() -> None:
    """Checks that the refusal message (the historical console text,
    prefixes included) is the string form of the raised exception."""
    with pytest.raises(SopaDataError) as excinfo:
        raise SopaDataError("ERROR: data not yet read.")
    assert str(excinfo.value) == "ERROR: data not yet read."
