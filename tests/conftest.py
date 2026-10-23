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

"""Pytest configuration shared by the whole SOPA test suite.

It selects the non-interactive Agg matplotlib backend before any module gets
the chance to import pyplot (PyQt6 is installed in the project virtualenv, so
without this an active display session would auto-select the QtAgg backend)
and provides an autouse fixture that closes all matplotlib figures after
every test, so that figures leaked by one test never reach the next one.
"""

import matplotlib

matplotlib.use("Agg")       # non-interactive backend; must be set before any pyplot import

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True)
def close_matplotlib_figures() -> Iterator[None]:
    """Closes every matplotlib figure left open by a test, immediately after it."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")
