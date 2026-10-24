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

"""Permanent import-every-module check of the sopa library (deferred from
phase 2, assigned to phase 3 cycle 2 by the project status and technical-debt
pages).

Every non-__init__ module under src/sopa is imported by name; importing them
transitively imports every package __init__ as well, so the whole tree is
covered. The module list is derived from the resolved sopa package directory,
so a module added to the library in the future is picked up automatically.
"""

import importlib
import sys
from pathlib import Path

import pytest

import sopa


def _library_module_names() -> list[str]:
    """Returns the dotted names of all non-__init__ modules of the sopa package."""
    package_dir = Path(sopa.__file__).parent
    return sorted(
        "sopa." + p.relative_to(package_dir).with_suffix("").as_posix().replace("/", ".")
        for p in package_dir.rglob("*.py")
        if p.name != "__init__.py"
    )


@pytest.mark.parametrize("module_name", _library_module_names())
def test_library_module_imports(module_name: str) -> None:
    """Imports one library module by dotted name; an import error fails the test."""
    importlib.import_module(module_name)
    assert module_name in sys.modules
