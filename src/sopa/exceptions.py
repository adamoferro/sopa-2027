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

"""Exception hierarchy of the sopa library: a single library root with leaf
classes dual-rooted in the matching built-in exceptions, so existing
`except OSError`/`except ValueError` handlers keep working while a single
`except SopaError` isolates every refusal raised by the library. The leaf
classes carry the refusal message of the site that raises them (the historical
console texts, prefixes included).
"""


class SopaError(Exception):
    """Root of the sopa library's exception hierarchy: every error the
    library reports derives from it."""


class SopaFileError(SopaError, OSError):
    """A missing or unreadable file: a label, auxiliary table, science data
    file, DEM or ENVI product that is not present or cannot be read."""


class SopaDataError(SopaError, ValueError):
    """Malformed or unsupported data, or a data object accessed in a state
    that does not allow it: an unparseable or undecodable header, an
    unsupported format variant, an inconsistent auxiliary file, or an access
    to data that have not been loaded or generated yet."""


class SopaSelectionError(SopaError, ValueError):
    """An invalid frame selection or geometry request: invalid frame limits,
    an end frame beyond the input data or before the start frame, a computed
    frame selection that comes out empty, or a degenerate geographic request
    (an empty ROI set, a circle area without a defined bounding box)."""
