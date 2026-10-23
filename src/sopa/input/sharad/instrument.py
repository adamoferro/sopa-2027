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

"""SHARAD instrument parameters.
"""

from sopa.input.instrument_base import RSInstrument


class SHARAD(RSInstrument):
    """Fixed-parameter definition of the SHARAD instrument: 20 MHz center frequency, 10 MHz bandwidth, 80/3 MHz fast-time sampling frequency, 85.05 us chirp duration, 3397 m/s reference platform velocity."""

    def __init__(self) -> None:
        super().__init__("SHARAD", 20e6, 10e6, 80e6/3., 85.05e-6, 3397.)
