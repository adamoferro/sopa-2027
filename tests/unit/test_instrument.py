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

"""Unit tests of the instrument definitions (sopa.input.instrument_base,
sopa.input.sharad.instrument): the base-class attribute storage and string
form, and the fixed SHARAD parameter set.
"""

from sopa.input.instrument_base import RSInstrument
from sopa.input.sharad.instrument import SHARAD


def test_base_stores_parameters() -> None:
    """Checks that the base class stores the physical parameters it is given."""
    i = RSInstrument("TEST", 20e6, 10e6, 100.0, 1e-3, 3000.0)
    assert i.name == "TEST"
    assert i.frequency == 20e6
    assert i.bandwidth == 10e6
    assert i.fast_time_sampling_frequency == 100.0
    assert i.chirp_duration == 1e-3
    assert i.ref_v_platform == 3000.0
    assert str(i) == "TEST, fc=20.0 MHz, fb=10.0 MHz"


def test_sharad_fixed_parameters() -> None:
    """Checks the SHARAD instrument parameter set: 20 MHz center frequency,
    10 MHz bandwidth, 80/3 MHz sampling, 85.05 us chirp, 3397 m/s reference
    velocity."""
    s = SHARAD()
    assert s.name == "SHARAD"
    assert s.frequency == 20e6
    assert s.bandwidth == 10e6
    assert s.fast_time_sampling_frequency == 80e6 / 3.0
    assert s.chirp_duration == 85.05e-6
    assert s.ref_v_platform == 3397.0
    assert isinstance(s, RSInstrument)
    assert str(s) == "SHARAD, fc=20.0 MHz, fb=10.0 MHz"
