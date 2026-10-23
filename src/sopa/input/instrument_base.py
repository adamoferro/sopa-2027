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

"""Base class for radar-sounder instrument definitions.
"""

class RSInstrument():
    """Stores the physical and acquisition parameters common to radar sounders: center frequency, bandwidth, fast-time sampling frequency, chirp duration and platform reference velocity."""

    def __init__(self, name: str, frequency: float, bandwidth: float, fast_time_sampling_frequency: float, chirp_duration: float, ref_v_platform: float) -> None:
        self.name = name
        self.frequency = frequency                                          # Hz
        self.bandwidth = bandwidth                                          # Hz
        self.fast_time_sampling_frequency = fast_time_sampling_frequency    # Hz
        self.chirp_duration = chirp_duration                                # s
        self.ref_v_platform = ref_v_platform                                # m/s

    def __str__(self) -> str:
        return f"{self.name}, fc={self.frequency/1e6:.1f} MHz, fb={self.bandwidth/1e6:.1f} MHz"
