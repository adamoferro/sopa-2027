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

"""Containers of the simulation and focusing parameters shared along the
processing chain.
"""

# TODO:
# - express half aperture in seconds or meters (when expressed in frames it depends on presumming)
# - check correctness of parameters
# - add Doppler max search and adjacent Doppler windows (as in SOFA "original")


class _BaseProcessingParameters:
    """Common frame-range, aperture and range-weighting parameters. `frame_start` is clamped to `frame_half_aperture`; when `meters_skip` is given it takes priority over `frame_skip` (frame selection by equal ground distance instead of a fixed frame count)."""

    def __init__(self, frame_start: int = 0, frame_end: int = -1, frame_skip: int = 1, meters_skip: float | None = None, frame_half_aperture: int = 768, squint_angle: float = 0, range_weighting_function: str = "hanning", range_weighting_kaiser_beta: float = 2) -> None:
        # frame_half_aperture               samples
        # squint_angle                      degrees
        # range_weighting_function          ["hanning","hamming","kaiser","none"]
        # range_weighting_kaiser_beta       float > 0
        # meters_skip has priority on frame_skip

        self.frame_start = max([frame_start, frame_half_aperture])
        self.frame_end = frame_end
        if meters_skip is not None:
            self.meters_skip = meters_skip
            self.frame_skip = 0
        else:
            self.frame_skip = frame_skip
            self.meters_skip = 0

        self.frame_half_aperture = frame_half_aperture
        self.squint_angle = squint_angle
        self.range_weighting_function = range_weighting_function
        self.range_weighting_kaiser_beta = range_weighting_kaiser_beta


class SimulatorParameters(_BaseProcessingParameters):
    """Parameter set for the clutter simulators: frame range, frame/meters skip, synthetic-aperture half length, squint angle and range weighting window."""
    def __init__(self, frame_start: int = 0, frame_end: int = -1, frame_skip: int = 1, meters_skip: float | None = None, frame_half_aperture: int = 768, squint_angle: float = 0, range_weighting_function: str = "hanning", range_weighting_kaiser_beta: float = 2) -> None:
        super().__init__(frame_start, frame_end, frame_skip, meters_skip, frame_half_aperture, squint_angle, range_weighting_function, range_weighting_kaiser_beta)


class FocuserParameters(_BaseProcessingParameters):
    """Parameter set for the focuser: adds the multilooking Doppler bandwidth, the range-migration and Doppler-phase compensation switches, the azimuth weighting window, and the Gaussian-filter multilooking decomposition (number of filters and alpha coefficient)."""
    def __init__(self, frame_start: int = 0, frame_end: int = -1, frame_skip: int = 1, meters_skip: float | None = None, frame_half_aperture: int = 768, squint_angle: float = 0, multilooking_doppler_bandwidth: float = 0.8, correct_range_migration: bool = True, compensate_doppler_phase: bool = True, range_weighting_function: str = "hanning", range_weighting_kaiser_beta: float = 2, azimuth_weighting_function: str = "hanning", azimuth_weighting_kaiser_beta: float = 2, gauss_rec_n_filters: int = 0, gauss_rec_alpha: float = 0.66) -> None:
        # multilooking_doppler_bandwidth    Hz
        # correct_range_migration           boolean
        # compensate_doppler_phase          boolean
        # azimuth_weighting_function        ["hanning","hamming","kaiser","none"]
        # azimuth_weighting_kaiser_beta     float > 0
        # gauss_rec_n_filters               int >= 0
        # gauss_rec_alpha                   float > 0

        super().__init__(frame_start, frame_end, frame_skip, meters_skip, frame_half_aperture, squint_angle, range_weighting_function, range_weighting_kaiser_beta)
        self.multilooking_doppler_bandwidth = multilooking_doppler_bandwidth
        self.correct_range_migration = correct_range_migration
        self.compensate_doppler_phase = compensate_doppler_phase
        self.azimuth_weighting_function = azimuth_weighting_function
        self.azimuth_weighting_kaiser_beta = azimuth_weighting_kaiser_beta
        self.gauss_rec_n_filters = gauss_rec_n_filters
        self.gauss_rec_alpha = gauss_rec_alpha
