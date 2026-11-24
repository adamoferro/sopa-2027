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

import math
from dataclasses import dataclass

# TODO:
# - express half aperture in meters (when expressed in frames it depends on
#   presumming; seconds are supported via FocuserParameters.coherence_time_s)
# - check correctness of parameters
# - add Doppler max search and adjacent Doppler windows (as in SOFA "original")


def half_aperture_frames_from_coherence_time(
    coherence_time_s: float, pri_data_s: float
) -> int:
    """Converts a requested coherence time in seconds into the
    synthetic-aperture half length in frames for data sampled with the
    per-frame interval pri_data_s (the data PRI multiplied by the presumming
    factor): the half length is the floor of
    coherence_time_s / (2 * pri_data_s), so the resulting odd aperture
    length of 2 * half + 1 frames (center frame included) has an achieved
    coherence time of 2 * half * pri_data_s that never exceeds the requested
    one. Raises ValueError when coherence_time_s or pri_data_s is not
    strictly positive (a non-positive pri_data_s means the data carry no
    usable PRI: check that the data declare a PRI and the presumming
    factor), or when the requested time is below the minimum coherence time
    of 2 * pri_data_s (one frame on each side of the center frame).
    """
    if coherence_time_s <= 0:
        raise ValueError(
            f"ERROR: coherence_time_s must be > 0, got {coherence_time_s!r} s."
        )
    if pri_data_s <= 0:
        raise ValueError(
            f"ERROR: pri_data_s must be > 0, got {pri_data_s!r} s (the data "
            "carry no usable PRI: check that the data declare a PRI and the "
            "presumming factor)."
        )
    half = math.floor(coherence_time_s / (2.0 * pri_data_s))
    if half < 1:
        raise ValueError(
            f"ERROR: coherence_time_s={coherence_time_s!r} s is below the "
            f"minimum coherence time {2 * pri_data_s!r} s (= 2 * pri_data_s) "
            f"for pri_data_s={pri_data_s!r} s: no aperture frame fits."
        )
    return int(half)


class _BaseProcessingParameters:
    """Common frame-range, aperture and range-weighting parameters. `frame_start` is clamped to `frame_half_aperture`; when `meters_skip` is given it takes priority over `frame_skip` (frame selection by equal ground distance instead of a fixed frame count) and is stored with `frame_skip` zeroed, otherwise `frame_skip` is stored with `meters_skip` left None; a non-positive meters_skip raises ValueError, as does a non-positive frame_skip when meters_skip is not given."""

    def __post_init__(self) -> None:
        """Applies the shared construction normalization: clamps `frame_start`
        to `frame_half_aperture` and selects the skip mode (meters-based with
        `frame_skip` zeroed when `meters_skip` is given, frame-based
        otherwise)."""
        self.frame_start = max([self.frame_start, self.frame_half_aperture])
        if self.meters_skip is not None:
            if self.meters_skip <= 0:
                raise ValueError("ERROR: meters_skip must be positive.")
            self.frame_skip = 0
        else:
            if self.frame_skip <= 0:
                raise ValueError(
                    "ERROR: frame_skip must be positive when meters_skip is not given."
                )
            self.meters_skip = None


@dataclass(eq=False, repr=False)
class SimulatorParameters(_BaseProcessingParameters):
    """Parameter set for the clutter simulators: frame range, frame/meters skip, synthetic-aperture half length, squint angle and range weighting window."""

    # frame_half_aperture               samples
    # squint_angle                      degrees
    # range_weighting_function          ["hanning","hamming","kaiser","none"]
    # range_weighting_kaiser_beta       float > 0
    # meters_skip has priority on frame_skip
    frame_start: int = 0
    frame_end: int = -1
    frame_skip: int = 1
    meters_skip: float | None = None
    frame_half_aperture: int = 768
    squint_angle: float = 0
    range_weighting_function: str = "hanning"
    range_weighting_kaiser_beta: float = 2


@dataclass(eq=False, repr=False)
class FocuserParameters(_BaseProcessingParameters):
    """Parameter set for the focuser: adds the multilooking Doppler bandwidth, the range-migration and Doppler-phase compensation switches, the azimuth weighting window, and the Gaussian-filter multilooking decomposition (number of filters and alpha coefficient). The synthetic-aperture half length is given either in frames (frame_half_aperture, the frames mode, default 768: odd aperture length of 2 * frame_half_aperture + 1 frames) or in seconds as a coherence time (coherence_time_s, mutually exclusive with frame_half_aperture): in the time mode the frame_half_aperture attribute reads 0, frame_start is not clamped at construction, and the focuser resolves the half aperture in frames from the requested time against the data PRI after presumming, then clamps frame_start to it."""

    # frame_half_aperture               frames, None in the coherence-time mode
    # coherence_time_s                  seconds, None in the frames mode
    # multilooking_doppler_bandwidth    Hz
    # correct_range_migration           boolean
    # compensate_doppler_phase          boolean
    # azimuth_weighting_function        ["hanning","hamming","kaiser","none"]
    # azimuth_weighting_kaiser_beta     float > 0
    # gauss_rec_n_filters               int >= 0
    # gauss_rec_alpha                   float > 0
    frame_start: int = 0
    frame_end: int = -1
    frame_skip: int = 1
    meters_skip: float | None = None
    frame_half_aperture: int | None = None
    coherence_time_s: float | None = None
    squint_angle: float = 0
    multilooking_doppler_bandwidth: float = 0.8
    correct_range_migration: bool = True
    compensate_doppler_phase: bool = True
    range_weighting_function: str = "hanning"
    range_weighting_kaiser_beta: float = 2
    azimuth_weighting_function: str = "hanning"
    azimuth_weighting_kaiser_beta: float = 2
    gauss_rec_n_filters: int = 0
    gauss_rec_alpha: float = 0.66

    def __post_init__(self) -> None:
        """Validates the aperture specification (mutually exclusive
        frame_half_aperture and coherence_time_s, non-positive coherence
        time, negative half aperture), resolves the coherence-time mode into
        the stored frame_half_aperture (0 in the time mode, 768 when neither
        is given), then applies the shared normalization of the base class."""
        if self.frame_half_aperture is not None and self.coherence_time_s is not None:
            raise ValueError(
                f"ERROR: frame_half_aperture ({self.frame_half_aperture!r} frames) "
                f"and coherence_time_s ({self.coherence_time_s!r} s) are mutually "
                "exclusive: give the half aperture in frames or the coherence "
                "time in seconds, not both."
            )
        if self.coherence_time_s is not None and self.coherence_time_s <= 0:
            raise ValueError(
                f"ERROR: coherence_time_s must be > 0, got {self.coherence_time_s!r} s."
            )
        if self.frame_half_aperture is not None and self.frame_half_aperture < 0:
            raise ValueError(
                f"ERROR: frame_half_aperture must be >= 0, got {self.frame_half_aperture!r}."
            )

        self.frame_half_aperture = (
            768
            if (self.frame_half_aperture is None and self.coherence_time_s is None)
            else (0 if self.coherence_time_s is not None else self.frame_half_aperture)
        )
        super().__post_init__()
