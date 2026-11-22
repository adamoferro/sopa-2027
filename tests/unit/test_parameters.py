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

"""Unit tests of the processing parameter containers
(sopa.processing.parameters): defaults, the frame_start clamp against the
synthetic-aperture half length, the meters-skip priority over the frame skip,
the non-positive skip validation, the focuser-specific additions, and the
coherence-time aperture specification (the seconds-to-frames conversion
helper and the mutually exclusive focuser aperture modes).
"""

import pytest

from sopa.processing.parameters import (
    FocuserParameters,
    SimulatorParameters,
    half_aperture_frames_from_coherence_time,
)


def test_simulator_parameters_defaults() -> None:
    """Checks the simulator parameter defaults: the frame start is clamped to
    the default half aperture of 768 frames."""
    p = SimulatorParameters()
    assert p.frame_start == 768
    assert p.frame_end == -1
    assert p.frame_skip == 1
    assert p.meters_skip is None
    assert p.frame_half_aperture == 768
    assert p.squint_angle == 0
    assert p.range_weighting_function == "hanning"
    assert p.range_weighting_kaiser_beta == 2


def test_frame_start_clamped_to_half_aperture() -> None:
    """Checks that a frame start smaller than the half aperture is clamped,
    and a larger one is kept."""
    assert (
        SimulatorParameters(frame_start=100, frame_half_aperture=200).frame_start == 200
    )
    assert (
        SimulatorParameters(frame_start=1000, frame_half_aperture=200).frame_start
        == 1000
    )


def test_meters_skip_takes_priority() -> None:
    """Checks that giving meters_skip switches the selection to the
    meters-based skip (frame_skip zeroed, meters stored)."""
    p = SimulatorParameters(meters_skip=123.5)
    assert p.frame_skip == 0
    assert p.meters_skip == 123.5


def test_frame_skip_without_meters() -> None:
    """Checks the frame-based selection: meters_skip stays None."""
    p = SimulatorParameters(frame_skip=4)
    assert p.frame_skip == 4
    assert p.meters_skip is None


def test_rejects_non_positive_meters_skip() -> None:
    """Checks that a non-positive meters_skip refuses at construction for
    both parameter containers: the former zero sentinel cannot be used to
    switch the skip mode anymore, and a negative ground distance has no
    meaning."""
    with pytest.raises(ValueError, match="ERROR: meters_skip must be positive"):
        SimulatorParameters(meters_skip=0)
    with pytest.raises(ValueError, match="ERROR: meters_skip must be positive"):
        SimulatorParameters(meters_skip=-2500.0)
    with pytest.raises(ValueError, match="ERROR: meters_skip must be positive"):
        FocuserParameters(meters_skip=0.0)


def test_rejects_non_positive_frame_skip() -> None:
    """Checks that a non-positive frame_skip refuses at construction when
    meters_skip is not given, for both parameter containers: a zero or
    negative arange step selects no frame and used to crash the consumers
    on the empty selection."""
    with pytest.raises(
        ValueError, match="ERROR: frame_skip must be positive when meters_skip"
    ):
        SimulatorParameters(frame_skip=0)
    with pytest.raises(
        ValueError, match="ERROR: frame_skip must be positive when meters_skip"
    ):
        SimulatorParameters(frame_skip=-5)
    with pytest.raises(
        ValueError, match="ERROR: frame_skip must be positive when meters_skip"
    ):
        FocuserParameters(frame_skip=0)


def test_focuser_parameters_defaults() -> None:
    """Checks the focuser parameter defaults beyond the shared base ones."""
    p = FocuserParameters()
    assert p.frame_skip == 1
    assert p.meters_skip is None
    assert p.multilooking_doppler_bandwidth == 0.8
    assert p.correct_range_migration is True
    assert p.compensate_doppler_phase is True
    assert p.azimuth_weighting_function == "hanning"
    assert p.azimuth_weighting_kaiser_beta == 2
    assert p.gauss_rec_n_filters == 0
    assert p.gauss_rec_alpha == 0.66
    assert p.range_weighting_function == "hanning"


def test_focuser_parameters_custom_values() -> None:
    """Checks that custom focuser values are stored and the shared base
    clamping still applies."""
    p = FocuserParameters(
        frame_start=1,
        frame_half_aperture=50,
        squint_angle=1.0,
        multilooking_doppler_bandwidth=0.6,
        correct_range_migration=False,
        compensate_doppler_phase=False,
        range_weighting_function="kaiser",
        range_weighting_kaiser_beta=3.0,
        azimuth_weighting_function="hamming",
        gauss_rec_n_filters=4,
        gauss_rec_alpha=0.5,
    )
    assert p.frame_start == 50
    assert p.squint_angle == 1.0
    assert p.multilooking_doppler_bandwidth == 0.6
    assert p.correct_range_migration is False
    assert p.compensate_doppler_phase is False
    assert p.range_weighting_function == "kaiser"
    assert p.range_weighting_kaiser_beta == 3.0
    assert p.azimuth_weighting_function == "hamming"
    assert p.gauss_rec_n_filters == 4
    assert p.gauss_rec_alpha == 0.5


def test_half_aperture_frames_from_coherence_time_floor_rule() -> None:
    """Checks the seconds-to-frames conversion rule: the half aperture is the
    floor of the requested coherence time over twice the per-frame interval,
    so the achieved coherence time of the resulting odd frame length
    2 * half + 1 never exceeds the request."""
    assert half_aperture_frames_from_coherence_time(1.0, 0.25) == 2
    assert half_aperture_frames_from_coherence_time(1.1, 0.25) == 2
    assert half_aperture_frames_from_coherence_time(0.99, 0.25) == 1


def test_half_aperture_frames_rejects_invalid_inputs() -> None:
    """Checks the ValueError cases of the seconds-to-frames conversion: a
    non-positive requested coherence time, a non-positive per-frame
    interval, and a requested time below the minimum coherence time
    2 * pri_data_s."""
    with pytest.raises(ValueError):
        half_aperture_frames_from_coherence_time(0.0, 0.25)
    with pytest.raises(ValueError):
        half_aperture_frames_from_coherence_time(-1.0, 0.25)
    with pytest.raises(ValueError):
        half_aperture_frames_from_coherence_time(1.0, 0.0)
    with pytest.raises(ValueError):
        half_aperture_frames_from_coherence_time(1.0, -0.25)
    with pytest.raises(ValueError):
        half_aperture_frames_from_coherence_time(0.3, 0.25)


def test_focuser_parameters_frames_mode_defaults() -> None:
    """Checks the focuser frames mode: the default half aperture of 768
    frames with the usual frame_start clamp, and the coherence time
    unset."""
    p = FocuserParameters()
    assert p.frame_half_aperture == 768
    assert p.frame_start == 768
    assert p.coherence_time_s is None


def test_focuser_parameters_coherence_time_mode() -> None:
    """Checks the focuser coherence-time mode: the requested time is stored,
    the frame_half_aperture attribute reads 0, and frame_start is kept as
    given instead of being clamped at construction."""
    p = FocuserParameters(frame_start=10, coherence_time_s=0.5)
    assert p.coherence_time_s == 0.5
    assert p.frame_half_aperture == 0
    assert p.frame_start == 10


def test_focuser_parameters_rejects_conflicting_specs() -> None:
    """Checks that giving both aperture specifications, the half aperture in
    frames and the coherence time in seconds, raises ValueError."""
    with pytest.raises(ValueError):
        FocuserParameters(frame_half_aperture=64, coherence_time_s=0.5)


def test_focuser_parameters_rejects_invalid_values() -> None:
    """Checks the ValueError cases of the focuser parameters: a non-positive
    coherence time and a negative frame half aperture, while a zero half
    aperture stays valid."""
    with pytest.raises(ValueError):
        FocuserParameters(coherence_time_s=0.0)
    with pytest.raises(ValueError):
        FocuserParameters(coherence_time_s=-1.0)
    with pytest.raises(ValueError):
        FocuserParameters(frame_half_aperture=-1)
    assert FocuserParameters(frame_half_aperture=0).frame_half_aperture == 0
