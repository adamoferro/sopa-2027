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

"""Structural interfaces and shared types for the objects that cross module
boundaries in the sopa package: typing Protocols for the instrument, geometry
conversion, DEM, dataset and processing-parameter objects and for the
message logger of the dataset readers, plus the echo-matrix and orbit-data
types and the frame-simulation callback contract shared by the clutter
simulators. The Protocols are matched structurally by static type checkers
only; nothing in the package performs runtime checks against them, and the
Protocol and alias definitions have no runtime cost beyond their definition.
"""

from collections.abc import Callable
from typing import Protocol

import numpy as np
import numpy.typing as npt

type EchoMatrix = npt.NDArray[np.float64] | npt.NDArray[np.complex64]
# The per-reader echo matrix: float64 for the EDR readers and the US RDR
# reader, complex64 for the focused RDR reader.

type OrbitDataValue = npt.NDArray[np.float64] | npt.NDArray[np.float32] | float
# One orbit-data entry: a float64 per-frame vector in every generator, the
# float32 receive-window opening times of the EDR and RDR PDS readers, or a
# scalar (the dt value).

type OrbitData = dict[str, OrbitDataValue]
# The per-frame orbit-data dictionary of a dataset reader (X, Y, Z, Lat, Lon,
# Radius, rxwin_time_s, time_offset_s, dt, and the attitude/geometry fields).

# The frame-simulation callbacks are the simulators' `_simulate_frame_range`
# static methods: nineteen positional parameters (range id, range count, the
# per-frame orbit position vectors, the frame range, the sample count, the
# local direction and UTM vectors, the UTM zone list, the receive-window
# reference, the shift-alignment time, the time step, the DEM and geometry
# objects, the simulation matrices, the Gaussian-filter sigma, the engine
# parameters and the ROI image) plus the keyword-only alt_unit_vector_carry
# and verbose that the dispatch paths pass; both keyword-only parameters have
# defaults on the concrete callbacks, which a parameterized Callable cannot
# express, so the parameter list is documented here instead of spelled out.
type FrameSimulationCallback = Callable[..., FrameSimulationResult]

type FrameSimulationResult = tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.uint8],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    tuple[dict[int, npt.NDArray[np.float64]], ...],
]
# One whole-track frame simulation: the simulated image, its uncertainty
# mask, the first-return lat/lon vectors, the ground-distance min/max images
# and the per-dictionary ROI position tuples (two dictionaries for dRSsim,
# four for sfRSsim).


class MessageLogger(Protocol):
    """Message-logging interface consumed by the dataset readers: buffers
    log messages and appends the buffer to a text file on request (the
    concrete implementations are `sopa.log.log.logger` and
    `sopa.log.log.fake_logger`)."""

    def log_messages(self, new_message: str) -> None: ...

    def log_write(self, filename: str) -> None: ...


class Instrument(Protocol):
    """Instrument-parameter interface consumed by the SOFA focuser and by
    the simulators' chirp compression: center frequency, bandwidth and
    fast-time sampling frequency in Hz, chirp duration in s and platform
    reference velocity in m/s (the concrete class is
    `sopa.input.instrument_base.RSInstrument`)."""

    frequency: float

    bandwidth: float

    fast_time_sampling_frequency: float

    chirp_duration: float

    ref_v_platform: float


class GeometryConverter(Protocol):
    """Geometry-conversion interface consumed by the SOFA focuser and the
    clutter simulators: ellipsoid radius lookup, planetocentric lat/lon to
    cartesian conversion, cartesian distances, UTM conversion and frame
    selection by ground distance (the concrete class is
    `sopa.geometry.coordinate_converter.CoordinateConverter`)."""

    def euclidean_distance_cart(
        self,
        x1: npt.ArrayLike,
        y1: npt.ArrayLike,
        z1: npt.ArrayLike,
        x2: npt.ArrayLike,
        y2: npt.ArrayLike,
        z2: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]: ...

    def to_xyz_from_latlon_and_radius(
        self, latitude: npt.ArrayLike, longitude: npt.ArrayLike, radius: npt.ArrayLike
    ) -> tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ]: ...

    def get_frame_ids_from_skip_distance(
        self,
        lat: npt.NDArray[np.float64],
        lon: npt.NDArray[np.float64],
        skip_distance: float,
    ) -> tuple[npt.NDArray[np.int64], float, float]: ...

    def get_ellipsoid_radius_from_latlon(
        self, latitude: npt.ArrayLike, longitude: npt.ArrayLike
    ) -> npt.NDArray[np.float64]: ...

    def UTM_to_LL(
        self,
        UTMNorthing: npt.NDArray[np.float64],
        UTMEasting: npt.NDArray[np.float64],
        fixed_UTMZone: str,
    ) -> tuple[npt.NDArray[np.float64] | float, npt.NDArray[np.float64] | float]: ...


class ScaledDEMRadiusSampler(Protocol):
    """Radius-sampler interface of the smoothed/upscaled sub-DEM returned by
    `DEM.get_scaled_subdem_from_latlon_limits` and consumed by the sfRSsim
    simulator: the dummy value of the replaced MOLA view plus the radius
    lookup on the scaled grid."""

    dummy_value: int

    def get_dem_radius_from_lat_lon(
        self, lat: float | npt.NDArray[np.float64], lon: float | npt.NDArray[np.float64]
    ) -> npt.NDArray[np.int32]: ...


class DEM(Protocol):
    """Digital-elevation-model interface consumed by the ground-track plots
    and the clutter simulators: the degree sampling and dummy value, the
    lat/lon-pixel conversions, radius lookup by pixel ranges and by lat/lon,
    and the smoothed/upscaled sub-DEM sampler (the concrete classes are the
    `sopa.dem` readers, e.g. `sopa.dem.mola.mola.MOLA128`)."""

    dummy_value: int

    deg_sampling: float

    def to_XY_from_LL(
        self,
        lat: float | npt.NDArray[np.float64],
        long: float | npt.NDArray[np.float64],
        return_as_float: bool = False,
    ) -> (
        tuple[float, float]
        | tuple[
            npt.NDArray[np.float64] | npt.NDArray[np.int64],
            npt.NDArray[np.float64] | npt.NDArray[np.int64],
        ]
    ): ...

    def to_LL_from_XY(self, x: float, y: float) -> tuple[float, float]: ...

    def XY_ranges_to_radius(
        self,
        x_min: int,
        x_max: int,
        y_min: int,
        y_max: int,
        add_offset: bool = True,
    ) -> npt.NDArray[np.int32]: ...

    def get_dem_radius_from_lat_lon(
        self, lat: float | npt.NDArray[np.float64], lon: float | npt.NDArray[np.float64]
    ) -> npt.NDArray[np.int32] | None: ...

    def get_scaled_subdem_from_latlon_limits(
        self,
        lat_n: float,
        lat_s: float,
        lon_e: float,
        lon_w: float,
        scale_factor: float = 4,
    ) -> ScaledDEMRadiusSampler | None: ...


class RawDataset(Protocol):
    """Raw-dataset interface consumed by the SOFA focuser: the echo matrix
    and the orbit data through their read-only accessors, the raw-data type
    tag the focuser validates, and the PRI and presumming factor (the
    concrete class is the raw-data EDR reader,
    `sopa.input.sharad.data.edr.pds.EDR` or
    `sopa.input.sharad.data.edr.cosharps.EDR`)."""

    _data_type: str

    pri: float

    presumming: int

    data: EchoMatrix | None

    orbit_data: OrbitData | None


class FocusingParameters(Protocol):
    """Focuser-parameter interface consumed by the SOFA focuser: frame range
    and skip, synthetic-aperture half length (in frames, after the
    coherence-time mode is resolved at construction), squint angle,
    range-migration and Doppler-phase compensation switches, weighting
    windows, multilooking Doppler bandwidth and the Gaussian-filter
    multilooking decomposition (the concrete class is
    `sopa.processing.parameters.FocuserParameters`)."""

    frame_start: int

    frame_end: int

    meters_skip: float | None

    frame_skip: int

    frame_half_aperture: int

    coherence_time_s: float | None

    squint_angle: float

    multilooking_doppler_bandwidth: float

    correct_range_migration: bool

    compensate_doppler_phase: bool

    range_weighting_function: str

    range_weighting_kaiser_beta: float

    azimuth_weighting_function: str

    azimuth_weighting_kaiser_beta: float

    gauss_rec_n_filters: int

    gauss_rec_alpha: float


class SimulationParameters(Protocol):
    """Simulator-parameter interface consumed by the clutter simulators'
    base class: frame range and skip and the synthetic-aperture half length
    (the concrete class is `sopa.processing.parameters.SimulatorParameters`)."""

    frame_start: int

    frame_end: int

    meters_skip: float | None

    frame_skip: int

    frame_half_aperture: int
