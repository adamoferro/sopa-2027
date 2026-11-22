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

"""Unit tests of the coordinate conversions (sopa.geometry.coordinate_converter).

The cartesian/planetocentric conversions are checked against hand-computed
literals on the principal axes, the ellipsoid radius against the equatorial
and polar radii, the UTM projection against a forward/inverse roundtrip on
the IAU 2000 Mars ellipsoid, and the remaining helpers against their
geometric definitions.
"""

import numpy as np
import numpy.typing as npt
import pytest

from sopa.geometry.coordinate_converter import MARS_IAU2000, CoordinateConverter


@pytest.fixture()
def cc() -> MARS_IAU2000:
    """The Mars IAU 2000 converter used across the tests."""
    return MARS_IAU2000()


def test_mars_ellipsoid_constants(cc: MARS_IAU2000) -> None:
    """Checks the IAU 2000 Mars ellipsoid parameters and the derived
    flattening/eccentricity."""
    assert cc._ELLIPSOID_EQUATOR_RADIUS == 3396190.0
    assert cc._ELLIPSOID_POLAR_RADIUS == 3376200.0
    assert cc._MEAN_RADIUS == 3389500.0
    assert cc._FLATTENING == pytest.approx((3396190.0 - 3376200.0) / 3396190.0)
    assert cc._ECCENTRICITY_SQUARED == pytest.approx(
        (3396190.0**2 - 3376200.0**2) / 3396190.0**2
    )


def test_to_radius_from_xyz() -> None:
    """Checks the cartesian norm on a 3-4-5 triangle."""
    r = CoordinateConverter.to_radius_from_xyz(
        np.array([3.0]), np.array([4.0]), np.array([0.0])
    )
    assert r[0] == pytest.approx(5.0)


def test_euclidean_distance_cart() -> None:
    """Checks the euclidean distance between two cartesian points."""
    d = CoordinateConverter.euclidean_distance_cart(
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([3.0]),
        np.array([4.0]),
        np.array([0.0]),
    )
    assert d[0] == pytest.approx(5.0)


def test_to_xyz_from_latlon_and_radius_axes() -> None:
    """Checks the lat/lon/radius to cartesian conversion on the principal
    axes: lat 90 deg maps to +Z, the equator at lon 180 deg to -X and the
    equator at lon 90 deg to +Y."""
    r = 1000.0
    lat = np.array([90.0, 0.0, 0.0])
    lon = np.array([0.0, 180.0, 90.0])
    radius = np.full(3, r)
    x, y, z = CoordinateConverter.to_xyz_from_latlon_and_radius(lat, lon, radius)
    np.testing.assert_allclose(x, [0.0, -r, 0.0], atol=1e-9)
    np.testing.assert_allclose(y, [0.0, 0.0, r], atol=1e-9)
    np.testing.assert_allclose(z, [r, 0.0, 0.0], atol=1e-9)


def test_to_xyz_norm_preserves_radius() -> None:
    """Checks that the conversion preserves the radius (norm of the output
    equals the input radius) on arbitrary points."""
    rng = np.random.default_rng(3)
    lat = rng.uniform(-90.0, 90.0, 16)
    lon = rng.uniform(0.0, 360.0, 16)
    r = rng.uniform(3.0e6, 4.0e6, 16)
    x, y, z = CoordinateConverter.to_xyz_from_latlon_and_radius(lat, lon, r)
    norms = CoordinateConverter.to_radius_from_xyz(x, y, z)
    np.testing.assert_allclose(norms, r)


def test_add_radius_offset_to_orbit(cc: MARS_IAU2000) -> None:
    """Checks that the orbit-data radius offset raises every radius by the
    per-frame offset and recomputes X/Y/Z consistently."""
    lat = np.array([0.0, 1.0])
    lon = np.array([10.0, 11.0])
    radius = np.array([3646190.0, 3646290.0])
    x, y, z = cc.to_xyz_from_latlon_and_radius(lat, lon, radius)
    orbit_data: dict[str, npt.NDArray[np.float64] | float] = {
        "Lat": lat,
        "Lon": lon,
        "Radius": radius,
        "X": x,
        "Y": y,
        "Z": z,
    }
    offset = np.array([100.0, 200.0])
    new_orbit_data = cc.add_radius_offset_to_orbit(orbit_data, offset)
    np.testing.assert_allclose(new_orbit_data["Radius"], radius + offset)
    nx, ny, nz = cc.to_xyz_from_latlon_and_radius(
        lat, lon, np.asarray(new_orbit_data["Radius"])
    )
    np.testing.assert_allclose(new_orbit_data["X"], nx)
    np.testing.assert_allclose(new_orbit_data["Y"], ny)
    np.testing.assert_allclose(new_orbit_data["Z"], nz)
    assert orbit_data["Radius"][0] == 3646190.0  # input dict not modified


def test_to_latlon_area_from_circle_area_equator(cc: MARS_IAU2000) -> None:
    """Checks the bounding box of a circle centered on the equator, where the
    lat and lon half-extents coincide (sin of the reference latitude is 1)."""
    box = cc.to_latlon_area_from_circle_area(0.0, 0.0, 100000.0)
    assert box is not None
    lat_n, lat_s, lon_e, lon_w = box
    expected = 180.0 / np.pi * (100000.0 / cc._MEAN_RADIUS)
    assert lat_n == pytest.approx(expected)
    assert lat_s == pytest.approx(-expected)
    assert lon_e == pytest.approx(expected)
    assert lon_w == pytest.approx(-expected)


def test_to_latlon_area_from_circle_area_none_over_half_sphere(
    cc: MARS_IAU2000,
) -> None:
    """Checks that a radius reaching half the sphere returns None."""
    assert (
        cc.to_latlon_area_from_circle_area(0.0, 0.0, cc._MEAN_RADIUS * np.pi / 2.0)
        is None
    )


def test_to_latlon_area_from_circle_area_pole_clamping(cc: MARS_IAU2000) -> None:
    """Checks the pole cases: a circle overlapping the north (or south) pole
    spans all longitudes with the pole latitude clamped."""
    lat_n, lat_s, lon_e, lon_w = cc.to_latlon_area_from_circle_area(
        89.9999, 10.0, 500.0
    )
    assert lat_n == 90.0 and (lon_w, lon_e) == (0.0, 360.0)
    assert lat_s == pytest.approx(89.9999 - 180.0 / np.pi * (500.0 / cc._MEAN_RADIUS))
    lat_n, lat_s, lon_e, lon_w = cc.to_latlon_area_from_circle_area(
        -89.9999, 10.0, 500.0
    )
    assert lat_s == -90.0 and (lon_w, lon_e) == (0.0, 360.0)


def test_ellipsoid_radius_at_reference_latitudes(cc: MARS_IAU2000) -> None:
    """Checks the local ellipsoid radius: equatorial radius at the equator,
    polar radius at the poles, in between at mid-latitudes. The longitude
    argument is required by the signature but unused by the computation."""
    assert cc.get_ellipsoid_radius_from_latlon(np.array([0.0]), np.array([0.0]))[
        0
    ] == pytest.approx(3396190.0)
    assert cc.get_ellipsoid_radius_from_latlon(np.array([90.0]), np.array([0.0]))[
        0
    ] == pytest.approx(3376200.0)
    mid = cc.get_ellipsoid_radius_from_latlon(np.array([45.0]), np.array([0.0]))[0]
    assert 3376200.0 < mid < 3396190.0


def test_ellipsoid_xyz_norm_equals_local_radius(cc: MARS_IAU2000) -> None:
    """Checks that points on the reference ellipsoid have a cartesian norm
    equal to the local ellipsoid radius."""
    lat = np.array([0.0, 30.0, 60.0, 90.0])
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    x, y, z = cc.get_ellipsoid_xyz_from_latlon(lat, lon)
    norms = cc.to_radius_from_xyz(x, y, z)
    np.testing.assert_allclose(norms, cc.get_ellipsoid_radius_from_latlon(lat, lon))


def test_angle_between_basic_vectors() -> None:
    """Checks the vector angle on the orthogonal, identical and opposite
    cases (the examples of the function docstring)."""
    assert CoordinateConverter.angle_between((1, 0, 0), (0, 1, 0)) == pytest.approx(
        np.pi / 2
    )
    assert CoordinateConverter.angle_between((1, 0, 0), (1, 0, 0)) == pytest.approx(0.0)
    assert CoordinateConverter.angle_between((1, 0, 0), (-1, 0, 0)) == pytest.approx(
        np.pi
    )


def test_angle_between_matrices() -> None:
    """Checks the matrix variant: a constant reference vector against four
    column vectors arranged in a 2x2 grid (the grid shape must match, as in
    the simulator's per-facet usage)."""
    v1 = np.zeros((3, 2, 2))
    v1[0, :, :] = 1.0
    v2 = np.zeros((3, 2, 2))
    v2[:, 0, 0] = [1.0, 0.0, 0.0]
    v2[:, 0, 1] = [0.0, 1.0, 0.0]
    v2[:, 1, 0] = [1.0, 0.0, 0.0]
    v2[:, 1, 1] = [-1.0, 0.0, 0.0]
    angles = CoordinateConverter.angle_between(v1, v2)
    np.testing.assert_allclose(angles, [[0.0, np.pi / 2.0], [0.0, np.pi]])


def _angle_between_matrix_loop_reference(
    v1: npt.NDArray[np.float64], v2: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Reference implementation of the pre-vectorization matrix branch of
    `angle_between`: the same unit-vector normalization, then a per-element
    np.dot double loop and the same NaN fallback, including its whole-matrix
    semantics (one joint equality scalar over all NaN columns)."""
    v1_u = CoordinateConverter.unit_vector(v1)
    v2_u = CoordinateConverter.unit_vector(v2)
    angle = np.zeros([v2.shape[1], v2.shape[2]])
    for i_y in np.arange(angle.shape[0]):
        for i_x in np.arange(angle.shape[1]):
            angle[i_y, i_x] = np.dot(v1_u[:, i_y, i_x], v2_u[:, i_y, i_x])
    angle = np.arccos(angle)
    nans = np.isnan(angle)
    angle[nans] = ((v1_u[:, nans] == v2_u[:, nans]).all()) * np.pi
    return angle


def test_angle_between_matrix_bitwise_against_loop_reference() -> None:
    """Pins the vectorized matrix branch of `angle_between` bitwise against
    the pre-vectorization per-element np.dot loop, on fixed-seed random
    matrices and on crafted columns with exact-parallel and exact-antiparallel
    pairs whose normalized dot products round past +/-1 and drive arccos to
    NaN, exercising the fallback and its whole-matrix semantics."""
    rng = np.random.default_rng(2027)
    v1_random = rng.standard_normal((3, 7, 9))
    v2_random = rng.standard_normal((3, 7, 9))
    np.testing.assert_array_equal(
        CoordinateConverter.angle_between(v1_random, v2_random),
        _angle_between_matrix_loop_reference(v1_random, v2_random),
    )

    # a single parallel column rounds past +1 and falls back alone: the joint
    # equality over the NaN columns is True, so the fallback value is pi
    v1_single = np.ones((3, 1, 3))
    v2_single = np.zeros((3, 1, 3))
    v2_single[:, 0, 0] = [1.0, 1.0, 1.0]
    v2_single[:, 0, 1] = [1.0, -1.0, 0.0]
    v2_single[:, 0, 2] = [0.0, 1.0, -1.0]
    raw_single = np.einsum(
        "i...,i...->...",
        CoordinateConverter.unit_vector(v1_single),
        CoordinateConverter.unit_vector(v2_single),
        optimize=False,
    )
    assert raw_single[0, 0] > 1.0  # the parallel column reaches the fallback
    angles_single = CoordinateConverter.angle_between(v1_single, v2_single)
    np.testing.assert_array_equal(
        angles_single,
        _angle_between_matrix_loop_reference(v1_single, v2_single),
    )
    assert angles_single[0, 0] == np.pi  # exact parallel, fallback value

    # parallel and antiparallel columns round past +/-1 together: the joint
    # equality over all NaN columns is False, so by the whole-matrix quirk
    # every fallback position (the antiparallel one included) receives 0.0
    v1_mixed = np.ones((3, 1, 4))
    v2_mixed = np.zeros((3, 1, 4))
    v2_mixed[:, 0, 0] = [1.0, 1.0, 1.0]
    v2_mixed[:, 0, 1] = [-1.0, -1.0, -1.0]
    v2_mixed[:, 0, 2] = [2.0, 2.0, 2.0]
    v2_mixed[:, 0, 3] = [1.0, -1.0, 0.0]
    raw_mixed = np.einsum(
        "i...,i...->...",
        CoordinateConverter.unit_vector(v1_mixed),
        CoordinateConverter.unit_vector(v2_mixed),
        optimize=False,
    )
    assert raw_mixed[0, 0] > 1.0  # parallel column reaches the fallback
    assert raw_mixed[0, 1] < -1.0  # antiparallel column reaches the fallback
    angles_mixed = CoordinateConverter.angle_between(v1_mixed, v2_mixed)
    np.testing.assert_array_equal(
        angles_mixed,
        _angle_between_matrix_loop_reference(v1_mixed, v2_mixed),
    )
    nans_mixed = np.isnan(np.arccos(raw_mixed))
    assert (angles_mixed[nans_mixed] == 0.0).all()  # quirk: 0.0, not pi


def test_utm_roundtrip_northern_hemisphere(cc: MARS_IAU2000) -> None:
    """Checks the UTM projection and inverse on a mid-latitude point: zone
    46T for lon 90 deg, and a sub-microradian inverse roundtrip."""
    lat = np.array([45.0])
    lon = np.array([90.0])
    northing, easting, zones, zone_numbers = cc.LL_to_UTM(lat, lon)
    assert zones[0] == "46T" and zone_numbers[0] == 46
    assert 2.6e6 < northing[0] < 2.7e6  # meridian distance at Mars scale
    assert easting[0] < 500000.0  # lon 90 is west of the zone 46 origin (93 deg)
    lat_back, lon_back = cc.UTM_to_LL(northing, easting, "46T")
    assert lat_back[0] == pytest.approx(45.0, abs=1e-6)
    assert lon_back[0] == pytest.approx(90.0, abs=1e-6)


def test_utm_roundtrip_southern_hemisphere(cc: MARS_IAU2000) -> None:
    """Checks the southern-hemisphere 10000000 m northing offset and its
    inverse roundtrip."""
    lat = np.array([-45.0])
    lon = np.array([90.0])
    northing, easting, zones, _ = cc.LL_to_UTM(lat, lon)
    assert zones[0] == "46G"
    assert 7.3e6 < northing[0] < 7.4e6  # southing plus the 10000000 m offset
    lat_back, lon_back = cc.UTM_to_LL(northing, easting, "46G")
    assert lat_back[0] == pytest.approx(-45.0, abs=1e-6)
    assert lon_back[0] == pytest.approx(90.0, abs=1e-6)


def test_utm_with_given_zone_matches_auto_zone(cc: MARS_IAU2000) -> None:
    """Checks that forcing the automatically selected zone reproduces the
    automatic projection exactly."""
    lat = np.array([45.0])
    lon = np.array([90.0])
    northing_auto, easting_auto, _, zone_numbers = cc.LL_to_UTM(lat, lon)
    northing_fixed, easting_fixed = cc.LL_to_UTM_with_given_zone_number(
        lat, lon, int(zone_numbers[0])
    )
    np.testing.assert_allclose(northing_fixed, northing_auto)
    np.testing.assert_allclose(easting_fixed, easting_auto)


def test_utm_letter_designator_boundaries(cc: MARS_IAU2000) -> None:
    """Checks the UTM latitude letter designators at representative
    boundaries of the table (Mars-modified upper band included)."""
    lats = np.array([85.0, 72.0, 64.0, 45.0, 0.0, -45.0, -80.0])
    assert cc.UTMLetterDesignator(lats) == ["X", "X", "W", "T", "N", "G", "C"]


def test_get_frame_ids_from_skip_distance(cc: MARS_IAU2000) -> None:
    """Checks the frame subsampling by ground distance on a straight equatorial
    track: identifiers start at the first frame, increase strictly, and the
    per-interval frame-count estimates are consistent."""
    lat = np.zeros(12)
    lon = np.linspace(0.0, 0.55, 12)
    ids, min_n_frames, max_n_frames = cc.get_frame_ids_from_skip_distance(
        lat, lon, 6000.0
    )
    assert ids[0] == 0
    assert np.all(np.diff(ids) > 0)
    assert len(ids) >= 2
    assert 1 <= min_n_frames <= max_n_frames


def test_get_frame_ids_rejects_non_positive_skip_distance(
    cc: MARS_IAU2000,
) -> None:
    """Checks the entry validation: a zero or negative skip distance refuses
    with the house-style ValueError before any computation. A zero skip used
    to push the required-distance arange past numpy's maximum array size,
    and a negative one selected an empty required-distance array whose
    statistics reduction raised."""
    lat = np.zeros(4)
    lon = np.linspace(0.0, 0.15, 4)
    with pytest.raises(ValueError, match="ERROR: skip distance must be positive"):
        cc.get_frame_ids_from_skip_distance(lat, lon, 0.0)
    with pytest.raises(ValueError, match="ERROR: skip distance must be positive"):
        cc.get_frame_ids_from_skip_distance(lat, lon, -5.0)


def test_get_frame_ids_single_point_window_returns_empty(
    cc: MARS_IAU2000,
) -> None:
    """Checks the degenerate window with fewer than two track points: no
    per-step distance exists, so the helper returns an empty id array with
    zero interval statistics instead of reducing an empty array."""
    ids, min_n_frames, max_n_frames = cc.get_frame_ids_from_skip_distance(
        np.array([0.0]), np.array([10.0]), 1000.0
    )
    np.testing.assert_array_equal(ids, np.array([], dtype=np.int64))
    assert min_n_frames == 0.0 and max_n_frames == 0.0


def test_get_frame_ids_zero_length_track_returns_empty(
    cc: MARS_IAU2000,
) -> None:
    """Checks a window whose track points coincide (zero total track
    length): no required distance fits below the zero total, so the tree
    query returns nothing and the helper hands back an empty id array with
    zero interval statistics."""
    ids, min_n_frames, max_n_frames = cc.get_frame_ids_from_skip_distance(
        np.zeros(4), np.full(4, 10.0), 1000.0
    )
    np.testing.assert_array_equal(ids, np.array([], dtype=np.int64))
    assert min_n_frames == 0.0 and max_n_frames == 0.0


def test_get_frame_ids_single_frame_selection_reports_floor_statistics(
    cc: MARS_IAU2000,
) -> None:
    """Checks the valid single-frame selection (a skip larger than the
    window's total ground distance): one frame id and interval statistics of
    1.0 instead of a reduction over the empty interval differences — the
    value the SOFA block-partition heuristic's floor of 1 makes unobservable
    in the products."""
    ids, min_n_frames, max_n_frames = cc.get_frame_ids_from_skip_distance(
        np.array([0.0, 1.0]), np.array([0.0, 0.0]), 1.0e6
    )
    np.testing.assert_array_equal(ids, np.array([0]))
    assert min_n_frames == 1.0 and max_n_frames == 1.0
