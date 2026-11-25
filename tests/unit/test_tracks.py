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

"""Unit tests of the ground-track display classes (sopa.plot_utils.tracks)
under the Agg backend, on a small synthetic DEM: the plain track plot, the
first-return overlay, and the ROI variants (no ROIs, plain points,
alpha-coded by incidence angle, color/alpha-coded by along-track angle),
plus the tick-label regression for degenerate track windows (fewer than 9
DEM pixels wide, or zero-extent).
Pixel content is not verified.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest

from sopa.dem.mola.mola import MOLA128
from sopa.exceptions import SopaSelectionError
from sopa.plot_utils.tracks import (
    TrackAndFirstReturnsPlotOnDEM,
    TrackAndROIReturnsPlotOnDEM,
    TrackPlotOnDEM,
)

N_POINTS = 30


@pytest.fixture()
def dem() -> MOLA128:
    """A small synthetic DEM covering lat 2..0.4375, lon 0..1 deg."""
    rng = np.random.default_rng(51)
    values = rng.integers(-100, 100, size=(200, 128)).astype("int16")
    return MOLA128(image=values, max_lat=2.0)


@pytest.fixture()
def track() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A synthetic short spacecraft track inside the DEM coverage."""
    rng = np.random.default_rng(0)
    return np.linspace(1.1, 1.3, N_POINTS) + rng.uniform(
        -1e-4, 1e-4, N_POINTS
    ), np.linspace(0.3, 0.7, N_POINTS)


@pytest.fixture()
def narrow_track() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A synthetic track spanning 6.4 DEM pixels per axis, below the
    9-pixel window the tick-label step division requires."""
    return np.linspace(1.10, 1.15, N_POINTS), np.linspace(0.30, 0.35, N_POINTS)


@pytest.fixture()
def single_point_track() -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """A single-point track landing exactly on DEM pixel (x=64, y=64):
    a zero-extent window."""
    return np.array([1.5]), np.array([0.5])


def test_track_plot(
    dem: MOLA128, track: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
) -> None:
    """Checks the plain track plot: exactly one figure is produced."""
    lats, lons = track
    TrackPlotOnDEM(dem, lats, lons).plot()
    assert len(plt.get_fignums()) == 1


def test_track_plot_auto_show_off(
    dem: MOLA128, track: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
) -> None:
    """Checks that show() without explicit request and without auto_show
    leaves the state unchanged (no error, figure stays open for the caller)."""
    lats, lons = track
    plotter = TrackPlotOnDEM(dem, lats, lons)
    plotter.show()
    assert len(plt.get_fignums()) == 0


def test_track_plot_narrow_window(
    dem: MOLA128,
    narrow_track: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> None:
    """Checks that a track window narrower than 9 DEM pixels no longer
    divides by zero in the tick-label construction and instead labels
    every window pixel (one tick per DEM pixel, starting at the window
    start)."""
    lats, lons = narrow_track
    TrackPlotOnDEM(dem, lats, lons).plot()
    assert len(plt.get_fignums()) == 1
    ax = plt.gcf().axes[0]
    np.testing.assert_array_equal(ax.get_xticks(), np.arange(7))
    np.testing.assert_array_equal(ax.get_yticks(), np.arange(8))


def test_track_plot_single_point_window(
    dem: MOLA128,
    single_point_track: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> None:
    """Checks that a zero-extent (single-point) track window no longer
    divides by zero in the tick-label construction and simply produces
    no tick labels."""
    lats, lons = single_point_track
    TrackPlotOnDEM(dem, lats, lons).plot()
    assert len(plt.get_fignums()) == 1
    ax = plt.gcf().axes[0]
    assert ax.get_xticks().size == 0
    assert ax.get_yticks().size == 0


def test_track_plot_tick_labels_unchanged_on_working_window(
    dem: MOLA128, track: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
) -> None:
    """Locks the exact tick positions and label strings of a working
    (9 pixels or wider) window, which the degenerate-window step clamp
    must leave untouched."""
    lats, lons = track
    TrackPlotOnDEM(dem, lats, lons).plot()
    assert len(plt.get_fignums()) == 1
    fig = plt.gcf()
    fig.canvas.draw()
    ax = fig.axes[0]
    assert ax.get_xticks().tolist() == [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    assert [t.get_text() for t in ax.get_xticklabels()] == [
        "0.3",
        "0.34",
        "0.38",
        "0.41",
        "0.45",
        "0.49",
        "0.53",
        "0.57",
        "0.61",
        "0.65",
        "0.69",
    ]
    assert ax.get_yticks().tolist() == [0, 3, 6, 9, 12, 15, 18, 21, 24]
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        "1.3",
        "1.28",
        "1.26",
        "1.23",
        "1.21",
        "1.19",
        "1.16",
        "1.14",
        "1.12",
    ]


def test_track_and_first_returns_plot(
    dem: MOLA128, track: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
) -> None:
    """Checks the first-return overlay plot: exactly one figure is produced."""
    lats, lons = track
    TrackAndFirstReturnsPlotOnDEM(dem, lats, lons, lats + 0.005, lons + 0.005).plot()
    assert len(plt.get_fignums()) == 1


def test_roi_plot_without_rois(
    dem: MOLA128,
    track: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]],
) -> None:
    """Checks the ROI plot with an empty ROI set: it refuses with
    SopaSelectionError and produces no figure."""
    lats, lons = track
    with pytest.raises(SopaSelectionError, match="ERROR: no ROIs to plot."):
        TrackAndROIReturnsPlotOnDEM(dem, lats, lons, ({}, {})).plot()
    assert len(plt.get_fignums()) == 0


def test_roi_plot_plain_points(
    dem: MOLA128, track: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
) -> None:
    """Checks the ROI plot with plain per-ROI scatter (no incidence coding)."""
    lats, lons = track
    roi_pos: tuple[dict[int, npt.NDArray[np.float64]], ...] = (
        {1: lats + 0.002},
        {1: lons},
    )
    TrackAndROIReturnsPlotOnDEM(dem, lats, lons, roi_pos).plot()
    assert len(plt.get_fignums()) == 1


def test_roi_plot_alpha_coded(
    dem: MOLA128, track: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
) -> None:
    """Checks the ROI plot alpha-coded by the global incidence angle (four
    position components)."""
    lats, lons = track
    ias = np.linspace(0.0, 0.01, N_POINTS)
    roi_pos: tuple[dict[int, npt.NDArray[np.float64]], ...] = (
        {1: lats + 0.002},
        {1: lons},
        {1: ias},
        {1: np.zeros(N_POINTS)},
    )
    TrackAndROIReturnsPlotOnDEM(dem, lats, lons, roi_pos).plot()
    assert len(plt.get_fignums()) == 1


def test_roi_plot_alt_angle_coded(
    dem: MOLA128, track: tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
) -> None:
    """Checks the ROI plot color/alpha-coded by the along-track incidence
    angle (four position components, plot_alt_angle=True): the alternate-angle
    branch completes and exactly one figure is produced."""
    lats, lons = track
    alt_ias = np.linspace(-0.005, 0.005, N_POINTS)
    roi_pos: tuple[dict[int, npt.NDArray[np.float64]], ...] = (
        {1: lats + 0.002},
        {1: lons},
        {1: np.zeros(N_POINTS)},
        {1: alt_ias},
    )
    TrackAndROIReturnsPlotOnDEM(dem, lats, lons, roi_pos, plot_alt_angle=True).plot()
    assert len(plt.get_fignums()) == 1
