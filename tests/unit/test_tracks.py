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
alpha-coded by incidence angle, color/alpha-coded by along-track angle).
Pixel content is not verified.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest

from sopa.dem.mola.mola import MOLA128
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
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Checks the ROI plot with an empty ROI set: it reports the missing
    ROIs and produces no figure."""
    lats, lons = track
    TrackAndROIReturnsPlotOnDEM(dem, lats, lons, ({}, {})).plot()
    assert "ERROR" in capsys.readouterr().out
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
