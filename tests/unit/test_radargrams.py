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

"""Unit tests of the radargram display classes (sopa.plot_utils.radargrams)
under the Agg backend: each class produces exactly one figure for a small
synthetic image, with the histogram-based, fixed-limit and not-a-number-aware
dynamic-range variants. Pixel content is not verified.
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pytest

from sopa.plot_utils.radargrams import GroundDistancePlot, RadargramPlot, SimulationPlot


@pytest.fixture()
def image() -> npt.NDArray[np.float64]:
    """A small positive synthetic radargram."""
    rng = np.random.default_rng(41)
    return rng.lognormal(mean=1.0, sigma=1.0, size=(400, 60))


def _one_figure_produced() -> None:
    """Asserts that the plotting call left exactly one open figure."""
    assert len(plt.get_fignums()) == 1


def test_radargram_plot_default_histogram_limits(
    image: npt.NDArray[np.float64],
) -> None:
    """Checks the focused-radargram plot defaults: dB conversion with
    histogram-based limits."""
    RadargramPlot(image).plot()
    _one_figure_produced()


def test_radargram_plot_fixed_limits(image: npt.NDArray[np.float64]) -> None:
    """Checks the fixed-limit variant without dB conversion."""
    RadargramPlot(
        image, db_output=False, min_value=0.0, max_value=10.0, method=None
    ).plot()
    _one_figure_produced()


def test_simulation_plot_with_crop(image: npt.NDArray[np.float64]) -> None:
    """Checks the simulation plot with a vertical crop."""
    SimulationPlot(image, top_point=10, bottom_point=200).plot()
    _one_figure_produced()


def test_ground_distance_plot_with_nans() -> None:
    """Checks the ground-distance plot: not-a-number cells are rendered
    black through the copied colormap."""
    rng = np.random.default_rng(42)
    gd = rng.uniform(0.0, 20.0, size=(300, 40))
    gd[:100, :] = np.nan
    GroundDistancePlot(gd).plot()
    _one_figure_produced()
