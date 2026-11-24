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

"""Matplotlib display of radargrams (focused, simulated or ground-distance
products) with optional dB conversion and histogram- or percentile-based
dynamic range selection.
"""

import matplotlib
import numpy as np
import numpy.typing as npt
from matplotlib import pyplot as pp
from matplotlib.backends import backend_registry


class _BaseRadargramPlot:
    """Common implementation of the radargram plots: prepares the image (optional dB conversion), the display limits (fixed, from the data, or from the histogram of the central 0.5-99.5% of the values) and the vertical crop."""

    def __init__(
        self,
        image: npt.NDArray[np.float64],
        db_output: bool = False,
        min_value: float = -1,
        max_value: float = -1,
        method: str | None = None,
        top_point: int = 0,
        bottom_point: int = -1,
        title: str = "",
    ) -> None:
        self._db_output = db_output
        if self._db_output:
            self._image = 10.0 * np.log10(image)
        else:
            self._image = image

        if method == "hist" and (min_value == -1 or max_value == -1):
            hist_min, hist_max = self._hist_limits()

        if min_value == -1:
            if method != "hist":
                self._min_value = np.min(self._image)
            else:
                self._min_value = hist_min
        else:
            self._min_value = min_value
        if max_value == -1:
            if method != "hist":
                self._max_value = np.max(self._image)
            else:
                self._max_value = hist_max
        else:
            self._max_value = max_value
        self._top_point = top_point
        if bottom_point == -1:
            self._bottom_point = self._image.shape[0]
        else:
            self._bottom_point = bottom_point
        self._title = title

    def _hist_limits(self, ll: float = 0.5, ul: float = 0.995) -> tuple[float, float]:
        lower_perc_limit = ll
        upper_perc_limit = ul
        h, be = np.histogram(self._image, bins=1000, density=True)
        bw = be[1] - be[0]
        h *= bw
        hs = np.cumsum(h)
        h_ok_ids = np.where((hs >= lower_perc_limit) & (hs <= upper_perc_limit))[0]
        return be[h_ok_ids[0]], be[h_ok_ids[-1]]

    def plot(self, cmap: str | matplotlib.colors.Colormap = "gray") -> None:
        """Shows the prepared image with the given colormap and limits; the
        blocking show call runs only when the active backend has a GUI
        framework (non-interactive backends skip it without a warning)."""
        pp.figure(figsize=(12, 6))
        pp.imshow(
            self._image[self._top_point : self._bottom_point, :],
            cmap=cmap,
            vmin=self._min_value,
            vmax=self._max_value,
            interpolation="none",
        )
        pp.title(self._title, fontweight="bold")
        pp.ylabel("Samples", fontweight="bold")
        pp.xlabel("Frames", fontweight="bold")
        pp.colorbar()
        _, gui_framework = backend_registry.resolve_backend(matplotlib.get_backend())
        if gui_framework is not None:
            pp.show()


class RadargramPlot(_BaseRadargramPlot):
    """Displays a focused radargram, converted to dB and with histogram-based limits by default."""

    def __init__(
        self,
        image: npt.NDArray[np.float64],
        db_output: bool = True,
        min_value: float = -1,
        max_value: float = -1,
        method: str = "hist",
        top_point: int = 0,
        bottom_point: int = -1,
        title: str = "",
    ) -> None:
        super().__init__(
            image,
            db_output,
            min_value,
            max_value,
            method,
            top_point,
            bottom_point,
            title,
        )


class SimulationPlot(_BaseRadargramPlot):
    """Displays a simulated radargram (linear power, no dB conversion)."""

    def __init__(
        self,
        image: npt.NDArray[np.float64],
        min_value: float = -1,
        max_value: float = -1,
        method: str = "hist",
        top_point: int = 0,
        bottom_point: int = -1,
        title: str = "",
    ) -> None:
        super().__init__(
            image, False, min_value, max_value, method, top_point, bottom_point, title
        )


class GroundDistancePlot(_BaseRadargramPlot):
    """Displays the minimum/maximum ground-distance images produced by the simulators (no dB conversion, default range 0-20 m); not-a-number values are rendered black."""

    def __init__(
        self,
        image: npt.NDArray[np.float64],
        min_value: float = 0,
        max_value: float = 20,
        top_point: int = 0,
        bottom_point: int = -1,
        title: str = "",
    ) -> None:
        super().__init__(
            image, False, min_value, max_value, None, top_point, bottom_point, title
        )

    def plot(self) -> None:
        """Same display as the base class, with a colormap whose not-a-number values are rendered black."""
        # 2026 port: matplotlib.cm.get_cmap was removed
        nan_cmap = matplotlib.colormaps["gist_rainbow"].with_extremes(bad="black")
        super().plot(cmap=nan_cmap)
