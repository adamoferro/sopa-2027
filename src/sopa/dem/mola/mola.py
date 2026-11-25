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

"""Reader for MOLA MEGDR digital elevation maps: ENVI images of radius
values with lat/lon <-> pixel conversions, per-point radius lookup and
interpolated, upscaled sub-DEM extraction.
"""

# TODO add limit checks

import numpy as np
import numpy.typing as npt
import scipy.ndimage as snd

from sopa.dem.dem_base import DEMBase
from sopa.exceptions import SopaDataError, SopaFileError
from sopa.io_utils import envi


class MOLA(DEMBase):
    """MOLA MEGDR 1/X degrees DEM"""

    def __init__(
        self,
        dem_name: str = "MOLA",
        filename_base: str = "",
        image: npt.NDArray[np.int16] | npt.NDArray[np.float64] | None = None,
        dummy_value: int = 0,
        max_lat: float = 88,
        min_long: float = 0,
        sampling_den: float = 128,
        copy_image: bool = True,
    ) -> None:
        """Sets the DEM metadata and, when an image is provided, adopts it as the DEM image: by default the image is copied, with `copy_image=False` the instance becomes the sole owner of the passed array without copying (for freshly created images with no other owner)."""
        super().__init__(dem_name, filename_base, dummy_value)
        self._MOLA_MAX_LAT = max_lat  # degrees
        self._MOLA_MIN_LONG = min_long  # degrees
        self._MOLA_SAMPLING = 1.0 / sampling_den  # degrees per pixel

        self._MOLA_MIN_LAT = 0  # both to be calculated when loading
        self._MOLA_MAX_LONG = 0  #

        self._mask = None

        self.deg_sampling = self._MOLA_SAMPLING
        self.offset_to_be_added = 3396000  # as per MOLA labels

        self.dem = None
        if image is not None:
            self.dem = image.copy() if copy_image else image
            self._set_info()

    def __str__(self) -> str:
        return f"{self.dem_name}, base filename: {self.filename_base}, extension: lat {self._MOLA_MIN_LAT}, {self._MOLA_MAX_LAT}; long {self._MOLA_MIN_LONG}, {self._MOLA_MAX_LONG}"

    def _set_info(self) -> None:
        self.dem_shape = self.dem.shape
        self._MOLA_MIN_LAT = self._MOLA_MAX_LAT - self.dem_shape[0] * self.deg_sampling
        self._MOLA_MAX_LONG = (
            self._MOLA_MIN_LONG + self.dem_shape[1] * self.deg_sampling
        )

    def read(self) -> bool:
        """Loads the DEM image with the ENVI reader, unless an image was already provided to the constructor or loaded before; raises `SopaDataError` when the image is already loaded or the header filename is not set, and `SopaFileError` when the ENVI product cannot be read."""
        if self.dem is not None:
            raise SopaDataError(
                "WARNING: dem has been already loaded or set by init. Not loading again."
            )
        elif self.filename_base != "":
            try:
                self.dem = envi.read(self.filename_base)
            except (SopaFileError, OSError) as err:
                raise SopaFileError("ERROR: cannot read MOLA128 DEM file.") from err
            self._set_info()
            return True
        else:
            raise SopaDataError("ERROR: header filename not set.")

    def reset(self) -> None:
        """Frees the loaded DEM image."""
        del self.dem
        self.dem = None

    def get_dem_radius_from_lat_lon(
        self, lat: float | npt.NDArray[np.float64], lon: float | npt.NDArray[np.float64]
    ) -> npt.NDArray[np.int32]:
        """Returns the DEM radius (in m, MOLA offset included) at the requested lat/lon points; points outside the DEM coverage receive the dummy value. Raises `SopaDataError` when the DEM image is not loaded."""
        if self.dem is not None:
            mola_x, mola_y = self.to_XY_from_LL(lat, lon)
            return self._XY_vectors_to_radius(mola_x, mola_y)
        else:
            raise SopaDataError("ERROR: MOLA DEM not yet loaded.")

    def get_scaled_subdem_from_latlon_limits(
        self,
        lat_n: float,
        lat_s: float,
        lon_e: float,
        lon_w: float,
        scale_factor: float = 4,
    ) -> _MOLAScaledSubdemSampler:
        """Extracts the sub-image covering the given lat/lon limits (snapped to the DEM grid), smooths it and returns a lazy sampler evaluating the sub-image upscaled by `scale_factor` through the same spline interpolation as the previously materialized `scipy.ndimage.zoom` image, bit-identical over the same geographic limits; used by the simulators to sample facet geometry on a grid finer than the native DEM. Raises `SopaDataError` when the DEM image is not loaded."""
        if self.dem is not None:
            lat_n_local = np.ceil(lat_n / self.deg_sampling) * self.deg_sampling
            lat_n_local = (
                lat_n_local if lat_n_local <= self._MOLA_MAX_LAT else self._MOLA_MAX_LAT
            )
            lat_s_local = np.floor(lat_s / self.deg_sampling) * self.deg_sampling
            lat_s_local = (
                lat_s_local if lat_s_local >= self._MOLA_MIN_LAT else self._MOLA_MIN_LAT
            )

            lon_e_local = lon_e if lon_e > 0 else lon_e + 360
            lon_e_local = np.ceil(lon_e_local / self.deg_sampling) * self.deg_sampling
            lon_e_local = (
                lon_e_local
                if lon_e_local <= self._MOLA_MAX_LONG
                else self._MOLA_MAX_LONG
            )
            lon_w_local = lon_w if lon_w > 0 else lon_w + 360
            lon_w_local = np.floor(lon_w_local / self.deg_sampling) * self.deg_sampling
            lon_w_local = (
                lon_w_local
                if lon_w_local >= self._MOLA_MIN_LONG
                else self._MOLA_MIN_LONG
            )

            mola_x_min, mola_y_min = self.to_XY_from_LL(lat_n_local, lon_w_local)
            mola_x_max, mola_y_max = self.to_XY_from_LL(lat_s_local, lon_e_local)
            subdem_image = self.XY_ranges_to_radius(
                mola_x_min[0],
                mola_x_max[0],
                mola_y_min[0],
                mola_y_max[0],
                add_offset=False,
            )
            smoothed_subdem_image = snd.gaussian_filter(subdem_image.astype("float"), 1)
            return _MOLAScaledSubdemSampler(
                smoothed_subdem_image,
                max_lat=self._MOLA_MAX_LAT - mola_y_min * self.deg_sampling,
                min_long=self._MOLA_MIN_LONG + mola_x_min * self.deg_sampling,
                sampling_den=1.0 / self.deg_sampling * scale_factor,
                scale_factor=scale_factor,
                dummy_value=self.dummy_value,
            )

        else:
            raise SopaDataError("ERROR: MOLA DEM not yet loaded.")

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
    ):
        """Converts lat/lon (degrees) to DEM pixel coordinates, by default rounded to integer pixel indices, with `return_as_float` scalar inputs returned as scalar floats and array inputs as a float array. Out-of-range points — a longitude exactly on the right edge included — are flagged in the internal mask and clamped to the upper-left pixel (to be replaced by the dummy value by the callers); non-finite coordinates take the same masked path without reaching the integer cast."""
        mola_x = (long - self._MOLA_MIN_LONG) / self.deg_sampling
        mola_y = (self._MOLA_MAX_LAT - lat) / self.deg_sampling

        scalar_input = np.ndim(mola_x) == 0
        if not return_as_float:
            # non-finite coordinates must never reach the int cast: casting a
            # non-finite float is platform-dependent (ADR 0037, INT64_MIN on
            # x86-64 vs 0 on aarch64), so they are masked to the
            # out-of-coverage sentinel on the float array first
            mola_x = np.round(np.where(np.isfinite(mola_x), mola_x, -1.0)).astype("int")
            mola_y = np.round(np.where(np.isfinite(mola_y), mola_y, -1.0)).astype("int")

        if scalar_input:
            mola_x = np.array([mola_x])
            mola_y = np.array([mola_y])

        self._mask = (
            (mola_x < 0)
            | (mola_y < 0)
            | (mola_x >= self.dem_shape[1])
            | (mola_y >= self.dem_shape[0])
        )

        mola_x[mola_x < 0] = 0
        mola_y[mola_y < 0] = 0
        mola_x[mola_x >= self.dem_shape[1]] = 0
        mola_y[mola_y >= self.dem_shape[0]] = 0

        if scalar_input and return_as_float:
            return (float(mola_x[0]), float(mola_y[0]))
        return (mola_x, mola_y)

    def to_LL_from_XY(self, x: float, y: float) -> tuple[float, float]:
        """Converts DEM pixel coordinates back to lat/lon (degrees)."""
        lat = -y * self.deg_sampling + self._MOLA_MAX_LAT
        lon = x * self.deg_sampling + self._MOLA_MIN_LONG
        return (lat, lon)

    def XY_ranges_to_radius(
        self, x_min: int, x_max: int, y_min: int, y_max: int, add_offset: bool = True
    ) -> npt.NDArray[np.int32]:
        """Returns the DEM sub-image for the given pixel ranges as radii in m, adding the MOLA radius offset (3396000 m) on request. The int32 upcast is required by numpy>=2 (NEP 50), which no longer promotes int16+int silently (see the in-code comment)."""
        # astype(int32): numpy>=2 (NEP 50) raises OverflowError when adding the
        # MOLA radius offset (3396000 m) to an int16 DEM array (old numpy
        # silently promoted the result type)
        output_dem = (
            self.dem[y_min : y_max + 1, x_min : x_max + 1].copy().astype("int32")
            + int(add_offset) * self.offset_to_be_added
        )
        return output_dem

    def _XY_vectors_to_radius(
        self,
        x_vect: npt.NDArray[np.int64],
        y_vect: npt.NDArray[np.int64],
        add_offset: bool = True,
    ) -> npt.NDArray[np.int32]:
        """Per-point variant of `XY_ranges_to_radius` for coordinate vectors: out-of-DEM points flagged by the last `to_XY_from_LL` call receive the dummy value."""
        # get vector containing MOLA radius corresponding to each frame to be processed (in the same order)
        output_dem = (
            self.dem[y_vect, x_vect].copy().astype("int32")
            + int(add_offset) * self.offset_to_be_added
        )
        output_dem[self._mask] = self.dummy_value
        return output_dem


class MOLA128(MOLA):
    """MOLA MEGDR 1/128 degrees DEM"""

    def __init__(
        self,
        dem_name: str = "MOLA128",
        filename_base: str = "",
        image: npt.NDArray[np.int16] | npt.NDArray[np.float64] | None = None,
        dummy_value: int = 0,
        max_lat: float = 88,
        min_long: float = 0,
    ) -> None:
        super().__init__(
            dem_name,
            filename_base,
            image,
            dummy_value,
            max_lat=max_lat,
            min_long=min_long,
            sampling_den=128,
        )


class _MOLAScaledSubdemSampler:
    """Lazy radius sampler for the smoothed/upscaled sub-DEM extracted by
    `MOLA.get_scaled_subdem_from_latlon_limits`: it holds the smoothed native
    sub-image and evaluates, on every lookup, the same order-2 spline
    interpolation the previously materialized `scipy.ndimage.zoom` upscale was
    built from, at the exact zoom lattice coordinates
    (`index * (n_in - 1) / (n_out - 1)` in float64), so the returned radii are
    bit-identical to reading the materialized image. It exposes the radius
    lookup and the dummy value of the MOLA view it replaces."""

    def __init__(
        self,
        smoothed_subdem: npt.NDArray[np.float64],
        max_lat: float | npt.NDArray[np.float64],
        min_long: float | npt.NDArray[np.float64],
        sampling_den: float,
        scale_factor: float,
        dummy_value: int,
    ) -> None:
        """Stores the smoothed native sub-image with the geographic bookkeeping of the scaled MOLA view it replaces: `max_lat`/`min_long` are the snapped limits of the native sub-image, `sampling_den` the sampling denominator of the scaled view and `scale_factor` the upscale factor."""
        self._smoothed_subdem = smoothed_subdem

        n_in_lat, n_in_long = smoothed_subdem.shape
        n_out_lat = int(round(n_in_lat * scale_factor))
        n_out_long = int(round(n_in_long * scale_factor))
        self.dem_shape = (n_out_lat, n_out_long)
        # zoom lattice factors, bit-equal to scipy.ndimage.zoom's internal
        # np.divide(n_in - 1, n_out - 1), which defaults to 1 where n_out == 1
        self._zoom_lat = (
            (n_in_lat - 1) / (n_out_lat - 1) if (n_out_lat - 1) != 0 else 1.0
        )
        self._zoom_long = (
            (n_in_long - 1) / (n_out_long - 1) if (n_out_long - 1) != 0 else 1.0
        )

        self._MOLA_MAX_LAT = max_lat  # degrees
        self._MOLA_MIN_LONG = min_long  # degrees
        self.deg_sampling = 1.0 / sampling_den  # degrees per pixel
        self.offset_to_be_added = 3396000  # as per MOLA labels
        self.dummy_value = dummy_value
        self._mask: npt.NDArray[np.bool_] | None = None

    def get_dem_radius_from_lat_lon(
        self, lat: float | npt.NDArray[np.float64], lon: float | npt.NDArray[np.float64]
    ) -> npt.NDArray[np.int32]:
        """Returns the sub-DEM radius (in m, MOLA offset included) at the requested lat/lon points on the scaled grid; points outside the scaled sub-DEM coverage — non-finite coordinates included — receive the dummy value."""
        mola_x = (lon - self._MOLA_MIN_LONG) / self.deg_sampling
        mola_y = (self._MOLA_MAX_LAT - lat) / self.deg_sampling

        # non-finite coordinates must never reach the int cast: casting a
        # non-finite float is platform-dependent (ADR 0037, INT64_MIN on
        # x86-64 vs 0 on aarch64), so they are masked to the
        # out-of-coverage sentinel on the float array first
        mola_x = np.round(np.where(np.isfinite(mola_x), mola_x, -1.0)).astype("int")
        mola_y = np.round(np.where(np.isfinite(mola_y), mola_y, -1.0)).astype("int")

        if len(mola_x.shape) == 0:
            mola_x = np.array([mola_x])
            mola_y = np.array([mola_y])

        self._mask = (
            (mola_x < 0)
            | (mola_y < 0)
            | (mola_x >= self.dem_shape[1])
            | (mola_y >= self.dem_shape[0])
        )

        mola_x[mola_x < 0] = 0
        mola_y[mola_y < 0] = 0
        mola_x[mola_x >= self.dem_shape[1]] = 0
        mola_y[mola_y >= self.dem_shape[0]] = 0

        # the same order-2 spline evaluation the materialized
        # snd.zoom(..., scale_factor, order=2) image was built with, at the
        # zoom lattice coordinates (identical order/mode/cval/prefilter);
        # integer indices are exactly representable in float64, so the
        # coordinate products are bit-equal to zoom's per-pixel computation
        radius = snd.map_coordinates(
            self._smoothed_subdem,
            np.stack(
                (
                    mola_y.astype(np.float64) * self._zoom_lat,
                    mola_x.astype(np.float64) * self._zoom_long,
                )
            ),
            order=2,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )
        # replicate the MOLA view's _XY_vectors_to_radius: truncate toward
        # zero into int32, add the MOLA offset, mask out-of-coverage points
        output_dem = radius.astype("int32") + self.offset_to_be_added
        output_dem[self._mask] = self.dummy_value
        return output_dem
