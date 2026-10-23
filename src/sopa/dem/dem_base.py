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

import numpy.typing as npt


class DEMBase():
    """docstring for DEMBase."""

    def __init__(self, dem_name: str = "no name", filename_base: str = "", dummy_value: int = 0) -> None:
        self.dem_name = dem_name
        self.filename_base = filename_base
        self.dummy_value = dummy_value
        self.dem = None
        self.deg_sampling = 0
        self.offset_to_be_added = 0

    def __str__(self) -> str:
        return self.dem_name + ", base filename: " + self.filename_base

    def read(self) -> bool:
        raise NotImplementedError

    def get_dem_radius_from_lat_lon(self, lat: npt.ArrayLike, lon: npt.ArrayLike) -> npt.NDArray | None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError
