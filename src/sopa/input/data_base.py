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

"""Base class of the radar-sounder dataset readers: file loading, orbit data
handling and read-only access to the echo matrix.
"""

# TODO:
# - modify print with logger

import gzip

from sopa._typing import EchoMatrix, MessageLogger, OrbitData
from sopa.log.log import fake_logger


class RSData:
    """Base class of the dataset readers.

    It stores the dataset metadata (name, number of frames, PRI, presumming
    factor, base filename), selects the file-open function (plain UTF-8 or
    gzip/ISO-8859-1) and defines the loading interface that format-specific
    subclasses implement: `load` dispatches to `_load_from_file`, and
    `generate_orbit_data` builds the per-frame orbit-data dictionary. The
    echo matrix and the orbit data are exposed read-only through the `data`
    and `orbit_data` properties.
    """

    def __init__(
        self,
        dataset_name: str = "no name",
        pri: float = 0,
        presumming: int = 0,
        filename_base: str = "",
        use_gzip_and_iso8859_1: bool = False,
        logger: MessageLogger | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.n_frames = 0
        self.pri = pri  # in s
        self.presumming = presumming
        self.filename_base = filename_base
        self.use_gzip_and_iso8859_1 = use_gzip_and_iso8859_1

        self._orbit_data = None
        self._data = None
        self._data_type = ""

        # select the "open" function according to the selected encoding
        self.encoding = "UTF-8"
        self.open_function = open
        if use_gzip_and_iso8859_1:
            self.encoding = "ISO-8859-1"
            self.open_function = gzip.open

        if logger is None:
            self.logger = fake_logger()
        else:
            self.logger = logger

    def _correct_lon(self) -> None:
        if self._orbit_data is not None:
            self._orbit_data["Lon"][self._orbit_data["Lon"] < 0] = (
                360 + self._orbit_data["Lon"][self._orbit_data["Lon"] < 0]
            )

    @property
    def data(self) -> EchoMatrix | None:
        """Echo matrix (read-only); accessing it before loading prints an error and returns None."""
        if self._data is None:
            print("ERROR: data not yet read.")
            return None
        else:
            return self._data

    @data.setter
    def data(self, value: EchoMatrix) -> None:
        """Assignment is refused: it prints an error and leaves the stored echo matrix unchanged, without inspecting or validating the assigned value."""
        print("ERROR: cannot set data manually.")

    @property
    def orbit_data(self) -> OrbitData | None:
        """Orbit-data dictionary (read-only); accessing it before generation prints an error and returns None."""
        if self._orbit_data is None:
            print("ERROR: orbit data not yet generated.")
            return None
        else:
            return self._orbit_data

    @orbit_data.setter
    def orbit_data(self, value: OrbitData) -> None:
        """Assignment is refused: it prints an error and leaves the stored orbit-data dictionary unchanged, without inspecting or validating the assigned value."""
        print("ERROR: cannot set orbit data manually.")

    def __str__(self) -> str:
        output_str = f"{self.dataset_name}, n_frames={self.n_frames}, PRI={self.pri * 1e6:.1f} us, presum={self.presumming}"
        if self.filename_base != "":
            output_str += f", base filename: {self.filename_base}"
        return output_str

    def load(
        self, mode: str = "full", force_reload: bool = False
    ) -> (
        bool | None
    ):  # mode can be: "all", "ancillary", "data". Not all implementation may be able to provide different behaviors
        """Loads the dataset identified by `filename_base`, either completely (mode="full") or limiting the work to the ancillary data or to the echo data; with `force_reload` the already-read parts are read again."""
        if self.filename_base != "":
            return self._load_from_file(mode, force_reload)
        else:
            print("ERROR: dataset base filename not set.")
            return False

    def set_orbit_data_from_ext(self, ext_orbit_data: OrbitData) -> None:
        """Installs externally produced orbit data, bypassing `generate_orbit_data`."""
        self._orbit_data = ext_orbit_data

    def _load_from_file(
        self, mode: str | None, force_reload: bool
    ) -> bool | None:  # "virtual" method
        raise NotImplementedError

    def generate_orbit_data(
        self, skip: int
    ) -> None:  # generates orbit data which match 1:1 with the data
        """Builds the per-frame orbit-data dictionary of the dataset; implemented by the format-specific subclasses."""
        raise NotImplementedError
