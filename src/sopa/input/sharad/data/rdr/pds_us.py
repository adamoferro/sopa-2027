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

"""Reader for SHARAD RDR products produced by the US SHARAD team: a
real-valued focused radargram (_rgram.img, 3600 samples per frame) plus a
matching geometric table (_geom.tab).
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt

from sopa._typing import MessageLogger
from sopa.exceptions import SopaDataError, SopaFileError, SopaSelectionError
from sopa.geometry.coordinate_converter import CoordinateConverter
from sopa.input.data_base import RSData
from sopa.input.sharad.data.rdr import pds_us_log as log_messages

# TODO use more the logger


class RDR(RSData):
    """PDS US RDR dataset: reads the geometric table and the focused radargram; no label file is required and the frame number is taken from the geometric table."""

    def __init__(
        self,
        dataset_name: str = "no name",
        pri: float = 0,
        presumming: int = 0,
        filename_base: str = "",
        use_gzip_and_iso8859_1: bool = False,
        logger: MessageLogger | None = None,
    ) -> None:
        super().__init__(
            dataset_name, pri, presumming, filename_base, use_gzip_and_iso8859_1, logger
        )

        # input file suffixes
        self._DATA_SUFFIX = "_rgram.img"
        self._ORBIT_SUFFIX = "_geom.tab"

        self._DT = 3 / 80e6

        self._ancillary_already_read = False
        self._data_already_read = False

        self._geom_data = None  # numpy array, dtype={
        #                "names": ("column_id", "timestamp", "Lat", "Lon", "MarsRadius", "Radius", "Vradial", "Vtang", "SZA", "PhaseDist"),
        #                "formats": ("i", "datetime64[ms]", "d", "d", "d", "d", "d", "d", "d", "d")
        #              }

        self._data_type = "focused"

    def _load_from_file(
        self, mode: str = "full", force_reload: bool = False
    ) -> bool:  # filename_base is set, checked by super
        self._check_file_presence(mode)
        load_ok1 = True
        load_ok2 = True
        if (mode == "full" or mode == "ancillary") and (
            (not self._ancillary_already_read) or force_reload
        ):
            load_ok1 = self._load_ancillary_data()
        if (mode == "full" or mode == "data") and (
            (not self._data_already_read) or force_reload
        ):
            load_ok2 = self._load_rdr_data()
        return load_ok1 and load_ok2

    def _check_file_presence(self, mode: str = "full") -> None:
        files_ok1 = True
        files_ok2 = True
        if mode == "full" or mode == "ancillary":
            self._ORBIT_FILENAME = self.filename_base + self._ORBIT_SUFFIX
            files_ok1 = Path(self._ORBIT_FILENAME).is_file()
            if not files_ok1:
                raise SopaFileError(log_messages.ERR_MESSAGES["orbit_file_not-found"])
        if (mode == "full" and files_ok1) or mode == "data":
            self._DATA_FILENAME = self.filename_base + self._DATA_SUFFIX
            files_ok2 = Path(self._DATA_FILENAME).is_file()
            if not files_ok2:
                raise SopaFileError(log_messages.ERR_MESSAGES["other_file_not-found"])

    def _load_ancillary_data(self) -> bool:
        # orbit data parsing
        self._geom_data = self._read_rdr_data_geom_file()
        self._ancillary_already_read = True
        return True

    def _read_rdr_data_geom_file(self) -> npt.NDArray[np.void]:
        def parsetime(v: str) -> np.datetime64:
            """Converts a text timestamp to numpy.datetime64."""
            return np.datetime64(v)

        try:
            with self.open_function(
                self._ORBIT_FILENAME, mode="rt", encoding=self.encoding
            ) as fp:
                geom_data = np.loadtxt(
                    fp,
                    dtype={
                        "names": (
                            "column_id",
                            "timestamp",
                            "Lat",
                            "Lon",
                            "MarsRadius",
                            "Radius",
                            "Vradial",
                            "Vtang",
                            "SZA",
                            "PhaseDist",
                        ),
                        "formats": (
                            "i",
                            "datetime64[ms]",
                            "d",
                            "d",
                            "d",
                            "d",
                            "d",
                            "d",
                            "d",
                            "d",
                        ),
                    },
                    delimiter=",",
                    skiprows=0,
                    converters={1: parsetime},
                )
                self.n_frames = geom_data.shape[0]
                return geom_data
        except OSError as err:
            raise SopaFileError(log_messages.ERR_MESSAGES["orbit_file_read"]) from err
        except UnicodeDecodeError as err:
            raise SopaDataError(log_messages.ERR_MESSAGES["orbit_file_decode"]) from err

    def _load_rdr_data(self, start_frame: int = 0, end_frame: int = -1) -> bool:
        """
        Read PDS US RDR data (filenames in the form s_xxxxxxxx_rgram.img) into
        self._data as a (frames, samples) float64 matrix. The .img file is
        stored sample-major: 3600 range-sample rows, each row holding one
        PC_REAL (float32) value per frame (label-derived ground truth for
        s_01260201: RECORD_BYTES = 18872 = 4718 x 4, LINES = 3600,
        LINE_SAMPLES = 4718, and 3600 x 4718 x 4 bytes = the exact file size,
        with the _geom.tab rows matching the 4718 frames 1:1). The file is
        therefore read whole, reshaped to (samples, frames), transposed to
        (frames, samples) and sliced afterwards; this holds one extra float32
        copy of the full radargram in memory (~68 MB for s_01260201) instead
        of seeking per request.
        - start_frame and end_frame can be in the range [0, OBS_NUMBER_OF_FRAMES-1]
            (Python array indexing convention)
        - end_frame == -1 means "all the frames starting from start_frame"
        - if end_frame is greater than the available number of frames, only the
            available data are returned without any error or warning
        - raises SopaSelectionError on invalid frame limits and SopaFileError
            when the data file cannot be read
        """
        N_SAMPLES = 3600  # range samples per frame (SHARAD series convention)
        if start_frame < 0 or (end_frame != -1 and start_frame > end_frame):
            raise SopaSelectionError(log_messages.ERR_MESSAGES["data_frame-limits"])
        try:
            with open(self._DATA_FILENAME, "rb") as fp:
                total_frames = Path(self._DATA_FILENAME).stat().st_size // (
                    N_SAMPLES * np.dtype(np.float32).itemsize
                )
                frames_major = (
                    np.fromfile(fp, dtype=np.float32).reshape((N_SAMPLES, -1)).T
                )
                if end_frame == -1:
                    end_frame = total_frames  # "all the frames from start_frame"
                self._data = frames_major[start_frame:end_frame].astype("float64")
                return True
        except OSError as err:
            raise SopaFileError(log_messages.ERR_MESSAGES["data_read"]) from err

    def generate_orbit_data(self, skip: int = 1) -> None:
        """
        Saves in self._orbit_data the following fields as a dictionary:
        "time_offset_s", "rxwin_time_s", "Lat", "Lon", "Radius", "Vtang", "Vradial", "X", "Y", "Z", "SZA"
        There is no need for interpolation, as geometric data are already matched 1:1 with RDR data.
        The field rxwin_time_s is arbitrarily set to 0, as it is not given in the data.
        Values of X, Y and Z are calculated, as they are not present in the data either.
        Raises SopaDataError when the ancillary data have not been loaded yet.
        """
        print(
            "WARNING: orbit data obtained from RDR PDS US data may be not sufficiently precise for simulation and/or focusing."
        )
        geom_fields_to_include = ["Lat", "Lon", "Radius", "Vtang", "Vradial", "SZA"]
        if self._ancillary_already_read:
            n_output_frames = int(self.n_frames / skip)
            orbit_data = dict()
            orbit_data["rxwin_time_s"] = np.zeros(n_output_frames)

            for field in geom_fields_to_include:
                orbit_data[field] = self._geom_data[field][::skip].copy()
            orbit_data["Radius"] *= 1000.0
            orbit_data["X"], orbit_data["Y"], orbit_data["Z"] = (
                CoordinateConverter.to_xyz_from_latlon_and_radius(
                    orbit_data["Lat"], orbit_data["Lon"], orbit_data["Radius"]
                )
            )
            orbit_data["time_offset_s"] = (
                self._geom_data["timestamp"][::skip].astype("float64")
                - self._geom_data["timestamp"][0].astype("float64")
            ) / 1000.0
            orbit_data["dt"] = self._DT
            self._orbit_data = orbit_data
            self._correct_lon()

        else:
            raise SopaDataError("ERROR: ancillary data not yet loaded.")
