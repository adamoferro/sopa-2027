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

"""Reader for SHARAD EDR products in the CO-SHARPS format: one
tab-separated header file, one CSV aux file, one housekeeping file, one or
more orbit files and one 8-bit science data file per acquisition mode.
"""

# TODO:
# - add skip to _load_raw_data
# - handle data where more "Mode"s are present
# - modify print with logger

import datetime
import glob
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import ndimage as nd
from scipy.interpolate import interp1d

from sopa.input.data_base import RSData
from sopa.input.sharad.data.edr import cosharps_log as log_messages


class EDR(RSData):
    """CO-SHARPS EDR dataset: reads and validates the header fields, the aux file (frame time offsets and receive-window delays), the latest available orbit file and, on request, the housekeeping telemetry, then loads the 8-bit raw echo data of the first acquisition mode."""

    def __init__(
        self,
        dataset_name: str = "no name",
        pri: float = 0,
        presumming: int = 0,
        filename_base: str = "",
        use_gzip_and_iso8859_1: bool = False,
        logger: Any = None,
        read_hk_data: bool = False,
        keep_data_with_problems: bool = False,
    ) -> None:
        super().__init__(
            dataset_name, pri, presumming, filename_base, use_gzip_and_iso8859_1, logger
        )

        # input file suffixes
        self._HEADER_SUFFIX = "_Header.txt"
        self._AUX_SUFFIX = "_Aux_001.csv"
        self._HK_SUFFIX = "_HK.dat"
        self._DATA_SUFFIX = "_Mode_001.dat"
        self._ORBIT_SUFFIX_PRE = "_Orbit_"
        self._ORBIT_SUFFIX_POST = ".txt"

        # required fields and values in the ancillary data to consider the data valid
        # note: it is mandatory to set these listis and they must have the same length
        self._FIELDS_TO_SEARCH = [
            "Number_of_Modes",
            "Oper_Mode",
            "PRI_Value",
            "Presumming",
            "Resolution",
            "Phase_Comp",
            "Compression",
            "Tracking",
            "Data_Blocks",
        ]
        self._FIELDS_TO_SEARCH_DTYPE = [int, str, float, int, int, str, str, str, int]
        self._FIELDS_TO_SEARCH_CONSTRAIN = [
            1,
            "SS",
            1428,
            "",
            "",
            "None",
            "Static",
            "Disabled",
            "",
        ]

        self._keep_data_with_problems = keep_data_with_problems
        if not self._keep_data_with_problems:
            self._FIELDS_TO_SEARCH += ["Lost_Packets", "Wrong_Packets", "Anomalies"]
            self._FIELDS_TO_SEARCH_DTYPE += [int, int, int]
            self._FIELDS_TO_SEARCH_CONSTRAIN += [0, 0, 0]

        self._N_FIELDS_TO_SEARCH = len(self._FIELDS_TO_SEARCH)

        self._HEADER_FIELD_N_FRAMES = "Data_Blocks"
        self._HEADER_FIELD_PRI = "PRI_Value"
        self._HEADER_FIELD_PRESUMMING = "Presumming"
        self._READ_HK_DATA_FLAG = read_hk_data

        self._DT = 3 / 80e6
        self._DCG_DELAY = 11.98e-6  # digital chirp generator delay, used in the calculation of the rxwin opening time

        self._ancillary_already_read = False
        self._data_already_read = False

        # output
        self._header_data = None  # dictionary, general values (e.g., PRI, presumming)
        self._aux_data = None  # numpy array, two columns: "time_offset_s", time offset of each frame in s wrt the first frame; "rxwin_time_us", RX window delay in us, taking into account PRI interleaving but not DCG delay
        self._geom_data = None  # numpy array, dtype={
        #                "names": ("timestamp", "Lat", "Lon", "Radius", "Vtang", "Vradial", "X", "Y", "Z", "VX", "VY", "VZ", "Roll", "Pitch", "Yaw", "HGAin", "HGAout", "SAPXin", "SAPXout", "SAMXin", "SAMXout", "SZA", "Mag_field", "Sun_dist"),
        #                "formats": ("datetime64[ms]", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d", "d")
        #              }
        self._hk_data = None  # dictionary: "timestamp_ms", "des_temp", "des_5v", "des_12v", "des_2v5", "rx_temp", "tx_temp", "tx_lev", "tx_curr"
        #   timestamps are not related to frames

        self._data_type = "raw"

    def _load_from_file(
        self, mode: str = "full", force_reload: bool = False
    ) -> bool:  # filename_base is set, checked by super
        if self._check_file_presence(mode):
            load_ok1 = True
            load_ok2 = True
            if (mode == "full" or mode == "ancillary") and (
                (not self._ancillary_already_read) or force_reload
            ):
                load_ok1 = self._load_ancillary_data()
            if (mode == "full" or mode == "data") and (
                (not self._data_already_read) or force_reload
            ):
                load_ok2 = self._load_raw_data()
            return load_ok1 and load_ok2
        return False

    def _check_file_presence(self, mode: str = "full") -> bool:
        files_ok1 = True
        files_ok2 = True
        if mode == "full" or mode == "ancillary":
            files_ok1 = self._check_file_presence_ancillary()
        if (mode == "full" and files_ok1) or mode == "data":
            files_ok2 = self._check_file_presence_data()
        return files_ok1 and files_ok2

    def _check_file_presence_ancillary(self) -> bool:
        self._HEADER_FILENAME = self.filename_base + self._HEADER_SUFFIX
        self._AUX_FILENAME = self.filename_base + self._AUX_SUFFIX
        self._HK_FILENAME = self.filename_base + self._HK_SUFFIX
        files_ok = False
        if (
            Path(self._HEADER_FILENAME).is_file()
            and Path(self._AUX_FILENAME).is_file()
            and Path(self._HK_FILENAME).is_file()
        ):
            available_orbit_files = glob.glob(
                self.filename_base
                + self._ORBIT_SUFFIX_PRE
                + "[0-9]"
                + self._ORBIT_SUFFIX_POST
            )
            n_available_orbit_files = len(available_orbit_files)
            if n_available_orbit_files > 0:
                self._ORBIT_FILENAME = (
                    self.filename_base
                    + self._ORBIT_SUFFIX_PRE
                    + str(n_available_orbit_files)
                    + self._ORBIT_SUFFIX_POST
                )
                files_ok = True
            else:
                self.logger.log_messages(
                    log_messages.ERR_MESSAGES["orbit_file_not-found"]
                )
        else:
            self.logger.log_messages(log_messages.ERR_MESSAGES["other_file_not-found"])
        return files_ok

    def _check_file_presence_data(self) -> bool:
        self._DATA_FILENAME = self.filename_base + self._DATA_SUFFIX
        files_ok = False
        if Path(self._DATA_FILENAME).is_file():
            files_ok = True
        else:
            self.logger.log_messages(log_messages.ERR_MESSAGES["other_file_not-found"])
        return files_ok

    def _load_ancillary_data(self) -> bool:
        # header parsing
        header_data_grouped = self._read_raw_data_header()

        if header_data_grouped is not None:
            header_data, header_error_field = header_data_grouped

            if header_error_field != "":
                err_msg = f"{log_messages.ERR_MESSAGES['discarded']} {header_error_field} = {header_data[header_error_field]}"
                self.logger.log_messages(err_msg)
                return False
            else:
                self._header_data = header_data
                self.n_frames = self._header_data[self._HEADER_FIELD_N_FRAMES]
                self.pri = self._header_data[self._HEADER_FIELD_PRI] / 1e6
                self.presumming = self._header_data[self._HEADER_FIELD_PRESUMMING]
        else:
            self.logger.log_messages(log_messages.ERR_MESSAGES["header_file_generic"])
            return False

        # aux file parsing
        aux_data = self._read_raw_data_aux_file()

        # consistency check
        if aux_data is None:
            self.logger.log_messages(log_messages.ERR_MESSAGES["aux_file_generic"])
            return False
        elif len(aux_data) != header_data[self._HEADER_FIELD_N_FRAMES]:
            self.logger.log_messages(log_messages.ERR_MESSAGES["no-consistency"])
            return False
        else:
            self._aux_data = aux_data

        # orbit data parsing
        geom_data = self._read_raw_data_geom_file()
        if geom_data is None:
            self.logger.log_messages(log_messages.ERR_MESSAGES["orbit_file_generic"])
            return False
        else:
            self._geom_data = geom_data

        # HK data reading
        if self._READ_HK_DATA_FLAG:
            hk_data = self._read_raw_data_hk_file()
            if hk_data is None:
                self.logger.log_messages(log_messages.ERR_MESSAGES["hk_file_generic"])
                return False
            else:
                self._hk_data = hk_data
        self._ancillary_already_read = True
        return True

    def _read_raw_data_header(self) -> tuple[dict[str, Any], str] | None:
        try:
            with self.open_function(
                self._HEADER_FILENAME, mode="rt", encoding=self.encoding
            ) as fp:
                header_data = dict()
                error_field = ""
                i_field_to_search = 0
                line = fp.readline()
                while line and i_field_to_search < self._N_FIELDS_TO_SEARCH:
                    # print(line)
                    if line[0] != "#" and len(line) > 1:
                        unpacked_line = line.split("\t")
                        field, value = unpacked_line[
                            0:2
                        ]  # avoid problems when lines have in-line comments after field value
                        value = value.strip()
                        if field == self._FIELDS_TO_SEARCH[i_field_to_search]:
                            # print(field, "=", value)

                            value = self._FIELDS_TO_SEARCH_DTYPE[i_field_to_search](
                                value
                            )
                            header_data[self._FIELDS_TO_SEARCH[i_field_to_search]] = (
                                value
                            )
                            if (
                                self._FIELDS_TO_SEARCH_CONSTRAIN[i_field_to_search]
                                != ""
                            ):
                                if (
                                    value
                                    != self._FIELDS_TO_SEARCH_CONSTRAIN[
                                        i_field_to_search
                                    ]
                                ):
                                    error_field = self._FIELDS_TO_SEARCH[
                                        i_field_to_search
                                    ]
                                    break

                            i_field_to_search += 1
                    line = fp.readline()

                if len(header_data) != self._N_FIELDS_TO_SEARCH and error_field == "":
                    self.logger.log_messages(
                        log_messages.ERR_MESSAGES["header_file_generic"]
                    )
                else:
                    return header_data, error_field
        except OSError:
            self.logger.log_messages(log_messages.ERR_MESSAGES["header_file_read"])
        except UnicodeDecodeError:
            self.logger.log_messages(log_messages.ERR_MESSAGES["header_file_decode"])
        return None

    def _read_raw_data_aux_file(self) -> npt.NDArray | None:
        try:
            aux_data = np.loadtxt(
                self._AUX_FILENAME,
                dtype={
                    "names": ("time_offset_s", "rxwin_time_us", "not_used"),
                    "formats": ("d", "d", "i4"),
                },
                delimiter=",",
            )
            aux_data = aux_data[["time_offset_s", "rxwin_time_us"]]
            return aux_data
        except OSError:
            self.logger.log_messages(log_messages.ERR_MESSAGES["aux_file_read"])
        return None

    def _read_raw_data_geom_file(self) -> npt.NDArray | None:
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
                            "timestamp",
                            "Lat",
                            "Lon",
                            "Radius",
                            "Vtang",
                            "Vradial",
                            "X",
                            "Y",
                            "Z",
                            "VX",
                            "VY",
                            "VZ",
                            "Roll",
                            "Pitch",
                            "Yaw",
                            "HGAin",
                            "HGAout",
                            "SAPXin",
                            "SAPXout",
                            "SAMXin",
                            "SAMXout",
                            "SZA",
                            "Mag_field",
                            "Sun_dist",
                        ),
                        "formats": (
                            "datetime64[ms]",
                            "d",
                            "d",
                            "d",
                            "d",
                            "d",
                            "d",
                            "d",
                            "d",
                            "d",
                            "d",
                            "d",
                            "d",
                            "d",
                            "d",
                            "d",
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
                    delimiter="\t",
                    skiprows=7,
                    converters={0: parsetime},
                )
                return geom_data
        except OSError:
            self.logger.log_messages(log_messages.ERR_MESSAGES["orbit_file_read"])
        except UnicodeDecodeError:
            self.logger.log_messages(log_messages.ERR_MESSAGES["orbit_file_decode"])
        return None

    def _read_raw_data_hk_file(self) -> dict[str, npt.NDArray[np.float64]] | None:
        """Parses the housekeeping telemetry lines (hex-encoded engineering data of the subsurface sounding mode) into timestamped temperatures, voltages and transmitter levels."""
        date_time_ref_obj = datetime.datetime.fromtimestamp(315532800, tz=datetime.UTC)
        try:
            with open(self._HK_FILENAME) as fp:
                lines_all = fp.readlines()
                lines = [
                    line for line in lines_all if "ENG:7EE5" in line
                ]  # engineering data label + telemetry start (7E) + engineering ID (E) + subsurface sounding mode (5)
                n_lines = len(lines)
                hk_data = {
                    "timestamp_ms": np.zeros(n_lines),
                    "des_temp": np.zeros(n_lines),
                    "des_5v": np.zeros(n_lines),
                    "des_12v": np.zeros(n_lines),
                    "des_2v5": np.zeros(n_lines),
                    "rx_temp": np.zeros(n_lines),
                    "tx_temp": np.zeros(n_lines),
                    "tx_lev": np.zeros(n_lines),
                    "tx_curr": np.zeros(n_lines),
                }
                for i_line in np.arange(n_lines):
                    line = lines[i_line][4:]
                    delta_t = datetime.timedelta(
                        seconds=int(line[4:12], 16) + int(line[12:16], 16) * 2**-16
                    )
                    hk_data["timestamp_ms"][i_line] = (
                        date_time_ref_obj + delta_t
                    ).timestamp() * 1000.0
                    hk_data["des_temp"][i_line] = int(line[32:34], 16)
                    hk_data["des_5v"][i_line] = int(line[34:36], 16)
                    hk_data["des_12v"][i_line] = int(line[36:38], 16)
                    hk_data["des_2v5"][i_line] = int(line[38:40], 16)
                    hk_data["rx_temp"][i_line] = int(line[40:42], 16)
                    hk_data["tx_temp"][i_line] = int(line[42:44], 16)
                    hk_data["tx_lev"][i_line] = int(line[44:46], 16)
                    hk_data["tx_curr"][i_line] = int(line[46:48], 16)
                return hk_data
        except OSError:
            self.logger.log_messages(log_messages.ERR_MESSAGES["hk-data_read"])
        return None

    def _load_raw_data(self, start_frame: int = 0, end_frame: int = -1) -> bool | None:
        """
        Read CO-SHARPS raw data (filenames in the form OBS_xxxxxxxxx_y_Mode_00z.dat)
        - start_frame and end_frame can be in the range [0, OBS_NUMBER_OF_FRAMES-1]
            (Python array indexing convention)
        - end_frame == -1 means "all the frames starting from start_frame"
        - if end_frame is greater than the available number of frames, only the
            available data are returned without any error or warning
        - if an error occurs, the function returns False

        author: Adamo Ferro, 2019
        """
        N_SAMPLES = 3600
        frame_count = -1
        if start_frame < 0 or (end_frame != -1 and start_frame > end_frame):
            self.logger.log_messages(log_messages.ERR_MESSAGES["data_frame-limits"])
            return None
        elif end_frame != -1:
            frame_count = end_frame - start_frame
        try:
            with open(self._DATA_FILENAME, "rb") as fp:
                fp.seek(start_frame * N_SAMPLES)
                raw_data = (
                    np.fromfile(fp, dtype=np.int8, count=frame_count * N_SAMPLES)
                    .reshape((-1, 3600))
                    .T.astype("float64")
                )
                self._data = raw_data
                return True
        except OSError:
            self.logger.log_messages(log_messages.ERR_MESSAGES["data_read"])
        self._data_already_read = True
        return False

    @staticmethod
    def _fill_data_holes(
        data: npt.NDArray[np.float64], hole_max_value: float = 0
    ) -> npt.NDArray[np.float64]:
        """
        Fill holes in data using a nearest neighbour approach.
        """
        ids = nd.distance_transform_edt(
            data <= hole_max_value, return_distances=False, return_indices=True
        )
        return data[tuple(ids)]

    def generate_orbit_data(self, skip: int = 1) -> None:
        """
        Saves in self._orbit_data the following fields as a dictionary,
        interpolated to match 1:skip the frames of the raw data:
        "time_offset_s", "rxwin_time_s", "Lat", "Lon", "Radius", "Vtang", "Vradial", "X", "Y", "Z", "Roll", "Pitch", "Yaw", "SZA"
        """
        geom_fields_to_interpolate = [
            "Lat",
            "Lon",
            "Radius",
            "Vtang",
            "Vradial",
            "X",
            "Y",
            "Z",
            "Roll",
            "Pitch",
            "Yaw",
            "SZA",
        ]
        geom_fields_to_convert_to_m = ["Radius", "X", "Y", "Z"]
        if self._ancillary_already_read:
            orbit_data = dict()
            orbit_data["time_offset_s"] = self._aux_data["time_offset_s"].copy()[::skip]
            orbit_data["rxwin_time_s"] = (
                self._aux_data["rxwin_time_us"][::skip] / 1e6 - self._DCG_DELAY
            )
            if self._keep_data_with_problems:
                orbit_data["rxwin_time_s"] = self._fill_data_holes(
                    orbit_data["rxwin_time_s"]
                )

            # unwrapping of longitude data to avoid wrong interpolation on the 0-360 degrees border
            self._geom_data["Lon"] = np.rad2deg(
                np.unwrap(np.deg2rad(self._geom_data["Lon"]))
            )

            geom_data_time_offset_s_float = (
                self._geom_data["timestamp"].astype("float64") / 1000
            )
            geom_data_time_offset_s_float -= geom_data_time_offset_s_float[0]
            for field in geom_fields_to_interpolate:
                f_interp = interp1d(
                    geom_data_time_offset_s_float, self._geom_data[field], kind="cubic"
                )
                orbit_data[field] = f_interp(orbit_data["time_offset_s"])

            # re-map longitude data to the 0-360 degrees range
            self._geom_data["Lon"] = np.mod(self._geom_data["Lon"], 360)
            orbit_data["Lon"] = np.mod(orbit_data["Lon"], 360)

            for field in geom_fields_to_convert_to_m:
                orbit_data[field] *= 1000.0

            orbit_data["dt"] = self._DT

            self._orbit_data = orbit_data
            self._correct_lon()

        else:
            print("ERROR: ancillary data not yet loaded.")
            return None
