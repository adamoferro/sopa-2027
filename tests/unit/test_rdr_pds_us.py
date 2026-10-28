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

"""Unit tests of the US-team PDS RDR reader (sopa.input.sharad.data.rdr.pds_us
and its companion log-message module), on a tiny synthetic dataset: the
float32 radargram load in the verified sample-major file layout (a (frames,
samples) matrix pinned with position markers), partial reads and the lenient
end-of-range handling, the file-size-derived frame count, the geometric-table
load (frame count taken from the table), orbit-data generation with computed
cartesian positions, the data-only load mode and the failure paths.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from sopa.geometry.coordinate_converter import MARS_IAU2000
from sopa.input.sharad.data.rdr import pds_us_log
from sopa.input.sharad.data.rdr.pds_us import RDR
from sopa.log.log import logger

N_FRAMES = 3
N_SAMPLES = 3600

GEOM_LAT = np.array([1.0, 1.1, 1.2])
GEOM_LON = np.array([10.0, 10.1, 10.2])
GEOM_RADIUS_KM = np.array([3396.0, 3396.1, 3396.2])


def _markers(n_frames: int = N_FRAMES) -> npt.NDArray[np.float64]:
    """Position markers of the synthetic radargram: frame f, sample r carries
    the value f * N_SAMPLES + r, unique over the whole matrix and exactly
    representable in float32."""
    return np.arange(n_frames * N_SAMPLES, dtype="float64").reshape(n_frames, N_SAMPLES)


def _write_dataset(directory: Path, n_img_frames: int = N_FRAMES) -> None:
    """Writes the synthetic radargram image and the geometric table. The .img
    follows the verified sample-major PDS layout: the marker of frame f,
    sample r is written as float32 bytes at row r, column f."""
    (directory / "s_test_rgram.img").write_bytes(
        _markers(n_img_frames).T.astype(np.float32).tobytes()
    )
    rows = [
        f"{i},{np.datetime64('2026-01-01T00:00:00.000') + i * np.timedelta64(1000, 'ms')},"
        f"{GEOM_LAT[i]},{GEOM_LON[i]},{GEOM_RADIUS_KM[i] + 0.19},{GEOM_RADIUS_KM[i]},0.5,3200.0,45.0,1.0"
        for i in range(N_FRAMES)
    ]
    (directory / "s_test_geom.tab").write_text("\n".join(rows) + "\n")


def _make_rdr(directory: Path) -> tuple[RDR, logger]:
    """Builds a US RDR reader with an in-memory logger."""
    lg = logger(verbose=False)
    return RDR(
        dataset_name="test", filename_base=str(directory / "s_test"), logger=lg
    ), lg


def test_log_message_tables_complete() -> None:
    """Checks that the reader's log-message module exposes the message keys
    used by the reader, each mapped to a non-empty string."""
    for key in [
        "orbit_file_read",
        "orbit_file_decode",
        "orbit_file_generic",
        "other_file_not-found",
        "data_frame-limits",
        "data_read",
    ]:
        assert key in pds_us_log.ERR_MESSAGES
        assert (
            isinstance(pds_us_log.ERR_MESSAGES[key], str)
            and pds_us_log.ERR_MESSAGES[key] != ""
        )


def test_load_full(tmp_path: Path) -> None:
    """Checks a successful full load against the verified PDS orientation:
    the sample-major .img (marker of frame f, sample r stored at row r,
    column f) is exposed as a (frames, samples) float64 matrix with the
    marker values preserved, and the frame count comes from the geometric
    table."""
    _write_dataset(tmp_path)
    d, _ = _make_rdr(tmp_path)
    assert d.load(mode="full") is True
    assert d.n_frames == N_FRAMES
    data = d.data
    assert data is not None
    assert data.shape == (N_FRAMES, N_SAMPLES)
    assert data.dtype == np.dtype("float64")
    np.testing.assert_array_equal(data, _markers())


def test_load_partial_frame_range(tmp_path: Path) -> None:
    """Checks a partial load: start_frame/end_frame select the
    [start_frame, end_frame) frame rows of the (frames, samples) matrix,
    values preserved."""
    _write_dataset(tmp_path)
    d, _ = _make_rdr(tmp_path)
    assert d.load(mode="data") is True
    assert d._load_rdr_data(start_frame=1, end_frame=3) is True
    data = d.data
    assert data is not None
    assert data.shape == (2, N_SAMPLES)
    np.testing.assert_array_equal(data, _markers()[1:3])


def test_load_end_beyond_available_clips(tmp_path: Path) -> None:
    """Checks the documented lenient semantics: an end_frame beyond the
    available frames loads only the available ones, without error."""
    _write_dataset(tmp_path)
    d, _ = _make_rdr(tmp_path)
    assert d.load(mode="data") is True
    assert d._load_rdr_data(start_frame=1, end_frame=1000) is True
    data = d.data
    assert data is not None
    assert data.shape == (N_FRAMES - 1, N_SAMPLES)
    np.testing.assert_array_equal(data, _markers()[1:])


def test_load_end_frame_minus_one(tmp_path: Path) -> None:
    """Checks that end_frame == -1 loads all frames from start_frame to the
    end of the file."""
    _write_dataset(tmp_path)
    d, _ = _make_rdr(tmp_path)
    assert d.load(mode="data") is True
    assert d._load_rdr_data(start_frame=2, end_frame=-1) is True
    data = d.data
    assert data is not None
    assert data.shape == (1, N_SAMPLES)
    np.testing.assert_array_equal(data, _markers()[2:])


def test_frame_count_from_file_size(tmp_path: Path) -> None:
    """Checks that the loaded frame count derives from the .img file size,
    not from the geometric table: a radargram carrying more frames than the
    table has rows loads all of its frames."""
    _write_dataset(tmp_path, n_img_frames=N_FRAMES + 2)
    d, _ = _make_rdr(tmp_path)
    assert d.load(mode="data") is True
    data = d.data
    assert data is not None
    assert data.shape == (N_FRAMES + 2, N_SAMPLES)
    np.testing.assert_array_equal(data, _markers(N_FRAMES + 2))


def test_generate_orbit_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Checks orbit-data generation: radius converted to meters, cartesian
    positions computed from lat/lon/radius, zero receive-window times and
    time offsets in seconds from the timestamps."""
    _write_dataset(tmp_path)
    d, _ = _make_rdr(tmp_path)
    assert d.load(mode="full") is True
    d.generate_orbit_data()
    orbit = d.orbit_data
    assert orbit is not None
    np.testing.assert_allclose(orbit["Radius"], GEOM_RADIUS_KM * 1000.0)
    cc = MARS_IAU2000()
    x, y, z = cc.to_xyz_from_latlon_and_radius(
        GEOM_LAT, GEOM_LON, GEOM_RADIUS_KM * 1000.0
    )
    np.testing.assert_allclose(orbit["X"], x, rtol=1e-12)
    np.testing.assert_allclose(orbit["Y"], y, rtol=1e-12)
    np.testing.assert_allclose(orbit["Z"], z, rtol=1e-12)
    np.testing.assert_array_equal(orbit["rxwin_time_s"], np.zeros(N_FRAMES))
    np.testing.assert_allclose(
        orbit["time_offset_s"], np.arange(N_FRAMES, dtype="float64")
    )
    np.testing.assert_allclose(orbit["Vtang"], np.full(N_FRAMES, 3200.0))
    assert orbit["dt"] == 3 / 80e6
    assert "WARNING" in capsys.readouterr().out


def test_generate_orbit_data_before_ancillary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Checks that orbit-data generation before the ancillary load reports an
    error and leaves the orbit data unset."""
    _write_dataset(tmp_path)
    d, _ = _make_rdr(tmp_path)
    assert d.generate_orbit_data() is None
    assert "ERROR" in capsys.readouterr().out
    assert d.orbit_data is None


def test_data_only_load(tmp_path: Path) -> None:
    """Checks the data-only mode: the radargram loads without the geometric
    table and the frame count stays unset."""
    _write_dataset(tmp_path)
    d, _ = _make_rdr(tmp_path)
    assert d.load(mode="data") is True
    assert d.n_frames == 0
    assert d.data is not None
    assert d.data.shape == (N_FRAMES, N_SAMPLES)


def test_missing_files_rejected(tmp_path: Path) -> None:
    """Checks that a base filename with no files behind it returns False."""
    d, _ = _make_rdr(tmp_path)
    assert d.load(mode="full") is False
    assert d.load(mode="data") is False


def test_frame_limits_rejected(tmp_path: Path) -> None:
    """Checks that invalid frame limits are reported and return None."""
    _write_dataset(tmp_path)
    d, lg = _make_rdr(tmp_path)
    assert d.load(mode="full") is True
    assert d._load_rdr_data(start_frame=-1) is None
    assert any("frame" in m for m in lg.LOG_MESSAGES)
