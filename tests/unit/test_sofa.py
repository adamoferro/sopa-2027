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

"""Unit tests of the SOFA focuser (sopa.processing.focusers.sofa): the raw-data
type validation, the derived focusing parameters (coherence time, Doppler
window, Gaussian-filter multilook decomposition, weighting windows), the
frame-range validation of focus(), a small end-to-end focus run on a tiny
synthetic raw dataset, and the byte-identity of the chunked aperture loop in
_block_focus across chunk sizes.
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest

from sopa.geometry.coordinate_converter import MARS_IAU2000
from sopa.input.data_base import RSData
from sopa.input.sharad.instrument import SHARAD
from sopa.processing.focusers import sofa as sofa_module
from sopa.processing.focusers.sofa import SOFA
from sopa.processing.parameters import FocuserParameters

N_FRAMES = 20
N_INPUT_SAMPLES = 900
PRI = 1428e-6
PRESUMMING = 16
N_HALF_APERTURE = 3

# Fresh-interpreter probe for the peak-RSS budget test: _block_focus at the
# heavy configuration's sizes (4096-point FFT, 1517 aperture frames, 1374
# output samples, 4 focused frames), default aperture-FFT chunk. The peak is
# read from /proc/self/status VmHWM -- the current mm's high-water mark, which
# exec resets -- and NOT from getrusage ru_maxrss: that one is task-lifetime
# and a forked child inherits the forking parent's resident pages between
# fork and exec (measured: a child of a ~700 MiB parent reports ru_maxrss
# ~712 MiB against its own VmHWM of ~12 MiB). Printed as the last line for
# the parent test to parse.
_HEAVY_PROBE_SCRIPT = """\
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sopa.processing.focusers.sofa import SOFA

rng = np.random.default_rng(20261026)
size_fft1 = 4096
n_output_samples = 1374
output_samples = np.arange(n_output_samples)
n_half_aperture_frames = 758
n_aperture_frames = 2 * n_half_aperture_frames + 1
n_block_columns = 1530
n_input_samples = 512
frames_to_be_focused_in_this_block = np.array([758, 762, 766, 770], dtype="int64")
lmbda_min = 299792458.0 / (20.0e6 + 5.0e6 / 2.0)

input_theta = rng.uniform(0.0, 2.0 * np.pi, n_block_columns)
input_phi = rng.uniform(0.1, np.pi - 0.1, n_block_columns)
spacecraft_radius = 3.6e6
input_scr_data = np.full(n_block_columns, spacecraft_radius)
input_scx_data = spacecraft_radius * np.cos(input_theta) * np.sin(input_phi)
input_scy_data = spacecraft_radius * np.sin(input_theta) * np.sin(input_phi)
input_scz_data = spacecraft_radius * np.cos(input_phi)
dem_radius_profile = np.full(n_block_columns, 3396000, dtype="int32")
input_rxwin_data_t = 5.0e-3 + rng.uniform(-1.0e-7, 1.0e-7, n_block_columns)
delay_start_shifts = rng.uniform(0.0, 1.0e-7, n_block_columns)
chirp_fft_conj = rng.standard_normal(size_fft1) + 1j * rng.standard_normal(size_fft1)
fftfreqs = np.fft.fftfreq(size_fft1, 7.5e-9)
t_frame_base = 7.5e-9 * output_samples
data_obj = SimpleNamespace(data=rng.standard_normal((n_input_samples, n_block_columns)))

output_image = SOFA._block_focus(
    0,
    0,
    n_block_columns,
    frames_to_be_focused_in_this_block,
    0,
    size_fft1,
    SOFA._C,
    input_rxwin_data_t,
    chirp_fft_conj,
    n_output_samples,
    output_samples,
    input_scr_data,
    input_theta,
    input_phi,
    n_half_aperture_frames,
    n_aperture_frames,
    True,
    input_scx_data,
    input_scy_data,
    input_scz_data,
    True,
    lmbda_min,
    np.ones([n_output_samples, 1]),
    True,
    0,
    [],
    [],
    0,
    0,
    t_frame_base,
    dem_radius_profile,
    fftfreqs,
    delay_start_shifts,
    data_obj,
)
assert output_image.shape == (n_output_samples, len(frames_to_be_focused_in_this_block))
hwm_kib = next(
    int(line.split()[1])
    for line in Path("/proc/self/status").read_text().splitlines()
    if line.startswith("VmHWM:")
)
print(f"PEAK_RSS_KIB={hwm_kib}")
"""

# Budget ceiling for the probe above. Measured ~428 MiB (VmHWM) in a fresh
# interpreter on this machine (2026-10-26); 480 MiB adds ~12% headroom while
# staying far below the naive-lift failure class: batching all 1517 aperture
# frames at once instead of chunks of 256 adds two 4096 x 1517 complex128
# buffers (~192 MiB total) in place of the 256-wide ones, putting the child
# process around ~590 MiB.
_PEAK_RSS_BUDGET_MIB = 480


class _StubRawData(RSData):
    """Minimal in-memory raw-data holder satisfying the SOFA constructor and
    the focus-time data access."""

    def __init__(
        self,
        data: npt.NDArray[np.float64],
        orbit_data: dict[str, npt.NDArray[np.float64] | float],
    ) -> None:
        super().__init__(dataset_name="stub", pri=PRI, presumming=PRESUMMING)
        self._data_type = "raw"
        self._data = data
        self.set_orbit_data_from_ext(orbit_data)

    def _load_from_file(
        self, mode: str | None = None, force_reload: bool = False
    ) -> bool:
        """Never used by the focuser in these tests."""
        return True

    def generate_orbit_data(self, skip: int = 1) -> None:
        """Never used by the focuser in these tests."""
        raise NotImplementedError


def _orbit_data() -> dict[str, npt.NDArray[np.float64] | float]:
    """Builds a synthetic orbit-data dictionary on an equatorial track with
    constant radius and receive-window time."""
    cc = MARS_IAU2000()
    lat = np.zeros(N_FRAMES)
    lon = np.linspace(0.0, 0.38, N_FRAMES)
    radius = np.full(N_FRAMES, 3646190.0)
    x, y, z = cc.to_xyz_from_latlon_and_radius(lat, lon, radius)
    return {
        "Lat": lat,
        "Lon": lon,
        "X": x,
        "Y": y,
        "Z": z,
        "Radius": radius,
        "rxwin_time_s": np.full(N_FRAMES, 5e-3),
        "dt": 37.5e-9,
    }


@pytest.fixture()
def stub_data() -> _StubRawData:
    """A small synthetic raw dataset (random real echoes) with its orbit data."""
    rng = np.random.default_rng(31)
    data = rng.standard_normal((N_INPUT_SAMPLES, N_FRAMES))
    return _StubRawData(data, _orbit_data())


def test_constructor_rejects_non_raw_data() -> None:
    """Checks that SOFA refuses non-raw input data."""
    holder = _StubRawData(np.zeros((8, N_FRAMES)), _orbit_data())
    holder._data_type = "focused"
    with pytest.raises(ValueError):
        SOFA(
            SHARAD(),
            MARS_IAU2000(),
            np.full(N_FRAMES, 3396000, dtype="int32"),
            holder,
            FocuserParameters(),
        )


def test_constructor_derived_parameters(stub_data: _StubRawData) -> None:
    """Checks the derived focusing parameters: PRI-based coherence time and
    spacing, aperture length, and the Doppler half window clamped to zero for
    a short aperture."""
    pars = FocuserParameters(
        frame_start=0, frame_end=1, frame_half_aperture=N_HALF_APERTURE
    )
    f = SOFA(
        SHARAD(),
        MARS_IAU2000(),
        np.full(N_FRAMES, 3396000, dtype="int32"),
        stub_data,
        pars,
    )
    assert f._pri_data == pytest.approx(PRI * PRESUMMING)
    assert f._n_aperture_frames == 2 * N_HALF_APERTURE + 1
    assert f._coherence_time == pytest.approx((2 * N_HALF_APERTURE) * PRI * PRESUMMING)
    assert f._du_global == pytest.approx(PRI * PRESUMMING * 3397.0)
    assert f._doppler_half_window_width == 0
    assert f._gf_power_coefficients == []
    assert f._gf_std_devs == []
    assert f._SIZE_RANGE_FFT == 4096


def test_constructor_gaussian_filter_decomposition(stub_data: _StubRawData) -> None:
    """Checks the Gaussian multilook decomposition setup for three filters:
    four power coefficients summing to one, starting with alpha, and the
    linearly increasing standard deviations up to the rounded maximum."""
    pars = FocuserParameters(
        frame_start=0,
        frame_end=1,
        frame_half_aperture=100,
        gauss_rec_n_filters=3,
        gauss_rec_alpha=0.66,
    )
    f = SOFA(
        SHARAD(),
        MARS_IAU2000(),
        np.full(N_FRAMES, 3396000, dtype="int32"),
        stub_data,
        pars,
    )
    coefficients = f._gf_power_coefficients
    assert len(coefficients) == 4
    assert coefficients[0] == pytest.approx(0.66)
    assert sum(coefficients) == pytest.approx(1.0, abs=1e-12)
    std_devs = f._gf_std_devs
    assert len(std_devs) == 3
    assert std_devs[0] < std_devs[1] < std_devs[2]
    assert std_devs[-1] == pytest.approx(1.8)
    assert f._doppler_half_window_width == 1


def test_constructor_weighting_windows(stub_data: _StubRawData) -> None:
    """Checks the range and azimuth weighting windows built from the named
    window functions, including the kaiser variant with its beta."""
    pars = FocuserParameters(
        frame_start=0,
        frame_end=1,
        frame_half_aperture=N_HALF_APERTURE,
        range_weighting_function="kaiser",
        range_weighting_kaiser_beta=2.5,
        azimuth_weighting_function="kaiser",
        azimuth_weighting_kaiser_beta=1.5,
    )
    f = SOFA(
        SHARAD(),
        MARS_IAU2000(),
        np.full(N_FRAMES, 3396000, dtype="int32"),
        stub_data,
        pars,
    )
    np.testing.assert_allclose(
        f._range_weighting_window_original, np.kaiser(4096 / 2.0, 2.5)
    )
    np.testing.assert_allclose(
        f._azimuth_weighting_window_original, np.kaiser(2 * N_HALF_APERTURE + 1, 1.5)
    )


def test_constructor_weighting_windows_none(stub_data: _StubRawData) -> None:
    """Checks the weighting-window fallbacks for the "none" functions: an
    all-ones float64 range window over half the range FFT length and the
    scalar azimuth window that focus() broadcasts over the output samples."""
    pars = FocuserParameters(
        frame_start=0,
        frame_end=1,
        frame_half_aperture=N_HALF_APERTURE,
        range_weighting_function="none",
        azimuth_weighting_function="none",
    )
    f = SOFA(
        SHARAD(),
        MARS_IAU2000(),
        np.full(N_FRAMES, 3396000, dtype="int32"),
        stub_data,
        pars,
    )
    np.testing.assert_array_equal(f._range_weighting_window_original, np.ones(2048))
    assert f._range_weighting_window_original.dtype == np.float64
    assert f._azimuth_weighting_window_original == 1.0


def test_constructor_n_processes_clamping(
    stub_data: _StubRawData, capsys: pytest.CaptureFixture[str]
) -> None:
    """Checks the n_processes interface since the parallel reactivation:
    values below one fall back to one, one is kept, values above one are
    preserved, and the constructor prints no notice in any case (the former
    parallelization-is-not-implemented notice is gone)."""
    pars = FocuserParameters(
        frame_start=0, frame_end=1, frame_half_aperture=N_HALF_APERTURE
    )
    for requested, expected in ((-3, 1), (0, 1), (1, 1), (4, 4)):
        f = SOFA(
            SHARAD(),
            MARS_IAU2000(),
            np.full(N_FRAMES, 3396000, dtype="int32"),
            stub_data,
            pars,
            n_processes=requested,
        )
        assert f.n_processes == expected
        assert capsys.readouterr().out == ""


def test_focus_rejects_end_after_data(
    stub_data: _StubRawData, capsys: pytest.CaptureFixture[str]
) -> None:
    """Checks that an end frame at or after the input length stops the focus
    before any processing."""
    pars = FocuserParameters(
        frame_start=3, frame_end=N_FRAMES, frame_half_aperture=N_HALF_APERTURE
    )
    f = SOFA(
        SHARAD(),
        MARS_IAU2000(),
        np.full(N_FRAMES, 3396000, dtype="int32"),
        stub_data,
        pars,
    )
    assert f.focus() is None
    assert "ERROR" in capsys.readouterr().out


def test_focus_rejects_empty_range(
    stub_data: _StubRawData, capsys: pytest.CaptureFixture[str]
) -> None:
    """Checks that a frame range ending before its (clamped) start stops the
    focus before any processing."""
    pars = FocuserParameters(
        frame_start=5, frame_end=4, frame_half_aperture=N_HALF_APERTURE
    )
    f = SOFA(
        SHARAD(),
        MARS_IAU2000(),
        np.full(N_FRAMES, 3396000, dtype="int32"),
        stub_data,
        pars,
    )
    assert f.focus() is None
    assert "ERROR" in capsys.readouterr().out


def test_focus_tiny_run(stub_data: _StubRawData) -> None:
    """Checks a complete focus run on the tiny dataset: one output column per
    focused frame (arange(3, 10) = 7), all values finite and non-negative,
    and positive total power."""
    pars = FocuserParameters(
        frame_start=3, frame_end=10, frame_half_aperture=N_HALF_APERTURE, squint_angle=0
    )
    f = SOFA(
        SHARAD(),
        MARS_IAU2000(),
        np.full(N_FRAMES, 3396000, dtype="int32"),
        stub_data,
        pars,
        n_processes=1,
    )
    output = f.focus()
    assert output is not None
    assert output.shape == (667 * 2, 7)
    assert np.all(np.isfinite(output))
    assert np.all(output >= 0)
    assert np.sum(output) > 0


def test_focus_tiny_run_unweighted(stub_data: _StubRawData) -> None:
    """Checks the same focus run as test_focus_tiny_run with both weighting
    functions set to "none": the run completes, one output column per focused
    frame (arange(3, 10) = 7), all values finite and non-negative, and
    positive total power."""
    pars = FocuserParameters(
        frame_start=3,
        frame_end=10,
        frame_half_aperture=N_HALF_APERTURE,
        squint_angle=0,
        range_weighting_function="none",
        azimuth_weighting_function="none",
    )
    f = SOFA(
        SHARAD(),
        MARS_IAU2000(),
        np.full(N_FRAMES, 3396000, dtype="int32"),
        stub_data,
        pars,
        n_processes=1,
    )
    output = f.focus()
    assert output is not None
    assert output.shape == (667 * 2, 7)
    assert np.all(np.isfinite(output))
    assert np.all(output >= 0)
    assert np.sum(output) > 0


def test_focus_progress_callback_multi_block(stub_data: _StubRawData) -> None:
    """Checks the focus() progress callback on a serial multi-block run
    (read_block_size_max 3 caps the 14 processed frames to blocks of 3, which
    yields one focused frame per block, 7 blocks for arange(3, 10)): the
    focus()-passed callback (which takes precedence over the constructor's)
    receives the initial partition report at 0.0 and one report per completed
    block at (blocks completed)/7, a monotonically non-decreasing sequence
    ending at 1.0, with non-empty notes, while the constructor's callback
    receives nothing."""
    pars = FocuserParameters(
        frame_start=3,
        frame_end=10,
        frame_half_aperture=N_HALF_APERTURE,
        squint_angle=0,
    )
    reports: list[tuple[float, str]] = []
    unused_reports: list[tuple[float, str]] = []
    f = SOFA(
        SHARAD(),
        MARS_IAU2000(),
        np.full(N_FRAMES, 3396000, dtype="int32"),
        stub_data,
        pars,
        n_processes=1,
        read_block_size_max=3,
        progress_callback=lambda fraction, note: unused_reports.append(
            (fraction, note)
        ),
    )
    output = f.focus(
        progress_callback=lambda fraction, note: reports.append((fraction, note))
    )
    assert output is not None
    assert output.shape == (667 * 2, 7)
    fractions = [fraction for fraction, _ in reports]
    assert fractions == [0.0] + [i / 7 for i in range(1, 8)]
    assert all(note for _, note in reports)
    assert unused_reports == []


def test_focus_progress_callback_output_identical(stub_data: _StubRawData) -> None:
    """Checks that the progress callback is call-only on the same tiny run as
    test_focus_tiny_run (default block size, one block): the run driven with a
    constructor-provided recording callback returns an output array equal
    (np.array_equal) to the run without any callback, and the callback
    received exactly the partition report at 0.0 and the block report at
    1.0."""
    pars = FocuserParameters(
        frame_start=3,
        frame_end=10,
        frame_half_aperture=N_HALF_APERTURE,
        squint_angle=0,
    )
    output_without_callback = SOFA(
        SHARAD(),
        MARS_IAU2000(),
        np.full(N_FRAMES, 3396000, dtype="int32"),
        stub_data,
        pars,
        n_processes=1,
    ).focus()
    reports: list[tuple[float, str]] = []
    output_with_callback = SOFA(
        SHARAD(),
        MARS_IAU2000(),
        np.full(N_FRAMES, 3396000, dtype="int32"),
        stub_data,
        pars,
        n_processes=1,
        progress_callback=lambda fraction, note: reports.append((fraction, note)),
    ).focus()
    assert output_without_callback is not None
    assert output_with_callback is not None
    assert np.array_equal(output_with_callback, output_without_callback)
    assert [fraction for fraction, _ in reports] == [0.0, 1.0]


def test_block_focus_chunked_matches_per_column_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Checks that the chunked aperture loop of _block_focus returns
    byte-identical output images for chunk sizes 1, 32 and 256 on four
    parameter combinations covering the range-migration, Doppler-correction,
    Gaussian-filter and azimuth-weighting branches; chunk size 1 is the
    per-column reference of the former loop."""
    rng = np.random.default_rng(20261026)
    size_fft1 = 4096
    n_output_samples = 200
    output_samples = np.arange(n_output_samples)
    n_half_aperture_frames = 60
    n_aperture_frames = 2 * n_half_aperture_frames + 1
    n_block_columns = 170
    n_input_samples = 512
    frames_to_be_focused_in_this_block = np.array([60, 65, 70, 75], dtype="int64")
    lmbda_min = 299792458.0 / (20.0e6 + 5.0e6 / 2.0)

    input_theta = rng.uniform(0.0, 2.0 * np.pi, n_block_columns)
    input_phi = rng.uniform(0.1, np.pi - 0.1, n_block_columns)
    spacecraft_radius = 3.6e6
    input_scr_data = np.full(n_block_columns, spacecraft_radius)
    input_scx_data = spacecraft_radius * np.cos(input_theta) * np.sin(input_phi)
    input_scy_data = spacecraft_radius * np.sin(input_theta) * np.sin(input_phi)
    input_scz_data = spacecraft_radius * np.cos(input_phi)
    dem_radius_profile = np.full(n_block_columns, 3396000, dtype="int32")
    input_rxwin_data_t = 5.0e-3 + rng.uniform(-1.0e-7, 1.0e-7, n_block_columns)
    delay_start_shifts = rng.uniform(0.0, 1.0e-7, n_block_columns)
    chirp_fft_conj = rng.standard_normal(size_fft1) + 1j * rng.standard_normal(
        size_fft1
    )
    fftfreqs = np.fft.fftfreq(size_fft1, 7.5e-9)
    t_frame_base = 7.5e-9 * output_samples
    data_obj = SimpleNamespace(
        data=rng.standard_normal((n_input_samples, n_block_columns))
    )

    # (correct_range_migration, use_doppler_correction, n_gaussian_filters,
    #  std_devs, gf_power_coefficients, doppler_half_window_width,
    #  azimuth_weighting_windows, azimuth_weighting_is_unity)
    parameter_combinations = (
        (True, True, 0, [], [], 0, np.ones([n_output_samples, 1]), True),
        (False, False, 0, [], [], 0, np.ones([n_output_samples, 1]), True),
        (
            True,
            False,
            2,
            [9.0, 18.0],
            [0.66, 0.2244, 0.1156],
            5,
            np.ones([n_output_samples, 1]),
            True,
        ),
        (
            False,
            True,
            0,
            [],
            [],
            5,
            rng.uniform(0.5, 1.5, [n_output_samples, 1]),
            False,
        ),
    )

    for (
        correct_range_migration,
        use_doppler_correction,
        n_gaussian_filters,
        std_devs,
        gf_power_coefficients,
        doppler_half_window_width,
        azimuth_weighting_windows,
        azimuth_weighting_is_unity,
    ) in parameter_combinations:

        def run_block_focus(chunk_size: int) -> npt.NDArray[np.float64]:
            """Runs _block_focus once with the given aperture-FFT chunk size."""
            monkeypatch.setattr(sofa_module, "_APERTURE_FFT_CHUNK", chunk_size)
            return SOFA._block_focus(
                0,  # block_id
                0,  # input_data_frame_start
                n_block_columns,  # input_data_frame_end
                frames_to_be_focused_in_this_block,
                0,  # frames_to_be_processed_start
                size_fft1,
                SOFA._C,
                input_rxwin_data_t,
                chirp_fft_conj,
                n_output_samples,
                output_samples,
                input_scr_data,
                input_theta,
                input_phi,
                n_half_aperture_frames,
                n_aperture_frames,
                correct_range_migration,
                input_scx_data,
                input_scy_data,
                input_scz_data,
                use_doppler_correction,
                lmbda_min,
                azimuth_weighting_windows,
                azimuth_weighting_is_unity,
                n_gaussian_filters,
                std_devs,
                gf_power_coefficients,
                doppler_half_window_width,
                0,  # squint_offset_approx_frames
                t_frame_base,
                dem_radius_profile,
                fftfreqs,
                delay_start_shifts,
                data_obj,
            )

        reference_bytes = run_block_focus(1).tobytes()
        for chunk_size in (32, 256):
            chunked_bytes = run_block_focus(chunk_size).tobytes()
            assert chunked_bytes == reference_bytes, (
                f"chunk size {chunk_size} differs from the per-column reference "
                f"(range migration = {correct_range_migration}, "
                f"Doppler correction = {use_doppler_correction})"
            )


def test_block_focus_peak_rss_budget() -> None:
    """Checks that _block_focus at the heavy configuration's sizes stays under
    a fixed peak-RSS ceiling when run with the module-default aperture-FFT
    chunk, in a fresh interpreter so the reading is not polluted by the pytest
    process. This pins the P3 memory discipline: chunked inner-loop batching
    must not regress toward the naive full-width lift, which exceeds the
    ceiling. Measured value, headroom rationale, and why the probe reads
    /proc/self/status VmHWM instead of getrusage ru_maxrss: see the
    _HEAVY_PROBE_SCRIPT and _PEAK_RSS_BUDGET_MIB constants."""
    src_dir = Path(__file__).resolve().parents[2] / "src"
    env = {**os.environ, "MPLBACKEND": "Agg"}
    env["PYTHONPATH"] = (
        str(src_dir)
        if not env.get("PYTHONPATH")
        else f"{src_dir}{os.pathsep}{env['PYTHONPATH']}"
    )
    probe = subprocess.run(
        [sys.executable, "-c", _HEAVY_PROBE_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
    )
    assert probe.returncode == 0, f"probe failed:\n{probe.stderr}"

    matches = re.findall(r"PEAK_RSS_KIB=(\d+)", probe.stdout)
    assert matches, f"probe printed no PEAK_RSS_KIB marker:\n{probe.stdout}"
    peak_rss_mib = int(matches[-1]) // 1024

    assert peak_rss_mib <= _PEAK_RSS_BUDGET_MIB, (
        f"heavy-shape _block_focus peak RSS {peak_rss_mib} MiB exceeds the "
        f"{_PEAK_RSS_BUDGET_MIB} MiB budget (chunked inner-loop memory "
        f"discipline violated; VmHWM high-water mark of the probe process)"
    )


def test_worker_budget_formula_boundaries() -> None:
    """Checks the memory-based worker-budget formula boundaries: with too
    little memory to admit even one budgeted worker the result is 1, with
    ample memory it is the CPU count, and in between it is the floored
    affordable worker count of the module constants."""
    assert sofa_module._max_memory_limited_pool_workers(4, 1) == 1
    assert sofa_module._max_memory_limited_pool_workers(4, 2**40) == 4
    assert (
        sofa_module._max_memory_limited_pool_workers(
            4,
            sofa_module._WORKER_BASE_RESERVE_BYTES
            + 2 * sofa_module._WORKER_MEMORY_BUDGET_BYTES,
        )
        == 2
    )
    assert (
        sofa_module._max_memory_limited_pool_workers(
            8,
            sofa_module._WORKER_BASE_RESERVE_BYTES
            + 10 * sofa_module._WORKER_MEMORY_BUDGET_BYTES,
        )
        == 8
    )
    assert sofa_module._max_memory_limited_pool_workers(1, 2**40) == 1


def test_available_memory_bytes_reading() -> None:
    """Checks that the MemAvailable reading either fails cleanly (None, e.g.
    off Linux) or returns a positive byte count."""
    available = sofa_module._available_memory_bytes()
    assert available is None or available > 0


def test_focus_tiny_run_no_doppler_correction(stub_data: _StubRawData) -> None:
    """Checks the same focus run as test_focus_tiny_run with the Doppler phase
    compensation disabled: the run completes, one output column per focused
    frame (arange(3, 10) = 7), all values finite and non-negative, and
    positive total power."""
    pars = FocuserParameters(
        frame_start=3,
        frame_end=10,
        frame_half_aperture=N_HALF_APERTURE,
        squint_angle=0,
        compensate_doppler_phase=False,
    )
    f = SOFA(
        SHARAD(),
        MARS_IAU2000(),
        np.full(N_FRAMES, 3396000, dtype="int32"),
        stub_data,
        pars,
        n_processes=1,
    )
    output = f.focus()
    assert output is not None
    assert output.shape == (667 * 2, 7)
    assert np.all(np.isfinite(output))
    assert np.all(output >= 0)
    assert np.sum(output) > 0
