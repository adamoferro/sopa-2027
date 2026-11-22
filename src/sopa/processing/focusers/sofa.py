#!/usr/bin/python3

#   Copyright 2016-2021 Adamo Ferro
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

"""SOFA (SHARAD Focuser): frequency-domain synthetic-aperture focuser for
SHARAD raw echo data.
"""

# TODO
# - estimate platform velocity from data instead of using reference velocity
# - calculate _SIZE_RANGE_FFT automatically? (next pow2?)
# - estimate of average number of frames corresponding to the defined squint offset using only the frames required for processing and not the whole track

import concurrent.futures
import ctypes
import multiprocessing
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d

from sopa.processing.parameters import half_aperture_frames_from_coherence_time
from sopa.processing.utils import ProcessingUtils

# Number of aperture frames per batched ifft call in _block_focus. The batch
# axis is bounded by this constant, never by block or aperture size, so the
# chunk buffers stay fixed-size regardless of the processing scale.
_APERTURE_FFT_CHUNK = 256

# Worker budget of the parallel block focus (phase-6 P5): the pool is capped
# by the CPU count and by the memory the machine can still admit,
# n = min(cpu_count, max(1, floor((MemAvailable - reserve) / per_worker))).
# The reserve (2 GiB) covers parent-side footprint growth that remains
# possible after the MemAvailable reading (the audited serial in-parent focus
# growth is ~0.9 GiB: 5942 -> 6845 MiB peak on the heavy configuration); the
# per-worker budget (1 GiB) carries ~80% headroom over the audited marginal
# cost of one worker (~0.55 GiB: ~8 GiB total peak at 4 workers over the same
# ~5.9 GiB base, of which ~225 MiB are the chunked focus buffers and the rest
# copy-on-write-faulted private pages and pickled block parameters). On the
# reference machine (16 GiB RAM, MemAvailable ~7 GiB at focus time on the
# heavy configuration) this admits 4 workers.
_WORKER_BASE_RESERVE_BYTES = 2 * 1024**3
_WORKER_MEMORY_BUDGET_BYTES = 1024**3

# Thread-control entry points of the (vendored) OpenBLAS libraries. numpy and
# scipy load differently named scipy-openblas builds whose symbols carry
# scipy_/size-suffix decorations; the first symbol found in a library wins.
_OPENBLAS_SET_NUM_THREADS_SYMBOLS = (
    "scipy_openblas_set_num_threads64_",
    "scipy_openblas_set_num_threads_64_",
    "scipy_openblas_set_num_threads",
    "scipy_openblas_set_num_threads_",
    "openblas_set_num_threads64_",
    "openblas_set_num_threads",
    "goto_set_num_threads",
)

# Raw-data object handed to the parallel block focus: set by SOFA.focus before
# the fork-based pool is created and resolved by the workers through fork
# copy-on-write, so the multi-GB raw array is never pickled through submit().
_FORK_BLOCK_FOCUS_DATA_OBJ: Any | None = None


def _available_memory_bytes() -> int | None:
    """Reads MemAvailable from /proc/meminfo and returns it in bytes, or None when the reading is not possible (the kernel-documented estimate of the memory available for starting new work without swapping)."""
    try:
        meminfo = Path("/proc/meminfo").read_text()
    except OSError:
        return None
    for line in meminfo.splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) >= 2:
                try:
                    return int(fields[1]) * 1024
                except ValueError:
                    return None
    return None


def _max_memory_limited_pool_workers(
    cpu_count: int, available_memory_bytes: int
) -> int:
    """Computes the worker-count ceiling for the parallel block focus from the CPU count and the available memory: the affordable worker count floor((available memory - reserve) / per-worker budget) over the module constants _WORKER_BASE_RESERVE_BYTES and _WORKER_MEMORY_BUDGET_BYTES, at least 1 and at most the CPU count."""
    affordable_workers = (
        available_memory_bytes - _WORKER_BASE_RESERVE_BYTES
    ) // _WORKER_MEMORY_BUDGET_BYTES
    return min(cpu_count, max(1, affordable_workers))


def _pin_openblas_threads(n_threads: int) -> None:
    """Pins the OpenBLAS thread pool of every OpenBLAS library loaded in this process (numpy's and scipy's vendored scipy-openblas builds among them) to n_threads via ctypes on the already-mapped shared objects. Each forked pool worker calls this before any compute: the OpenBLAS thread pool inherited from the forking parent must not spin up in the child, and single-threaded workers avoid oversubscribing the machine's cores."""
    try:
        maps = Path("/proc/self/maps").read_text()
    except OSError:
        return
    for library_path in sorted(set(re.findall(r"\S+openblas\S*", maps))):
        try:
            library = ctypes.CDLL(library_path)
        except OSError:
            continue
        for symbol_name in _OPENBLAS_SET_NUM_THREADS_SYMBOLS:
            set_num_threads = getattr(library, symbol_name, None)
            if set_num_threads is not None:
                set_num_threads.argtypes = [ctypes.c_int]
                set_num_threads(n_threads)
                break


def _pool_worker_initializer() -> None:
    """Pool initializer of the parallel block focus: pins the worker's OpenBLAS thread pools to one thread before any compute runs in the worker."""
    _pin_openblas_threads(1)


def _pool_block_focus(block_id: int, *input_parameters: Any) -> npt.NDArray[np.float64]:
    """Worker-side entry point of the parallel block focus: resolves the raw-data object from the fork-inherited module reference (never from pickled arguments) and runs the same _block_focus operation sequence as the serial path, with the worker-side block prints suppressed because the parent prints the per-block progress."""
    if _FORK_BLOCK_FOCUS_DATA_OBJ is None:
        raise RuntimeError(
            "ERROR: the fork-inherited raw-data reference is not set; "
            "_pool_block_focus must run in a worker forked after SOFA.focus "
            "set it."
        )
    return SOFA._block_focus(
        block_id,
        *input_parameters,
        _FORK_BLOCK_FOCUS_DATA_OBJ,
        verbose=False,
    )


class SOFA:
    """Synthetic-aperture focuser for SHARAD raw data.

    For every frame to focus it range-compresses the echoes of the synthetic
    aperture (matched filter in the frequency domain), applies the optional
    range-migration correction (computed against the DEM radius profile) and
    the optional Doppler phase compensation, Fourier-transforms the aperture
    in azimuth, and reduces the azimuth power spectrum to one output sample
    by multilooking. For each output sample the multilook Doppler window is
    centered on the maximum of the power spectrum found within a search
    window around the aperture center: the search half-width is a parameter
    of the focusing code (currently zero, i.e. fixed zero-Doppler centering),
    and wider values de-center the window so that it follows the Doppler
    maximum. The multilook power can be decomposed into a weighted
    combination of Gaussian-filtered spectra (gauss_rec_n_filters /
    gauss_rec_alpha). The squint angle offsets the aperture with respect to
    the frame to focus (positive: look ahead, negative: look behind). Frames
    are processed in blocks; with `n_processes` = 1 (the default) the blocks
    are processed serially, with values above 1 they are dispatched to a
    fork-based process pool whose worker count is capped by the CPU count and
    by a memory-based worker budget, the workers sharing the raw data with
    the parent copy-on-write. An optional progress callback (constructor
    keyword progress_callback, or a per-call override as the focus()
    argument) receives coarse progress reports during focus(); it is
    call-only — it never influences the numerical processing — and its
    absence leaves the code path byte-identical.
    """

    _C = 299792458.0

    def __init__(
        self,
        instr_obj: Any,
        geom_obj: Any,
        dem_radius_profile: npt.NDArray[np.int32],
        data_obj: Any,
        param_obj: Any,
        n_processes: int = 1,
        read_block_size_max: int = 5000,
        debug_mode: bool = False,
        *,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> None:
        self.geom_obj = geom_obj
        self.data_obj = data_obj

        if self.data_obj._data_type != "raw":
            raise ValueError("ERROR: input data are not raw.")

        self._dem_radius_profile = dem_radius_profile

        self._frequency = instr_obj.frequency
        self._bandwidth = instr_obj.bandwidth
        self._fast_time_sampling_frequency = instr_obj.fast_time_sampling_frequency
        self._lmbda_min = self._C / (
            instr_obj.frequency + instr_obj.bandwidth / 2.0
        )  # wavelength at highest frequency
        self._dt = 1.0 / instr_obj.fast_time_sampling_frequency
        self._chirp_duration = instr_obj.chirp_duration
        self._n_input_frames = data_obj.data.shape[1]
        self._n_input_samples = data_obj.data.shape[0]

        self._pri = data_obj.pri
        self._presumming = data_obj.presumming

        self._frames_to_be_focused_start = param_obj.frame_start
        self._frames_to_be_focused_end = param_obj.frame_end
        if param_obj.meters_skip is not None:
            self._skip_meters = param_obj.meters_skip
            self._skip_frames = 0
        else:
            self._skip_frames = param_obj.frame_skip
            self._skip_meters = None
        self._n_half_aperture_frames = param_obj.frame_half_aperture

        self._SIZE_RANGE_FFT = 4096
        self._N_OUTPUT_SAMPLES = 667 * 2

        self._CORRECT_RANGE_MIGRATION = param_obj.correct_range_migration
        self._COMPENSATE_DOPPLER_PHASE = param_obj.compensate_doppler_phase

        self._n_gaussian_filters = param_obj.gauss_rec_n_filters
        self._gaussian_alpha = param_obj.gauss_rec_alpha

        self._squint_angle_deg = (
            param_obj.squint_angle
        )  # =0, look nadir; >0, look ahead; <0 look behind

        self._read_block_size_max = read_block_size_max
        self._debug_mode = debug_mode

        # ----------- CALCULATED PARAMETERS -----------
        self._pri_data = self._pri * self._presumming
        if param_obj.coherence_time_s is not None:
            self._n_half_aperture_frames = half_aperture_frames_from_coherence_time(
                param_obj.coherence_time_s, self._pri_data
            )
            self._frames_to_be_focused_start = max(
                [self._frames_to_be_focused_start, self._n_half_aperture_frames]
            )
        self._n_aperture_frames = (
            2 * self._n_half_aperture_frames + 1
        )  # center sample is included and in the middle of aperture
        self._coherence_time = (self._n_aperture_frames - 1) * self._pri_data
        self._du_global = self._pri_data * instr_obj.ref_v_platform

        # TODO
        self._focusing_ML_doppler_half_bandwidth = (
            param_obj.multilooking_doppler_bandwidth / 2.0
        )
        self._doppler_half_window_width = (
            np.round(
                self._focusing_ML_doppler_half_bandwidth * self._coherence_time
            ).astype("int")
            - 1
        )
        self._doppler_half_window_width = np.max(
            [0, self._doppler_half_window_width]
        )  # avoid negative window widths

        if self._debug_mode:
            print("n_input_frames =", self._n_input_frames)
            print("coherence time =", self._coherence_time)
            print("Doppler half window width =", self._doppler_half_window_width)
            print("du_global =", self._du_global)

        self._squint_angle_rad = self._squint_angle_deg * np.pi / 180.0

        # Parallel block focusing is implemented again since phase-6 P5: a
        # fork-based process pool over the focus blocks (see focus()).
        # Values < 1 fall back to 1; 1 keeps the original serial path, which
        # is the exact pre-P5 processing code.
        if n_processes < 1:
            n_processes = 1

        self.n_processes = n_processes

        self.progress_callback = progress_callback

        # gaussian filter parameters
        #   power coefficients
        self._gf_power_coefficients = list()
        if self._n_gaussian_filters > 0:
            self._gf_power_coefficients.append(self._gaussian_alpha)
            for i_gf in np.arange(self._n_gaussian_filters - 1):
                self._gf_power_coefficients.append(
                    (1.0 - np.sum(self._gf_power_coefficients)) * self._gaussian_alpha
                )
            self._gf_power_coefficients.append(1 - np.sum(self._gf_power_coefficients))

        #   std devs
        self._gf_std_devs = list()
        gf_std_dev_max = (
            np.round(
                (self._focusing_ML_doppler_half_bandwidth * self._coherence_time) * 10.0
            )
        ) / 10.0
        self._gf_std_devs = [
            (i_gf + 1) * gf_std_dev_max / self._n_gaussian_filters
            for i_gf in np.arange(self._n_gaussian_filters)
        ]

        if debug_mode:
            print("Gaussian filters number:", self._n_gaussian_filters)
            print(
                "   power coefficients =",
                self._gf_power_coefficients,
                "(including no filter coeff.)",
            )
            print("   std devs =", self._gf_std_devs)
        else:
            if debug_mode:
                print("NO Gaussian filtering.")

        self._range_weighting_window_original = np.ones(int(self._SIZE_RANGE_FFT / 2.0))
        rwfs = param_obj.range_weighting_function
        if rwfs != "none":
            if rwfs == "kaiser":
                self._range_weighting_window_original = np.kaiser(
                    self._SIZE_RANGE_FFT / 2.0, param_obj.range_weighting_kaiser_beta
                )
            else:
                self._range_weighting_window_original = eval(
                    "np." + rwfs + "(self._SIZE_RANGE_FFT/2.)"
                )

        self._azimuth_weighting_window_original = 1.0
        awfs = param_obj.azimuth_weighting_function
        if awfs != "none":
            if awfs == "kaiser":
                self._azimuth_weighting_window_original = np.kaiser(
                    self._n_aperture_frames, param_obj.azimuth_weighting_kaiser_beta
                )
            else:
                self._azimuth_weighting_window_original = eval(
                    "np." + awfs + "(self._n_aperture_frames)"
                )

    def focus(
        self, *, progress_callback: Callable[[float, str], None] | None = None
    ) -> npt.NDArray[np.float64] | None:
        """Focuses the selected frames and returns the output image (one column per focused frame); validates the frame range against the input data and computes the actual frame selection (skip in frames or meters, aperture extent and squint offset), returning None on invalid ranges and when the computed frame selection is empty. With n_processes above 1 the per-block focus is dispatched to a fork-based process pool (worker count capped by the CPU count and the memory-based worker budget) and the output is assembled in block order, which leaves it independent of the worker count; the block size is derived per requested process, so a smaller effective worker count only queues more blocks.

        The optional keyword-only progress_callback (a callable taking a
        fraction in [0, 1] and a short note string; when omitted, the
        constructor's callback applies, and this argument takes precedence
        over it when both are given) receives coarse progress reports: one
        call with fraction 0.0 once the frame selection and block partition
        are computed, then one call per completed focus block with the
        fraction (blocks completed)/(total blocks), reaching 1.0 with the
        last block. The fractions are monotonically non-decreasing; the
        callback is invoked only in the parent process — the pool workers
        never call it, and it is never passed across the fork boundary. When
        focus() returns None on an invalid frame range the callback is not
        called. Exceptions raised by the callback propagate to the caller and
        abort the focus run. The callback is call-only: it never influences
        the numerical processing, and without one (the default) the code path
        is byte-identical to the previous behavior.
        """
        if progress_callback is None:
            progress_callback = self.progress_callback

        if self._frames_to_be_focused_end >= self._n_input_frames:
            print("ERROR: last frame to be focused after the end of the input data.")
            return None

        # Estimation of average number of frames corresponding to the defined squint offset
        input_sc_alt = self.data_obj.orbit_data["Radius"] - self._dem_radius_profile
        mean_altitude = np.average(input_sc_alt)
        squint_offset_approx_distance = (
            -(mean_altitude * np.tan(self._squint_angle_rad)) * 1.01
        )  # TODO *** 1.01 because (abs) result is tipically underestimated
        squint_offset_approx_frames = int(
            np.round(squint_offset_approx_distance / self._du_global)
        )

        # calculate actual start and end frames according to aperture and squint angle
        if self._frames_to_be_focused_start == -1:
            self._frames_to_be_focused_start = np.max(
                [self._n_half_aperture_frames - squint_offset_approx_frames + 1, 0]
            )

        if self._frames_to_be_focused_end == -1:
            self._frames_to_be_focused_end = np.min(
                [
                    self._n_input_frames
                    - self._n_half_aperture_frames
                    - 1
                    - squint_offset_approx_frames,
                    self._n_input_frames - 1,
                ]
            )

        if self._frames_to_be_focused_end < self._frames_to_be_focused_start:
            print("ERROR: last frame to be focused before first frame to be focused.")
            return None

        if self._debug_mode:
            print("SC mean altitude =", mean_altitude)
            print("squint_offset_approx_distance =", squint_offset_approx_distance)
            print("squint_offset_approx_frames =", squint_offset_approx_frames)

        # calculation of range of frames to be focused
        if self._skip_frames != 0:  # when the skip is expressed in frames
            frames_to_be_focused = np.arange(
                self._frames_to_be_focused_start,
                self._frames_to_be_focused_end,
                self._skip_frames,
            )
            min_skip_frames = self._skip_frames
            max_skip_frames = self._skip_frames
        else:  # when the skip is expressed in meters
            frames_to_be_focused, min_skip_frames, max_skip_frames = (
                self.geom_obj.get_frame_ids_from_skip_distance(
                    self.data_obj.orbit_data["Lat"][
                        self._frames_to_be_focused_start : self._frames_to_be_focused_end
                    ],
                    self.data_obj.orbit_data["Lon"][
                        self._frames_to_be_focused_start : self._frames_to_be_focused_end
                    ],
                    self._skip_meters,
                )
            )
            frames_to_be_focused += self._frames_to_be_focused_start

        if len(frames_to_be_focused) == 0:
            print("ERROR: no frame to be focused in the selected range.")
            return None

        # correction of the last frame to be focused
        self._frames_to_be_focused_end = frames_to_be_focused[-1]
        n_frames_to_be_focused = len(frames_to_be_focused)

        if self._debug_mode:
            print("frames_to_be_focused_start =", self._frames_to_be_focused_start)
            print("frames_to_be_focused_end =", self._frames_to_be_focused_end)
            print("n_frames_to_be_focused =", n_frames_to_be_focused)
            print("last frame to be focused =", frames_to_be_focused[-1])

        # calculation of range of frames to be processed to focus the selected frames
        frames_to_be_processed_start = (
            self._frames_to_be_focused_start
            - self._n_half_aperture_frames
            + squint_offset_approx_frames
        )
        if frames_to_be_processed_start > self._frames_to_be_focused_start:
            frames_to_be_processed_start = self._frames_to_be_focused_start
            if self._debug_mode:
                print(
                    "\nNOTICE: first frame to be processed = first frame to be focused, as aperture is very delayed."
                )

        frames_to_be_processed_end = (
            self._frames_to_be_focused_end
            + self._n_half_aperture_frames
            + squint_offset_approx_frames
            + 1
        )
        if frames_to_be_processed_end < self._frames_to_be_focused_end:
            frames_to_be_processed_end = self._frames_to_be_focused_end
            if self._debug_mode:
                print(
                    "\nNOTICE: last frame to be processed = last frame to be focused, as aperture is very anticipated.\n"
                )

        frames_to_be_processed = np.arange(
            frames_to_be_processed_start, frames_to_be_processed_end + 1, dtype="int"
        )
        n_frames_to_be_processed = len(frames_to_be_processed)

        if self._debug_mode:
            print("n_frames_to_be_processed =", n_frames_to_be_processed)
            print("frames_to_be_processed_start =", frames_to_be_processed_start)
            print("frames_to_be_processed_end =", frames_to_be_processed_end)

        if frames_to_be_processed_start < 0:
            print(
                "ERROR: first frame to be processed is before the beginning of input raw data."
            )
            return None

        if frames_to_be_processed_end >= self._n_input_frames:
            print(
                "ERROR: the last frame to be processed is after the end of input raw data."
            )
            return None

        # calculation of ids of frames to be focused wrt frames_to_be_processed
        frames_to_be_focused_within_frames_to_be_processed = (
            frames_to_be_focused - frames_to_be_processed_start
        )

        # select only data related to frames to be processed
        input_rxwin_data_t = self.data_obj.orbit_data["rxwin_time_s"][
            frames_to_be_processed
        ]
        input_scx_data = self.data_obj.orbit_data["X"][frames_to_be_processed]
        input_scy_data = self.data_obj.orbit_data["Y"][frames_to_be_processed]
        input_scz_data = self.data_obj.orbit_data["Z"][frames_to_be_processed]
        input_lat_data = self.data_obj.orbit_data["Lat"][frames_to_be_processed]
        input_long_data = self.data_obj.orbit_data["Lon"][frames_to_be_processed]
        input_scr_data = self.data_obj.orbit_data["Radius"][frames_to_be_processed]
        dem_radius_profile = self._dem_radius_profile[frames_to_be_processed]

        input_theta = np.arctan2(input_scy_data, input_scx_data)
        input_phi = np.arccos(input_scz_data / input_scr_data)

        # shifts for final frame alignment
        input_scr_data_alignment = (
            input_scr_data
            - self.geom_obj.get_ellipsoid_radius_from_latlon(
                input_lat_data, input_long_data
            )
        )
        input_scr_data_shifts_alignment = (
            max(
                input_scr_data_alignment[
                    frames_to_be_focused_within_frames_to_be_processed
                ]
            )
            - input_scr_data_alignment
        )
        input_scr_data_shifts_alignment_t = (
            input_scr_data_shifts_alignment * 2.0 / self._C
        )

        input_rxwin_data_t_shifts = input_rxwin_data_t - min(
            input_rxwin_data_t[frames_to_be_focused_within_frames_to_be_processed]
        )

        # frame sample offsets for final frame alignment
        sample_start_shifts = np.round(
            (input_rxwin_data_t_shifts + input_scr_data_shifts_alignment_t) / self._dt
        ).astype("int")
        sample_start_shifts_max = np.max(
            sample_start_shifts[frames_to_be_focused_within_frames_to_be_processed]
        )

        self._N_OUTPUT_SAMPLES += sample_start_shifts_max  # update N_OUTPUT_SAMPLES
        OUTPUT_SAMPLES = np.arange(self._N_OUTPUT_SAMPLES)

        # create here azimuth weighting windows because _N_OUTPUT_SAMPLES has been updated
        self._azimuth_weighting_windows = (
            self._azimuth_weighting_window_original
            * np.ones([self._N_OUTPUT_SAMPLES, 1])
        )
        azimuth_weighting_is_unity = np.isscalar(
            self._azimuth_weighting_window_original
        )

        delay_start_shifts = (
            input_rxwin_data_t_shifts + input_scr_data_shifts_alignment_t
        )

        chirp_fft_conj = ProcessingUtils.create_synthetic_chirp_conj(
            self._frequency,
            self._bandwidth,
            self._chirp_duration,
            self._fast_time_sampling_frequency,
            self._range_weighting_window_original,
            self._SIZE_RANGE_FFT,
        )

        t_frame_base = self._dt * OUTPUT_SAMPLES
        fftfreqs = np.fft.fftfreq(self._SIZE_RANGE_FFT, self._dt)

        # calculate block size for parallel processing
        read_block_size_optimal_approx = np.ceil(
            n_frames_to_be_processed / self.n_processes
        )  # not used anymore + self._block_overlapping_size
        if read_block_size_optimal_approx > self._read_block_size_max:
            read_block_size_optimal_approx = self._read_block_size_max
        n_frames_to_be_focused_within_block_max = np.max(
            [
                int(
                    np.ceil(
                        (
                            read_block_size_optimal_approx
                            - self._n_half_aperture_frames * 2
                        )
                        / min_skip_frames
                    )
                    + 1
                ),
                1,
            ]
        )
        n_blocks = int(
            np.ceil(n_frames_to_be_focused / n_frames_to_be_focused_within_block_max)
        )
        n_frames_to_be_focused_in_last_block = (
            n_frames_to_be_focused
            - (n_blocks - 1) * n_frames_to_be_focused_within_block_max
        )
        n_frames_to_be_focused_per_block = np.concatenate(
            (
                n_frames_to_be_focused_within_block_max * np.ones(n_blocks - 1),
                [n_frames_to_be_focused_in_last_block],
            )
        ).astype("int")
        n_frames_to_be_focused_per_block_cum = np.cumsum(
            n_frames_to_be_focused_per_block
        )

        if self._debug_mode:
            msg = ""
            if read_block_size_optimal_approx == self._read_block_size_max:
                msg = "(MAXIMUM ALLOWED SIZE)"
            print(
                "read_block_size_optimal_approx =", read_block_size_optimal_approx, msg
            )
            print(
                "n_frames_to_be_focused_within_block_max =",
                n_frames_to_be_focused_within_block_max,
            )
            print(
                "n_frames_to_be_focused_in_last_block =",
                n_frames_to_be_focused_in_last_block,
            )
            print("N BLOCKS =", n_blocks)

        # initialize output images
        output_image = np.zeros([self._N_OUTPUT_SAMPLES, n_frames_to_be_focused])

        # create processing input block parameters
        input_parameters = list()
        for block_id in np.arange(n_blocks):
            frames_to_be_focused_in_this_block = frames_to_be_focused[
                int(block_id * n_frames_to_be_focused_within_block_max) : int(
                    block_id * n_frames_to_be_focused_within_block_max
                    + n_frames_to_be_focused_per_block[block_id]
                )
            ]
            block_frame_to_be_focused_start = frames_to_be_focused_in_this_block[0]
            block_frame_to_be_focused_end = frames_to_be_focused_in_this_block[-1]
            input_data_frame_start = (
                block_frame_to_be_focused_start
                - self._n_half_aperture_frames
                + squint_offset_approx_frames
            )
            input_data_frame_end = np.min(
                [
                    block_frame_to_be_focused_end
                    + self._n_half_aperture_frames
                    + 1
                    + squint_offset_approx_frames,
                    frames_to_be_processed_end,
                ]
            )
            input_parameters.append(
                (
                    input_data_frame_start,
                    input_data_frame_end,
                    frames_to_be_focused_in_this_block,
                    frames_to_be_processed_start,
                    self._SIZE_RANGE_FFT,
                    self._C,
                    input_rxwin_data_t,
                    chirp_fft_conj,
                    self._N_OUTPUT_SAMPLES,
                    OUTPUT_SAMPLES,
                    input_scr_data,
                    input_theta,
                    input_phi,
                    self._n_half_aperture_frames,
                    self._n_aperture_frames,
                    self._CORRECT_RANGE_MIGRATION,
                    input_scx_data,
                    input_scy_data,
                    input_scz_data,
                    self._COMPENSATE_DOPPLER_PHASE,
                    self._lmbda_min,
                    self._azimuth_weighting_windows,
                    azimuth_weighting_is_unity,
                    self._n_gaussian_filters,
                    self._gf_std_devs,
                    self._gf_power_coefficients,
                    self._doppler_half_window_width,
                    squint_offset_approx_frames,
                    t_frame_base,
                    dem_radius_profile,
                    fftfreqs,
                    delay_start_shifts,
                )
            )

        if progress_callback is not None:
            progress_callback(0.0, "frame selection and block partition computed")

        # Effective worker count: the requested n_processes, capped by the
        # CPU count and the memory-based worker budget. A cap down to 1 runs
        # the serial branch below (the exact pre-P5 processing path); a cap
        # between 2 and n_processes only queues more blocks, because the
        # block size is derived per requested process.
        n_workers = 1
        if self.n_processes > 1:
            available_memory_bytes = _available_memory_bytes()
            memory_worker_cap = (
                _max_memory_limited_pool_workers(
                    os.process_cpu_count(), available_memory_bytes
                )
                if available_memory_bytes is not None
                else os.process_cpu_count()
            )
            n_workers = min(self.n_processes, memory_worker_cap)

        if n_workers > 1:
            # Parallel block focus (phase-6 P5): the pool uses the explicit
            # fork context so the workers inherit the raw data copy-on-write
            # (with the default forkserver start method the multi-GB array
            # would have to be pickled per task). The fork happens at the
            # first submit, before any heavy compute of this focus run, and
            # the worker initializer pins each worker to one OpenBLAS thread.
            global _FORK_BLOCK_FOCUS_DATA_OBJ
            _FORK_BLOCK_FOCUS_DATA_OBJ = self.data_obj
            print(
                f"Focusing {n_blocks} block(s) on {n_workers} worker "
                f"processes (requested: {self.n_processes})."
            )
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=multiprocessing.get_context("fork"),
                initializer=_pool_worker_initializer,
            ) as executor:
                block_futures = [
                    executor.submit(
                        _pool_block_focus,
                        block_id,
                        *(input_parameters[block_id]),
                    )
                    for block_id in np.arange(n_blocks)
                ]
                for block_id, block_future in enumerate(block_futures):
                    column_start = (
                        0
                        if block_id == 0
                        else n_frames_to_be_focused_per_block_cum[block_id - 1]
                    )
                    output_image[
                        :,
                        column_start : column_start
                        + n_frames_to_be_focused_per_block[block_id],
                    ] = block_future.result()
                    focused_frames_in_block = input_parameters[block_id][2]
                    print(
                        f"PROCESSED BLOCK {block_id} ({block_id + 1}/{n_blocks}):"
                        f" focused from abs {int(focused_frames_in_block[0])}"
                        f" to {int(focused_frames_in_block[-1])}"
                        f" ({len(focused_frames_in_block)} frame(s))"
                    )
                    if progress_callback is not None:
                        progress_callback(
                            float((block_id + 1) / n_blocks),
                            f"focused block {block_id + 1}/{n_blocks}",
                        )
        else:
            output_image[:, 0 : n_frames_to_be_focused_per_block[0]] = (
                SOFA._block_focus(0, *(input_parameters[0]), self.data_obj)
            )
            if progress_callback is not None:
                progress_callback(1.0 / n_blocks, f"focused block 1/{n_blocks}")
            for block_id in np.arange(1, n_blocks):
                output_image[
                    :,
                    n_frames_to_be_focused_per_block_cum[
                        block_id - 1
                    ] : n_frames_to_be_focused_per_block_cum[block_id - 1]
                    + n_frames_to_be_focused_per_block[block_id],
                ] = SOFA._block_focus(
                    block_id,
                    *(input_parameters[block_id]),
                    self.data_obj,
                )
                if progress_callback is not None:
                    progress_callback(
                        float((block_id + 1) / n_blocks),
                        f"focused block {int(block_id) + 1}/{n_blocks}",
                    )

        return output_image

    @staticmethod
    def _block_focus(
        block_id: int,
        input_data_frame_start: int,
        input_data_frame_end: int,
        frames_to_be_focused_in_this_block: npt.NDArray[np.int64],
        frames_to_be_processed_start: int,
        sizeFFT1: int,
        c: float,
        input_rxwin_data_t: npt.NDArray[np.float64],
        chirp_fft_conj: npt.NDArray[np.complex128],
        N_OUTPUT_SAMPLES: int,
        OUTPUT_SAMPLES: npt.NDArray[np.int64],
        input_scr_data: npt.NDArray[np.float64],
        input_theta: npt.NDArray[np.float64],
        input_phi: npt.NDArray[np.float64],
        n_half_aperture_frames: int,
        n_aperture_frames: int,
        CORRECT_RANGE_MIGRATION: bool,
        input_scx_data: npt.NDArray[np.float64],
        input_scy_data: npt.NDArray[np.float64],
        input_scz_data: npt.NDArray[np.float64],
        USE_DOPPLER_CORRECTION: bool,
        lmbda_min: float,
        azimuth_weighting_windows: npt.NDArray[np.float64],
        azimuth_weighting_is_unity: bool,
        N_GAUSSIAN_FILTERS: int,
        std_devs: list[float],
        gf_power_coefficients: list[float],
        doppler_half_window_width: int,
        squint_offset_approx_frames: int,
        t_frame_base: npt.NDArray[np.float64],
        dem_radius_profile: npt.NDArray[np.int32],
        fftfreqs: npt.NDArray[np.float64],
        delay_start_shifts: npt.NDArray[np.float64],
        data_obj: Any,
        *,
        verbose: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Focuses one block of frames (range compression, per-aperture-frame delays, range-migration and Doppler phase corrections, azimuth FFT and multilooking) and returns the block image. The per-block progress prints run only with verbose (the parallel path calls this with verbose=False in the silent pool workers; the parent prints the progress instead)."""

        input_data = data_obj.data[:, input_data_frame_start:input_data_frame_end]
        read_block_size_real = input_data.shape[1]

        # NOTE: frames to be focused could be OUTSIDE the block because of squint!!!
        frames_to_be_focused_within_block = frames_to_be_focused_in_this_block
        n_frames_to_be_focused_within_block = len(frames_to_be_focused_within_block)

        if verbose:
            print("PROCESSING BLOCK", block_id)
            print(
                "  FOCUSING FROM ABS ",
                frames_to_be_focused_within_block[0],
                "TO",
                frames_to_be_focused_within_block[-1],
            )
            print(
                "  n_frames_to_be_focused_within_block =",
                n_frames_to_be_focused_within_block,
            )

        output_image = np.zeros([N_OUTPUT_SAMPLES, n_frames_to_be_focused_within_block])
        signals = np.zeros([sizeFFT1, read_block_size_real], dtype="complex")

        signals[: input_data.shape[0], :] = input_data

        if verbose:
            print("...matched filtering...")
        signals_fft = np.fft.fft(signals, axis=0)  # spectrum: 0-positive ... negative-0
        signals_fft *= chirp_fft_conj[:, None]

        del signals

        i_frame_focusing = 0

        # aperture has final length = 2*n_half_aperture_frames+1
        aperture_range_offset = np.arange(
            squint_offset_approx_frames - n_half_aperture_frames,
            squint_offset_approx_frames + n_half_aperture_frames + 1,
            dtype="int",
        )
        # Allocated once per block and reused across ref frames; every element
        # is overwritten before use on each ref frame.
        signals_matched_within_aperture = np.zeros(
            [N_OUTPUT_SAMPLES, n_aperture_frames], dtype="complex"
        )
        delay_ramp_prefactor = -1j * 2.0 * np.pi * fftfreqs
        ifft_chunk_buffer = np.empty(
            [sizeFFT1, min(_APERTURE_FFT_CHUNK, n_aperture_frames)], dtype="complex"
        )
        for ref_frame_abs in frames_to_be_focused_within_block:
            ref_frame = (
                ref_frame_abs - frames_to_be_processed_start
            )  # referred to the frames_to_be_processed interval
            aperture_frames = ref_frame + aperture_range_offset

            if CORRECT_RANGE_MIGRATION:
                # range migration is corrected wrt DEM radius: not necessary to correct for every single sample, error is negligible
                r_sample_ref = dem_radius_profile[ref_frame]
                x_sample_ref = (
                    r_sample_ref
                    * (np.cos(input_theta[ref_frame]))
                    * (np.sin(input_phi[ref_frame]))
                )
                y_sample_ref = (
                    r_sample_ref
                    * (np.sin(input_theta[ref_frame]))
                    * (np.sin(input_phi[ref_frame]))
                )
                z_sample_ref = r_sample_ref * np.cos(input_phi[ref_frame])

                sc_distance_to_ref_sample_within_aperture = np.sqrt(
                    (input_scx_data[aperture_frames] - x_sample_ref) ** 2
                    + (input_scy_data[aperture_frames] - y_sample_ref) ** 2
                    + (input_scz_data[aperture_frames] - z_sample_ref) ** 2
                )
                sc_distance_to_ref_sample_within_aperture_ref = np.sqrt(
                    (input_scx_data[ref_frame] - x_sample_ref) ** 2
                    + (input_scy_data[ref_frame] - y_sample_ref) ** 2
                    + (input_scz_data[ref_frame] - z_sample_ref) ** 2
                )

                # t of beginning of frames wrt tx -rxwin
                t_within_frames_within_aperture = (
                    sc_distance_to_ref_sample_within_aperture * 2.0 / c
                    - input_rxwin_data_t[aperture_frames]
                )
                t_within_frames_within_aperture_ref = (
                    sc_distance_to_ref_sample_within_aperture_ref * 2.0 / c
                    - input_rxwin_data_t[ref_frame]
                )

                t_sample_compr_ref = t_frame_base.T + input_rxwin_data_t[ref_frame]
                r_sample_ref = np.array(
                    [input_scr_data[ref_frame] - t_sample_compr_ref * c / 2.0]
                ).T
                x_sample_ref = (
                    r_sample_ref
                    * (np.cos(input_theta[ref_frame]))
                    * (np.sin(input_phi[ref_frame]))
                )
                y_sample_ref = (
                    r_sample_ref
                    * (np.sin(input_theta[ref_frame]))
                    * (np.sin(input_phi[ref_frame]))
                )
                z_sample_ref = r_sample_ref * np.cos(input_phi[ref_frame])

                sc_distance_to_ref_sample_within_aperture_matrix = np.sqrt(
                    (input_scx_data[aperture_frames] - x_sample_ref) ** 2
                    + (input_scy_data[aperture_frames] - y_sample_ref) ** 2
                    + (input_scz_data[aperture_frames] - z_sample_ref) ** 2
                )

                delays_within_aperture = -(
                    t_within_frames_within_aperture
                    - t_within_frames_within_aperture_ref
                )

                if USE_DOPPLER_CORRECTION:
                    # phase correction "US style", only referred to a reference sample
                    doppler_phase_corr_within_aperture = (
                        -4.0
                        * np.pi
                        * sc_distance_to_ref_sample_within_aperture_matrix
                        / lmbda_min
                    )
                else:
                    doppler_phase_corr_within_aperture = np.zeros(
                        [N_OUTPUT_SAMPLES, n_aperture_frames]
                    )
            else:
                delays_within_aperture = np.zeros(n_aperture_frames)
                doppler_phase_corr_within_aperture = np.zeros(
                    [N_OUTPUT_SAMPLES, n_aperture_frames]
                )

            first_aperture_column = (
                aperture_frames[0]
                + frames_to_be_processed_start
                - input_data_frame_start
            )
            for i_chunk_start in range(0, n_aperture_frames, _APERTURE_FFT_CHUNK):
                i_chunk_end = min(
                    i_chunk_start + _APERTURE_FFT_CHUNK, n_aperture_frames
                )
                n_chunk_frames = i_chunk_end - i_chunk_start
                chunk_product = ifft_chunk_buffer[:, :n_chunk_frames]
                np.multiply(
                    delay_ramp_prefactor[:, None],
                    (
                        delays_within_aperture[i_chunk_start:i_chunk_end]
                        + delay_start_shifts[ref_frame]
                    )[None, :],
                    out=chunk_product,
                )
                np.exp(chunk_product, out=chunk_product)
                chunk_first_column = first_aperture_column + i_chunk_start
                chunk_last_column = first_aperture_column + i_chunk_end
                np.multiply(
                    signals_fft[:, chunk_first_column:chunk_last_column],
                    chunk_product,
                    out=chunk_product,
                )
                # OUTPUT_SAMPLES is arange(N_OUTPUT_SAMPLES), so the per-column
                # fancy index of the former loop is this prefix slice.
                signals_matched_within_aperture[:, i_chunk_start:i_chunk_end] = (
                    np.fft.ifft(chunk_product, axis=0)[:N_OUTPUT_SAMPLES, :]
                )

            # Doppler correction on whole aperture
            if USE_DOPPLER_CORRECTION:
                signals_matched_within_aperture *= np.exp(
                    1j * doppler_phase_corr_within_aperture
                )
            if not azimuth_weighting_is_unity:
                signals_matched_within_aperture *= azimuth_weighting_windows

            signals_matched_within_aperture_azimuth_fft = np.fft.fftshift(
                np.fft.fft(signals_matched_within_aperture, axis=1), axes=(1,)
            )
            signals_matched_within_aperture_azimuth_fft = (
                np.abs(signals_matched_within_aperture_azimuth_fft) ** 2.0
            )

            if N_GAUSSIAN_FILTERS > 0:
                signals_matched_within_aperture_upsampled_azimuth_fft_g = list()

                for std_dev_tmp in std_devs:
                    signals_matched_within_aperture_upsampled_azimuth_fft_g.append(
                        gaussian_filter1d(
                            signals_matched_within_aperture_azimuth_fft,
                            std_dev_tmp,
                            axis=1,
                        )
                    )

                signals_matched_within_aperture_azimuth_fft *= gf_power_coefficients[0]
                for i_std_dev in np.arange(N_GAUSSIAN_FILTERS):
                    signals_matched_within_aperture_azimuth_fft += (
                        gf_power_coefficients[N_GAUSSIAN_FILTERS - 1 - i_std_dev]
                        * signals_matched_within_aperture_upsampled_azimuth_fft_g[
                            i_std_dev
                        ]
                    )

            zero_doppler_search_half_window_width = 0
            frame_power_max_index = (
                n_half_aperture_frames - zero_doppler_search_half_window_width
            ) + np.argmax(
                signals_matched_within_aperture_azimuth_fft[
                    :,
                    n_half_aperture_frames
                    - zero_doppler_search_half_window_width : n_half_aperture_frames
                    + zero_doppler_search_half_window_width
                    + 1,
                ],
                axis=1,
            )
            frame_multilooking_start = frame_power_max_index - doppler_half_window_width
            output_frame_power = np.lib.stride_tricks.sliding_window_view(
                signals_matched_within_aperture_azimuth_fft,
                doppler_half_window_width * 2 + 1,
                axis=1,
            )[OUTPUT_SAMPLES, frame_multilooking_start[OUTPUT_SAMPLES]].sum(axis=1) / (
                doppler_half_window_width * 2 + 1
            )

            output_image[OUTPUT_SAMPLES, i_frame_focusing] = output_frame_power[:]

            i_frame_focusing += 1

        return output_image
