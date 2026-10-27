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

"""Base class of the surface-clutter simulators: frame selection, fast-time
alignment of the track, and dispatch of the per-frame computation to the
format-specific simulation callback.
"""

import concurrent.futures
import multiprocessing
import os
from typing import Any

import numpy as np
import numpy.typing as npt

from sopa.geometry.coordinate_converter import CoordinateConverter as cc
from sopa.processing.focusers.sofa import (
    _available_memory_bytes,
    _max_memory_limited_pool_workers,
    _pool_worker_initializer,
)
from sopa.processing.utils import ProcessingUtils

# Simulation data handed to the parallel frame simulation (phase-6 P6): set by
# _SIMBase.simulate before the fork-based pool is created and resolved by the
# workers through fork copy-on-write, so the multi-GB DEM object and the
# per-track arrays are never pickled through submit().
_FORK_SIMULATION_CONTEXT: Any | None = None


def _alt_unit_vector_carry(
    orbit_xyz: tuple[
        npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
    ],
    previous_frame: int,
) -> npt.NDArray[np.float64]:
    """Computes the along-track unit vector that the serial simulation loop carries from the simulated frame `previous_frame` into the following one: the unit vector of the spacecraft-position difference between `previous_frame` and its successor, exactly the value `_simulate_frame_range` computes for a non-final frame."""
    return cc.unit_vector(
        np.array(
            [
                [[orbit_xyz[0][previous_frame + 1] - orbit_xyz[0][previous_frame]]],
                [[orbit_xyz[1][previous_frame + 1] - orbit_xyz[1][previous_frame]]],
                [[orbit_xyz[2][previous_frame + 1] - orbit_xyz[2][previous_frame]]],
            ]
        )
    )


def _pool_simulate_frame_range(
    range_id: int, n_ranges: int, *range_parameters: Any
) -> Any:
    """Worker-side entry point of the parallel frame simulation: resolves the shared simulation data from the fork-inherited module reference (never from pickled arguments) and runs the same `_simulate_frame_range` operation sequence as the serial path for one frame range, with the worker-side progress prints suppressed because the parent prints the per-range progress."""
    if _FORK_SIMULATION_CONTEXT is None:
        raise RuntimeError(
            "ERROR: the fork-inherited simulation data reference is not set; "
            "_pool_simulate_frame_range must run in a worker forked after "
            "_SIMBase.simulate set it."
        )
    (
        sim_callback,
        orbit_xyz,
        sim_n_samples,
        local_direction_g,
        UTMNorthing_g,
        UTMEasting_g,
        UTMZone_g,
        input_scr_data_shifts_alignment_t,
        dem_obj,
        geom_obj,
        alt_sim_matrix,
        act_sim_matrix,
        roi_img,
    ) = _FORK_SIMULATION_CONTEXT
    (
        range_start,
        range_stop,
        rxwin_ref,
        dt,
        gaussian_filter_sigma,
        further_pars,
        alt_unit_vector_carry,
    ) = range_parameters
    return sim_callback(
        range_id,
        n_ranges,
        orbit_xyz,
        np.arange(range_start, range_stop),
        sim_n_samples,
        local_direction_g,
        UTMNorthing_g,
        UTMEasting_g,
        UTMZone_g,
        rxwin_ref,
        input_scr_data_shifts_alignment_t,
        dt,
        dem_obj,
        geom_obj,
        alt_sim_matrix,
        act_sim_matrix,
        gaussian_filter_sigma,
        further_pars,
        roi_img,
        alt_unit_vector_carry=alt_unit_vector_carry,
        verbose=False,
    )


def _assemble_range_results(
    range_results: list[tuple[Any, ...]],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.uint8],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    tuple[dict[int, npt.NDArray[np.float64]], ...],
]:
    """Assembles the per-frame-range simulation results into the whole-track outputs: the image blocks and first-return values are concatenated in frame order (the per-frame writes are column-disjoint, so the result is independent of the range partition), and the per-range ROI dictionaries are merged key by key in frame order."""
    sim_image = np.concatenate([result[0] for result in range_results], axis=1)
    uncert_image = np.concatenate([result[1] for result in range_results], axis=1)
    first_return_lats = np.concatenate([result[2] for result in range_results])
    first_return_lons = np.concatenate([result[3] for result in range_results])
    ground_distance_min_image = np.concatenate(
        [result[4] for result in range_results], axis=1
    )
    ground_distance_max_image = np.concatenate(
        [result[5] for result in range_results], axis=1
    )
    roi_pos: list[dict[int, npt.NDArray[np.float64]]] = []
    for roi_dict_index in range(len(range_results[0][6])):
        merged: dict[int, npt.NDArray[np.float64]] = {}
        for result in range_results:
            for roi_id, values in result[6][roi_dict_index].items():
                if roi_id in merged:
                    merged[roi_id] = np.concatenate((merged[roi_id], values))
                else:
                    merged[roi_id] = values
        roi_pos.append(merged)
    return (
        sim_image,
        uncert_image,
        first_return_lats,
        first_return_lons,
        ground_distance_min_image,
        ground_distance_max_image,
        tuple(roi_pos),
    )


class _SIMBase:
    """Shared implementation of the clutter simulators.

    It selects the frames to simulate (skip in frames or meters), aligns the
    frames in fast time (receive-window and spacecraft-radius shifts), maps
    the track to UTM to estimate the local ground direction, and hands the
    per-frame computation to the subclass' static `_simulate_frame_range`
    callback. With `simulate_chirp` the simulated image is convolved with
    the synthetic chirp (pulse compression). With `n_processes` = 1 (the
    default) the frames run serially in a single call, which is the original
    processing path; with `n_processes` above 1 the selected frames are
    partitioned into contiguous ranges (derived per requested process by
    `_define_frame_ranges`) and the ranges are dispatched to a fork-based
    process pool whose worker count is capped by the CPU count and the
    memory-based worker budget, with the results assembled in frame order.
    """

    _C = 299792458.0

    def __init__(
        self,
        geom_obj: Any,
        dem_obj: Any,
        param_obj: Any,
        n_processes: int = 1,
        use_data_dt: bool = True,
        simulate_chirp: bool = False,
        instr_obj: Any = None,
        roi_img: npt.NDArray | None = None,
    ) -> None:
        self.DEFAULT_DT = 37.5e-9
        self.USE_DATA_DT = use_data_dt

        if simulate_chirp and instr_obj is None:
            raise ValueError(
                "ERROR: simulate_chirp set to True but intrument object not provided."
            )

        self.SIMULATE_CHIRP = simulate_chirp
        self.instr_obj = instr_obj

        self.dt = self.DEFAULT_DT
        self._N_OUTPUT_SAMPLES = 667 * 2
        self.sim_top_offset_samples = 200

        self.geom_obj = geom_obj
        self.dem_obj = dem_obj

        self.roi_img = roi_img

        self._frames_to_be_simulated_start = param_obj.frame_start
        self._frames_to_be_simulated_end = param_obj.frame_end
        if param_obj.frame_skip == 0:
            self._skip_meters = param_obj.meters_skip
            self._skip_frames = 0
        else:
            self._skip_frames = param_obj.frame_skip
            self._skip_meters = None
        self._n_half_aperture_frames = param_obj.frame_half_aperture

        # Parallel frame simulation is implemented again since phase-6 P6: a
        # fork-based process pool over the frame ranges (see simulate()).
        # Values < 1 fall back to 1; 1 keeps the original serial path, which
        # is the exact pre-P6 processing code.
        if n_processes < 1:
            n_processes = 1

        self.n_processes = n_processes

    def _pre_simulate(
        self, orbit_data: dict[str, npt.NDArray[np.float64] | float]
    ) -> (
        tuple[
            int,
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            float,
            float,
        ]
        | None
    ):
        """Validates the frame range, selects the frames to simulate, prepares the aligned track vectors and receive-window reference, and allocates the output images; returns None on invalid ranges."""
        n_input_frames = orbit_data["X"].shape[0]

        if self._frames_to_be_simulated_end >= n_input_frames:
            print("ERROR: last frame to be simulated after the end of the input data.")
            return None

        if self.USE_DATA_DT:
            if "dt" in orbit_data:
                self.dt = orbit_data["dt"]
            else:
                print(
                    f"WARNING: orbit data do not contain the dt value. Using the default dt = {self.DEFAULT_DT} s."
                )

        # calculate actual start and end frames according to aperture
        if self._frames_to_be_simulated_start == -1:
            self._frames_to_be_simulated_start = np.max(
                [self._n_half_aperture_frames, 0]
            )

        if self._frames_to_be_simulated_end == -1:
            self._frames_to_be_simulated_end = np.min(
                [n_input_frames - self._n_half_aperture_frames - 1, n_input_frames - 1]
            )

        if self._frames_to_be_simulated_end < self._frames_to_be_simulated_start:
            print(
                "ERROR: last frame to be simulated before first frame to be simulated."
            )
            return None

        # calculation of range of frames to be simulated
        if self._skip_frames != 0:
            frames_to_be_simulated = np.arange(
                self._frames_to_be_simulated_start,
                self._frames_to_be_simulated_end,
                self._skip_frames,
            )
        else:
            frames_to_be_simulated, _, _ = (
                self.geom_obj.get_frame_ids_from_skip_distance(
                    orbit_data["Lat"][
                        self._frames_to_be_simulated_start : self._frames_to_be_simulated_end
                    ],
                    orbit_data["Lon"][
                        self._frames_to_be_simulated_start : self._frames_to_be_simulated_end
                    ],
                    self._skip_meters,
                )
            )
            frames_to_be_simulated += self._frames_to_be_simulated_start

        # correction of the last frame to be simulated
        self._frames_to_be_simulated_end = frames_to_be_simulated[-1]
        n_frames_to_be_simulated = len(frames_to_be_simulated)

        # select only data related to frames to be processed
        input_rxwin_data_t = orbit_data["rxwin_time_s"][frames_to_be_simulated]
        input_scx_data = orbit_data["X"][frames_to_be_simulated]
        input_scy_data = orbit_data["Y"][frames_to_be_simulated]
        input_scz_data = orbit_data["Z"][frames_to_be_simulated]
        input_lat_data = orbit_data["Lat"][frames_to_be_simulated]
        input_long_data = orbit_data["Lon"][frames_to_be_simulated]
        input_scr_data = orbit_data["Radius"][frames_to_be_simulated]

        # shifts for final frame alignment
        input_scr_data_alignment = (
            input_scr_data
            - self.geom_obj.get_ellipsoid_radius_from_latlon(
                input_lat_data, input_long_data
            )
        )
        input_scr_data_shifts_alignment = (
            max(input_scr_data_alignment) - input_scr_data_alignment
        )
        input_scr_data_shifts_alignment_t = (
            input_scr_data_shifts_alignment * 2.0 / self._C
        )

        # # calculate reference rxwin and sc radius to be used as reference for frame time alignment
        # sc_radius_ref = np.min(input_scr_data)
        # ellipsoid_radius = self.geom_obj.get_ellipsoid_radius_from_latlon(input_lat_data, input_long_data)
        # dem_radius = self.dem_obj.get_dem_radius_from_lat_lon(input_lat_data, input_long_data)
        # dem_radius_ref_id = np.argmax(dem_radius)
        # ellipsoid_radius_ref = ellipsoid_radius[dem_radius_ref_id]
        # input_scr_data_shifts_alignment_t = ((sc_radius_ref - input_scr_data) + (ellipsoid_radius - ellipsoid_radius_ref)) * 2. / self._C

        if np.max(input_rxwin_data_t) == 0:  # if rxwin is not set in the orbit data
            print("NOTICE: field rxwin_time_s not set. Using the default value.")

            # rxwin is calculated as the time delay that sets the "highest point" of the simulation at sim_top_offset_samples from the top border
            dem_radius = self.dem_obj.get_dem_radius_from_lat_lon(
                input_lat_data, input_long_data
            )
            t_surface_vect_ref = 2 * (input_scr_data - dem_radius) / self._C
            rxwin_ref = (
                np.min(t_surface_vect_ref) - self.sim_top_offset_samples * self.dt
            )
            self._N_OUTPUT_SAMPLES += int(
                np.max(np.round((input_scr_data_shifts_alignment_t) / self.dt))
            )
        else:
            rxwin_ref = np.min(input_rxwin_data_t)
            self._N_OUTPUT_SAMPLES += int(
                np.max(
                    np.round(
                        (
                            input_scr_data_shifts_alignment_t
                            + (input_rxwin_data_t - rxwin_ref)
                        )
                        / self.dt
                    )
                )
            )

        self.sim_image = np.zeros(
            [self._N_OUTPUT_SAMPLES, n_frames_to_be_simulated]
        )  # , dtype="int16")   # see comment below
        self.first_return_lats = np.zeros(n_frames_to_be_simulated)
        self.first_return_lons = np.zeros(n_frames_to_be_simulated)

        return (
            n_frames_to_be_simulated,
            input_rxwin_data_t,
            input_scx_data,
            input_scy_data,
            input_scz_data,
            input_lat_data,
            input_long_data,
            input_scr_data,
            rxwin_ref,
            input_scr_data_shifts_alignment_t,
        )

    def _define_frame_ranges(
        self, n_frames_to_be_simulated: int
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        n_frames_per_subprocess = int(
            np.ceil(n_frames_to_be_simulated / self.n_processes)
        )
        sub_frame_range_starts = np.arange(
            0, n_frames_to_be_simulated, n_frames_per_subprocess
        )
        sub_frame_range_stops = sub_frame_range_starts + n_frames_per_subprocess
        sub_frame_range_stops[-1] = np.min(
            [sub_frame_range_stops[-1], n_frames_to_be_simulated]
        )
        return sub_frame_range_starts, sub_frame_range_stops

    def _simulate_chirp(self) -> None:
        """Convolves the simulated image with the synthetic chirp in the frequency domain (pulse compression of the simulated echo)."""
        SIZE_RANGE_FFT = 4096
        range_weighting_window_ones = np.ones(int(SIZE_RANGE_FFT / 2.0))
        range_weighting_window_original = np.hanning(SIZE_RANGE_FFT / 2.0)
        chirp_fft = np.conj(
            np.array(
                [
                    ProcessingUtils.create_synthetic_chirp_conj(
                        self.instr_obj.frequency,
                        self.instr_obj.bandwidth,
                        self.instr_obj.chirp_duration,
                        self.instr_obj.fast_time_sampling_frequency,
                        range_weighting_window_ones,
                        SIZE_RANGE_FFT,
                    )
                ]
            ).T
        )
        chirp_fft_conj = np.array(
            [
                ProcessingUtils.create_synthetic_chirp_conj(
                    self.instr_obj.frequency,
                    self.instr_obj.bandwidth,
                    self.instr_obj.chirp_duration,
                    self.instr_obj.fast_time_sampling_frequency,
                    range_weighting_window_original,
                    SIZE_RANGE_FFT,
                )
            ]
        ).T
        chirp_fft_combined = chirp_fft * chirp_fft_conj
        sim_image_fft = np.fft.fft(self.sim_image, n=SIZE_RANGE_FFT, axis=0)
        sim_image_fft *= chirp_fft_combined * np.ones([1, self.sim_image.shape[1]])
        self.sim_image = np.abs(
            np.fft.ifft(sim_image_fft, axis=0)[: self.sim_image.shape[0], :]
        )

    def simulate(
        self,
        orbit_data: dict[str, npt.NDArray[np.float64] | float],
        sim_callback: Any,
        further_pars: list[float] = [],
    ) -> (
        tuple[
            npt.NDArray[np.float64],
            npt.NDArray[np.uint8],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            npt.NDArray[np.float64],
            tuple[Any, ...],
        ]
        | None
    ):
        """Runs the simulation over the selected frames through `sim_callback` and returns the simulated image, its uncertainty mask, the first-return lat/lon, the ground-distance min/max images and the ROI positions; returns None on input errors. With n_processes above 1 the per-frame-range simulation is dispatched to a fork-based process pool (worker count capped by the CPU count and the memory-based worker budget) and the outputs are assembled in frame order, which leaves them independent of the worker count; the frame ranges are derived per requested process, so a smaller effective worker count only queues more ranges."""

        pre_simulate_output = self._pre_simulate(orbit_data)
        if pre_simulate_output is not None:
            (
                n_frames_to_be_simulated,
                input_rxwin_data_t,
                input_scx_data,
                input_scy_data,
                input_scz_data,
                input_lat_data,
                input_long_data,
                input_scr_data,
                rxwin_ref,
                input_scr_data_shifts_alignment_t,
            ) = pre_simulate_output
        else:
            return None

        UTMNorthing_g, UTMEasting_g, UTMZone_g, UTMZoneNumber = self.geom_obj.LL_to_UTM(
            input_lat_data, input_long_data
        )

        # frame before, with same UTMZone of central frame
        UTMNorthing_before, UTMEasting_before = (
            self.geom_obj.LL_to_UTM_with_given_zone_number(
                input_lat_data[: -self.n_frames_offset_for_dir_estimation],
                input_long_data[: -self.n_frames_offset_for_dir_estimation],
                UTMZoneNumber[self.n_frames_offset_for_dir_estimation :],
            )
        )
        UTMNorthing_before = np.concatenate(
            (
                np.ones(self.n_frames_offset_for_dir_estimation)
                * UTMNorthing_before[0],
                UTMNorthing_before,
            ),
            0,
        )
        UTMEasting_before = np.concatenate(
            (
                np.ones(self.n_frames_offset_for_dir_estimation) * UTMEasting_before[0],
                UTMEasting_before,
            ),
            0,
        )

        # frame after, with same UTMZone of central frame
        UTMNorthing_after, UTMEasting_after = (
            self.geom_obj.LL_to_UTM_with_given_zone_number(
                input_lat_data[self.n_frames_offset_for_dir_estimation :],
                input_long_data[self.n_frames_offset_for_dir_estimation :],
                UTMZoneNumber[: -self.n_frames_offset_for_dir_estimation],
            )
        )
        UTMNorthing_after = np.concatenate(
            (
                UTMNorthing_after,
                np.ones(self.n_frames_offset_for_dir_estimation)
                * UTMNorthing_after[-1],
            ),
            0,
        )
        UTMEasting_after = np.concatenate(
            (
                UTMEasting_after,
                np.ones(self.n_frames_offset_for_dir_estimation) * UTMEasting_after[-1],
            ),
            0,
        )

        # local direction = 0º --> Easting, = 90º --> Northing
        local_direction_g = np.arctan2(
            UTMNorthing_after - UTMNorthing_before, UTMEasting_after - UTMEasting_before
        )

        orbit_xyz = (input_scx_data, input_scy_data, input_scz_data)

        if self.n_processes == 1:
            # Serial simulation: the original, unchanged processing path (the
            # fidelity gate runs this path).
            (
                self.sim_image,
                self.uncert_image,
                self.first_return_lats,
                self.first_return_lons,
                self.ground_distance_min_image,
                self.ground_distance_max_image,
                self.roi_pos,
            ) = sim_callback(
                0,
                1,
                orbit_xyz,
                np.arange(n_frames_to_be_simulated),
                self._N_OUTPUT_SAMPLES,
                local_direction_g,
                UTMNorthing_g,
                UTMEasting_g,
                UTMZone_g,
                rxwin_ref,
                input_scr_data_shifts_alignment_t,
                self.dt,
                self.dem_obj,
                self.geom_obj,
                self.alt_sim_matrix,
                self.act_sim_matrix,
                self.gaussian_filter_sigma,
                further_pars,
                self.roi_img,
            )
        else:
            # Parallel frame simulation (phase-6 P6). The frame partition is
            # derived per requested process, so a worker count capped down by
            # the CPU count or the memory budget only queues more (smaller)
            # ranges; fewer workers than requested is the memory-safe
            # direction.
            sub_frame_range_starts, sub_frame_range_stops = self._define_frame_ranges(
                n_frames_to_be_simulated
            )
            n_ranges = len(sub_frame_range_starts)

            range_parameters: list[tuple[Any, ...]] = []
            for range_id in np.arange(n_ranges):
                range_start = int(sub_frame_range_starts[range_id])
                range_stop = int(sub_frame_range_stops[range_id])
                # The along-track unit vector of the track's last simulated
                # frame is reused from the previous simulated frame inside
                # `_simulate_frame_range`; a range starting exactly at that
                # frame is handed the value the serial loop would have
                # carried into it (a pure function of orbit_xyz).
                if range_start == 0:
                    alt_unit_vector_carry = None
                else:
                    alt_unit_vector_carry = _alt_unit_vector_carry(
                        orbit_xyz, range_start - 1
                    )
                range_parameters.append(
                    (
                        range_start,
                        range_stop,
                        rxwin_ref,
                        self.dt,
                        self.gaussian_filter_sigma,
                        further_pars,
                        alt_unit_vector_carry,
                    )
                )

            # Effective worker count: the requested n_processes, capped by
            # the CPU count and the memory-based worker budget (the P5
            # formula and constants, conservative for the simulators'
            # ~200-250 MB per-worker peaks). A cap down to 1 runs the serial
            # per-range loop below.
            available_memory_bytes = _available_memory_bytes()
            memory_worker_cap = (
                _max_memory_limited_pool_workers(
                    os.process_cpu_count(), available_memory_bytes
                )
                if available_memory_bytes is not None
                else os.process_cpu_count()
            )
            n_workers = min(self.n_processes, memory_worker_cap)

            range_results: list[Any] = []
            if n_workers > 1:
                # The pool uses the explicit fork context so the workers
                # inherit the DEM object and the per-track arrays
                # copy-on-write (with the default forkserver start method the
                # multi-GB DEM would have to be pickled per task). The fork
                # happens at the first submit, before any frame compute of
                # this simulation run, and the worker initializer pins each
                # worker to one OpenBLAS thread.
                global _FORK_SIMULATION_CONTEXT
                _FORK_SIMULATION_CONTEXT = (
                    sim_callback,
                    orbit_xyz,
                    self._N_OUTPUT_SAMPLES,
                    local_direction_g,
                    UTMNorthing_g,
                    UTMEasting_g,
                    UTMZone_g,
                    input_scr_data_shifts_alignment_t,
                    self.dem_obj,
                    self.geom_obj,
                    self.alt_sim_matrix,
                    self.act_sim_matrix,
                    self.roi_img,
                )
                print(
                    f"Simulating {n_ranges} frame range(s) on {n_workers} worker "
                    f"processes (requested: {self.n_processes})."
                )
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=n_workers,
                    mp_context=multiprocessing.get_context("fork"),
                    initializer=_pool_worker_initializer,
                ) as executor:
                    range_futures = [
                        executor.submit(
                            _pool_simulate_frame_range,
                            range_id,
                            n_ranges,
                            *(range_parameters[range_id]),
                        )
                        for range_id in np.arange(n_ranges)
                    ]
                    # Indexed futures: the results are collected in frame
                    # (range) order, never in completion order.
                    for range_id, range_future in enumerate(range_futures):
                        range_results.append(range_future.result())
                        print(
                            f"PROCESSED RANGE {range_id} ({range_id + 1}/{n_ranges}):"
                            f" simulated local frames"
                            f" {int(sub_frame_range_starts[range_id])}"
                            f" to {int(sub_frame_range_stops[range_id]) - 1}"
                            f" ({int(sub_frame_range_stops[range_id] - sub_frame_range_starts[range_id])} frame(s))"
                        )
            else:
                for range_id in np.arange(n_ranges):
                    (
                        range_start,
                        range_stop,
                        rxwin_ref,
                        dt,
                        gaussian_filter_sigma,
                        further_pars,
                        alt_unit_vector_carry,
                    ) = range_parameters[range_id]
                    range_results.append(
                        sim_callback(
                            range_id,
                            n_ranges,
                            orbit_xyz,
                            np.arange(range_start, range_stop),
                            self._N_OUTPUT_SAMPLES,
                            local_direction_g,
                            UTMNorthing_g,
                            UTMEasting_g,
                            UTMZone_g,
                            rxwin_ref,
                            input_scr_data_shifts_alignment_t,
                            dt,
                            self.dem_obj,
                            self.geom_obj,
                            self.alt_sim_matrix,
                            self.act_sim_matrix,
                            gaussian_filter_sigma,
                            further_pars,
                            self.roi_img,
                            alt_unit_vector_carry=alt_unit_vector_carry,
                            verbose=False,
                        )
                    )
                    print(
                        f"PROCESSED RANGE {range_id} ({range_id + 1}/{n_ranges}):"
                        f" simulated local frames {range_start}"
                        f" to {range_stop - 1} ({range_stop - range_start} frame(s))"
                    )

            # Assemble in frame order: the per-frame writes are column-disjoint
            # by construction and the ROI accumulations are concatenated per
            # key in frame order, so the assembled results do not depend on
            # the range partition or the workers' completion order.
            (
                self.sim_image,
                self.uncert_image,
                self.first_return_lats,
                self.first_return_lons,
                self.ground_distance_min_image,
                self.ground_distance_max_image,
                self.roi_pos,
            ) = _assemble_range_results(range_results)

        if (len(self.roi_pos) < 5) and self.SIMULATE_CHIRP:
            self._simulate_chirp()

        # self.sim_image = self.sim_image.astype("int16")
        # self.uncert_image = self.uncert_image.astype("uint8")
        return (
            self.sim_image,
            self.uncert_image,
            self.first_return_lats,
            self.first_return_lons,
            self.ground_distance_min_image,
            self.ground_distance_max_image,
            self.roi_pos,
        )
