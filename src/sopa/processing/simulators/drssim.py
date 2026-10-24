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

"""Discrete-point surface-clutter simulator: every point of the DEM surface
grid contributes one count to the range sample corresponding to its
two-way travel time.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter

from sopa.processing.simulators.sim_base import _SIMBase


class dRSsim(_SIMBase):
    """Discrete-point clutter simulator.

    Around each frame's sub-spacecraft point it builds a surface grid
    (plus/minus 500 m along track, plus/minus 50000 m across track, at the
    nominal 50 m spacing), samples the smoothed DEM on it, and adds 1 to the
    range sample of every grid point's two-way travel time: no facet
    weighting, unlike `sfRSsim`. For each frame it also records the
    first-return position, the minimum/maximum ground distance associated to
    each simulated sample, and the lat/lon of the points falling in the
    ranges of the ROI image, when one is provided.
    """

    def __init__(
        self,
        geom_obj: Any,
        dem_obj: Any,
        param_obj: Any,
        n_processes: int = 1,
        use_data_dt: bool = True,
        simulate_chirp: bool = True,
        instr_obj: Any = None,
        roi_img: npt.NDArray | None = None,
    ) -> None:
        super().__init__(
            geom_obj,
            dem_obj,
            param_obj,
            n_processes,
            use_data_dt,
            simulate_chirp,
            instr_obj,
            roi_img,
        )
        self.n_frames_offset_for_dir_estimation = 1
        self.alt_sim_spacing = 50
        self.act_sim_spacing = 50
        self.gaussian_filter_sigma = 4
        self.alt_sim_forward_max_distance = 500
        self.alt_sim_backward_max_distance = 500
        self.act_sim_max_distance = 50000

        alt_sim_range = np.arange(
            -self.alt_sim_backward_max_distance,
            self.alt_sim_forward_max_distance + self.alt_sim_spacing,
            self.alt_sim_spacing,
        )
        act_sim_range = np.arange(
            -self.act_sim_max_distance,
            self.act_sim_max_distance + self.act_sim_spacing,
            self.act_sim_spacing,
        )
        self.alt_sim_matrix = np.array([alt_sim_range]).T * np.ones(
            [1, len(act_sim_range)]
        )
        self.act_sim_matrix = np.array([act_sim_range]) * np.ones(
            [len(alt_sim_range), 1]
        )

    def simulate(
        self, orbit_data: dict[str, npt.NDArray[np.float64] | float]
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
        """Runs the point-based simulation over the selected frames of the given track."""
        return super().simulate(orbit_data, dRSsim._simulate_frame_range)

    @staticmethod  # defined static to allow calling from ray avoiding the "pickling" of the object
    def _simulate_frame_range(
        id: int,
        n_processes: int,
        orbit_xyz: tuple[
            npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
        ],
        frame_range: npt.NDArray[np.int64],
        sim_n_samples: int,
        local_direction_g: npt.NDArray[np.float64],
        UTMNorthing_g: npt.NDArray[np.float64],
        UTMEasting_g: npt.NDArray[np.float64],
        UTMZone_g: list[str],
        rxwin_ref: float,
        input_scr_data_shifts_alignment_t: npt.NDArray[np.float64],
        dt: float,
        dem_obj: Any,
        geom_obj: Any,
        alt_sim_matrix: npt.NDArray[np.float64],
        act_sim_matrix: npt.NDArray[np.float64],
        gaussian_filter_sigma: int,
        further_pars: list[float] = [],
        roi_img: npt.NDArray | None = None,
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.uint8],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        tuple[dict[int, npt.NDArray[np.float64]], dict[int, npt.NDArray[np.float64]]],
    ]:
        """Per-frame point computation described in the class docstring: surface-grid DEM sampling, two-way travel times, image accumulation, and first-return/ground-distance/ROI bookkeeping."""
        _C = 299792458.0

        n_frames_to_be_simulated = len(frame_range)
        sim_image = np.zeros(
            [sim_n_samples, n_frames_to_be_simulated]
        )  # , dtype="int16")   # int16 and uint8 saves memory but considerably slows down the processing as numpy cannot handle ints in a fast way
        uncert_image = np.zeros(sim_image.shape, dtype="uint8")
        ground_distance_min_image = np.nan * np.ones(sim_image.shape)
        ground_distance_max_image = np.nan * np.ones(sim_image.shape)
        first_return_lats = np.zeros(n_frames_to_be_simulated)
        first_return_lons = np.zeros(n_frames_to_be_simulated)
        roi_lats = dict()
        roi_lons = dict()

        status_str_pre = f"\t{'|\t\t' * id}"
        status_str_post = "\t\t|" * (n_processes - id - 1)
        last_status = -1
        for count, i_frame_tbs in enumerate(frame_range):
            if count == n_frames_to_be_simulated - 1:
                new_status = 100
            else:
                new_status = int(count / n_frames_to_be_simulated * 10) * 10
            if new_status != last_status:
                print(f"{status_str_pre}{new_status:>2}{status_str_post}")
                last_status = new_status

            local_direction = local_direction_g[i_frame_tbs]
            UTMNorthing = UTMNorthing_g[i_frame_tbs]
            UTMEasting = UTMEasting_g[i_frame_tbs]
            UTMZone = UTMZone_g[i_frame_tbs]
            sc_x_cart_local = orbit_xyz[0][i_frame_tbs]
            sc_y_cart_local = orbit_xyz[1][i_frame_tbs]
            sc_z_cart_local = orbit_xyz[2][i_frame_tbs]

            UTMNorthing_surface_matrix = UTMNorthing + (
                alt_sim_matrix * np.sin(local_direction)
                - act_sim_matrix * np.cos(local_direction)
            )
            UTMEasting_surface_matrix = UTMEasting + (
                alt_sim_matrix * np.cos(local_direction)
                + act_sim_matrix * np.sin(local_direction)
            )
            Lat_surface_matrix, Long_surface_matrix = geom_obj.UTM_to_LL(
                UTMNorthing_surface_matrix, UTMEasting_surface_matrix, UTMZone
            )
            dem_surface_radius_matrix = dem_obj.get_dem_radius_from_lat_lon(
                Lat_surface_matrix, Long_surface_matrix
            )

            ground_distance_from_nadir = np.sqrt(
                (UTMNorthing_surface_matrix - UTMNorthing) ** 2
                + (UTMEasting_surface_matrix - UTMEasting) ** 2
            )

            dummy_check_matrix = dem_surface_radius_matrix == dem_obj.dummy_value

            # smooth the DEM to avoid "stripes" in the simulation
            dem_surface_radius_matrix = gaussian_filter(
                dem_surface_radius_matrix, gaussian_filter_sigma
            )

            if dummy_check_matrix.any():
                uncert_image[:, count] = 1
                dem_surface_radius_matrix[dummy_check_matrix] = (
                    0  # radius too far for being recorded in the simulation
                )

            x_cart_surface_matrix, y_cart_surface_matrix, z_cart_surface_matrix = (
                geom_obj.to_xyz_from_latlon_and_radius(
                    Lat_surface_matrix, Long_surface_matrix, dem_surface_radius_matrix
                )
            )

            d_surface = geom_obj.euclidean_distance_cart(
                sc_x_cart_local,
                sc_y_cart_local,
                sc_z_cart_local,
                x_cart_surface_matrix,
                y_cart_surface_matrix,
                z_cart_surface_matrix,
            )

            t_surface = 2.0 * d_surface / _C
            sample_pos_surface = np.round(
                (t_surface - rxwin_ref + input_scr_data_shifts_alignment_t[i_frame_tbs])
                / dt
            ).astype("int")  # use rxwin_ref instead of input_rxwin_data_t_local

            # set out-of-range sample positions to the farthest position (useful later for detecting the first-return position)
            sample_pos_surface[
                (sample_pos_surface < 0) | (sample_pos_surface >= sim_n_samples)
            ] = sim_n_samples

            # find first return coordinates
            first_return_ids = np.unravel_index(
                np.argmin(sample_pos_surface), sample_pos_surface.shape
            )
            first_return_lats[count] = Lat_surface_matrix[first_return_ids]
            first_return_lons[count] = Long_surface_matrix[first_return_ids]

            if roi_img is not None:
                roi_samples = np.where(roi_img[:, i_frame_tbs])[
                    0
                ]  # check where the roi_img is filled in this frame
                if len(roi_samples) > 0:
                    roi_ids = roi_img[roi_samples, i_frame_tbs]
                    roi_ids_unique = np.unique(roi_ids)
                    for ir in roi_ids_unique:
                        tmp_roi_samples = roi_samples[roi_ids == ir]
                        tmp_sample_pos = np.where(
                            np.isin(sample_pos_surface, tmp_roi_samples)
                        )
                        if ir not in roi_lats.keys():
                            roi_lats[ir] = np.array([])
                            roi_lons[ir] = np.array([])
                        roi_lats[ir] = np.concatenate(
                            (
                                roi_lats[ir],
                                Lat_surface_matrix[
                                    tmp_sample_pos[0], tmp_sample_pos[1]
                                ],
                            )
                        )
                        roi_lons[ir] = np.concatenate(
                            (
                                roi_lons[ir],
                                Long_surface_matrix[
                                    tmp_sample_pos[0], tmp_sample_pos[1]
                                ],
                            )
                        )

            ground_distance_from_nadir = ground_distance_from_nadir.flatten()
            sample_pos_surface = sample_pos_surface.flatten()

            # find and keep only the sample_pos_surface items within the simulation range
            sample_pos_surface_ok_ids = np.where(sample_pos_surface < sim_n_samples)[0]
            sample_pos_surface = sample_pos_surface[sample_pos_surface_ok_ids]

            # increment sim_image values by 1 each time they correspond to a valid sample_pos_surface
            np.add.at(sim_image[:, count], sample_pos_surface, 1)

            # keep only ground_distance_from_nadir items corresponding to valid sample_pos_surface items (selected before)
            ground_distance_from_nadir = ground_distance_from_nadir[
                sample_pos_surface_ok_ids
            ]

            # sort distances and sample positions both by increasing distance
            sorting_indexes = np.argsort(ground_distance_from_nadir)
            ground_distance_from_nadir = ground_distance_from_nadir[sorting_indexes]
            sample_pos_surface = sample_pos_surface[sorting_indexes]

            # for each simulation sample detect the corresponding ground distance
            # note: as the arrays were first ordered by ground distance, and considering that
            # "unique" selects the first instance of the array items, the "shortest" distance
            # associated to a certain sample is kept
            unique_sample_pos_surface, unique_sample_pos_surface_ids = np.unique(
                sample_pos_surface, return_index=True
            )
            unique_ground_distance_from_nadir = ground_distance_from_nadir[
                unique_sample_pos_surface_ids
            ]
            ground_distance_min_image[unique_sample_pos_surface, count] = (
                unique_ground_distance_from_nadir
            )

            # reverse the ordered arrays and use the sample principle to save the "longest" distances for every sample
            sample_pos_surface = np.flip(sample_pos_surface)
            ground_distance_from_nadir = np.flip(ground_distance_from_nadir)
            unique_sample_pos_surface, unique_sample_pos_surface_ids = np.unique(
                sample_pos_surface, return_index=True
            )
            unique_ground_distance_from_nadir = ground_distance_from_nadir[
                unique_sample_pos_surface_ids
            ]
            ground_distance_max_image[unique_sample_pos_surface, count] = (
                unique_ground_distance_from_nadir
            )

        return (
            sim_image,
            uncert_image,
            first_return_lats,
            first_return_lons,
            ground_distance_min_image,
            ground_distance_max_image,
            (roi_lats, roi_lons),
        )
