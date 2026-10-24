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

"""Shared fast-time (range) processing helpers.
"""

import numpy as np
import numpy.typing as npt
from scipy.signal import chirp


class ProcessingUtils:
    """Container of static helpers for the range compression of the echo data."""
    def __init__(self) -> None:
        pass

    @staticmethod
    def create_synthetic_chirp_conj(frequency: float, bandwidth: float, chirp_duration: float, fast_time_sampling_frequency: float, range_weighting_window_original: npt.NDArray[np.float64] | float, SIZE_RANGE_FFT: int) -> npt.NDArray[np.complex128]:
        """Builds the conjugate spectrum of the synthetic TX chirp (down-chirp, zero-padded to the range FFT length and weighted by the given range window), to be multiplied with the echo spectra for matched filtering (pulse compression)."""
        t_chirp = np.arange(0, chirp_duration, 1. / fast_time_sampling_frequency)
        range_weighting_window = np.concatenate((range_weighting_window_original, np.zeros(int(SIZE_RANGE_FFT/2))), 0)

        f0 = frequency + bandwidth/2.
        t1 = chirp_duration
        f1 = frequency - bandwidth/2.
        chirp2_real = chirp(t_chirp, f0, t1, f1)

        chirp_fft = (np.fft.fft(chirp2_real, SIZE_RANGE_FFT)) / np.sqrt(SIZE_RANGE_FFT)
        chirp_fft = chirp_fft*range_weighting_window
        chirp_fft_conj = np.conj(chirp_fft).T

        return chirp_fft_conj
