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

"""Unit tests of the shared range-processing helper
(sopa.processing.utils.ProcessingUtils): the shape and dtype of the synthetic
chirp conjugate spectrum, the effect of the range weighting window, and the
matched-filter property that the compressed chirp peaks at zero lag.
"""

import numpy as np
from scipy.signal import chirp

from sopa.processing.utils import ProcessingUtils

FREQUENCY = 20e6
BANDWIDTH = 10e6
CHIRP_DURATION = 85.05e-6
FS = 80e6 / 3.0
SIZE_RANGE_FFT = 4096


def test_shape_and_dtype() -> None:
    """Checks that the chirp conjugate spectrum is a complex128 vector of the
    requested FFT length."""
    out = ProcessingUtils.create_synthetic_chirp_conj(FREQUENCY, BANDWIDTH, CHIRP_DURATION, FS, np.ones(SIZE_RANGE_FFT // 2), SIZE_RANGE_FFT)
    assert out.shape == (SIZE_RANGE_FFT,)
    assert out.dtype == np.dtype("complex128")


def test_zero_window_zeroes_spectrum() -> None:
    """Checks that a null range weighting window nulls the whole spectrum."""
    out = ProcessingUtils.create_synthetic_chirp_conj(FREQUENCY, BANDWIDTH, CHIRP_DURATION, FS, np.zeros(SIZE_RANGE_FFT // 2), SIZE_RANGE_FFT)
    np.testing.assert_array_equal(out, np.zeros(SIZE_RANGE_FFT, dtype="complex128"))


def test_matched_filter_peaks_at_zero_lag() -> None:
    """Checks the matched-filter property: multiplying the chirp spectrum by
    the conjugate synthetic spectrum and transforming back yields the chirp
    autocorrelation, whose maximum is at zero lag."""
    window = np.ones(SIZE_RANGE_FFT // 2)
    chirp_conj = ProcessingUtils.create_synthetic_chirp_conj(FREQUENCY, BANDWIDTH, CHIRP_DURATION, FS, window, SIZE_RANGE_FFT)
    t_chirp = np.arange(0, CHIRP_DURATION, 1.0 / FS)
    f0 = FREQUENCY + BANDWIDTH / 2.0
    f1 = FREQUENCY - BANDWIDTH / 2.0
    chirp_time = chirp(t_chirp, f0, CHIRP_DURATION, f1)
    compressed = np.fft.ifft(np.fft.fft(chirp_time, SIZE_RANGE_FFT) * chirp_conj)
    assert int(np.argmax(np.abs(compressed))) == 0
    assert np.abs(compressed[0]) > np.abs(compressed[1:]).max()
