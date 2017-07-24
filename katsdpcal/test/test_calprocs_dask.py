"""Tests for :mod:`katsdpcal.calprocs_dask`."""

import unittest
import time

import numpy as np
import dask.array as da

from katsdpcal import calprocs, calprocs_dask
from . import test_calprocs


def as_dask(arr):
    return da.from_array(arr, chunks=arr.shape)


def unit(value):
    return value / np.abs(value)


class TestStefcal(test_calprocs.TestStefcal):
    """Tests for :func:`katsdpcal.calprocs_dask.stefcal`."""
    def _call_stefcal(self, rawvis, num_ants, corrprod_lookup, weights=None,
                      ref_ant=0, init_gain=None, *args, **kwargs):
        rawvis = da.asarray(rawvis)
        if weights is not None:
            weights = da.asarray(weights)
        if init_gain is not None:
            init_gain = da.asarray(init_gain)
        return calprocs_dask.stefcal(rawvis, num_ants, corrprod_lookup, weights, ref_ant,
                                     init_gain, *args, **kwargs).compute()

    def _wrap_array(self, array):
        # Overloads the base class to convert arrays to dask arrays
        if array is None:
            return None
        elif array.ndim >= 1 and array.shape[0] == 8192:
            # It's a timing test. Do something sensible to get good performance
            return da.from_array(array, chunks=(512,) + array.shape[1:])
        else:
            # Give it some random chunking, to check that the chunks of
            # multiple arrays get mapped together correctly.
            chunks = []
            for s in array.shape:
                if s > 1:
                    chunks.append(self.random_state.randint(1, s - 1))
                else:
                    chunks.append(1)
            return da.from_array(array, chunks=tuple(chunks))


class TestWavg(unittest.TestCase):
    """Tests for :func:`katsdpcal.calprocs_dask.wavg`"""
    def setUp(self):
        shape = (5, 10, 3, 10)
        self.data = np.ones(shape, np.complex64)
        self.weights = np.ones(shape, np.float32)
        self.flags = np.zeros(shape, np.uint8)

    def test_fallback(self):
        """Test the behaviour of the fallback path"""
        # Use 3D arrays, since that puts us off the numba path
        data = self.data[0]
        weights = self.weights[0]
        flags = self.flags[0]
        # Put in some NaNs and flags to check that they're handled correctly
        data[:, 1, 1] = [1 + 1j, 2j, np.nan, 4j, np.nan, 5, 6, 7, 8, 9]
        weights[:, 1, 1] = [np.nan, 1, 0, 1, 0, 2, 3, 4, 5, 6]
        flags[:, 1, 1] = [4, 0, 0, 4, 0, 0, 0, 0, 4, 4]
        # A completely NaN column and a completely flagged column => NaNs in output
        data[:, 2, 2] = np.nan
        flags[:, 0, 3] = 4

        expected = np.ones((3, 10), np.complex64)
        expected[1, 1] = 5.6 + 0.2j
        expected[2, 2] = np.nan
        expected[0, 3] = np.nan
        actual = calprocs_dask.wavg(as_dask(data), as_dask(flags), as_dask(weights))
        self.assertEqual(np.complex64, actual.dtype)
        np.testing.assert_allclose(expected, actual, rtol=1e-6)

    def test_numba(self):
        rs = np.random.RandomState(seed=1)
        shape = self.data.shape
        self.data[:] = rs.standard_normal(shape) + 1j * rs.standard_normal(shape)
        self.data[:] += rs.choice([0, np.nan], shape, p=[0.95, 0.05])
        self.weights[:] = rs.uniform(0.0, 4.0, shape)
        self.weights[:] += rs.choice([0, np.nan], shape, p=[0.95, 0.05])
        self.flags[:] = rs.choice([0, 4], shape, p=[0.95, 0.05])
        for axis in range(0, 2):
            expected = calprocs_dask._wavg_fallback(
                as_dask(self.data), as_dask(self.flags), as_dask(self.weights), axis=axis)
            actual = calprocs_dask.wavg(
                as_dask(self.data), as_dask(self.flags), as_dask(self.weights), axis=axis)
            np.testing.assert_allclose(expected, actual, rtol=1e-6)


class TestWavgFull(unittest.TestCase):
    """Tests for :func:`katsdpcal.calprocs_dask.wavg_full_t`"""
    def setUp(self):
        shape = (10, 5, 3, 10)
        self.data = np.ones(shape, np.complex64)
        self.weights = np.ones(shape, np.float32)
        self.flags = np.zeros(shape, np.uint8)
        # Put in some NaNs and flags to check that they're handled correctly
        self.data[:, 0, 1, 1] = [1 + 1j, 2j, np.nan, 4j, np.nan, 5, 6, 7, 8, 9]
        self.weights[:, 0, 1, 1] = [np.nan, 1, 0, 1, 0, 2, 3, 4, 5, 6]
        self.flags[:, 0, 1, 1] = [4, 0, 0, 4, 0, 0, 0, 0, 4, 4]
        # A completely NaN column and a completely flagged column => NaNs in output
        self.data[:, 1, 2, 2] = np.nan
        self.flags[:, 2, 0, 3] = 4

    def test_basic(self):
        out_shape = (3, 5, 3, 10)
        expected_data = np.ones(out_shape, np.complex64)
        expected_weights = np.ones(out_shape, np.float32) * 4
        expected_weights[2, ...] = 2    # Only two samples added together
        expected_flags = np.zeros(out_shape, np.bool_)
        expected_data[:, 0, 1, 1] = [2j, 56.0 / 9.0, np.nan]
        expected_weights[:, 0, 1, 1] = [1, 9, 0]
        expected_flags[:, 0, 1, 1] = [True, False, True]

        expected_data[:, 1, 2, 2] = np.nan
        expected_weights[:, 1, 2, 2] = 0

        expected_data[:, 2, 0, 3] = np.nan
        expected_weights[:, 2, 0, 3] = 0
        expected_flags[:, 2, 0, 3] = True

        out_data, out_flags, out_weights = calprocs_dask.wavg_full_t(
            self.data, self.flags, self.weights, 4)
        np.testing.assert_allclose(expected_data, out_data, rtol=1e-6)
        np.testing.assert_equal(expected_flags, out_flags)
        np.testing.assert_allclose(expected_weights, out_weights, rtol=1e-6)

    def test_threshold(self):
        """Test thresholding on flags"""
        # This assumes the threshold default is 0.3 - it's not currently
        # settable via wavg_full_t.
        self.flags[:2, 0, 0, 0] = 4
        self.flags[:4, 0, 0, 1] = 4
        out_data, out_flags, out_weights = calprocs_dask.wavg_full_t(
            self.data, self.flags, self.weights, 10)
        self.assertEqual(False, out_flags[0, 0, 0, 0])
        self.assertEqual(True, out_flags[0, 0, 0, 1])
