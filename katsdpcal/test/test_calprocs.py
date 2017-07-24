"""Tests for the calprocs module."""

import unittest
import time

import numpy as np
import dask.array as da

from katsdpcal import calprocs


def unit(value):
    return value / np.abs(value)


class TestCalprocs(unittest.TestCase):
    def setUp(self):
        self.random_state = np.random.RandomState(seed=1)

    def test_solint_from_nominal(self):
        # requested interval shorter than dump
        self.assertEqual((4.0, 1), calprocs.solint_from_nominal(0.5, 4.0, 6))
        # requested interval longer than scan
        self.assertEqual((24.0, 6), calprocs.solint_from_nominal(100.0, 4.0, 6))
        # adjust interval to evenly divide the scan
        self.assertEqual((28.0, 7), calprocs.solint_from_nominal(32.0, 4.0, 28))
        # no way to adjust interval to evenly divide the scan
        self.assertEqual((32.0, 8), calprocs.solint_from_nominal(29.0, 4.0, 31))

    def _assert_gains_equal(self, expected, actual, *args, **kwargs):
        """Compares two sets of gains, allowing for a phase shift between
        them that is uniform along the last dimension.
        """
        actual = actual * unit(expected[..., [0]]) / unit(actual[..., [0]])
        np.testing.assert_allclose(expected, actual, *args, **kwargs)

    def _test_stefcal(self, nants=7, delta=1e-3, noise=None, **kwargs):
        """Check that specified stefcal calculates gains correct to within
        specified limit (default 1e-3)

        Parameters
        ----------
        nants     : number of antennas to use in the simulation, int
        delta     : limit to assess gains as equal, float
        noise     : amount of noise to use noise in the simulation, optional
        """
        vis, bl_ant_list, gains = calprocs.fake_vis(nants, noise=noise,
                                                    random_state=self.random_state)
        # solve for gains
        calc_gains = calprocs.stefcal(vis, nants, bl_ant_list, **kwargs)
        self._assert_gains_equal(gains, calc_gains, rtol=delta)

    def test_stefcal(self):
        """Check that stefcal calculates gains correct to within 1e-3"""
        self._test_stefcal()

    def test_stefcal_noise(self):
        """Check that stefcal calculates gains correct to within 1e-3, on noisy data"""
        self._test_stefcal(noise=1e-4)

    def test_stefcal_scalar_weight(self):
        """Test stefcal with a single scalar weight"""
        self._test_stefcal(weights=2.5)

    def test_stefcal_weight(self):
        """Test stefcal with non-trivial weights."""
        vis, bl_ant_list, gains = calprocs.fake_vis(7, random_state=self.random_state)
        weights = self.random_state.uniform(0.5, 4.0, vis.shape)
        # deliberate mess up one of the visibilities, but down-weight it
        assert bl_ant_list[1, 0] != bl_ant_list[1, 1]  # Check that it isn't an autocorr
        vis[1] = 1e6 - 1e6j
        weights[1] = 1e-12
        # solve for gains
        calc_gains = calprocs.stefcal(vis, 7, bl_ant_list, weights=weights)
        self._assert_gains_equal(gains, calc_gains, rtol=1e-3)

    def test_stefcal_init_gain(self):
        """Test stefcal with initial gains provided.

        This is just to check that it doesn't break. It doesn't test the effect
        on the number of iterations required.
        """
        init_gain = self.random_state.random_sample(7) + 1j * self.random_state.random_sample(7)
        self._test_stefcal(init_gain=init_gain)

    def test_stefcal_neg_ref_ant(self):
        """Test stefcal with negative `ref_ant`.

        This applies some normalisation to the returned phase. This doesn't
        check that aspect, just that it doesn't break anything.
        """
        self._test_stefcal(ref_ant=-1)

    def test_stefcal_multi_dimensional(self):
        """Test stefcal with higher number of dimensions."""
        vis = np.empty((4, 8, 28), np.complex128)
        gains = np.empty((4, 8, 7), np.complex128)
        for i in range(4):
            for j in range(8):
                vis[i, j], bl_ant_list, gains[i, j] = \
                    calprocs.fake_vis(7, random_state=self.random_state)
        calc_gains = calprocs.stefcal(vis, 7, bl_ant_list)
        self._assert_gains_equal(gains, calc_gains, rtol=1e-3)

    def test_stefcal_single_precision(self):
        """Test that single precision input yields a single precision output"""
        vis, bl_ant_list, gains = calprocs.fake_vis(7, random_state=self.random_state)
        vis = vis.astype(np.complex64)
        # solve for gains
        calc_gains = calprocs.stefcal(vis, 7, bl_ant_list)
        self.assertEqual(np.complex64, calc_gains.dtype)
        self._assert_gains_equal(gains, calc_gains, rtol=1e-3)

    def _test_stefcal_timing(self, ntimes=5, nants=32, nchans=8192, dtype=np.complex128, noise=None):
        """Time comparisons of the stefcal algorithms. Simulates data and solves for gains.

        Parameters
        ----------
        ntimes : number of times to run simulation and solve, int
        nants  : number of antennas to use in the simulation, int
        nchans : number of channel to use in the simulation, int
        noise  : whether to use noise in the simulation, boolean
        """

        elapsed = 0.0

        for i in range(ntimes):
            vis, bl_ant_list, gains = calprocs.fake_vis(nants, noise=noise,
                                                        random_state=self.random_state)
            vis = vis.astype(dtype)
            # add channel dimension, if required
            if nchans > 1:
                vis = np.repeat(vis[np.newaxis, :], nchans, axis=0)

            # solve for gains
            t0 = time.time()
            gains = calprocs.stefcal(vis, nants, bl_ant_list, num_iters=100)
            self.assertEqual(dtype, gains.dtype)
            t1 = time.time()

            elapsed += t1-t0

        print 'average time:', elapsed/ntimes

    def test_stefcal_timing(self):
        """Time comparisons of the stefcal algorithms. Simulates data and solves for gains."""
        print '\nStefcal comparison:'
        self._test_stefcal_timing()

    def test_stefcal_timing_single_precision(self):
        """Time comparisons of the stefcal algorithms. Simulates data and solves for gains."""
        print '\nStefcal comparison (single precision):'
        self._test_stefcal_timing(dtype=np.complex64)

    def test_stefcal_timing_noise(self):
        """Time comparisons of the stefcal algorithms.

        Simulates data with noise and solves for gains.
        """
        print '\nStefcal comparison with noise:'
        self._test_stefcal_timing(noise=1e-3)


class TestWavg(unittest.TestCase):
    """Tests for :func:`katsdpcal.calprocs.wavg`"""
    def setUp(self):
        shape = (10, 5, 3, 10)
        self.data = np.ones(shape, np.complex64)
        self.weights = np.ones(shape, np.float32)
        self.flags = np.zeros(shape, np.uint8)

    def test_basic(self):
        """Hand-coded test"""
        # Put in some NaNs and flags to check that they're handled correctly
        self.data[:, 0, 1, 1] = [1 + 1j, 2j, np.nan, 4j, np.nan, 5, 6, 7, 8, 9]
        self.weights[:, 0, 1, 1] = [np.nan, 1, 0, 1, 0, 2, 3, 4, 5, 6]
        self.flags[:, 0, 1, 1] = [4, 0, 0, 4, 0, 0, 0, 0, 4, 4]
        # A completely NaN column and a completely flagged column => NaNs in output
        self.data[:, 3, 2, 2] = np.nan
        self.flags[:, 4, 0, 3] = 4

        expected = np.ones((5, 3, 10), np.complex64)
        expected[0, 1, 1] = 5.6 + 0.2j
        expected[3, 2, 2] = np.nan
        expected[4, 0, 3] = np.nan
        actual = calprocs.wavg(da.asarray(self.data), da.asarray(self.flags), da.asarray(self.weights))
        self.assertEqual(np.complex64, actual.dtype)
        np.testing.assert_allclose(expected, actual, rtol=1e-6)


class TestWavgFull(unittest.TestCase):
    """Tests for :func:`katsdpcal.calprocs.wavg_full_t`"""
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

        out_data, out_flags, out_weights = calprocs.wavg_full_t(
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
        out_data, out_flags, out_weights = calprocs.wavg_full_t(
            self.data, self.flags, self.weights, 10)
        self.assertEqual(False, out_flags[0, 0, 0, 0])
        self.assertEqual(True, out_flags[0, 0, 0, 1])
