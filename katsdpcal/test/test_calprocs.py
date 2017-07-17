"""Tests for the calprocs module."""

import unittest
import time

import numpy as np

from katsdpcal.calprocs import fake_vis, stefcal, solint_from_nominal, wavg, _wavg_fallback


class TestCalprocs(unittest.TestCase):
    def test_solint_from_nominal(self):
        # requested interval shorter than dump
        self.assertEqual((4.0, 1), solint_from_nominal(0.5, 4.0, 6))
        # requested interval longer than scan
        self.assertEqual((24.0, 6), solint_from_nominal(100.0, 4.0, 6))
        # adjust interval to evenly divide the scan
        self.assertEqual((28.0, 7), solint_from_nominal(32.0, 4.0, 28))
        # no way to adjust interval to evenly divide the scan
        self.assertEqual((32.0, 8), solint_from_nominal(29.0, 4.0, 31))

    def _test_stefcal(self, nants=7, delta=1e-3, noise=None):
        """Check that specified stefcal calculates gains correct to within
        specified limit (default 1e-3)

        Parameters
        ----------
        nants     : number of antennas to use in the simulation, int
        delta     : limit to assess gains as equal, float
        noise     : amount of noise to use noise in the simulation, optional
        """
        rs = np.random.RandomState(seed=1)
        vis, bl_ant_list, gains = fake_vis(nants, noise=noise, random_state=rs)
        # solve for gains
        vis_and_conj = np.hstack((vis, vis.conj()))
        calc_gains = stefcal(vis_and_conj, nants, bl_ant_list, weights=1.0, num_iters=100,
                             ref_ant=0, init_gain=None)
        for i in range(len(gains)):
            self.assertAlmostEqual(gains[i], calc_gains[i], delta=delta)

    def test_stefcal(self):
        """Check that stefcal calculates gains correct to within 1e-3"""
        self._test_stefcal()

    def test_stefcal_noise(self):
        """Check that stefcal calculates gains correct to within 1e-3, on noisy data"""
        self._test_stefcal(noise=1e-4)

    def _test_stefcal_timing(self, ntimes=200, nants=7, nchans=1, noise=None):
        """Time comparisons of the stefcal algorithms. Simulates data and solves for gains.

        Parameters
        ----------
        ntimes : number of times to run simulation and solve, int
        nants  : number of antennas to use in the simulation, int
        nchans : number of channel to use in the simulation, int
        noise  : whether to use noise in the simulation, boolean
        """

        elapsed = 0.0

        rs = np.random.RandomState(seed=1)
        for i in range(ntimes):
            vis, bl_ant_list, gains = fake_vis(nants, noise=noise, random_state=rs)
            # add channel dimension, if required
            if nchans > 1:
                vis = np.repeat(vis[:, np.newaxis], nchans, axis=1).T
            vis_and_conj = np.concatenate((vis, vis.conj()), axis=-1)

            # solve for gains
            t0 = time.time()
            stefcal(vis_and_conj, nants, bl_ant_list, weights=1.0, num_iters=100,
                    ref_ant=0, init_gain=None)
            t1 = time.time()

            elapsed += t1-t0

        print 'average time:', elapsed/ntimes

    def test_stefcal_timing(self):
        """Time comparisons of the stefcal algorithms. Simulates data and solves for gains."""
        print '\nStefcal comparison:'
        self._test_stefcal_timing()

    def test_stefcal_timing_noise(self):
        """Time comparisons of the stefcal algorithms.

        Simulates data with noise and solves for gains.
        """
        print '\nStefcal comparison with noise:'
        self._test_stefcal_timing(noise=1e-3)

    def test_stefcal_timing_chans(self):
        """Time comparisons of the stefcal algorithms.

        Simulates multi-channel data and solves for gains.
        """
        nchans = 600
        print '\nStefcal comparison, {0} chans:'.format(nchans)
        self._test_stefcal_timing(nchans=nchans)


class TestWavg(unittest.TestCase):
    """Tests for :func:`katsdpcal.calprocs.wavg`"""
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
        weights[:, 1, 1] = [np.nan, 1.0, 0.0, 1.0, 0.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        flags[:, 1, 1] = [4, 0, 0, 4, 0, 0, 0, 0, 4, 4]
        # A completely NaN column and a completely flagged column => NaNs in output
        data[:, 2, 2] = np.nan
        flags[:, 0, 3] = 4

        expected = np.ones((3, 10), np.complex64)
        expected[1, 1] = 5.6 + 0.2j
        expected[2, 2] = np.nan
        expected[0, 3] = np.nan
        actual = wavg(data, flags, weights)
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
            expected = _wavg_fallback(self.data, self.flags, self.weights, axis=axis)
            actual = wavg(self.data, self.flags, self.weights, axis=axis)
            np.testing.assert_allclose(expected, actual, rtol=1e-6)
