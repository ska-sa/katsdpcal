"""Tests for the calprocs module."""

import unittest
import time

import numpy as np

from katsdpcal import calprocs


def unit(value):
    return value / np.abs(value)


class TestCalprocs(unittest.TestCase):
    def test_solint_from_nominal(self):
        # requested interval shorter than dump
        self.assertEqual((4.0, 1), calprocs.solint_from_nominal(0.5, 4.0, 6))
        # requested interval longer than scan
        self.assertEqual((24.0, 6), calprocs.solint_from_nominal(100.0, 4.0, 6))
        # adjust interval to evenly divide the scan
        self.assertEqual((28.0, 7), calprocs.solint_from_nominal(32.0, 4.0, 28))
        # no way to adjust interval to evenly divide the scan
        self.assertEqual((32.0, 8), calprocs.solint_from_nominal(29.0, 4.0, 31))
        # single dump
        self.assertEqual((4.0, 1), calprocs.solint_from_nominal(4.0, 4.0, 1))


class TestStefcal(unittest.TestCase):
    """Tests for :func:`katsdpcal.calprocs.stefcal`."""
    def setUp(self):
        self.random_state = np.random.RandomState(seed=1)

    def _call_stefcal(self, *args, **kwargs):
        """Wrapper to call stefcal. This is overloaded in subclasses to test
        different versions of stefcal.
        """
        return calprocs.stefcal(*args, **kwargs)

    def _wrap_array(self, array):
        """Wrapper to convert an array to the type needed for the test. This is
        overloaded in subclasses.
        """
        if array is None:
            return None
        elif array.ndim == 0:
            return array[()]   # Convert 0-d arrays to scalars
        else:
            return array

    def _assert_gains_equal(self, expected, actual, *args, **kwargs):
        """Compares two sets of gains, allowing for a phase shift between
        them that is uniform along the last dimension.
        """
        actual = actual * unit(expected[..., [0]]) / unit(actual[..., [0]])
        np.testing.assert_allclose(expected, actual, *args, **kwargs)

    def fake_data(self, shape=(7,), weights_shape=None, init_gain_shape=None,
                  dtype=np.complex128, noise=None):
        """Generate a set of fake data for a stefcal case.

        Parameters
        ----------
        shape : tuple
            Shape of the gains (last dimension is number of antennas)
        weights_shape : tuple, optional
            If specified, shape of returned weights. Otherwise, no weights
            will be generated.
        init_gain_shape : tuple, optional
            If specified, shape of returned initial gains. Otherwise, no
            initial gains will be generated.
        dtype : np.dtype
            Complex data type for the visibilities
        noise : float, optional
            Noise level to add

        Returns
        -------
        vis, weight, init_gain : array-like
            Data arrays for stefcal (wrapped by :meth:`_wrap_array`)
        bl_ant_list : array
            Baseline list to pass to stefcal
        gains : np.ndarray
            Expected gains (*not* wrapped by :meth:`_wrap_array`)
        """
        vis, bl_ant_list, gains = calprocs.fake_vis(
            shape, noise=noise, random_state=self.random_state)
        vis = vis.astype(dtype)
        if weights_shape is not None:
            weights = self.random_state.uniform(0.5, 2.0, weights_shape)
            weights = weights.astype(vis.real.dtype)
        else:
            weights = None
        if init_gain_shape is not None:
            init_gain = self.random_state.standard_normal(init_gain_shape) \
                    + 1j * self.random_state.standard_normal(init_gain_shape)
            init_gain = init_gain.astype(vis.dtype)
        else:
            init_gain = None
        return (self._wrap_array(vis), self._wrap_array(weights), self._wrap_array(init_gain),
                bl_ant_list, gains)

    def _test_stefcal(self, shape=(7,), weights_shape=None, init_gain_shape=None,
                      dtype=np.complex128, noise=None, delta=1e-3, ref_ant=0, *args, **kwargs):
        """Check that specified stefcal calculates gains correct to within
        specified limit (default 1e-3)

        Parameters
        ----------
        shape : tuple
            Shape of the gains (last dimension corresponding to number of antennas)
        weights_shape : tuple, optional
            Shape of `weights` parameter, or ``None`` to not use weights
        init_gain_shape : tuple, optional
            Shape of `init_gain` parameter, or ``None`` to not use initial gains
        dtype : np.dtype, optional
            Complex dtype of the visibilities
        noise : float, optional
            Noise level to add to data
        delta : float, optional
            Floating-point tolerance
        *args, **kwargs
            Passed to stefcal
        """
        vis, weights, init_gain, bl_ant_list, gains = self.fake_data(
            shape, weights_shape, init_gain_shape, dtype, noise)
        calc_gains = self._call_stefcal(vis, shape[-1], bl_ant_list, weights, ref_ant,
                                        init_gain, *args, **kwargs)
        self._assert_gains_equal(gains, calc_gains, rtol=delta)
        self.assertEqual(dtype, calc_gains.dtype)

    def test_stefcal(self):
        """Check that stefcal calculates gains correct to within 1e-3"""
        self._test_stefcal()

    def test_stefcal_noise(self):
        """Check that stefcal calculates gains correct to within 1e-3, on noisy data"""
        self._test_stefcal(noise=1e-4)

    def test_stefcal_scalar_weight(self):
        """Test stefcal with a single scalar weight"""
        self._test_stefcal(weights_shape=())

    def test_stefcal_weight(self):
        """Test stefcal with non-trivial weights."""
        vis, weights, init_gain, bl_ant_list, gains = self.fake_data((7,), (28,))
        # Deliberately mess up one of the visibilities, but down-weight it.
        # Have to unwrap it (in case it is a dask array) and rewrap it.
        assert bl_ant_list[1, 0] != bl_ant_list[1, 1]  # Check that it isn't an autocorr
        vis = np.array(vis)
        weights = np.array(weights)
        vis[1] = 1e6 - 1e6j
        weights[1] = 1e-12
        vis = self._wrap_array(vis)
        weights = self._wrap_array(weights)
        # solve for gains
        calc_gains = self._call_stefcal(vis, 7, bl_ant_list, weights=weights)
        self._assert_gains_equal(gains, calc_gains, rtol=1e-3)

    def test_stefcal_init_gain(self):
        """Test stefcal with initial gains provided.

        This is just to check that it doesn't break. It doesn't test the effect
        on the number of iterations required.
        """
        self._test_stefcal(init_gain_shape=(7,))

    def test_stefcal_neg_ref_ant(self):
        """Test stefcal with negative `ref_ant`.

        This applies some normalisation to the returned phase. This test
        doesn't check that aspect, just that it doesn't break anything.
        """
        self._test_stefcal(ref_ant=-1)

    def test_stefcal_multi_dimensional(self):
        """Test stefcal with higher number of dimensions."""
        self._test_stefcal(shape=(4, 8, 7))

    def test_stefcal_broadcast(self):
        """Different shapes for the inputs, to test broadcasting"""
        self._test_stefcal(shape=(4, 8, 7), weights_shape=(4, 1, 28), init_gain_shape=(8, 7))

    def test_stefcal_single_precision(self):
        """Test that single precision input yields a single precision output"""
        self._test_stefcal(dtype=np.complex64)

    def _test_stefcal_timing(self, ntimes=5, shape=(8192, 32), *args, **kwargs):
        """Time comparisons of the stefcal algorithms. Simulates data and solves for gains.

        Parameters
        ----------
        ntimes : number of times to run simulation and solve, int
        nants  : number of antennas to use in the simulation, int
        nchans : number of channel to use in the simulation, int
        noise  : whether to use noise in the simulation, boolean
        """

        elapsed = 0.0

        vis, weights, init_gain, bl_ant_list, gains = self.fake_data(shape, *args, **kwargs)
        for i in range(ntimes):
            # solve for gains
            t0 = time.time()
            gains = self._call_stefcal(vis, shape[-1], bl_ant_list, weights, 0, init_gain)
            t1 = time.time()
            if 'dtype' in kwargs:
                self.assertEqual(kwargs['dtype'], gains.dtype)
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


class TestWavgFull_F(unittest.TestCase):
    """Tests for :func:`katsdpcal.calprocs.wavg_full_f`"""
    def setUp(self):
        shape = (5, 10, 3, 10)
        self.data = np.ones(shape, np.complex64)
        self.weights = np.ones(shape, np.float32)
        self.flags = np.zeros(shape, np.uint8)
        # Put in some NaNs and flags to check that they're handled correctly
        self.data[0, :, 1, 1] = [1 + 1j, 2j, np.nan, 4j, np.nan, 5, 6, 7, 8, 9]
        self.weights[0, :, 1, 1] = [np.nan, 1, 0, 1, 0, 2, 3, 4, 5, 6]
        self.flags[0, :, 1, 1] = [4, 0, 0, 4, 0, 0, 0, 0, 4, 4]
        # A completely NaN column and a completely flagged column => NaNs in output
        self.data[1, :, 2, 2] = np.nan
        self.flags[2, :, 0, 3] = 4

    def test_basic(self):
        out_shape = (5, 3, 3, 10)
        expected_data = np.ones(out_shape, np.complex64)
        expected_weights = np.ones(out_shape, np.float32) * 4
        expected_weights[:, 2, ...] = 2    # Only two samples added together
        expected_flags = np.zeros(out_shape, np.bool_)
        expected_data[0, :, 1, 1] = [2j, 56.0 / 9.0, np.nan]
        expected_weights[0, :, 1, 1] = [1, 9, 0]
        expected_flags[0, :, 1, 1] = [False, False, True]

        expected_data[1, :, 2, 2] = np.nan
        expected_weights[1, :, 2, 2] = 0

        expected_data[2, :, 0, 3] = np.nan
        expected_weights[2, :, 0, 3] = 0
        expected_flags[2, :, 0, 3] = True

        out_data, out_flags, out_weights = calprocs.wavg_full_f(
            self.data, self.flags, self.weights, 4)
        np.testing.assert_allclose(expected_data, out_data, rtol=1e-6)
        np.testing.assert_equal(expected_flags, out_flags)
        np.testing.assert_allclose(expected_weights, out_weights, rtol=1e-6)

    def test_threshold(self):
        """Test thresholding on flags"""
        # This assumes the threshold default is 0.8 - it's not currently
        # settable via wavg_full_f.
        self.flags[0, :8, 0, 0] = 4
        self.flags[0, :9, 0, 1] = 4
        out_data, out_flags, out_weights = calprocs.wavg_full_f(
            self.data, self.flags, self.weights, 10)
        self.assertEqual(False, out_flags[0, 0, 0, 0])
        self.assertEqual(True, out_flags[0, 0, 0, 1])
