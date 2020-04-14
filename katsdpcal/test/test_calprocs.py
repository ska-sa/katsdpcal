"""Tests for the calprocs module."""

import unittest
import time

import numpy as np

from katsdpcal import calprocs

from katsdpcal.solutions import CalSolution, CalSolutionStore


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

    def test_stefcal_all_zeros(self):
        """Test stefcal with visibilities set to zero"""
        vis, weights, init_gain, bl_ant_list, gains = self.fake_data((10, 7))
        # Deliberately set the visibilities to zero in all baselines for one channel.
        # Set the gain to NaN in this channel
        # Have to unwrap it (in case it is a dask array) and rewrap it.
        vis = np.array(vis)
        vis[1] = 0j
        gains[1] = np.nan
        vis = self._wrap_array(vis)
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

        print('average time:', elapsed/ntimes)

    def test_stefcal_timing(self):
        """Time comparisons of the stefcal algorithms. Simulates data and solves for gains."""
        print('\nStefcal comparison:')
        self._test_stefcal_timing()

    def test_stefcal_timing_single_precision(self):
        """Time comparisons of the stefcal algorithms. Simulates data and solves for gains."""
        print('\nStefcal comparison (single precision):')
        self._test_stefcal_timing(dtype=np.complex64)

    def test_stefcal_timing_noise(self):
        """Time comparisons of the stefcal algorithms.

        Simulates data with noise and solves for gains.
        """
        print('\nStefcal comparison with noise:')
        self._test_stefcal_timing(noise=1e-3)


class TestWavgFullF(unittest.TestCase):
    """Tests for :func:`katsdpcal.calprocs.wavg_full_f`"""
    def setUp(self):
        shape = (5, 10, 3, 10)
        self.data = np.ones(shape, np.complex64)
        self.weights = np.ones(shape, np.float32)
        self.flags = np.zeros(shape, np.uint8)
        # Put in some NaNs, zeros, high weights and flags to check that they're handled correctly
        self.data[0, :, 1, 1] = [1 + 1j, 2j, np.nan, 4j, np.nan, 5, 0, 7, 8, 9]
        self.weights[0, :, 1, 1] = [np.nan, 1, 0, 1, 0, 2, 3, 2e19, 5, 6]
        self.flags[0, :, 1, 1] = [4, 0, 0, 4, 0, 0, 0, 0, 4, 4]

        # A completely NaN column and a completely flagged column => zeros in output
        self.data[1, :, 2, 2] = np.nan
        self.flags[2, :, 0, 3] = 4
        # A completely zero column and column with all high weights => Zeros in output
        self.data[1, :, 2, 3] = 0j
        self.weights[2, :, 0, 4] = 2e19

    def test_basic(self):
        out_shape = (5, 3, 3, 10)
        expected_data = np.ones(out_shape, np.complex64)
        expected_weights = np.ones(out_shape, np.float32) * 4
        expected_weights[:, 2, ...] = 2    # Only two samples added together
        expected_flags = np.zeros(out_shape, np.bool_)
        expected_data[0, :, 1, 1] = [2j, 5, 0j]
        expected_weights[0, :, 1, 1] = [1, 2, 0]
        expected_flags[0, :, 1, 1] = [False, False, True]

        expected_data[1, :, 2, 2] = 0j
        expected_weights[1, :, 2, 2] = 0
        expected_flags[1, :, 2, 2] = True

        expected_data[2, :, 0, 3] = 0j
        expected_weights[2, :, 0, 3] = 0
        expected_flags[2, :, 0, 3] = True

        expected_data[1, :, 2, 3] = 0j
        expected_weights[1, :, 2, 3] = 0
        expected_flags[1, :, 2, 3] = True

        expected_data[2, :, 0, 4] = 0j
        expected_weights[2, :, 0, 4] = 0
        expected_flags[2, :, 0, 4] = True

        out_data, out_flags, out_weights = calprocs.wavg_full_f(
            self.data, self.flags, self.weights, 4)

        np.testing.assert_allclose(expected_data, out_data, rtol=1e-6)
        np.testing.assert_equal(expected_flags, out_flags)
        np.testing.assert_allclose(expected_weights, out_weights, rtol=1e-6)

    def test_threshold(self):
        """Test thresholding on flags"""
        # This assumes the threshold default is 0.8 - it's not currently
        # settable via wavg_full_f.
        self.flags[0, :7, 0, 0] = 4
        self.flags[0, :8, 0, 1] = 4
        out_data, out_flags, out_weights = calprocs.wavg_full_f(
            self.data, self.flags, self.weights, 10)
        self.assertEqual(False, out_flags[0, 0, 0, 0])
        self.assertEqual(True, out_flags[0, 0, 0, 1])


class TestWavgFlagsF(unittest.TestCase):
    def test_4dim(self):
        flags = np.zeros((3, 6, 4, 1), np.uint8)
        excise = np.zeros(flags.shape, np.bool_)
        expected = np.zeros((3, 3, 4, 1), np.uint8)
        # No excision
        flags[0, 2, 1, 0] = 0xa0
        flags[0, 3, 1, 0] = 0x2a
        expected[0, 1, 1, 0] = 0xaa
        # Partial excision
        flags[1, 0, 0:2, 0] = 0xa0
        flags[1, 1, 0:2, 0] = 0x2a
        excise[1, 0, 0, 0] = True
        excise[1, 1, 1, 0] = True
        expected[1, 0, 0, 0] = 0x2a
        expected[1, 0, 1, 0] = 0xa0
        # Full excision
        flags[2, 4, 3, 0] = 0x70
        flags[2, 5, 3, 0] = 0x17
        excise[2, 4:6, 3, 0] = True
        expected[2, 2, 3, 0] = 0x77
        # Test
        actual = calprocs.wavg_flags_f(flags, 2, excise, axis=1)
        np.testing.assert_array_equal(actual, expected)
        # Test again with axis wrapping
        actual = calprocs.wavg_flags_f(flags, 2, excise, axis=-3)
        np.testing.assert_array_equal(actual, expected)

    def test_1dim(self):
        flags = np.array([0xa0, 0x2a, 0x70, 0x17, 0xa2, 0x0a], np.uint8)
        excise = np.array([False, False, True, False, True, True])
        expected = np.array([0xaa, 0x17, 0xaa], np.uint8)
        actual = calprocs.wavg_flags_f(flags, 2, excise, axis=0)
        np.testing.assert_array_equal(actual, expected)

    def test_bad_axis(self):
        flags = np.zeros((6, 6), np.uint8)
        excise = np.zeros((6, 6), np.uint8)
        with self.assertRaises(np.AxisError):
            calprocs.wavg_flags_f(flags, 2, excise, axis=2)
        with self.assertRaises(np.AxisError):
            calprocs.wavg_flags_f(flags, 2, excise, axis=-3)

    def test_unsupported_axis(self):
        flags = np.zeros((6, 6, 6, 6), np.uint8)
        excise = np.zeros((6, 6, 6, 6), np.uint8)
        with self.assertRaises(NotImplementedError):
            calprocs.wavg_flags_f(flags, 2, excise, axis=0)
        with self.assertRaises(NotImplementedError):
            calprocs.wavg_flags_f(flags, 2, excise, axis=2)

    def test_shape_mismatch(self):
        flags = np.zeros((6, 5), np.uint8)
        excise = np.zeros((6, 6), np.uint8)
        with self.assertRaises(ValueError):
            calprocs.wavg_flags_f(flags, 2, excise, axis=0)

    def test_unequal_split(self):
        flags = np.zeros((3, 6, 4, 1), np.uint8)
        excise = np.zeros(flags.shape, np.bool_)
        with self.assertRaises(ValueError):
            calprocs.wavg_flags_f(flags, 4, excise, axis=1)


class TestNormaliseComplex(unittest.TestCase):
    """Tests for :func:`katsdpcal.calprocs.normalise_complex`"""
    def setUp(self):
        shape = (6, 7, 2)
        self.data = np.ones(shape, np.complex64) + 1.j
        self.data[:, 2, 0] += 1+1j
        # Put in some zeros and NaNs to check they're handled correctly
        self.data[2, 2, 0] = 0
        self.data[3, 1, 0] = np.nan
        self.data[:, 0, 0] = [np.sqrt(2)+0j, 1+1j, 2+2j, 1+1j, 0, np.nan]
        self.data[5, :, 0] = np.nan
        # columns which consist of only NaNs and/or zeros => 1 + 0j in the normalisation factor
        self.data[:, 4, 0] = np.nan
        self.data[:, 5, 0] = 0
        self.data[:, 6, 0] = [0, 0, 0, np.nan, np.nan, np.nan]

        self.expected = np.ones((1, 7, 2), np.complex64)
        self.expected *= 1/np.sqrt(2) * np.exp(-1j * np.pi/4)
        expected_angle = -1j * np.pi / 4 * np.array([3. / 5., 1, 4. / 5., 1, 0, 0, 0])
        expected_amp = np.array([1, 1, 5. / 8, 1, 0, 0, 0]) / np.sqrt(2)
        self.expected[0, :, 0] = expected_amp * np.exp(expected_angle)
        self.expected[0, 4:, 0] = 1

    def test_unweighted(self):
        """Test normalisation factor is correct without weights"""
        # check for all axes
        for i in [0, 1, 2, -1, -2, -3]:
            data_i = np.moveaxis(self.data, 0, i)
            expected_i = np.moveaxis(self.expected, 0, i)
            norm_factor = calprocs.normalise_complex(data_i, axis=i)
            # check normalisation factor is correct
            np.testing.assert_allclose(expected_i, norm_factor, rtol=1e-6)
            # check results are the same for unweighted data and all equal weights
            weights_i = np.ones(data_i.shape) * 2
            norm_factor = calprocs.normalise_complex(data_i, weights_i, axis=i)
            np.testing.assert_allclose(expected_i, norm_factor, rtol=1e-6)

    def test_weighted(self):
        """Test normalisation factor is correct with weights"""
        weights = np.ones(self.data.shape)
        # Include some NaNs and zeros in weights to check they are handled correctly
        # Data with NaN weights should be ignored in normalisation factor calculation
        weights[:, 0, 0] = [1, 0, 2, np.nan, 1, 5]
        expected = self.expected
        expected_angle = -1j * np.pi / 8
        expected_amp = 4 / (5 * np.sqrt(2))
        expected[:, 0, 0] = expected_amp * np.exp(expected_angle)

        # check for all axes
        for i in [0, 1, 2, -1, -2, -3]:
            data_i = np.moveaxis(self.data, 0, i)
            weights_i = np.moveaxis(weights, 0, i)
            expected_i = np.moveaxis(expected, 0, i)

            norm_factor = calprocs.normalise_complex(data_i, weights_i, axis=i)
            np.testing.assert_allclose(expected_i, norm_factor, rtol=1e-6)


class TestKAnt(unittest.TestCase):
    """Tests for :func:`katsdpcal.calprocs.K_ant'"""
    def test(self):
        ntimes = 3
        nchans = 2
        nants = 5

        uvw = np.ones((3, ntimes, nants))
        uvw[:, 1, 3] = [70, 5, 1]
        l = 0.1  # noqa: E741
        m = 0.2
        n = np.sqrt(1 - l * l - m * m)
        wl = np.array([0.1, 0.4])
        kant = np.zeros((ntimes, nchans, nants), np.complex64)
        kant = calprocs.K_ant(uvw, l, m, wl, kant)

        expected_kant = np.zeros((ntimes, nchans, nants), np.complex64)
        expected_kant[:, 0] = np.exp(2j * np.pi * (3 + (n - 1) / 0.1))
        expected_kant[:, 1] = np.exp(2j * np.pi * (0.75 + (n - 1) / 0.4))
        expected_kant[1, 0, 3] = np.exp(2j * np.pi * (80 + (n - 1) / 0.1))
        expected_kant[1, 1, 3] = np.exp(2j * np.pi * (20 + (n - 1) / 0.4))

        np.testing.assert_equal(kant, expected_kant)


class TestMeasureFlux(unittest.TestCase):
    """Tests for :func:`katsdpcal.calprocs.measure_flux'"""
    def setUp(self):
        ntimes = 4
        self.gains1 = np.ones((ntimes, 5), dtype=np.complex64)
        self.times1 = np.linspace(0.2, 0.3, ntimes)
        self.targ1 = ['targ1'] * ntimes

        self.gains2 = np.ones((ntimes, 5), dtype=np.float32) * 10. + \
            1j * np.ones((ntimes, 5), dtype=np.float32) * 10.
        self.times2 = np.linspace(0.3, 0.4, ntimes)
        self.targ2 = ['targ2'] * ntimes

        # All gains NaN
        self.gains3 = np.full((1, 5), np.nan, dtype=np.complex64)
        self.times3 = [0.5]
        self.targ3 = ['targ3']

        # Gains with mismatched shape
        self.gains4 = np.ones((2, 6), dtype=np.complex64)
        self.times4 = 0.6
        self.targ4 = 'targ4'

        self.gains1[0, :] = np.nan
        self.gains1[:, 2] = np.nan
        self.gains1[2, 3] = np.nan
        self.gains1[3, 2] = 10.0

        flux = np.ones((ntimes, 5), dtype=np.float32) * 1. / np.sqrt(2) + \
            1. / np.sqrt(2) * 1j * np.ones((ntimes, 5), dtype=np.float32)
        timesflux = np.linspace(0.1, 0.2, ntimes)
        targflux = ['flux'] * ntimes
        flux[0, 2] = np.nan + 1j*np.nan
        flux[1, :] = np.nan + 1j*np.nan
        self.f_store = self.Gsolution_store(targflux, timesflux, flux)

        self.expect_f1 = 1.0
        expect_ratio1 = [1., 1., 10., 1., 1.]
        self.expect_std1 = 2. * 1.253 * np.std(expect_ratio1) / np.sqrt(5)
        self.expect_f2 = (np.abs(10 + 10j) / 1.) ** 2.

    def Gsolution_store(self, targs, times, gains):
        store = CalSolutionStore('G')
        for targ, tm, gain in zip(targs, times, gains):
            soln = CalSolution('G', gain, tm, targ)
            store.add(soln)
        return store

    def test_basic(self):
        # Check that measure_flux function produces expected result
        g_store = self.Gsolution_store(self.targ1, self.times1, self.gains1)
        prod_f, prod_f_std = calprocs.measure_flux(self.f_store, g_store, 0., 1.)
        self.assertTrue('targ1' in prod_f)
        self.assertTrue('targ1' in prod_f_std)
        np.testing.assert_almost_equal(prod_f['targ1'], self.expect_f1)
        np.testing.assert_almost_equal(prod_f_std['targ1'], self.expect_std1)

    def test_multiple_targets(self):
        # Test the measure_flux function with gains produced from multiple targets
        g_store = self.Gsolution_store(self.targ1 + self.targ2 + self.targ3,
                                       np.hstack([self.times1, self.times2, self.times3]),
                                       np.vstack([self.gains1, self.gains2, self.gains3]))
        prod_f, _ = calprocs.measure_flux(self.f_store, g_store, 0., 1.)
        self.assertTrue('targ1' in prod_f)
        self.assertTrue('targ2' in prod_f)
        self.assertTrue('targ3' in prod_f)
        np.testing.assert_almost_equal(prod_f['targ1'], self.expect_f1)
        np.testing.assert_almost_equal(prod_f['targ2'], self.expect_f2)
        np.testing.assert_equal(prod_f['targ3'], np.nan)

    def test_mismatched_shapes(self):
        # Test a set of gains with mismatched shapes don't produce an F product
        g_store = self.Gsolution_store(self.targ1, self.times1, self.gains1)
        g_store.add(CalSolution('G', self.gains4, self.times4, self.targ4))
        prod_f, _ = calprocs.measure_flux(self.f_store, g_store, 0., 1.)
        self.assertTrue('targ1' in prod_f)
        self.assertFalse('targ4' in prod_f)

    def test_no_scaled_gains(self):
        # Test that without scaled gains there is no F product.
        g_store = self.Gsolution_store(self.targ2, self.times2, self.gains2)
        prod_f, _ = calprocs.measure_flux(self.f_store, g_store, 0.3, 1.)
        self.assertFalse(prod_f)


class TestAddModelVis(unittest.TestCase):
    """Tests for :func:`katsdpcal.calprocs.add_model_vis`"""
    def test(self):
        shape = (3, 5, 4)
        kant = np.ones(shape, np.complex64)
        kant[1, 3, :] = [1, 2, 1+1j, 2+2j]
        ant1 = np.array([0, 0, 0, 1, 1, 2])
        ant2 = np.array([1, 2, 3, 2, 3, 3])

        out_shape = (3, 5, 6)
        model = np.ones(out_shape, np.complex64)
        S = np.array([5, 4, 3, 2, 1])

        out_model = calprocs.add_model_vis(kant, ant1, ant2, S, model)

        expected_model = np.ones(out_shape, np.complex64) + S[np.newaxis, :, np.newaxis]
        expected_model[1, 3, :] = [5, 3-2j, 5-4j, 5-4j, 9-8j, 9]

        np.testing.assert_equal(out_model, expected_model)


class TestCalcSnr(unittest.TestCase):
    """Tests for :func:`katsdpcal.calprocs.calc_snr`"""
    def setUp(self):
        shape = (7, 3, 2)
        self.data = np.ones(shape, np.float32)
        # Put in some NaNs to check they're handled correctly
        self.data[:, 0, 0] = [1, 0, 2, 3, np.nan, 4, 5]
        self.data[4, 1, 0] = np.nan
        # A completely NaN column => NaNs in the output
        self.data[:, 2, 0] = np.nan

        self.weights = np.ones(shape, np.float32)
        # Set some weights to zeros
        self.weights[:, 0, 0] = [1, 2, 3, 2, 5, 0, 0]

        self.expected_rms = np.ones((3, 2), np.float32)
        self.expected_rms[:, 0] = [np.sqrt(31 / 8), 1, np.nan]

    def test_basic(self):
        for axis in [0, 1, 2, -1, -2, -3]:
            data = np.moveaxis(self.data, 0, axis)
            weights = np.moveaxis(self.weights, 0, axis)
            rms = calprocs.calc_rms(data, weights, axis)
            np.testing.assert_equal(self.expected_rms, rms)


class TestPoorAntennaFlags(unittest.TestCase):
    """Tests for :func:`katsdpcal.calprocs.poor_antenna_flags`"""
    def test(self):
        nants = 5
        shape = (2, 10, 2, nants*(nants-1)//2)
        vis = np.ones(shape, np.complex64)
        weights = np.ones(shape, np.float32)
        bls_lookup = np.array([
            (0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
            (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)
            ])

        # Add some noise to visibility angles
        random_state = np.random.RandomState(seed=1)
        angle_noise = 0.2*(random_state.random_sample(shape)) - 0.1
        vis *= np.exp(1.j * angle_noise)

        # Add higher noise and NaNs to baselines to particular antennas
        # high noise antennas should be flagged, all NaN'd antennas should not
        high_bls = [np.any(1 == b) for b in bls_lookup]
        nan_bls = [np.any(2 == b) for b in bls_lookup]
        high_noise = 0.8 * random_state.random_sample((4, 10)) - 0.4
        vis[0, :, 0, high_bls] = np.exp(1.j * high_noise)
        vis[0, :, 0, nan_bls] = np.nan

        expected_flags = np.zeros(shape).astype(bool)
        expected_flags[0, :, 0, high_bls] = True

        flags = calprocs.poor_antenna_flags(vis, weights, bls_lookup, 0.2)
        np.testing.assert_equal(expected_flags, flags)


class TestSnrAntenna(unittest.TestCase):
    """Tests for :func:`katsdpcal.calprocs.snr_antenna`"""
    def setUp(self):
        nants = 5
        self.shape = (2, 10, 2, nants*(nants-1)//2)
        self.vis = np.ones(self.shape, np.complex64)
        self.weights = np.ones(self.shape, np.float32)
        self.bls_lookup = np.array([
            (0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
            (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)
            ])

        # alter visibility angles from zero
        self.vis *= np.exp(1.j * 0.01)
        vis_angle = np.array([(0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.04, 0.04, 0.01)])
        self.vis[0] = np.exp(1.j * vis_angle)

        # alter the weights
        self.weights[0] = np.array([(2, 2, 2, 1, 2, 2, 1, 1, 1, 4)])

        # Put in a NaN and check it is excluded from calculated SNR
        self.vis[0, 5, 1, 0] = np.nan

        # Put in a completely NaN'ed antenna => NaN snr
        nan_bls = [np.any(2 == b) for b in self.bls_lookup]
        self.vis[1, :, 0, nan_bls] = np.nan

        # Set one antenna to have higher noise than the others
        self.high_bls = [np.any(1 == b) for b in self.bls_lookup]
        self.vis[1, :, 1, self.high_bls] = np.exp(1.j * 0.5)

        self.expected = 100 * np.ones((2, 2, nants), np.float32)
        self.expected[0] = [100, 100 / np.sqrt(16 / 7), 100 / np.sqrt(6),
                            100 / np.sqrt(30 / 9), 100 / np.sqrt(25 / 7)]
        self.expected[0, 1, 1] = 100 / (np.sqrt(158 / 68))
        self.expected[1, 0, 2] = np.nan

    def test_basic(self):
        expected = self.expected
        expected[1, 1, :] = 100 / np.sqrt(2503 / 4)
        expected[1, 1, 1] = 100 / 50

        snr = calprocs.snr_antenna(self.vis, self.weights, self.bls_lookup)
        np.testing.assert_allclose(expected, snr, rtol=1e-3)

    def test_mask(self):
        expected = self.expected
        ant_flags = np.ones(self.shape, bool)
        # Flag the noisier antenna, check that SNR excludes baselines
        # to this antenna when calculating SNR for other antennas.
        ant_flags[1, :, 1, self.high_bls] = True

        expected[1, 1, 1] = 2
        snr = calprocs.snr_antenna(self.vis, self.weights, self.bls_lookup, ant_flags)
        np.testing.assert_allclose(expected, snr, rtol=1e3)


class TestFluxDensityModel(unittest.TestCase):
    """Tests for :class:`katsdpcal.calprocs.FluxDensityModel`"""
    def _test_str(self, desc_str, expected_flux):
        """Tests flux density produced with a string input."""
        flux_model = calprocs.FluxDensityModel(desc_str)
        flux = flux_model.flux_density([100, 2000])
        np.testing.assert_equal(flux, expected_flux)

    def _test_inputs(self, min_freq, max_freq, coefs, expected_flux):
        """Tests flux density produced with min_freq, max_freq and coefs inputs"""
        flux_model = calprocs.FluxDensityModel(min_freq, max_freq, coefs)
        flux = flux_model.flux_density([100, 2000])
        np.testing.assert_equal(flux, expected_flux)

    def test_baars(self):
        """Correct flux produced with Baars_MHz polynomial"""
        self._test_str('10.0 1000.0 5 2 3 0 0', [10 ** 21.0, np.nan])
        self._test_str('(10.0 1000.0 baars_mhz 5 2 3)', [10 ** 21.0, np.nan])
        self._test_inputs(10.0, 1000.0, (5, 2, 3), [10 ** 21.0, np.nan])
        self._test_inputs(10.0, 1000.0, ('baars_mhz', 5, 2, 3), [10 ** 21.0, np.nan])

    def test_wsclean_ord(self):
        """Correct flux produced with wsclean ordinary polynomial"""
        self._test_str('10.0 1000.0 wsclean_ord_mhz 20.0 2 3 1', [30.0, np.nan])
        self._test_inputs(10.0, 1000.0, ('wsclean_ord_mhz', 20, 2, 3, 1),
                          [30.0, np.nan])

    def test_wsclean_log(self):
        desc_str = '10.0 1000.0 wsclean_log_mhz 1.0 0.0001 2 3 -1 0 0.25'
        self._test_str(desc_str, [np.exp(12.0), np.nan])
        self._test_inputs(10, 1000, ('wsclean_log_mhz', 1, 0.0001, 2, 3, -1, 0 , 0.25),
                          [np.exp(12.0), np.nan])

    def test_unknown_polynomial(self):
        desc_str = '10.0 1000.0 baars_ghz 5 2 3'
        flux_model = calprocs.FluxDensityModel(desc_str)
        with self.assertRaises(ValueError):
            flux_model.flux_density([100, 2000])
