"""Tests for the calprocs module."""

import unittest


class TestCalprocs(unittest.TestCase):

    def _test_stefcal(self, nants=7, delta=1e-3, noise=None):
        """Check that specified stefcal calculates gains correct to within
        specified limit (default 1e-3)

        Parameters
        ----------
        nants     : number of antennas to use in the simulation, int
        delta     : limit to assess gains as equal, float
        noise     : whether to use noise in the simulation, boolean
        """
        from katsdpcal.calprocs import fake_vis
        vis, bl_ant_list, gains = fake_vis(nants, noise=noise)
        # solve for gains
        import numpy as np
        vis_and_conj = np.hstack((vis, vis.conj()))
        from katsdpcal.calprocs import stefcal
        calc_gains = stefcal(vis_and_conj, nants, bl_ant_list, weights=1.0, num_iters=100,
                             ref_ant=0, init_gain=None)
        for i in range(len(gains)):
            self.assertAlmostEqual(gains[i], calc_gains[i], delta=delta)

    def test_stefcal(self):
        """Check that stefcal calculates gains correct to within 1e-3"""
        self._test_stefcal()

    def test_stefcal_noise(self):
        """Check that stefcal calculates gains correct to within 1e-3, on noisy data"""
        self._test_stefcal(noise=1e-3)

    def _test_stefcal_timing(self, ntimes=200, nants=7, nchans=1, noise=None):
        """Time comparisons of the stefcal algorithms. Simulates data and solves for gains.

        Parameters
        ----------
        ntimes : number of times to run simulation and solve, int
        nants  : number of antennas to use in the simulation, int
        nchans : number of channel to use in the simulation, int
        noise  : whether to use noise in the simulation, boolean
        """
        import time
        import numpy as np
        from katsdpcal.calprocs import fake_vis
        from katsdpcal.calprocs import stefcal

        elapsed = 0.0

        for i in range(ntimes):
            vis, bl_ant_list, gains = fake_vis(nants, noise=noise)
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
