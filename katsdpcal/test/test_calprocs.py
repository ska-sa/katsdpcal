"""Tests for the calprocs module."""

import unittest

class TestCalprocs(unittest.TestCase):

    def _test_stefcal(self,algorithm,nants=7,delta=1e-3,noise=False):
        """Check that specified stefcal calculates gains correct to within specified limit (default 1e-3)

        Parameters
        ----------
        algorithm : stefcal algorithm, string
        nants     : number of antennas to use in the simulation, int
        delta     : limit to asses gains as equal, float
        noise     : whether to use noise in the simulation, boolean
        """
        from katsdpcal.calprocs import fake_vis
        vis, bl_ant_list, gains = fake_vis(nants)
        # solve for gains
        import numpy as np
        vis_and_conj = np.hstack((vis, vis.conj()))
        from katsdpcal.calprocs import stefcal
        calc_gains = stefcal(vis_and_conj, nants, bl_ant_list, weights=1.0, num_iters=100, ref_ant=0, init_gain=None, algorithm=algorithm)
        for i in range(len(gains)): self.assertAlmostEqual(gains[i],calc_gains[i],delta=1.e-3)

    def test_adi_stefcal(self):
        """Check that adi stefcal calculates gains correct to within 1e-3"""
        self._test_stefcal('adi')
        
    def test_adi_stefcal_noise(self):
        """Check that adi stefcal calculates gains correct to within 1e-3, on noisy data"""
        self._test_stefcal('adi',noise=True)
    
    def test_schwardt_stefcal(self):
        """Check that schwardt stefcal calculates gains correct to within 1e-3"""
        self._test_stefcal('schwardt')
        
    def test_schwardt_stefcal_noise(self):
        """Check that schwardt stefcal calculates gains correct to within 1e-3, on noisy data"""
        self._test_stefcal('schwardt',noise=True)

    def test_schwardt_adi_stefcal(self):
        """Check that schwardt_adi stefcal calculates gains correct to within 1e-3"""
        self._test_stefcal('schwardt_adi')
        
    def test_schwardt_adi_stefcal_noise(self):
        """Check that schwardt_adi stefcal calculates gains correct to within 1e-3, on noisy data"""
        self._test_stefcal('schwardt_adi',noise=True)
        
    def test_stefcal_comparison(self):
        """Check that schwardt_adi and adi stefcal gains are equivalent within 1e-3"""
        from katsdpcal.calprocs import fake_vis
        nants = 7
        vis, bl_ant_list, gains = fake_vis(nants)
        # solve for gains
        import numpy as np
        vis_and_conj = np.hstack((vis, vis.conj()))
        from katsdpcal.calprocs import stefcal
        calc_gains_0 = stefcal(vis_and_conj, nants, bl_ant_list, weights=1.0, num_iters=100, ref_ant=0, init_gain=None, algorithm='adi')
        calc_gains_1 = stefcal(vis_and_conj, nants, bl_ant_list, weights=1.0, num_iters=100, ref_ant=0, init_gain=None, algorithm='schwardt_adi')

        for i in range(len(gains)): self.assertAlmostEqual(calc_gains_0[i],calc_gains_1[i],delta=1.e-3)

    def _test_stefcal_timing(self,ntimes=200,nants=7,nchans=1,noise=False):
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

        t_adi, t_schwardt, t_adi_schwardt, t_default = 0, 0, 0, 0

        for i in range(ntimes):
            vis, bl_ant_list, gains = fake_vis(nants, noise=noise)
            # add channel dimension, if required
            if nchans > 1: vis = np.repeat(vis[:,np.newaxis],nchans,axis=1).T
            vis_and_conj = np.concatenate((vis, vis.conj()),axis=-1)

            # solve for gains, using each algorithm
            t0 = time.time()
            calc_gains = stefcal(vis_and_conj, nants, bl_ant_list, weights=1.0, num_iters=100, ref_ant=0, init_gain=None, algorithm='adi')
            t1 = time.time()
            calc_gains = stefcal(vis_and_conj, nants, bl_ant_list, weights=1.0, num_iters=100, ref_ant=0, init_gain=None, algorithm='schwardt')
            t2 = time.time()
            calc_gains = stefcal(vis_and_conj, nants, bl_ant_list, weights=1.0, num_iters=100, ref_ant=0, init_gain=None, algorithm='adi_schwardt')
            t3 = time.time()
            calc_gains = stefcal(vis_and_conj, nants, bl_ant_list, weights=1.0, num_iters=100, ref_ant=0, init_gain=None)
            t4 = time.time()

            t_adi += t1-t0
            t_schwardt += t2-t1
            t_adi_schwardt += t3-t2
            t_default += t4-t3

        print 'adi:         ', t_adi/ntimes
        print 'schwardt:    ', t_schwardt/ntimes
        print 'adi schwardt:', t_adi_schwardt/ntimes
        print 'default:     ', t_default/ntimes

    def test_stefcal_timing(self):
        """Time comparisons of the stefcal algorithms. Simulates data and solves for gains."""
        print '\nStefcal comparison:'
        self._test_stefcal_timing()

    def test_stefcal_timing_noise(self):
        """Time comparisons of the stefcal algorithms. Simulates data with noise and solves for gains."""
        print '\nStefcal comparison with noise:'
        self._test_stefcal_timing(noise=True)

    def test_stefcal_timing_chans(self):
        """Time comparisons of the stefcal algorithms. Simulates multi-channel data and solves for gains."""
        nchans = 600
        print '\nStefcal comparison, {0} chans:'.format(nchans)
        self._test_stefcal_timing(nchans=nchans)
