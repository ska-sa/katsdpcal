"""Tests for the calprocs module."""

import unittest

class TestCalprocs(unittest.TestCase):
            
    def fake_gain_solution(self,algorithm,delta=1e-3,noise=False,gains=None):
        """Create fake visibbilities from random gains, then calculates gains from the visibilities"""
        import numpy as np
    
        # create antenna lists
        nants = 6
        antlist = range(nants)
    
        list1 = np.array([])
        for a,i in enumerate(range(nants-1,0,-1)):
            list1 = np.hstack([list1,np.ones(i)*a])
        list1 = np.hstack([list1,antlist])
        list1 = np.int_(list1)

        list2 = np.array([], dtype=np.int)
        mod_antlist = antlist[1:]
        for i in (range(0,len(mod_antlist))):
            list2 = np.hstack([list2,mod_antlist[:]])
            mod_antlist.pop(0)
        list2 = np.hstack([list2,antlist])
    
        # create fake gains, if gains are not given as input
        if gains is None: gains = np.random.random(nants)
    
        # create fake corrupted visibilities
        nbl = nants*(nants+1)/2
        vis = np.ones([nbl])
        # corrupt vis with gains
        for i,j in zip(list1,list2): vis[(list1==i)&(list2==j)] *= gains[i]*gains[j]
        # if requested, corrupt vis with noise
        if noise:
            vis_noise = (np.random.random(vis.shape)-0.5)*1.e-4
            vis = vis+vis_noise

        # set up solver inputs
        vis_and_conj = np.hstack((vis, vis.conj())) 
        antlist1 = np.concatenate((list1, list2))
        antlist2 = np.concatenate((list2, list1))

        # solve for gains
        from katcal.calprocs import stefcal    
        calc_gains = stefcal(vis_and_conj, nants, antlist1, antlist2, weights=1.0, num_iters=100, ref_ant=0, init_gain=None, algorithm=algorithm)
        
        return gains, calc_gains
        
    def _test_stefcal(self,algorithm,delta=1e-3,noise=False,gains=None):
        """Check that specified stefcal calculates gains correct to within 1e-3"""
        gains, calc_gains = self.fake_gain_solution('adi')
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
        
    def test_stefcal_comparison(self):
        """Check that schwardt and adi stefcal gains are equivalent within 1e-3, on noisy data"""
        gains, adi = self.fake_gain_solution('adi',noise=True)
        gains, schwardt = self.fake_gain_solution('schwardt',noise=True,gains=gains)
        for i in range(len(gains)): self.assertAlmostEqual(adi[i],schwardt[i],delta=1.e-3)
        