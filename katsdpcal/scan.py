"""Scan class to data and operations on data."""

import numpy as np
import copy

from katcal import calprocs
from katcal.calprocs import CalSolution

#--------------------------------------------------------------------------------------------------
#--- CLASS :  Scan
#--------------------------------------------------------------------------------------------------

class Scan(object):
    """
    Single scan of data with auxillary information.

    Parameters
    ----------
    data : dictionary
        Buffer of correlator data. Contains arrays of visibility, flag, weight and time.
    ti0, ti1: int
        Start and stop indices of the scan in the buffer arrays.
    dump_period : float
        Dump period of correlator data.
    nant : int
        Number of antennas in the array.
    bls_lookup : list of int, shape (2, number of baselines)
        List of antenna pairs for each baseline.
    target : string
        Name of target observed in the scan.
    chans : array of float
        Array of channel frequencies.
    corr : string, optional
        String to select correlation product, 'xc' for cross-correlations.

    Attributes
    ----------
    dump_period : float
        Dump period of correlator data.
    bl_mask : array of bool
        General mask for selecting usbset of the correlation products.
    corrprod_lookup : list of int, shape (number_of_baselines, 2)
        List of antenna pairs for each baseline.
    vis : array of float, complex64 (ntime, nchan, npol, nbl)
        Visibility data.
    flags : array of uint8, shape (ntime, nchan, npol, nbl)
        Flag data.
    weights : array of float64, shape (ntime, nchan, npol, nbl)
        Weight data.
    times : array of float, shape (ntime, nchan, npol, nbl)
        Times.
    target : string
        Name of target obsrved in the scan.
    modvis : array of float, complex64 (ntime, nchan, npol, nbl)
        Intermediate visibility product.
    nchan : int
        Number of frequency channels in the data.
    channel_freqs : list of float
        List of frequencies corresponding to channels (or channel indices, in the absence of frequencies)
    nant : int
        Number of antennas in the data.
    npol : int
        Number of polarisations in the data.

    """

    def __init__(self, data, ti0, ti1, dump_period, nant, bls_lookup, target, chans=None, corr='xc'):

        # cross-correlation mask. Must be np array so it can be used for indexing
        # if scan has explicitly been set up as a cross-correlation scan, seelct XC data only
        xc_mask = np.array([b0!=b1 for b0,b1 in bls_lookup])
        self.bl_mask = xc_mask if corr is 'xc' else slice(None)
        self.corrprod_lookup = bls_lookup[self.bl_mask]

        # get references to this time chunk of data
        # -- just using first polarisation for now
        # data format is:   (time x channels x pol x bl)
        #self._vis = data['vis'][ti0:ti1+1,:,0:2,:]
        self.vis = data['vis'][ti0:ti1+1,:,0:2,self.bl_mask]
        self.flags = data['flags'][ti0:ti1+1,:,0:2,self.bl_mask]
        self.weights = np.ones_like(self.flags,dtype=np.float)
        self.times = data['times'][ti0:ti1+1]
        self.target = target

        # intermediate product visibility - use sparingly!
        self.modvis = None

        # scan meta-data
        self.dump_period = dump_period
        self.nchan = self.vis.shape[1]
        self.channel_freqs = range(self.nchan) if chans is None else list(chans)
        self.nant = nant
        self.npol = 4

    def solint_from_nominal(self, input_solint):
        """
        determine appropriaye solution interval given nominal solution interval

        Inputs:
        ------
        input_solint : nominal solution interval

        Returns:
        solint : calculated optimal solution interval
        dumps_per_solint : number of dumps per solution interval
        """

        solint, dumps_per_solint = calprocs.solint_from_nominal(input_solint,self.dump_period,len(self.times))

    # ---------------------------------------------------------------------------------------------
    # Calibration solution functions

    def g_sol(self,input_solint,g0,REFANT,pre_apply=[]):
        """
        Solve for Gain

        Returns:
           CalSolution with soltype 'G'. Solution values have shape (time, pol, nant)
        """

        if len(pre_apply) > 0:
            self.modvis = copy.deepcopy(self.vis)
        else:
            self.modvis = self.vis

        for soln in pre_apply:
           self.modvis = self.apply(soln,origvis=False)

        # set up solution interval
        solint, dumps_per_solint = calprocs.solint_from_nominal(input_solint,self.dump_period,len(self.times))

        # first averge in time over solution interval

        ave_vis, ave_flags, ave_weights, av_sig, ave_times = calprocs.wavg_full_t(self.modvis,self.flags,self.weights,
                dumps_per_solint,axis=0,times=self.times)
        # second average over all channels
        ave_vis = calprocs.wavg(ave_vis,ave_flags,ave_weights,axis=1)

        # solve for gains G
        g_soln = calprocs.g_fit(ave_vis,self.corrprod_lookup,g0,REFANT)

        return CalSolution('G', g_soln, ave_times)

    def k_sol(self,chan_sample,k0,bp0,REFANT,pre_apply=[]):
        """
        Solve for Delay

        Returns:
           CalSolution with soltype 'K'. Solution values have shape (nant)
        """

        if len(pre_apply) > 0:
            self.modvis = copy.deepcopy(self.vis)
        else:
            self.modvis = self.vis

        for soln in pre_apply:
           self.modvis = self.apply(soln,origvis=False)

        # average over all time (no averaging over channel)
        ave_vis, ave_time = calprocs.wavg(self.modvis,self.flags,self.weights,times=self.times,axis=0)

        # solve for delay K
        k_soln = calprocs.k_fit(ave_vis,self.corrprod_lookup,self.channel_freqs,k0,bp0,REFANT,chan_sample=chan_sample)

        return CalSolution('K', k_soln, ave_time)

    def b_sol(self,bp0,REFANT,pre_apply=[]):
        """
        Solve for Bandpass

        Returns:
           CalSolution with soltype 'B'. Solution values have shape (chan, pol, nant)
        """

        if len(pre_apply) > 0:
            self.modvis = copy.deepcopy(self.vis)
        else:
            self.modvis = self.vis

        for soln in pre_apply:
           self.modvis = self.apply(soln,origvis=False)

        # average over all time (no averaging over channel)
        ave_vis, ave_time = calprocs.wavg(self.modvis,self.flags,self.weights,times=self.times,axis=0)

        # solve for bandpass
        b_soln = calprocs.bp_fit(ave_vis,self.corrprod_lookup,bp0,REFANT)

        return CalSolution('B', b_soln, ave_time)

    # ---------------------------------------------------------------------------------------------
    # solution application

    def _apply(self, solval, origvis=True, inplace=False):
        """
        Applies calibration solutions.
        Must already be interpolated to either full time or full frequency.

        Inputs:
        ------
        solval : multiplicative solution values to be applied to visibility data
        """

        if inplace is True:
            self._apply_inplace(solval)
            return
        else:
            return self._apply_newvis(solval, origvis=origvis)


    def _apply_newvis(self, solval, origvis=True):
        """
        Applies calibration solutions.
        Must already be interpolated to either full time or full frequency.

        Inputs:
        ------
        solval : multiplicative solution values to be applied to visibility data
                 ndarray, shape (time, chan, pol, ant) where time and chan are optional
        """

        outvis = copy.deepcopy(self.vis) if origvis else copy.deepcopy(self.modvis)

        # check solution and vis shapes are compatible
        if solval.shape[-2] !=  outvis.shape[-2]: raise Exception('Polarisation axes do not match!')

        for cp in range(len(self.corrprod_lookup)):
            outvis[...,cp] /= solval[...,self.corrprod_lookup[cp][0]]*(solval[...,self.corrprod_lookup[cp][1]].conj())

        return outvis

    def _apply_inplace(self, solval):
        """
        Applies calibration solutions.
        Must already be interpolated to either full time or full frequency.

        Inputs:
        ------
        solval : multiplicative solution values to be applied to visibility data
        """

        for cp in range(len(self.corrprod_lookup)):
            self.vis[...,cp] /= solval[...,self.corrprod_lookup[cp][0]]*(solval[...,self.corrprod_lookup[cp][1]].conj())

    def apply(self, soln, origvis=True, inplace=False):
        # set up more complex interpolation methods later
        if soln.soltype is 'G':
            # add empty channel dimension
            full_sol = np.expand_dims(soln.values,axis=1)
            return self._apply(full_sol,origvis=origvis,inplace=inplace)
        elif soln.soltype is 'K':
            # want shape (ntime, nchan, npol, nant)
            gain_shape = tuple(list(self.vis.shape[:-1]) + [self.nant])
            g_from_k = np.zeros(gain_shape,dtype=np.complex)
            for ci, c in enumerate(self.channel_freqs):
                g_from_k[:,ci,:,:] = np.exp(1.0j*soln.values*c)
            return self._apply(g_from_k,origvis=origvis,inplace=inplace)
        elif soln.soltype is 'B':
            return self._apply(soln.values,origvis=origvis,inplace=inplace)
        else:
            return ValueError('Solution type is invalid.')

    # ---------------------------------------------------------------------------------------------
    # interpolation

    def interpolate(self, solns):
        # set up more complex interpolation methods later
        soltype = solns.soltype
        if soltype is 'G':
            #return self.self_interpolate(num_dumps, **kwargs)
            return self.linear_interpolate(solns)
        if soltype is 'K':
            return self.inf_interpolate(solns)
        if soltype is 'B':
            return self.inf_interpolate(solns)

    def linear_interpolate(self, solns):
        values = solns.values
        times = solns.times

        real_interp = calprocs.interp_extrap_1d(times, values.real, kind='linear', axis=0)
        imag_interp = calprocs.interp_extrap_1d(times, values.imag, kind='linear', axis=0)

        interp_solns = real_interp(self.times) + 1.0j*imag_interp(self.times)
        return CalSolution(solns.soltype, interp_solns, self.times)

    def inf_interpolate(self, solns):
        values = solns.values
        interp_solns = np.repeat(np.expand_dims(values,axis=0),len(self.times),axis=0)
        return CalSolution(solns.soltype, interp_solns, self.times)





