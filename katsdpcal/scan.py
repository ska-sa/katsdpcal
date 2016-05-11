"""Scan class to data and operations on data."""

from . import calprocs
from .calprocs import CalSolution

import numpy as np
import copy

import ephem
import katpoint

import logging
logger = logging.getLogger(__name__)

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
    time_slice: slice
        Time slice of the scan in the buffer arrays.
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
    bl_slice : slice
        Slice for selecting susbset of the correlation products.
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
    model : list
        List of model components
    nchan : int
        Number of frequency channels in the data.
    channel_freqs : list of float
        List of frequencies corresponding to channels (or channel indices, in the absence of frequencies)
    nant : int
        Number of antennas in the data.
    npol : int
        Number of polarisations in the data.

    """

    def __init__(self, data, time_slice, dump_period, nant, bls_lookup, target, chans=None, corr='xc', logger=logger):

        # cross-correlation mask. Must be np array so it can be used for indexing
        # if scan has explicitly been set up as a cross-correlation scan, select XC data only
        # NOTE: This makes the asumption that the XC data are grouped at the beginning of the bl ordering,
        #       Followed by the AC data.
        # ******* Fancy indexing will not work here, as it returns a copy, not a view *******
        xc_mask = np.array([b0!=b1 for b0,b1 in bls_lookup])
        xc_slice = slice(np.where(xc_mask)[0][0], np.where(xc_mask)[0][-1]+1)
        self.bl_slice = xc_slice if corr is 'xc' else slice(None)
        self.corrprod_lookup = bls_lookup[self.bl_slice]

        # get references to this time chunk of data
        # -- just using first polarisation for now
        # data format is:   (time x channels x pol x bl)
        self.vis = data['vis'][time_slice,:,0:2,self.bl_slice]
        self.flags = data['flags'][time_slice,:,0:2,self.bl_slice]
        self.weights = np.ones_like(self.flags,dtype=np.float)

        self.cross_vis = data['vis'][time_slice,:,2:,self.bl_slice]
        self.cross_flags = data['flags'][time_slice,:,2:,self.bl_slice]
        self.cross_weights = np.ones_like(self.cross_flags,dtype=np.float)

        self.times = data['times'][time_slice]
        self.target = katpoint.Target(target)

        # intermediate product visibility - use sparingly!
        self.modvis = None

        # scan meta-data
        self.dump_period = dump_period
        self.nchan = self.vis.shape[1]
        # note - keep an eye on ordering of frequencies - increasing with index, or decreasing?
        self.channel_freqs = range(self.nchan) if chans is None else list(chans)
        self.nant = nant
        self.npol = 4

        # initialise model to unity
        self.model = 1.
        self.freqavg_model = 1.
        self.model_raw_params = None

        self.logger = logger

    def solint_from_nominal(self, input_solint):
        """
        Determine appropriate solution interval given nominal solution interval

        Inputs:
        ------
        input_solint : nominal solution interval

        Returns
        -------
        solint : calculated optimal solution interval
        dumps_per_solint : number of dumps per solution interval
        """

        solint, dumps_per_solint = calprocs.solint_from_nominal(input_solint,self.dump_period,len(self.times))

    # ---------------------------------------------------------------------------------------------
    # Calibration solution functions

    def g_sol(self,input_solint,g0,REFANT,pre_apply=[],**kwargs):
        """
        Solve for gain

        Parameters
        ----------
        input_solint : nominal solution interval to use for the fit
        g0 : initial estimate of gains for solver, shape (time, pol, nant)
        REFANT : reference antenna, int
        pre_apply : calibration solutions to apply, list of CalSolutions, optional

        Returns
        -------
        Gain CalSolution with soltype 'G', shape (time, pol, nant)
        """

        # initialise model, if this scan target has an associated model
        self._init_model(spectral=False)

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
        g_soln = calprocs.g_fit(ave_vis,self.corrprod_lookup,g0,REFANT,model=self.freqavg_model,**kwargs)

        return CalSolution('G', g_soln, ave_times)

    def kcross_sol(self,chan_ave,pre_apply=[]):
        """
        Solve for cross hand delay offset

        Parameters
        ----------
        chan_ave : channels to average together prior during fit
        pre_apply : calibration solutions to apply, list of CalSolutions, optional

        Returns
        -------
        Cross hand polarisation delay offset CalSolution with soltype 'KCROSS', shape (nant)
        """

        if len(pre_apply) > 0:
            self.modvis = copy.deepcopy(self.cross_vis)
        else:
            self.modvis = self.cross_vis

        for soln in pre_apply:
           self.modvis = self.apply(soln,origvis=False)

        # average over all time (no averaging over channel)
        ave_vis, av_flags, av_weights, av_sig = calprocs.wavg_full(self.modvis,self.cross_flags,self.cross_weights,axis=0)

        # solve for cross hand delay KCROSS
        # note that the kcross solver needs the flags because it averages the data
        #  (strictly it should need weights too, but deal with that leter when weights are meaningful)
        kcross_soln = calprocs.kcross_fit(ave_vis,av_flags,self.channel_freqs,chan_ave=chan_ave)
        return CalSolution('KCROSS', kcross_soln, np.average(self.times))

    def k_sol(self,REFANT,bchan=1,echan=0,chan_sample=1,pre_apply=[]):
        """
        Solve for delay

        Parameters
        ----------
        REFANT : reference antenna, int
        bchan : start channel for delay fit, int, optional
        echan : end channel for delay fit, int, optional
        chan_sample : channel sampling to use in delay fit, optional
        pre_apply : calibration solutions to apply, list of CalSolutions, optional

        Returns
        -------
        Delay CalSolution with soltype 'K', shape (2, nant)
        """

        if len(pre_apply) > 0:
            self.modvis = copy.deepcopy(self.vis)
        else:
            self.modvis = self.vis

        for soln in pre_apply:
           self.modvis = self.apply(soln,origvis=False)

        # average over all time, for specified channel range (no averaging over channel)
        if echan == 0: echan = None
        chan_slice = [slice(None),slice(bchan,echan),slice(None),slice(None)]
        ave_vis, ave_time = calprocs.wavg(self.modvis[chan_slice],self.flags[chan_slice],self.weights[chan_slice],times=self.times,axis=0)

        # solve for delay K, using specified channel range
        k_freqs = self.channel_freqs[bchan:echan]
        k_soln = calprocs.k_fit(ave_vis,self.corrprod_lookup,k_freqs,REFANT,chan_sample=chan_sample)

        return CalSolution('K', k_soln, ave_time)

    def b_sol(self,bp0,REFANT,pre_apply=[]):
        """
        Solve for bandpass

        Parameters
        ----------
        bp0 : initial estimate of bandpass for solver, shape (chan, pol, nant)
        REFANT : reference antenna, int
        pre_apply : calibration solutions to apply, list of CalSolutions, optional

        Returns
        -------
        Bandpass CalSolution with soltype 'B', shape (chan, pol, nant)
        """
        # initialise model, if this scan target has an associated model
        self._init_model(spectral=True)

        if len(pre_apply) > 0:
            self.modvis = copy.deepcopy(self.vis)
        else:
            self.modvis = self.vis

        for soln in pre_apply:
           self.modvis = self.apply(soln,origvis=False)

        # average over all time (no averaging over channel)
        ave_vis, ave_time = calprocs.wavg(self.modvis,self.flags,self.weights,times=self.times,axis=0)

        # solve for bandpass
        b_soln = calprocs.bp_fit(ave_vis,self.corrprod_lookup,bp0,REFANT,model=self.model)

        return CalSolution('B', b_soln, ave_time)

    # ---------------------------------------------------------------------------------------------
    # solution application

    def _apply(self, solval, origvis=True, inplace=False):
        """
        Applies calibration solutions.
        Must already be interpolated to either full time or full frequency.

        Parameters
        ----------
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

        Parameters
        ----------
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

        Parameters
        ----------
        solval : multiplicative solution values to be applied to visibility data
        """

        for cp in range(len(self.corrprod_lookup)):
            self.vis[...,cp] /= solval[...,self.corrprod_lookup[cp][0]]*(solval[...,self.corrprod_lookup[cp][1]].conj())

    def apply(self, soln, origvis=True, inplace=False):
        # set up more complex interpolation methods later
        if soln.soltype is 'G':
            # add empty channel dimension if necessary
            full_sol = np.expand_dims(soln.values,axis=1) if len(soln.values.shape) < 4 else soln.values
            return self._apply(full_sol,origvis=origvis,inplace=inplace)
        elif soln.soltype is 'K':
            # want shape (ntime, nchan, npol, nant)
            gain_shape = tuple(list(self.vis.shape[:-1]) + [self.nant])
            g_from_k = np.zeros(gain_shape,dtype=np.complex)
            for ci, c in enumerate(self.channel_freqs):
                g_from_k[:,ci,:,:] = np.exp(1.0j*2.*np.pi*soln.values*c)
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

        if len(times) < 2:
            # case of only one solution value being interpolated
            return self.inf_interpolate(solns)
        else:
            real_interp = calprocs.interp_extrap_1d(times, values.real, kind='linear', axis=0)
            imag_interp = calprocs.interp_extrap_1d(times, values.imag, kind='linear', axis=0)

            interp_solns = real_interp(self.times) + 1.0j*imag_interp(self.times)
            return CalSolution(solns.soltype, interp_solns, self.times)

    def inf_interpolate(self, solns):
        values = solns.values
        interp_solns = np.repeat(np.expand_dims(values,axis=0),len(self.times),axis=0)
        return CalSolution(solns.soltype, interp_solns, self.times)

    # ---------------------------------------------------------------------------------------------
    # model related functions

    def _create_model(self, max_offset=8., spectral=False):
        """
        Creates numpy array models for use in the solver
        *** more complex models will be divided through from the data - watch this space ***

        Models are currently implimented for:
          * a flat spectrum single point source in the phase centre
          * a varied spectrum point source in the phase centre

        Inputs
        ======
        max_offset : float
            The difference in positon away from the phase centre for a point to be considered at the phase centre [arcseconds]
            Default: 8 (= meerkat beam)
        spectral : boolean
            whether the model will have spectral dimensions

        Returns
        =======
        Model of the source. Dimensions can be:
           * scalar values 1
           * (nant, nant)
           * (nchan, 2, nant, nant)  (where the 2 is for polarisation)

        """
        # if models parameters have not been set for the scan, return unity model

        if self.model_raw_params is None:
            return 1.0

        ra0, dec0 = self.target.radec()

        # deal with easy case first!
        if (self.model_raw_params is not None) and (self.model_raw_params.size == 1):
            # check if source is at the phase centre
            ra = ephem.hours(self.model_raw_params['RA'].item())
            dec = ephem.degrees(self.model_raw_params['DEC'].item())

            if ephem.separation((ra,dec),(ra0,dec0)) < calprocs.arcsec_to_rad(max_offset):

                if (('a1' not in self.model_raw_params.dtype.names or self.model_raw_params['a1'] == 0) and
                    ('a2' not in self.model_raw_params.dtype.names or self.model_raw_params['a2'] == 0) and
                    ('a3' not in self.model_raw_params.dtype.names or self.model_raw_params['a3'] == 0)):
                    # spectral index is zero
                    S = 10.**self.model_raw_params['a0'].item()
                    model = S * (1.0 - np.eye(self.nant, dtype=np.complex))
                    self.logger.info('     Model: single point source, flat spectrum, flux: {0:03.4f} Jy'.format(S,))
                else:
                    model_params = [self.model_raw_params[a].item() for a in ['a0','a1','a2','a3']]
                    if spectral:
                        # full spectral model
                        freq_model = calprocs.flux(model_params,self.channel_freqs)
                        pol_freq_model = np.vstack([freq_model,freq_model]).T
                        self.logger.info('     Model: single point source, spectral model, average flux over {0:03.3f}-{1:03.3f} GHz: {2:03.4f} Jy'.format(self.channel_freqs[0]/1.e9, self.channel_freqs[-1]/1.e9, np.mean(freq_model)))
                        model = pol_freq_model[:,:,np.newaxis,np.newaxis] * (1.0 - np.eye(self.nant, dtype=np.complex))
                    else:
                        # use flux at the centre frequency
                        S_freq = calprocs.flux(model_params,self.channel_freqs[len(self.channel_freqs)/2])
                        model = S_freq * (1.0 - np.eye(self.nant, dtype=np.complex))
                        self.logger.info('     Model: single point source, flux at centre frequency {0:03.3f} GHz: {1:03.4f} Jy'.format(self.channel_freqs[len(self.channel_freqs)/2]/1.e9, S_freq))
        else:
            print 'do something more complex!'

        return model

    def _init_model(self, max_offset=8., spectral=False):
        """
        Initialises models for use in the solver.
        Checks for existing model scan attributes and creates models if models are not yet present

        Inputs
        ======
        max_offset : float
            The difference in positon away from the phase centre for a point to be considered at the phase centre [arcseconds]
            Default: 8 (= meerkat beam)
        spectral : boolean
            whether the model will have spectral dimensions
        """

        if spectral:
            if np.isscalar(self.model) and np.all(self.model == 1.):
                self.model = self._create_model(max_offset, spectral=True)
        else:
            if np.isscalar(self.freqavg_model) and np.all(self.freqavg_model == 1.):
                self.freqavg_model = self._create_model(max_offset, spectral=False)
