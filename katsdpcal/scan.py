"""Scan class to data and operations on data."""

from . import calprocs
from .calprocs import CalSolution

import numpy as np
import copy
from time import time

import ephem
import katpoint
from scipy.constants import c as light_speed

import logging
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# --- CLASS :  Scan
# --------------------------------------------------------------------------------------------------


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
    ants : array of string
        Array of antenna description strings.
    refant : int
        Index of reference antenna in antenna description list.
    array_position : string
        Description string of array centre position.
    corr : string, optional
        String to select correlation product, 'xc' for cross-correlations.
    logger : logger
        Logger

    Attributes
    ----------
    xc_mask : numpy array of bool
        Mask for selecting cross-correlation data
    bl_slice : slice
        Slice for selecting susbset of the correlation products.
    corrprod_lookup : list of int, shape (number_of_baselines, 2)
        List of antenna pairs for each baseline.
    vis : array of float, complex64 (ntime, nchan, 2, nbl)
        Visibility data.
    flags : array of uint8, shape (ntime, nchan, 2, nbl)
        Flag data.
    weights : array of float32, shape (ntime, nchan, 2, nbl)
        Weight data.
    cross_vis : array of float, complex64 (ntime, nchan, 2, nbl)
        Cross polarisation visibility data.
    cross_flags : array of uint8, shape (ntime, nchan, 2, nbl)
        Cross polarisation flag data.
    cross_weights : array of float32, shape (ntime, nchan, 2, nbl)
        Cross polarisation weight data.
    timestamps : array of float, shape (ntime, nchan, npol, nbl)
        Times.
    target : katpoint Target
        Phase centre of the scan.
    modvis : array of float, complex64 (ntime, nchan, npol, nbl)
        Intermediate visibility product.
    uvw : array of float, shape (3, ntime, nchan, nbl)
        UVW coordinates
    dump_period : float
        Dump period of correlator data.
    nchan : int
        Number of frequency channels in the data.
    channel_freqs : list of float
        List of frequencies corresponding to channels (or channel indices, in the absence of frequencies).
    npol : int
        Number of polarisations in the data.
    nant : int
        Number of antennas in the data
    antenna_descriptions : list of string
        Description strings for each antenna
    refant : int
        Index of reference antenna in antenna description list
    array_position : string
        Description sctring for array position
    model_raw_params : list
        List of model components
    model : scalar or array
        Model of the visibilities
    logger : logger
        logger
    """

    def __init__(self, data, time_slice, dump_period, nant, bls_lookup, target, chans=None, ants=None, refant=0, array_position=None, corr='xc', logger=logger):

        # cross-correlation mask. Must be np array so it can be used for indexing
        # if scan has explicitly been set up as a cross-correlation scan, select XC data only
        # NOTE: This makes the asumption that the XC data are grouped at the beginning of the bl ordering,
        #       Followed by the AC data.
        # ******* Fancy indexing will not work here, as it returns a copy, not a view *******
        self.xc_mask = np.array([b0 != b1 for b0, b1 in bls_lookup])
        try:
            xc_slice = slice(np.where(self.xc_mask)[0][0], np.where(self.xc_mask)[0][-1]+1)
        except IndexError:
            # not XC data
            xc_slice = slice(None)
        self.bl_slice = xc_slice if corr is 'xc' else slice(None)
        self.corrprod_lookup = bls_lookup[self.bl_slice]

        # get references to this time chunk of data, parallel hand polarisations only
        # data format is:   (time x channels x pol x bl)
        self.vis = data['vis'][time_slice, :, 0:2, self.bl_slice]
        self.flags = data['flags'][time_slice, :, 0:2, self.bl_slice]
        self.weights = np.ones_like(data['weights'][time_slice, :, 0:2, self.bl_slice])
        # cross hand polarisations
        self.cross_vis = data['vis'][time_slice, :, 2:, self.bl_slice]
        self.cross_flags = data['flags'][time_slice, :, 2:, self.bl_slice]
        self.cross_weights = np.ones_like(data['weights'][time_slice, :, 2:, self.bl_slice])

        self.timestamps = data['times'][time_slice]
        self.target = katpoint.Target(target)

        # intermediate product visibility - use sparingly!
        self.modvis = None
        # uvw coordinates
        self.uvw = None

        # scan meta-data
        self.dump_period = dump_period
        self.nchan = self.vis.shape[1]
        # note - keep an eye on ordering of frequencies - increasing with index, or decreasing?
        self.channel_freqs = np.arange(self.nchan) if chans is None else np.array(chans)
        self.npol = 4
        self.nant = nant
        self.antenna_descriptions = ants
        self.refant = refant
        self.array_position = array_position

        # initialise models
        self.model_raw_params = None
        self.model = None

        self.logger = logger

    def logsolutiontime(f):
        """
        Decorator to log time duration of solver functions
        """
        def timed(*args, **kw):
            ts = time()
            result = f(*args, **kw)
            te = time()

            scanlogger = args[0].logger
            scanlogger.info('  - Solution time: {0} s'.format(te-ts,))
            return result
        return timed

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

        solint, dumps_per_solint = calprocs.solint_from_nominal(input_solint, self.dump_period, len(self.timestamps))

    # ---------------------------------------------------------------------------------------------
    # Calibration solution functions

    @logsolutiontime
    def g_sol(self, input_solint, g0, bchan=1, echan=0, pre_apply=[], **kwargs):
        """
        Solve for gain

        Parameters
        ----------
        input_solint : nominal solution interval to use for the fit
        g0 : initial estimate of gains for solver, shape (time, pol, nant)
        bchan : start channel for fit, int, optional
        echan : end channel for fit, int, optional
        pre_apply : calibration solutions to apply, list of CalSolutions, optional

        Returns
        -------
        Gain CalSolution with soltype 'G', shape (time, pol, nant)
        """
        if len(pre_apply) > 0:
            self.modvis = copy.deepcopy(self.vis)
        else:
            self.modvis = self.vis

        for soln in pre_apply:
            self.logger.info('  - Pre-apply {0} solution to {1}'.format(soln.soltype,self.target.name))
            self.modvis = self.apply(soln,origvis=False)

        # set up solution interval
        solint, dumps_per_solint = calprocs.solint_from_nominal(input_solint, self.dump_period, len(self.timestamps))
        self.logger.info('  - G solution interval: {0} s'.format(solint,))
        # determine channel range for fit
        if echan == 0: echan = None
        chan_slice = [slice(None), slice(bchan, echan), slice(None), slice(None)]

        # initialise and apply model, for if this scan target has an associated model
        self._init_model()
        fitvis = self._get_solver_model(chan_select=chan_slice)

        # first averge in time over solution interval, for specified channel range (no averaging over channel)
        ave_vis, ave_flags, ave_weights, av_sig, ave_times = calprocs.wavg_full_t(fitvis, self.flags[chan_slice],
                self.weights[chan_slice], dumps_per_solint, axis=0, times=self.timestamps)
        # secondly, average channels
        ave_vis = calprocs.wavg(ave_vis, ave_flags, ave_weights, axis=1)
        # solve for gain
        g_soln = calprocs.g_fit(ave_vis, self.corrprod_lookup, g0, self.refant, **kwargs)

        return CalSolution('G', g_soln, ave_times)

    @logsolutiontime
    def kcross_sol(self,bchan=1,echan=0,chan_ave=1,pre_apply=[]):
        """
        Solve for cross hand delay offset
        *** doesn't currently use models ***

        Parameters
        ----------
        bchan : start channel for fit, int, optional
        echan : end channel for fit, int, optional
        chan_ave : channels to average together prior during fit, int, optional
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
            self.logger.info('  - Pre-apply {0} solution to {1}'.format(soln.soltype, self.target.name))
            self.modvis = self.apply(soln, origvis=False)

        # average over all time, for specified channel range (no averaging over channel)
        if echan == 0: echan = None
        chan_slice = [slice(None), slice(bchan, echan), slice(None), slice(None)]
        ave_vis, av_flags, av_weights, av_sig = calprocs.wavg_full(self.modvis[chan_slice], self.cross_flags[chan_slice], self.cross_weights[chan_slice], axis=0)

        # solve for cross hand delay KCROSS
        # note that the kcross solver needs the flags because it averages the data
        #  (strictly it should need weights too, but deal with that leter when weights are meaningful)
        kcross_soln = calprocs.kcross_fit(ave_vis, av_flags, self.channel_freqs[bchan:echan], chan_ave=chan_ave)
        return CalSolution('KCROSS', kcross_soln, np.average(self.timestamps))

    @logsolutiontime
    def k_sol(self, bchan=1, echan=0, chan_sample=1, pre_apply=[]):
        """
        Solve for delay

        Parameters
        ----------
        bchan : start channel for fit, int, optional
        echan : end channel for fit, int, optional
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
            self.logger.info('  - Pre-apply {0} solution to {1}'.format(soln.soltype,self.target.name))
            self.modvis = self.apply(soln, origvis=False)

        # determine channel range for fit
        if echan == 0: echan = None
        chan_slice = [slice(None), slice(bchan, echan), slice(None), slice(None)]
        # use specified channel range for frequencies
        k_freqs = self.channel_freqs[bchan:echan]

        # initialise model, if this scan target has an associated model
        self._init_model()
        # for delay case, only apply case C full visibility model (other models don't impact delay)
        if self.model is None:
            fitvis = self.modvis[chan_slice]
        elif self.model.shape[-1] == 1:
            fitvis = self.modvis[chan_slice]
        else:
            fitvis = self._get_solver_model(chan_select=chan_slice)

        # average over all time, for specified channel range (no averaging over channel)
        ave_vis, ave_time = calprocs.wavg(fitvis, self.flags[chan_slice], self.weights[chan_slice], times=self.timestamps, axis=0)
        # fit for delay
        k_soln = calprocs.k_fit(ave_vis, self.corrprod_lookup, k_freqs, self.refant, chan_sample=chan_sample)

        return CalSolution('K', k_soln, ave_time)

    @logsolutiontime
    def b_sol(self, bp0, pre_apply=[]):
        """
        Solve for bandpass

        Parameters
        ----------
        bp0 : initial estimate of bandpass for solver, shape (chan, pol, nant)
        pre_apply : calibration solutions to apply, list of CalSolutions, optional

        Returns
        -------
        Bandpass CalSolution with soltype 'B', shape (chan, pol, nant)
        """
        if len(pre_apply) > 0:
            self.modvis = copy.deepcopy(self.vis)
        else:
            self.modvis = self.vis

        for soln in pre_apply:
            self.logger.info('  - Pre-apply {0} solution to {1}'.format(soln.soltype,self.target.name))
            self.modvis = self.apply(soln, origvis=False)

        # initialise and apply model, for if this scan target has an associated model
        self._init_model()
        fitvis = self._get_solver_model()

        # first average in time
        ave_vis, ave_time = calprocs.wavg(fitvis, self.flags, self.weights, times=self.timestamps, axis=0)
        # solve for bandpass
        b_soln = calprocs.bp_fit(ave_vis, self.corrprod_lookup, bp0)

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
        if solval.shape[-2] != outvis.shape[-2]: raise Exception('Polarisation axes do not match!')

        for cp in range(len(self.corrprod_lookup)):
            outvis[..., cp] /= solval[..., self.corrprod_lookup[cp][0]]*(solval[..., self.corrprod_lookup[cp][1]].conj())

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
            self.vis[..., cp] /= solval[..., self.corrprod_lookup[cp][0]]*(solval[..., self.corrprod_lookup[cp][1]].conj())

    def apply(self, soln, origvis=True, inplace=False):
        # set up more complex interpolation methods later
        if soln.soltype is 'G':
            # add empty channel dimension if necessary
            full_sol = np.expand_dims(soln.values, axis=1) if len(soln.values.shape) < 4 else soln.values
            return self._apply(full_sol, origvis=origvis, inplace=inplace)
        elif soln.soltype is 'K':
            # want shape (ntime, nchan, npol, nant)
            gain_shape = tuple(list(self.vis.shape[:-1]) + [self.nant])
            g_from_k = np.zeros(gain_shape, dtype=np.complex)
            for ci, c in enumerate(self.channel_freqs):
                g_from_k[:, ci, :, :] = np.exp(1.0j*2.*np.pi*soln.values*c)
            return self._apply(g_from_k, origvis=origvis, inplace=inplace)
        elif soln.soltype is 'B':
            return self._apply(soln.values, origvis=origvis, inplace=inplace)
        else:
            return ValueError('Solution type is invalid.')

    # ---------------------------------------------------------------------------------------------
    # interpolation

    def interpolate(self, solns):
        # set up more complex interpolation methods later
        soltype = solns.soltype
        if soltype is 'G':
            return self.linear_interpolate(solns)
        if soltype is 'K':
            return self.inf_interpolate(solns)
        if soltype is 'B':
            return self.inf_interpolate(solns)

    def linear_interpolate(self, solns):
        values = solns.values
        timestamps = solns.times

        if len(timestamps) < 2:
            # case of only one solution value being interpolated
            return self.inf_interpolate(solns)
        else:
            real_interp = calprocs.interp_extrap_1d(timestamps, values.real, kind='linear', axis=0)
            imag_interp = calprocs.interp_extrap_1d(timestamps, values.imag, kind='linear', axis=0)

            interp_solns = real_interp(self.timestamps) + 1.0j*imag_interp(self.timestamps)
            return CalSolution(solns.soltype, interp_solns, self.timestamps)

    def inf_interpolate(self, solns):
        values = solns.values
        interp_solns = np.repeat(np.expand_dims(values, axis=0), len(self.timestamps), axis=0)
        return CalSolution(solns.soltype, interp_solns, self.timestamps)

    # ---------------------------------------------------------------------------------------------
    # model related functions

    def _create_model(self, max_offset=8., timestamps=None):
        """
        Creates models from raw model parameters. *** models are currently unpolarised ***

        Models are currently implimented for three cases:
        * A - Point source at the phase centre, no spectral slope -- model is a scalar
        * B - Point source at the phase centre, with spectral slope -- model is an array of fluxes of shape (nchan)
        * C - Complex model requiring calculation via uvw coordinates -- model is an array of the same shape as self.vis

        Inputs
        ======
        max_offset : float
            The difference in positon away from the phase centre for a point to be considered at the phase centre [arcseconds]
            Default: 8 (= meerkat beam)
        timestamps: array of floats
            Timestamps (optional, only necessary for complex models)
        """
        # phase centre position
        ra0, dec0 = self.target.radec()
        # position of first source
        ra = ephem.hours(self.model_raw_params[0]['RA'].item())
        dec = ephem.degrees(self.model_raw_params[0]['DEC'].item())

        # deal with easy case first - single point at the phase centre
        if (self.model_raw_params.size == 1) and (ephem.separation((ra, dec), (ra0, dec0)) < calprocs.arcsec_to_rad(max_offset)):
                if (('a1' not in self.model_raw_params.dtype.names or self.model_raw_params['a1'] == 0) and
                    ('a2' not in self.model_raw_params.dtype.names or self.model_raw_params['a2'] == 0) and
                    ('a3' not in self.model_raw_params.dtype.names or self.model_raw_params['a3'] == 0)):
                    #### CASE A - Point source as the phase centre, no spectral slope ####
                    # spectral index is zero
                    self.model = np.array([10.**self.model_raw_params['a0'].item()])
                    self.logger.info('     Model: single point source, flat spectrum, flux: {0:03.4f} Jy'.format(self.model,))
                else:
                    #### CASE B - Point source at the phase centre, with spectral slope ####
                    source_coefs = [self.model_raw_params[a].item() for a in ['a0', 'a1', 'a2', 'a3']]
                    # full spectral model
                    # katpoint.FluxDensityModel needs upper and lower limits of applicability of the model.
                    #   we make the assumptions that models provided will apply to our data so ignore this functionality by
                    #   using frequency range 0 -- 100 GHz (==100e3 MHz)
                    source_flux = katpoint.FluxDensityModel(0, 100.0e3, coefs=source_coefs)
                    # katsdpcal flux model parameters referenced to GHz, not MHz (so use frequencies in GHz)
                    self.model = source_flux.flux_density(self.channel_freqs/1.0e9)[np.newaxis,:,np.newaxis,np.newaxis]
                    self.logger.info('     Model: single point source, spectral model, average flux over {0:03.3f}-{1:03.3f} GHz: {2:03.4f} Jy'.format(self.channel_freqs[0]/1.e9, self.channel_freqs[-1]/1.e9, np.mean(self.model)))
        #### CASE C - Complex model requiring calculation via uvw coordinates ####
        # If not one of the simple cases above, make a proper full model
        else:
            self.logger.info('     Model: {0} point sources'.format(len(np.atleast_1d(self.model_raw_params)),))

            # calculate uvw, if it hasn't already been calculated
            if self.uvw is None:
                wl = light_speed/self.channel_freqs
                self.uvw = calprocs.calc_uvw_wave(self.target, self.timestamps, self.corrprod_lookup, self.antenna_descriptions, wl, self.array_position)

            # set up model visibility
            complexmodel = np.zeros_like(self.vis)

            # iteratively add sources to the model
            for source in np.atleast_1d(self.model_raw_params):
                # source spectral flux
                source_coefs = [source[a].item() for a in ['a0', 'a1', 'a2', 'a3']]
                # katpoint.FluxDensityModel needs upper and lower limits of applicability of the model.
                #   we make the assumptions that models provided will apply to our data so ignore this functionality by
                #   using frequency range 0 -- 100 GHz (==100e3 MHz)
                source_flux = katpoint.FluxDensityModel(0, 100.0e3, coefs=source_coefs)
                # katsdpcal flux model parameters referenced to GHz, not MHz (so use frequencies in GHz)
                #   currently using the same flux model for both polarisations
                S = source_flux.flux_density(self.channel_freqs/1.0e9)[np.newaxis, :, np.newaxis, np.newaxis]
                # source position
                ra = ephem.hours(source['RA'].item())
                dec = ephem.degrees(source['DEC'].item())
                l = np.cos(dec)*np.sin(ra-ra0)
                m = np.sin(dec)*np.cos(dec0)-np.cos(dec)*np.sin(dec0)*np.cos(ra - ra0)
                # the axis awkwardness is to allow appropriate broadcasting
                #    this may be a ***terrible*** way to do this - will see!
                complexmodel += S*(np.exp(2.*np.pi*1j*(self.uvw[0]*l + self.uvw[1]*m + self.uvw[2]*(np.sqrt(1.0-l**2.0-m**2.0)-1.0)))[:, :, np.newaxis, :])
            self.model = complexmodel
        return

    def _init_model(self, max_offset=8.0):
        """
        Initialises models for use in the solver.
        Checks for existing model and creates models if it is not yet present

        Inputs
        ======
        max_offset : float
            The difference in positon away from the phase centre for a point to be considered at the phase centre [arcseconds]
            Default: 8 (= meerkat beam)
        """
        # if models parameters have not been set for the scan, return unity model
        if self.model_raw_params is None:
            self.model = None
            return

        if self.model is None:
            self._create_model(max_offset)

    def add_model(self, model_raw_params):
        """
        Add raw parameters for model
        """
        self.model_raw_params = np.atleast_1d(model_raw_params)

    def _get_solver_model(self, chan_select=None):
        """
        Get model to supply to solver

        Inputs
        ======
        chan_select : channel selection, slice

        Returns
        =======
        Model for solver. This is either:
           * self.modvis if there is no model; or
           * self.modvis divided by the model, over the selected channel range

        """
        if chan_select is None: chan_select = slice(None)
        if self.model is None:
            return self.modvis[chan_select]
        else:
            # if the channel axis has dimension 1 (== no channel axis), select all data
            if (self.model.shape[1] == 1): chan_select = slice(None)
            return self.modvis[chan_select]/self.model[chan_select]
