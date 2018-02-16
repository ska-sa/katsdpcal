"""Scan class to data and operations on data."""

from time import time
import functools
import logging

import numpy as np
import dask.array as da
from scipy.constants import c as light_speed
import scipy.interpolate

from katdal.h5datav3 import FLAG_NAMES
import katpoint

from . import calprocs, calprocs_dask, inplace
from .solutions import CalSolution, CalSolutions

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# --- CLASS :  Scan
# --------------------------------------------------------------------------------------------------


def _rfi(vis, flags, flagger, out_bit):
    out_value = np.uint8(2**out_bit)
    # flagger doesn't handle a separate pol axis, so the caller must use
    # a chunk size of 1
    assert flags.shape[2] == 1
    # Mask out the output bit from the input. This ensures that the process
    # gives invariant values even if some chunks are updated in-place before
    # they are used to compute other chunks (which shouldn't happen, but the
    # safety check in the inplace module isn't smart enough to detect this).
    flagger_mask = calprocs.asbool(flags[:, :, 0, :] & ~out_value)
    out_flags = flagger.get_flags(vis[:, :, 0, :], flagger_mask)
    return flags | (out_flags[:, :, np.newaxis, :] * out_value)


class ScanData(object):
    """Data in a scan with particular chunking scheme.

    A :class:`Scan` stores several instances of :class:`ScanData`, representing
    the same data but with chunking optimised for different algorithms.

    TODO: document parameters
    """
    def __init__(self, vis, flags, weights, chunks=None, keep_splits=True):
        if chunks is not None:
            vis = self._rechunk(vis, chunks, keep_splits)
            flags = self._rechunk(flags, chunks, keep_splits)
            weights = self._rechunk(weights, chunks, keep_splits)
        self.vis = vis
        self.flags = flags
        self.weights = weights

    def __getitem__(self, idx):
        return ScanData(self.vis[idx], self.flags[idx], self.weights[idx])

    @property
    def shape(self):
        return self.vis.shape

    def rechunk(self, chunks, keep_splits=True):
        """Create a new view of the data with specified chunk sizes.

        Parameters
        ----------
        chunks
            New chunking scheme, in any format accepted by dask
        keep_splits : bool
            If true (default), existing chunk boundaries will remain in the new
            scheme.
        """
        return ScanData(self.vis, self.flags, self.weights, chunks, keep_splits)

    @classmethod
    def _intersect_chunks(cls, chunks1, chunks2):
        splits = set(np.cumsum(chunks1))
        splits.update(np.cumsum(chunks2))
        splits.add(0)
        return tuple(np.diff(sorted(splits)))

    @classmethod
    def _rechunk(cls, array, chunks, keep_splits):
        chunks = da.core.normalize_chunks(chunks, array.shape)
        if keep_splits:
            chunks = tuple(cls._intersect_chunks(c, e) for (c, e) in zip(chunks, array.chunks))
        return da.rechunk(array, chunks)


class ScanDataPair(object):
    """Wraps a pair of :class:`ScanData` objects, one for auto-polarisations,
    the other for cross-hand polarisations.
    """
    def __init__(self, auto, cross):
        self.auto = auto
        self.cross = cross

    def __getitem__(self, idx):
        return ScanDataPair(self.auto[idx], self.cross[idx])

    def rechunk(self, idx):
        return ScanDataPair(self.auto.rechunk(idx), self.cross.rechunk(idx))


class Scan(object):
    """
    Single scan of data with auxillary information.

    Parameters
    ----------
    data : dictionary
        Buffer of correlator data. Contains arrays of visibility, flag, weight and time.
        The `vis`, `flags` and `weights` arrays are :class:`dask.Arrays`, while `times`
        is a numpy array.
    time_slice: slice
        Time slice of the scan in the buffer arrays.
    dump_period : float
        Dump period of correlator data.
    bls_lookup : list of int, shape (2, number of baselines)
        List of antenna pairs for each baseline.
    target : string
        Name of target observed in the scan.
    chans : array of float
        Array of channel frequencies.
    ants : array of :class:`katpoint.Antenna`
        Array of antennas.
    refant : int
        Index of reference antenna in antenna description list.
    array_position : :class:`katpoint.Antenna`
        Array centre position.
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
    timestamps : array of float, shape (ntime,)
        Times.
    target : katpoint Target
        Phase centre of the scan.
    uvw : array of float, shape (3, ntime, nchan, nbl)
        UVW coordinates
    dump_period : float
        Dump period of correlator data.
    nchan : int
        Number of frequency channels in the data.
    channel_freqs : list of float
        List of frequencies corresponding to channels (or channel indices, in
        the absence of frequencies).
    npol : int
        Number of polarisations in the data.
    antennas : list of :class:`katpoint.Antenna`
        The antennas
    refant : int
        Index of reference antenna in antenna description list
    array_position : :class:`katpoint.Antenna`
        Array centre position
    model_raw_params : list
        List of model components
    model : scalar or array
        Model of the visibilities
    logger : logger
        logger
    """

    def __init__(self, data, time_slice, dump_period, bls_lookup, target,
                 chans, ants, refant=0, array_position=None, corr='xc', logger=logger):

        # cross-correlation mask. Must be np array so it can be used for indexing
        # if scan has explicitly been set up as a cross-correlation scan, select XC data only
        # NOTE: This makes the asumption that the XC data are grouped at the
        # beginning of the bl ordering, followed by the AC data.
        # ******* Fancy indexing will not work here, as it returns a copy, not a view *******
        self.xc_mask = np.array([b0 != b1 for b0, b1 in bls_lookup])
        try:
            xc_slice = slice(np.where(self.xc_mask)[0][0], np.where(self.xc_mask)[0][-1]+1)
        except IndexError:
            # not XC data
            xc_slice = slice(None)
        self.bl_slice = xc_slice if corr is 'xc' else slice(None)
        self.corrprod_lookup = bls_lookup[self.bl_slice]

        # get references to this time chunk of data
        # data format is:   (time x channels x pol x bl).
        all_data = ScanData(data['vis'], data['flags'], data['weights'])
        all_data = all_data[time_slice, :, :, self.bl_slice]
        self.orig = ScanDataPair(all_data[:, :, 0:2, :], all_data[:, :, 2:, :])
        self.reset_chunked()

        self.timestamps = data['times'][time_slice]
        self.target = katpoint.Target(target)

        # uvw coordinates
        self.uvw = None

        # scan meta-data
        self.dump_period = dump_period
        self.nchan = self.orig.auto.shape[1]
        # note - keep an eye on ordering of frequencies - increasing with index, or decreasing?
        self.channel_freqs = np.array(chans, dtype=np.float32)
        self.npol = all_data.vis.shape[3]
        self.antennas = ants
        self.refant = refant
        self.array_position = array_position

        # initialise models
        self.model_raw_params = None
        self.model = None

        self.logger = logger

    def reset_chunked(self):
        """Recreate the :attr:`tf` and :attr:`pb` attributes after changing :attr:`orig`."""
        # Arrays chunked up in time and frequency
        self.tf = self.orig.rechunk((4, 4096, None, None))
        # Arrays chunked up in polarization and baseline
        self.pb = self.orig.rechunk((None, None, 1, 16))

    def logsolutiontime(f):
        """
        Decorator to log time duration of solver functions
        """
        @functools.wraps(f)
        def timed(*args, **kw):
            ts = time()
            result = f(*args, **kw)
            te = time()

            scanlogger = args[0].logger
            scanlogger.info('  - Solution time ({0}): {1} s'.format(f.__name__, te-ts,))
            return result
        return timed

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
        pre_apply : calibration solutions to apply, list of :class:`~.CalSolutions`, optional

        Returns
        -------
        :class:`~.CalSolutions`
            Solutions with soltype 'G', shape (time, pol, nant)
        """
        modvis = self.pre_apply(pre_apply)

        # set up solution interval
        solint, dumps_per_solint = calprocs.solint_from_nominal(input_solint, self.dump_period,
                                                                len(self.timestamps))
        self.logger.info(
            '  - G solution interval: {} s ({} dumps)'.format(solint, dumps_per_solint))
        # determine channel range for fit
        if echan == 0:
            echan = None
        chan_slice = np.s_[:, bchan:echan, :, :]

        # initialise and apply model, for if this scan target has an associated model
        self._init_model()
        fitvis = self._get_solver_model(modvis, chan_select=chan_slice)

        # first averge in time over solution interval, for specified channel
        # range (no averaging over channel)
        ave_vis, ave_flags, ave_weights, ave_times = calprocs_dask.wavg_full_t(
            fitvis, self.tf.auto.flags[chan_slice],
            self.tf.auto.weights[chan_slice], dumps_per_solint, times=self.timestamps)
        # secondly, average channels
        ave_vis = calprocs_dask.wavg(ave_vis, ave_flags, ave_weights, axis=1)
        # solve for gain
        g_soln = calprocs.g_fit(ave_vis.compute(), self.corrprod_lookup, g0, self.refant, **kwargs)

        return CalSolutions('G', g_soln, ave_times)

    @logsolutiontime
    def kcross_sol(self, bchan=1, echan=0, chan_ave=1, pre_apply=[]):
        """
        Solve for cross hand delay offset, for full pol data sets (four polarisation products)
        *** doesn't currently use models ***

        Parameters
        ----------
        bchan : start channel for fit, int, optional
        echan : end channel for fit, int, optional
        chan_ave : channels to average together prior during fit, int, optional
        pre_apply : calibration solutions to apply, list of :class:`~.CalSolutions`, optional

        Returns
        -------
        :class:`~.CalSolution`
            Cross hand polarisation delay offset CalSolution with soltype 'KCROSS', shape (nant)
        """
        if self.npol < 4:
            self.logger.info('Cant solve for KCROSS without four polarisation products')
            return
        else:
            # TODO: pre_apply doesn't do the right thing for cross-hand data
            modvis = self.pre_apply(pre_apply, self.tf.cross)

            # average over all time, for specified channel range (no averaging over channel)
            if echan == 0:
                echan = None
            chan_slice = np.s_[:, bchan:echan, :, :]
            av_vis, av_flags, av_weights = calprocs_dask.wavg_full(
                modvis[chan_slice], self.tf.cross.flags[chan_slice],
                self.tf.cross.weights[chan_slice])

            # solve for cross hand delay KCROSS
            # note that the kcross solver needs the flags because it averages the data
            #  (strictly it should need weights too, but deal with that leter
            #  when weights are meaningful)
            av_vis, av_flags = da.compute(av_vis, av_flags)
            kcross_soln = calprocs.kcross_fit(av_vis, av_flags, self.channel_freqs[bchan:echan],
                                              chan_ave=chan_ave)
            return CalSolutions('KCROSS', kcross_soln, np.average(self.timestamps))

    @logsolutiontime
    def k_sol(self, bchan=1, echan=0, chan_sample=1, pre_apply=[]):
        """
        Solve for delay

        Parameters
        ----------
        bchan : start channel for fit, int, optional
        echan : end channel for fit, int, optional
        chan_sample : channel sampling to use in delay fit, optional
        pre_apply : calibration solutions to apply, list of :class:`CalSolutions`, optional

        Returns
        -------
        :class:`~.CalSolution`
            Delay solution with soltype 'K', shape (2, nant)
        """

        modvis = self.pre_apply(pre_apply)

        # determine channel range for fit
        if echan == 0:
            echan = None
        chan_slice = np.s_[:, bchan:echan, :, :]
        # use specified channel range for frequencies
        k_freqs = self.channel_freqs[bchan:echan]

        # initialise model, if this scan target has an associated model
        self._init_model()
        # for delay case, only apply case C full visibility model (other models don't impact delay)
        if self.model is None:
            fitvis = modvis[chan_slice]
        elif self.model.shape[-1] == 1:
            fitvis = modvis[chan_slice]
        else:
            fitvis = self._get_solver_model(modvis, chan_select=chan_slice)

        # average over all time, for specified channel range (no averaging over channel)
        ave_vis, ave_time = calprocs_dask.wavg(fitvis, self.tf.auto.flags[chan_slice],
                                               self.tf.auto.weights[chan_slice],
                                               times=self.timestamps, axis=0)
        # fit for delay
        k_soln = calprocs.k_fit(ave_vis.compute(), self.corrprod_lookup, k_freqs, self.refant,
                                chan_sample=chan_sample)

        return CalSolution('K', k_soln, ave_time)

    @logsolutiontime
    def b_sol(self, bp0, pre_apply=[]):
        """
        Solve for bandpass

        Parameters
        ----------
        bp0 : initial estimate of bandpass for solver, shape (chan, pol, nant)
        pre_apply : calibration solutions to apply, list of :class:`CalSolutions`, optional

        Returns
        -------
        :class:`~.CalSolution `
            Bandpass with soltype 'B', shape (chan, pol, nant)
        """
        modvis = self.pre_apply(pre_apply)

        # initialise and apply model, for if this scan target has an associated model
        self._init_model()
        fitvis = self._get_solver_model(modvis)

        # first average in time
        ave_vis, ave_time = calprocs_dask.wavg(fitvis, self.tf.auto.flags, self.tf.auto.weights,
                                               times=self.timestamps, axis=0)
        # solve for bandpass
        b_soln = calprocs_dask.bp_fit(ave_vis, self.corrprod_lookup, bp0).compute()

        return CalSolution('B', b_soln, ave_time)

    # ---------------------------------------------------------------------------------------------
    # solution application

    def _apply(self, solval, vis):
        """
        Applies calibration solutions.
        Must already be interpolated to either full time or full frequency.

        Parameters
        ----------
        solval : dask.array
            multiplicative solution values to be divided out of visibility data
        vis : dask.array
            input visibilities to be corrected
        """

        # check solution and vis shapes are compatible
        if solval.shape[-2] != vis.shape[-2]:
            raise Exception('Polarisation axes do not match!')

        # If the solution was (accidentally) computed at double precision while
        # the visibilities are single precision, then we force the solution down
        # to single precision, but warn so that the promotion to double can be
        # tracked down.
        if solval.dtype != vis.dtype:
            logger.warn('Applying solution of type %s to visibilities of type %s',
                        solval.dtype, vis.dtype)
        inv_solval = da.reciprocal(solval, dtype=vis.dtype)
        index0 = [cp[0] for cp in self.corrprod_lookup]
        index1 = [cp[1] for cp in self.corrprod_lookup]
        correction = inv_solval[..., index0] * inv_solval[..., index1].conj()
        return vis * correction

    def apply(self, soln, vis):
        # set up more complex interpolation methods later
        soln_values = da.asarray(soln.values)
        if soln.soltype == 'G':
            # add empty channel dimension if necessary
            full_sol = soln_values[:, np.newaxis, :, :] \
                if soln_values.ndim < 4 else soln_values
            return self._apply(full_sol, vis)
        elif soln.soltype == 'K':
            # want shape (ntime, nchan, npol, nant)
            channel_freqs = da.asarray(self.channel_freqs)
            g_from_k = da.exp(2j * np.pi * soln.values[:, np.newaxis, :, :]
                              * channel_freqs[np.newaxis, :, np.newaxis, np.newaxis])
            return self._apply(g_from_k, vis)
        elif soln.soltype == 'B':
            return self._apply(soln_values, vis)
        else:
            raise ValueError('Solution type {} is invalid.'.format(soln.soltype))

    def pre_apply(self, pre_apply_solns, data=None):
        """Apply a set of solutions to the visibilities.

        Parameters
        ----------
        pre_apply_solns : list of :class:`~katsdpcal.calprocs.CalSolutions`
            Solutions to apply
        data : :class:`ScanData`
            Data view to which to apply solutions. Defaults to
            ``self.tf.auto``. Cross-hand visibilities are not currently
            supported.

        Returns
        -------
        modvis : array
            Corrected visibilities. If `pre_apply_solns` is empty this will
            just be the original visibilities (not a copy), otherwise it will
            be a new array.
        """
        if data is None:
            data = self.tf.auto
        modvis = data.vis
        for soln in pre_apply_solns:
            self.logger.info(
                '  - Pre-apply {0} solution to {1}'.format(soln.soltype, self.target.name))
            modvis = self.apply(soln, modvis)
        return modvis

    @logsolutiontime
    def apply_inplace(self, solns_to_apply):
        """Apply a set of solutions to the visibilities, overwriting them.

        Parameters
        ----------
        solns_to_apply : list of :class:`~katsdpcal.calprocs.CalSolution`
            Solutions to apply
        """
        # TODO: also apply to cross-pols once such an apply function exists
        vis = self.tf.auto.vis
        for soln in solns_to_apply:
            self.logger.info(
                '  - Apply {0} solution to {1} (inplace)'.format(soln.soltype, self.target.name))
            vis = self.apply(soln, vis)
        inplace.store_inplace(vis, self.tf.auto.vis)
        # bust any dask caches
        inplace.rename(self.orig.auto.vis)
        self.reset_chunked()

    # ---------------------------------------------------------------------------------------------
    # interpolation

    def interpolate(self, solns):
        """Interpolate a solution to the timestamps of the scan.

        Converts either a :class:`CalSolution` or :class:`CalSolutions` to a
        :class:`CalSolutions`. A :class:`CalSolution` is simply expanded to a
        :class:`CalSolutions` with the new timestamps (relying on
        broadcasting), while a :class:`CalSolutions` undergoes linear
        interpolation.
        """

        # set up more complex interpolation methods later
        if isinstance(solns, CalSolutions):
            return self.linear_interpolate(solns)
        else:
            return self.inf_interpolate(solns)

    def linear_interpolate(self, solns):
        values = solns.values
        timestamps = solns.times

        if len(timestamps) < 2:
            # case of only one solution value being interpolated
            return CalSolutions(solns.soltype, solns.values, self.timestamps)
        else:
            real_interp = scipy.interpolate.interp1d(
                timestamps, values.real, kind='linear', axis=0, fill_value='extrapolate')
            imag_interp = scipy.interpolate.interp1d(
                timestamps, values.imag, kind='linear', axis=0, fill_value='extrapolate')
            # interp1d gives float64 answers even given float32 inputs
            interp_solns = real_interp(self.timestamps).astype(np.float32) \
                + 1.0j * imag_interp(self.timestamps).astype(np.float32)
            return CalSolutions(solns.soltype, interp_solns, self.timestamps)

    def inf_interpolate(self, soln):
        """Expand a single solution to span all timestamps of the scan"""
        values = soln.values
        interp_solns = np.expand_dims(values, axis=0)
        return CalSolutions(soln.soltype, interp_solns, self.timestamps)

    # ---------------------------------------------------------------------------------------------
    # model related functions

    def _create_model(self, max_offset=8., timestamps=None):
        """
        Creates models from raw model parameters. *** models are currently unpolarised ***

        Models are currently implemented for three cases:
        * A - Point source at the phase centre, no spectral slope -- model is a scalar
        * B - Point source at the phase centre, with spectral slope -- model is
              an array of fluxes of shape (nchan)
        * C - Complex model requiring calculation via uvw coordinates -- model
              is an array of the same shape as self.vis

        Inputs
        ======
        max_offset : float
            The difference in positon away from the phase centre for a point to
            be considered at the phase centre [arcseconds]
            Default: 8 (= meerkat beam)
        timestamps: array of floats
            Timestamps (optional, only necessary for complex models)
        """
        # phase centre position
        ra0, dec0 = self.target.radec()
        # position of first source
        first_source = katpoint.construct_radec_target(self.model_raw_params[0]['RA'].item(),
                                                       self.model_raw_params[0]['DEC'].item())
        position_offset = self.target.separation(first_source,
                                                 antenna=self.array_position)

        # deal with easy case first - single point at the phase centre
        if (self.model_raw_params.size == 1) \
                and (position_offset < calprocs.arcsec_to_rad(max_offset)):
            if (('a1' not in self.model_raw_params.dtype.names
                 or self.model_raw_params['a1'] == 0) and
                ('a2' not in self.model_raw_params.dtype.names
                 or self.model_raw_params['a2'] == 0) and
                ('a3' not in self.model_raw_params.dtype.names
                 or self.model_raw_params['a3'] == 0)):
                # CASE A - Point source as the phase centre, no spectral slope
                # spectral index is zero
                self.model = np.array([10.**self.model_raw_params['a0'].item()], np.float32)
                self.logger.info(
                    '     Model: single point source, flat spectrum, flux: {0:03.4f} Jy'.format(
                        self.model[0],))
            else:
                # CASE B - Point source at the phase centre, with spectral slope
                source_coefs = [self.model_raw_params[a].item() for a in ['a0', 'a1', 'a2', 'a3']]
                # full spectral model
                # katpoint.FluxDensityModel needs upper and lower limits of
                # applicability of the model.
                #   we make the assumptions that models provided will apply to
                #   our data so ignore this functionality by using frequency
                #   range 0 -- 100 GHz (==100e3 MHz)
                source_flux = katpoint.FluxDensityModel(0, 100.0e3, coefs=source_coefs)
                # katsdpcal flux model parameters referenced to GHz, not MHz
                # (so use frequencies in GHz)
                self.model = source_flux.flux_density(
                    self.channel_freqs / 1.0e9)[np.newaxis, :, np.newaxis, np.newaxis]
                self.model = np.require(self.model, dtype=np.float32)
                self.logger.info(
                    '     Model: single point source, spectral model, average flux over '
                    '{0:03.3f}-{1:03.3f} GHz: {2:03.4f} Jy'.format(
                        self.channel_freqs[0] / 1.e9, self.channel_freqs[-1] / 1.e9,
                        np.mean(self.model)))
        # CASE C - Complex model requiring calculation via uvw coordinates ####
        # If not one of the simple cases above, make a proper full model
        else:
            self.logger.info(
                '     Model: {0} point sources'.format(len(np.atleast_1d(self.model_raw_params)),))

            # calculate uvw, if it hasn't already been calculated
            if self.uvw is None:
                wl = light_speed / self.channel_freqs
                self.uvw = calprocs.calc_uvw_wave(
                    self.target, self.timestamps, self.corrprod_lookup,
                    self.antennas, wl, self.array_position)

            # set up model visibility
            complexmodel = np.zeros_like(self.vis)

            # iteratively add sources to the model
            for source in np.atleast_1d(self.model_raw_params):
                # source spectral flux
                source_coefs = [source[a].item() for a in ['a0', 'a1', 'a2', 'a3']]
                # katpoint.FluxDensityModel needs upper and lower limits of
                # applicability of the model. we make the assumptions that
                # models provided will apply to our data so ignore this
                # functionality by using frequency range
                # 0 -- 100 GHz (==100e3 MHz)
                source_flux = katpoint.FluxDensityModel(0, 100.0e3, coefs=source_coefs)
                # katsdpcal flux model parameters referenced to GHz, not MHz
                # (so use frequencies in GHz)
                #   currently using the same flux model for both polarisations
                S = source_flux.flux_density(
                    self.channel_freqs / 1.0e9)[np.newaxis, :, np.newaxis, np.newaxis]
                # source position
                source_position = katpoint.construct_radec_target(source['RA'].item(),
                                                                  source['DEC'].item())
                l, m = self.target.sphere_to_plane(
                    *source_position.radec(), projection_type='SIN', coord_system='radec')
                # the axis awkwardness is to allow appropriate broadcasting
                #    this may be a ***terrible*** way to do this - will see!
                complexmodel += \
                    S * (np.exp(2. * np.pi * 1j
                         * (self.uvw[0]*l + self.uvw[1]*m
                            + self.uvw[2]*(np.sqrt(1.0-l**2.0-m**2.0)-1.0)))[:, :, np.newaxis, :])
            self.model = complexmodel
        return

    def _init_model(self, max_offset=8.0):
        """
        Initialises models for use in the solver.
        Checks for existing models and creates them if they are not yet present

        Inputs
        ======
        max_offset : float
            The difference in positon away from the phase centre for a point to
            be considered at the phase centre [arcseconds]
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

    def _get_solver_model(self, modvis, chan_select=None):
        """
        Get model to supply to solver

        Parameters
        ----------
        modvis : dask.array
            Input visibilities with gain corrections pre-applied
        chan_select : slice, optional
            Channel selection

        Returns
        -------
        model
            Model for solver. This is either:
               * `modvis` if there is no model; or
               * `modvis` divided by the model, over the selected channel range

        """
        if chan_select is None:
            chan_select = np.s_[:]
        if self.model is None:
            return modvis[chan_select]
        else:
            # for model without channel axis divide through selected channels by model
            if len(self.model.shape) < 2 or self.model.shape[1] == 1:
                return modvis[chan_select] / self.model
            else:
                # for full model, divide through selected channels by same
                # channel selection in the model
                return modvis[chan_select] / self.model[chan_select]

    # ----------------------------------------------------------------------
    # RFI Functions
    @logsolutiontime
    def rfi(self, flagger, mask=None, cross=False):
        """Detect flags in the visibilities. Detected flags
        are added to the cal_rfi bit of the flag array.
        Optionally provide a channel mask, which is added to
        the static bit of the flag array.

        Parameters
        ----------
        flagger : :class:`SumThresholdFlagger`
            Flagger, with :meth:`get_flags` to detect rfi
        mask : 1d array, boolean, optional
            Channel mask to apply
        cross : boolean, optional
            If True, flag the cross-pol data, otherwise
            flag single-pol data (default).
        """
        if cross:
            self.logger.info('  - Flag cross-pols')
            data = self.pb.cross
            orig = self.orig.cross
        else:
            self.logger.info('  - Flag single-pols')
            data = self.pb.auto
            orig = self.orig.auto
        vis = data.vis
        flags = data.flags

        total_size = np.multiply.reduce(data.flags.shape) / 100.
        self.logger.info('  - Start flags: %.3f%%',
                         (da.sum(calprocs.asbool(data.flags)) / total_size).compute())

        # Get the relevant flag bits from katdal
        static_bit = FLAG_NAMES.index('static')
        cal_rfi_bit = FLAG_NAMES.index('cal_rfi')

        # do we have an rfi mask? In which case, use it
        # Add to 'static' flag bit.
        # TODO: Should mask_flags already be in static bit?
        if mask is not None:
            flags |= (mask[np.newaxis, :, np.newaxis, np.newaxis] * np.uint8(2**static_bit))

        flags = da.atop(_rfi, 'TFpb', vis, 'tfpb', flags, 'tfpb',
                        dtype=np.uint8,
                        new_axes={'T': vis.shape[0], 'F': vis.shape[1]}, concatenate=True,
                        flagger=flagger, out_bit=cal_rfi_bit)
        # _rfi takes care to be idempotent, so we can use safe=False. The
        # safety check doesn't handle the case of some chunks being
        # concatenated, processed, then split back to the original chunks.
        inplace.store_inplace(flags, data.flags, safe=False)
        # Bust any caches of the old values
        inplace.rename(orig.flags)
        self.reset_chunked()
        if cross:
            flags = self.tf.cross.flags
        else:
            flags = self.tf.auto.flags
        # Count the new flags
        self.logger.info('  - New flags: %.3f%%',
                         (da.sum(calprocs.asbool(flags)) / total_size).compute())
