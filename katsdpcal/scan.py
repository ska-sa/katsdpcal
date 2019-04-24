"""Scan class to data and operations on data."""

import time
import functools
import logging
import operator

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
    return out_flags[:, :, np.newaxis, :] * out_value


class ScanData(object):
    """Data in a scan with particular chunking scheme.

    A :class:`Scan` stores several instances of :class:`ScanData`, representing
    the same data but with chunking optimised for different algorithms.

    TODO: document parameters
    """
    def __init__(self, vis, flags, weights, chunks=None):
        if chunks is not None:
            vis = self._rechunk(vis, chunks)
            flags = self._rechunk(flags, chunks)
            weights = self._rechunk(weights, chunks)
        self.vis = vis
        self.flags = flags
        self.weights = weights

    def __getitem__(self, idx):
        return ScanData(self.vis[idx], self.flags[idx], self.weights[idx])

    @property
    def shape(self):
        return self.vis.shape

    def rechunk(self, chunks):
        """Create a new view of the data with specified chunk sizes.

        Parameters
        ----------
        chunks
            New chunking scheme, in any format accepted by dask
        """
        return ScanData(self.vis, self.flags, self.weights, chunks)

    @classmethod
    def _intersect_chunks(cls, chunks1, chunks2):
        splits = set(np.cumsum(chunks1))
        splits.update(np.cumsum(chunks2))
        splits.add(0)
        return tuple(np.diff(sorted(splits)))

    @classmethod
    def _rechunk(cls, array, chunks):
        chunks = da.core.normalize_chunks(chunks, array.shape)
        chunks = tuple(cls._intersect_chunks(c, e) for (c, e) in zip(chunks, array.chunks))
        if chunks != array.chunks:
            array = da.rechunk(array, chunks)
            # Workaround for https://github.com/dask/dask/issues/3139
            # This flattens the graph to be simply a bunch of getitem calls on
            # the original arrays, which for unknown reasons works around
            # the performance issue.
            array.dask = array.__dask_optimize__(array.dask, array.__dask_keys__(),
                                                 fast_functions=(da.core.getter, operator.getitem),
                                                 rename_fused_keys=False)
        return array


class ScanDataPair(object):
    """Wraps a pair of :class:`ScanData` objects, one for auto-polarisations,
    the other for cross-hand polarisations.
    """
    def __init__(self, auto, cross):
        self.auto_pol = auto
        self.cross_pol = cross

    def __getitem__(self, idx):
        return ScanDataPair(self.auto_pol[idx], self.cross_pol[idx])

    def rechunk(self, idx):
        return ScanDataPair(self.auto_pol.rechunk(idx), self.cross_pol.rechunk(idx))


class ScanDataGroupBl(object):
    """
    Selects a subset of baselines from a :class: `ScanData` object

    Parameters
    ----------
    scan_data : :class: `ScanData`
        data in the scan
    bls_lookup : list of int, shape (2, number of baselines)
        List of antenna pairs for each baseline.
    corr_mask : :class: ndarray of bool
        True for baselines to be selected

    Attributes
    ----------
    orig : :class:`ScanDataPair` of :class:`ScanData`
        data for baselines given by corr_mask
    bls_lookup : list of int, shape (2, number of selected baselines)
        List of antenna pairs for selected baselines
    """
    def __init__(self, scan_data, bls_lookup, corr_mask):
        # NOTE: This makes the asumption that the XC data are grouped at the
        # beginning of the bl ordering, followed by the AC data.
        # ******* Fancy indexing will not work here, as it returns a copy, not a view *******
        # check if data in mask is contiguous
        transitions = (corr_mask.astype(int)[1:] != corr_mask.astype(int)[:-1]).sum()
        if transitions > 1:
            raise ValueError('Cross-correlation data is not contiguous')

        mask_slice = slice(np.where(corr_mask)[0][0], np.where(corr_mask)[0][-1]+1)
        mask_data = scan_data[..., mask_slice]
        self.bls_lookup = bls_lookup[mask_slice]
        self.orig = ScanDataPair(mask_data[:, :, 0:2, :], mask_data[:, :, 2:, :])
        self.reset_chunked()

    def reset_chunked(self):
        """Recreate the :attr:`tf` and :attr:`pb` attributes after changing :attr:`orig`."""
        # Arrays chunked up in time and frequency
        self.tf = self.orig.rechunk((4, 4096, None, None))
        # Arrays chunked up in polarization and baseline
        self.pb = self.orig.rechunk((None, None, 1, 16))


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
    logger : logger
        Logger

    Attributes
    ----------
    xc_mask : numpy array of bool
    ac_mask : numpy array of bool
        Mask for selecting auto-correlation data
    cross_ant : :class:`ScanDataGroupBl` of :class:`ScanData`
        Cross-correlation data for time_slice
    auto_ant : :class:`ScanDataGroupBl` of :class:`ScanData`
        Auto-correlation data for time_slice
    timestamps : array of float, shape (ntime)
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
                 chans, ants, refant=0, array_position=None, logger=logger):
        # cross-correlation and auto-correlation masks.
        # Must be np arrays so they can be used for indexing
        self.xc_mask = np.array([b0 != b1 for b0, b1 in bls_lookup])
        self.ac_mask = np.array([b0 == b1 for b0, b1 in bls_lookup])
        all_data = ScanData(data['vis'], data['flags'], data['weights'])
        all_data = all_data[time_slice]
        self.cross_ant = ScanDataGroupBl(all_data, bls_lookup, self.xc_mask)
        self.auto_ant = ScanDataGroupBl(all_data, bls_lookup, self.ac_mask)
        self.timestamps = data['times'][time_slice]
        self.target = katpoint.Target(target)

        # uvw coordinates
        self.uvw = None

        # scan meta-data
        self.dump_period = dump_period
        self.nchan = self.cross_ant.orig.auto_pol.shape[1]
        # note - keep an eye on ordering of frequencies - increasing with index, or decreasing?
        self.channel_freqs = np.array(chans, dtype=np.float32)
        self.npol = all_data.vis.shape[3]
        self.antennas = ants
        self.refant = refant
        self.array_position = array_position
        self.nant = len(ants)
        # initialise models
        self.model_raw_params = None
        self.model = None

        self.logger = logger

    def logsolutiontime(f):
        """
        Decorator to log time duration of solver functions
        """
        @functools.wraps(f)
        def timed(*args, **kw):
            ts = time.time()
            result = f(*args, **kw)
            te = time.time()

            scanlogger = args[0].logger
            scanlogger.info('  - Solution time ({0}): {1} s'.format(f.__name__, te-ts,))
            return result
        return timed

    # ---------------------------------------------------------------------------------------------
    @logsolutiontime
    def refant_find(self, bchan=1, echan=None, chan_sample=1):
        """
        Find a good reference antenna candidate

        Parameters
        ----------
        bchan : start channel for fit, int, optional
        echan : end channel for fit, int, optional
        chan_sample : channel sampling to use in delay fit, optional

        Returns
        -------
        refant_index : int
            index of preferred refant
        """

        modvis = self.cross_ant.tf.auto_pol.vis

        # determine channel range for fit
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
        ave_vis = calprocs_dask.wavg(
            fitvis,
            self.cross_ant.tf.auto_pol.flags[chan_slice],
            self.cross_ant.tf.auto_pol.weights[chan_slice])

        # fit for delay
        ave_vis = ave_vis.compute()
        refant_index = calprocs.best_refant(ave_vis, self.cross_ant.bls_lookup, k_freqs)
        return refant_index

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
            fitvis, self.cross_ant.tf.auto_pol.flags[chan_slice],
            self.cross_ant.tf.auto_pol.weights[chan_slice], dumps_per_solint, times=self.timestamps)
        # secondly, average channels
        ave_vis, ave_flags, ave_weights = calprocs_dask.wavg_full(ave_vis, ave_flags, ave_weights,
                                                                  axis=1)
        # solve for gain
        ave_vis, ave_weights = da.compute(ave_vis, ave_weights)
        g_soln = calprocs.g_fit(ave_vis, ave_weights,
                                self.cross_ant.bls_lookup, g0,
                                self.refant, **kwargs)

        return CalSolutions('G', g_soln, ave_times)

    @logsolutiontime
    def kcross_sol(self, bchan=1, echan=None, chan_ave=1, pre_apply=[], nd=None, auto_ant=False):
        """
        Solve for cross hand delay offset, for full pol data sets (four polarisation products)
        *** doesn't currently use models ***

        Parameters
        ----------
        bchan : int, optional
            start channel for fit
        echan : int, optional
            end channel for fit
        chan_ave : int, optional
            channels to average together prior during fit
        pre_apply :  list of :class:`~.CalSolutions` CalSolutions, optional
            calibration solutions to apply
        nd : :class:`np.ndarray` of bool, optional
            True for antennas with noise diode on, otherwise False.
            If auto_ant is True and nd is False for an antenna, set the solution
            for that antenna to NaN
        auto_ant : bool, optional
            if True solve for KCROSS_DIODE using auto-correlations, else solve for KCROSS
            using cross-correlations

        Returns
        -------
        :class:`~.CalSolution`
            Cross hand polarisation delay offset CalSolution with soltype 'KCROSS', shape(pol, 1)
            or 'KCROSS_DIODE', shape (pol, nant). The second polarisation has delays set to zero.
        """
        if auto_ant:
            corr = self.auto_ant
            soln_type = 'KCROSS_DIODE'
        else:
            corr = self.cross_ant
            soln_type = 'KCROSS'

        # pre_apply solutions
        modvis = self.pre_apply(pre_apply, corr, cross_pol=True)

        # average over all time, for specified channel range (no averaging over channel)
        chan_slice = np.s_[:, bchan:echan, :, :]
        av_vis, av_flags, av_weights = calprocs_dask.wavg_full(
            modvis[chan_slice], corr.tf.cross_pol.flags[chan_slice],
            corr.tf.cross_pol.weights[chan_slice])
        # average over channel if specified
        if chan_ave > 1:
            av_vis, av_flags, av_weights = calprocs_dask.wavg_full_f(av_vis,
                                                                     av_flags,
                                                                     av_weights,
                                                                     chan_ave)

        # solve for cross hand delay KCROSS_DIODE per antenna if auto_ant is True, else
        # average across all baselines and solve for a single KCROSS
        weighted_data, flagged_weights = calprocs_dask.weight_data(av_vis, av_flags, av_weights)
        av_weights = da.sum(flagged_weights, axis=-2)
        av_vis = (weighted_data[:, 0, :] +
                  np.conjugate(weighted_data[:, 1, :]) /
                  av_weights)
        # replace NaN'ed data caused by dividing by zero weights with a zero.
        isnan = da.isnan(av_vis)
        av_vis = calprocs_dask.where(isnan, av_vis.dtype.type(0), av_vis)

        if not auto_ant:
            av_flags = da.any(av_flags, axis=-2)
            av_vis, av_flags, av_weights = calprocs_dask.wavg_full(av_vis, av_flags, av_weights,
                                                                   axis=-1)
            # add antenna axis
            av_vis = av_vis[..., np.newaxis]
            av_weights = av_weights[..., np.newaxis]

        chans = self.channel_freqs[bchan:echan]
        ave_chans = np.add.reduceat(
                chans, list(range(0, len(chans), chan_ave))) / chan_ave

        av_vis, av_weights = da.compute(av_vis, av_weights)
        kcross_soln = calprocs.k_fit(av_vis, av_weights,
                                     corr.bls_lookup, ave_chans, self.refant, False,
                                     chan_sample=1)
        # set delay in the second polarisation axis to zero
        kcross_soln = np.vstack([kcross_soln, np.zeros_like(kcross_soln)])
        # if auto_ant is True set soln to NaN for antennas where the noise diode didn't fire
        if nd is not None and auto_ant:
            kcross_soln = np.where(~nd, np.nan, kcross_soln)
        return CalSolution(soln_type, kcross_soln, np.average(self.timestamps))

    @logsolutiontime
    def k_sol(self, bchan=1, echan=None, chan_sample=1, pre_apply=[]):
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
        ave_vis, ave_flags, ave_weights = calprocs_dask.wavg_full(
            fitvis,
            self.cross_ant.tf.auto_pol.flags[chan_slice],
            self.cross_ant.tf.auto_pol.weights[chan_slice])

        ave_time = np.average(self.timestamps, axis=0)
        # fit for delay
        ave_vis, ave_weights = da.compute(ave_vis, ave_weights)
        k_soln = calprocs.k_fit(ave_vis, ave_weights,
                                self.cross_ant.bls_lookup,
                                k_freqs, self.refant, True, chan_sample)

        return CalSolution('K', k_soln, ave_time)

    @logsolutiontime
    def b_sol(self, bp0, pre_apply=[], bp_flagger=None):
        """
        Solve for bandpass

        Parameters
        ----------
        bp0 : :class: `np.ndarray`, complex, shape (chan, pol, nant) or None
            initial estimate of bandpass for solver
        pre_apply : list of :class:`CalSolutions`, optional
            calibration solutions to apply
        bp_flagger : :class:`SumThresholdFlagger`, optional
            Flagger, with :meth:`get_flags` to detect rfi in bandpass

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
        ave_vis, ave_flags, ave_weights = calprocs_dask.wavg_full(
            fitvis,
            self.cross_ant.tf.auto_pol.flags,
            self.cross_ant.tf.auto_pol.weights)

        ave_time = np.average(self.timestamps, axis=0)
        # solve for bandpass
        b_soln = calprocs_dask.bp_fit(ave_vis, ave_weights,
                                      self.cross_ant.bls_lookup, bp0, self.refant)

        # flag bandpass
        if bp_flagger is not None:
            # add time axis to be compatible with shape expected by flagger
            b_soln = b_soln[np.newaxis].rechunk((None, None, 1, None))
            # turn bandpass NaN's into input flags for the flagger,
            # the input flags are stored in bit 0
            flags = da.isnan(b_soln).astype(np.uint8)
            rfi_flags = da.atop(_rfi, 'TFpa', b_soln, 'tfpa', flags, 'tfpa',
                                dtype=np.uint8,
                                new_axes={'T': b_soln.shape[0], 'F': b_soln.shape[1]},
                                concatenate=True,
                                flagger=bp_flagger, out_bit=1)

            # OR flags across polarisation
            rfi_flags |= da.flip(rfi_flags, axis=2)
            b_soln = da.where(rfi_flags, np.nan, b_soln)
            b_soln = b_soln[0]

        # normalise after flagging solution
        b_soln = b_soln.compute()

        return CalSolution('B', b_soln, ave_time)

    @logsolutiontime
    def b_norm(self, b_soln, bchan=1, echan=None):
	"""
        Calculates an array of complex numbers which normalises
        b_soln in the specified channel range

        Parameters:
        -----------
        bchan : int, optional
            start channel to use in normalisation
        echan : int, optional
            end channel to use in normalisation
        b_soln : :class: `~.CalSolution`
            Solution with soltype 'B' to be normalised, shape(nchans, npols, nants)

        Returns:
        -------
        :class:`np.ndarray`
            normalisation factor, complex, shape(npols, nants)
        """
        if b_soln.soltype == 'B':
            b_soln = b_soln.values[bchan:echan]
            norm_fact = calprocs.normalise_complex(b_soln)
        else:
            raise ValueError('b_soln has soltype {}, expected soltype B'.format(b_soln.soltype))
        return norm_fact

    # ---------------------------------------------------------------------------------------------
    # solution application

    def _apply(self, solval, vis, cross_pol=False):
        """
        Applies calibration solutions.
        Must already be interpolated to either full time or full frequency.

        Parameters
        ----------
        solval : dask.array
            multiplicative solution values to be divided out of visibility data
        vis : dask.array
            input visibilities to be corrected
        cross_pol : bool, optional
            Apply corrections appropriate for cross hand data if True, else apply
            parallel hand corrections.
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
        # check if data is auto-corr or cross-corr
        if vis.shape[-1] == self.nant:
            index0 = [cp[0] for cp in self.auto_ant.bls_lookup]
            index1 = [cp[1] for cp in self.auto_ant.bls_lookup]
        else:
            index0 = [cp[0] for cp in self.cross_ant.bls_lookup]
            index1 = [cp[1] for cp in self.cross_ant.bls_lookup]
        if cross_pol:
            correction = inv_solval[..., index0] * da.flip(inv_solval, axis=-2).conj()[..., index1]
        else:
            correction = inv_solval[..., index0] * inv_solval[..., index1].conj()
        return vis * correction

    def apply(self, soln, vis, cross_pol=False):
        # set up more complex interpolation methods later
        soln_values = da.asarray(soln.values)
        if soln.soltype == 'G':
            # add empty channel dimension if necessary
            full_sol = soln_values[:, np.newaxis, :, :] \
                if soln_values.ndim < 4 else soln_values
            return self._apply(full_sol, vis, cross_pol)

        elif soln.soltype == 'K':
            # want shape (ntime, nchan, npol, nant)
            channel_freqs = da.asarray(self.channel_freqs)
            g_from_k = da.exp(2j * np.pi * soln.values[:, np.newaxis, :, :]
                              * channel_freqs[np.newaxis, :, np.newaxis, np.newaxis])
            return self._apply(g_from_k, vis, cross_pol)
        elif soln.soltype in ['KCROSS_DIODE', 'KCROSS']:
            # average HV delay over all antennas
            channel_freqs = da.asarray(self.channel_freqs)
            soln = np.nanmedian(soln.values, axis=-1, keepdims=True)
            g_from_k = da.exp(2j * np.pi * soln[:, np.newaxis, :, :]
                              * channel_freqs[np.newaxis, :, np.newaxis, np.newaxis])
            g_from_k = da.tile(g_from_k, self.nant)
            return self._apply(g_from_k, vis, cross_pol)

        elif soln.soltype is 'B':
            return self._apply(soln_values, vis, cross_pol)
        else:
            raise ValueError('Solution type {} is invalid.'.format(soln.soltype))

    def pre_apply(self, pre_apply_solns, data=None, cross_pol=False):
        """Apply a set of solutions to the visibilities. It always uses
        tf chunking of the data.

        Parameters
        ----------
        pre_apply_solns : list of :class:`~katsdpcal.calprocs.CalSolutions`
            Solutions to apply
        data : :class:`ScanDataGroupBl`
            Data group to which to apply solutions. Defaults to
            ``self.cross_ant.``.
        cross_pol : bool, optional
            Apply corrections to cross hand data if True, else apply to
            parallel hand.

        Returns
        -------
        modvis : array
            Corrected visibilities. If `pre_apply_solns` is empty this will
            just be the original visibilities (not a copy), otherwise it will
            be a new array.
        """
        if data is None:
            data = self.cross_ant
        if cross_pol:
            modvis = data.tf.cross_pol.vis
        else:
            modvis = data.tf.auto_pol.vis
        for soln in pre_apply_solns:
            self.logger.info(
                '  - Pre-apply {0} solution to {1}'.format(soln.soltype, self.target.name))
            modvis = self.apply(soln, modvis, cross_pol)
        return modvis

    @logsolutiontime
    def apply_inplace(self, solns_to_apply):
        """Apply a set of solutions to the visibilities, overwriting them.

        Parameters
        ----------
        solns_to_apply : list of :class:`~katsdpcal.calprocs.CalSolution`
            Solutions to apply
        """
        vis_xc_auto = self.cross_ant.tf.auto_pol.vis
        vis_xc_cross = self.cross_ant.tf.cross_pol.vis
        vis_ac_auto = self.auto_ant.tf.auto_pol.vis
        vis_ac_cross = self.auto_ant.tf.cross_pol.vis
        for soln in solns_to_apply:
            self.logger.info(
                '  - Apply {0} solution to {1} (inplace)'.format(soln.soltype, self.target.name))
            vis_xc_auto = self.apply(soln, vis_xc_auto)
            vis_xc_cross = self.apply(soln, vis_xc_cross, cross_pol=True)
            vis_ac_auto = self.apply(soln, vis_ac_auto)
            vis_ac_cross = self.apply(soln, vis_ac_cross, cross_pol=True)
        inplace.store_inplace(vis_xc_auto, self.cross_ant.tf.auto_pol.vis)
        inplace.store_inplace(vis_xc_cross, self.cross_ant.tf.cross_pol.vis)
        inplace.store_inplace(vis_ac_auto, self.auto_ant.tf.auto_pol.vis)
        inplace.store_inplace(vis_ac_cross, self.auto_ant.tf.cross_pol.vis)
        # bust any dask caches
        inplace.rename(self.cross_ant.orig.auto_pol.vis)
        inplace.rename(self.cross_ant.orig.cross_pol.vis)
        inplace.rename(self.auto_ant.orig.auto_pol.vis)
        inplace.rename(self.auto_ant.orig.cross_pol.vis)
        self.cross_ant.reset_chunked()
        self.auto_ant.reset_chunked()

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
                    self.target, self.timestamps, self.cross_ant.bls_lookup,
                    self.antennas, wl, self.array_position)

            # set up model visibility
            complexmodel = np.zeros_like(self.cross_ant.orig.auto_pol.vis)

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
    # Summarize Functions
    def summarize_flags(self, av_corr, nchans=1024):
        """
        Average flags to nchans channels. Sum these averaged flags across time
        and average them across baselines. Add the summed and averaged quantities
        to a dictionary

        Parameters
        ----------
        av_corr : collections.defaultdict
            dict to add summarized flags to
        nchans : int, optional
            number of channels in the summarized flags
        """
        # sum flags over time and average in blocks of frequency
        t_sum_flags = da.sum(calprocs.asbool(self.cross_ant.tf.auto_pol.flags),
                             axis=0, dtype=np.float32)

        orig_chans = self.cross_ant.tf.auto_pol.flags.shape[1]
        chanav = max(1, orig_chans // nchans)
        t_sum_flags = calprocs_dask.av_blocks(t_sum_flags, chanav)

        # average flags over baseline and time
        n_times = np.float32(len(self.timestamps))
        bl_sum_flags = da.mean(t_sum_flags, axis=-1) / n_times
        t_sum_flags, bl_sum_flags = da.compute(t_sum_flags, bl_sum_flags)

        # add summed and average flags to av_corr
        av_corr['t_flags'].insert(0, t_sum_flags)
        av_corr['bl_flags'].insert(0, (bl_sum_flags, np.average(self.timestamps)))

    def summarize_full(self, av_corr, key, data=None, nchans=8):
        """
        Average visibilities per scan, per antenna and into nchans frequency blocks. Add these
        averaged visibilities, flags and weights to a dictionary with given key.
        Prefix the target_name to the dictionary key.

        Parameters
        ----------
        av_corr : collections.defaultdict
            dict to add averaged visibilities to
        key : str
            dictionary key
        data : tuple of :class: `da.Array`, optional
            vis, flags, weights of data to average. Defaults to
            vis, flags and weights of self.cross_ant.tf.auto_pol
        nchans : int, optional
            number of channels in the averaged visibilities
        """
        if not data:
            vis = self.cross_ant.tf.auto_pol.vis
            flags = self.cross_ant.tf.auto_pol.flags
            weights = self.cross_ant.tf.auto_pol.weights
        else:
            vis, flags, weights = data

        # average in frequency blocks, per scan and per antenna
        av_vis, av_flags, av_weights = calprocs_dask.wavg_t_f(vis, flags, weights, nchans)
        av_vis, av_flags, av_weights = calprocs_dask.wavg_ant(av_vis, av_flags, av_weights,
                                                              self.antennas,
                                                              self.cross_ant.bls_lookup)
        # add avg spectrum per antenna to corrected data
        av_vis, av_flags, av_weights = da.compute(av_vis, av_flags, av_weights)
        av_corr[key].insert(0, (av_vis, av_flags, av_weights))

    def summarize(self, av_corr, key, data=None, nchans=8, avg_ant=False,
                  refant_only=False):
        """
        Average visibilites per scan and into nchans frequency blocks.
        Optionally average per antenna. Optionally select only baselines to the reference antenna.
        Add these averaged visibilities to a defaultdict with the given key.

        Parameters:
        -----------
        av_corr : collections.defaultdict
            dict to add averaged visibilities to
        key : str
            dictionary key
        data : tuple of :class: `da.Array`, optional
            vis, flags, weights of data to average. Defaults to
            vis, flags and weights of self.cross_ant.tf.auto_pol
        nchans : int, optional
            number of channels in the averaged visibilities
        avg_ant : bool, optional
            average per antenna
        refant_only : bool, optional
            select baselines to the reference antenna
        """
        if not data:
            vis = self.cross_ant.tf.auto_pol.vis
            flags = self.cross_ant.tf.auto_pol.flags
            weights = self.cross_ant.tf.auto_pol.weights
        else:
            vis, flags, weights = data

        # average per scan, in frequency blocks
        av_vis, av_flags, av_weights = calprocs_dask.wavg_t_f(vis, flags, weights, nchans)

        # select only baselines to refant
        if refant_only:
            ant_idx = np.where(((self.cross_ant.bls_lookup[:, 0] == self.refant)
                               ^ (self.cross_ant.bls_lookup[:, 1] == self.refant)))[0]
            av_vis = av_vis[..., ant_idx]
            av_flags = av_flags[..., ant_idx]
        # Average per antenna
        if avg_ant:
            av_vis, av_flags, av_weights = calprocs_dask.wavg_ant(av_vis, av_flags, av_weights,
                                                                  self.antennas,
                                                                  self.cross_ant.bls_lookup)

        av_vis[av_flags] = np.nan
        val = (av_vis.compute(), np.average(self.timestamps))
        av_corr[key].insert(0, val)

    # ----------------------------------------------------------------------
    # RFI Functions
    @logsolutiontime
    def rfi(self, flagger, mask=None, auto_ant=False, sensors=None):
        """Detect flags in the visibilities. Detected flags
        are added to the cal_rfi bit of the flag array. Flags
        are detected using cross-pol data but added to both
        the cross_pol and auto_pol flags.
        Optionally provide a channel mask, which is added to
        the static bit of the flag array.

        Parameters
        ----------
        flagger : :class:`SumThresholdFlagger`
            Flagger, with :meth:`get_flags` to detect rfi
        mask : 1d array, boolean, optional
            Channel mask to apply
        auto_ant : boolean, optional
            If True, flag the auto_ant data, otherwise
            flag cross_ant data (default).
        sensors : dict, optional
            Sensors available in the calling parent. If set, it is expected
            that sensors named pipeline-[start|final]-flag-fraction-[single|cross]
            will exist in the dict.
        """
        if auto_ant:
            scandata = self.auto_ant
            label = ', auto-corrs'
            if mask is not None:
                mask = mask[..., self.ac_mask]
        else:
            scandata = self.cross_ant
            label = ''
            if mask is not None:
                mask = mask[..., self.xc_mask]

        # Get the relevant flag bits from katdal
        static_bit = FLAG_NAMES.index('static')
        cal_rfi_bit = FLAG_NAMES.index('cal_rfi')

        data = {}
        total_size = np.multiply.reduce(scandata.pb.cross_pol.shape) / 100.
        for key in ['auto_pol', 'cross_pol']:
            data[key] = getattr(scandata.pb, key)
            data[key + '_orig'] = getattr(scandata.orig, key)
            data[key + '_flags'] = data[key].flags
            data[key + '_start_flag_fraction'] = (
                da.sum(calprocs.asbool(data[key+'_flags'])) / total_size).compute()
            # do we have an rfi mask? In which case, use it
            # Add to 'static' flag bit.
            # TODO: Should mask_flags already be in static bit?
            if mask is not None:
                data[key + '_flags'] |= (mask * np.uint8(2**static_bit))

        rfi_flags = da.atop(
            _rfi, 'TFpb', data['cross_pol'].vis, 'tfpb', data['cross_pol_flags'], 'tfpb',
            dtype=np.uint8,
            new_axes={'T': data['cross_pol'].vis.shape[0], 'F': data['cross_pol'].vis.shape[1]},
            concatenate=True, flagger=flagger, out_bit=cal_rfi_bit
            )

        # OR the rfi flags across polarisations
        rfi_flags |= da.flip(rfi_flags, axis=2)
        # OR the rfi flags with the original flags
        data['cross_pol_flags'] |= rfi_flags
        data['auto_pol_flags'] |= rfi_flags

        # _rfi takes care to be idempotent, so we can use safe=False. The
        # safety check doesn't handle the case of some chunks being
        # concatenated, processed, then split back to the original chunks.
        inplace.store_inplace([data['cross_pol_flags'], data['auto_pol_flags']],
                              [data['cross_pol'].flags, data['auto_pol'].flags],
                              safe=False)
        # Bust any caches of the old values
        inplace.rename(data['cross_pol_orig'].flags)
        inplace.rename(data['auto_pol_orig'].flags)
        self.cross_ant.reset_chunked()

        for key in ['auto_pol', 'cross_pol']:
            tf_flags = getattr(scandata.tf, key).flags
            data[key + '_final_flag_fraction'] = (
                da.sum(calprocs.asbool(tf_flags)) / total_size).compute()

            flag_type = key.replace('_', '-')
            self.logger.info('  - Flag %ss%s', flag_type, label)
            self.logger.info('  - Start flags : %.3f%%', data[key+'_start_flag_fraction'])
            self.logger.info('  - New flags: %.3f%%', data[key+'_final_flag_fraction'])
            if sensors:
                now = time.time()
                sensors['pipeline-start-flag-fraction-{}'.format(flag_type)].set_value(
                    data[key + '_start_flag_fraction'], timestamp=now)
                sensors['pipeline-final-flag-fraction-{}'.format(flag_type)].set_value(
                    data[key + '_final_flag_fraction'], timestamp=now)
