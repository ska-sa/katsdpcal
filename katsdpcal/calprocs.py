"""
Calibration procedures for MeerKAT calibration pipeline
=======================================================

Solvers and averagers for use in the MeerKAT calibration pipeline.
"""

from __future__ import print_function
import time
import logging

import numpy as np
import scipy.fftpack
import numba

import katpoint


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------------
# --- Modelling procedures
# --------------------------------------------------------------------------------------------------


def radec_to_lm(ra, dec, ra0, dec0):
    """
    Parameteters
    ------------
    ra : float
        Right Ascension, radians
    dec : float
        Declination, radians
    ra0 : float
        Right Ascention of centre, radians
    dec0 : float
        Declination of centre, radians

    Returns
    -------
    l, m : float
        direction cosines
    """
    l = np.cos(dec)*np.sin(ra - ra0)
    m = np.sin(dec)*np.cos(dec0) - np.cos(dec)*np.sin(dec0)*np.cos(ra-ra0)
    return l, m


def list_to_model(model_list, centre_position):
    lmI_list = []

    ra0, dec0 = centre_position

    for ra, dec, I in model_list:
        l, m = radec_to_lm(ra, dec, ra0, dec0)
        lmI_list.append([l, m, I])

    return lmI_list


def to_ut(t):
    """
    Converts MJD seconds into Unix time in seconds

    Parameters
    ----------
    t : float
        time in MJD seconds

    Returns
    -------
    float
        Unix time in seconds
    """
    return (t/86400. - 2440587.5 + 2400000.5)*86400.


def calc_uvw_wave(phase_centre, timestamps, corrprod_lookup, ant_descriptions,
                  wavelengths=None, array_centre=None):
    """
    Calculate uvw coordinates

    Parameters
    ----------
    phase_centre : :class:`katpoint.Target`
        phase centre position
    timestamps : array of float
        times, shape(nrows)
    corrprod_lookup : array
        lookup table of antenna indices for each baseline, shape(nant,2)
    antenna_descriptions : list of str
        description strings for the antennas, same order as antlist
    wavelengths : array or scalar
        wavelengths, single value or array shape(nchans)
    array_centre : str
        description string for array centre position

    Returns
    -------
    uvw_wave
        uvw coordinates, normalised by wavelength
    """
    uvw = calc_uvw(phase_centre, timestamps, corrprod_lookup, ant_descriptions, array_centre)
    if wavelengths is None:
        return uvw
    elif np.isscalar(wavelengths):
        return uvw/wavelengths
    else:
        return uvw[:, :, np.newaxis, :]/wavelengths[:, np.newaxis]


def calc_uvw(phase_centre, timestamps, corrprod_lookup, ant_descriptions, array_centre=None):
    """
    Calculate uvw coordinates

    Parameters
    ----------
    phase_centre : :class:`katpoint.Target`
        phase centre position
    timestamps : array of float
        times, shape(nrows)
    corrprod_lookup : array
        lookup table of antenna indices for each baseline, shape(nant,2)
    antenna_descriptions : list of str
        description strings for the antennas, same order as antlist
    array_centre : str
        description string for array centre position

    Returns
    -------
    uvw_wave
        uvw coordinates
    """
    if array_centre is not None:
        array_reference_position = katpoint.Antenna(array_centre)
    else:
        # if no array centre position is given, use lat-long-alt of first
        # antenna in the antenna list
        refant = katpoint.Antenna(ant_descriptions[0])
        array_reference_position = katpoint.Antenna('array_position', *refant.ref_position_wgs84)

    # use the array reference position for the basis
    basis = phase_centre.uvw_basis(timestamp=timestamps, antenna=array_reference_position)
    antenna_uvw = np.empty([len(ant_descriptions), 3, len(timestamps)])

    for i, antenna in enumerate(ant_descriptions):
        ant = katpoint.Antenna(antenna)
        enu = np.array(ant.baseline_toward(array_reference_position))
        antenna_uvw[i, ...] = np.tensordot(basis, enu, ([1], [0]))

    baseline_uvw = np.empty([3, len(timestamps), len(corrprod_lookup)])
    for i, [a1, a2] in enumerate(corrprod_lookup):
        baseline_uvw[..., i] = antenna_uvw[a2] - antenna_uvw[a1]

    return baseline_uvw

# --------------------------------------------------------------------------------------------------
# --- Solvers
# --------------------------------------------------------------------------------------------------


def get_bl_ant_pairs(corrprod_lookup):
    """
    Get antenna lists in solver format, from `corrprod_lookup`

    Inputs:
    -------
    corrprod_lookup : array
        lookup table of antenna indices for each baseline, array shape(nant,2)

    Returns:
    --------
    antlist1, antlist2 : list
        lists of antennas matching the correlation_product lookup table,
        appended with their conjugates (format required by stefcal)
    """

    # NOTE: no longer need hh and vv masks as we re-ordered the data to be
    # ntime x nchan x nbl x npol

    # get antenna number lists for stefcal - need vis then vis.conj (assume
    # constant over an observation)
    # assume same for hh and vv
    antlist1 = np.concatenate((corrprod_lookup[:, 0], corrprod_lookup[:, 1]))
    antlist2 = np.concatenate((corrprod_lookup[:, 1], corrprod_lookup[:, 0]))

    return antlist1, antlist2


@numba.guvectorize(['c8[:], int_[:], int_[:], f4[:], int_[:], c8[:], int_[:], f4[:], c8[:]',
                    'c16[:], int_[:], int_[:], f8[:], int_[:], c16[:], int_[:], f8[:], c16[:]'],
                   '(n),(n),(n),(n),(),(a),(),()->(a)', nopython=True)
def _stefcal_gufunc(rawvis, ant1, ant2, weights, ref_ant, init_gain, num_iters, conv_thresh, g):
    ref_ant2 = max(ref_ant[0], 0)
    num_ants = init_gain.shape[0]
    R = np.zeros((num_ants, num_ants), rawvis.dtype)    # Weighted visibility matrix
    M = np.zeros((num_ants, num_ants), weights.dtype)   # Weighted model
    # Convert list of baselines into a covariance matrix
    for i in range(len(ant1)):
        a = ant1[i]
        b = ant2[i]
        if a != b:
            weighted_vis = rawvis[i] * weights[i]
            R[a, b] = weighted_vis
            R[b, a] = np.conj(weighted_vis)
            M[a, b] = weights[i]
            M[b, a] = weights[i]
    g_old = init_gain.copy()
    for n in range(num_iters[0]):
        for p in range(num_ants):
            gn = g.dtype.type(0)
            gd = g.real.dtype.type(0)
            for i in range(num_ants):
                z = g_old[i] * M[p, i]
                gn += R[p, i] * z       # exploiting R[p, i] = R[i, p].conj()
                gd += z.real * z.real + z.imag * z.imag
            g[p] = gn / gd
        ref = g[ref_ant2]
        g *= np.conj(ref) / np.abs(ref)
        # Salvini & Wijnholds tweak gains every *even* iteration but their counter starts at 1
        if n % 2:
            # Check for convergence of relative l_2 norm of change in gain vector
            dnorm2 = g.real.dtype.type(0)
            gnorm2 = g.real.dtype.type(0)
            for i in range(num_ants):
                delta = g[i] - g_old[i]
                dnorm2 += delta.real * delta.real + delta.imag * delta.imag
                gnorm2 += g[i].real * g[i].real + g[i].imag * g[i].imag
            # The sense of this if test is carefully chosen so that if
            # g contains nans, the loop will exit (there is no point
            # continuing, since the nans can only spread.
            if dnorm2 >= conv_thresh[0] * conv_thresh[0] * gnorm2:
                # Avoid getting stuck bouncing between two gain vectors by
                # going halfway in between
                for i in range(num_ants):
                    g[i] = (g[i] + g_old[i]) / 2
            else:
                break
        g_old[:] = g
    if ref_ant[0] < 0:
        middle_angle = np.median(g.real) - np.median(g.imag) * g.dtype.type(1j)
        middle_angle /= np.abs(middle_angle)
        g *= middle_angle


def stefcal(rawvis, num_ants, corrprod_lookup, weights=None, ref_ant=0,
            init_gain=None, num_iters=30, conv_thresh=0.0001):
    """Solve for antenna gains using StEFCal.

    The observed visibilities are provided in a NumPy array of any shape and
    dimension, as long as the last dimension represents baselines. The gains
    are then solved in parallel for the rest of the dimensions. For example,
    if the *vis* array has shape (T, F, B) containing *T* dumps / timestamps,
    *F* frequency channels and *B* baselines, the resulting gain array will be
    of shape (T, F, num_ants), where *num_ants* is the number of antennas.

    Parameters
    ----------
    rawvis : array of complex, shape (M, ..., N)
        Complex cross-correlations between antennas A and B, assuming *N*
        baselines or antenna pairs on the last dimension
    num_ants : int
        Number of antennas
    corrprod_lookup : numpy array of int, shape (N, 2)
        First and second antenna indices associated with visibilities
    weights : float or array of float, shape (M, ..., N), optional
        Visibility weights (positive real numbers)
    ref_ant : int, optional
        Reference antenna for which phase will be forced to 0.0. Alternatively,
        if *ref_ant* is -1, the median gain phase will be 0.
    init_gain : array of complex, shape(num_ants,), optional
        Initial gain vector (all equal to 1.0 by default)
    num_iters : int, optional
        Number of iterations
    conv_thresh : float, optional
        Convergence threshold (max relative l_2 norm of change in gain vector)

    Returns
    -------
    gains : array of complex, shape (M, ..., num_ants)
        Complex gains per antenna

    Notes
    -----
    The model visibilities are assumed to be 1, implying a point source model.
    The algorithm is iterative but should converge in a small number of
    iterations (10 to 30).

    References
    ----------
    .. [1] Salvini, Wijnholds, "Fast gain calibration in radio astronomy using
       alternating direction implicit methods: Analysis and applications," 2014,
       preprint at `<http://arxiv.org/abs/1410.2101>`_
    """
    if init_gain is None:
        init_gain = np.ones(num_ants, rawvis.dtype)
    if init_gain.shape[-1] != num_ants:
        raise ValueError('initial gains have wrong length {} for number of antennas {}'.format(
            init_gain.shape[-1], num_ants))
    if weights is None:
        weights = rawvis.real.dtype.type(1.0)
    if not all(0 <= x < num_ants for x in corrprod_lookup.flat):
        raise ValueError('invalid antenna index in corrprod_lookup')
    if ref_ant >= num_ants:
        raise ValueError('invalid reference antenna')
    weights = np.broadcast_to(weights, rawvis.shape)
    return _stefcal_gufunc(rawvis, corrprod_lookup[:, 0], corrprod_lookup[:, 1], weights,
                           ref_ant, init_gain, num_iters, conv_thresh)


# --------------------------------------------------------------------------------------------------
# --- Calibration helper functions
# --------------------------------------------------------------------------------------------------

def g_from_K(chans, K):
    g_array = np.ones(K.shape+(len(chans),), dtype=np.complex)
    for i, c in enumerate(chans):
        g_array[:, :, i] = np.exp(1.0j * 2. * np.pi * K * c)
    return g_array


def ants_from_bllist(bllist):
    return len(set([item for sublist in bllist for item in sublist]))


def g_fit(data, corrprod_lookup, g0=None, refant=0, **kwargs):
    """
    Fit complex gains to visibility data.

    Parameters
    ----------
    data : visibility data, array of complex, shape(num_sol, num_chans, baseline)
    g0 : array of complex, shape(num_ants) or None
    corrprod_lookup : antenna mappings, for first then second antennas in bl pair
    refant : reference antenna

    Returns
    -------
    g_array : Array of gain solutions, shape(num_sol, num_ants)
    """
    num_ants = ants_from_bllist(corrprod_lookup)
    return stefcal(data, num_ants, corrprod_lookup, weights=1.0,
                   ref_ant=refant, init_gain=g0, **kwargs)


def k_fit(data, corrprod_lookup, chans=None, refant=0, chan_sample=1, **kwargs):
    """
    Fit delay (phase slope across frequency) to visibility data.

    Parameters
    ----------
    data : array of complex, shape(num_chans, num_pols, baseline)
    corrprod_lookup : antenna mappings, for first then second antennas in bl pair
    chans : list or array of channel frequencies, shape(num_chans), optional
    refant : reference antenna, integer, optional
    algorithm : stefcal algorithm, string, optional

    Returns
    -------
    ksoln : delay, shape(num_pols, num_ants)
    """
    # OVER THE TOP NOTE:
    # This solver was written to deal with extreme gains, which can wrap many
    # many times across the solution window, to the extent that solving for a
    # per-channel per-antenna bandpass, then fitting a line to the per-antenna
    # bandpass breaks down. It breaks down because the phase unwrapping is
    # unreliable when the phases (a) wrap so extremely, and (b) are also
    # noisy.
    #
    # This situation was adressed by first doing a coarse delay fit through an
    # FFT, for each antenna to the reference antenna. Then the course delay is
    # applied, and a secondary linear phase fit is performed with the corrected
    # data. With the coarse delays removed there should be little, or minimal
    # wrapping, and the unwrapping prior to the linear fit usually works.
    # BUT:
    # * Things fall over in the FFT if there are NaNs in the data - the NaNs
    #   propagate and everything becomes NaNs There is a temporary fix for this
    #   below, but if there are chunks of NaNs this won't work (see below)
    # * This algorithm is most effective if the delays are of similar scales.
    #   How to ensure this is to choose a reference antenna that has high delays
    #   on as many baselines as possible.  Perhaps this could be done by looking
    #   through all of the FFT results, grouped by antenna, to find the highest
    #   set of delays?
    # * In the future world where not every delay fit needs to deal with
    #   extreme delay values, maybe we could separate out the
    #   bandpass-linear-phase-fit component of this algorithm, and use just that
    #   for delay solutions we know won't be too extreme. Then this extreme delay
    #  solver could solve for the coarse delay then apply it and call the fine
    #  delay bandpass-linear-phase-fit solver.

    # -----------------------------------------------------
    # if channel sampling is specified, thin down the data and channel list
    if chan_sample != 1:
        data = data[::chan_sample, ...]
        if np.any(chans):
            chans = chans[::chan_sample]

    # set up parameters
    num_ants = ants_from_bllist(corrprod_lookup)
    num_pol = data.shape[-2] if len(data.shape) > 2 else 1
    if chans is None:
        chans = range(data.shape[0])
    nchans = len(chans)
    chan_increment = chans[1]-chans[0]

    # -----------------------------------------------------
    # initialise values for solver
    kdelay = []
    if not(np.any(chans)):
        chans = np.arange(data.shape[0], dtype=np.float32)

    # NOTE: I know that iterating over polarisation is horrible, but I ran out
    # of time to fix this up I initially just assumed we would always have two
    # polarisations, but shouldn't make that assumption especially for use of
    # calprocs offline on an h5 file
    for p in range(num_pol):
        pol_data = data[:, p, :] if len(data.shape) > 2 else data

        # if there is a nan in a channel on any baseline, remove that channel
        # from all baselines for the fit
        bad_chan_mask = np.logical_or.reduce(np.isnan(pol_data), axis=1)
        good_pol_data = pol_data[~bad_chan_mask]
        good_chans = chans[~bad_chan_mask]
        # NOTE: This removal of channels is not a proper fix for if there are a lot of NaNs.
        #   In that case there will be large gaps in the data leaving to what
        #   the FFT below sees as phase jumps This removal of NaN channels is
        #   just a temporary fix for the case of a few bad channels, so that
        #   the NaN's don't propagate and lead to NaNs throughout and NaN delay
        #   values

        # -----------------------------------------------------
        # FT to find visibility space delays
        # NOTE: This is a bit inefficient at the moment as the FFT for al
        # baselines is calculated but only the baselines to the reference
        # antenna are used for the coarse delay
        # Also note that scipy.fftpack will do a single-precision FFT given
        # single-precision input, unlike np.fft.fft which converts it to
        # double precision.
        ft_vis = scipy.fftpack.fft(good_pol_data, axis=0)
        # get index of FT maximum
        k_arg = np.argmax(np.abs(ft_vis), axis=0)
        if nchans % 2 == 0:
            k_arg[k_arg > ((nchans / 2) - 1)] -= nchans
        else:
            k_arg[k_arg > ((nchans - 1) / 2)] -= nchans

        # calculate vis space K from FT sample frequencies
        vis_k = np.float32(k_arg) / (chan_increment * nchans)

        # now determine per-antenna K values
        coarse_k = np.zeros(num_ants, np.float32)
        for ai in range(num_ants):
            k = vis_k[..., (corrprod_lookup[:, 0] == ai) & (corrprod_lookup[:, 1] == refant)]
            if k.size > 0:
                coarse_k[..., ai] = np.squeeze(k)
        for ai in range(num_ants):
            k = vis_k[..., (corrprod_lookup[:, 1] == ai) & (corrprod_lookup[:, 0] == refant)]
            if k.size > 0:
                coarse_k[..., ai] = np.squeeze(-1.0 * k)

        # apply coarse K values to the data and solve for bandpass
        v_corrected = np.zeros_like(good_pol_data)
        for vi in range(v_corrected.shape[-1]):
            for ci, c in enumerate(good_chans):
                v_corrected[ci, vi] = good_pol_data[ci, vi] \
                    * np.exp(-2.0j * np.pi * c * coarse_k[corrprod_lookup[vi, 0]]) \
                    * np.exp(2.0j * np.pi * c * coarse_k[corrprod_lookup[vi, 1]])
        bpass = stefcal(v_corrected, num_ants, corrprod_lookup, weights=1.0,
                        num_iters=100, ref_ant=refant, init_gain=None, **kwargs)

        # find slope of the residual bandpass
        delta_k = np.empty_like(coarse_k)
        for i, bp in enumerate(bpass.T):
            # np.unwrap falls over in the case of bad RFI - robustify this later
            bp_phase = np.unwrap(np.angle(bp), discont=1.9 * np.pi)
            A = np.array([good_chans, np.ones(len(good_chans))])
            delta_k[i] = np.linalg.lstsq(A.T, bp_phase)[0][0] / (2. * np.pi)

        kdelay.append(coarse_k + delta_k)

    return np.atleast_2d(kdelay)


def kcross_fit(data, flags, chans=None, chan_ave=1):
    """
    Fit delay (phase slope across frequency) offset to visibility data.

    Parameters
    ----------
    data : array of complex, shape(num_chans, num_pols, baseline)
    flags : array of bool or uint8, shape(num_chans, num_pols, baseline)
    chans : list or array of channel frequencies, shape(num_chans), optional
    chan_ave : number of channels to average together before fit, int, optional

    Returns
    -------
    kcrosssoln : delay, float
    """

    if not(np.any(chans)):
        chans = np.arange(data.shape[0])
    else:
        chans = np.array(chans)

    # average channels as specified
    ave_crosshand = np.average(
        (data[:, 0, :] * ~flags[:, 0, :] + np.conjugate(data[:, 1, :] * ~flags[:, 1, :])) / 2.,
        axis=-1)
    ave_crosshand = np.add.reduceat(
        ave_crosshand, range(0, len(ave_crosshand), chan_ave)) / chan_ave
    ave_chans = np.add.reduceat(
        chans, range(0, len(chans), chan_ave)) / chan_ave

    nchans = len(ave_chans)
    chan_increment = ave_chans[1] - ave_chans[0]

    # FT the visibilities to get course estimate of kcross
    #   with noisy data, this is more robust than unwrapping the phase
    ft_vis = np.fft.fft(ave_crosshand)

    # get index of FT maximum
    k_arg = np.argmax(ft_vis)
    if nchans % 2 == 0:
        if k_arg > ((nchans / 2) - 1):
            k_arg = k_arg - nchans
    else:
        if k_arg > ((nchans - 1) / 2):
            k_arg = k_arg - nchans

    # calculate kcross from FT sample frequencies
    coarse_kcross = k_arg / (chan_increment*nchans)

    # correst data with course kcross to solve for residual delta kcross
    corrected_crosshand = ave_crosshand*np.exp(-2.0j * np.pi * coarse_kcross * ave_chans)
    # solve for residual kcross through linear phase fit
    crossphase = np.angle(corrected_crosshand)
    # remove any offset from zero
    crossphase -= np.median(crossphase)
    # linear fit
    A = np.array([ave_chans, np.ones(len(ave_chans))])
    delta_kcross = np.linalg.lstsq(A.T, crossphase)[0][0]/(2. * np.pi)

    # total kcross
    return coarse_kcross + delta_kcross


def asbool(arr):
    """View an array as boolean.

    If possible it simply returns a view, otherwise a copy. It works on both
    dask and numpy arrays.
    """
    if arr.dtype in (np.uint8, np.int8, np.bool_):
        return arr.view(np.bool_)
    else:
        return arr.astype(np.bool_)


def solint_from_nominal(solint, dump_period, num_times):
    """
    Given nominal solint, modify it by up to 20percent to optimally fit the scan length
    and dump period. Times are assumed to be contiguous.

    Parameters
    ----------
    solint      : nominal solint
    dump_period : dump period of the data
    num_times   : number of time dumps in the scan

    Returns
    -------
    nsolint     : new optimal solint
    dumps_per_solint : number of dump periods in a solution interval
    """

    if dump_period > solint:
        # case of solution interval that is shorter than dump time
        dumps_per_solint = 1
    elif dump_period * num_times < solint:
        # case of solution interval that is longer than scan
        dumps_per_solint = num_times
    else:
        # requested number of dumps per nominal solution interval
        req_dumps_per_solint = solint / dump_period

        # range for searching: nominal solint +-20%
        dumps_per_solint_low = int(np.round(req_dumps_per_solint * 0.8))
        dumps_per_solint_high = int(np.round(req_dumps_per_solint * 1.2))
        solint_check_range = np.arange(dumps_per_solint_low,
                                       dumps_per_solint_high + 1).astype(np.int)

        # compute size of final partial interval (in dumps)
        tail = (num_times - 1) % solint_check_range + 1
        delta = solint_check_range - tail

        # choose a solint to minimise the change in the final fractional solution interval
        solint_index = np.argmin(delta)
        logger.debug('solint index: {0}'.format(solint_index,))
        dumps_per_solint = solint_check_range[solint_index]
    return dumps_per_solint * dump_period, dumps_per_solint


# --------------------------------------------------------------------------------------------------
# --- Baseline ordering
# --------------------------------------------------------------------------------------------------

def get_ant_bls(pol_bls_ordering):
    """
    Given baseline list with polarisation information, return pure antenna list
    e.g. from
    [['ant0h','ant0h'],['ant0v','ant0v'],['ant0h','ant0v'],['ant0v','ant0h'],['ant1h','ant1h']...]
    to [['ant0,ant0'],['ant1','ant1']...]
    """

    # get antenna only names from pol-included bls orderding
    ant_bls = np.array([[a1[:-1], a2[:-1]] for a1, a2 in pol_bls_ordering])
    ant_dtype = ant_bls[0, 0].dtype
    # get list without repeats (ie no repeats for pol)
    #     start with array of empty strings, shape (num_baselines x 2)
    bls_ordering = np.empty([len(ant_bls) / 4, 2], dtype=ant_dtype)

    # iterate through baseline list and only include non-repeats
    #   I know this is horribly non-pythonic. Fix later.
    bls_ordering[0] = ant_bls[0]
    bl = 0
    for c in ant_bls:
        if not np.all(bls_ordering[bl] == c):
            bl += 1
            bls_ordering[bl] = c

    return bls_ordering


def get_pol_bls(bls_ordering, pol):
    """
    Given baseline ordering and polarisation ordering, return full baseline-pol ordering array

    Inputs:
    -------
    bls_ordering
        list of correlation products without polarisation information, string shape(nbl,2)
    pol
        list of polarisation pairs, string shape (npol_pair, 2)

    Returns:
    --------
    pol_bls_ordering : :class:`np.ndarray`
        correlation products without polarisation information, shape(nbl*4, 2)
    """
    pol_ant_dtype = np.array(bls_ordering[0][0] + 'h').dtype
    nbl = len(bls_ordering)
    pol_bls_ordering = np.empty([nbl * len(pol), 2], dtype=pol_ant_dtype)
    for i, p in enumerate(pol):
        for b, bls in enumerate(bls_ordering):
            pol_bls_ordering[nbl * i + b] = bls[0] + p[0], bls[1] + p[1]
    return pol_bls_ordering


def get_reordering(antlist, bls_ordering, output_order_bls=None, output_order_pol=None):
    """
    Determine reordering necessary to change given bls_ordering into desired ordering

    Inputs:
    -------
    antlist : list of antennas, string shape(nants), or string of csv antenna names
    bls_ordering : list of correlation products, string shape(nbl,2)
    output_order_bls : desired baseline output order, string shape(nbl,2), optional
    output_order_pol : desired polarisation output order, string shape(npolcorr,2), optional

    Returns:
    --------
    ordering
        ordering array necessary to change given bls_ordering into desired
        ordering, numpy array shape(nbl*4, 2)
    bls_wanted
        ordering of baselines, without polarisation, list shape(nbl, 2)
    pol_order
        ordering of polarisations, list shape (4, 2)

    """
    # convert antlist to list, if it is a csv string
    if isinstance(antlist, str):
        antlist = antlist.split(',')
    # convert bls_ordering to a list, if it is not a list (e.g. np.ndarray)
    if not isinstance(bls_ordering, list):
        bls_ordering = bls_ordering.tolist()

    # get current antenna list without polarisation
    bls_ordering_nopol = [[b[0][0:4], b[1][0:4]] for b in bls_ordering]
    # find unique elements
    unique_bls = []
    for b in bls_ordering_nopol:
        if b not in unique_bls:
            unique_bls.append(b)

    # determine polarisation ordering
    if output_order_pol is not None:
        pol_order = output_order_pol
    else:
        # get polarisation products from current antenna list
        bls_ordering_polonly = [[b[0][4], b[1][4]] for b in bls_ordering]
        # find unique pol combinations
        unique_pol = []
        for pbl in bls_ordering_polonly:
            if pbl not in unique_pol:
                unique_pol.append(pbl)
        # put parallel polarisations first
        pol_order = []
        if len(bls_ordering_polonly) > 2:
            for pp in unique_pol:
                if pp[0] == pp[1]:
                    pol_order.insert(0, pp)
                else:
                    pol_order.append(pp)
    npol = len(pol_order)

    if output_order_bls is None:
        # default output order is XC then AC
        bls_wanted = [b for b in unique_bls if b[0] != b[1]]

        # number of baselines including autocorrelations
        nants = len(antlist)
        nbl_ac = nants * (nants + 1) / 2

        # add AC to bls list, if necessary
        # assume that AC are either included or not (no partial states)
        if len(bls_ordering) == (nbl_ac * npol):
            # data include XC and AC
            bls_wanted.extend([b for b in unique_bls if b[0] == b[1]])

        bls_pol_wanted = get_pol_bls(bls_wanted, pol_order)
    else:
        bls_pol_wanted = get_pol_bls(output_order_bls, pol_order)
    # note: bls_pol_wanted must be an np array for list equivalence below

    # find ordering necessary to change given bls_ordering into desired ordering
    # note: ordering must be a numpy array to be used for indexing later
    ordering = np.array([np.all(bls_ordering == bls, axis=1).nonzero()[0][0]
                         for bls in bls_pol_wanted])

    # how to use this:
    # print bls_ordering[ordering]
    # print bls_ordering[ordering].reshape([4,nbl,2])
    return ordering, bls_wanted, pol_order


def get_reordering_nopol(antlist, bls_ordering, output_order_bls=None):
    """
    Determine reordering necessary to change given bls_ordering into desired ordering

    Inputs:
    -------
    antlist : list of antennas, string shape(nants), or string of csv antenna names
    bls_ordering : list of correlation products, string shape(nbl,2)
    output_order_bls : desired baseline output order, string shape(nbl,2), optional

    Returns:
    --------
    ordering
        ordering array necessary to change given bls_ordering into desired
        ordering, numpy array, shape(nbl*4, 2)
    bls_wanted
        ordering of baselines, without polarisation, numpy array, shape(nbl, 2)

    """
    # convert antlist to list, if it is a csv string
    if isinstance(antlist, str):
        antlist = antlist.split(',')
    # convert bls_ordering to a list, if it is not a list (e.g. np.ndarray)
    if not isinstance(bls_ordering, list):
        bls_ordering = bls_ordering.tolist()

    # find unique elements
    unique_bls = []
    for b in bls_ordering:
        if b not in unique_bls:
            unique_bls.append(b)

    # defermine output ordering:
    if output_order_bls is None:
        # default output order is XC then AC
        bls_wanted = [b for b in unique_bls if b[0] != b[1]]
        # add AC to bls list
        bls_wanted.extend([b for b in unique_bls if b[0] == b[1]])
    else:
        # convert to list in case it is not a list
        bls_wanted = output_order_bls
    # note: bls_wanted must be an np array for list equivalence below
    bls_wanted = np.array(bls_wanted)

    # find ordering necessary to change given bls_ordering into desired ordering
    # note: ordering must be a numpy array to be used for indexing later
    ordering = np.array([np.all(bls_ordering == bls, axis=1).nonzero()[0][0]
                        for bls in bls_wanted])
    # how to use this:
    # print bls_ordering[ordering]
    # print bls_ordering[ordering].reshape([4,nbl,2])
    return ordering, bls_wanted


def get_bls_lookup(antlist, bls_ordering):
    """
    Get correlation product antenna mapping

    Inputs:
    -------
    antlist : list of antenna names, string shape (nant)
    bls_ordering : list of correlation products, string shape(nbl,2)

    Returns:
    --------
    corrprod_lookup : lookup table of antenna indices for each baseline, shape(nbl,2)
    """

    # make polarisation and corr_prod lookup tables (assume this doesn't change
    # over the course of an observaton)
    antlist_index = dict([(antlist[i], i) for i in range(len(antlist))])
    return np.array([[antlist_index[a1[0:4]], antlist_index[a2[0:4]]] for a1, a2 in bls_ordering])


# --------------------------------------------------------------------------------------------------
# --- Simulation
# --------------------------------------------------------------------------------------------------

def fake_vis(shape=(7,), gains=None, noise=None, random_state=None):
    """Create fake point source visibilities, corrupted by given or random gains. The
    final dimension of `shape` corresponds to the number of antennas.
    """
    # create antenna lists
    if isinstance(shape, (int, long)):
        shape = (shape,)
    shape = tuple(shape)
    nants = shape[-1]

    antlist = range(nants)
    list1 = np.array([])
    for a, i in enumerate(range(nants - 1, 0, -1)):
        list1 = np.hstack([list1, np.ones(i) * a])
    list1 = np.hstack([list1, antlist])
    list1 = np.int_(list1)

    list2 = np.array([], dtype=np.int)
    mod_antlist = antlist[1:]
    for i in range(0, len(mod_antlist)):
        list2 = np.hstack([list2, mod_antlist[:]])
        mod_antlist.pop(0)
    list2 = np.hstack([list2, antlist])

    # create fake gains, if gains are not given as input
    if random_state is None:
        random_state = np.random
    if gains is None:
        gains = random_state.random_sample(shape) + 1j * random_state.random_sample(shape)
    else:
        assert shape == gains.shape

    # create fake corrupted visibilities
    nbl = nants * (nants + 1) // 2
    vis = np.ones(tuple(shape[:-1]) + (nbl,), gains.dtype)
    # corrupt vis with gains
    for i, (a, b) in enumerate(zip(list1, list2)):
        vis[..., i] *= gains[..., a] * gains[..., b].conj()
    # if requested, corrupt vis with noise
    if noise is not None:
        vis_noise = random_state.standard_normal(vis.shape) * noise
        vis = vis + vis_noise

    # return useful info
    bl_pair_list = np.column_stack([list1, list2])
    return vis, bl_pair_list, gains


# --------------------------------------------------------------------------------------------------
# --- CLASS :  CalSolution
# --------------------------------------------------------------------------------------------------

class CalSolution(object):
    """Calibration solution store."""
    def __init__(self, soltype, solvalues, soltimes):
        self.soltype = soltype
        self.values = solvalues
        self.times = soltimes
        self.ts_solname = 'cal_product_{}'.format(soltype)

    def __str__(self):
        """String representation of calibration solution to help identify it."""
        # Obtain human-friendly timestamp representing the centre of solutions
        timestamp = time.strftime("%H:%M:%S", time.gmtime(np.mean(self.times)))
        return "{} {} {}".format(self.soltype, self.values.shape, timestamp)


# --------------------------------------------------------------------------------------------------
# --- General helper functions
# --------------------------------------------------------------------------------------------------

def arcsec_to_rad(angle):
    """
    Convert angle in arcseconds to angle in radians
    """
    return np.deg2rad(angle / 60. / 60.)
