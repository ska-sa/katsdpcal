"""
Calibration procedures for MeerKAT calibration pipeline
=======================================================

Solvers and averagers for use in the MeerKAT calibration pipeline.
"""

import numpy as np
import copy
import katpoint

import logging
logger = logging.getLogger(__name__)

#--------------------------------------------------------------------------------------------------
#--- Modelling procedures
#--------------------------------------------------------------------------------------------------

def radec_to_lm(ra, dec, ra0, dec0):
    """
    Parameteters
    ------------
    ra : Right Ascension, radians
    dec : Declination, radians
    ra0 : Right Ascention of centre, radians
    dec0 : Declination of centre, radians

    Returns
    -------
    l, m : direction cosines
    """
    l = np.cos(dec)*np.sin(ra-ra0)
    m = np.sin(dec)*np.cos(dec0)-np.cos(dec)*np.sin(dec0)*np.cos(ra-ra0)
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
    t : time in MJD seconds

    Returns
    -------
    Unix time in seconds
    """
    return (t/86400. - 2440587.5 + 2400000.5)*86400.

def calc_uvw_wave(phase_centre, timestamps, corrprod_lookup, ant_descriptions, wavelengths=None, array_centre=None):
    """
    Calculate uvw coordinates

    Parameters
    ----------
    phase_centre : katpoint target for phase centre position
    timestamps : times, array of floats, shape(nrows)
    corrprod_lookup : lookup table of antenna indices for each baseline, array shape(nant,2)
    antenna_descriptions : description strings for the antennas, same order as antlist, list of string
    wavelengths : wavelengths, single value or array shape(nchans)
    array_centre : description string for array centre position, string

    Returns
    -------
    uvw_wave : uvw coordinates, normalised by wavelength
    """
    uvw = calc_uvw(phase_centre, timestamps, corrprod_lookup, ant_descriptions, array_centre)
    if wavelengths is None:
        return uvw
    elif np.isscalar(wavelengths):
        return uvw/wavelengths
    else:
        return uvw[:,:,np.newaxis,:]/wavelengths[:,np.newaxis]

def calc_uvw(phase_centre, timestamps, corrprod_lookup, ant_descriptions, array_centre=None):
    """
    Calculate uvw coordinates

    Parameters
    ----------
    phase_centre : katpoint target for phase centre position
    timestamps : times, array of floats, shape(nrows)
    corrprod_lookup : lookup table of antenna indices for each baseline, array shape(nant,2)
    antenna_descriptions : description strings for the antennas, same order as antlist, list of string
    array_centre : description string for array centre position, string

    Returns
    -------
    uvw_wave : uvw coordinates
    """
    if array_centre is not None:
        array_reference_position = katpoint.Antenna(array_centre)
    else:
        # if no array centre position is given, use lat-long-alt of first antenna in the antenna list
        refant = katpoint.Antenna(ant_descriptions[0])
        array_reference_position = katpoint.Antenna('array_position',*refant.ref_position_wgs84)

    # use the array reference position for the basis
    basis = phase_centre.uvw_basis(timestamp=timestamps, antenna=array_reference_position)
    antenna_uvw = np.empty([len(ant_descriptions),3,len(timestamps)])

    for i, antenna in enumerate(ant_descriptions):
        ant = katpoint.Antenna(antenna)
        enu = np.array(ant.baseline_toward(array_reference_position))
        v = np.tensordot(basis, enu, ([1], [0]))
        antenna_uvw[i,...] = np.tensordot(basis, enu, ([1], [0]))

    baseline_uvw = np.empty([3,len(timestamps),len(corrprod_lookup)])
    for i,[a1,a2] in enumerate(corrprod_lookup):
        baseline_uvw[...,i] = antenna_uvw[a2] - antenna_uvw[a1]

    return baseline_uvw

#--------------------------------------------------------------------------------------------------
#--- Solvers
#--------------------------------------------------------------------------------------------------

def get_bl_ant_pairs(corrprod_lookup):
    """
    Get antenna lists in solver format, from corr_prod lookup

    Inputs:
    -------
    corrprod_lookup : lookup table of antenna indices for each baseline, array shape(nant,2)

    Returns:
    --------
    antlist 1, antlist 2 : lists of antennas matching the correlation_product lookup table,
        appended with their conjugates (format required by stefcal)
    """

    # NOTE: no longer need hh and vv masks as we re-ordered the data to be ntime x nchan x nbl x npol

     # get antenna number lists for stefcal - need vis then vis.conj (assume constant over an observation)
    # assume same for hh and vv
    antlist1 = np.concatenate((corrprod_lookup[:,0], corrprod_lookup[:,1]))
    antlist2 = np.concatenate((corrprod_lookup[:,1], corrprod_lookup[:,0]))

    return antlist1, antlist2

def stefcal(vis, num_ants, corrprod_lookup, weights=1.0, ref_ant=0, init_gain=None,
    model=1.0, algorithm='adi_schwardt', num_iters=100, conv_thresh=0.0001, verbose=False):
    """Solve for antenna gains using ADI StefCal.
    ADI StefCal implimentation from:
    'Fast gain calibration in radio astronomy using alternating direction implicit methods:
    Analysis and applications', Salvini & Winjholds, 2014

    Parameters
    ----------
    vis : array of complex, shape (N,)
        Complex cross-correlations between antennas A and B
    num_ants : int
        Number of antennas
    corrprod_lookup : numpy array of int, array of shape (2,N)
        First and second antenna indices associated with visibilities
    num_iters : int, optional
        Number of iterations
    ref_ant : int, optional
        Reference antenna whose gain will be forced to be 1.0
    init_gain : array of complex, shape(num_ants,) or None, optional
        Initial gain vector (all equal to 1.0 by default)
    conv_thresh : float, optional
        Convergence threshold, only for ADI stefcal
    algorithm : string, optional
        Stefcal algorithm:
        'adi' -      ADI stefcal (default)
        'schwardt' - Schwardt stefcal

    Returns
    -------
    gains : array of complex, shape (num_ants,)
        Complex gains, one per antenna

    """
    bl_ant_pairs = get_bl_ant_pairs(corrprod_lookup)
    if algorithm == 'adi':
        return adi_stefcal(vis, num_ants, bl_ant_pairs, weights, ref_ant, init_gain, model,
                num_iters, conv_thresh, verbose)
    if algorithm == 'adi_nonnumpy':
        # this algorithm can only run on 2D data, shape (N, nbl)
        vis2d = np.atleast_2d(vis)
        gains = np.zeros([vis2d.shape[0],num_ants], dtype=np.complex)
        for i, c_vis in enumerate(vis2d):
            gains[i] = adi_stefcal_nonparallel(c_vis, num_ants, bl_ant_pairs, weights, ref_ant,
                init_gain, model, num_iters, conv_thresh, verbose)
        return gains	
    elif algorithm == 'schwardt':
        return schwardt_stefcal(vis, num_ants, bl_ant_pairs, weights, ref_ant,
                init_gain, num_iters, verbose)
    elif algorithm == 'adi_schwardt':
        return adi_schwardt_stefcal(vis, num_ants, bl_ant_pairs, weights, ref_ant,
                init_gain, model, num_iters, conv_thresh, verbose)
    else:
        raise ValueError(' '+algorithm+' is not a valid stefcal implimentation.')

def adi_stefcal_nonparallel(vis, num_ants, bl_ant_pairs, weights=1.0, ref_ant=0, init_gain=None, model=1.,
	num_iters=100, conv_thresh=0.0001, verbose=False):
    """Solve for antenna gains using ADI StefCal. Non parallel version of the algorithm.
    ADI StefCal implimentation from:
    'Fast gain calibration in radio astronomy using alternating direction implicit methods:
    Analysis and applications', Salvini & Winjholds, 2014
    Parameters
    ----------
    vis         : array of complex, shape (N,)
        Complex cross-correlations between antennas A and B
    num_ants    : int
        Number of antennas
    bl_ant_pairs : numpy array of int, shape (2,2*N)
        First and second antenna indices associated with visibilities, repeated twice
    num_iters   : int, optional
        Number of iterations
    ref_ant     : int, optional
        Reference antenna whose gain will be forced to be 1.0
    init_gain   : array of complex, shape(num_ants,) or None, optional
        Initial gain vector (all equal to 1.0 by default)
    model       : array of complex, shape(num_ants, num_ants) or None, optional
        Sky model
    conv_thresh : float, optional
        Convergence threshold
    Returns
    -------
    gains : array of complex, shape (num_ants,)
        Complex gains, one per antenna
    """
    # Initialise gain matrix
    g_prev = np.ones(num_ants, dtype=np.complex) if init_gain is None else init_gain
    g_curr = 1.0*g_prev
    # initialise calibrator source model
    #   default is unity with zeros along the diagonals to ignore autocorr
    M = model * (1.0 - np.eye(num_ants, dtype=np.complex)) if np.isscalar(model) else model

    antA, antB = bl_ant_pairs
    for i in range(num_iters):
        # iterate through antennas, solving for each gain
        for p in range(num_ants):
            # z <- g_prev odot M[:,p]
            z = [g*m for g,m in zip(g_prev,M[:,p])]
            z.pop(p)

            # R[:,p]
            antA_vis = vis[antA==p]
            # antenna order of R[:,p]
            antB_order = antB[antA==p].tolist()

            # g[p] <- (R[:,p] dot z)/(z^H dot z)
            antlist = range(num_ants)
            antlist.pop(p)
            g_curr[p] = np.sum([antA_vis[antB_order.index(j)]*zj for j,zj in zip(antlist,z)])/(np.dot(np.conjugate(z),z))

            # Force reference gain to be zero phase
            g_curr *= abs(g_curr[ref_ant])/g_curr[ref_ant]

        # for even iterations, check convergence
        # for odd iterations, average g_curr and g_prev
        #  note - i starts at zero, unlike Salvini & Winjholds (2014) algorithm
        if np.mod(i,2) == 1:
            # convergence criterion:
            # abs(g_curr - g_prev) / abs(g_curr) < tau
            diff = np.sum(np.abs(g_curr - g_prev)/np.abs(g_curr))
            if diff < conv_thresh:
                break
        else:
            # G_curr <- (G_curr + G_prev)/2
            g_curr = (g_curr + g_prev)/2.0

        # for next iteration, set g_prev to g_curr
        g_prev = 1.0*g_curr

        # if max iters reached without convergence, log it
        if i==num_iters-1: logger.debug('ADI stefcal convergence not reached after {0} iterations'.format(num_iters,))

    return g_curr

def adi_stefcal(vis, num_ants, bl_ant_pairs, weights=1.0, ref_ant=0, init_gain=None, model=1.,
	 num_iters=100, conv_thresh=0.0001, verbose=False):
    """Solve for antenna gains using ADI StefCal. Parallel version of the algorithm.
    ADI StefCal implimentation from:
    'Fast gain calibration in radio astronomy using alternating direction implicit methods:
    Analysis and applications', Salvini & Winjholds, 2014

    Parameters
    ----------
    vis : array of complex, shape (M, ..., N)
        Complex cross-correlations between antennas A and B, assuming *N*
        baselines or antenna pairs on the last dimension
    num_ants : int
        Number of antennas
    bl_ant_pairs : numpy array of int, shape (2,2*N)
        First and second antenna indices associated with visibilities, repeated twice
    num_iters   : int, optional
        Number of iterations
    ref_ant     : int, optional
        Reference antenna whose gain will be forced to be 1.0
    init_gain   : array of complex, shape(M, ..., num_ants) or None, optional
        Initial gain vector (all equal to 1.0 by default)
    model       : array of complex, shape(num_ants, num_ants) or None, optional
        Sky model
    conv_thresh : float, optional
        Convergence threshold

    Returns
    -------
    gains : array of complex, shape (M, ..., num_ants)
        Complex gains per antenna
    """
    # Initialise gain matrix
    gain_shape = tuple(list(vis.shape[:-1]) + [num_ants])
    g_prev = np.ones(gain_shape, dtype=np.complex) if init_gain is None else init_gain
    logger.debug("StefCal solving for %s gains from vis with shape %s" %
                 ('x'.join(str(gs) for gs in gain_shape), vis.shape))
    g_curr = copy.copy(g_prev)

    # initialise calibrator source model
    #   default is unity with zeros along the diagonals to ignore autocorr
    M = model * (1.0 - np.eye(num_ants, dtype=np.complex)) if np.isscalar(model) else model

    antA, antB = bl_ant_pairs
    for i in range(num_iters):
        # iterate through antennas, solving for each gain
        for p in range(num_ants):
            # z <- g_prev odot M[:,p]
            z = g_prev*M[...,p]
            z = np.delete(z,p,axis=-1)

            # R[:,p]
            antA_vis = vis[...,antA==p]
            # antenna order of R[:,p]
            antB_order = antB[antA==p].tolist()

            antlist = range(num_ants)
            antlist.pop(p)
            antB_order_indices = [antB_order.index(j) for j in antlist]

            # g[p] <- (R[:,p] dot z)/(z^H dot z)
            g_curr[...,p] = np.sum(antA_vis[...,antB_order_indices]*z,axis=-1)/np.sum(z.conj()*z,axis=-1)

            # Force reference gain to be zero phase
            #norm_val = abs(g_curr[...,ref_ant][..., np.newaxis])/g_curr[...,ref_ant][..., np.newaxis]
            #g_curr *= norm_val

            # Get gains for reference antenna (or first one, if none given)
            g_ref = g_curr[..., max(ref_ant, 0)][..., np.newaxis].copy()
            # Force reference gain to have zero phase
            g_curr *= np.abs(g_ref) / g_ref

        # for even iterations, check convergence
        # for odd iterations, average g_curr and g_prev
        #  note - i starts at zero, unlike Salvini & Winjholds (2014) algorithm
        if np.mod(i,2) == 1:
            # convergence criterion:
            # abs(g_curr - g_prev) / abs(g_curr) < tau
            diff = np.sum(np.abs(g_curr - g_prev)/np.abs(g_curr))
            if diff < conv_thresh:
                break
        else:
            # G_curr <- (G_curr + G_prev)/2
            g_curr = (g_curr + g_prev)/2.0

        # for next iteration, set g_prev to g_curr
        g_prev = copy.copy(g_curr)

        # if max iters reached without convergence, log it
        if i==num_iters-1: logger.debug('ADI stefcal convergence not reached after {0} iterations'.format(num_iters,))

    return g_curr

def adi_stefcal_acorr(vis, num_ants, bl_ant_pairs, weights=1.0, ref_ant=0, init_gain=None,
	model=1.,  num_iters=100, conv_thresh=0.0001, verbose=False):
    """Solve for antenna gains using ADI StefCal, including fake autocorr data. Non parallel version of
    the algorithm.
    ADI StefCal implimentation from:
    'Fast gain calibration in radio astronomy using alternating direction implicit methods:
    Analysis and applications', Salvini & Winjholds, 2014

    Parameters
    ----------
    vis         : array of complex, shape (N,)
        Complex cross-correlations between antennas A and B
    num_ants    : int
        Number of antennas
    bl_ant_pairs : numpy array of int, shape (2,2*N)
        First and second antenna indices associated with visibilities, repeated twice
    num_iters   : int, optional
        Number of iterations
    ref_ant     : int, optional
        Reference antenna whose gain will be forced to be 1.0
    init_gain   : array of complex, shape(num_ants,) or None, optional
        Initial gain vector (all equal to 1.0 by default)
    model       : array of complex, shape(num_ants, num_ants) or None, optional
        Sky model
    conv_thresh : float, optional
        Convergence threshold

    Returns
    -------
    gains : array of complex, shape (num_ants,)
        Complex gains, one per antenna
    """
    # fudge add pretend autocorr data to the end
    vis = np.concatenate((vis,np.zeros(num_ants,dtype=np.complex)))
    antA = np.concatenate((antA,range(num_ants)))
    antB = np.concatenate((antB,range(num_ants)))

    # Initialise gain matrix
    g_prev = np.ones(num_ants, dtype=np.complex) if init_gain is None else init_gain
    g_curr = 1.0*g_prev
    # initialise calibrator source model
    #   default is unity with zeros along the diagonals to ignore autocorr
    M = model * (1.0 - np.eye(num_ants, dtype=np.complex)) if np.isscalar(model) else model

    antA, antB = bl_ant_pairs
    for i in range(num_iters):
        # iterate through antennas, solving for each gain
        for p in range(num_ants):
            # z <- g_prev odot M[:,p]
            z = [g*m for g,m in zip(g_prev,M[:,p])]

            # R[:,p]
            antA_vis = vis[antA==p]
            # antenna order of R[:,p]
            antB_order = antB[antA==p].tolist()

            # g[p] <- (R[:,p] dot z)/(z^H dot z)
            g_curr[p] = np.sum([antA_vis[antB_order.index(j)]*z[j] for j in range(num_ants)])/(np.dot(np.conjugate(z),z))

            # Force reference gain to be zero phase
            g_curr *= abs(g_curr[ref_ant])/g_curr[ref_ant]

        # for even iterations, check convergence
        # for odd iterations, average g_curr and g_prev
        #  note - i starts at zero, unlike Salvini & Winjholds (2014) algorithm
        if np.mod(i,2) == 1:
            # convergence criterion:
            # abs(g_curr - g_prev) / abs(g_curr) < tau
            diff = np.sum(np.abs(g_curr - g_prev)/np.abs(g_curr))
            if diff < conv_thresh:
                break
        else:
            # G_curr <- (G_curr + G_prev)/2
            g_curr = (g_curr + g_prev)/2.0

        # for next iteration, set g_prev to g_curr
        g_prev = 1.0*g_curr

    return g_curr

def adi_schwardt_stefcal(rawvis, num_ants, bl_ant_pairs, weights=1.0, ref_ant=0,
	init_gain=None, model=None, num_iters=30, conv_thresh=0.0001, verbose=False):
    """Solve for antenna gains using StEFCal (array dot product version).
    The observed visibilities are provided in a NumPy array of any shape and
    dimension, as long as the last dimension represents baselines. The gains
    are then solved in parallel for the rest of the dimensions. For example,
    if the *vis* array has shape (T, F, B) containing *T* dumps / timestamps,
    *F* frequency channels and *B* baselines, the resulting gain array will be
    of shape (T, F, num_ants), where *num_ants* is the number of antennas.
    In order to get a proper solution it is important to include the conjugate
    visibilities as well by reversing antenna pairs, e.g. by forming
    full_vis = np.concatenate((vis, vis.conj()), axis=-1)
    full_antA = np.r_[antA, antB]
    full_antB = np.r_[antB, antA]

    Parameters
    ----------
    rawvis : array of complex, shape (M, ..., N)
        Complex cross-correlations between antennas A and B, assuming *N*
        baselines or antenna pairs on the last dimension
    num_ants : int
        Number of antennas
    antA, antB : array of int, shape (N,)
        Antenna indices associated with visibilities
    weights : float or array of float, shape (M, ..., N), optional
        Visibility weights (positive real numbers)
    num_iters : int, optional
        Number of iterations
    ref_ant : int, optional
        Reference antenna for which phase will be forced to 0.0. Alternatively,
        if *ref_ant* is -1, the median gain phase will be 0.
    init_gain : array of complex, shape(num_ants,) or None, optional
        Initial gain vector (all equal to 1.0 by default)
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
    # ignore autocorr data
    antA, antB = bl_ant_pairs
    xcorr = antA!=antB
    if np.all(xcorr):
        vis = rawvis
    else:
        vis = rawvis[...,xcorr]
        antA_new = antA[xcorr]
        antB = antB[xcorr]
        antA = antA_new
        # log a warning as the XC visiblilties are copied
        logger.warning('Autocorr visibilities present in StEFCal solver. Solver running on copy of crosscorr visibilities.')

    # Each row of this array contains the indices of baselines with the same antA
    baselines_per_antA = np.array([(antA == m).nonzero()[0] for m in range(num_ants)])
    # Each row of this array contains the corresponding antB indices with the same antA
    antB_per_antA = antB[baselines_per_antA]
    weighted_vis = weights * vis
    weighted_vis = weighted_vis[..., baselines_per_antA]
    # Initial estimate of gain vector
    gain_shape = tuple(list(vis.shape[:-1]) + [num_ants])
    g_curr = np.ones(gain_shape, dtype=np.complex) if init_gain is None else init_gain
    logger.debug("StEFCal solving for %s gains from vis with shape %s" %
                 ('x'.join(str(gs) for gs in gain_shape), vis.shape))
    for n in range(num_iters):
        # Basis vector (collection) represents gain_B* times model (assumed 1)
        g_basis = g_curr[..., antB_per_antA]
        # Do scalar least-squares fit of basis vector to vis vector for whole collection in parallel
        g_new = (g_basis * weighted_vis).sum(axis=-1) / (g_basis.conj() * g_basis).sum(axis=-1)
        # Get gains for reference antenna (or first one, if none given)
        g_ref = g_new[..., max(ref_ant, 0)][..., np.newaxis].copy()
        # Force reference gain to have zero phase
        g_new *= np.abs(g_ref) / g_ref
        logger.debug("Iteration %d: mean absolute gain change = %f" %
                     (n + 1, 0.5 * np.abs(g_new - g_curr).mean()))
        # Salvini & Wijnholds tweak gains every *even* iteration but their counter starts at 1
        if n % 2:
            # Check for convergence of relative l_2 norm of change in gain vector
            error = np.linalg.norm(g_new - g_curr, axis=-1) / np.linalg.norm(g_new, axis=-1)
            if np.max(error) < conv_thresh:
                g_curr = g_new
                break
            else:
                # Avoid getting stuck bouncing between two gain vectors by going halfway in between
                g_curr = 0.5 * (g_new + g_curr)
        else:
            # Normal update every *odd* iteration
            g_curr = g_new
    if ref_ant < 0:
        middle_angle = np.arctan2(np.median(g_curr.imag, axis=-1),
                                  np.median(g_curr.real, axis=-1))
        g_curr *= np.exp(-1j * middle_angle)[..., np.newaxis]
    return g_curr

def schwardt_stefcal(rawvis, num_ants, bl_ant_pairs, weights=1.0, ref_ant=0, init_gain=None, num_iters=100, verbose=False):
    """Solve for antenna gains using StefCal (array dot product version).
    The observed visibilities are provided in a NumPy array of any shape and
    dimension, as long as the last dimension represents baselines. The gains
    are then solved in parallel for the rest of the dimensions. For example,
    if the *vis* array has shape (T, F, B) containing *T* dumps / timestamps,
    *F* frequency channels and *B* baselines, the resulting gain array will be
    of shape (T, F, num_ants), where *num_ants* is the number of antennas.
    In order to get a proper solution it is important to include the conjugate
    visibilities as well by reversing antenna pairs, e.g. by forming
    full_vis = np.concatenate((vis, vis.conj()), axis=-1)
    full_antA = np.r_[antA, antB]
    full_antB = np.r_[antB, antA]
    Parameters
    ----------
    rawvis : array of complex, shape (M, ..., N)
        Complex cross-correlations between antennas A and B, assuming *N*
        baselines or antenna pairs on the last dimension
    num_ants : int
        Number of antennas
    bl_ant_pairs : numpy array of int, shape (2,2*N)
        First and second antenna indices associated with visibilities, repeated twice
    weights : float or array of float, shape (M, ..., N), optional
        Visibility weights (positive real numbers)
    num_iters : int, optional
        Number of iterations
    ref_ant : int, optional
        Reference antenna whose gain will be forced to be 1.0. Alternatively,
        if *ref_ant* is -1, the average gain magnitude will be 1 and the median
        gain phase will be 0.
    init_gain : array of complex, shape(num_ants,) or None, optional
        Initial gain vector (all equal to 1.0 by default)
    Returns
    -------
    gains : array of complex, shape (M, ..., num_ants)
        Complex gains per antenna
    Notes
    -----
    The model visibilities are assumed to be 1, implying a point source model.
    The algorithm is iterative but should converge in a small number of
    iterations (10 to 30).
    """
    # ignore autocorr data
    antA, antB = bl_ant_pairs
    xcorr = antA!=antB
    if np.all(xcorr):
        vis = rawvis
    else:
        vis = rawvis[...,xcorr]
        antA_new = antA[xcorr]
        antB = antB[xcorr]
        antA = antA_new
        # log a warning as the XC visiblilties are copied
        logger.warning('Autocorr visibilities present in StEFCal solver. Solver running on copy of crosscorr visibilities.')

    # Each row of this array contains the indices of baselines with the same antA
    baselines_per_antA = np.array([(antA == m).nonzero()[0] for m in range(num_ants)])
    # Each row of this array contains the corresponding antB indices with the same antA
    antB_per_antA = antB[baselines_per_antA]
    weighted_vis = weights * vis
    weighted_vis = weighted_vis[..., baselines_per_antA]
    # Initial estimate of gain vector
    gain_shape = tuple(list(vis.shape[:-1]) + [num_ants])
    g_curr = np.ones(gain_shape, dtype=np.complex) if init_gain is None else init_gain
    logger.debug("StefCal solving for %s gains from vis with shape %s" %
                 ('x'.join(str(gs) for gs in gain_shape), vis.shape))
    for n in range(num_iters):
        # Basis vector (collection) represents gain_B* times model (assumed 1)
        g_basis = g_curr[..., antB_per_antA]
        # Do scalar least-squares fit of basis vector to vis vector for whole collection in parallel
        g_new = (g_basis * weighted_vis).sum(axis=-1) / (g_basis.conj() * g_basis).sum(axis=-1)
        # ----------------
        # THIS BIT HACKED BY LAURA
        # Normalise g_new to match g_curr so that taking their average and
        # difference make sense (without copy the elements of g_new are mangled up)
   #     g_new /= (g_new[..., ref_ant][..., np.newaxis].copy() if ref_ant >= 0 else
   #                  g_new[..., 0][..., np.newaxis].copy())
        if ref_ant >= 0:
            g_new = g_new*abs(g_new[..., ref_ant][..., np.newaxis])/g_new[..., ref_ant][..., np.newaxis]
        else:
            g_new = g_new*abs(g_new[..., 0][..., np.newaxis])/g_new[..., 0][..., np.newaxis]
        # ----------------
        logger.debug("Iteration %d: mean absolute gain change = %f" %
                     (n + 1, 0.5 * np.abs(g_new - g_curr).mean()))
        # Avoid getting stuck during iteration
        g_curr = 0.5 * (g_new + g_curr)
    if ref_ant < 0:
        avg_amp = np.mean(np.abs(g_curr), axis=-1)
        middle_angle = np.arctan2(np.median(g_curr.imag, axis=-1),
                                  np.median(g_curr.real, axis=-1))
        g_curr /= (avg_amp * np.exp(1j * middle_angle))[..., np.newaxis]
    return g_curr

#--------------------------------------------------------------------------------------------------
#--- Calibration helper functions
#--------------------------------------------------------------------------------------------------

def g_from_K(chans,K):
    g_array = np.ones(K.shape+(len(chans),), dtype=np.complex)
    for i,c in enumerate(chans):
        g_array[:,:,i] = np.exp(1.0j*2.*np.pi*K*c)
    return g_array

def nanAve(x,axis=0):
    return np.nansum(x,axis=axis)/np.sum(~np.isnan(x),axis=axis)

def ants_from_xcbl(bl):
    """
    Returns the number of antennas calculated from the number of cross-correlation baselines
    """
    return int((1+np.sqrt(1+8*bl))/2)

def ants_from_allbl(bl):
    """
    Returns the number of antennas calculated from the number of cross-correlation and auto-correlation baselines
    """
    return int((np.sqrt(1+8*bl)-1)/2)

def ants_from_bllist(bllist):
    return len(set([item for sublist in bllist for item in sublist]))

def xcbl_from_ants(a):
    """
    Returns the number of cross-correlation baselines calculated from the number of antennas
    """
    return a*(a-1)/2

def g_fit(data,corrprod_lookup,g0=None,refant=0,**kwargs):
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
    num_sol = data.shape[0]
    num_ants = ants_from_bllist(corrprod_lookup)

    # ------------
    # stefcal needs the visibilities as a list of [vis,vis.conjugate]
    vis_and_conj = np.concatenate((data, data.conj()),axis=-1)
    return stefcal(vis_and_conj, num_ants, corrprod_lookup, weights=1.0, ref_ant=refant, init_gain=g0, **kwargs)

def bp_fit(data,corrprod_lookup,bp0=None,refant=0,normalise=True,**kwargs):
    """
    Fit bandpass to visibility data.
    The bandpass phase is centred on zero.

    Parameters
    ----------
    data : array of complex, shape(num_chans, num_pols, baselines)
    bp0 : array of complex, shape(num_chans, num_pols, num_ants) or None
    corrprod_lookup : antenna mappings, for first then second antennas in bl pair
    refant : reference antenna
    normalise : bool, True to normalise the bandpass amplitude

    Returns
    -------
    bpass : Bandpass, shape(num_chans, num_pols, num_ants)
    """

    n_ants = ants_from_bllist(corrprod_lookup)
    n_chans = data.shape[0]

    # -----------------------------------------------------
    # solve for the bandpass over the channel range

    # stefcal needs the visibilities as a list of [vis,vis.conjugate]
    vis_and_conj = np.concatenate((data, data.conj()),axis=-1)
    bp = stefcal(vis_and_conj, n_ants, corrprod_lookup, weights=1.0, num_iters=100, init_gain=bp0, **kwargs)
    # centre the phase on zero
    centre_rotation = np.exp(-1.0j*np.nanmedian(np.angle(bp), axis=0))
    rotated_bp = bp*centre_rotation
    # normalise bandpasses by dividing through by the average
    if normalise:
        rotated_bp /= (np.nansum(np.abs(rotated_bp), axis=0)/n_chans)
    return rotated_bp

def k_fit(data,corrprod_lookup,chans=None,refant=0,chan_sample=1,**kwargs):
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
    # This solver was written to deal with extreme gains, which can wrap many many times across the
    # solution window, to the extent that solving for a per-channel per-antenna bandpass, then fitting a
    # line to the per-antenna bandpass breaks down. It breaks down because the phase unwrapping is unreliable
    # when the phases (a) wrap so extremely, and (b) are also noisy.
    # This situation was adressed by first doing a coarse delay fit through an FFT, for each antenna to the
    # reference antenna. Then the course delay is applied, and a secondary linear phase fit is performed with
    # the corrected data. With the coarse delays removed there should be little, or minimal wrapping, and the
    # unwrapping prior to the linear fit usually works.
    # BUT:
    # * Things fall over in the FFT if there are NaNs in the data - the NaNs propagate and everything becomes NaNs
    #   There is a temporary fix for this below, but if there are chunks of NaNs this won't work (see below)
    # * The coarse delays are calculated relative to the reference antenna. If we happen to choose a reference
    #   antenna with a small delay relative to most other antennas then coarse fit delays are actually relatively
    #   small and we don't have the desired effect of removing the wrapping.
    #   ~~~~~if some of the delays are extreme, then we need to choose a reference antenna that has large delay
    #   relative to other antennas~~~~~
    #   Perhaps this could be done by looking through all of the FFT results, grouped by antenna, to find the
    #   highest set of delays?
    # * In the future world where not every delay fit needs to deal with extreme delay values, maybe we could 
    #   separate out the bandpass-linear-phase-fit component of this algorithm, and use just that for delay
    #   solutions we know won't be too extreme. Then this extreme delay solver could solve for the coarse delay 
    #   then apply it and call the fine delay bandpass-linear-phase-fit solver.

    # -----------------------------------------------------
    # if channel sampling is specified, thin down the data and channel list
    if chan_sample != 1:
        data = data[::chan_sample,...]
        if np.any(chans): chans = chans[::chan_sample]

    # set up parameters
    num_ants = ants_from_bllist(corrprod_lookup)
    num_pol = data.shape[-2] if len(data.shape)>2 else 1
    if chans == None:
        chans = range(data.shape[0])
    nchans = len(chans)
    chan_increment = chans[1]-chans[0]

    # -----------------------------------------------------
    # initialise values for solver
    kdelay = []
    if not(np.any(chans)): chans = np.arange(data.shape[0])

    # NOTE: I know that iterating over polarisation is horrible, but I ran out of time to fix this up
    # I initially just assumed we would always have two polarisations, but shouldn't make that assumption
    # especially for use of calprocs offline on an h5 file
    for p in range(num_pol):
        pol_data = data[:,p,:] if len(data.shape)>2 else data

        # if there is a nan in a channel on any baseline, remove that channel from all baselines for the fit
        bad_chan_mask = np.logical_or.reduce(np.isnan(pol_data), axis=1)
        good_pol_data = pol_data[~bad_chan_mask]
        good_chans = chans[~bad_chan_mask]
        # NOTE: This removal of channels is not a proper fix for if there are a lot of NaNs. 
        #   In that case there will be large gaps in the data leaving to what the FFT below sees as phase jumps
        #   This removal of NaN channels is just a temporary fix for the case of a few bad channels, so that the
        #   NaN's don't propagate and lead to NaNs throughout and NaN delay values

        # -----------------------------------------------------
        # FT to find visibility space delays
        # NOTE: This is a bit inefficient at the moment as the FFT for al baselines is calculated but only
        #  the baselines to the reference antenna are used for the coarse delay
        ft_vis = np.fft.fft(good_pol_data, axis=0)
        # get index of FT maximum
        k_arg = np.argmax(np.abs(ft_vis),axis=0)
        if nchans%2 == 0:
            k_arg[k_arg > ((nchans/2) - 1)] -= nchans
        else:
            k_arg[k_arg > ((nchans-1)/2)] -= nchans

        # calculate vis space K from FT sample frequencies
        vis_k = 1.*k_arg/(chan_increment*nchans)

        # now determine per-antenna K values
        coarse_k = np.zeros(num_ants)
        for ai in range(num_ants):
            k = vis_k[...,(corrprod_lookup[:,0]==ai)&(corrprod_lookup[:,1]==refant)]
            if k.size > 0: coarse_k[...,ai] = np.squeeze(k)
        for ai in range(num_ants):
            k = vis_k[...,(corrprod_lookup[:,1]==ai)&(corrprod_lookup[:,0]==refant)]
            if k.size > 0: coarse_k[...,ai] = np.squeeze(-1.0*k)

        # apply coarse K values to the data and solve for bandpass
        v_corrected = np.zeros_like(good_pol_data)
        for vi in range(v_corrected.shape[-1]):
            for ci, c in enumerate(good_chans):
                v_corrected[ci,vi] = good_pol_data[ci,vi] * np.exp(-1.0j*2.*np.pi*c*coarse_k[corrprod_lookup[vi,0]]) * np.exp(1.0j*2.*np.pi*c*coarse_k[corrprod_lookup[vi,1]])
        # stefcal needs the visibilities as a list of [vis,vis.conjugate]
        vis_and_conj = np.concatenate((v_corrected, v_corrected.conj()),axis=-1)
        bpass = stefcal(vis_and_conj, num_ants, corrprod_lookup, weights=1.0, num_iters=100, ref_ant=refant, init_gain=None, **kwargs)

        # find slope of the residual bandpass
        delta_k = np.empty_like(coarse_k)
        for i,bp in enumerate(bpass.T):
            # np.unwrap falls over in the case of bad RFI - robustify this later
            bp_phase = np.unwrap(np.angle(bp),discont=1.9*np.pi)
            A = np.array([good_chans, np.ones(len(good_chans))])
            delta_k[i] = np.linalg.lstsq(A.T,bp_phase)[0][0]/(2.*np.pi)

        kdelay.append(np.squeeze(coarse_k + delta_k))

    return np.squeeze(kdelay)

def kcross_fit(data,flags,chans=None,chan_ave=1):
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
    ave_crosshand = np.average((data[:,0,:]*~flags[:,0,:]+np.conjugate(data[:,1,:]*~flags[:,1,:]))/2.,axis=-1)
    ave_crosshand = np.add.reduceat(ave_crosshand,range(0,len(ave_crosshand),chan_ave))/chan_ave
    ave_chans = np.add.reduceat(chans,range(0,len(chans),chan_ave))/chan_ave

    nchans = len(ave_chans)
    chan_increment = ave_chans[1]-ave_chans[0]

    # FT the visibilities to get course estimate of kcross
    #   with noisy data, this is more robust than unwrapping the phase
    ft_vis = np.fft.fft(ave_crosshand)

    # get index of FT maximum
    k_arg = np.argmax(ft_vis)
    if nchans%2 == 0:
        if k_arg > ((nchans/2) - 1): k_arg = k_arg - nchans
    else:
	    if k_arg > ((nchans-1)/2): k_arg = k_arg - nchans

    # calculate kcross from FT sample frequencies
    coarse_kcross = k_arg/(chan_increment*nchans)

    # correst data with course kcross to solve for residual delta kcross
    corrected_crosshand = ave_crosshand*np.exp(-1.0j*2.*np.pi*coarse_kcross*ave_chans)
    # solve for residual kcross through linear phase fit
    crossphase = np.angle(corrected_crosshand)
    # remove any offset from zero
    crossphase -= np.median(crossphase)
    # linear fit
    A = np.array([ave_chans, np.ones(len(ave_chans))])
    delta_kcross = np.linalg.lstsq(A.T,crossphase)[0][0]/(2.*np.pi)

    # total kcross
    return coarse_kcross + delta_kcross

def wavg(data,flags,weights,times=False,axis=0):
    """
    Perform weighted average of data, applying flags,
    over specified axis

    Parameters
    ----------
    data    : array of complex
    flags   : array of uint8 or boolean
    weights : array of floats
    times   : array of times. If times are given, average times are returned
    axis    : axis to average over

    Returns
    -------
    vis, times : weighted average of data and, optionally, times
    """

    if flags.dtype is np.uint8:
        fg = flags.view(np.bool)
    elif np.issubdtype(flags.dtype,bool):
        fg = flags
    else:
        raise TypeError('Incompatible flag type!')

    vis = np.nansum(data*weights*(~fg),axis=axis)/np.nansum(weights*(~fg),axis=axis)
    return vis if times is False else (vis, np.average(times,axis=axis))

def wavg_full(data,flags,weights,axis=0,threshold=0.3):
    """
    Perform weighted average of data, flags and weights,
    applying flags, over specified axis

    Parameters
    ----------
    data       : array of complex
    flags      : array of uint8 or boolean
    weights    : array of floats
    axis    : axis to average over

    Returns
    -------
    av_data    : weighted average of data
    av_flags   : weighted average of flags
    av_weights : weighted average of weights
    av_sig     : sigma of weighted data
    """

    av_sig = np.nanstd(data*weights*(~flags))
    av_data = np.nansum(data*weights*(~flags),axis=axis)/np.nansum(weights*(~flags),axis=axis)
    av_flags = np.nansum(flags,axis=axis) > flags.shape[0]*threshold

    # fake weights for now
    av_weights = np.ones_like(av_data,dtype=np.float32)

    return av_data, av_flags, av_weights, av_sig

def wavg_full_t(data,flags,weights,solint,axis=0,times=None):
    """
    Perform weighted average of data, flags and weights,
    applying flags, over specified axis, for specified
    solution interval increments

    Parameters
    ----------
    data       : array of complex
    flags      : array of boolean
    weights    : array of floats
    solint     : index interval over which to average, integer
    axis       : axis to average over
    times      : optional array of times to average, array of floats

    Returns
    -------
    av_data    : weighted average of data
    av_flags   : weighted average of flags
    av_weights : weighted average of weights
    av_sig     : sigma of averaged data
    av_times   : optional average of times
    """
    # ensure solint is an intager
    solint = np.int(solint)
    inc_array = np.arange(0,data.shape[axis],solint)
    wavg = np.array([wavg_full(data[ti:ti+solint],flags[ti:ti+solint],weights[ti:ti+solint],axis=0) for ti in inc_array])

    av_data, av_flags, av_weights, av_sig = [], [], [], []
    for ti in inc_array:
        w_out = wavg_full(data[ti:ti+solint],flags[ti:ti+solint],weights[ti:ti+solint],axis=0)
        av_data.append(w_out[0])
        av_flags.append(np.bool_(w_out[1]))
        av_weights.append(w_out[2])
        av_sig.append(w_out[3])
    av_data = np.array(av_data)
    av_flags = np.array(av_flags)
    av_weights = np.array(av_weights)
    av_sig = np.array(av_sig)

    if np.any(times):
        av_times = np.array([np.average(times[ti:ti+solint],axis=0) for ti in inc_array])
        return av_data, av_flags, av_weights, av_sig, av_times
    else:
        return av_data, av_flags, av_sig, av_weights

def solint_from_nominal(solint,dump_period,num_times):
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

    # case of solution interval that is shorter than dump time
    if dump_period > solint:
        return dump_period, 1.0
    # case of solution interval that is longer than scan
    if dump_period*num_times < solint:
        return solint, np.round(solint/dump_period)

    # number of dumps per nominal solution interval
    dumps_per_solint = np.round(solint/dump_period)

    # range for searching: nominal solint +-20%
    delta_dumps_per_solint = int(dumps_per_solint*0.2)
    solint_check_range = range(-delta_dumps_per_solint,delta_dumps_per_solint+1)

    smallest_inc = np.empty(len(solint_check_range))
    for i,s in enumerate(solint_check_range):
        # solution intervals across the total time range
        intervals = num_times/(dumps_per_solint+s)
        # the size of the final fractional solution interval
        if int(intervals) == 0:
            smallest_inc[i] = 0
        else:
            smallest_inc[i] = intervals % int(intervals)

    # choose a solint to minimise the final fractional solution interval
    solint_index = np.where(smallest_inc==max(smallest_inc))[0][0]
    logger.debug('solint index: {0}'.format(solint_index,))
    nsolint = solint+solint_check_range[solint_index]
    # calculate new dumps per solints
    dumps_per_solint = np.round(nsolint/dump_period)

    return nsolint, dumps_per_solint

#--------------------------------------------------------------------------------------------------
#--- Interpolation
#--------------------------------------------------------------------------------------------------

import scipy.interpolate

class interp_extrap_1d(scipy.interpolate.interp1d):
    """
    Subclasses the scipy.interpolate interp1d to be able to extrapolate
    Extrapolated points are just == to edge interpolated values

    scipy.interpolate interpolates a 1-D function.

    `x` and `y` are arrays of values used to approximate some function f:
    ``y = f(x)``.  This class returns a function whose call method uses
    interpolation to find the value of new points.

    Parameters
    ----------
    x : (N,) array_like
        A 1-D array of real values.
    y : (...,N,...) array_like
        A N-D array of real values. The length of `y` along the interpolation
        axis must be equal to the length of `x`.
    kind : str or int, optional
        Specifies the kind of interpolation as a string
        ('linear', 'nearest', 'zero', 'slinear', 'quadratic, 'cubic'
        where 'slinear', 'quadratic' and 'cubic' refer to a spline
        interpolation of first, second or third order) or as an integer
        specifying the order of the spline interpolator to use.
        Default is 'linear'.
    axis : int, optional
        Specifies the axis of `y` along which to interpolate.
        Interpolation defaults to the last axis of `y`.
    copy : bool, optional
        If True, the class makes internal copies of x and y.
        If False, references to `x` and `y` are used. The default is to copy.
    assume_sorted : bool, optional
        If False, values of `x` can be in any order and they are sorted first.
        If True, `x` has to be an array of monotonically increasing values.

    Methods
    -------
    __call__

    """

    def __call__(cls, x, **kwds):
        x_new = copy.copy(x)
        # for input x values that are less than the lowest value of the interpolation function x-range, set them to the lowest value
        x_new[x_new < cls.x[0]] = cls.x[0]
        # likewise, for input x values that are greater than the greatest value of the interpolation function x-range, set them to the highest value
        x_new[x_new > cls.x[-1]] = cls.x[-1]
        # this is a sneaky way to force extrapolated y values to be the same as the y exterior values of the interpolation range
        # by setting the x-values outside to the interpolation range to the same values as the x exterior values of the interpolation range
        return scipy.interpolate.interp1d.__call__(cls, x_new, **kwds)

#--------------------------------------------------------------------------------------------------
#--- Baseline ordering
#--------------------------------------------------------------------------------------------------

def get_ant_bls(pol_bls_ordering):
    """
    Given baseline list with polarisation information, return pure antenna list
    e.g. from [['ant0h','ant0h'],['ant0v','ant0v'],['ant0h','ant0v'],['ant0v','ant0h'],['ant1h','ant1h']...]
     to [['ant0,ant0'],['ant1','ant1']...]
    """

    # get antenna only names from pol-included bls orderding
    ant_bls = np.array([[a1[:-1],a2[:-1]] for a1,a2 in pol_bls_ordering])
    ant_dtype = ant_bls[0,0].dtype
    # get list without repeats (ie no repeats for pol)
    #     start with array of empty strings, shape (num_baselines x 2)
    bls_ordering = np.empty([len(ant_bls)/4,2],dtype=ant_dtype)

    # iterate through baseline list and only include non-repeats
    #   I know this is horribly non-pythonic. Fix later.
    bls_ordering[0] = ant_bls[0]
    bl = 0
    for c in ant_bls:
        if not np.all(bls_ordering[bl] == c):
            bl += 1
            bls_ordering[bl] = c

    return bls_ordering

def get_pol_bls(bls_ordering,pol):
    """
    Given baseline ordering and polarisation ordering, return full baseline-pol ordering array

    Inputs:
    -------
    bls_ordering : list of correlation products without polarisation information, string shape(nbl,2)
    pol : list of polarisation pairs, string shape (npol_pair, 2)

    Returns:
    --------
    pol_bls_ordering : correlation products without polarisation information, numpy array shape(nbl*4, 2)
    """
    pol_ant_dtype = np.array(bls_ordering[0][0]+'h').dtype
    nbl = len(bls_ordering)
    pol_bls_ordering = np.empty([nbl*len(pol),2],dtype=pol_ant_dtype)
    for i,p in enumerate(pol):
        for b,bls in enumerate(bls_ordering):
            pol_bls_ordering[nbl*i+b] = bls[0]+p[0], bls[1]+p[1]
    return pol_bls_ordering

def get_reordering(antlist,bls_ordering,output_order_bls=None,output_order_pol=None):
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
    ordering : ordering array necessary to change given bls_ordering into desired ordering, numpy array shape(nbl*4, 2)
    bls_wanted : ordering of baselines, without polarisation, list shape(nbl, 2)
    pol_order : ordering of polarisations, list shape (4, 2)

    """
    # convert antlist to list, if it is a csv string
    if isinstance(antlist,str): antlist = antlist.split(',')
    # convert bls_ordering to a list, if it is not a list (e.g. np.ndarray)
    if not isinstance(bls_ordering,list): bls_ordering = bls_ordering.tolist()

    # get current antenna list without polarisation
    bls_ordering_nopol = [[b[0][0:4],b[1][0:4]] for b in bls_ordering]
    # find unique elements
    unique_bls = []
    for b in bls_ordering_nopol:
        if b not in unique_bls: unique_bls.append(b)

    # determine polarisation ordering
    if output_order_pol is not None:
        pol_order = output_order_pol
    else:
        # get polarisation products from current antenna list
        bls_ordering_polonly = [[b[0][4],b[1][4]] for b in bls_ordering]
        # find unique pol combinations
        unique_pol = []
        for pbl in bls_ordering_polonly:
            if pbl not in unique_pol: unique_pol.append(pbl)
        # put parallel polarisations first
        pol_order = []
        if len(bls_ordering_polonly) > 2:
            for pp in unique_pol:
                if pp[0]==pp[1]:
                    pol_order.insert(0,pp)
                else:
                    pol_order.append(pp)
    npol = len(pol_order)

    if output_order_bls == None:
        # default output order is XC then AC
        bls_wanted = [b for b in unique_bls if b[0]!=b[1]]

        # number of baselines including autocorrelations
        nants = len(antlist)
        nbl_ac = nants*(nants+1)/2

        # add AC to bls list, if necessary
        # assume that AC are either included or not (no partial states)
        if len(bls_ordering) == (nbl_ac * npol):
            # data include XC and AC
            bls_wanted.extend([b for b in unique_bls if b[0]==b[1]])

        bls_pol_wanted = get_pol_bls(bls_wanted,pol_order)
    else:
        bls_pol_wanted = get_pol_bls(output_order_bls,pol_order)
    # note: bls_pol_wanted must be an np array for list equivalence below

    # find ordering necessary to change given bls_ordering into desired ordering
    # note: ordering must be a numpy array to be used for indexing later
    ordering = np.array([np.all(bls_ordering==bls,axis=1).nonzero()[0][0] for bls in bls_pol_wanted])

    # how to use this:
    #print bls_ordering[ordering]
    #print bls_ordering[ordering].reshape([4,nbl,2])
    return ordering, bls_wanted, pol_order

def get_reordering_nopol(antlist,bls_ordering,output_order_bls=None):
    """
    Determine reordering necessary to change given bls_ordering into desired ordering

    Inputs:
    -------
    antlist : list of antennas, string shape(nants), or string of csv antenna names
    bls_ordering : list of correlation products, string shape(nbl,2)
    output_order_bls : desired baseline output order, string shape(nbl,2), optional

    Returns:
    --------
    ordering : ordering array necessary to change given bls_ordering into desired ordering, numpy array, shape(nbl*4, 2)
    bls_wanted : ordering of baselines, without polarisation, numpy array, shape(nbl, 2)

    """
    # convert antlist to list, if it is a csv string
    if isinstance(antlist,str): antlist = antlist.split(',')
    nants = len(antlist)
    nbl = nants*(nants+1)/2
    # convert bls_ordering to a list, if it is not a list (e.g. np.ndarray)
    if not isinstance(bls_ordering,list): bls_ordering = bls_ordering.tolist()

    # find unique elements
    unique_bls = []
    for b in bls_ordering:
        if not b in unique_bls: unique_bls.append(b)

    # defermine output ordering:
    if output_order_bls is None:
        # default output order is XC then AC
        bls_wanted = [b for b in unique_bls if b[0]!=b[1]]
        # add AC to bls list
        bls_wanted.extend([b for b in unique_bls if b[0]==b[1]])
    else:
        # convert to list in case it is not a list
        bls_wanted = output_order_bls
    # note: bls_wanted must be an np array for list equivalence below
    bls_wanted = np.array(bls_wanted)

    # find ordering necessary to change given bls_ordering into desired ordering
    # note: ordering must be a numpy array to be used for indexing later
    ordering = np.array([np.all(bls_ordering==bls,axis=1).nonzero()[0][0] for bls in bls_wanted])
    # how to use this:
    #print bls_ordering[ordering]
    #print bls_ordering[ordering].reshape([4,nbl,2])
    return ordering, bls_wanted

def get_bls_lookup(antlist,bls_ordering):
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

    # make polarisation and corr_prod lookup tables (assume this doesn't change over the course of an observaton)
    antlist_index = dict([(antlist[i], i) for i in range(len(antlist))])
    return np.array([[antlist_index[a1[0:4]],antlist_index[a2[0:4]]] for a1,a2 in bls_ordering])

#--------------------------------------------------------------------------------------------------
#--- Simulation
#--------------------------------------------------------------------------------------------------

def fake_vis(nants=7,gains=None,noise=None): #   (self,algorithm,delta=1e-3,noise=False,gains=None):
    """Create fake point source visibilities, corrupted by given or random gains"""
    import numpy as np

    # create antenna lists
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
    if noise != None:
        vis_noise = (np.random.random(vis.shape)-0.5)*noise
        vis = vis+vis_noise

    # return useful info
    bl_pair_list = np.column_stack([list1, list2])
    return vis, bl_pair_list, gains

#--------------------------------------------------------------------------------------------------
#--- CLASS :  CalSolution
#--------------------------------------------------------------------------------------------------

class CalSolution(object):

    def __init__(self, soltype, solvalues, soltimes):
        self.soltype = soltype
        self.values = solvalues
        self.times = soltimes
        self.ts_solname =  'cal_product_{0}'.format(soltype,)

#--------------------------------------------------------------------------------------------------
#--- General helper functions
#--------------------------------------------------------------------------------------------------

def arcsec_to_rad(angle):
    """
    Convert angle in arcseconds to angle in radians
    """
    return np.deg2rad(angle/60./60.)
