"""
Calibration procedures for MeerKAT calibration pipeline
=======================================================

Solvers and averagers for use in the MeerKAT calibration pipeline.
"""

import numpy as np
import logging
import copy

logger = logging.getLogger(__name__)

#--------------------------------------------------------------------------------------------------
#--- Solvers
#--------------------------------------------------------------------------------------------------

def get_bl_ant_pairs(corrprod_lookup):
    """
    Get antenna lists in solver format, from corr_prod lookup

    Inputs:
    -------
    corrprod_lookup : lookup table of antenna indices for each baseline, list shape(nant,2)

    Returns:
    --------
    antlist 1, antlist 2 : lists of antennas matching the correlation_product lookup table,
        appended with their conjugates (format required by stefcal)
    """

    # NOTE: no longer need hh and vv masks as we re-ordered the data to be ntime x nchan x nbl x npol

     # get antenna number lists for stefcal - need vis then vis.conj (assume constant over an observation)
    # assume same for hh and vv
    print corrprod_lookup
    antlist1 = np.concatenate((corrprod_lookup[:,0], corrprod_lookup[:,1]))
    antlist2 = np.concatenate((corrprod_lookup[:,1], corrprod_lookup[:,0]))

    return antlist1, antlist2

def stefcal(vis, num_ants, corrprod_lookup, weights=1.0, num_iters=100, ref_ant=0, init_gain=None,
    model=None, algorithm='adi', conv_thresh=0.0001, verbose=False):
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
    corrprod_lookup : numpy array of int, shape (2,N)
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
        return adi_stefcal(vis, num_ants, bl_ant_pairs, weights, num_iters, ref_ant,
                init_gain, model, conv_thresh, verbose)
    elif algorithm == 'schwardt':
        return schwardt_stefcal(vis, num_ants, bl_ant_pairs, weights, num_iters, ref_ant,
                init_gain, verbose)
    else:
        raise ValueError(' '+algorithm+' is not a valid stefcal implimentation.')

def adi_stefcal_nonparallel(vis, num_ants, bl_ant_pairs, weights=1.0, num_iters=100, ref_ant=0, init_gain=None, model=None, conv_thresh=0.0001, verbose=False):
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
    M = 1.0 - np.eye(num_ants, dtype=np.complex) if model is None else model

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

def adi_stefcal(vis, num_ants, bl_ant_pairs, weights=1.0, num_iters=100, ref_ant=0, init_gain=None, model=None, conv_thresh=0.0001, verbose=False):
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
    M = 1.0 - np.eye(num_ants, dtype=np.complex) if model is None else model

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
            norm_val = abs(g_curr[...,ref_ant][..., np.newaxis])/g_curr[...,ref_ant][..., np.newaxis]
            g_curr *= norm_val

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

def adi_stefcal_acorr(vis, num_ants, bl_ant_pairs, weights=1.0, num_iters=100, ref_ant=0, init_gain=None, model=None, conv_thresh=0.0001, verbose=False):
    """Solve for antenna gains using ADI StefCal, including fake autocorr data. Non parallel version of the algorithm.
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
    M = 1.0 - np.eye(num_ants, dtype=np.complex) if model is None else model

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

def schwardt_stefcal(vis, num_ants, bl_ant_pairs, weights=1.0, num_iters=10, ref_ant=-1, init_gain=None, verbose=False):
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
    vis : array of complex, shape (M, ..., N)
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
    vis = vis[...,xcorr]
    antA_new = antA[xcorr]
    antB = antB[xcorr]
    antA = antA_new

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
        g_array[:,:,i] = np.exp(1.0j*K*c)
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

def g_fit(data,corrprod_lookup,g0=None,refant=0,algorithm='adi'):
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
    return stefcal(vis_and_conj, num_ants, corrprod_lookup, weights=1.0, num_iters=100, ref_ant=refant, init_gain=g0, algorithm=algorithm)

def bp_fit(data,corrprod_lookup,bp0=None,refant=0,algorithm='adi'):
    """
    Fit bandpass to visibility data.

    Parameters
    ----------
    data : array of complex, shape(num_chans, baselines)
    bp0 : array of complex, shape(num_chans, num_ants) or None
    corrprod_lookup : antenna mappings, for first then second antennas in bl pair
    refant : reference antenna

    Returns
    -------
    bpass : Bandpass, shape(num_chans, num_ants)
    """

    num_ants = ants_from_bllist(corrprod_lookup)

    # -----------------------------------------------------
    # initialise values for solver
    bpsoln = np.empty([data.shape[0],num_ants],dtype=np.complex) # Make empty array to fill bandpass into

    # -----------------------------------------------------
    # solve for the bandpass over the channel range

    # stefcal needs the visibilities as a list of [vis,vis.conjugate]
    vis_and_conj = np.concatenate((data, data.conj()),axis=-1)
    return stefcal(vis_and_conj, num_ants, corrprod_lookup, weights=1.0, num_iters=100, ref_ant=refant, init_gain=bp0, algorithm=algorithm)

def k_fit(data,corrprod_lookup,chans=None,k0=None,bp0=None,refant=0,chan_sample=None,algorithm='adi'):
    """
    Fit bandpass to visibility data.

    Parameters
    ----------
    data : array of complex, shape(num_chans, num_pols, baseline)
    corrprod_lookup : antenna mappings, for first then second antennas in bl pair
    k0 : array of complex, shape(num_chans, num_pols, num_ants) or None
    bp0 : array of complex, shape(num_chans, num_pols, num_ants) or None
    refant : reference antenna

    Returns
    -------
    ksoln : Bandpass, shape(num_chans, num_ants)
    """

    num_ants = ants_from_bllist(corrprod_lookup)

    # -----------------------------------------------------
    # if channel sampling is specified, thin down the data and channel list
    data = data[::chan_sample,...]
    if np.any(chans): chans = chans[::chan_sample]

    # -----------------------------------------------------
    # initialise values for solver
    kdelay = np.empty([2,num_ants],dtype=np.complex) # Make empty array to fill delay into
    if not(np.any(chans)): chans = np.arange(data.shape[0])

    # -----------------------------------------------------
    # solve for the bandpass over the channel range

    # stefcal needs the visibilities as a list of [vis,vis.conjugate]
    vis_and_conj = np.concatenate((data, data.conj()),axis=-1)
    bpass = stefcal(vis_and_conj, num_ants, corrprod_lookup, weights=1.0, num_iters=100, ref_ant=refant, init_gain=None, algorithm=algorithm)

    # -----------------------------------------------------
    # find bandpass phase slopes (delays)
    for i,bp in enumerate(bpass.T):
        # polarisation
        for p in range(2):
            # unwrap angles before fitting for slope
            bp_phase = np.unwrap(np.angle(bp[p]))
            A = np.array([ chans, np.ones(len(chans))])
            kdelay[p,i] = np.linalg.lstsq(A.T,bp_phase)[0][0]

    return kdelay

def wavg(data,flags,weights,times=False,axis=0):
    """
    Perform weighted average of data, applying flags,
    over specified axis

    Parameters
    ----------
    data    : array of complex
    flags   :u array of int8 or boolean
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
    threshold = 0
    av_flags = np.nansum(flags,axis=axis) > flags.shape[0]*threshold

    # fake weights for now
    av_weights = np.ones_like(av_data,dtype=np.float)

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
    solint     : index interval over which to average
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
    """

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
        smallest_inc[i] = intervals % int(intervals)

    # choose a solint to minimise the final fractional solution interval
    nsolint = solint+solint_check_range[np.where(smallest_inc==max(smallest_inc))[0]]
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
    """

    def __call__(cls, x, **kwds):
        x_new = copy.copy(x)

        x_new[x_new < cls.x[0]] = cls.x[0]
        x_new[x_new > cls.x[-1]] = cls.x[-1]

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
    pol_bls_ordering = np.empty([nbl*4,2],dtype=pol_ant_dtype)
    for i,p in enumerate(pol):
        for b,bls in enumerate(bls_ordering):
            pol_bls_ordering[nbl*i+b] = bls[0]+p[0], bls[1]+p[1]
    return pol_bls_ordering

def get_reordering(antlist,bls_ordering):
    """
    Determine reordering necessary to change given bls_ordering into desired ordering

    Inputs:
    -------
    antlist : list of antennas, string shape(nants), or string of csv antenna names
    bls_ordering : list of correlation products, string shape(nbl,2)

    Returns:
    --------
    ordering : ordering array necessary to change given bls_ordering into desired ordering, numpy array shape(nbl*4, 2)
    bls_wanted : ordering of baselines, without polarisation, list shape(nbl, 2)
    pol_order : ordering of polarisations, list shape (4, 2)

    """
    # convert antlist to list, if it is a csv string
    if isinstance(antlist,str): antlist = antlist.split(',')
    nants = len(antlist)
    nbl = nants*(nants+1)/2
    # convert bls_ordering to a list, if it is not a list (e.g. np.ndarray)
    if not isinstance(bls_ordering,list): bls_ordering = bls_ordering.tolist()

    # get current antenna list without polarisation
    bls_ordering_nopol = [[b[0][0:4],b[1][0:4]] for b in bls_ordering]
    # find unique elements
    unique_bls = []
    for b in bls_ordering_nopol:
        if not b in unique_bls: unique_bls.append(b)

    # re-order into XC then AC
    bls_wanted = [b for b in unique_bls if b[0]!=b[1]]
    bls_wanted.extend([b for b in unique_bls if b[0]==b[1]])

    #   add polarisation indices
    pol_order = [['h','h'],['v','v'],['h','v'],['v','h']]
    bls_pol_wanted = get_pol_bls(bls_wanted,pol_order)

    # find ordering necessary to change given bls_ordering into desired ordering
    # note: ordering must be a numpy array to be used for indexing later
    ordering = np.array([np.all(bls_ordering==bls,axis=1).nonzero()[0][0] for bls in bls_pol_wanted])
    # how to use this:
    #print bls_ordering[ordering]
    #print bls_ordering[ordering].reshape([4,nbl,2])
    return ordering, bls_wanted, pol_order

def get_bls_lookup(antlist,bls_ordering):
    """
    Get correlation product antenna mapping

    Inputs:
    -------
    antlist : csv list of antennas, string
    bls_ordering : list of correlation products, string shape(nbl,2)

    Returns:
    --------
    corrprod_lookup : lookup table of antenna indices for each baseline, shape(nbl,2)
    """

    antlist = antlist.split(',')
    # make polarisation and corr_prod lookup tables (assume this doesn't change over the course of an observaton)
    antlist_index = dict([(antlist[i], i) for i in range(len(antlist))])
    return np.array([[antlist_index[a1[0:4]],antlist_index[a2[0:4]]] for a1,a2 in bls_ordering])

#--------------------------------------------------------------------------------------------------
#--- CLASS :  CalSolution
#--------------------------------------------------------------------------------------------------

class CalSolution(object):

    def __init__(self, soltype, solvalues, soltimes):
        self.soltype = soltype
        self.values = solvalues
        self.times = soltimes
        self.ts_solname =  'cal_product_{0}'.format(soltype,)
