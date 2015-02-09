"""
Calibration procedures for MeerKAT calibration pipeline
=======================================================

Solvers and averagers for use in the MeerKAT calibration pipeline.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def stefcal(vis, num_ants, antA, antB, weights=1.0, num_iters=100, ref_ant=0, init_gain=None, 
    model=None, algorithm='adi', conv_thresh=0.001, verbose=False):
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
    antA, antB : numpy array of int, shape (N,)
        Antenna indices associated with visibilities
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
    #algorithm = 'schwardt'
    if algorithm == 'adi':
        return adi_stefcal(vis, num_ants, antA, antB, weights, num_iters, ref_ant, 
                init_gain, model, conv_thresh, verbose)
    elif algorithm == 'schwardt':
        return schwardt_stefcal(vis, num_ants, antA, antB, weights, num_iters, ref_ant, 
                init_gain, verbose)
    else:
        raise ValueError(' '+algorithm+' is not a valid stefcal implimentation.')
        
def adi_stefcal(vis, num_ants, antA, antB, weights=1.0, num_iters=100, ref_ant=0, init_gain=None, model=None, conv_thresh=0.001, verbose=False):
    """Solve for antenna gains using ADI StefCal.
    ADI StefCal implimentation from:
    'Fast gain calibration in radio astronomy using alternating direction implicit methods: 
    Analysis and applications', Salvini & Winjholds, 2014

    Parameters
    ----------
    vis         : array of complex, shape (N,)
        Complex cross-correlations between antennas A and B
    num_ants    : int
        Number of antennas
    antA, antB  : numpy array of int, shape (N,)
        Antenna indices associated with visibilities
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
    
def adi_stefcal_acorr(vis, num_ants, antA, antB, weights=1.0, num_iters=100, ref_ant=0, init_gain=None, model=None, conv_thresh=0.001, verbose=False):
    """Solve for antenna gains using ADI StefCal, including fake autocorr data
    ADI StefCal implimentation from:
    'Fast gain calibration in radio astronomy using alternating direction implicit methods: 
    Analysis and applications', Salvini & Winjholds, 2014
    
    Parameters
    ----------
    vis         : array of complex, shape (N,)
        Complex cross-correlations between antennas A and B
    num_ants    : int
        Number of antennas
    antA, antB  : numpy array of int, shape (N,)
        Antenna indices associated with visibilities
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
    
def schwardt_stefcal(vis, num_ants, antA, antB, weights=1.0, num_iters=10, ref_ant=0, init_gain=None, verbose=False):
    """Solve for antenna gains using StefCal.
    (Stolen from Ludwig's antsol solver.py)

    Parameters
    ----------
    vis : array of complex, shape (N,)
        Complex cross-correlations between antennas A and B
    num_ants : int
        Number of antennas
    antA, antB : array of int, shape (N,)
        Antenna indices associated with visibilities
    weights : float or array of float, shape (N,), optional
        Visibility weights (positive real numbers)
    num_iters : int, optional
        Number of iterations
    ref_ant : int, optional
        Reference antenna whose gain will be forced to be 1.0
    init_gain : array of complex, shape(num_ants,) or None, optional
        Initial gain vector (all equal to 1.0 by default)

    Returns
    -------
    gains : array of complex, shape (num_ants,)
        Complex gains, one per antenna

    """
    # Initialise design matrix for solver
    g_prev = np.zeros((len(vis), num_ants), dtype=np.complex)
    rows = np.arange(len(vis))
    weighted_vis = weights * vis
    weights = np.atleast_2d(weights).T
    # Initial estimate of gain vector
    g_curr = np.ones(num_ants, dtype=np.complex) if init_gain is None else init_gain
    for n in range(num_iters):
        # Insert current gain into design matrix as gain B
        # g_prev has shape(num_blx2,num_ants)
        g_prev[rows, antA] = g_curr[antB].conj()
        g_new = np.linalg.lstsq(weights * g_prev, weighted_vis)[0]
        # Force reference gain to be zero phase
        g_new = abs(g_new[ref_ant])*g_new/g_new[ref_ant]
        # Force reference gain to be 1
        #g_new /= g_new[ref_ant]
        if verbose: print "Iteration %d: mean absolute gain change = %f" % (n + 1, 0.5 * np.abs(g_new - g_curr).mean())
        # Avoid getting stuck
        g_curr = 0.5 * (g_new + g_curr)
    return g_curr
    
def g_from_K(chans,K):
    g_array = np.ones(K.shape+(len(chans),), dtype=np.complex)
    for i,c in enumerate(chans):
        g_array[:,:,i] = np.cos(2*np.pi*K*c) + 1.0j*np.sin(2*np.pi*K*c)
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
   
def xcbl_from_ants(a):
    """
    Returns the number of cross-correlation baselines calculated from the number of antennas
    """
    return a*(a-1)/2
   
def g_fit(data,g0,antlist1,antlist2,refant):
    """
    Fit gains to visibility data.
   
    Parameters
    ----------
    data     : array of complex, shape(baselines)
    g0       : array of complex, shape(num_ants) or None
    antlist1 : antenna mapping, for first antenna in bl pair 
    antlist2 : antenna mapping, for second antenna in bl pair 
    refant   : reference antenna

    Returns
    ------- 
    gainsoln : Gain solutions, shape(num_ants)
    """
    num_ants = int(ants_from_allbl(data.shape[0]))
   
    # -----------------------------------------------------
    # initialise values for solver
    gainsoln = np.empty(num_ants,dtype=np.complex) # Make empty array to fill bandpass into

    # -----------------------------------------------------
    # solve for the gains

    # stefcal needs the visibilities as a list of [vis,vis.conjugate]
    vis_and_conj = np.concatenate((data, data.conj())) 
    gainsoln = stefcal(vis_and_conj, num_ants, antlist1, antlist2, weights=1.0, num_iters=100, ref_ant=refant, init_gain=g0)   
      
    return gainsoln
   
def g_fit_per_solint(data,dumps_per_solint,antlist1,antlist2,g0=None,refant=0):
    """
    Fit complex gains to visibility data.
   
    Parameters
    ----------
    data     : visibility data, array of complex, shape(num_sol, num_chans, baseline)
    dumps_per_solint : number of dumps to average for a solution, integer
    g0       : array of complex, shape(num_ants) or None
    antlist1 : antenna mapping, for first antenna in bl pair 
    antlist2 : antenna mapping, for second antenna in bl pair 
    refant   : reference antenna

    Returns
    ------- 
    g_array  : Array of gain solutions, shape(num_sol, num_ants)
    """
    num_sol = data.shape[0]
    num_ants = ants_from_allbl(data.shape[1])

    # empty arrays for solutions
    g_array = np.empty([num_sol,num_ants],dtype=np.complex)
    t_array = np.empty([num_sol],dtype=np.complex)
    # solve for G for each solint
    for i in range(num_sol):
        g_array[i] = g_fit(data[i],g0,antlist1,antlist2,refant)

    return g_array
    
def bp_fit(data,antlist1,antlist2,bp0=None,refant=0):
    """
    Fit bandpass to visibility data.
   
    Parameters
    ----------
    data     : array of complex, shape(num_chans, baselines)
    bp0      : array of complex, shape(num_chans, num_ants) or None
    antlist1 : antenna mapping, for first antenna in bl pair 
    antlist2 : antenna mapping, for second antenna in bl pair 
    refant   : reference antenna

    Returns
    ------- 
    bpass : Bandpass, shape(num_chans, num_ants)
    """
   
    num_ants = ants_from_allbl(data.shape[1])
   
    # -----------------------------------------------------
    # initialise values for solver
    bpsoln = np.empty([data.shape[0],num_ants],dtype=np.complex) # Make empty array to fill bandpass into
    if not(np.any(bp0)): bp0 = np.empty([data.shape[0]], dtype=object) # set array init values to None if necessary

    # -----------------------------------------------------
    # solve for the bandpass over the channel range

    for c in range(data.shape[0]):
        # stefcal needs the visibilities as a list of [vis,vis.conjugate]
        vis_and_conj = np.concatenate((data[c], data[c].conj())) 
        fitted_bp = stefcal(vis_and_conj, num_ants, antlist1, antlist2, weights=1.0, num_iters=100, ref_ant=refant, init_gain=bp0[c])   
        bpsoln[c] = fitted_bp
      
    return bpsoln
   
def k_fit(data,antlist1,antlist2,chans=None,k0=None,bp0=None,refant=0,chan_sample=None):
    """
    Fit bandpass to visibility data.
   
    Parameters
    ----------
    data     : array of complex, shape(num_chans, baseline)
    antlist1 : antenna mapping, for first antenna in bl pair 
    antlist2 : antenna mapping, for second antenna in bl pair 
    k0       : array of complex, shape(num_chans, num_ants) or None
    bp0      : array of complex, shape(num_chans, num_ants) or None
    refant   : reference antenna

    Returns
    ------- 
    ksoln : Bandpass, shape(num_chans, num_ants)
    """
   
    num_ants = ants_from_allbl(data.shape[1])
   
    # -----------------------------------------------------
    # if channel sampling is specified, thin down the data and channel list
    data = data[::chan_sample,:]
    if np.any(chans): chans = chans[::chan_sample]
   
    # -----------------------------------------------------
    # initialise values for solver
    bpass = np.empty([data.shape[0],num_ants],dtype=np.complex) # Make empty array to fill bandpass into
    kdelay = np.empty(num_ants,dtype=np.complex) # Make empty array to fill delay into
    if not(np.any(bp0)): bp0 = np.empty([data.shape[0]], dtype=object) # set array init values to None if necessary
    if not(np.any(chans)): chans = np.arange(data.shape[0])

    # -----------------------------------------------------
    # solve for the bandpass over the channel range
    for c in range(data.shape[0]):
        # stefcal needs the visibilities as a list of [vis,vis.conjugate]
        vis_and_conj = np.concatenate((data[c], data[c].conj())) 
        fitted_bp = stefcal(vis_and_conj, num_ants, antlist1, antlist2, weights=1.0, num_iters=100, ref_ant=refant, init_gain=bp0[c])   
        bpass[c] = fitted_bp
      
    # -----------------------------------------------------
    # find bandpass phase slopes (delays)
    for i,bp_phase in enumerate(np.angle(bpass).T):
        A = np.array([ chans, np.ones(len(chans))])
        kdelay[i] = np.linalg.lstsq(A.T,bp_phase)[0][0]
   
    return kdelay      
   
def wavg(data,flags,weights,axis=0):
    """
    Perform weighted average of data, applying flags, 
    over specified axis
   
    Parameters
    ----------
    data    : array of complex
    flags   : array of boolean
    weights : array of floats
    axis    : axis to average over
   
    Returns
    -------
    average : weighted average of data  
    """
   
    return np.nansum(data*weights*(~flags),axis=axis)/np.nansum(weights*(~flags),axis=axis)
   
def wavg_full(data,flags,weights,axis=0):
    """
    Perform weighted average of data, flags and weights, 
    applying flags, over specified axis
   
    Parameters
    ----------
    data       : array of complex
    flags      : array of boolean
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
    #print np.nansum(weights*(~flags),axis=axis)
    #pri
    # fake flags and weights for now
    av_flags = np.zeros_like(av_data,dtype=np.bool)
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
#--- CLASS :  CalSolution
#--------------------------------------------------------------------------------------------------

class CalSolution(Solution):
   
    def __init__(self, soltype, values, times):
        self.soltype = soltype
        self.solvalues = solvalues
        self.times = times


