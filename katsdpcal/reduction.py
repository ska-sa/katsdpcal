
import numpy as np
import optparse
import sys
import copy

import pickle
import os

from katcal import plotting
from katcal import calprocs
from katcal.calsolution import CalSolution
from katcal import parameters
from katcal.simulator import SimData
from katcal import report

from time import time

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ThreadLoggingAdapter(logging.LoggerAdapter):
    """
    Logging adaptor which prepends a bracketed string to the log message.
    The string is passed via a 'connid' key.    
    """
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['connid'], msg), kwargs

def rfi():
    """
    Place holder for RFI detection algorithms to come.
    """
    print 'Some sort of RFI flagging?!'
    
def get_nearest_from_ts(ts,key,t,dt=15.0):
    """
    Given a time, extract the value of a key in the Telescope State closest to that time.
    
    Inputs:
    =======
    ts  : TelescopeState
    key : string, key to be extracted from the ts
    t   : time that value of the key is desired 
    dt  : range of times over which to search for key value, bracketing time t
    
    Returns:
    ========
    Value of a key in the Telescope State closest to time t
    
    """    
    key_list = np.array(ts.get_range(key,st=t-dt,et=t+dt))
    time_diffs = [np.float(line[1]) - t for line in key_list]
    return key_list[np.argmin(time_diffs)] 

def pipeline(data, ts, thread_name):
    """
    Pipeline calibration
    """

    # ----------------------------------------------------------    
    # set up logging adapter for the pipeline thread
    pipeline_logger = ThreadLoggingAdapter(logger, {'connid': thread_name})
    
    # ----------------------------------------------------------
    # set up timing file

    # leave timing for now
    #timing_file = 'timing.txt'
    #if os.path.isfile(timing_file): os.remove(timing_file)
    #timing_file = open("timing.txt", "w")

    # ----------------------------------------------------------
    # some general constants
    #    will mostly be in the Telescope Model later?
    #    or combination of TM and param file?

    params = parameters.set_params()

    per_scan_plots = params['per_scan_plots']
    closing_plots = params['closing_plots']
    REFANT = params['refant']

    # solution intervals
    bp_solint = params['bp_solint'] #seconds
    k_solint = params['k_solint'] #seconds
    k_chan_sample = params['k_chan_sample']
    g_solint = params['g_solint'] #seconds

    # plots per scan
    #per_scan_plots = True

    # ----------------------------------------------------------
    # extract values we need frequently from the TM

    nant = ts.num_ants
    nbl = nant*(nant+1)/2
    nchan = ts.echan - ts.bchan
    chan = np.arange(nchan)
    dump_period = ts.dump_period
    antlist = ts.antlist
    corr_products = ts.corr_products

    # ----------------------------------------------------------
    # antenna mapping

    # make polarisation and corr_prod lookup tables (assume this doesn't change over the course of an observaton)
    antlist_index = dict([(antlist[i], i) for i in range(len(antlist))])
    corr_products_lookup = np.array([[antlist_index[a1[0:4]],antlist_index[a2[0:4]]] for a1,a2 in corr_products])
    
    # from full list of correlator products, get list without repeats (ie no repeats for pol)
    corrprod_lookup = -1*np.ones([nbl,2],dtype=np.int) # start with array of -1
    bl = -1
    for c in corr_products_lookup: 
        if not np.all(corrprod_lookup[bl] == c): 
            bl += 1
            corrprod_lookup[bl] = c
  
    # NOTE: no longer need hh and vv masks as we re-ordered the data to be ntime x nchan x nbl x npol

    # get antenna number lists for stefcal - need vis then vis.conj (assume constant over an observation)
    # assume same for hh and vv
    antlist1 = np.concatenate((corrprod_lookup[:,0], corrprod_lookup[:,1]))
    antlist2 = np.concatenate((corrprod_lookup[:,1], corrprod_lookup[:,0]))

    # ----------------------------------------------------------
    # set initial values for fits

    bp0_h = None
    k0_h = None
    g0_h = None
    target_h = None

    # ----------------------------------------------------------
    


    #print "Scan %3d Target: %s" % (scan_ind, target.name)   

    # -------------------------------------------
    # data has shape(ntime x nchan x nbl x npol)
    vis = data['vis']
    times = data['times']
    flags = data['flags']
    weights = np.ones_like(flags,dtype=np.float)
    
    # iterate through the track scans accumulated into the data buffer
    for i in range(len(data['track_start_indices'])-1):
        # start and end indices for this track in the data buffer
        ti0 = data['track_start_indices'][i]
        ti1 = data['track_start_indices'][i+1]
    
        # start time
        t0 = times[ti0]
    
        # extract scan info from the TS
        target = get_nearest_from_ts(ts,'target',t0)[0]
        scan_state = get_nearest_from_ts(ts,'scan_state',t0)[0]
        taglist = get_nearest_from_ts(ts,'tag',t0)[0]
        print 'Pipeline calibration of target: ', target, scan_state, taglist
        

        

        # -------------------------------------------
        # extract hh and vv
        vis_hh = data['vis'][ti0:ti1,:,:,0] #np.array([vis_row[:,hh_mask] for vis_row in vis])
        flags_hh = data['flags'][ti0:ti1,:,:,0] #np.array([flag_row[:,hh_mask] for flag_row in flags])
        weights_hh = data['weights'][ti0:ti1,:,:,0] #np.array([weight_row[:,hh_mask] for weight_row in weights])
        times_hh = data['times'][ti0:ti1]

        # fudge for now to add delay cal tag to bpcals
        #if 'bpcal' in tags: tags.append('delaycal')
        #taglist = [tags]
        # -------------------------------------------

        # initial RFI flagging
        rfi()

        # -------------------------------------------
        # perform calibration as appropriate, from scan intent tags
        
        if any('gaincal' in k for k in taglist):
            # ---------------------------------------
            # Apply K and BP solutions
            # Apply K solution
            #k_current = TM['K_current']
            #k_soln_hh = CalSolution('K', k_current, np.ones(num_ants), 'inf', corrprod_lookup_hh)
            #k_to_apply = k_soln_hh.interpolate(times)
            #vis_hh = k_to_apply.apply(vis_hh, chans)

            #bp_current = TM['BP_current']
            #bp_soln_hh = CalSolution('B',bp_current, np.ones(num_ants), 'inf', corrprod_lookup_hh)
            #bp_to_apply = bp_soln_hh.interpolate(times) 
            #vis_hh = bp_to_apply.apply(vis_hh)
        
            # ---------------------------------------
            # Preliminary G solution
            # set up solution interval
            solint, dumps_per_solint = calprocs.solint_from_nominal(g_solint,dump_period,len(times))
            # first averge solution interval then channel
            
            print vis_hh.shape
            print flags_hh.shape
            print weights_hh.shape
            
            ave_vis_hh, ave_flags_hh, ave_weights_hh, av_sig_hh, ave_times_hh = calprocs.wavg_full_t(vis_hh,flags_hh,weights_hh,
                    dumps_per_solint,axis=0,times=times)
            ave_vis_hh = calprocs.wavg(ave_vis_hh,ave_flags_hh,ave_weights_hh,axis=1)
            # solve for gains
            g_soln_hh = CalSolution('G', calprocs.g_fit_per_solint(ave_vis_hh,dumps_per_solint,antlist1,antlist2,g0_h,REFANT), 
                    ave_times_hh, solint, corrprod_lookup)
                
            print 'gains - ' #, g_soln_hh.values
            pipeline_logger.info('Saving gains to TS')
            ts.add('g_soln',g_soln_hh.values)
