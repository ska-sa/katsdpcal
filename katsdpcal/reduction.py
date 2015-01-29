
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
    timing_file = 'timing.txt'
    if os.path.isfile(timing_file): os.remove(timing_file)
    timing_file = open("timing.txt", "w")

    # ----------------------------------------------------------
    # some general constants
    #    will mostly be in the Telescope Model later?
    #    or combination of TM and param file?

    params = parameters.set_params()

    #per_scan_plots = params['per_scan_plots']
    #closing_plots = params['closing_plots']
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

    nant = ts.nant
    nbl = nant*(nant+1)/2
    nchan = ts.nchan
    chans = np.arange(nchan)
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
        # fudge for now to add delay cal tag to bpcals
        if 'bpcal' in taglist: taglist.append('delaycal')
        
        print 'Pipeline calibration of target: ', target, scan_state, taglist

        # -------------------------------------------
        # extract hh and vv
        vis_hh = data['vis'][ti0:ti1,:,:,0] #np.array([vis_row[:,hh_mask] for vis_row in vis])
        flags_hh = data['flags'][ti0:ti1,:,:,0] #np.array([flag_row[:,hh_mask] for flag_row in flags])
        weights_hh = data['weights'][ti0:ti1,:,:,0] #np.array([weight_row[:,hh_mask] for weight_row in weights])
        times_hh = data['times'][ti0:ti1]
        # -------------------------------------------

        # initial RFI flagging
        rfi()

        # -------------------------------------------
        # perform calibration as appropriate, from scan intent tags        
        
        # process data depending on tag
        run_t0 = time()
        
        if any('delaycal' in k for k in taglist):
            # ---------------------------------------
            # Preliminary G solution
            pipeline_logger.info('Solving for preliminary gain on delay calibrator {0}'.format(target.split(',')[0],))
            
            # set up solution interval
            solint, dumps_per_solint = calprocs.solint_from_nominal(k_solint,dump_period,len(times_hh))
            # first averge solution interval then channel
            ave_vis_hh, ave_flags_hh, ave_weights_hh, av_sig_hh, ave_times_hh = calprocs.wavg_full_t(vis_hh,flags_hh,weights_hh,
                    dumps_per_solint,axis=0,times=times_hh)
            ave_vis_hh = calprocs.wavg(ave_vis_hh,ave_flags_hh,ave_weights_hh,axis=1)
            # solve for gains
            pre_g_soln_hh = CalSolution('G', calprocs.g_fit_per_solint(ave_vis_hh,dumps_per_solint,antlist1,antlist2,g0_h,REFANT), 
                    ave_times_hh, solint, corrprod_lookup)
                  
            # ---------------------------------------
            # Apply preliminary G solution
            g_to_apply = pre_g_soln_hh.interpolate(times_hh,dump_period=dump_period) 
            vis_hh = g_to_apply.apply(vis_hh)    
      
            # ---------------------------------------
            # K solution
            pipeline_logger.info('Solving for delay on delay calibrator {0}'.format(target.split(',')[0],))
            
            # average over all time
            ave_vis_hh = calprocs.wavg(vis_hh,flags_hh,weights_hh,axis=0)
            # solve for K
            k_soln_hh = CalSolution('K',calprocs.k_fit(ave_vis_hh,antlist1,antlist2,chans,k0_h,bp0_h,REFANT,chan_sample=k_chan_sample),
              np.ones(nant), 'inf', corrprod_lookup)          
            # solve for std deviation
            #if options.keep_stats:
            #    stddev_hh = np.std(np.abs(ave_vis_hh),axis=0)
            #    std_soln_hh = CalSolution('STD',calprocs.g_fit(stddev_hh,None,antlist1,antlist2,REFANT),np.ones(num_ants), 'inf', corrprod_lookup_hh)
          
            # ---------------------------------------  
            # update TS
            ts.add('K',k_soln_hh.values)
            #if options.keep_stats:
            #    ts.add('K_std',std_soln_hh.values)
      
            # ---------------------------------------
            timing_file.write("Delay cal:    %s \n" % (np.round(time()-run_t0,3),))
            run_t0 = time()
            
        if any( 'bpcal' in k for k in taglist ):  
            # ---------------------------------------
            # Apply K solution
            try:
                pipeline_logger.info('Applying delay to bandpass calibrator {0}'.format(target.split(',')[0],))
                k_current = ts.get('K')
                k_soln_hh = CalSolution('K', k_current, np.ones(nant), 'inf', corrprod_lookup)
                k_to_apply = k_soln_hh.interpolate(times_hh)
                vis_hh = k_to_apply.apply(vis_hh, chans)
            except KeyError:
                # TS doesn't yet contain 'K'
                pipeline_logger.info('Solving for gain prior to delay correction')
      
            # ---------------------------------------
            # Preliminary G solution
            pipeline_logger.info('Solving for preliminary gain on delay calibrator {0}'.format(target.split(',')[0],))
            
            # set up solution interval
            solint, dumps_per_solint = calprocs.solint_from_nominal(k_solint,dump_period,len(times_hh))
            # first averge solution interval then channel
            ave_vis_hh, ave_flags_hh, ave_weights_hh, av_sig_hh, ave_times_hh = calprocs.wavg_full_t(vis_hh,flags_hh,weights_hh,
                    dumps_per_solint,axis=0,times=times_hh)
            ave_vis_hh = calprocs.wavg(ave_vis_hh,ave_flags_hh,ave_weights_hh,axis=1)
            # solve for gains
            pre_g_soln_hh = CalSolution('G', calprocs.g_fit_per_solint(ave_vis_hh,dumps_per_solint,antlist1,antlist2,g0_h,REFANT), 
                    ave_times_hh, solint, corrprod_lookup)
                  
            # ---------------------------------------
            # Apply preliminary G solution
            g_to_apply = pre_g_soln_hh.interpolate(times_hh,dump_period=dump_period) 
            vis_hh = g_to_apply.apply(vis_hh) 
      
            # ---------------------------------------
            # BP solution
            #   solve for bandpass
        
            # first average over all time
            ave_vis_hh = calprocs.wavg(vis_hh,flags_hh,weights_hh,axis=0)
            # then solve for BP    
            bp_soln_hh = CalSolution('B',calprocs.bp_fit(ave_vis_hh,antlist1,antlist2,bp0_h,REFANT),
              np.ones(nant), 'inf', corrprod_lookup)
              
            # ---------------------------------------  
            # update TM
            ts.add('B',bp_soln_hh.values)
      
            # ---------------------------------------
            timing_file.write("Bandpass cal:    %s \n" % (np.round(time()-run_t0,3),))
            run_t0 = time()
        
        if any('gaincal' in k for k in taglist):
            # Apply K solution
            try:
                pipeline_logger.info('Applying delay to gain calibrator {0}'.format(target.split(',')[0],))
                k_current = ts.get('K')
                k_soln_hh = CalSolution('K', k_current, np.ones(nant), 'inf', corrprod_lookup)
                k_to_apply = k_soln_hh.interpolate(times_hh)
                vis_hh = k_to_apply.apply(vis_hh, chans)
            except KeyError:
                # TS doesn't yet contain 'K'
                pipeline_logger.info('Solving for gain prior to delay correction')
                
            # Apply BP solution    
            try:
                pipeline_logger.info('Applying bandpass to gain calibrator {0}'.format(target.split(',')[0],))
                bp_current = ts.get('BP')
                bp_soln_hh = CalSolution('B',bp_current, np.ones(nant), 'inf', corrprod_lookup)
                bp_to_apply = bp_soln_hh.interpolate(times_hh) 
                vis_hh = bp_to_apply.apply(vis_hh)
            except KeyError:
                # TS doesn't yet contain 'B'
                pipeline_logger.info('Solving for gain prior to bandpass correction')
        
            # ---------------------------------------
            # Preliminary G solution
            pipeline_logger.info('Solving for preliminary gain on gain calibrator {0}'.format(target.split(',')[0],))
            
            # set up solution interval
            solint, dumps_per_solint = calprocs.solint_from_nominal(g_solint,dump_period,len(times_hh))
            # first averge solution interval then channel
            
            ave_vis_hh, ave_flags_hh, ave_weights_hh, av_sig_hh, ave_times_hh = calprocs.wavg_full_t(vis_hh,flags_hh,weights_hh,
                    dumps_per_solint,axis=0,times=times_hh)
            ave_vis_hh = calprocs.wavg(ave_vis_hh,ave_flags_hh,ave_weights_hh,axis=1)
            # solve for gains
            g_soln_hh = CalSolution('G', calprocs.g_fit_per_solint(ave_vis_hh,dumps_per_solint,antlist1,antlist2,g0_h,REFANT), 
                    ave_times_hh, solint, corrprod_lookup)
                
            print 'gains - ' #, g_soln_hh.values
            pipeline_logger.info('Saving gains to TS')
            ts.add('G',g_soln_hh.values)
            

