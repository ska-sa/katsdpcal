
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

from katcal.telescope_model import TelescopeModel

from time import time

# ----------------------------------------------------------
# place holders

def rfi():
    print 'Some sort of RFI flagging?!'
    
def get_nearest_from_tm(tm,value,t,dt=15.0):    
    value_list = np.array(tm.get(value,st=t-dt,et=t+dt))
    time_diffs = [np.float(line[1]) - t for line in value_list]
    return value_list[np.argmin(time_diffs)] 
    
def find_tracks(tm,start_time,end_time):
    states = tm.get('scan_state',st=start_time,et=end_time)
    print states
    pri

    

def pipeline(data):
    
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
    # start TM
    tm = TelescopeModel(host='127.0.0.1',db=1)

    # ----------------------------------------------------------
    # extract values we need frequently from the TM

    nant = tm.num_ants
    
    nbl = nant*(nant+1)/2
    #num_chans = TM['num_channels'] 
    # just using innver 600 channels for now
    nchan = tm.echan - tm.bchan
    
    chan = np.arange(nchan)
    dump_period = tm.dump_period
    antlist = tm.antlist
    corr_products = tm.corr_products

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

    # True for h pol:
    #pol_lookup = np.array([[a1[4]=='h',a2[4]=='h'] for a1,a2 in corr_products])

    # -------------------------------------------    
    # no longer need hh and vv masks as we re-ordered the data to be ntime x nchan x nbl x npol
    
    # masks for extracting hh and vv (assume constant over an observation)
    #hh_mask = np.array([(corrprod_lookup[i][0]!=corrprod_lookup[i][1])and(pol_lookup[i][0])and(pol_lookup[i][1]) for i in range(len(corr_products))])
    #vv_mask = np.array([(corrprod_lookup[i][0]!=corrprod_lookup[i][1])and(~pol_lookup[i][0])and(~pol_lookup[i][1]) for i in range(len(corr_products))])

    # antenna name lookup for hh and vv (assume constant over an observation)
    #corrprod_lookup_hh = corrprod_lookup[hh_mask]
    #corrprod_lookup_vv = corrprod_lookup[vv_mask]

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

    # initialise list to hold figures
    fig_list = []

    # ----------------------------------------------------------
    # iterate through scans and calibrate scan by scan

    #num_dumps = times.shape[0]

    #if scan_state != 'track':
        #print "    scan %3d (%4d samples) skipped '%s' - not a track" % (scan_ind, num_dumps, scan_state)
    #    print '!'
    #elif num_dumps < 2:
    #    #print "    scan %3d (%4d samples) skipped - too short" % (scan_ind, num_dumps)
    #    print '!'

    #else:
    
    if True:

        #print "Scan %3d Target: %s" % (scan_ind, target.name)   

        # -------------------------------------------
        # data has shape(num_times, num_chans, num_baselines)
        vis = data['vis']
        times = data['times']
        print '***', times
        print vis[0,0,0,0]
        #pri
        print
        time_offset = times[0]
        flags = data['flags']
        weights = np.ones_like(flags,dtype=np.float)
        
        
        for i in range(len(data['track_start_indices'])-1):
            ti0 = data['track_start_indices'][i]
            ti1 = data['track_start_indices'][i+1]
            print 'time indicess - ', ti0, ti1
        
            # -------------------------------------------
            t0 = times[ti0]
        
            target = get_nearest_from_tm(tm,'target',t0)[0]
            scan_state = get_nearest_from_tm(tm,'scan_state',t0)[0]
            taglist = get_nearest_from_tm(tm,'tag',t0)[0]
            print target, scan_state, taglist
            
            if 'track' in scan_state:
                

                # -------------------------------------------
                # extract hh and vv
                vis_hh = data['vis'][ti0:ti1,:,:,0] #np.array([vis_row[:,hh_mask] for vis_row in vis])
                flags_hh = data['flags'][ti0:ti1,:,:,0] #np.array([flag_row[:,hh_mask] for flag_row in flags])
                weights_hh = data['weights'][ti0:ti1,:,:,0] #np.array([weight_row[:,hh_mask] for weight_row in weights])
                times_hh = data['times'][ti0:ti1]

                # fudge for now to add delay cal tag to bpcals
                #if 'bpcal' in tags: tags.append('delaycal')
                #taglist = [tags]

                #t0 = time()

                # initial RFI flagging?
                rfi()
            
                # gain cal

            
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
                    
                    tm.add('g_soln',g_soln_hh.values)
