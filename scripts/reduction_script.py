
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

# ----------------------------------------------------------
# place holders

def rfi():
    print 'Some sort of RFI flagging?!'

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

per_scan_plots = params['per_scan_plots']
closing_plots = params['closing_plots']
REFANT = params['refant']

# solution intervals
bp_solint = params['bp_solint'] #seconds
k_solint = params['k_solint'] #seconds
k_chan_sample = params['k_chan_sample']
g_solint = params['g_solint'] #seconds

# Hacky way to stop the simulated data 
NUM_ITERS = 500

# only use inner channels for simulation
BCHAN = params['bchan']
ECHAN = params['echan']

# plots per scan
#per_scan_plots = True

# ----------------------------------------------------------
# H5 file to use for simulation
#   simulation of data and Teselcope Model (TM)

parser = optparse.OptionParser(usage="%prog [options] <filename.h5>", description='Run MeerKAT calibration pipeline on H5 file')
parser.add_option("-C", "--channel-range", help="Range of frequency channels to process (zero-based inclusive 'first_chan,last_chan', default is all channels)")
parser.add_option("-t", "--write-target", action="store_true", default=False, help="Write the corrected target data back into the H5 file")
parser.add_option("-b", "--write-bandpass", action="store_true", default=False, help="Write the corrected bandpass calibrator data back into the H5 file")
parser.add_option("-g", "--write-gaincal", action="store_true", default=False, help="Write the corrected gain calibrator data back into the H5 file")
parser.add_option("-s", "--keep-stats", action="store_true", default=False, help="Keep stats of calibration solution data along with the calibration solution")
(options, args) = parser.parse_args()

if len(args) < 1 or not args[0].endswith(".h5"):
    print "Please provide an H% filename as argument"
    sys.exit(1)
        
file_name = args[0]
project_name = file_name.split('.')[0]
simdata = SimData(file_name)
    
# Select frequency channel range
#   if channel range is set in the parser options it overrides the parameter file
if options.channel_range is not None:
    channel_range = [int(chan_str) for chan_str in options.channel_range.split(',')]
    BCHAN, ECHAN = channel_range[0], channel_range[1]

    if (BCHAN < 0) or (ECHAN >= simdata.shape[1]):
        print "Requested channel range outside data set boundaries. Set channels in the range [0,%s]" % (simdata.shape[1]-1,)
        sys.exit(1)
        
simdata.select(channels=slice(BCHAN,ECHAN))

# ----------------------------------------------------------
# Fake Telescope Model dictionary
#   faking it up for now, until such time as 
#   kattelmod is more developed.
TMfile = 'TM.pickle'
# use parameters from parameter file as defaults, but override with TMfile
TM = simdata.setup_TM(TMfile,params)

# ----------------------------------------------------------
# extract values we need frequently from the TM

num_ants = TM['num_ants']
#num_chans = TM['num_channels'] 
# just using innver 600 channels for now
num_chans = ECHAN-BCHAN
chans = np.arange(num_chans)
dump_period = TM['dump_period']

# ----------------------------------------------------------
# antenna mapping

# make polarisation and corr_prod lookup tables (assume this doesn't change over the course of an observaton)
antlist_index = dict([(TM['antlist'][i], i) for i in range(TM['num_ants'])])
corrprod_lookup = np.array([[antlist_index[a1[0:4]],antlist_index[a2[0:4]]] for a1,a2 in TM['corr_products']])
# True for h pol:
pol_lookup = np.array([[a1[4]=='h',a2[4]=='h'] for a1,a2 in TM['corr_products']])

# -------------------------------------------    
# masks for extracting hh and vv (assume constant over an observation)
hh_mask = np.array([(corrprod_lookup[i][0]!=corrprod_lookup[i][1])and(pol_lookup[i][0])and(pol_lookup[i][1]) for i in range(len(TM['corr_products']))])
vv_mask = np.array([(corrprod_lookup[i][0]!=corrprod_lookup[i][1])and(~pol_lookup[i][0])and(~pol_lookup[i][1]) for i in range(len(TM['corr_products']))])

# antenna name lookup for hh and vv (assume constant over an observation)
corrprod_lookup_hh = corrprod_lookup[hh_mask]
corrprod_lookup_vv = corrprod_lookup[vv_mask]

# get antenna number lists for stefcal - need vis then vis.conj (assume constant over an observation)
# assume same for hh and vv
antlist1 = np.concatenate((corrprod_lookup_hh[:,0], corrprod_lookup_hh[:,1]))
antlist2 = np.concatenate((corrprod_lookup_hh[:,1], corrprod_lookup_hh[:,0]))

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

scan_iter = 0

for scan_ind, scan_state, target in simdata.scans():
  #if target.name == '0637-752':   
  #if scan_ind < 8: 
    
    scan_iter = scan_iter+1

    num_dumps = simdata.shape[0]
   
    if scan_state != 'track':
        #print "    scan %3d (%4d samples) skipped '%s' - not a track" % (scan_ind, num_dumps, scan_state)
        continue
    elif num_dumps < 2:
        #print "    scan %3d (%4d samples) skipped - too short" % (scan_ind, num_dumps)
        continue
    else:
        print "Scan %3d Target: %s" % (scan_ind, target.name)   

    # -------------------------------------------
    # data has shape(num_times, num_chans, num_baselines)
    vis = simdata.vis[:]
    times = simdata.timestamps[:] #-simdata.timestamps[:][0]
    time_offset = simdata.timestamps[:][0]
    flags = simdata.flags()[:]
    weights = simdata.weights()[:]

    # -------------------------------------------
    # extract hh and vv
    vis_hh = np.array([vis_row[:,hh_mask] for vis_row in vis])
    flags_hh = np.array([flag_row[:,hh_mask] for flag_row in flags])
    weights_hh = np.array([weight_row[:,hh_mask] for weight_row in weights])
   
    # fudge for now to add delay cal tag to bpcals
    if 'bpcal' in target.tags: target.tags.append('delaycal')
    taglist = [target.tags]
   
    t0 = time()

    # initial RFI flagging?
    rfi()
   
    if any( 'target' in k for k in taglist ):
        # ---------------------------------------
        # save target data for processing after the gain cal solution
        #   this assumes a model where we get data in chunks of 
        #   [target, gaincal], [target, gaincal]...
        #   mocking it up for now
        target_h = vis_hh
        target_chan_mask = simdata._freq_keep
        target_time_mask = simdata._time_keep
        target_times = times
        target_num_dumps = num_dumps
      
        pre_target_g_hh = copy.deepcopy(g_soln_hh)
   
    # process data depending on tag
    if any( 'delaycal' in k for k in taglist ):
        # ---------------------------------------
        # Preliminary G solution
        # set up solution interval
        solint, dumps_per_solint = calprocs.solint_from_nominal(k_solint,dump_period,len(times))
        # first averge solution interval then channel
        ave_vis_hh, ave_flags_hh, ave_weights_hh, av_sig_hh, ave_times_hh = calprocs.wavg_full_t(vis_hh,flags_hh,weights_hh,
                dumps_per_solint,axis=0,times=times)
        ave_vis_hh = calprocs.wavg(ave_vis_hh,ave_flags_hh,ave_weights_hh,axis=1)
        # solve for gains
        pre_g_soln_hh = CalSolution('G', calprocs.g_fit_per_solint(ave_vis_hh,dumps_per_solint,antlist1,antlist2,g0_h,REFANT), 
                ave_times_hh, solint, corrprod_lookup_hh)

        # plot the G solutions
        #if per_scan_plots: plotting.plot_g_solns(g_array)
      
        # ---------------------------------------
        # Apply preliminary G solution
        g_to_apply = pre_g_soln_hh.interpolate(times,dump_period=dump_period) 
        vis_hh = g_to_apply.apply(vis_hh)    
            
        # plot data with G solutions applied:
        if per_scan_plots: fig_list.append(plotting.plot_bp_data(vis_hh,plotavg=True))
      
        # ---------------------------------------
        # K solution
        # average over all time
        ave_vis_hh = calprocs.wavg(vis_hh,flags_hh,weights_hh,axis=0)
        # solve for K
        k_soln_hh = CalSolution('K',calprocs.k_fit(ave_vis_hh,antlist1,antlist2,chans,k0_h,bp0_h,REFANT,chan_sample=k_chan_sample),
          np.ones(num_ants), 'inf', corrprod_lookup_hh)          
        # solve for std deviation
        if options.keep_stats:
            stddev_hh = np.std(np.abs(ave_vis_hh),axis=0)
            std_soln_hh = CalSolution('STD',calprocs.g_fit(stddev_hh,None,antlist1,antlist2,REFANT),np.ones(num_ants), 'inf', corrprod_lookup_hh)
          
        # ---------------------------------------  
        # update TM
        TM['K'].append(k_soln_hh.values)
        TM['K_current'] = k_soln_hh.values
        if options.keep_stats:
            TM['K_std'].append(std_soln_hh.values)
            TM['K_std_current'] = std_soln_hh.values
      
        # ---------------------------------------
        timing_file.write("Delay cal:    %s \n" % (np.round(time()-t0,3),))
        t0 = time()

    if any( 'bpcal' in k for k in taglist ):      
        # ---------------------------------------
        # Apply K solution
        k_current = TM['K_current']
        k_soln_hh = CalSolution('K', k_current, np.ones(num_ants), 'inf', corrprod_lookup_hh)
        k_to_apply = k_soln_hh.interpolate(times)
        vis_hh = k_to_apply.apply(vis_hh, chans)
      
        # plot data with K solutions applied:
        #if per_scan_plots: plotting.plot_bp_data(vis_hh,plotavg=True)
      
        # ---------------------------------------
        # Preliminary G solution
        # set up solution interval
        solint, dumps_per_solint = calprocs.solint_from_nominal(bp_solint,dump_period,len(times))
        # first averge solution interval then channel
        ave_vis_hh, ave_flags_hh, ave_weights_hh, av_sig_hh, ave_times_hh = calprocs.wavg_full_t(vis_hh,flags_hh,weights_hh,
                dumps_per_solint,axis=0,times=times)
        ave_vis_hh = calprocs.wavg(ave_vis_hh,ave_flags_hh,ave_weights_hh,axis=1)
        # solve for gains
        pre_g_soln_hh = CalSolution('G', calprocs.g_fit_per_solint(ave_vis_hh,dumps_per_solint,antlist1,antlist2,g0_h,REFANT), 
                ave_times_hh, solint, corrprod_lookup_hh)
                
        # solve for std deviation
        if options.keep_stats:
            stddev_hh = np.std(np.abs(ave_vis_hh),axis=0)
            print av_sig_hh.shape, stddev_hh.shape
            #pri
            std_soln_hh = CalSolution('STD',calprocs.g_fit(stddev_hh,None,antlist1,antlist2,REFANT),np.ones(num_ants), 'inf', corrprod_lookup_hh)
         
        # plot the G solutions
        #if per_scan_plots: plotting.plot_g_solns(g_array)
      
        # ---------------------------------------
        # Apply preliminary G solution
        g_to_apply = pre_g_soln_hh.interpolate(times,dump_period=dump_period) 
        vis_hh = g_to_apply.apply(vis_hh)
            
        # plot data with G solutions applied:
        #if per_scan_plots: plotting.plot_bp_data(vis_hh,plotavg=True)
      
        # ---------------------------------------
        # BP solution
        #   solve for bandpass
        
        # first average over all time
        ave_vis_hh = calprocs.wavg(vis_hh,flags_hh,weights_hh,axis=0)
        # then solve for BP    
        bp_soln_hh = CalSolution('B',calprocs.bp_fit(ave_vis_hh,antlist1,antlist2,bp0_h,REFANT),
          np.ones(num_ants), 'inf', corrprod_lookup_hh)
              
        # ---------------------------------------  
        # update TM
        TM['BP'].append(bp_soln_hh.values)
        TM['BP_current'] = bp_soln_hh.values
        if options.keep_stats:
            TM['BP_std'].append(std_soln_hh.values)
            TM['BP_std_current'] = std_soln_hh.values
      
        # ---------------------------------------
        # Apply BP solution 
        bp_to_apply = bp_soln_hh.interpolate(times) 
        vis_hh = bp_to_apply.apply(vis_hh)
      
        # plot data with K solutions applied:
        if per_scan_plots: fig_list.append(plotting.plot_bp_data(vis_hh,plotavg=True))
        # plot all data:
        #if per_scan_plots: plotting.plot_bp_solns(bp_soln_hh.values)   
      
        # ---------------------------------------
        # write the calibrated bandpass data back to h5 file
        if options.write_bandpass: simdata.write_h5(vis_hh,hh_mask)
      
        # ---------------------------------------
        timing_file.write("Bandpass cal: %s \n" % (np.round(time()-t0,3),))
        t0 = time()
      
    if any( 'fluxcal' in k for k in taglist ):
        print "   flux scaling not yet implimented"
        continue
      
        timing_file.write("Amp cal: %s \n" % (np.round(time()-t0,3),))
        t0 = time()
      
    if any( 'gaincal' in k for k in taglist ):
        # ---------------------------------------
        # Apply K and BP solutions
        # Apply K solution
        k_current = TM['K_current']
        k_soln_hh = CalSolution('K', k_current, np.ones(num_ants), 'inf', corrprod_lookup_hh)
        k_to_apply = k_soln_hh.interpolate(times)
        vis_hh = k_to_apply.apply(vis_hh, chans)
        
        bp_current = TM['BP_current']
        bp_soln_hh = CalSolution('B',bp_current, np.ones(num_ants), 'inf', corrprod_lookup_hh)
        bp_to_apply = bp_soln_hh.interpolate(times) 
        vis_hh = bp_to_apply.apply(vis_hh)
      
        # ---------------------------------------
        # Preliminary G solution
        # set up solution interval
        solint, dumps_per_solint = calprocs.solint_from_nominal(g_solint,dump_period,len(times))
        # first averge solution interval then channel
        ave_vis_hh, ave_flags_hh, ave_weights_hh, av_sig_hh, ave_times_hh = calprocs.wavg_full_t(vis_hh,flags_hh,weights_hh,
                dumps_per_solint,axis=0,times=times)
        ave_vis_hh = calprocs.wavg(ave_vis_hh,ave_flags_hh,ave_weights_hh,axis=1)
        # solve for gains
        g_soln_hh = CalSolution('G', calprocs.g_fit_per_solint(ave_vis_hh,dumps_per_solint,antlist1,antlist2,g0_h,REFANT), 
                ave_times_hh, solint, corrprod_lookup_hh)

        # plot the G solutions
        #if per_scan_plots: plotting.plot_g_solns(g_array)
         
        # plot the G solutions
        if per_scan_plots: fig_list.append(plotting.plot_g_solns(g_soln_hh.times,g_soln_hh.values))
      
        # ---------------------------------------
        # RFI flagging
      
        #temporarily apply preliminary gains for RFI flagging
        # ... ?
        rfi()
      
        # ---------------------------------------
        # Next G solution
        # set up solution interval: just solve for two intervals per G scan
        dumps_per_solint = np.ceil(num_dumps/2.0)
        solint = dumps_per_solint*dump_period
        # first averge solution interval then channel
        ave_vis_hh, ave_flags_hh, ave_weights_hh, av_sig_hh, ave_times_hh = calprocs.wavg_full_t(vis_hh,flags_hh,weights_hh,
                dumps_per_solint,axis=0,times=times)
        if options.keep_stats:
            stddev_hh = np.std(np.abs(ave_vis_hh),axis=1)
            # fidge for nans - fix later
            stddev_hh[np.isnan(stddev_hh)] = np.nanmean(stddev_hh)

        ave_vis_hh = calprocs.wavg(ave_vis_hh,ave_flags_hh,ave_weights_hh,axis=1)
        # solve for gains
        g_soln_hh = CalSolution('G', calprocs.g_fit_per_solint(ave_vis_hh,dumps_per_solint,antlist1,antlist2,g0_h,REFANT), 
                ave_times_hh, solint, corrprod_lookup_hh)
                
        # solve for std deviation
        #std_soln_hh = CalSolution('STD',calprocs.g_fit_per_solint(av_sig_hh,dumps_per_solint,antlist1,antlist2,g0,REFANT),np.ones(num_ants), 'inf', corrprod_lookup_hh)
       
        # plot the G solutions
        #if per_scan_plots: plotting.plot_g_solns(g_array)
      
        # ---------------------------------------  
        # update TM
        for g in g_soln_hh.values: TM['G'].append(g)
        for t in g_soln_hh.times: TM['G_times'].append(t)
        #TM['G'].append(np.ravel(g_soln_hh.values))
        #TM['G_times'].append(np.ravel(g_soln_hh.times))
        TM['G_current'] = g_soln_hh.values
    
        if options.keep_stats:
            for s in std_soln_hh.values: TM['G_std'].append(s)
            TM['G_std_current'] = std_soln_hh.values

        # ---------------------------------------
        # write the calibrated gaincal data back to h5 file
        #    apply G solution 
        #g_to_apply = g_soln_hh.interpolate(times) 
        #vis_hh = g_to_apply.apply(vis_hh)
        if options.write_gaincal: simdata.write_h5(vis_hh,hh_mask)
      
        # ---------------------------------------
        timing_file.write("Gain cal: %s \n" % (np.round(time()-t0,3),))
        t0 = time()
      
        if target_h is not None:
            # ---------------------------------------
            # apply gains to target
            #   this assumes a model where we get data in chunks of 
            #   [target, gaincal], [target, gaincal]...
            #   mocking it up for now
      
            # ---------------------------------------
            # Apply K and BP solutions
            k_to_apply = k_soln_hh.interpolate(target_times) 
            target_h = k_to_apply.apply(target_h, chans)
            bp_to_apply = bp_soln_hh.interpolate(target_times) 
            target_h = bp_to_apply.apply(target_h)
      
            # ---------------------------------------
            # Apply G solutions
         
            cal_g_sol_hh = g_soln_hh.concat(pre_target_g_hh)
            g_sol_hh_to_apply = cal_g_sol_hh.interpolate(target_times)
            target_h = g_sol_hh_to_apply.apply(target_h)
      
            # ---------------------------------------
            # RFI flagging
            rfi()
      
            # ---------------------------------------
            # write the calibrated target data back to h5 file
            if options.write_target: simdata.write_h5(target_h,hh_mask,tmask=target_time_mask,cmask=target_chan_mask)
      
            # ---------------------------------------
            timing_file.write("Source (cal application): %s \n" % (np.round(time()-t0,3),))
            t0 = time()
      

    print
    if scan_iter>NUM_ITERS: break
        

timing_file.close()
# save TM
#pickle.dump(TM, open('TM.pickle', 'wb'))

if closing_plots:
    # plot BP solutions from TM
   
    # plot G solutions from TM
    g_solns = np.array(TM['G'])
    if options.keep_stats: g_std = np.array(TM['G_std'])
    g_soln_times = np.array(TM['G_times'])
    #plotting.plot_g_solns(g_soln_times,g_solns)
    if options.keep_stats:
        fig_list.append(plotting.plot_g_solns_with_errors(g_soln_times,g_solns,g_std))
    else:
        fig_list.append(plotting.plot_g_solns(g_soln_times,g_solns)) 
           
    # plot BP solutions
    fig_list.append(plotting.plot_bp_soln_list(np.array(TM['BP'])))
    
report_name = file_name.replace('.h5','.pdf')
#plotting.flush_plots(fig_list,report_name)
report.make_cal_report(project_name,fig_list)

