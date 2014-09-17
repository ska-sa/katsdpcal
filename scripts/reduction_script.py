

from pyrap.tables import table
import numpy as np
#import matplotlib.pylab as plt
from scipy.stats import nanmean
from math import pi

import pickle
import os

import h5py
import katdal

from katcal import plotting
from katcal import calprocs
from katcal.calsolution import CalSolution
import katcalparams

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

params = katcalparams.set_params()

do_plots = params['do_plots']

REFANT = params['refant']

# solution intervals
bp_solint = params['bp_solint'] #seconds
k_solint = params['k_solint'] #seconds
k_chan_sample = params['k_chan_sample']
g_solint = params['g_solint'] #seconds

# Hacky way to stop the simulated data 
NUM_ITERS = 50

# only use inner channels for simulation
BCHAN = params['bchan']
ECHAN = params['echan']

# ----------------------------------------------------------
# H5 file to use for simulation
#   simulation of data and Temelcope Model (TM)

h5filename = 'tst.h5' #1345897247.h5' #1345967709.h5' #1337360375.h5' #1373441586.h5'
file_prefix = h5filename.split('.')[0]
h5 = katdal.open(h5filename)

# ----------------------------------------------------------
# Fake Telescope Model dictionary
#   faking it up for now, until such time as 
#   kattelmod is more developed.

# load initial bandpass from pickle
TMfile = 'TM.pickle'
TM = pickle.load(open(TMfile, 'rb')) if os.path.isfile(TMfile) else {}

# empty solutions - start with no solutions
TM['BP'] = []
TM['K'] = []
TM['G'] = []

# set siulated TM values from h5 file
TM['antlist'] = [ant.name for ant in h5.ants]
TM['num_ants'] = len(h5.ants)
TM['num_channels'] = len(h5.channels)
#antdesclist = [ant.description for ant in h5.ants]
TM['corr_products'] = h5.corr_products
TM['dump_period'] = h5.dump_period

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

bp0_hh = None
k0_hh = None
g0 = None

# ----------------------------------------------------------

scan_iter = 0

for scan_ind, scan_state, target in h5.scans():
   scan_iter = scan_iter+1

   num_dumps = h5.shape[0]
   
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
   vis = h5.vis[:,BCHAN:ECHAN,:]
   times = h5.timestamps[:]-h5.timestamps[:][0]
   time_offset = h5.timestamps[:][0]
   flags = h5.flags()[:,BCHAN:ECHAN,:]
   weights = h5.weights()[:,BCHAN:ECHAN,:]

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
   
   # process data depending on tag
   if any( 'delaycal' in k for k in taglist ):
      # ---------------------------------------
      # Preliminary G solution
      # set up solution interval
      solint, dumps_per_solint = calprocs.solint_from_nominal(k_solint,dump_period,len(times))
      # first averge solution interval then channel
      ave_vis_hh, ave_flags_hh, ave_weights_hh, times_hh = calprocs.wavg_full_t(vis_hh,flags_hh,weights_hh,
         dumps_per_solint,axis=0,times=times)
      ave_vis_hh = calprocs.wavg(ave_vis_hh,ave_flags_hh,ave_weights_hh,axis=1)
      # solve for gains
      g_soln_hh = CalSolution('G', calprocs.g_fit_per_solint(ave_vis_hh,dumps_per_solint,antlist1,antlist2,g0,REFANT), 
         times_hh, solint, corrprod_lookup_hh)

      # plot the G solutions
      #if do_plots: plotting.plot_g_solns(g_array)
      
      # ---------------------------------------
      # Apply preliminary G solution
      g_to_apply = g_soln_hh.interpolate(num_dumps,dump_period=dump_period) 
      vis_hh = g_to_apply.apply(vis_hh)    
            
      # plot data with G solutions applied:
      if do_plots: plotting.plot_bp_data(vis_hh,plotavg=True)
      
      # ---------------------------------------
      # K solution
      # average over all time
      ave_vis_hh = calprocs.wavg(vis_hh,flags_hh,weights_hh,axis=0)
      # solve for K
      k_soln_hh = CalSolution('K',calprocs.k_fit(ave_vis_hh,antlist1,antlist2,chans,k0_hh,bp0_hh,REFANT,chan_sample=k_chan_sample),
          np.ones(num_ants), 'inf', corrprod_lookup_hh)
      # update TM
      TM['K'].append(k_soln_hh.values)
      TM['K_current'] = k_soln_hh.values
      
      # ---------------------------------------
      timing_file.write("Delay cal:    %s \n" % (np.round(time()-t0,3),))
      t0 = time()

   if any( 'bpcal' in k for k in taglist ):      
      # ---------------------------------------
      # Apply K solution
      k_current = TM['K_current']
      k_soln_hh = CalSolution('K', k_current, np.ones(num_ants), 'inf', corrprod_lookup_hh)
      k_to_apply = k_soln_hh.interpolate(num_dumps)
      vis_hh = k_to_apply.apply(vis_hh, chans)
      
      # plot data with K solutions applied:
      #if do_plots: plotting.plot_bp_data(vis_hh,plotavg=True)
      
      # ---------------------------------------
      # Preliminary G solution
      # set up solution interval
      solint, dumps_per_solint = calprocs.solint_from_nominal(bp_solint,dump_period,len(times))
      # first averge solution interval then channel
      ave_vis_hh, ave_flags_hh, ave_weights_hh, times_hh = calprocs.wavg_full_t(vis_hh,flags_hh,weights_hh,
         dumps_per_solint,axis=0,times=times)
      ave_vis_hh = calprocs.wavg(ave_vis_hh,ave_flags_hh,ave_weights_hh,axis=1)
      # solve for gains
      g_soln_hh = CalSolution('G', calprocs.g_fit_per_solint(ave_vis_hh,dumps_per_solint,antlist1,antlist2,g0,REFANT), 
         times_hh, solint, corrprod_lookup_hh)
         
      # plot the G solutions
      #if do_plots: plotting.plot_g_solns(g_array)
      
      # ---------------------------------------
      # Apply preliminary G solution
      g_to_apply = g_soln_hh.interpolate(num_dumps,dump_period=dump_period) 
      vis_hh = g_to_apply.apply(vis_hh)
            
      # plot data with G solutions applied:
      #if do_plots: plotting.plot_bp_data(vis_hh,plotavg=True)
      
      # ---------------------------------------
      # BP solution
      #   solve for bandpass
        
      # first average over all time
      ave_vis_hh = calprocs.wavg(vis_hh,flags_hh,weights_hh,axis=0)
      # then solve for BP    
      bp_soln_hh = CalSolution('B',calprocs.bp_fit(ave_vis_hh,antlist1,antlist2,bp0_hh,REFANT),
          np.ones(num_ants), 'inf', corrprod_lookup_hh)
      # update TM
      TM['BP'].append(bp_soln_hh.values)
      TM['BP_current'] = bp_soln_hh.values
      
      # ---------------------------------------
      # Apply BP solution 
      bp_to_apply = bp_soln_hh.interpolate(num_dumps) 
      vis_hh = bp_to_apply.apply(vis_hh)
      
      # plot data with K solutions applied:
      if do_plots: plotting.plot_bp_data(vis_hh,plotavg=True)
      # plot all data:
      #if do_plots: plotting.plot_bp_solns(bp_soln_hh.values)   
      
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
      k_to_apply = k_soln_hh.interpolate(num_dumps) 
      vis_hh = k_to_apply.apply(vis_hh, chans)
      bp_to_apply = bp_soln_hh.interpolate(num_dumps) 
      vis_hh = bp_to_apply.apply(vis_hh)
      
      # ---------------------------------------
      # Preliminary G solution
      # set up solution interval
      solint, dumps_per_solint = calprocs.solint_from_nominal(g_solint,dump_period,len(times))
      # first averge solution interval then channel
      ave_vis_hh, ave_flags_hh, ave_weights_hh, times_hh = calprocs.wavg_full_t(vis_hh,flags_hh,weights_hh,
         dumps_per_solint,axis=0,times=times)
      ave_vis_hh = calprocs.wavg(ave_vis_hh,ave_flags_hh,ave_weights_hh,axis=1)
      # solve for gains
      g_soln_hh = CalSolution('G', calprocs.g_fit_per_solint(ave_vis_hh,dumps_per_solint,antlist1,antlist2,g0,REFANT), 
         times_hh, solint, corrprod_lookup_hh)

      # plot the G solutions
      #if do_plots: plotting.plot_g_solns(g_array)
         
      # plot the G solutions
      if do_plots: plotting.plot_g_solns(g_array)
      
      # ---------------------------------------
      # RFI flagging
      rfi()
      
      # ---------------------------------------
      # Next G solution
      # set up solution interval: just solve for two intervals per G scan
      dumps_per_solint = num_dumps/2.0
      solint = dumps_per_solint*dump_period
      # first averge solution interval then channel
      ave_vis_hh, ave_flags_hh, ave_weights_hh, times_hh = calprocs.wavg_full_t(vis_hh,flags_hh,weights_hh,
         dumps_per_solint,axis=0,times=times)
      ave_vis_hh = calprocs.wavg(ave_vis_hh,ave_flags_hh,ave_weights_hh,axis=1)
      # solve for gains
      g_soln_hh = CalSolution('G', calprocs.g_fit_per_solint(ave_vis_hh,dumps_per_solint,antlist1,antlist2,g0,REFANT), 
         times_hh, solint, corrprod_lookup_hh)
       
      # plot the G solutions
      if do_plots: plotting.plot_g_solns(g_array)
      
      timing_file.write("Gain cal: %s \n" % (np.round(time()-t0,3),))
      t0 = time()
      
   if any( 'target' in k for k in taglist ):
      # ---------------------------------------
      # Apply K and BP solutions
      k_to_apply = k_soln_hh.interpolate(num_dumps) 
      vis_hh = k_to_apply.apply(vis_hh, chans)
      bp_to_apply = bp_soln_hh.interpolate(num_dumps) 
      vis_hh = bp_to_apply.apply(vis_hh)
      
      # ---------------------------------------
      # RFI flagging
      rfi()
      
      # ---------------------------------------
      # save calibrated data
      data_indices = np.squeeze(np.where(h5._time_keep))
      indi, indf = min(data_indices), max(data_indices)+1       
      h5._vis[indi:indf,BCHAN:ECHAN,hh_mask,0] = vis_hh.real
      h5._vis[indi:indf,BCHAN:ECHAN,hh_mask,1] = vis_hh.imag
      
      # ---------------------------------------
      timing_file.write("Source (cal application): %s \n" % (np.round(time()-t0,3),))
      t0 = time()
      

   print
   if scan_iter>NUM_ITERS: break
        

timing_file.close()
pickle.dump(TM, open('TM.pickle', 'wb'))


