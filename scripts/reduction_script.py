

from pyrap.tables import table
import numpy as np
#import matplotlib.pylab as plt
from scipy.stats import nanmean
from math import pi

import pickle
import os.path

import h5py
import katdal

from katcal import calprocs
from katcal import calplots
from katcal.calsolution import CalSolution

PLOT_COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] 

BCHAN = 200
ECHAN = 800

REFANT = 4
SOLINT = 10.0 #seconds

NUM_ITERS = 100

# ----------------------------------------------------------
# H5 file to use for simulation
#   simulation of data and Temelcope Model (TM)

h5filename = '1345897247.h5' #1345967709.h5' #1337360375.h5' #1373441586.h5'
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
num_chans = 600
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
   print "scan %3d (%4d samples) loaded. Target: '%s'." % (scan_ind, num_dumps, target.name)
   if scan_state != 'track':
      print "    scan %3d (%4d samples) skipped '%s' - not a track" % (scan_ind, num_dumps, scan_state)
      continue
   if num_dumps < 2:
      print "    scan %3d (%4d samples) skipped - too short" % (scan_ind, num_dumps)
      continue

   # -------------------------------------------
   # data has shape(num_times, num_chans, num_baselines)
   vis = h5.vis[:,BCHAN:ECHAN,:]
   times = h5.timestamps[:]-h5.timestamps[:][0]
   time_offset = h5.timestamps[:][0]
   flags = h5.flags()[:,BCHAN:ECHAN,:]
   weights = h5.weights()[:,BCHAN:ECHAN,:]

   # -------------------------------------------
   # data preparation

   # extract hh and vv
   vis_hh = np.array([vis_row[:,hh_mask] for vis_row in vis])
   flags_hh = np.array([flag_row[:,hh_mask] for flag_row in flags])
   weights_hh = np.array([weight_row[:,hh_mask] for weight_row in weights])

   if 'bpcal' in target.tags:
      # plot all data:
      #calplots.plot_bp_data(vis_hh,plotavg=True)
      
      # ---------------------------------------
      # Preliminary G solution
      
      # first average over time, for each solution interval
      dumps_per_solint = int(SOLINT/np.round(dump_period,3))
      solint_increments = range(0,num_dumps,dumps_per_solint)
      g_array = np.empty([len(solint_increments),num_ants],dtype=np.complex)
      t_array = np.empty([len(solint_increments)],dtype=np.complex)

      # solve for G for each solint
      for i,ts in enumerate(solint_increments):
         ave_vis_hh, ave_flags_hh, ave_weights_hh = calprocs.wavg_full(vis_hh[ts:ts+dumps_per_solint],
            flags_hh[ts:ts+dumps_per_solint],weights_hh[ts:ts+dumps_per_solint],axis=0)
         ave_vis_hh = calprocs.wavg(ave_vis_hh,ave_flags_hh,ave_weights_hh,axis=0)

         g_array[i] = calprocs.g_fit(ave_vis_hh,g0,antlist1,antlist2,REFANT)
         t_array[i] = times[ts]
         
      g_soln_hh = CalSolution('G', g_array, t_array, SOLINT, corrprod_lookup_hh)
         
      # plot the G solutions
      #calplots.plot_g_solns(g_array)
      
      # ---------------------------------------
      # Apply preliminary G solution
      
      g_to_apply = g_soln_hh.interpolate(dump_period,num_dumps) #  np.repeat(g_array,dumps_per_solint,axis=0)[0:num_dumps]
      #import matplotlib.pylab as plt
      #plt.plot(np.abs(g_to_apply.values))
      #plt.show()
      vis_hh = g_to_apply.apply(vis_hh)
      
      #for ti in range(num_dumps):
      #   for cp in range(len(corrprod_lookup_hh)):
      #      vis_hh[ti,:,cp] /= gains_to_apply[ti,corrprod_lookup_hh[cp][0]]*(gains_to_apply[ti,corrprod_lookup_hh[cp][1]].conj())
            
      # plot all data:
      #calplots.plot_bp_data(vis_hh,plotavg=True)
            
      # ---------------------------------------
      # BP solution
        
      # set up data for BP solution
      # average over time, including flags:
      #    ave data shape num_chans,num_bl)       
      ave_vis_hh = calprocs.wavg(vis_hh,flags_hh,weights_hh,axis=0)
      # solve for BP      
      bp_soln_hh = CalSolution('B',calprocs.bp_fit(ave_vis_hh,antlist1,antlist2,bp0_hh,REFANT),
          np.ones(num_ants), 'inf', corrprod_lookup_hh)
      # update TM
      TM['BP'].append(bp_soln_hh.values)
      TM['BP_current'] = bp_soln_hh.values

      # plot all data:
      calplots.plot_bp_solns(bp_soln_hh.values)
      
      # ---------------------------------------
      # K solution
      #   solve for delay 
      k_soln_hh = CalSolution('K',calprocs.k_fit(ave_vis_hh,antlist1,antlist2,chans,k0_hh,bp0_hh,REFANT,chan_sample=10),
          np.ones(num_ants), 'inf', corrprod_lookup_hh)
      # update TM
      TM['K'].append(k_soln_hh.values)
      TM['K_current'] = k_soln_hh.values

      #import matplotlib.pylab as plt
      #plt.plot(360.*(np.array([k*chans for k in k_soln_hh]).T)/(2.*np.pi),'-*')
      #plt.show()
      #pri
      
      # ---------------------------------------
      # K solution
         
      # apply the solns
      for ti in range(num_dumps):
         for c in chans:
            for cp in range(len(corrprod_lookup_hh)):
               gains_to_apply = np.exp(1.0j*k_soln_hh.values*c) 
               #print '888', k_solns.shape
               #plt.plot(gains_to_apply)
               #plt.show()
               vis_hh[ti,c,cp] /= gains_to_apply[corrprod_lookup_hh[cp][0]]*(gains_to_apply[corrprod_lookup_hh[cp][1]].conj())
               
      #plt.show()
      
      # plot all data:
      calplots.plot_bp_data(vis_hh,plotavg=True)
      
      
                     





   if scan_iter>NUM_ITERS: break
        



pickle.dump(TM, open('TM.pickle', 'wb'))


