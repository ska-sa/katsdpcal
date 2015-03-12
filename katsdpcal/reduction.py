
import numpy as np
import optparse
import sys
import copy

import pickle
import os

from katcal import plotting
from katcal import calprocs
from katcal.simulator import SimData
from katcal import report
from katcal.scan import Scan

from katcal.calprocs import CalSolution

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
    #print 'Some sort of RFI flagging?!'
    pass

def get_solns_to_apply(s,ts,sol_list,target_name,pipeline_logger,time_range=[]):
    solns_to_apply = []
    
    for X in sol_list: 
        ts_solname = 'cal_product_{0}'.format(X,)   
        try:
            # get most recent solution value
            sol, soltime = ts.get_range(ts_solname)
            if X is not 'G':
                soln = CalSolution(X, sol, soltime)
            else:    
                # get G values for an hour range on either side of target scan
                t0, t1 = time_range
                gsols = ts.get_range(ts_solname,st=t0-60.*60.,et=t1+60.*60,return_format='recarray')
                solval, soltime = gsols['value'], gsols['time']
                soln = CalSolution('G', solval, soltime)
                
            solns_to_apply.append(s.interpolate(soln))   
            pipeline_logger.info('Apply {0} solution to {1}'.format(X,target_name))   
            
        except KeyError:
            # TS doesn't yet contain 'X'
            pipeline_logger.info('No {0} correction present'.format(X,))
            
    return solns_to_apply 

def pipeline(data, ts, task_name):
    """
    Pipeline calibration
    """

    # ----------------------------------------------------------    
    # set up logging adapter for the pipeline thread/process
    pipeline_logger = ThreadLoggingAdapter(logger, {'connid': task_name})
    
    # ----------------------------------------------------------
    # set up timing file
    # at the moment this is re-made every scan! fix later!
    timing_file = 'timing.txt'
    #print timing_file
    if os.path.isfile(timing_file): os.remove(timing_file)
    timing_file = open("timing.txt", "w")

    # ----------------------------------------------------------
    # extract some some commonly used constants from the TS

    #per_scan_plots = ts.cal_per_scan_plots
    #closing_plots = ts.cal_closing_plots

    # solution intervals
    bp_solint = ts.cal_bp_solint #seconds
    k_solint = ts.cal_k_solint #seconds
    k_chan_sample = ts.cal_k_chan_sample
    g_solint = ts.cal_g_solint #seconds
    
    dump_period = ts.sdp_l0_int_time
    
    antlist = ts.antenna_mask.split(',')
    # refant index number in the antenna list
    refant_ind = antlist.index(ts.cal_refant)
    
    # get names of activity and target TS keys, using TS reference antenna
    target_key = '{0}_target'.format(ts.cal_refant,)
    activity_key = '{0}_activity'.format(ts.cal_refant,)
    
    # plots per scan
    #per_scan_plots = True

    # ----------------------------------------------------------
    # set initial values for fits
    bp0_h = None
    k0_h = None
    g0_h = None
    target_h = None

    # ----------------------------------------------------------
    # iterate through the track scans accumulated into the data buffer
    #    first extract track scan indices from the buffer
    #    iterate backwards in time through the scans, 
    #    for the case where a gains need to be calculated from a gain scan after a target scan,
    #    for application to the target scan
    track_starts = data['track_start_indices'][0:np.where(data['track_start_indices']==-1)[0]]
    print 'track starts: ', track_starts

    for i in range(len(track_starts)-1,0,-1):
        # start and end indices for this track in the data buffer
        ti0 = track_starts[i-1]
        ti1 = track_starts[i]-1

        # start time, end time
        t0 = data['times'][ti0]
        t1 = data['times'][ti1]

        # extract scan info from the TS
        #  target string contains: 'target name, tags, RA, DEC'
        target = ts.get_previous(target_key,t0,dt=10.)[0]
        scan_state = ts.get_previous(activity_key,t0,dt=10.)[0]
        taglist = target.split(',')[1].split()    
        # fudge for now to add delay cal tag to bpcals
        if 'bpcal' in taglist: taglist.append('delaycal')
        
        target_name = target.split(',')[0]
        pipeline_logger.info('Target: {0}'.format(target_name,))
        pipeline_logger.info('Tags:   {0}'.format(taglist,))
        
        # set up scan
        s = Scan(data, ti0, ti1, dump_period, ts.cbf_n_ants, ts.cal_bls_lookup)

        # initial RFI flagging
        pipeline_logger.info('Preliminary flagging')
        rfi()

        run_t0 = time()
        # perform calibration as appropriate, from scan intent tags           
        if any('delaycal' in k for k in taglist):
            # ---------------------------------------
            # preliminary G solution
            pipeline_logger.info('Solving for preliminary G on delay calibrator {0}'.format(target_name,))
            # solve and interpolate to scan timestamps
            pre_g_soln = s.g_sol(k_solint,g0_h,refant_ind)
            g_to_apply = s.interpolate(pre_g_soln)
            
            # ---------------------------------------
            # K solution
            pipeline_logger.info('Solving for K on delay calibrator {0}'.format(target_name,))
            k_soln = s.k_sol(k_chan_sample,k0_h,bp0_h,refant_ind,pre_apply=[g_to_apply])
            
            # ---------------------------------------
            # update TS
            pipeline_logger.info('Saving K to Telescope State')
            ts.add(k_soln.ts_solname,k_soln.values,ts=k_soln.times) 
            
            # ---------------------------------------
            timing_file.write("K cal:    %s \n" % (np.round(time()-run_t0,3),))
            run_t0 = time()     
            
        if any('bpcal' in k for k in taglist):  
            # ---------------------------------------
            # get K solutions to apply and interpolate it to scan timestamps
            solns_to_apply = get_solns_to_apply(s,ts,['K'],target_name,pipeline_logger)
        
            # ---------------------------------------
            # Preliminary G solution
            pipeline_logger.info('Solving for preliminary G on bandpass calibrator {0}'.format(target.split(',')[0],))
            # solve and interpolate to scan timestamps
            pre_g_soln = s.g_sol(bp_solint,g0_h,refant_ind,pre_apply=solns_to_apply)
            g_to_apply = s.interpolate(pre_g_soln)
            
            # ---------------------------------------
            # B solution
            pipeline_logger.info('Solving for B on bandpass calibrator {0}'.format(target.split(',')[0],))
            solns_to_apply.append(g_to_apply)
            b_soln = s.b_sol(bp0_h,refant_ind,pre_apply=solns_to_apply)

            # ---------------------------------------
            # update TS
            pipeline_logger.info('Saving B to Telescope State')
            ts.add(b_soln.ts_solname,b_soln.values,ts=b_soln.times) 
            
            # ---------------------------------------
            timing_file.write("B cal:    %s \n" % (np.round(time()-run_t0,3),))
            run_t0 = time()  
            
        if any('gaincal' in k for k in taglist):
            # ---------------------------------------
            # get K and B solutions to apply and interpolate them to scan timestamps
            solns_to_apply = get_solns_to_apply(s,ts,['K','B'],target_name,pipeline_logger)
        
            # ---------------------------------------
            # G solution
            pipeline_logger.info('Solving for G on gain calibrator {0}'.format(target.split(',')[0],))
            # set up solution interval: just solve for two intervals per G scan (ignore ts g_solint for now)
            dumps_per_solint = np.ceil((ti1-ti0)/2.0)
            g_solint = dumps_per_solint*dump_period            
            g_soln = s.g_sol(g_solint,g0_h,refant_ind,pre_apply=solns_to_apply)
            
            # ---------------------------------------
            # update TS
            pipeline_logger.info('Saving G to Telescope State')
            # add gains to TS, iterating through solution times 
            for v,t in zip(g_soln.values,g_soln.times): 
                ts.add(g_soln.ts_solname,v,ts=t) 
                
            # debug check
            #g_to_apply = s.interpolate(g_soln)
            #solns_to_apply.append(g_to_apply)
            #for soln in solns_to_apply:    
            #    s.apply(soln, inplace=True)
            #plotting.plot_bp_data(s.vis)            

            # ---------------------------------------
            timing_file.write("G cal:    %s \n" % (np.round(time()-run_t0,3),))
            run_t0 = time() 
            
        if any('target' in k for k in taglist):
            # ---------------------------------------
            # get K, B and G solutions to apply and interpolate it to scan timestamps
            solns_to_apply = get_solns_to_apply(s,ts,['K','B','G'],target_name,pipeline_logger,time_range=[t0,t1])            
            # apply solutions  
            for soln in solns_to_apply:    
                s.apply(soln, inplace=True)
                
            # return calibrated target data to be streamed to L1
            return s.vis, s.flags, s.weights, s.times
            
            
            
