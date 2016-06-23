
from . import plotting
from . import calprocs
from . import pipelineprocs as pp
from . import report
from .scan import Scan
from .rfi import threshold_avg_flagging

import numpy as np
import optparse
import sys
import copy

from . import lsm_dir

import pickle
import os

from time import time

import logging
logger = logging.getLogger(__name__)

class ThreadLoggingAdapter(logging.LoggerAdapter):
    """
    Logging adaptor which prepends a bracketed string to the log message.
    The string is passed via a 'connid' key.
    """
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['connid'], msg), kwargs

def rfi(s,thresholds,av_blocks,pipeline_logger):
    """
    Place holder for RFI detection algorithms to come.

    Inputs:
    -------
    s : Scan
        Scan to flag
    thresholds : list of float, shape(N)
        Tresholds to use for tN iterations of flagging
    av_blocks : list of int, shape(N-1, 2)
        List of block sizes to average over from the second iteration, in the
        form [time_block, channel_block]
    pipeline_logger : logger
        logger 
    """
    total_size = np.multiply.reduce(s.flags.shape)/100.
    pipeline_logger.info('     Start flags: {0:.3f}%'.format(np.sum(s.flags.view(np.bool))/total_size,))
    threshold_avg_flagging(s.vis,s.flags,thresholds,blocks=av_blocks,transform=np.abs)
    pipeline_logger.info('     New flags:   {0:.3f}%'.format(np.sum(s.flags.view(np.bool))/total_size,))

def get_tracks(data, ts):
    """
    Determines start and end indices of each track in the data buffer

    Inputs:
    -------
    data : data buffer, dictionary
    ts : telescope state, TelescopeState

    Returns:
    --------
    list of slices for each track in the buffer
    """
    max_indx = data['max_index'][0]

    activity_key = '{0}_activity'.format(ts.cal_refant,)
    activity = ts.get_range(activity_key,st=data['times'][0],et=data['times'][max_indx],include_previous=True)

    start_indx, stop_indx = [], []
    prev_state = ''
    for state, time in activity:
        nearest_time_indx = np.abs(time - data['times'][0:max_indx+1]).argmin()
        if 'track' in state:
            start_indx.append(nearest_time_indx)
            if 'track' in prev_state:
                stop_indx.append(nearest_time_indx-1)
        #    stop_indx.append(nearest_time_indx-1)
        if 'track' in prev_state:
            stop_indx.append(nearest_time_indx-1)
        prev_state = state

    # remove first slew time from stop indices
    if len(stop_indx) > 0:
        if stop_indx[0] == -1: stop_indx = stop_indx[1:]
    # add max index in buffer to stop indices of necessary
    if len(stop_indx) < len(start_indx): stop_indx.append(max_indx)

    return [slice(start, stop+1) for start, stop in zip(start_indx, stop_indx)]

def get_solns_to_apply(s,ts,sol_list,logger,time_range=[]):
    """
    For a given scan, extract and interpolate specified calibration solutions from TelescopeState

    Inputs:
    -------
    s : scan, Scan
    ts : telescope state, TelescopeState
    sol_list : list of calibration solutions to extract and interpolate, list of strings

    Returns:
    --------
    solns_to_apply : list of CalSolution solutions
    """
    solns_to_apply = []

    for X in sol_list:
        ts_solname = 'cal_product_{0}'.format(X,)
        try:
            # get most recent solution value
            sol, soltime = ts.get_range(ts_solname)[0]
            if X is not 'G':
                soln = calprocs.CalSolution(X, sol, soltime)
            else:
                # get G values for an hour range on either side of target scan
                t0, t1 = time_range
                gsols = ts.get_range(ts_solname,st=t0-2.*60.*60.,et=t1+2.*60.*60,return_format='recarray')
                solval, soltime = gsols['value'], gsols['time']
                soln = calprocs.CalSolution('G', solval, soltime)

            if len(soln.values) > 0: solns_to_apply.append(s.interpolate(soln))
            logger.info('{0} correction to be applied, shape {1}'.format(X,soln.values.shape))

        except KeyError:
            # TS doesn't yet contain 'X'
            logger.info('No {0} correction present'.format(X,))

    return solns_to_apply

def pipeline(data, ts, task_name='pipeline'):
    """
    Pipeline calibration

    Inputs:
    -------
    data : data buffer, dictionary
    ts : telescope state, TelescopeState
    task_name : name of pipeline task (used for logging), string

    Returns:
    --------
    list of slices for each target track in the buffer
    """

    # ----------------------------------------------------------
    # set up logging adapter for the pipeline thread/process
    pipeline_logger = ThreadLoggingAdapter(logger, {'connid': task_name})

    # ----------------------------------------------------------
    # set up timing file
    # at the moment this is re-made every scan! fix later!
    timing_file = 'timing.txt'
    #print timing_file
    #if os.path.isfile(timing_file): os.remove(timing_file)
    #timing_file = open("timing.txt", "w")

    # ----------------------------------------------------------
    # extract some some commonly used constants from the TS

    # solution intervals
    bp_solint = ts.cal_param_bp_solint #seconds
    k_solint = ts.cal_param_k_solint #seconds
    k_chan_sample = ts.cal_param_k_chan_sample
    g_solint = ts.cal_param_g_solint #seconds
    try:
        dump_period = ts.sdp_l0_int_time
    except:
        pipeline_logger.warning('Parameter sdp_l0_int_time not present in TS. Will be derived from data.')
        dump_period = data['times'][1] - data['times'][0]

    antlist = ts.cal_antlist
    n_ants = len(antlist)
    n_chans = ts.cbf_n_chans
    # refant index number in the antenna list
    refant_ind = antlist.index(ts.cal_refant)

    # set up parameters that are static during an observation, but only available when the first data starts to flow
    # list of antenna descriptions
    if not ts.has_key('cal_antlist_description'):
        description_list = [ts['{0}_observer'.format(ant,)] for ant in antlist]
        ts.add('cal_antlist_description',description_list,immutable=True)
    # channel frequencies
    if not ts.has_key('cal_channel_freqs'):
        sideband = ts['cal_sideband'] if ts.has_key('cal_sideband') else 1
        channel_freqs = ts.cbf_center_freq + sideband*(ts.cbf_bandwidth/n_chans)*(np.arange(n_chans) - n_chans / 2)
        ts.add('cal_channel_freqs',channel_freqs,immutable=True)

    # get names of activity and target TS keys, using TS reference antenna
    target_key = '{0}_target'.format(ts.cal_refant,)
    activity_key = '{0}_activity'.format(ts.cal_refant,)

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
    track_slices = get_tracks(data,ts)
    target_slices = []

    for scan_slice in reversed(track_slices):
        # start time, end time
        t0 = data['times'][scan_slice.start]
        t1 = data['times'][scan_slice.stop-1]

        # if we only have one or two timestamps in the scan, ignore it
        #  (a single timestamp can happen when there is no slew between tracks so we catch the first dump of the next track)
        n_times = scan_slice.stop - scan_slice.start
        if n_times < 3: continue

        # extract scan info from the TS
        # add the dump period here to account for scan start and activity change being closely timed
        scan_state = ts.get_range(activity_key,et=t0+dump_period)[0][0]
        #  target string contains: 'target name, tags, RA, DEC'
        target = ts.get_range(target_key,et=t0)[0][0]
        target_list = target.split(',')
        target_name = target_list[0]
        pipeline_logger.info('-----------------------------------')
        pipeline_logger.info('Target: {0}'.format(target_name,))
        pipeline_logger.info('   Timestamps:   {0}'.format(n_times,))

        # if there are no tags, don't process this scan
        if len(target_list) > 1:
            taglist = target_list[1].split()
        else:
            pipeline_logger.info('   Tags:   None')
            continue
        pipeline_logger.info('   Tags:   {0}'.format(taglist,))

        # ---------------------------------------
        # set up scan
        s = Scan(data, scan_slice, dump_period, n_ants, ts.cal_bls_lookup, target, chans=ts.cal_channel_freqs, logger=pipeline_logger)
        if s.xc_mask.size == 0:
            pipeline_logger.info('No XC data - no processing performed.')
            continue

        # Do we have a model for this source?
        model_key = 'cal_model_{0}'.format(target_name,)
        if model_key in ts.keys():
            s.model_raw_params = ts[model_key]
        else:
            model_params = pp.get_model(target_name, lsm_dir)
            if model_params is not None:
                s.model_raw_params =  model_params
                ts.add(model_key,model_params,immutable=True)
        pipeline_logger.debug('Model parameters for source {0}: {1}'.format(target_name, s.model_raw_params))

        # ---------------------------------------
        # initial RFI flagging
        pipeline_logger.info('   Preliminary flagging')
        rfi(s,[3.0,3.0,2.0,1.6],[[3,1],[3,5],[3,8]],pipeline_logger)

        run_t0 = time()

        # perform calibration as appropriate, from scan intent tags:

        # BEAMFORMER
        if any('bfcal' in k for k in taglist):
            # ---------------------------------------
            # K solution
            pipeline_logger.info('   Solving for K on beamformer calibrator {0}'.format(target_name,))
            k_soln = s.k_sol(refant_ind,ts.cal_param_k_bchan,ts.cal_param_k_echan)
            pipeline_logger.info('    - Saving K to Telescope State')
            ts.add(k_soln.ts_solname,k_soln.values,ts=k_soln.times)

            # ---------------------------------------
            # B solution
            pipeline_logger.info('   Solving for B on beamformer calibrator {0}'.format(target_name,))
            # get K solutions to apply and interpolate it to scan timestamps
            solns_to_apply = get_solns_to_apply(s,ts,['K'],pipeline_logger)
            b_soln = s.b_sol(bp0_h,refant_ind,pre_apply=solns_to_apply)
            pipeline_logger.info('    - Saving B to Telescope State')
            ts.add(b_soln.ts_solname,b_soln.values,ts=b_soln.times)

            # ---------------------------------------
            # G solution
            pipeline_logger.info('   Solving for G on beamformer calibrator {0}'.format(target_name,))
            # get B solutions to apply and interpolate them to scan timestamps along with K
            solns_to_apply.extend(get_solns_to_apply(s,ts,['B'],pipeline_logger))

            # use single solution interval
            dumps_per_solint = np.ceil(scan_slice.stop-scan_slice.start-1)
            g_solint = dumps_per_solint*dump_period
            g_soln = s.g_sol(g_solint,g0_h,refant_ind,ts.cal_param_g_bchan,ts.cal_param_g_echan,pre_apply=solns_to_apply)
            pipeline_logger.info('    - Saving G to Telescope State')
            # add gains to TS, iterating through solution times
            for v,t in zip(g_soln.values,g_soln.times):
                ts.add(g_soln.ts_solname,v,ts=t)

            # ---------------------------------------
            #timing_file.write("K cal:    %s \n" % (np.round(time()-run_t0,3),))
            run_t0 = time()

        # DELAY
        if any('delaycal' in k for k in taglist):
            # ---------------------------------------
            # preliminary G solution
            pipeline_logger.info('   Solving for preliminary G on delay calibrator {0}'.format(target_name,))
            # solve and interpolate to scan timestamps
            pre_g_soln = s.g_sol(k_solint,g0_h,refant_ind,ts.cal_param_k_bchan,ts.cal_param_k_echan)
            g_to_apply = s.interpolate(pre_g_soln)

            # ---------------------------------------
            # K solution
            pipeline_logger.info('   Solving for K on delay calibrator {0}'.format(target_name,))
            k_soln = s.k_sol(refant_ind,ts.cal_param_k_bchan,ts.cal_param_k_echan,k_chan_sample,pre_apply=[g_to_apply])

            # ---------------------------------------
            # update TS
            pipeline_logger.info('    - Saving K to Telescope State')
            ts.add(k_soln.ts_solname,k_soln.values,ts=k_soln.times)

            # ---------------------------------------
            #timing_file.write("K cal:    %s \n" % (np.round(time()-run_t0,3),))
            run_t0 = time()

        # DELAY POL OFFSET
        if any('polcal' in k for k in taglist):
            # ---------------------------------------
            # get K solutions to apply and interpolate them to scan timestamps
            pre_apply_solns = ['K','B']
            solns_to_apply = get_solns_to_apply(s,ts,pre_apply_solns,pipeline_logger)
            #solns_to_apply.append(g_to_apply)

            # ---------------------------------------
            # preliminary G solution
            pipeline_logger.info('   Solving for preliminary G on KCROSS calibrator {0}'.format(target_name,))
            # solve (pre-applying given solutions)
            pre_g_soln = s.g_sol(k_solint,g0_h,refant_ind,pre_apply=solns_to_apply)
            # interpolate to scan timestamps
            g_to_apply = s.interpolate(pre_g_soln)
            solns_to_apply.append(g_to_apply)

            # ---------------------------------------
            # KCROSS solution
            pipeline_logger.info('   Solving for KCROSS on delay calibrator {0}'.format(target_name,))
            kcross_soln = s.kcross_sol(ts.cal_param_k_bchan,ts.cal_param_k_echan,ts.cal_param_kcross_chanave,pre_apply=solns_to_apply)

            # ---------------------------------------
            # update TS
            pipeline_logger.info('    - Saving KCROSS to Telescope State')
            ts.add(kcross_soln.ts_solname,kcross_soln.values,ts=kcross_soln.times)

            # ---------------------------------------
            #timing_file.write("K cal:    %s \n" % (np.round(time()-run_t0,3),))
            run_t0 = time()

        # BANDPASS
        if any('bpcal' in k for k in taglist):
            # ---------------------------------------
            # get K solutions to apply and interpolate it to scan timestamps
            solns_to_apply = get_solns_to_apply(s,ts,['K'],pipeline_logger)

            # ---------------------------------------
            # Preliminary G solution
            pipeline_logger.info('   Solving for preliminary G on bandpass calibrator {0}'.format(target_name,))
            # solve and interpolate to scan timestamps
            pre_g_soln = s.g_sol(bp_solint,g0_h,refant_ind,pre_apply=solns_to_apply)
            g_to_apply = s.interpolate(pre_g_soln)

            # ---------------------------------------
            # B solution
            pipeline_logger.info('   Solving for B on bandpass calibrator {0}'.format(target_name,))
            solns_to_apply.append(g_to_apply)
            b_soln = s.b_sol(bp0_h,refant_ind,pre_apply=solns_to_apply)

            # ---------------------------------------
            # update TS
            pipeline_logger.info('    - Saving B to Telescope State')
            ts.add(b_soln.ts_solname,b_soln.values,ts=b_soln.times)

            # ---------------------------------------
            #timing_file.write("B cal:    %s \n" % (np.round(time()-run_t0,3),))
            run_t0 = time()

        # GAIN
        if any('gaincal' in k for k in taglist):
            # ---------------------------------------
            # get K and B solutions to apply and interpolate them to scan timestamps
            solns_to_apply = get_solns_to_apply(s,ts,['K','B'],pipeline_logger)

            # ---------------------------------------
            # G solution
            pipeline_logger.info('   Solving for G on gain calibrator {0}'.format(target_name,))
            # set up solution interval: just solve for two intervals per G scan (ignore ts g_solint for now)
            dumps_per_solint = np.ceil((scan_slice.stop-scan_slice.start-1)/2.0)
            g_solint = dumps_per_solint*dump_period
            g_soln = s.g_sol(g_solint,g0_h,refant_ind,ts.cal_param_g_bchan,ts.cal_param_g_echan,pre_apply=solns_to_apply)

            # ---------------------------------------
            # update TS
            pipeline_logger.info('    - Saving G to Telescope State')
            # add gains to TS, iterating through solution times
            for v,t in zip(g_soln.values,g_soln.times):
                ts.add(g_soln.ts_solname,v,ts=t)

            # ---------------------------------------
            #timing_file.write("G cal:    %s \n" % (np.round(time()-run_t0,3),))
            run_t0 = time()

        # TARGET
        if any('target' in k for k in taglist):
            # ---------------------------------------
            pipeline_logger.info('   Applying calibration solutions to target {0}:'.format(target_name,))

            # ---------------------------------------
            # get K, B and G solutions to apply and interpolate it to scan timestamps
            solns_to_apply = get_solns_to_apply(s,ts,['K','B','G'],pipeline_logger,time_range=[t0,t1])
            # apply solutions
            for soln in solns_to_apply:
                s.apply(soln, inplace=True)

            # accumulate list of target scans to be streamed to L1
            target_slices.append(scan_slice)

            # flag calibrated target
            pipeline_logger.info('   Flagging calibrated target {0}'.format(target_name,))
            rfi(s,[3.0,3.0,2.0],[[3,1],[5,8]],pipeline_logger)

    return target_slices
