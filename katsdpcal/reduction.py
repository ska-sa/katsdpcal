import time
import logging

import numpy as np
import dask.array as da

from katdal.sensordata import TelstateSensorData, SensorCache
from katdal.categorical import CategoricalData
from katdal.h5datav3 import SENSOR_PROPS

from katsdpsigproc.rfi.twodflag import SumThresholdFlagger

from . import calprocs
from . import calprocs_dask
from . import pipelineprocs as pp
from .scan import Scan
from . import lsm_dir

logger = logging.getLogger(__name__)


def init_flagger(ts, dump_period):
    """Set up SumThresholdFlagger objects for targets
    and calibrators.

    Parameters
    ----------
    ts : :class:`katsdptelstate.TelescopeState`
        The telescope state associated with this pipeline
    dump_period : float
        The dump period in seconds

    Returns
    -------
    calib_flagger : :class:`SumThresholdFlagger`
        A SumThresholdFlagger object for use with calibrators
    targ_flagger : :class:`SumThresholdFlagger`
        A SumThresholdFlagger object for use with targets
    """

    # Make windows a integer array
    rfi_windows_freq = np.array(ts.cal_param_rfi_windows_freq.split(','), dtype=np.int)
    spike_width_time = ts.cal_param_rfi_spike_width_time / dump_period
    calib_flagger = SumThresholdFlagger(outlier_nsigma=ts.cal_param_rfi_calib_nsigma,
                                        windows_freq=rfi_windows_freq,
                                        spike_width_time=spike_width_time,
                                        spike_width_freq=ts.cal_param_rfi_calib_spike_width_freq,
                                        average_freq=ts.cal_param_rfi_average_freq,
                                        freq_extend=ts.cal_param_rfi_extend_freq)
    targ_flagger = SumThresholdFlagger(outlier_nsigma=ts.cal_param_rfi_targ_nsigma,
                                       windows_freq=rfi_windows_freq,
                                       spike_width_time=spike_width_time,
                                       spike_width_freq=ts.cal_param_rfi_targ_spike_width_freq,
                                       average_freq=ts.cal_param_rfi_average_freq,
                                       freq_extend=ts.cal_param_rfi_extend_freq)
    return calib_flagger, targ_flagger


def get_tracks(data, ts, dump_period):
    """Determine the start and end indices of each track segment in data buffer.

    Inputs
    ------
    data : dict
        Data buffer
    ts : :class:`katsdptelstate.TelescopeState`
        The telescope state associated with this pipeline
    dump_period : float
        Dump period in seconds

    Returns
    -------
    segments : list of slice objects
        List of slices indicating dumps associated with each track in buffer

    """
    # Collect all receptor activity sensors from telstate
    cache = {}
    for ant in ts.cal_antlist:
        sensor_name = '{}_activity'.format(ant)
        cache[sensor_name] = TelstateSensorData(ts, sensor_name)
    num_dumps = data['times'].shape[0]
    timestamps = data['times']
    sensors = SensorCache(cache, timestamps, dump_period, props=SENSOR_PROPS)
    # Interpolate onto data timestamps and find dumps where all receptors track
    tracking = np.ones(num_dumps, dtype=bool)
    for activity in sensors:
        tracking &= (np.array(sensors[activity]) == 'track')
    # Convert sequence of flags into segments and return the ones that are True
    all_tracking = CategoricalData(tracking, range(num_dumps + 1))
    all_tracking.remove_repeats()
    return [segment for (segment, track) in all_tracking.segments() if track]


def get_solns_to_apply(s, ts, sol_list, time_range=[]):
    """
    For a given scan, extract and interpolate specified calibration solutions
    from TelescopeState.

    Inputs
    ------
    s : :class:`Scan`
        scan
    ts : :class:`katsdptelstate.TelescopeState`
        telescope state
    sol_list : list of str
        calibration solutions to extract and interpolate

    Returns
    -------
    solns_to_apply : list of :class:`CalSolution`
        solutions
    """
    solns_to_apply = []

    for X in sol_list:
        ts_solname = 'cal_product_{0}'.format(X,)
        try:
            # get most recent solution value
            sol, soltime = ts.get_range(ts_solname)[0]
            if X != 'G':
                soln = calprocs.CalSolution(X, sol, soltime)
            else:
                # get G values for an hour range on either side of target scan
                t0, t1 = time_range
                gsols = ts.get_range(ts_solname, st=t0 - 2. * 60. * 60., et=t1 + 2. * 60. * 60,
                                     return_format='recarray')
                solval, soltime = gsols['value'], gsols['time']
                soln = calprocs.CalSolution('G', solval, soltime)

            if len(soln.values) > 0:
                solns_to_apply.append(s.interpolate(soln))
            logger.info("Loaded solution '{}' from Telescope State".format(soln))

        except KeyError:
            # TS doesn't yet contain 'X'
            logger.info('No {} solutions found in Telescope State'.format(X))

    return solns_to_apply


def init_ts_params(ts, stream_name):
    """
    Initialise telescope state parameteters
    This is necessary to add telescope state parameters derived from
    information available only after data has started flowing.

    Inputs
    ------
    ts : :class:`katsdptelstate.TelescopeState`
        telescope state
    stream_name : str
        name of L0 data stream
    """
    description_list = [ts['{0}_observer'.format(ant,)] for ant in ts.cal_antlist]
    # list of antenna descriptions
    if 'cal_antlist_description' not in ts:
        ts.add('cal_antlist_description', description_list, immutable=True)

    # array reference position
    if 'cal_array_position' not in ts:
        # take lat-long-alt value from first antenna in antenna list as the array reference position
        ts.add('cal_array_position',
               'array_position, ' + ','.join(description_list[0].split(',')[1:-1]))

    # channel frequencies
    if 'cal_channel_freqs' not in ts:
        n_chans = ts[stream_name + '_n_chans']
        center_freq = ts[stream_name + '_center_freq']
        bandwidth = ts[stream_name + '_bandwidth']
        sideband = ts.get('cal_sideband', 1)
        channel_freqs = center_freq \
            + sideband * (bandwidth / n_chans) * (np.arange(n_chans) - n_chans / 2)
        ts.add('cal_channel_freqs', channel_freqs, immutable=True)


def pipeline(data, ts, stream_name):
    """
    Pipeline calibration

    Inputs
    ------
    data : dict
        Dictionary of data buffers. Keys `vis`, `flags` and `weights` reference
        :class:`dask.Arrays`, while `times` references a numpy array.
    ts : :class:`katsdptelstate.TelescopeState`
        Telescope state
    stream_name : str
        Name of the L0 data stream

    Returns
    -------
    slices : list of slice
        slices for each target track in the buffer
    av_corr : dict
        Dictionary containing time and frequency averaged, calibrated data. Keys `targets`, `vis`, 
    `flags`, `weights`, `times`, `ntimes` all reference numpy arrays.
    """

    # ----------------------------------------------------------
    # set up timing file
    # at the moment this is re-made every scan! fix later!
    # timing_file = 'timing.txt'
    # print timing_file
    # if os.path.isfile(timing_file): os.remove(timing_file)
    # timing_file = open("timing.txt", "w")

    # ----------------------------------------------------------
    # extract some some commonly used constants from the TS

    # solution intervals
    bp_solint = ts.cal_param_bp_solint  # seconds
    k_solint = ts.cal_param_k_solint  # seconds
    k_chan_sample = ts.cal_param_k_chan_sample
    g_solint = ts.cal_param_g_solint  # seconds
    try:
        dump_period = ts[stream_name + '_int_time']
    except KeyError:
        logger.warning(
            'Parameter %s_int_time not present in TS. Will be derived from data.', stream_name)
        dump_period = data['times'][1] - data['times'][0]

    antlist = ts.cal_antlist
    n_ants = len(antlist)
    pols = set()
    for a, b in ts[stream_name + '_bls_ordering']:
        pols.add(a[-1] + b[-1])
    n_pols = len(pols)
    # refant index number in the antenna list
    refant_ind = antlist.index(ts.cal_refant)

    # set up parameters that are static during an observation, but only
    # available when the first data starts to flow.
    # TODO: more of this is static now and could be dealt with at startup.
    init_ts_params(ts, stream_name)

    # Set up flaggers
    calib_flagger, targ_flagger = init_flagger(ts, dump_period)

    # get names of target TS key, using TS reference antenna
    target_key = '{0}_target'.format(ts.cal_refant)

    # ----------------------------------------------------------
    # set initial values for fits
    bp0_h = None
    g0_h = None

    # ----------------------------------------------------------
    # iterate through the track scans accumulated into the data buffer
    #    first extract track scan indices from the buffer
    #    iterate backwards in time through the scans,
    #    for the case where a gains need to be calculated from a gain scan
    #    after a target scan, for application to the target scan
    track_slices = get_tracks(data, ts, dump_period)
    target_slices = []
    # initialise corrected data
    av_corr = {'targets': [], 'vis': [], 'flags': [], 'weights': [],
               'times': [], 'n_flags': [], 't_stamps': []}

    for scan_slice in reversed(track_slices):
        # start time, end time
        t0 = data['times'][scan_slice.start]
        t1 = data['times'][scan_slice.stop - 1]
        n_times = scan_slice.stop - scan_slice.start

        #  target string contains: 'target name, tags, RA, DEC'
        target = ts.get_range(target_key, et=t0)[0][0]
        target_list = target.split(',')
        target_name = target_list[0]
        logger.info('-----------------------------------')
        logger.info('Target: {0}'.format(target_name,))
        logger.info('  Timestamps: {0}'.format(n_times,))
        logger.info('  Time:       {0} - {1}'.format(
            time.strftime("%H:%M:%S", time.gmtime(t0)), time.strftime("%H:%M:%S", time.gmtime(t1))))

        # if there are no tags, don't process this scan
        if len(target_list) > 1:
            taglist = target_list[1].split()
        else:
            logger.info('  Tags:   None')
            continue
        logger.info('  Tags:       {0}'.format(taglist,))

        # ---------------------------------------
        # set up scan
        s = Scan(data, scan_slice, dump_period, n_ants, n_pols, ts.cal_bls_lookup, target,
                 chans=ts.cal_channel_freqs, ants=ts.cal_antlist_description,
                 refant=refant_ind, array_position=ts.cal_array_position, logger=logger)
        if s.xc_mask.size == 0:
            logger.info('No XC data - no processing performed.')
            continue

        # Do we have a model for this source?
        model_key = 'cal_model_{0}'.format(target_name,)
        if model_key in ts.keys():
            s.add_model(ts[model_key])
        else:
            model_params, model_file = pp.get_model(target_name, lsm_dir)
            if model_params is not None:
                s.add_model(model_params)
                ts.add(model_key, model_params, immutable=True)
                logger.info('   Model file: {0}'.format(model_file,))
        logger.debug('Model parameters for source {0}: {1}'.format(
            target_name, s.model_raw_params))

        # ---------------------------------------
        # Calibrator RFI flagging
        if any(k.endswith('cal') for k in taglist):
            logger.info('Calibrator flagging')
            s.rfi(calib_flagger, ts.get('cal_rfi_mask'))
            # TODO: setup separate flagger for cross-pols
            s.rfi(calib_flagger, ts.get('cal_rfi_mask'), cross=True)

        # run_t0 = time.time()

        # perform calibration as appropriate, from scan intent tags:

        # BEAMFORMER
        if any('bfcal' in k for k in taglist):
            # ---------------------------------------
            # K solution
            logger.info('Solving for K on beamformer calibrator {0}'.format(target_name,))
            k_soln = s.k_sol(ts.cal_param_k_bchan, ts.cal_param_k_echan)
            logger.info("  - Saving solution '{}' to Telescope State".format(k_soln))
            ts.add(k_soln.ts_solname, k_soln.values, ts=k_soln.times)

            # ---------------------------------------
            # B solution
            logger.info('Solving for B on beamformer calibrator {0}'.format(target_name,))
            # get K solutions to apply and interpolate it to scan timestamps
            solns_to_apply = get_solns_to_apply(s, ts, ['K'])
            b_soln = s.b_sol(bp0_h, pre_apply=solns_to_apply)
            logger.info("  - Saving solution '{}' to Telescope State".format(b_soln))
            ts.add(b_soln.ts_solname, b_soln.values, ts=b_soln.times)

            # ---------------------------------------
            # G solution
            logger.info('Solving for G on beamformer calibrator {0}'.format(target_name,))
            # get B solutions to apply and interpolate them to scan timestamps along with K
            solns_to_apply.extend(get_solns_to_apply(s, ts, ['B']))

            # use single solution interval
            dumps_per_solint = scan_slice.stop - scan_slice.start
            g_solint = dumps_per_solint * dump_period
            g_soln = s.g_sol(g_solint, g0_h, ts.cal_param_g_bchan, ts.cal_param_g_echan,
                             pre_apply=solns_to_apply)
            logger.info("  - Saving solution '{}' to Telescope State".format(g_soln))
            # add gains to TS, iterating through solution times
            for v, t in zip(g_soln.values, g_soln.times):
                ts.add(g_soln.ts_solname, v, ts=t)

            # ---------------------------------------
            # timing_file.write("K cal:    %s \n" % (np.round(time.time()-run_t0,3),))
            # run_t0 = time.time()

        # DELAY
        if any('delaycal' in k for k in taglist):
            # ---------------------------------------
            # preliminary G solution
            logger.info('Solving for preliminary G on delay calibrator {0}'.format(
                target_name,))
            # solve and interpolate to scan timestamps
            pre_g_soln = s.g_sol(k_solint, g0_h, ts.cal_param_k_bchan, ts.cal_param_k_echan)
            g_to_apply = s.interpolate(pre_g_soln)

            # ---------------------------------------
            # K solution
            logger.info('Solving for K on delay calibrator {0}'.format(target_name,))
            k_soln = s.k_sol(ts.cal_param_k_bchan, ts.cal_param_k_echan, k_chan_sample,
                             pre_apply=[g_to_apply])

            # ---------------------------------------
            # update TS
            logger.info("  - Saving solution '{}' to Telescope State".format(k_soln))
            ts.add(k_soln.ts_solname, k_soln.values, ts=k_soln.times)

            # ---------------------------------------
            # timing_file.write("K cal:    %s \n" % (np.round(time.time()-run_t0,3),))
            # run_t0 = time.time()

        # DELAY POL OFFSET
        if any('polcal' in k for k in taglist):
            if n_pols < 4:
                logger.info('Cant solve for KCROSS without four polarisation products')
            else:
                # ---------------------------------------
                # get K solutions to apply and interpolate them to scan timestamps
                pre_apply_solns = ['K', 'B']
                solns_to_apply = get_solns_to_apply(s, ts, pre_apply_solns)
                # solns_to_apply.append(g_to_apply)

                # ---------------------------------------
                # preliminary G solution
                logger.info(
                    'Solving for preliminary G on KCROSS calibrator {0}'.format(target_name,))
                # solve (pre-applying given solutions)
                pre_g_soln = s.g_sol(k_solint, g0_h, pre_apply=solns_to_apply)
                # interpolate to scan timestamps
                g_to_apply = s.interpolate(pre_g_soln)
                solns_to_apply.append(g_to_apply)

                # ---------------------------------------
                # KCROSS solution
                logger.info(
                    'Solving for KCROSS on cross-hand delay calibrator {0}'.format(target_name,))
                kcross_soln = s.kcross_sol(ts.cal_param_k_bchan, ts.cal_param_k_echan,
                                           ts.cal_param_kcross_chanave, pre_apply=solns_to_apply)

                # ---------------------------------------
                # update TS
                logger.info(
                    "  - Saving solution '{}' to Telescope State".format(kcross_soln))
                ts.add(kcross_soln.ts_solname, kcross_soln.values, ts=kcross_soln.times)

                # ---------------------------------------
                # timing_file.write("K cal:    %s \n" % (np.round(time.time()-run_t0,3),))
                # run_t0 = time.time()

        # BANDPASS
        if any('bpcal' in k for k in taglist):
            # ---------------------------------------
            # get K solutions to apply and interpolate it to scan timestamps
            solns_to_apply = get_solns_to_apply(s, ts, ['K'])

            # ---------------------------------------
            # Preliminary G solution
            logger.info(
                'Solving for preliminary G on bandpass calibrator {0}'.format(target_name,))
            # solve and interpolate to scan timestamps
            pre_g_soln = s.g_sol(bp_solint, g0_h, pre_apply=solns_to_apply)
            g_to_apply = s.interpolate(pre_g_soln)

            # ---------------------------------------
            # B solution
            logger.info('Solving for B on bandpass calibrator {0}'.format(target_name,))
            solns_to_apply.append(g_to_apply)
            b_soln = s.b_sol(bp0_h, pre_apply=solns_to_apply)

            # ---------------------------------------
            # update TS
            logger.info("  - Saving solution '{}' to Telescope State".format(b_soln))
            ts.add(b_soln.ts_solname, b_soln.values, ts=b_soln.times)

            # ---------------------------------------
            # timing_file.write("B cal:    %s \n" % (np.round(time.time()-run_t0,3),))
            # run_t0 = time.time()

        # GAIN
        if any('gaincal' in k for k in taglist):
            # ---------------------------------------
            # get K and B solutions to apply and interpolate them to scan timestamps
            solns_to_apply = get_solns_to_apply(s, ts, ['K', 'B'])

            # ---------------------------------------
            # G solution
            logger.info('Solving for G on gain calibrator {0}'.format(target_name,))
            # set up solution interval: just solve for two intervals per G scan
            # (ignore ts g_solint for now)
            dumps_per_solint = np.ceil((scan_slice.stop - scan_slice.start - 1) / 2.0)
            g_solint = dumps_per_solint * dump_period
            g_soln = s.g_sol(g_solint, g0_h, ts.cal_param_g_bchan, ts.cal_param_g_echan,
                             pre_apply=solns_to_apply)

            # ---------------------------------------
            # update TS
            logger.info("  - Saving solution '{}' to Telescope State".format(g_soln))
            # add gains to TS, iterating through solution times
            for v, t in zip(g_soln.values, g_soln.times):
                ts.add(g_soln.ts_solname, v, ts=t)

            # ---------------------------------------
            # timing_file.write("G cal:    %s \n" % (np.round(time.time()-run_t0,3),))
            # run_t0 = time.time()

        # TARGET
        if any('target' in k for k in taglist):
            # ---------------------------------------
            logger.info(
                'Applying calibration solutions to target {0}:'.format(target_name,))

            # ---------------------------------------
            # get K, B and G solutions to apply and interpolate it to scan timestamps
            solns_to_apply = get_solns_to_apply(s, ts, ['K', 'B', 'G'],
                                                time_range=[t0, t1])
            s.apply_inplace(solns_to_apply)

            # accumulate list of target scans to be streamed to L1
            target_slices.append(scan_slice)

            # flag calibrated target
            logger.info('Flagging calibrated target {0}'.format(target_name,))
            rfi_mask = ts.get('cal_rfi_mask')
            s.rfi(targ_flagger, rfi_mask)
            # TODO: setup separate flagger for cross-pols
            s.rfi(targ_flagger, rfi_mask, cross=True)

        # apply solutions and average the corrected data
        solns_to_apply = get_solns_to_apply(s, ts, ['K', 'B', 'G'], time_range=[t0, t1])
        logger.info('Applying solutions to {0}:'.format(target_name,))
        vis = s.tf.auto.vis
        for soln in solns_to_apply:
            vis = s.apply(soln, vis)
        
        av_vis, av_flags, av_weights = da.compute(vis, s.tf.auto.flags, s.tf.auto.weights)        

        if vis.shape[1]>1024:
            av_vis, av_flags, av_weights = calprocs_dask.wavg_full_f(av_vis,
                                                            av_flags,
                                                            av_weights,
                                                            chanav=vis.shape[1] // 1024)       

        av_vis, av_flags, av_weights = calprocs_dask.wavg_full(av_vis,
                                                               av_flags,
                                                               av_weights)

        # collect corrected data and calibrator target list to send to report writer
        av_vis, av_flags, av_weights = da.compute(av_vis, av_flags, av_weights) 
        av_corr['targets'].append(target)
        av_corr['vis'].append(av_vis)
        av_corr['flags'].append(av_flags)
        av_corr['weights'].append(av_weights)
        av_corr['times'].append(np.average(s.timestamps))
        av_corr['n_flags'].append(da.sum(calprocs.asbool(s.tf.auto.flags),axis=0).compute())
        av_corr['t_stamps'].append(s.timestamps)

    return target_slices, av_corr
