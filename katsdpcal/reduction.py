import time
import logging
import threading
import katpoint

import numpy as np

from collections import defaultdict
from katdal.sensordata import TelstateSensorData, SensorCache
from katdal.h5datav3 import SENSOR_PROPS

from katsdpsigproc.rfi.twodflag import SumThresholdFlagger

from . import solutions
from . import pipelineprocs as pp
from .scan import Scan
from . import lsm_dir

logger = logging.getLogger(__name__)


def init_flagger(parameters, dump_period):
    """Set up SumThresholdFlagger objects for targets
    and calibrators.

    Parameters
    ----------
    parameters : dict
        Pipeline parameters
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
    rfi_windows_freq = np.array(parameters['rfi_windows_freq'], dtype=np.int)
    spike_width_time = parameters['rfi_spike_width_time'] / dump_period
    calib_flagger = SumThresholdFlagger(outlier_nsigma=parameters['rfi_calib_nsigma'],
                                        windows_freq=rfi_windows_freq,
                                        spike_width_time=spike_width_time,
                                        spike_width_freq=parameters['rfi_calib_spike_width_freq'],
                                        average_freq=parameters['rfi_average_freq'],
                                        freq_extend=parameters['rfi_extend_freq'],
                                        freq_chunks=parameters['rfi_freq_chunks'])
    targ_flagger = SumThresholdFlagger(outlier_nsigma=parameters['rfi_targ_nsigma'],
                                       windows_freq=rfi_windows_freq,
                                       spike_width_time=spike_width_time,
                                       spike_width_freq=parameters['rfi_targ_spike_width_freq'],
                                       average_freq=parameters['rfi_average_freq'],
                                       freq_extend=parameters['rfi_extend_freq'],
                                       freq_chunks=parameters['rfi_freq_chunks'])
    return calib_flagger, targ_flagger


def get_tracks(data, telstate, dump_period):
    """Determine the start and end indices of each track segment in data buffer.

    Inputs
    ------
    data : dict
        Data buffer
    telstate : :class:`katsdptelstate.TelescopeState`
        The telescope state associated with this pipeline
    dump_period : float
        Dump period in seconds

    Returns
    -------
    segments : list of slice objects
        List of slices indicating dumps associated with each track in buffer
    """
    sensor_name = 'obs_activity'
    cache = {sensor_name: TelstateSensorData(telstate, sensor_name)}
    sensors = SensorCache(cache, data['times'], dump_period, props=SENSOR_PROPS)
    activity = sensors.get(sensor_name)
    return [segment for (segment, a) in activity.segments() if a == 'track']


def check_noise_diode(telstate, ant_names, time_range):
    """Check if the noise diode is on at all per antenna within the time range.

    Inputs
    ------
    telstate : :class:`katsdptelstate.TelescopeState`
        Telescope state
    ant_names : sequence of str
        Antenna names
    time_range : sequence of 2 floats
        Time range as [start_time, end_time]

    Returns
    -------
    nd_on : :class:`np.ndarray` of bool, shape (len(ant_names),)
        True for each antenna with noise diode on at some time in `time_range`
    """
    sub_band = telstate['sub_band']
    nd_key = 'dig_{}_band_noise_diode'.format(sub_band)
    nd_on = np.full(len(ant_names), False)
    for n, ant in enumerate(ant_names):
        try:
            value_times = telstate.get_range('{}_{}'.format(ant, nd_key),
                                             st=time_range[0], et=time_range[1],
                                             include_previous=True)
        except KeyError:
            pass
        else:
            # Set to True if any noise diode value is positive, per antenna
            values = zip(*value_times)[0]
            nd_on[n] = max(values) > 0
    return nd_on


def get_solns_to_apply(s, solution_stores, sol_list, time_range=[]):
    """
    For a given scan, extract and interpolate specified calibration solutions
    from the solution stores.

    Inputs
    ------
    s : :class:`Scan`
        scan
    solution_stores : dict of :class:`solutions.SolutionStore`
        stored solutions
    sol_list : list of str
        calibration solutions to extract and interpolate

    Returns
    -------
    solns_to_apply : list of :class:`~.CalSolution`
        solutions
    """
    solns_to_apply = []

    for X in sol_list:
        if X != 'G':
            # get most recent solution value
            soln = solution_stores[X].latest
        else:
            # get G values for a two hour range on either side of target scan
            t0, t1 = time_range
            soln = solution_stores[X].get_range(t0 - 2. * 60. * 60., t1 + 2. * 60. * 60)

        if soln is not None and len(soln.values) > 0:
            solns_to_apply.append(s.interpolate(soln))
            logger.info("Loaded solution '%s' from solution store", soln)
        else:
            logger.info("No '%s' solutions found in solution store", X)

    return solns_to_apply


# For real use it doesn't need to be thread-local, but the unit test runs
# several servers in the same process.
_shared_solve_seq = threading.local()


def save_solution(telstate, key, solution_store, soln):
    """Write a calibration solution to telescope state and the solution store.

    The `soln` may be either a :class:`CalSolution` or a :class:`CalSolutions`.
    The `telstate` may be ``None`` to save only to the solution store.
    """
    def save_one(soln):
        if telstate is not None:
            telstate.add(key, soln.values, ts=soln.time)
        solution_store.add(soln)

    logger.info("  - Saving solution '%s' to Telescope State", soln)
    assert isinstance(soln, (solutions.CalSolution, solutions.CalSolutions))
    if isinstance(soln, solutions.CalSolution):
        save_one(soln)
    else:
        for v, t in zip(soln.values, soln.times):
            save_one(solutions.CalSolution(soln.soltype, v, t))


def shared_solve(telstate, parameters, solution_store, bchan, echan,
                 solver, *args, **kwargs):
    """Run a solver on one of the cal nodes.

    The one containing the relevant data actually does the calculation and
    stores it in telstate, while the others simply wait for the result.

    Parameters
    ----------
    telstate : :class:`katsdptelstate.TelescopeState`
        Telescope state in which the solution is stored
    parameters : dict
        Pipeline parameters
    solution_store : :class:`CalSolutionStore`-like
        Store in which to place the solution, in addition to the `telstate`. If it
        is ``None``, the solution is returned but not placed in any store (nor in
        a public part of telstate). Only :class:`~.CalSolution` or `~.CalSolutions` solutions can
        be stored. Other solution types can only be returned.
    bchan,echan : int
        Channel range containing the data, relative to the channels held by
        this server. It must lie either entirely inside or entirely outside
        [0, n_chans).
    solver : callable
        Function to do the actual computation. It is passed the remaining
        arguments, and is also passed `bchan` and `echan` by keyword. It must
        return a :class:`~.CalSolution`, :class:`~.CalSolutions`, int or np.ndarray.
    _seq : int, optional
        If specified, it is used as the sequence number instead of using the
        global counter. This is intended strictly for unit testing.
    """
    # telstate doesn't quite provide the sort of barrier primitives we'd like,
    # but we can kludge it. Each server maintains a sequence number of calls to
    # this function (which MUST be kept synchronised). Metadata needed to fetch the
    # actual result is inserted into an immutable key, whose name includes the
    # sequence number. If `name` is not given, the metadata contains the actual
    # values.
    def add_info(info):
        telstate.add(shared_key, info, immutable=True)
        logger.debug('Added shared key %s', shared_key)

    if '_seq' in kwargs:
        seq = kwargs.pop('_seq')
    else:
        try:
            seq = _shared_solve_seq.value
        except AttributeError:
            # First use
            seq = 0
        _shared_solve_seq.value = seq + 1
    shared_key = 'shared_solve_{}'.format(seq)

    if solution_store is not None:
        telstate_key = parameters['product_names'][solution_store.soltype]
    else:
        telstate_key = None

    n_chans = len(parameters['channel_freqs'])
    if 0 <= bchan and echan <= n_chans:
        kwargs['bchan'] = bchan
        kwargs['echan'] = echan
        try:
            soln = solver(*args, **kwargs)
            if isinstance(soln, (solutions.CalSolution, solutions.CalSolutions)):
                values = soln.values
                if solution_store is not None:
                    save_solution(telstate, telstate_key, solution_store, soln)
                    values = None
                if isinstance(soln, solutions.CalSolution):
                    info = ('CalSolution', soln.soltype, values, soln.time)
                else:
                    info = ('CalSolutions', soln.soltype, values, soln.times)
            elif isinstance(soln, (int, np.ndarray)):
                info = ('soln', soln)
                if solution_store is not None:
                    logger.warn('Solution is not of type :class:`~.CalSolution` or `~.CalSolutions`'
                                ' and won\'t be stored in solution store')
        except Exception as error:
            add_info(('Exception', error))
            raise
        else:
            add_info(info)
            return soln
    else:
        assert echan <= 0 or bchan >= n_chans, 'partial channel overlap'
        logger.debug('Waiting for shared key %s', shared_key)
        telstate.wait_key(shared_key)
        info = telstate[shared_key]
        logger.debug('Found shared key %s', shared_key)
        if info[0] == 'Exception':
            raise info[1]
        elif info[0] == 'CalSolution':
            soltype, values, time = info[1:]
            if values is None:
                saved = telstate.get_range(telstate_key, st=time, et=time, include_end=True)
                if len(saved) != 1:
                    print(len(telstate.get_range(telstate_key, st=0)))
                    raise ValueError('Expected exactly one solution with timestamp {}, found {}'
                                     .format(time, len(saved)))
                values = saved[0][0]
            soln = solutions.CalSolution(soltype, values, time)
        elif info[0] == 'CalSolutions':
            soltype, values, times = info[1:]
            if values is None:
                # Reassemble from telstate
                saved = telstate.get_range(telstate_key, st=times[0], et=times[-1],
                                           include_end=True)
                if not saved:
                    raise ValueError('Key {} not found in time interval [{}, {}]'
                                     .format(telstate_key, times[0], times[-1]))
                # Split (value, ts) pairs into separate lists
                values, saved_times = zip(*saved)
                if list(saved_times) != list(times):
                    raise ValueError('Timestamps for {} did not match expected values'
                                     .format(telstate_key))
                values = np.stack(values)
            soln = solutions.CalSolutions(soltype, values, times)
        elif info[0] == 'soln':
            soln = info[1]
        else:
            raise ValueError('Unknown info type {}'.format(info[0]))
        if solution_store is not None:
            if isinstance(soln, (solutions.CalSolution, solutions.CalSolutions)):
                # We don't pass telstate, because we got the value from telstate
                save_solution(None, None, solution_store, soln)
            else:
                logger.warn('Solution is not of type :class:`~.CalSolution` or `~.CalSolutions`'
                            ' and won\'t be stored in solution store')
        return soln


def pipeline(data, ts, parameters, solution_stores, stream_name, sensors=None):
    """
    Pipeline calibration

    Inputs
    ------
    data : dict
        Dictionary of data buffers. Keys `vis`, `flags` and `weights` reference
        :class:`dask.Arrays`, while `times` references a numpy array.
    ts : :class:`katsdptelstate.TelescopeState`
        Telescope state, scoped to the cal namespace within the capture block
    parameters : dict
        The pipeline parameters
    solution_stores : dict of :class:`~.CalSolutionStore`-like
        Solution stores for the capture block, indexed by solution type
    stream_name : str
        Name of the L0 data stream
    sensors : dict, optional
        Sensors available in the calling parent

    Returns
    -------
    slices : list of slice
        slices for each target track in the buffer
    av_corr : dict
        Dictionary containing time and frequency averaged, calibrated data.
        Keys `targets`, `vis`, `flags`, `weights`, `times`, `n_flags`,
        `timestamps` all reference numpy arrays.
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

    telstate_l0 = ts.view(stream_name)
    # solution intervals
    bp_solint = parameters['bp_solint']  # seconds
    k_solint = parameters['k_solint']  # seconds
    k_chan_sample = parameters['k_chan_sample']
    g_solint = parameters['g_solint']  # seconds
    try:
        dump_period = telstate_l0['int_time']
    except KeyError:
        logger.warning(
            'Parameter %s_int_time not present in TS. Will be derived from data.', stream_name)
        dump_period = data['times'][1] - data['times'][0]

    n_pols = len(parameters['bls_pol_ordering'])
    # refant index number in the antenna list
    refant_ind = parameters['refant_index']

    # Set up flaggers
    calib_flagger, targ_flagger = init_flagger(parameters, dump_period)

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
    av_corr = defaultdict(list)
    for scan_slice in reversed(track_slices):
        # start time, end time
        t0 = data['times'][scan_slice.start]
        t1 = data['times'][scan_slice.stop - 1]
        n_times = scan_slice.stop - scan_slice.start

        #  target string contains: 'target name, tags, RA, DEC'
        target_str = ts.get_range('cbf_target', et=t0)[0][0]
        target = katpoint.Target(target_str)
        target_name = target.name
        logger.info('-----------------------------------')
        logger.info('Target: {0}'.format(target_name))
        logger.info('  Timestamps: {0}'.format(n_times))
        logger.info('  Time:       {0} - {1}'.format(
            time.strftime("%H:%M:%S", time.gmtime(t0)),
            time.strftime("%H:%M:%S", time.gmtime(t1))))

        # get tags, ignore the first tag which is the body type tag
        taglist = target.tags[1:]
        # if there are no tags, don't process this scan
        if not taglist:
            logger.info('  Tags:   None')
            continue
        logger.info('  Tags:       {0}'.format(taglist,))
        # ---------------------------------------
        # set up scan
        s = Scan(data, scan_slice, dump_period,
                 parameters['bls_lookup'], target,
                 chans=parameters['channel_freqs'],
                 ants=parameters['antennas'],
                 refant=refant_ind,
                 array_position=parameters['array_position'], logger=logger)
        if s.xc_mask.size == 0:
            logger.info('No XC data - no processing performed.')
            continue

        # Do we have a model for this source?
        model_key = 'model_{0}'.format(target_name)
        try:
            model_params = ts[model_key]
        except KeyError:
            model_params, model_file = pp.get_model(target_name, lsm_dir)
            if model_params is not None:
                s.add_model(model_params)
                ts.add(model_key, model_params, immutable=True)
                logger.info('   Model file: {0}'.format(model_file))
        else:
            s.add_model(model_params)
        logger.debug('Model parameters for source {0}: {1}'.format(
            target_name, s.model_raw_params))

        # ---------------------------------------
        # Calibrator RFI flagging
        if any(k.endswith('cal') for k in taglist):
            logger.info('Calibrator flagging')
            s.rfi(calib_flagger, parameters['rfi_mask'], sensors=sensors)

            # Set a reference antenna if one isn't already set
            if s.refant is None:
                best_refant_index = shared_solve(ts, parameters, None,
                                                 parameters['k_bchan'], parameters['k_echan'],
                                                 s.refant_find)
                parameters['refant_index'] = best_refant_index
                parameters['refant'] = parameters['antennas'][best_refant_index]
                logger.info('Reference antenna set to %s', parameters['refant'].name)
                ts.add('refant', parameters['refant'], immutable=True)
                s.refant = best_refant_index

        # run_t0 = time.time()

        # perform calibration as appropriate, from scan intent tags:

        # BEAMFORMER
        if any('bfcal' in k for k in taglist):
            logger.info('Calibrator flagging, auto-correlations')
            s.rfi(calib_flagger, parameters['rfi_mask'], auto_ant=True, sensors=sensors)
            # ---------------------------------------
            # K solution
            logger.info('Solving for K on beamformer calibrator %s', target_name)
            k_soln = shared_solve(ts, parameters, solution_stores['K'],
                                  parameters['k_bchan'], parameters['k_echan'], s.k_sol)

            # ---------------------------------------
            # B solution
            logger.info('Solving for B on beamformer calibrator %s', target_name)
            # get K solutions to apply and interpolate it to scan timestamps
            solns_to_apply = [s.interpolate(k_soln)]
            b_soln = s.b_sol(bp0_h, pre_apply=solns_to_apply, bp_flagger=calib_flagger)
            b_norm_factor = shared_solve(ts, parameters, None,
                                         parameters['g_bchan'], parameters['g_echan'],
                                         s.b_norm, b_soln)
            b_soln.values *= b_norm_factor
            save_solution(ts, parameters['product_names']['B'], solution_stores['B'], b_soln)

            # ---------------------------------------
            # G solution
            logger.info('Solving for G on beamformer calibrator {0}'.format(target_name,))
            # get B solutions to apply and interpolate them to scan timestamps along with K
            solns_to_apply.append(s.interpolate(b_soln))

            # use single solution interval
            dumps_per_solint = scan_slice.stop - scan_slice.start
            g_solint = dumps_per_solint * dump_period
            shared_solve(ts, parameters, solution_stores['G'],
                         parameters['g_bchan'], parameters['g_echan'],
                         s.g_sol, g_solint, g0_h, pre_apply=solns_to_apply)

            # ----------------------------------------
            # KCROSS solution
            logger.info('Checking if the noise diode was fired')
            ant_names = [a.name for a in s.antennas]
            nd_on = check_noise_diode(ts, ant_names, [t0, t1])
            if any(nd_on):
                logger.info("Noise diode was fired,"
                            " solving for KCROSS_DIODE on beamformer calibrator %s", target_name)
                if n_pols < 4:
                    logger.info("Can't solve for KCROSS_DIODE without four polarisation products")
                elif s.ac_mask.size == 0:
                    logger.info("No AC data, can't solve for KCROSS_DIODE without AC data")
                else:
                    solns_to_apply = [s.interpolate(k_soln)]
                    kcross_soln = shared_solve(ts, parameters, solution_stores['KCROSS_DIODE'],
                                               parameters['k_bchan'], parameters['k_echan'],
                                               s.kcross_sol, pre_apply=solns_to_apply,
                                               nd=nd_on, auto_ant=True)

                    # apply solutions and put corrected data into the av_corr dictionary
                    solns_to_apply.append(s.interpolate(kcross_soln))
                    vis = s.auto_ant.tf.cross_pol.vis
                    for soln in solns_to_apply:
                        vis = s.apply(soln, vis, cross_pol=True)
                    logger.info('Averaging corrected auto-corr data for %s:', target_name)

                    data = (vis, s.auto_ant.tf.cross_pol.flags, s.auto_ant.tf.cross_pol.weights)
                    s.summarize(av_corr, 'auto_cross', data, nchans=1024)
            else:
                logger.info("Noise diode wasn't fired, no KCROSS_DIODE solution")

        # DELAY
        if any('delaycal' in k for k in taglist):
            # ---------------------------------------
            # preliminary G solution
            logger.info('Solving for preliminary G on delay calibrator %s', target_name)
            # solve and interpolate to scan timestamps
            pre_g_soln = shared_solve(ts, parameters, None,
                                      parameters['k_bchan'], parameters['k_echan'],
                                      s.g_sol, k_solint, g0_h)
            g_to_apply = s.interpolate(pre_g_soln)

            # ---------------------------------------
            # K solution
            logger.info('Solving for K on delay calibrator %s', target_name)
            shared_solve(ts, parameters, solution_stores['K'],
                         parameters['k_bchan'], parameters['k_echan'],
                         s.k_sol, chan_sample=k_chan_sample, pre_apply=[g_to_apply])

        # DELAY POL OFFSET
        if any('polcal' in k for k in taglist):
            if n_pols < 4:
                logger.info('Cant solve for KCROSS without four polarisation products')
            else:
                # ---------------------------------------
                # get K solutions to apply and interpolate them to scan timestamps
                pre_apply_solns = ['K', 'B']
                solns_to_apply = get_solns_to_apply(s, solution_stores, pre_apply_solns)
                # solns_to_apply.append(g_to_apply)

                # ---------------------------------------
                # preliminary G solution
                logger.info(
                    'Solving for preliminary G on KCROSS calibrator {0}'.format(target_name,))
                # solve (pre-applying given solutions)
                pre_g_soln = shared_solve(ts, parameters, None,
                                          parameters['k_bchan'], parameters['k_echan'],
                                          s.g_sol, k_solint, g0_h, pre_apply=solns_to_apply)
                # interpolate to scan timestamps
                g_to_apply = s.interpolate(pre_g_soln)
                solns_to_apply.append(g_to_apply)

                # ---------------------------------------
                # KCROSS solution
                logger.info('Solving for KCROSS on cross-hand delay calibrator %s', target_name)
                shared_solve(ts, parameters, solution_stores['KCROSS'],
                             parameters['k_bchan'], parameters['k_echan'],
                             s.kcross_sol, chan_ave=parameters['kcross_chanave'],
                             pre_apply=solns_to_apply)

        # BANDPASS
        if any('bpcal' in k for k in taglist):
            # ---------------------------------------
            # get K solutions to apply and interpolate it to scan timestamps
            solns_to_apply = get_solns_to_apply(s, solution_stores, ['K'])

            # ---------------------------------------
            # Preliminary G solution
            logger.info('Solving for preliminary G on bandpass calibrator %s', target_name)
            # solve and interpolate to scan timestamps
            pre_g_soln = shared_solve(ts, parameters, None,
                                      parameters['g_bchan'], parameters['g_echan'],
                                      s.g_sol, bp_solint, g0_h, pre_apply=solns_to_apply)
            g_to_apply = s.interpolate(pre_g_soln)

            # ---------------------------------------
            # B solution
            logger.info('Solving for B on bandpass calibrator %s', target_name)
            solns_to_apply.append(g_to_apply)
            b_soln = s.b_sol(bp0_h, pre_apply=solns_to_apply, bp_flagger=calib_flagger)
            b_norm_factor = shared_solve(ts, parameters, None,
                                         parameters['g_bchan'], parameters['g_echan'],
                                         s.b_norm, b_soln)
            b_soln.values *= b_norm_factor
            save_solution(ts, parameters['product_names']['B'], solution_stores['B'], b_soln)

        # GAIN
        if any('gaincal' in k for k in taglist):
            # ---------------------------------------
            # get K and B solutions to apply and interpolate them to scan timestamps
            solns_to_apply = get_solns_to_apply(s, solution_stores, ['K', 'B'])

            # ---------------------------------------
            # G solution
            logger.info('Solving for G on gain calibrator %s', target_name)
            # set up solution interval: just solve for two intervals per G scan
            # (ignore ts g_solint for now)
            dumps_per_solint = np.ceil((scan_slice.stop - scan_slice.start - 1) / 2.0)
            g_solint = dumps_per_solint * dump_period
            shared_solve(ts, parameters, solution_stores['G'],
                         parameters['g_bchan'], parameters['g_echan'],
                         s.g_sol, g_solint, g0_h, pre_apply=solns_to_apply)

        # Apply calibration
        cal_tags = ['gaincal', 'target', 'bfcal', 'bpcal', 'delaycal']
        if any(k in cal_tags for k in taglist):
            # ---------------------------------------
            logger.info('Applying calibration solutions to %s:', target_name)

            # ---------------------------------------
            # get K, B and G solutions to apply and interpolate it to scan timestamps
            solns_to_apply = get_solns_to_apply(s, solution_stores, ['K', 'B', 'G'],
                                                time_range=[t0, t1])
            s.apply_inplace(solns_to_apply)

            # TARGET
            if 'target' in taglist:
                # accumulate list of target scans to be streamed to L1
                target_slices.append(scan_slice)

                # flag calibrated target
                logger.info('Flagging calibrated target {0}'.format(target_name,))
                rfi_mask = parameters['rfi_mask']
                s.rfi(targ_flagger, rfi_mask, sensors=sensors)

            # summarize corrected data for data with cal tags
            logger.info('Averaging corrected data for %s:', target_name)

            av_corr['targets'].insert(0, (target, np.average(s.timestamps)))
            av_corr['timestamps'].insert(0, (s.timestamps, np.average(s.timestamps)))
            s.summarize_flags(av_corr)

            # summarize gain-calibrated targets
            gaintag = ['gaincal', 'target', 'bfcal']
            if any(k in gaintag for k in taglist):
                s.summarize_full(av_corr, target_name + '_g_spec', nchans=1024)
                s.summarize(av_corr, target_name + '_g_bls')
                if not any('target' in k for k in taglist):
                    s.summarize(av_corr, 'g_phase', avg_ant=True)
            # summarize non-gain calibrated targets
            else:
                s.summarize(av_corr, target_name + '_nog_spec', nchans=1024, refant_only=True)

    return target_slices, av_corr
