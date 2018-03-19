import time
import logging
import threading

import numpy as np
import dask.array as da

from katdal.sensordata import TelstateSensorData, SensorCache
from katdal.categorical import CategoricalData
from katdal.h5datav3 import SENSOR_PROPS

from katsdpsigproc.rfi.twodflag import SumThresholdFlagger

from . import calprocs, calprocs_dask, solutions
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


def get_tracks(data, ts, parameters, dump_period):
    """Determine the start and end indices of each track segment in data buffer.

    Inputs
    ------
    data : dict
        Data buffer
    ts : :class:`katsdptelstate.TelescopeState`
        The telescope state associated with this pipeline
    parameters : dict
        Pipeline parameters
    dump_period : float
        Dump period in seconds

    Returns
    -------
    segments : list of slice objects
        List of slices indicating dumps associated with each track in buffer

    """
    # Collect all receptor activity sensors from telstate
    cache = {}
    for ant in parameters['antennas']:
        sensor_name = '{}_activity'.format(ant.name)
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


def get_noise_diode(telstate, ant_names, time_range=[]):
    """
    For a given timerange check if the noise diode is on for
    antennas in ant_names

    Inputs
    ------
    telstate : :class:`katsdptelstate.TelescopeState`
        telescope state
    ant_names : list of str
        names of antennas
    time_range : list of float
        timerange to check the noise diode is on for

    Returns
    -------
    nd_on : :class:`np.ndarray` of bool
        True for antennas with noise diode on during time_range, otherwise False
    """
    sub_band = telstate['sub_band']
    nd_key = '_dig_{0}_band_noise_diode'.format(sub_band)
    nd_during = [telstate.get_range('{0}{1}'.format(a, nd_key),
                 st=time_range[0], et=time_range[1], include_previous=True)
                 for a in ant_names]

    nd_on = [min(zip(*values)[0]) > 0 for values in nd_during]
    return np.asarray(nd_on)


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
        a public part of telstate).
    bchan,echan : int
        Channel range containing the data, relative to the channels held by
        this server. It must lie either entirely inside or entirely outside
        [0, n_chans).
    solver : callable
        Function to do the actual computation. It is passed the remaining
        arguments, and is also passed `bchan` and `echan` by keyword. It must
        return a :class:`~.CalSolution` or `~.CalSolutions`.
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
            assert isinstance(soln, (solutions.CalSolution, solutions.CalSolutions))
            values = soln.values
            if solution_store is not None:
                save_solution(telstate, telstate_key, solution_store, soln)
                values = None
            if isinstance(soln, solutions.CalSolution):
                info = ('CalSolution', soln.soltype, values, soln.time)
            else:
                info = ('CalSolutions', soln.soltype, values, soln.times)
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
        else:
            raise ValueError('Unknown info type {}'.format(info[0]))
        if solution_store is not None:
            # We don't pass telstate, because we got the value from telstate
            save_solution(None, None, solution_store, soln)
        return soln


def pipeline(data, ts, parameters, solution_stores, stream_name):
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

    n_ants = len(parameters['antennas'])
    n_pols = len(parameters['bls_pol_ordering'])
    # refant index number in the antenna list
    refant_ind = parameters['refant_index']

    # Set up flaggers
    calib_flagger, targ_flagger = init_flagger(parameters, dump_period)

    # get names of target TS key, using TS reference antenna
    target_key = '{0}_target'.format(parameters['refant'].name)

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
    track_slices = get_tracks(data, ts, parameters, dump_period)
    target_slices = []
    # initialise corrected data
    av_corr = {'targets': [], 'vis': [], 'flags': [], 'weights': [],
               'times': [], 'n_flags': [], 'timestamps': [],
               'auto_cross': [], 'auto_timestamps': []}

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
        logger.info('Target: {0}'.format(target_name))
        logger.info('  Timestamps: {0}'.format(n_times))
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
            s.rfi(calib_flagger, parameters['rfi_mask'])
            # TODO: setup separate flagger for cross-pols
            s.rfi(calib_flagger, parameters['rfi_mask'], cross_pol=True)

        # run_t0 = time.time()

        # perform calibration as appropriate, from scan intent tags:

        # BEAMFORMER
        if any('bfcal' in k for k in taglist):
            logger.info('Calibrator flagging, auto-correlations')
            s.rfi(calib_flagger, parameters['rfi_mask'], cross_pol=True, auto_ant=True)
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
            b_soln = s.b_sol(bp0_h, pre_apply=solns_to_apply)
            save_solution(ts, parameters['product_names']['B'], solution_stores['B'], b_soln)

            # ---------------------------------------
            # G solution
            logger.info('Solving for G on beamformer calibrator {0}'.format(target_name,))
            # get B solutions to apply and interpolate them to scan timestamps along with K
            solns_to_apply.append(s.interpolate(b_soln))

            # use single solution interval
            dumps_per_solint = scan_slice.stop - scan_slice.start
            g_solint = dumps_per_solint * dump_period
            g_soln = shared_solve(ts, parameters, solution_stores['G'],
                                  parameters['g_bchan'], parameters['g_echan'],
                                  s.g_sol, g_solint, g0_h, pre_apply=solns_to_apply)

            # ----------------------------------------
            # KCROSS solution
            logger.info('Checking if the noise diode was fired')
            ant_names = [a.name for a in s.antennas]
            nd_on = get_noise_diode(ts, ant_names, [t0, t1])
            if any(nd_on):
                logger.info("Noise diode was fired,"
                            " solving for KCROSS_DIODE on beamformer calibrator %s", target_name)
                if n_pols < 4:
                    logger.info("Can't solve for KCROSS_DIODE without four polarisation products")
                elif s.ac_mask.size == 0:
                    logger.info("No AC data, can't solve for KCROSS_DIODE without AC data")
                else:
                    solns_to_apply.append(s.interpolate(g_soln))
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
                av_vis = vis
                av_flags = s.auto_ant.tf.cross_pol.flags
                av_weights = s.auto_ant.tf.cross_pol.weights
                if av_vis.shape[1] > 1024:
                    av_vis, av_flags, av_weights = calprocs_dask.wavg_full_f(
                        av_vis, av_flags, av_weights, chanav=av_vis.shape[1] // 1024)
                av_vis = calprocs_dask.wavg(av_vis, av_flags, av_weights)
                av_corr['auto_cross'].insert(0, av_vis.compute())
                av_corr['auto_timestamps'].insert(0, np.average(s.timestamps))
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
                             s.kcross_sol,
                             chan_ave=parameters['kcross_chanave'], pre_apply=solns_to_apply)

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
            b_soln = s.b_sol(bp0_h, pre_apply=solns_to_apply)
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

        # TARGET
        if any('target' in k for k in taglist):
            # ---------------------------------------
            logger.info('Applying calibration solutions to target {0}:'.format(target_name,))

            # ---------------------------------------
            # get K, B and G solutions to apply and interpolate it to scan timestamps
            solns_to_apply = get_solns_to_apply(s, solution_stores, ['K', 'B', 'G'],
                                                time_range=[t0, t1])
            s.apply_inplace(solns_to_apply)

            # accumulate list of target scans to be streamed to L1
            target_slices.append(scan_slice)

            # flag calibrated target
            logger.info('Flagging calibrated target {0}'.format(target_name,))
            rfi_mask = parameters['rfi_mask']
            s.rfi(targ_flagger, rfi_mask)
            # TODO: setup separate flagger for cross-pols
            s.rfi(targ_flagger, rfi_mask, cross=True)

        vis = s.cross_ant.tf.auto_pol.vis
        target_type = 'target'
        # apply solutions to calibrators
        if not any('target' in k for k in taglist):
            target_type = 'calibrator'
            solns_to_apply = get_solns_to_apply(s, solution_stores,
                                                ['K', 'B', 'G'], time_range=[t0, t1])
            logger.info('Applying solutions to calibrator %s:', target_name)
            for soln in solns_to_apply:
                vis = s.apply(soln, vis)

        # average the corrected data
        av_vis = vis
        av_flags = s.cross_ant.tf.auto_pol.flags
        av_weights = s.cross_ant.tf.auto_pol.weights

        logger.info('Averaging corrected data for {0} {1}:'.format(target_type, target_name,))
        if av_vis.shape[1] > 1024:
            av_vis, av_flags, av_weights = calprocs_dask.wavg_full_f(av_vis,
                                                                     av_flags,
                                                                     av_weights,
                                                                     chanav=vis.shape[1] // 1024)

        av_vis, av_flags, av_weights = calprocs_dask.wavg_full(av_vis,
                                                               av_flags,
                                                               av_weights)

        # collect corrected data and calibrator target list to send to report writer
        av_vis, av_flags, av_weights = da.compute(av_vis, av_flags, av_weights)
        sum_flags = da.sum(calprocs.asbool(s.cross_ant.tf.auto_pol.flags), axis=0).compute()
        av_corr['targets'].insert(0, target)
        av_corr['vis'].insert(0, av_vis)
        av_corr['flags'].insert(0, av_flags)
        av_corr['weights'].insert(0, av_weights)
        av_corr['times'].insert(0, np.average(s.timestamps))
        av_corr['n_flags'].insert(0, sum_flags)
        av_corr['timestamps'].insert(0, s.timestamps)

    return target_slices, av_corr
