import time
import logging
import threading
import pickle
import katpoint
from numbers import Integral

import numpy as np
import katsdptelstate
from katsdpcalproc import pointing

from collections import defaultdict
from katdal.sensordata import TelstateSensorGetter, SensorCache
from katdal.h5datav3 import SENSOR_PROPS

from katsdpcalproc import solutions
from katsdpcalproc.calprocs import interpolate_soln
from katsdpsigproc.rfi.twodflag import SumThresholdFlagger

from . import pipelineprocs as pp
from .scan import Scan
from . import lsm_dir

logger = logging.getLogger(__name__)


def init_flagger(parameters, dump_period):
    """Set up SumThresholdFlagger objects for targets and calibrators.

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
    rfi_windows_freq = np.array(parameters['rfi_windows_freq'], dtype=int)
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


def slot_slices_to_time_slices(slices):
    """Convert a list of slot slices to time slices"""
    start = 0
    for s in slices:
        step = s.stop - s.start
        yield slice(start, start+step)
        start = start+step


def slots_slices(slots):
    """Compresses a list of slot positions to a list of ranges (given as slices).

    This is a generator that yields the slices

    Example
    -------
    >>> list(_slots_slices([2, 3, 4, 6, 7, 8, 0, 1]))
    [slice(2, 5, None), slice(6, 9, None), slice(0, 2, None)]
    """
    start = None
    end = None
    for slot in slots:
        if end is not None and slot != end:
            yield slice(start, end)
            start = end = None
        if start is None:
            start = slot
        end = slot + 1
    if end is not None:
        yield slice(start, end)


def store_flags(slots, scan_slice, flag_array, scan, auto_ant=False):
    """Store scan flags in the shared memory flag buffer

    Inputs
    ------
    slots: list of slots
        buffer slots currently being processed by the pipeline
    scan_slice: slice
        slice of the current scan
    flag_array: :class:`np.ndarray` of uint8, shape (ntimes, nchans, npols, nbls)
        shared memory array containing flag data
    scan: :class:`Scan`
        scan corresponding to the scan_slice
    cross: bool
        cross-correlations or auto-correlations
    """
    # ******* Fancy indexing will not work here, as it returns a copy, not a view *******
    # all selections on the shared flag array must be done using slices
    if auto_ant:
        mask = scan.ac_mask
        corr = scan.auto_ant
    else:
        mask = scan.xc_mask
        corr = scan.cross_ant

    mask_slice = slice(np.where(mask)[0][0], np.where(mask)[0][-1]+1)
    slot_slices = list(slots_slices(slots[scan_slice]))
    time_slices = list(slot_slices_to_time_slices(slot_slices))
    for slots, times in zip(slot_slices, time_slices):
        flag_array[slots, :, :2, mask_slice] = corr.tf.auto_pol.flags[times].compute()
        flag_array[slots, :, 2:, mask_slice] = corr.tf.cross_pol.flags[times].compute()


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
    cache = {sensor_name: TelstateSensorGetter(telstate, sensor_name)}
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
            values = list(zip(*value_times))[0]
            nd_on[n] = max(values) > 0
    return nd_on


def get_solns_to_apply(s, solution_stores, sol_list, time_range=[], G_target=None):
    """Extract and interpolate specified cal solutions for a given scan.

    Inputs
    ------
    s : :class:`Scan`
        scan
    solution_stores : dict of :class:`solutions.SolutionStore`
        stored solutions
    sol_list : list of str
        calibration solutions to extract and interpolate
    G_target : str
        Only extract G solutions for given target. Use any target if None.

    Returns
    -------
    solns_to_apply : list of :class:`~.CalSolution`
        solutions
    """
    solns_to_apply = []

    for X in sol_list:
        if X.startswith('G'):
            # get G values for a two hour range on either side of target scan
            t0, t1 = time_range
            soln = solution_stores[X].get_range(t0 - 2. * 60. * 60.,
                                                t1 + 2. * 60. * 60, target=G_target)
        else:
            # get most recent solution value
            soln = solution_stores[X].latest

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
    The `solution_store` may be ``None`` to save only to the telstate.
    """
    def save_one(soln):
        if telstate is not None:
            telstate.add(key, soln.values, ts=soln.time)
            if soln.snr is not None:
                key_list = key.split('_')
                key_list.insert(1, 'SNR')
                snrkey = '_'.join(key_list)
                telstate.add(snrkey, soln.snr, ts=soln.time)
        if solution_store is not None:
            solution_store.add(soln)
    if telstate is not None:
        logger.info("  - Saving solution '%s' to Telescope State", soln)
    assert isinstance(soln, (solutions.CalSolution, solutions.CalSolutions))
    if isinstance(soln, solutions.CalSolution):
        save_one(soln)
    else:
        n = soln.target
        if soln.snr is not None:
            for v, t, s in zip(soln.values, soln.times, soln.snr):
                save_one(solutions.CalSolution(soln.soltype, v, t, n, s))
        else:
            for v, t in zip(soln.values, soln.times):
                save_one(solutions.CalSolution(soln.soltype, v, t, n))


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
        telstate[shared_key] = info
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
                    info = ('CalSolution', soln.soltype, values, soln.time, soln.target)
                else:
                    info = ('CalSolutions', soln.soltype, values, soln.times, soln.target)
            elif isinstance(soln, (Integral, np.ndarray)):
                info = ('soln', soln)
                if solution_store is not None:
                    logger.warning(
                        'Solution is not of type :class:`~.CalSolution` or `~.CalSolutions`'
                        ' and won\'t be stored in solution store')
            else:
                raise TypeError('Unhandled solution type {}'.format(type(soln)))
        except Exception as error:
            add_info(('Exception', pickle.dumps(error)))
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
            raise pickle.loads(info[1])
        elif info[0] == 'CalSolution':
            soltype, values, time, target = info[1:]
            if values is None:
                saved = telstate.get_range(telstate_key, st=time, et=time, include_end=True)
                if len(saved) != 1:
                    print(len(telstate.get_range(telstate_key, st=0)))
                    raise ValueError('Expected exactly one solution with timestamp {}, found {}'
                                     .format(time, len(saved)))
                values = saved[0][0]
            soln = solutions.CalSolution(soltype, values, time, target)
        elif info[0] == 'CalSolutions':
            soltype, values, times, target = info[1:]
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
            soln = solutions.CalSolutions(soltype, values, times, soltarget=target)
        elif info[0] == 'soln':
            soln = info[1]
        else:
            raise ValueError('Unknown info type {}'.format(info[0]))
        if solution_store is not None:
            if isinstance(soln, (solutions.CalSolution, solutions.CalSolutions)):
                # We don't pass telstate, because we got the value from telstate
                save_solution(None, None, solution_store, soln)
            else:
                logger.warning(
                    'Solution is not of type :class:`~.CalSolution` or `~.CalSolutions`'
                    ' and won\'t be stored in solution store')
        return soln


def shared_B_interp_nans(telstate, parameters, b_soln, st, et):
    """Interpolate over NaNs in the channel axis of the bandpass.

    If there are multiple cal nodes, retrieve all parts of the
    bandpass from telstate prior to interpolation.

    Parameters
    ----------
    telstate : :class:`katsdptelstate.TelescopeState`
        Telescope state in which the solution is stored
    parameters : dict
        Pipeline parameters
    b_soln : :class:`~.CalSolution`
        bandpass solution for this cal node
    st : float
        start time for retrieving solutions from telstate
    et : float
        end time for retrieving solutions from telstate

    Returns
    -------
    :class:`~.CalSolution`
        interpolated solution for this cal node
    """
    # if there are multiple cal nodes
    # retrieve parts of the bandpass computed by other cal nodes
    if parameters['servers'] > 1:
        parts = []
        for n in range(parameters['servers']):
            if n == parameters['server_id']:
                parts.append(b_soln.values)

            else:
                key = 'product_B{0}'.format(n)
                try:
                    telstate.wait_key(key, lambda value, ts: ts > st, 30)
                    valid_part = telstate.get_range('product_B{0}'.format(n), st, et)
                    parts.append(valid_part[0][0])

                except (KeyError, katsdptelstate.TimeoutError):
                    logger.info("Unable to retrieve %s from telstate,"
                                " filling with NaNs while interpolating", key)
                    parts.append(np.full_like(b_soln.values, np.nan))

        b_values = np.concatenate(parts)
    else:
        # else use the values computed by this (the only) node
        b_values = b_soln.values

    # interpolate over NaNs along the channel axis
    nchans, *_ = b_values.shape
    b_interp = interpolate_soln(np.arange(nchans), np.arange(nchans), b_values)

    # select channels processed by this cal node
    b_interp = b_interp[parameters['channel_slice']]

    return solutions.CalSolution('B', b_interp, b_soln.time, b_soln.target)


def set_refant(s, ts, parameters, sensors):
    """Select and update the reference antenna.

    Update the reference antenna in all relevant stores including sensors,
    parameters, telstate and scan attributes.

    Parameters
    ----------
    s : :class:`~.Scan`
        scan of data to select refant
    ts : :class:`katsdptelstate.TelescopeState`
        telstate used in solver and updated to new refant
    parameters : dict
        calibration parameters used by solver and updated to new refant
    sensors : dict
        Sensors available in the calling parent
    """
    # select a new refant if the old one is flagged
    best_refant_index = shared_solve(ts, parameters, None,
                                     parameters['k_bchan'], parameters['k_echan'],
                                     s.refant_find, refant_index=parameters['refant_index_prev'])

    # update parameters, telstate and scan with new refant
    best_refant = parameters['antenna_names'][best_refant_index]
    parameters['refant_index'] = s.refant = best_refant_index
    parameters['refant'] = ts['refant'] = best_refant

    # update sensors if refant has changed
    if best_refant_index != parameters['refant_index_prev']:
        logger.info('Reference antenna set to %s', parameters['refant'])
        sensors['pipeline-reference-antenna'].set_value(
            parameters['refant'], timestamp=s.timestamps[0])


def pipeline(data, ts, parameters, solution_stores, stream_name, flag_array, slots, sensors=None):
    """Pipeline calibration.

    Parameters
    ----------
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
    flag_array : :class: `np.ndarray`, np.uint8,  shape(ntimes, nchans, npol, nbls)
        shared memory flag array for all slots acquired from the accumulator
    slots : list of int
        list of slots acquired from the accumulator
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
            model_params, model_file = pp.get_model(s.target, lsm_dir, ts['sub_band'])
            if model_params is not None:
                s.add_model(model_params)
                ts[model_key] = model_params
                logger.info('   Model file: {0}'.format(model_file))
        else:
            s.add_model(model_params)
        logger.debug('Model parameters for source {0}: {1}'.format(
            target_name, s.model_raw_params))

        # ---------------------------------------
        # Calibrator RFI flagging
        if any(k.endswith('cal') for k in taglist):
            logger.info('Calibrator flagging')
            s.rfi(calib_flagger, sensors=sensors)
            store_flags(slots, scan_slice, flag_array, s)
            # Set a reference antenna for this cbid if one isn't already set
            if s.refant is None:
                set_refant(s, ts, parameters, sensors)

        # run_t0 = time.time()
        # perform calibration as appropriate, from scan intent tags:

        # BEAMFORMER
        if any('bfcal' in k for k in taglist):
            logger.info('Calibrator flagging, auto-correlations')
            s.rfi(calib_flagger, auto_ant=True, sensors=sensors)
            store_flags(slots, scan_slice, flag_array, s, auto_ant=True)
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
            # flagged bandpasses (with NaNs) are stored only in telstate
            save_solution(ts, parameters['product_names']['B'], None, b_soln)
            b_soln_nonans = shared_B_interp_nans(ts, parameters, b_soln,
                                                 s.timestamps[0], s.timestamps[-1])

            # interpolated bandpasses (without NaNs) are stored in solution store
            # so they can be applied to target/calibrator data without propagating NaNs
            solution_stores['B'].add(b_soln_nonans)

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
            # KCROSS_DIODE, BCROSS_DIODE and BCROSS_DIODE_SKY solutions
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
                    solns_to_apply = [s.interpolate(k_soln), s.interpolate(b_soln),
                                      s.interpolate(g_soln)]

                    kcross_soln = shared_solve(ts, parameters, solution_stores['KCROSS_DIODE'],
                                               parameters['k_bchan'], parameters['k_echan'],
                                               s.kcross_sol, pre_apply=solns_to_apply,
                                               nd=nd_on, auto_ant=True)

                    # solve for bcross_diode and save solution
                    solns_to_apply.append(s.interpolate(kcross_soln))
                    logger.info('Solving for BCROSS_DIODE on beamformer calibrator %s', target_name)
                    bcross_soln = s.bcross_sol(solns_to_apply, nd_on)
                    save_solution(ts, parameters['product_names']['BCROSS_DIODE'],
                                  solution_stores['BCROSS_DIODE'], bcross_soln)

                    # if a non-zero spline exists, then create the BCROSS_DIODE_SKY product
                    if np.any(parameters['bcross_sky_spline'][1]):
                        logger.info('Upgrading BCROSS_DIODE to BCROSS_DIODE_SKY')
                        bcross_sky_soln = s.bcross_to_sky(bcross_soln,
                                                          parameters['bcross_sky_spline'],
                                                          parameters['pol_ordering'])

                        # store bcross_sky_soln only in telstate,
                        # as it is never applied to the data by the pipeline
                        save_solution(ts, parameters['product_names']['BCROSS_DIODE_SKY'],
                                      None, bcross_sky_soln)
                    else:
                        logger.info('No spline correction available, skipping solve for '
                                    'BCROSS_DIODE_SKY')

                    # apply solutions and put corrected data into the av_corr dictionary
                    solns_to_apply.append(s.interpolate(bcross_soln))
                    vis = s.auto_ant.tf.cross_pol.vis
                    for soln in solns_to_apply:
                        vis = s.apply(soln, vis, cross_pol=True)
                    logger.info('Averaging corrected auto-corr data for %s:', target_name)
                    cross_data = (vis, s.auto_ant.tf.cross_pol.flags,
                                  s.auto_ant.tf.cross_pol.weights)
                    s.summarize(av_corr, target_name + '_auto_cross', cross_data, nchans=1024)
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
                                      s.g_sol, k_solint, g0_h, calc_snr=False, relative=True)
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
                                          s.g_sol, k_solint, g0_h, pre_apply=solns_to_apply,
                                          calc_snr=False)
                # interpolate to scan timestamps
                g_to_apply = s.interpolate(pre_g_soln)
                solns_to_apply.append(g_to_apply)

                # ---------------------------------------
                # KCROSS solution
                logger.info('Solving for KCROSS on cross-hand delay calibrator %s', target_name)
                kcross_soln = shared_solve(ts, parameters, solution_stores['KCROSS'],
                                           parameters['k_bchan'], parameters['k_echan'],
                                           s.kcross_sol, chan_ave=parameters['kcross_chanave'],
                                           pre_apply=solns_to_apply)
                solns_to_apply.append(s.interpolate(kcross_soln))
                vis = s.cross_ant.tf.cross_pol.vis
                for soln in solns_to_apply:
                    vis = s.apply(soln, vis, cross_pol=True)
                logger.info('Averaging corrected cross-pol data for %s:', target_name)
                cross_data = (vis, s.cross_ant.tf.cross_pol.flags, s.cross_ant.tf.cross_pol.weights)
                s.summarize(av_corr, target_name + '_cross', cross_data, nchans=1024,
                            refant_only=True)

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
                                      s.g_sol, bp_solint, g0_h, pre_apply=solns_to_apply,
                                      calc_snr=False, relative=True)
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
            # flagged bandpasses (with NaNs) are stored only in telstate
            save_solution(ts, parameters['product_names']['B'], None, b_soln)
            b_soln_nonans = shared_B_interp_nans(ts, parameters, b_soln,
                                                 s.timestamps[0], s.timestamps[-1])

            # interpolated solutions (without NaNs) are stored in the solution store
            # so they can be applied to target/calibrator data without propagating NaNs
            solution_stores['B'].add(b_soln_nonans)
            # --------------------------------------------------------------
            # G solution with flux model for BPCals
            logger.info('Solving for G on bandpass calibrator %s', target_name)
            # Only K and B for bandpass G
            solns_to_apply = get_solns_to_apply(s, solution_stores, ['K', 'B'])
            dumps_per_solint = scan_slice.stop - scan_slice.start
            g_solint = dumps_per_solint * dump_period
            g_soln = shared_solve(ts, parameters, None,
                                  parameters['g_bchan'], parameters['g_echan'],
                                  s.g_sol, g_solint, g0_h, pre_apply=solns_to_apply)

            if s.model is not None:
                # Save solution to solution stores G_FLUX
                # only if there is a model.
                save_solution(None, None, solution_stores['G_FLUX'], g_soln)
            elif 'gaincal' not in taglist:
                # If there is no model and the target isn't a gaincal as well
                # save to G
                save_solution(None, None, solution_stores['G'], g_soln)

        # POINTING
        if any('pointingcal' in k for k in taglist):
            # B solution on pointing cal while dishes are offset
            logger.info('Solving for B on pointing calibrator %s', target_name)
            b_soln = s.b_sol(bp0_h)
            # Save only to solution stores
            solution_stores['B_POINTING'].add(b_soln)

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
                         s.g_sol, g_solint, g0_h, pre_apply=solns_to_apply,
                         use_model=False)

        # Apply calibration
        cal_tags = ['gaincal', 'target', 'bfcal', 'bpcal', 'delaycal']
        if any(k in cal_tags for k in taglist):
            # ---------------------------------------
            logger.info('Applying calibration solutions to %s:', target_name)

            # ---------------------------------------
            # get K, B and G solutions to apply and interpolate it to scan timestamps
            if 'bfcal' in taglist:
                solns_to_apply = get_solns_to_apply(s, solution_stores, ['K', 'B', 'G'],
                                                    time_range=[t0, t1])
            elif 'bpcal' in taglist and solution_stores['G_FLUX'].has_target(target_name):
                solns_to_apply = get_solns_to_apply(s, solution_stores, ['K', 'B', 'G_FLUX'],
                                                    time_range=[t0, t1], G_target=target_name)
            elif solution_stores['G'].has_target(target_name):
                solns_to_apply = get_solns_to_apply(s, solution_stores, ['K', 'B', 'G'],
                                                    time_range=[t0, t1], G_target=target_name)
            else:
                # Interpolate to the target across all the available G solutions
                solns_to_apply = get_solns_to_apply(s, solution_stores, ['K', 'B', 'G'],
                                                    time_range=[t0, t1])

            s.apply_inplace(solns_to_apply)

            # save average model for report, if a model exists
            if s.model is not None:
                chan_sample = s.nchan//1024
                model_average = np.average(s.model, axis=(0, -2, -1))[::chan_sample]
                av_corr[target_name + '_model'].insert(0, model_average)

            # TARGET
            if 'target' in taglist:
                # accumulate list of target scans to be streamed to L1
                target_slices.append(scan_slice)

                # flag calibrated target
                logger.info('Flagging calibrated target {0}'.format(target_name,))
                s.rfi(targ_flagger, sensors=sensors)
                store_flags(slots, scan_slice, flag_array, s)

            # summarize corrected data for data with cal tags
            logger.info('Averaging corrected data for %s:', target_name)

            av_corr['targets'].insert(0, (target, np.average(s.timestamps)))
            av_corr['timestamps'].insert(0, (s.timestamps, np.average(s.timestamps)))
            s.summarize_flags(av_corr)

            # summarize gain-calibrated targets
            gaintag = ['gaincal', 'target', 'bfcal']
            nogaintag = ['bpcal', 'delaycal']
            if any(k in gaintag for k in taglist):
                s.summarize_full(av_corr, target_name + '_g_spec', nchans=1024)
                s.summarize(av_corr, target_name + '_g_bls')
                gaincaltag = ['gaincal', 'bfcal']
                if any(k in gaincaltag for k in taglist):
                    s.summarize(av_corr, target_name + '_g_phase', avg_ant=True)
            # summarize non-gain calibrated targets
            if any(k in nogaintag for k in taglist):
                s.summarize(av_corr, target_name + '_nog_spec', nchans=1024, refant_only=True)

    return target_slices, av_corr


def get_offsets(ts, parameters, target, t_stamps, temp, pres, humi):
    """Calculate offset co-ordinates relative to target for each pointing.

    Parameters
    ----------
    ts : :class:`katsdptelstate.TelescopeState`
        Telescope state, scoped to the cal namespace within the capture block
    parameters : dict
        The pipeline parameters
    target : :class:`katpoint.Target`
        Pointing calibrator as a katpoint target object
    t_stamps : list
        Timestamps at which offset co-ordinates must be determined
    temp, pres, humi : :class: 'float'
        Atmospheric conditions used for refraction correction

    Returns
    -------
    offsets : list of (x,y) co-ordinates for each pointing
    """
    rc = katpoint.RefractionCorrection()
    # Set refant index
    refant_ind = parameters['refant_index']
    refant = parameters['antennas'][refant_ind]
    # Get middle timestamp of each track slice
    offsets = []
    for j in t_stamps:
        # AZ/EL co-ordinates of target
        azel = target.azel(timestamp=j, antenna=refant)
        # apply refraction to target co-ordinates (already in radians)
        ref_corr_el = rc.apply(azel[1], temp, pres, humi)
        # Construct target object
        # TODO The following ideally needs katdal sensor framework for interpolation
        trgt = katpoint.construct_azel_target(azel[0], ref_corr_el)
        # Get offset az/el co-ordinates (direction in which reference antenna is pointing)
        az = ts.get_range(refant.name + '_pos_actual_scan_azim', et=j)
        el = ts.get_range(refant.name + '_pos_actual_scan_elev', et=j)
        az_actual = katpoint.deg2rad(az[0][0])
        el_actual = rc.apply(katpoint.deg2rad(el[0][0]), temp, pres, humi)
        # Project spherical coordinates to plane with target position as reference
        offset = trgt.sphere_to_plane(
            az_actual, el_actual, coord_system='azel', antenna=refant, timestamp=j
        )
        offset = (katpoint.rad2deg(offset[0]), katpoint.rad2deg(offset[1]))
        offsets.append(offset)

    return offsets


def _finish_pointing_cal(ts, parameters, b_solutions):
    """Complete pointing calibration by fitting beams to B gains and saving to telstate.

    Uses katsdpcalproc routines to fit primary beams using bandpass solutions
    generated on previous target tracks and saves beam solutions to telstate
    as calibration product EPOINT.
    """
    # Group the global frequency channels into this many sections (fewer per server)
    num_chunks = parameters['epoint_freq_chunks']
    # Extract some some commonly used constants from the TS and parameters
    # Middle time for each dump
    mid_times = b_solutions.times
    # TODO The following ideally needs katdal sensor framework for interpolation
    target_str = ts.get_range('cbf_target', et=mid_times[0])[0][0]
    target = katpoint.Target(target_str)
    # Atmospheric conditions
    soltime = np.mean(mid_times)
    pres = ts.get_range('anc_air_pressure', et=soltime)[0][0]
    temp = ts.get_range('anc_air_temperature', et=soltime)[0][0]
    humi = ts.get_range('anc_air_relative_humidity', et=soltime)[0][0]
    ants = parameters['antennas']
    channel_freqs = parameters['channel_freqs']
    pols = parameters['pol_ordering']

    # Calculate offset (x,y) co-ordinates for each pointing
    offsets = get_offsets(ts, parameters, target, mid_times, temp, pres, humi)
    # Extract gains per pointing offset, per receptor and per frequency chunk.
    data_points = pointing.get_offset_gains(b_solutions.values, offsets, ants, channel_freqs,
                                            pols, num_chunks)
    # Fit primary beams to the gains
    beams = pointing.beam_fit(data_points, ants, num_chunks)
    # Save fitted beams as CalSolution with shape (num_chunks, len(pols), len(ants), 5)
    beam_sol = np.full((num_chunks, len(pols), len(ants), 5), np.nan, dtype=np.float32)
    beam_sol_SNR = np.full((num_chunks, len(pols), len(ants), 5), np.nan, dtype=np.float32)
    for a, ant in enumerate(ants):
        for c, beam in enumerate(beams[ant.name]):
            if beam is None:
                continue
            beam_sol[c, :, a] = np.r_[beam.center, beam.width, beam.height]
            beam_sol_SNR[c, :, a] = 1 / np.r_[beam.std_center, beam.std_width, beam.std_height]
            if not beam.is_valid:
                beam_sol_SNR[c, :, a, -1] = 0.0
    beam_sol = solutions.CalSolution(soltype='EPOINT', soltime=soltime, solvalues=beam_sol,
                                     soltarget=target.name, solsnr=beam_sol_SNR)
    # Save fitted beam CalSolution and beam SNR to telstate
    save_solution(ts, parameters['product_names']['EPOINT'], None, beam_sol)


def flush_pipeline(ts, parameters, solution_stores):
    """Complete any unfinished calibration products remaining in the pipeline.

    Saves beam solutions and uncertainties to telstate.

    Parameters
    ----------
    ts : :class:`katsdptelstate.TelescopeState`
        Telescope state, scoped to the cal namespace within the capture block
    parameters : dict
        The pipeline parameters
    solution_stores : dict of :class:`~.CalSolutionStore`-like
        Solution stores for the capture block, indexed by solution type
    """
    logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    logger.info('Flushing pipeline')

    # See if there are any intermediate bandpass gains for pointing cal in solution stores
    b_solutions = solution_stores['B_POINTING'].get_range(start_time=0, end_time=time.time())
    n_pointings = len(b_solutions.times)
    if n_pointings > 0:
        logger.info('Finishing pointing cal on %d pointings', n_pointings)
        _finish_pointing_cal(ts, parameters, b_solutions)
