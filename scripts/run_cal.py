#! /usr/bin/env python
import numpy as np
import time
import os
import shutil
import signal
import manhole

from katsdptelstate import endpoint
import katsdpservices

from katsdpcal.control import init_accumulator_control, init_pipeline_control, shared_empty
from katsdpcal.control import end_transmit
from katsdpcal.report import make_cal_report
from katsdpcal.pipelineprocs import ts_from_file, setup_ts

from katsdpcal import param_dir, rfi_dir

import logging
logger = logging.getLogger(__name__)


def print_dict(dictionary, ident='', braces=1):
    """ Recursively prints nested dictionaries."""

    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            print '{0}{1}{2}{3}'.format(ident, braces * '[', key, braces * ']')
            print_dict(value, ident+'  ', braces+1)
        else:
            print '{0}{1} = {2}'.format(ident, key, value)


def log_dict(dictionary, ident='', braces=1):
    """ Recursively logs nested dictionaries."""

    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            logger.info('{0}{1}{2}{3}'.format(ident, braces * '[', key, braces*']'))
            log_dict(value, ident+'  ', braces+1)
        else:
            logger.info('{0}{1} = {2}'.format(ident, key, value))


def comma_list(type_):
    """Return a function which splits a string on commas and converts each element to
    `type_`."""

    def convert(arg):
        return [type_(x) for x in arg.split(',')]
    return convert


def parse_opts():
    parser = katsdpservices.ArgumentParser(
        description='Set up and wait for spead stream to run the pipeline.')
    parser.add_argument(
        '--num-buffers', type=int, default=2,
        help='Specify the number of data buffers to use. default: 2')
    parser.add_argument(
        '--buffer-maxsize', type=float,
        help='The amount of memory (in bytes) to allocate to each buffer.')
    parser.add_argument(
        '--no-auto', action='store_true',
        help='Pipeline data DOESNT include autocorrelations '
        + '[default: False (autocorrelations included)]')
    parser.set_defaults(no_auto=False)
    # note - the following lines extract various parameters from the MC config
    parser.add_argument(
        '--cbf-channels', type=int,
        help='The number of frequency channels in the visibility data. Default from MC config')
    parser.add_argument(
        '--cbf-pols', type=int,
        help='The number of polarisation products in the visibility data. Default from MC config')
    parser.add_argument(
        '--antenna-mask', type=comma_list(str),
        help='List of antennas in the L0 data stream. Default from MC config')
    # also need bls ordering
    parser.add_argument(
        '--l0-spectral-spead', type=endpoint.endpoint_list_parser(7200, single_port=True),
        default=':7200',
        help='endpoints to listen for L0 spead stream (including multicast IPs). '
        + '[<ip>[+<count>]][:port]. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument(
        '--l0-spectral-interface',
        help='interface to subscribe to for L0 spectral data. [default: auto]', metavar='INTERFACE')
    parser.add_argument(
        '--l1-spectral-spead', type=endpoint.endpoint_parser(7202), default='127.0.0.1:7202',
        help='destination for spectral L1 output. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument(
        '--l1-rate', type=float, default=5e7,
        help='L1 spead transmission rate. For laptops, recommend rate of 5e7. Default: 5e7')
    parser.add_argument(
        '--l1_level', default=0,
        help='Data to transmit to L1: 0 - none, 1 - target only, 2 - all [default: 0]')
    parser.add_argument(
        '--notthreading', action='store_false',
        help='Use threading to control pipeline and accumulator '
        + '[default: False (to use multiprocessing)]')
    parser.set_defaults(notthreading=True)
    parser.add_argument(
        '--parameter-file', type=str, default='',
        help='Default pipeline parameter file (will be over written by TelescopeState.')
    parser.add_argument(
        '--report-path', type=str, default='/var/kat/data',
        help='Path under which to save pipeline report. [default: /var/kat/data]')
    parser.add_argument(
        '--log-path', type=str, default=os.path.abspath('.'),
        help='Path under which to save pipeline logs. [default: current directory]')
    # parser.set_defaults(telstate='localhost')
    return parser.parse_args()


def setup_logger(log_name, log_path='.'):
    """
    Set up the pipeline logger.
    The logger writes to a pipeline.log file and to stdout.

    Inputs
    ======
    log_path : str
        path in which log file will be written
    log_name : str
        name of log file
    """
    katsdpservices.setup_logging()

    # logging to file
    log_path = os.path.abspath(log_path)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03dZ %(name)-24s %(levelname)-8s %(message)s')
    formatter.datefmt = '%Y-%m-%d %H:%M:%S'
    formatter.converter = time.gmtime

    handler = logging.FileHandler('{0}/{1}'.format(log_path, log_name))
    handler.setFormatter(formatter)
    logging.getLogger('').addHandler(handler)


def setup_observation_logger(log_name, log_path='.'):
    """
    Set up a pipeline logger to file.

    Inputs
    ======
    log_path : str
        path in which log file will be written
    """
    log_path = os.path.abspath(log_path)

    # logging to file
    # set format
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03dZ %(name)-24s %(levelname)-8s %(message)s')
    formatter.datefmt = '%Y-%m-%d %H:%M:%S'

    obs_log = logging.FileHandler('{0}/{1}'.format(log_path, log_name))
    obs_log.setFormatter(formatter)
    logging.getLogger('').addHandler(obs_log)
    return obs_log


def stop_observation_logger(obs_log):
    logging.getLogger('').removeHandler(obs_log)


def all_alive(process_list):
    """
    Check if all of the process in process list are alive and return True
    if they are or False if they are not.

    Inputs
    ======
    process_list:  list of multpirocessing.Process objects
    """

    alive = True
    for process in process_list:
        alive = alive and process.is_alive()
    return alive


def create_buffer_arrays(buffer_shape, mproc=True):
    """
    Create empty buffer record using specified dimensions
    """
    if mproc:
        factory = shared_empty
    else:
        factory = np.empty
    data = {}
    data['vis'] = factory(buffer_shape, dtype=np.complex64)
    data['flags'] = factory(buffer_shape, dtype=np.uint8)
    data['weights'] = factory(buffer_shape, dtype=np.float32)
    data['times'] = factory(buffer_shape[0], dtype=np.float)
    data['max_index'] = factory([1], dtype=np.int32)
    data['max_index'][0] = 0
    return data


def force_shutdown(accumulator, pipelines):
    # forces pipeline threads to shut down
    accumulator.stop()
    accumulator.join()
    # pipeline needs to be terminated, rather than stopped,
    # to end long running reduction.pipeline function
    map(lambda x: x.terminate(), pipelines)


def kill_shutdown():
    # brutal kill (for threading)
    os.kill(os.getpid(), signal.SIGKILL)


def run_threads(ts, cbf_n_chans, cbf_n_pols, antenna_mask, num_buffers=2,
                buffer_maxsize=None, auto=True,
                l0_endpoint=':7200', l0_interface=None,
                l1_endpoint='127.0.0.1:7202', l1_rate=5.0e7, l1_level=0,
                mproc=True, param_file='', report_path='', log_path='.', full_log=None):
    """
    Start the pipeline using 'num_buffers' buffers, each of size 'buffer_maxsize'.
    This will instantiate num_buffers + 1 threads; a thread for each pipeline and an
    extra accumulator thread the reads data from the spead stream into each buffer
    seen by the pipeline.

    Inputs
    ======
    ts : TelescopeState
        The telescope state, default: 'localhost' database 0
    cbf_n_chans : int
        The number of channels in the data stream
    cbf_n_pols : int
        The number of polarisations in the data stream
    antenna_mask : list of strings
        List of antennas present in the data stream
    num_buffers : int
        The number of buffers to use- this will create a pipeline thread for each buffer
        and an extra accumulator thread to read the spead stream.
    buffer_maxsize : float
        The maximum size of the buffer. Memory for each buffer will be allocated at first
        and then populated by the accumulator from the spead stream.
    auto : bool
        True for autocorrelations included in the data, False for cross-correlations only
    l0_endpoint : endpoint
        Endpoint to listen to for L0 stream, default: ':7200'
    l0_interface : str
        Name of interface to subscribe to for L0, or None to let the OS decide
    l1_endpoint : endpoint
        Destination endpoint for L1 stream, default: '127.0.0.1:7202'
    l1_rate : float
        Rate for L1 stream transmission, default 5e7
    l1_level : int
        Data to transmit to L1: 0 - none, 1 - target only, 2 - all
    mproc : bool
        True for control via multiprocessing, False for control via threading
    param_file : string
        File of default pipeline parameters
    report_path : string
        Path under which to save pipeline report
    log_path : string
        Path for pipeline logs
    """

    logger.info('Pipeline system input parameters')
    logger.info('   - antenna mask: {0}'.format(antenna_mask,))
    logger.info('   - number of channels: {0}'.format(cbf_n_chans,))
    logger.info('   - number of polarisation products: {0}'.format(cbf_n_pols,))

    # extract data shape parameters
    #   argument parser traversed TS config to find these
    if antenna_mask is not None:
        ts.add('antenna_mask', antenna_mask, immutable=True)
    elif 'antenna_mask' not in ts:
        raise RuntimeError("No antenna_mask set.")
    if len(ts.antenna_mask) < 4:
        # if we have less than four antennas, no katsdpcal necessary
        logger.info('Only {0} antenna present - stopping katsdpcal'.format(len(ts.antenna_mask,)))
        return

    # deal with required input parameters
    if cbf_n_chans is not None:
        ts.add('cbf_n_chans', cbf_n_chans, immutable=True)
    elif 'cbf_n_chans' not in ts:
        raise RuntimeError("No cbf_n_chans set.")
    if cbf_n_pols is not None:
        ts.add('cbf_n_pols', cbf_n_pols, immutable=True)
    elif 'cbf_n_pols' not in ts:
        logger.warning(
            'Number of polarisation inputs cbf_n_pols not set. Setting to default value 4')
        ts.add('cbf_n_pols', 4, immutable=True)

    # initialise TS from default parameter file
    #   defaults are used only for parameters missing from the TS
    if param_file == '':
        if ts.cbf_n_chans == 4096:
            param_filename = 'pipeline_parameters_meerkat_ar1_4k.txt'
            param_file = os.path.join(param_dir, param_filename)
            logger.info('Parameter file for 4k mode: {0}'.format(param_file,))
            rfi_filename = 'rfi_mask.pickle'
            rfi_file = os.path.join(rfi_dir, rfi_filename)
            logger.info('RFI mask file for 4k mode: {0}'.format(rfi_file,))
        else:
            param_filename = 'pipeline_parameters_meerkat_ar1_32k.txt'
            param_file = os.path.join(param_dir, param_filename)
            logger.info('Parameter file for 32k mode: {0}'.format(param_file,))
            rfi_filename = 'rfi_mask32K.pickle'
            rfi_file = os.path.join(rfi_dir, rfi_filename)
            logger.info('RFI mask file for 32k mode: {0}'.format(rfi_file,))
    else:
        logger.info('Parameter file: {0}'.format(param_file))
    logger.info('Inputting Telescope State parameters from parameter file.')
    ts_from_file(ts, param_file, rfi_file)
    # telescope state logs for debugging
    logger.info('Telescope state parameters:')
    for keyval in ts.keys():
        if keyval not in ['sdp_l0_bls_ordering', 'cbf_channel_freqs']:
            logger.info('{0} : {1}'.format(keyval, ts[keyval]))
    logger.info('Telescope state config graph:')
    log_dict(ts.config)

    # set up TS for pipeline use
    logger.info('Setting up Telescope State parameters for pipeline.')
    setup_ts(ts)

    nant = len(ts.cal_antlist)
    # number of baselines (may include autocorrelations)
    nbl = nant*(nant+1)/2 if auto else nant*(nant-1)/2

    # get buffer size
    if buffer_maxsize is not None:
        ts.add('cal_buffer_size', buffer_maxsize)
    else:
        if 'cal_buffer_size' in ts:
            buffer_maxsize = ts['cal_buffer_size']
        else:
            buffer_maxsize = 20.0e9
            ts.add('cal_buffer_size', buffer_maxsize)

    # buffer needs to include:
    #   visibilities, shape(time,channel,baseline,pol), type complex64 (8 bytes)
    #   flags, shape(time,channel,baseline,pol), type uint8 (1 byte)
    #   weights, shape(time,channel,baseline,pol), type float32 (4 bytes)
    #   time, shape(time), type float64 (8 bytes)
    # plus minimal extra for scan transition indices
    scale_factor = 8. + 1. + 4.  # vis + flags + weights
    time_factor = 8. + 0.1  # time + 0.1 for good measure (indiced)
    array_length = buffer_maxsize/((scale_factor*ts.cbf_n_chans*ts.cbf_n_pols*nbl) + time_factor)
    array_length = np.int(np.ceil(array_length))
    logger.info('Buffer size : {0} G'.format(buffer_maxsize/1.e9,))
    logger.info('Max length of buffer array : {0}'.format(array_length,))

    # Set up empty buffers
    buffer_shape = [array_length, ts.cbf_n_chans, ts.cbf_n_pols, nbl]
    buffers = [create_buffer_arrays(buffer_shape, mproc=mproc) for i in range(num_buffers)]

    # account for forced shutdown possibilities
    #  due to SIGTERM, keyboard interrupt, or unknown error
    forced_shutdown = False

    # get subarray ID
    subarray_id = ts['subarray_product_id'] if 'subarray_product_id' in ts else 'unknown_subarray'

    logger.info('Receiving L0 data from {0} via {1}'.format(
        l0_endpoint, 'default interface' if l0_interface is None else l0_interface))
    l0_interface_address = katsdpservices.get_interface_address(l0_interface)
    while not forced_shutdown:
        observation_log = '{0}_pipeline.log'.format(int(time.time()),)
        obs_log = setup_observation_logger(observation_log, log_path)

        logger.info('===========================')
        logger.info('   Starting new observation')

        # set up inter-task synchronisation primitives.
        # passed events to indicate buffer transfer, end-of-observation, or stop
        accum_pipeline_queues = [control_method.Queue() for i in range(num_buffers)]
        # signalled when the pipeline is finished with a buffer
        pipeline_accum_sems = [control_method.Semaphore(value=1) for i in range(num_buffers)]

        # Set up the accumulator
        accumulator = init_accumulator_control(control_method, control_task, buffers,
                                               buffer_shape,
                                               accum_pipeline_queues, pipeline_accum_sems,
                                               l0_endpoint, l0_interface_address, ts)
        # Set up the pipelines (one per buffer)
        pipelines = [init_pipeline_control(control_method, control_task,
                     buffers[i], buffer_shape,
                     accum_pipeline_queues[i], pipeline_accum_sems[i], i,
                     l1_endpoint, l1_level, l1_rate, ts) for i in range(num_buffers)]

        try:
            manhole.install(oneshot_on='USR1', locals={
                'ts': ts, 'accumulator': accumulator, 'pipelines': pipelines})
        # allow remote debug connections and expose telescope state, accumulator and pipelines
        except manhole.AlreadyInstalled:
            pass

        # Start the pipeline threads
        map(lambda x: x.start(), pipelines)
        # Start the accumulator thread
        accumulator.start()
        logger.info('Waiting for L0 data')

        # wait until the everything has shut down
        try:
            accumulator.join()
            logger.info('Accumulator stopped')
            for pipeline in pipelines:
                pipeline.join()
            logger.info('Pipelines stopped')
        except (KeyboardInterrupt, SystemExit):
            logger.info('Received interrupt! Quitting threads.')
            force_shutdown(accumulator, pipelines) if mproc else kill_shutdown()
            forced_shutdown = True
        except Exception, e:
            logger.error('Unknown error: {}'.format(e))
            force_shutdown(accumulator, pipelines)
            forced_shutdown = True

        # closing steps, if data transmission has stoped (skipped for forced early shutdown)
        if not forced_shutdown:
            # get observation end time
            if 'cal_obs_end_time' in ts:
                obs_end = ts.cal_obs_end_time
            else:
                logger.info('Unknown observation end time')
                obs_end = time.time()
            # get observation name
            try:
                obs_params = ts.get_range('obs_params', st=0, et=obs_end, return_format='recarray')
                obs_keys = obs_params['value']
                obs_times = obs_params['time']
                # choose most recent experiment id (last entry in the list), if
                # there are more than one
                experiment_id_string = [x for x in obs_keys if 'experiment_id' in x][-1]
                experiment_id = eval(experiment_id_string.split()[-1])
                obs_start = [t for x, t in zip(obs_keys, obs_times) if 'experiment_id' in x][-1]
            except (TypeError, KeyError, AttributeError):
                # TypeError, KeyError because this isn't properly implemented yet
                # AttributeError in case this key isnt in the telstate for whatever reason
                experiment_id = '{0}_unknown_project'.format(int(time.time()),)
                obs_start = None

            # make directory for this observation, for logs and report
            if not report_path:
                report_path = '.'
            report_path = os.path.abspath(report_path)
            obs_dir = '{0}/{1}_{2}_{3}'.format(
                report_path, int(time.time()), subarray_id, experiment_id)
            current_obs_dir = '{0}-current'.format(obs_dir,)
            try:
                os.mkdir(current_obs_dir)
            except OSError:
                logger.warning('Experiment ID directory {} already exits'.format(current_obs_dir,))

            # create pipeline report (very basic at the moment)
            try:
                make_cal_report(ts, current_obs_dir, experiment_id, st=obs_start, et=obs_end)
            except Exception as e:
                logger.info('Report generation failed: {0}'.format(e,))

            if l1_level != 0:
                # send L1 stop transmission
                #   wait for a couple of secs before ending transmission
                time.sleep(2.0)
                end_transmit(l1_endpoint.host, l1_endpoint.port)
                logger.info('L1 stream ended')

            logger.info('   Observation ended')
            logger.info('===========================')

            # copy log of this observation into the report directory
            shutil.move('{0}/{1}'.format(log_path, observation_log),
                        '{0}/pipeline_{1}.log'.format(current_obs_dir, experiment_id))
            stop_observation_logger(obs_log)
            if full_log is not None:
                shutil.copy('{0}/{1}'.format(log_path, full_log),
                            '{0}/{1}'.format(current_obs_dir, full_log))

            # change report and log directory to final name for archiving
            shutil.move(current_obs_dir, obs_dir)


if __name__ == '__main__':

    opts = parse_opts()

    # set up logging
    log_name = 'pipeline.log'
    log_path = os.path.abspath(opts.log_path)
    setup_logger(log_name, log_path)

    # threading or multiprocessing imports
    if opts.notthreading is True:
        logger.info("Using multiprocessing")
        import multiprocessing as control_method
        class control_task(control_method.Process):
            def start(self):
                # Block SIGINT while spawning the child, so that the child
                # won't get a KeyboardInterrupt if Ctrl-C is pressed. There
                # is a race condition where a Ctrl-C while in this function
                # would be lost, but since SIGINT is only intended for
                # interactive use, the user can just push it again. With Python
                # 3 it would be possible to fix this with
                # signal.pthread_sigmask to block rather than ignore the
                # signal.
                orig = signal.signal(signal.SIGINT, signal.SIG_IGN)
                super(control_task, self).start()
                signal.signal(signal.SIGINT, orig)
    else:
        logger.info("Using threading")
        import multiprocessing.dummy as control_method
        from multiprocessing.dummy import Process as control_task

    def force_exit(_signo=None, _stack_frame=None):
        logger.info("Exiting katsdpcal on SIGTERM")
        raise SystemExit

    signal.signal(signal.SIGTERM, force_exit)

    run_threads(
        opts.telstate,
        cbf_n_chans=opts.cbf_channels, cbf_n_pols=opts.cbf_pols, antenna_mask=opts.antenna_mask,
        num_buffers=opts.num_buffers, buffer_maxsize=opts.buffer_maxsize, auto=not(opts.no_auto),
        l0_endpoint=opts.l0_spectral_spead[0], l0_interface=opts.l0_spectral_interface,
        l1_endpoint=opts.l1_spectral_spead,
        l1_rate=opts.l1_rate, l1_level=opts.l1_level, mproc=opts.notthreading,
        param_file=opts.parameter_file,
        report_path=opts.report_path, log_path=log_path, full_log=log_name)
