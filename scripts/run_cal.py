#! /usr/bin/env python
import numpy as np
import time
import os
import signal
import manhole

from katsdpcal.simulator import SimData
from katsdpcal import parameters

from katsdptelstate.telescope_state import TelescopeState
from katsdptelstate import endpoint, ArgumentParser

from katsdpcal.control import init_accumulator_control, init_pipeline_control
from katsdpcal.control import end_transmit

from katsdpcal.report import make_cal_report

import logging
logger = logging.getLogger(__name__)

def print_dict(dictionary, ident = '', braces=1):
    """ Recursively prints nested dictionaries."""

    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
           print '%s%s%s%s' %(ident,braces*'[',key,braces*']')
           print_dict(value, ident+'  ', braces+1)
        else:
           print ident+'%s = %s' %(key, value)

def comma_list(type_):
    """Return a function which splits a string on commas and converts each element to
    `type_`."""

    def convert(arg):
        return [type_(x) for x in arg.split(',')]
    return convert

def parse_opts():
    parser = ArgumentParser(description = 'Set up and wait for spead stream to run the pipeline.')
    parser.add_argument('--num-buffers', type=int, default=2, help='Specify the number of data buffers to use. default: 2')
    parser.add_argument('--buffer-maxsize', type=float, default=1000e6, help='The amount of memory (in bytes?) to allocate to each buffer. default: 1e9')
    # note - the following lines extract various parameters from the MC config
    parser.add_argument('--cbf-channels', type=int, help='The number of frequency channels in the visibility data. Default from MC config')
    parser.add_argument('--antenna-mask', type=comma_list(str), help='List of antennas in the L0 data stream. Default from MC config')
    # also need bls ordering
    parser.add_argument('--l0-spectral-spead', type=endpoint.endpoint_list_parser(7200, single_port=True), default=':7200', help='endpoints to listen for L0 spead stream (including multicast IPs). [<ip>[+<count>]][:port]. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument('--l1-spectral-spead', type=endpoint.endpoint_parser(7202), default='127.0.0.1:7202', help='destination for spectral L1 output. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument('--l1-rate', type=float, default=5e7, help='L1 spead transmission rate. For laptops, recommend rate of 5e7. Default: 5e7')
    parser.add_argument('--full-l1', action='store_true', help='Send full data set to L1 [default: Only send target data to L1')
    parser.set_defaults(full_l1=False)
    parser.add_argument('--threading', action='store_true', help='Use threading to control pipeline and accumulator [default: False (to use multiprocessing)]')
    parser.set_defaults(threading=False)
    parser.add_argument('--report-path', type=str, default=os.path.abspath('.'), help='Path under which to save pipeline report. [default: current directory]')
    parser.add_argument('--log-path', type=str, default=os.path.abspath('.'), help='Path under which to save pipeline logs. [default: current directory]')
    #parser.set_defaults(telstate='localhost')
    return parser.parse_args()

def setup_logger(log_path):
    """
    Set up the pipeline logger.
    The logger writes to a pipeline.log file and to stdout.

    Inputs
    ======
    log_path : str
        path in which log file will be written
    """
    if not log_path: log_path = '.'
    log_path = os.path.abspath(log_path)

    # logging to file
    logging.basicConfig(filename='{0}/pipeline.log'.format(log_path,),
                        format='%(asctime)s %(name)-24s %(levelname)-8s %(message)s',
                        datefmt='%d-%m-%y %H:%M',)
    logger.setLevel(logging.INFO)

    # logging to stdout
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set format for console use
    formatter = logging.Formatter('%(asctime)s %(name)-24s %(levelname)-8s %(message)s')
    formatter.datefmt='%d-%m %H:%M'
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

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

def any_alive(process_list):
    """
    Check if any of the process in process list are alive and return True
    if any are, False otherwise.

    Inputs
    ======
    process_list:  list of multpirocessing.Process objects
    """

    alive = False
    for process in process_list:
        alive = alive or process.is_alive()
    return alive

def create_buffer_arrays(buffer_shape,mproc=True):
    """
    Create empty buffer record using specified dimensions
    """
    if mproc is True:
        return create_buffer_arrays_multiprocessing(buffer_shape)
    else:
        return create_buffer_arrays_threading(buffer_shape)

def create_buffer_arrays_multiprocessing(buffer_shape):
    """
    Create empty buffer record using specified dimensions,
    for multiprocessing shared memory appropriate buffers
    """
    data={}
    array_length = buffer_shape[0]
    buffer_size = reduce(lambda x, y: x*y, buffer_shape)
    data['vis'] = control_method.RawArray(c_float, buffer_size*2) # times two for complex
    data['flags'] = control_method.RawArray(c_ubyte, buffer_size)
    data['weights'] = control_method.RawArray(c_float, buffer_size)
    data['times'] = control_method.RawArray(c_double, array_length)
    data['max_index'] = control_method.sharedctypes.RawArray(c_int, 1)
    return data

def create_buffer_arrays_threading(buffer_shape):
    """
    Create empty buffer record using specified dimensions,
    for thread appropriat numpy buffers
    """
    data={}
    data['vis'] = np.empty(buffer_shape,dtype=np.complex64)
    data['flags'] = np.empty(buffer_shape,dtype=np.uint8)
    data['weights'] = np.empty(buffer_shape,dtype=np.float64)
    data['times'] = np.empty(buffer_shape[0],dtype=np.float)
    data['max_index'] = np.empty([0.0], dtype=np.int32)
    return data

def run_threads(ts, cbf_n_chans, antenna_mask, num_buffers=2, buffer_maxsize=1000e6,
           l0_endpoint=':7200', l1_endpoint='127.0.0.1:7202', l1_rate=5.0e7, full_l1=False,
           mproc=True,report_path=''):
    """
    Start the pipeline using 'num_buffers' buffers, each of size 'buffer_maxsize'.
    This will instantiate num_buffers + 1 threads; a thread for each pipeline and an
    extra accumulator thread the reads data from the spead stream into each buffer
    seen by the pipeline.

    Inputs
    ======
    ts: TelescopeState
        The telescope state, default: 'localhost' database 0
    cbf_n_chans: int
        The number of channels in the data stream
    antenna_mask: list of strings
        List of antennas present in the data stream
    num_buffers: int
        The number of buffers to use- this will create a pipeline thread for each buffer
        and an extra accumulator thread to read the spead stream.
    buffer_maxsize: float
        The maximum size of the buffer. Memory for each buffer will be allocated at first
        and then populated by the accumulator from the spead stream.
    l0_endpoint: endpoint
        Endpoint to listen to for L0 stream, default: ':7200'
    l1_endpoint: endpoint
        Destination endpoint for L1 stream, default: '127.0.0.1:7202'
    l1_rate : float
        Rate for L1 stream transmission, default 5e7
    full_l1 : bool
        True to transmit all of the data to L1, False to only transmit target data.
    mproc: bool
        True for control via multiprocessing, False for control via threading
    report_path : string
        Path under which to save pipeline report
    """

    # debug print outs
    print '\nTelescope state: '
    for k in ts.keys(): print k
    print '\nTelescope state config graph: '
    print_dict(ts.config)
    print

    # extract data shape parameters from TS
    #  antenna_mask and cbf_n_chans come from MC config if present, else try the TS 
    try:
        if antenna_mask is None: antenna_mask = ts.antenna_mask.split(',')
    except:
        raise RuntimeError("No antenna_mask set.")
    try:
        if cbf_n_chans is None: cbf_n_chans = ts.cbf_n_chans
        nchan = cbf_n_chans
    except:
        raise RuntimeError("No cbf_n_chans set.")

    # save L1 transmit preference to TS
    ts.add('cal_full_l1', full_l1, immutable=True)

    # ensure reference antenna is present in the dataset
    #   if reference antenna is not present,
    #   iterate through preferred antenna list of find next best option
    if ts.cal_refant not in antenna_mask:
        for ant in ts.cal_preferred_refants:
            if ant in antenna_mask:
                ts.cal_refant = ant
                break

    npol = 4
    nant = len(antenna_mask)
    # number of baselines includes autocorrelations
    nbl = nant*(nant+1)/2

    # buffer needs to include:
    #   visibilities, shape(time,channel,baseline,pol), type complex64 (8 bytes)
    #   flags, shape(time,channel,baseline,pol), type int8 (? confirm)
    #   weights, shape(time,channel,baseline,pol), type int8 (? confirm)
    #   time, shape(time), type float64 (8 bytes)
    # plus minimal extra for scan transition indices
    scale_factor = 8. + 1. + 1.  # vis + flags + weights
    time_factor = 8.
    array_length = buffer_maxsize/((scale_factor*nchan*npol*nbl) + time_factor)
    array_length = np.int(np.ceil(array_length))
    logger.info('Max length of buffer array : {0}'.format(array_length,))

    # Set up empty buffers
    buffer_shape = [array_length,nchan,npol,nbl]
    buffers = [create_buffer_arrays(buffer_shape,mproc=mproc) for i in range(num_buffers)]

    # set up conditions for the buffers
    scan_accumulator_conditions = [control_method.Condition() for i in range(num_buffers)]

    # Set up the accumulator
    accumulator = init_accumulator_control(control_method, control_task, buffers, buffer_shape, scan_accumulator_conditions, l0_endpoint, ts)
    # Set up the pipelines (one per buffer)
    pipelines = [init_pipeline_control(control_method, control_task, buffers[i], buffer_shape, scan_accumulator_conditions[i], i, \
        l1_endpoint, l1_rate, ts) for i in range(num_buffers)]

    manhole.install(oneshot_on='USR1', locals={'ts':ts, 'accumulator':accumulator, 'pipelines':pipelines})
     # allow remote debug connections and expose telescope state, accumulator and pipelines
    # Start the pipeline threads
    map(lambda x: x.start(), pipelines)
    # Start the accumulator thread
    accumulator.start()

    # account for forced shutdown possibilities
    #  due to SIGTERM, keyboard interrupt, or unknown error
    forced_shutdown = False

    def force_shutdown():
        # forces pipeline threads to shut down
        accumulator.stop()
        accumulator.join()
        # pipeline needs to be terminated, rather than stopped,
        # to end long running reduction.pipeline function
        map(lambda x: x.terminate(), pipelines)

    def force_hard_shutdown():
        # forces pipeline threads to shut down
        #  - faster but dirtier than force_shutdown()
        accumulator.terminate()
        # terminating accumulator leaves hanging wait in pipeline
        # - aggressive kill needed for running pipeline
        map(lambda x: os.kill(x.pid, signal.SIGKILL), pipelines)

    def kill_shutdown():
        # brutal kill (for threading)
        os.kill(os.getpid(), signal.SIGKILL)

    try:
        # run tasks until the observation has ended
        while all_alive([accumulator]+pipelines) and not accumulator.obs_finished():
            time.sleep(0.1)
    except (KeyboardInterrupt, SystemExit):
        logger.info('Received interrupt! Quitting threads.')
        force_shutdown() if mproc else kill_shutdown()
        forced_shutdown = True
    except:
        logger.error('Unknown error. ')
        force_shutdown()
        forced_shutdown = True

    # closing steps, if data transmission has stoped (skipped for forced early shutdown)
    if not forced_shutdown:
        # Stop pipelines first so they recieve correct signal before accumulator acquires the condition
        map(lambda x: x.stop(), pipelines)
        logger.info('Pipelines stopped')
        # then stop accumulator (releasing conditions)
        accumulator.stop_release()
        logger.info('Accumulator stopped')

        # join tasks
        accumulator.join()
        logger.info('Accumulator task closed')
        # wait till all pipeline runs finish then join
        while any_alive(pipelines):
            map(lambda x: x.join(), pipelines)
            time.sleep(0.5)
        logger.info('Pipeline tasks closed')

        # create pipeline report (very basic at the moment)
        make_cal_report(ts,report_path)
        logger.info('Report compiled, in directory {0}/{1}'.format(report_path,ts.experiment_id))

        if full_l1:
            # send L1 stop transmission
            #   wait for a couple of secs before ending transmission
            time.sleep(2.0)
            end_transmit(l1_endpoint)
            logger.info('L1 stream ended')

if __name__ == '__main__':

    opts = parse_opts()

    # set up logging
    setup_logger(opts.log_path)

    # threading or multiprocessing imports
    if opts.threading is False:
        import multiprocessing as control_method
        from multiprocessing import Process as control_task
        from ctypes import c_float, c_ubyte, c_double, c_int
    else:
        import threading as control_method
        from threading import Thread as control_task

    def force_exit(_signo=None, _stack_frame=None):
        logger.info("Exiting katsdpcal on SIGTERM")
        raise SystemExit

    signal.signal(signal.SIGTERM, force_exit)
     # mostly needed for Docker use since this process runs as PID 1
     # and does not get passed sigterm unless it has a custom listener

    run_threads(opts.telstate,
           cbf_n_chans=opts.cbf_channels, antenna_mask=opts.antenna_mask,
           num_buffers=opts.num_buffers, buffer_maxsize=opts.buffer_maxsize,
           l0_endpoint=opts.l0_spectral_spead[0], l1_endpoint=opts.l1_spectral_spead,
           l1_rate=opts.l1_rate, full_l1=opts.full_l1, mproc=not(opts.threading),
           report_path=opts.report_path)
