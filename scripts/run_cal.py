#! /usr/bin/env python
import time
import os
import signal
import logging
import manhole
import numpy as np

import trollius
from trollius import From
from tornado.platform.asyncio import AsyncIOMainLoop, to_asyncio_future

from katsdptelstate import endpoint
import katsdpservices

from katsdpcal.control import create_server, create_buffer_arrays
from katsdpcal.pipelineprocs import ts_from_file, setup_ts

from katsdpcal import param_dir, rfi_dir


logger = logging.getLogger(__name__)


def adapt_formatter(taskname):
    """Monkey-patches logging.Formatter to inject the task name into the format string.

    Parameters
    ----------
    taskname : {processName, threadName}
        The property name to inject
    """
    def new_init(self, fmt=None, datefmt=None):
        if fmt is None:
            fmt = '%(message)s'
        fmt = fmt.replace('%(message)s', '[%(' + taskname + ')s] %(message)s')
        old_init(self, fmt, datefmt)
    old_init = logging.Formatter.__init__
    logging.Formatter.__init__ = new_init


def log_dict(dictionary, ident='', braces=1):
    """Recursively logs nested dictionaries.

    Parameters
    ----------
    dictionary : dict
        Dictionary to print
    ident : str
        indentation string
    braces : int
        number of braces to surround item with
    """
    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            logger.info('%s%s%s%s', ident, braces * '[', key, braces * ']')
            log_dict(value, ident+'  ', braces+1)
        else:
            logger.info('%s%s = %s', ident, key, value)


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
        '--buffer-maxsize', type=float,
        help='The amount of memory (in bytes) to allocate for buffer.')
    parser.add_argument(
        '--l0-spectral-spead', type=endpoint.endpoint_list_parser(7200, single_port=True),
        default=':7200',
        help='endpoints to listen for L0 spead stream (including multicast IPs). '
        + '[<ip>[+<count>]][:port]. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument(
        '--l0-spectral-interface',
        help='interface to subscribe to for L0 spectral data. [default: auto]', metavar='INTERFACE')
    parser.add_argument(
        '--l0-spectral-name', default='sdp_l0',
        help='Name of the L0 stream for telstate metadata. [default: %(default)s]', metavar='NAME')
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
        '--threading', action='store_true',
        help='Use threading to control pipeline and accumulator '
        + '[default: use multiprocessing]')
    parser.add_argument(
        '--workers', type=int,
        help='Number of worker threads to use within the pipeline [default: cores - 1]')
    parser.add_argument(
        '--parameter-file', type=str, default='',
        help='Default pipeline parameter file (will be over written by TelescopeState.')
    parser.add_argument(
        '--report-path', type=str, default='/var/kat/data',
        help='Path under which to save pipeline report. [default: /var/kat/data]')
    parser.add_argument(
        '--log-path', type=str, default=os.path.abspath('.'),
        help='Path under which to save pipeline logs. [default: current directory]')
    parser.add_argument(
        '--dask-diagnostics', type=str, metavar='FILENAME.HTML',
        help='Write a file with dask diagnostics (requires bokeh). [default: none]')
    parser.add_argument(
        '--port', '-p', type=int, default=2048, help='katcp host port [%(default)s]')
    parser.add_argument(
        '--host', '-a', type=str, default='', help='katcp host address [all hosts]')
    return parser.parse_args()


def setup_logger(log_name, log_path='.'):
    """
    Set up the pipeline logger.
    The logger writes to a pipeline.log file and to stdout.

    Parameters
    ----------
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


@trollius.coroutine
def run(ts, stream_name, host, port,
        buffer_maxsize=None,
        l0_endpoints=':7200', l0_interface=None,
        l1_endpoint='127.0.0.1:7202', l1_rate=5.0e7, l1_level=0,
        mproc=True, param_file='', report_path='', log_path='.', full_log=None,
        diagnostics_file=None, num_workers=None):
    """
    Run the device server.

    Parameters
    ----------
    ts : TelescopeState
        The telescope state
    stream_name : str
        Name of the L0 input stream, for finding parameters in `ts`
    host : str
        Bind hostname for katcp server
    port : int
        Bind port for katcp server
    buffer_maxsize : float
        The size of the buffer. Memory for each buffer will be allocated at first
        and then populated by the accumulator from the SPEAD stream.
    l0_endpoints : list of :class:`katsdptelstate.endpoint.Endpoint`
        Endpoints to listen to for L0 stream
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
    param_file : str
        File of default pipeline parameters
    report_path : str
        Path under which to save pipeline report
    log_path : str
        Path for pipeline logs
    diagnostics_file : str
        Path to write dask diagnostics
    num_workers : int
        Number of worker threads to use in the pipeline
    """
    ioloop = AsyncIOMainLoop()
    ioloop.install()

    # deal with required input parameters
    n_chans = ts[stream_name + '_n_chans']
    baselines = ts[stream_name + '_bls_ordering']
    ants = set()
    pols = set()
    ant_baselines = set()
    for a, b in baselines:
        ants.add(a[:-1])
        ants.add(b[:-1])
        ant_baselines.add((a[:-1], b[:-1]))
        pols.add((a[-1], b[-1]))
    antlist = list(sorted(ants))

    logger.info('Pipeline system input parameters')
    logger.info('   - antennas: %s', antlist)
    logger.info('   - number of channels: %s', n_chans)
    logger.info('   - number of polarisation products: %s', len(pols))

    if len(antlist) < 4:
        # if we have less than four antennas, no katsdpcal necessary
        logger.info('Only %d antenna(s) present - stopping katsdpcal', len(antlist))
        return

    # initialise TS from default parameter file
    #   defaults are used only for parameters missing from the TS
    if param_file == '':
        if n_chans == 4096:
            param_filename = 'pipeline_parameters_meerkat_ar1_4k.txt'
            param_file = os.path.join(param_dir, param_filename)
            logger.info('Parameter file for 4k mode: %s', param_file)
            rfi_filename = 'rfi_mask.pickle'
            rfi_file = os.path.join(rfi_dir, rfi_filename)
            logger.info('RFI mask file for 4k mode: %s', rfi_file)
        else:
            param_filename = 'pipeline_parameters_meerkat_ar1_32k.txt'
            param_file = os.path.join(param_dir, param_filename)
            logger.info('Parameter file for 32k mode: %s', param_file)
            rfi_filename = 'rfi_mask32K.pickle'
            rfi_file = os.path.join(rfi_dir, rfi_filename)
            logger.info('RFI mask file for 32k mode: %s', rfi_file)
    else:
        logger.info('Parameter file: %s', param_file)
    logger.info('Inputting Telescope State parameters from parameter file.')
    ts_from_file(ts, param_file, rfi_file)
    # telescope state logs for debugging
    logger.info('Telescope state parameters:')
    for keyval in ts.keys():
        # don't print out the really long telescope state key values
        if keyval not in [stream_name + '_bls_ordering', 'cal_channel_freqs']:
            logger.info('%s : %s', keyval, ts[keyval])

    # set up TS for pipeline use
    logger.info('Setting up Telescope State parameters for pipeline.')
    setup_ts(ts, antlist)

    nant = len(antlist)
    # number of baselines (may include autocorrelations)
    nbl = len(ant_baselines)
    npols = len(pols)

    # get buffer size
    if buffer_maxsize is not None:
        ts.add('cal_buffer_size', buffer_maxsize)
    elif 'cal_buffer_size' in ts:
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
    array_length = buffer_maxsize/((scale_factor*n_chans*npols*nbl) + time_factor)
    array_length = np.int(np.ceil(array_length))
    logger.info('Buffer size : %f GB', buffer_maxsize / 1e9)
    logger.info('Total slots in buffer : %d', array_length)

    # Set up empty buffers
    buffer_shape = [array_length, n_chans, npols, nbl]
    buffers = create_buffer_arrays(buffer_shape, mproc)

    logger.info('Receiving L0 data from %s via %s',
                endpoint.endpoints_to_str(l0_endpoints),
                'default interface' if l0_interface is None else l0_interface)
    l0_interface_address = katsdpservices.get_interface_address(l0_interface)

    # Suppress SIGINT, so that the children inherit SIG_IGN. This ensures that
    # pressing Ctrl-C in a terminal will only deliver SIGINT to the parent.
    # There is a small window here where pressing Ctrl-C will have no effect at
    # all, which could be fixed in Python 3 with signal.pthread_sigmask.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    server = create_server(mproc, host, port, buffers,
                           l0_endpoints, l0_interface_address,
                           l1_endpoint, l1_level, l1_rate, ts, stream_name,
                           report_path, log_path, full_log,
                           diagnostics_file, num_workers)
    with server:
        ioloop.add_callback(server.start)

        # allow remote debug connections and expose telescope state and tasks
        manhole.install(oneshot_on='USR1', locals={
            'ts': ts,
            'server': server
        })

        # Now install the signal handlers (which won't be inherited).
        loop = trollius.get_event_loop()

        def signal_handler(signum, force):
            loop.remove_signal_handler(signum)
            # If we were asked to do a graceful shutdown, the next attempt should
            # be to try a hard kill, before going back to the default handler.
            if not force:
                loop.add_signal_handler(signum, signal_handler, signum, True)
            trollius.ensure_future(server.shutdown(force=force))
        loop.add_signal_handler(signal.SIGTERM, signal_handler, signal.SIGTERM, True)
        loop.add_signal_handler(signal.SIGINT, signal_handler, signal.SIGINT, False)
        logger.info('katsdpcal started')

        # Run until shutdown
        yield From(server.join())
        # If we were shut down by a katcp request, give a bit of time for the reply
        # to be sent, just to avoid getting warnings.
        yield From(trollius.sleep(0.01))
        yield From(to_asyncio_future(server.stop()))
        logger.info('Server stopped')


@trollius.coroutine
def main():
    opts = parse_opts()

    # set up logging. The Formatter class is replaced so that all log messages
    # show the process/thread name.
    if opts.threading:
        adapt_formatter('threadName')
    else:
        adapt_formatter('processName')
    log_name = 'pipeline.log'
    log_path = os.path.abspath(opts.log_path)
    setup_logger(log_name, log_path)

    yield From(run(
        opts.telstate, opts.l0_spectral_name,
        host=opts.host, port=opts.port,
        buffer_maxsize=opts.buffer_maxsize,
        l0_endpoints=opts.l0_spectral_spead, l0_interface=opts.l0_spectral_interface,
        l1_endpoint=opts.l1_spectral_spead,
        l1_rate=opts.l1_rate, l1_level=opts.l1_level, mproc=not opts.threading,
        param_file=opts.parameter_file,
        report_path=opts.report_path, log_path=log_path, full_log=log_name,
        diagnostics_file=opts.dask_diagnostics, num_workers=opts.workers))


if __name__ == '__main__':
    trollius.get_event_loop().run_until_complete(main())
    trollius.get_event_loop().close()
