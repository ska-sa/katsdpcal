#!/usr/bin/env python3
import time
import os
import signal
import logging
import numpy as np
import json
import asyncio
import argparse

import jsonschema

from katsdptelstate import endpoint
import katsdpservices

from katsdpcal.control import create_server, create_buffer_arrays, FlagsStream
from katsdpcal.pipelineprocs import (
    register_argparse_parameters, parameters_from_file, parameters_from_argparse,
    finalise_parameters, parameters_to_telstate)

from katsdpcal import param_dir, rfi_dir


FLAGS_STREAMS_SCHEMA = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "endpoints": {"type": "string"},
            "src_stream": {"type": "string"},
            "interface": {"type": "string"},
            "rate_ratio": {
                "type": "number",
                "minimum": 1.0,
                "exclusiveMinimum": True
            },
            "continuum_factor": {
                "type": "integer",
                "minimum": 1,
                "default": 1
            }
        },
        "additionalProperties": False,
        "required": ["name", "endpoints", "src_stream", "rate_ratio"]
    }
}

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
    for key, value in dictionary.items():
        if isinstance(value, dict):
            logger.info('%s%s%s%s', ident, braces * '[', key, braces * ']')
            log_dict(value, ident+'  ', braces+1)
        else:
            logger.info('%s%s = %s', ident, key, value)


def diagnostics_endpoint(value):
    try:
        return int(value)
    except ValueError:
        addr = endpoint.endpoint_parser(None)(value)
        if addr.port is None:
            raise argparse.ArgumentTypeError('port missing')
        return (addr.host, addr.port)


def parse_flags_streams(data, parser):
    try:
        schema = FLAGS_STREAMS_SCHEMA
        validator_cls = jsonschema.validators.validator_for(schema)
        validator_cls.check_schema(schema)
        validator = validator_cls(schema, format_checker=jsonschema.FormatChecker())
        validator.validate(data)
        return [
            FlagsStream(
                name=item['name'],
                endpoints=endpoint.endpoint_list_parser(7202)(item['endpoints']),
                src_stream=item['src_stream'],
                interface_address=katsdpservices.get_interface_address(item.get('interface')),
                rate_ratio=item['rate_ratio'],
                continuum_factor=int(item.get('continuum_factor', 1)))
            for item in data
        ]
    except (ValueError, jsonschema.ValidationError) as exc:
        parser.error(str(exc))


def parse_opts():
    parser = katsdpservices.ArgumentParser(
        description='Set up and wait for spead stream to run the pipeline.')
    parser.add_argument(
        '--buffer-maxsize', type=float, default=20e9,
        help='The amount of memory (in bytes) to allocate for buffer.')
    parser.add_argument(
        '--max-scans', type=int,
        help='Maximum number of scans per capture block to show in the report.')
    parser.add_argument(
        '--l0-spead', type=endpoint.endpoint_list_parser(7200, single_port=True),
        default=':7200',
        help='endpoints to listen for L0 spead stream (including multicast IPs). '
        + '[<ip>[+<count>]][:port]. [default=%(default)s]', metavar='ENDPOINTS')
    parser.add_argument(
        '--l0-interface',
        help='interface to subscribe to for L0 spectral data. [default: auto]', metavar='INTERFACE')
    parser.add_argument(
        '--l0-name', default='sdp_l0',
        help='Name of the L0 stream for telstate metadata. [default: %(default)s]', metavar='NAME')
    parser.add_argument(
        '--cal-name', default='cal',
        help='Name of the cal output in telstate. [default: %(default)s]', metavar='NAME')
    parser.add_argument(
        '--flags-spead',
        help='[DEPRECATED] endpoints for L1 flags. [default: none]', metavar='ENDPOINTS')
    parser.add_argument(
        '--flags-name', type=str, default='sdp_l1_flags',
        help='[DEPRECATED] name for the flags stream. [default: %(default)s]', metavar='NAME')
    parser.add_argument(
        '--flags-interface',
        help='[DEPRECATED] interface to send flags stream to. [default: auto]',
        metavar='INTERFACE')
    parser.add_argument(
        '--flags-rate-ratio', type=float, default=8.0,
        help='[DEPRECATED] speed to send flags, relative to incoming rate. [default: %(default)s]',
        metavar='RATIO')
    parser.add_argument(
        '--flags-streams', type=json.loads, default=[],
        help='JSON document describing flags streams to send. [default: none]', metavar='JSON')
    parser.add_argument(
        '--clock-ratio', type=float, default=1.0,
        help='Scale factor for transmission rate, smaller is faster [default: %(default)s]')
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
        '--dask-diagnostics', type=diagnostics_endpoint, metavar='HOST:PORT',
        help='Provide a web server with dask diagnostics (PORT binds to localhost, '
             'but :PORT binds to all interfaces). [default: none]')
    parser.add_argument(
        '--pipeline-profile', type=str, metavar='FILENAME',
        help='Write a file with a profile of the pipeline process. [default: none]')
    parser.add_argument(
        '--port', '-p', type=int, default=2048, help='katcp host port [%(default)s]')
    parser.add_argument(
        '--host', '-a', type=str, default='', help='katcp host address [all hosts]')
    parser.add_argument(
        '--servers', type=int, default=1,
        help='number of parallel servers producing the output [default: %(default)s]')
    parser.add_argument(
        '--server-id', type=int, default=1,
        help='index of this server amongst parallel servers (1-based) [default: %(default)s]')
    register_argparse_parameters(parser)
    args = parser.parse_args()
    if args.telstate is None:
        parser.error('--telstate is required')
    # This is done here rather than as a type converter because it needs to
    # happen even when args are parsed as a structure in telstate.
    args.flags_streams = parse_flags_streams(args.flags_streams, parser)
    if args.flags_spead is not None:
        data = [{
            'name': args.flags_name,
            'endpoints': args.flags_spead,
            'src_stream': args.l0_name,
            'interface': args.flags_interface,
            'rate_ratio': args.flags_rate_ratio
        }]
        args.flags_streams += parse_flags_streams(data, parser)

    return args


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


async def run(opts, log_path, full_log):
    """
    Run the device server.

    Parameters
    ----------
    opts : argparse.Namespace
        Command-line options
    log_path : str
        Path for pipeline logs
    """
    # deal with required input parameters
    telstate_l0 = opts.telstate.view(opts.l0_name)
    telstate_cal = opts.telstate.view(opts.cal_name)
    n_chans_all = telstate_l0['n_chans']

    # determine parameter file to use
    if opts.parameter_file == '':
        if n_chans_all == 1024:
            param_filename = 'pipeline_parameters_meerkat_L_1k.txt'
            param_file = os.path.join(param_dir, param_filename)
            logger.info('Parameter file for 1k mode: %s', param_file)
            rfi_filename = 'rfi_mask1K.pickle'
            rfi_file = os.path.join(rfi_dir, rfi_filename)
            logger.info('RFI mask file for 1k mode: %s', rfi_file)
        elif n_chans_all == 4096:
            param_filename = 'pipeline_parameters_meerkat_L_4k.txt'
            param_file = os.path.join(param_dir, param_filename)
            logger.info('Parameter file for 4k mode: %s', param_file)
            rfi_filename = 'rfi_mask.pickle'
            rfi_file = os.path.join(rfi_dir, rfi_filename)
            logger.info('RFI mask file for 4k mode: %s', rfi_file)
        else:
            param_filename = 'pipeline_parameters_meerkat_L_32k.txt'
            param_file = os.path.join(param_dir, param_filename)
            logger.info('Parameter file for 32k mode: %s', param_file)
            rfi_filename = 'rfi_mask32K.pickle'
            rfi_file = os.path.join(rfi_dir, rfi_filename)
            logger.info('RFI mask file for 32k mode: %s', rfi_file)
    else:
        param_file = opts.parameter_file
        logger.info('Parameter file: %s', param_file)

    logger.info('Loading parameters from parameter file.')
    parameters = parameters_from_file(param_file)
    # Override file settings with command-line settings
    parameters.update(parameters_from_argparse(opts))
    logger.info('Finalising parameters')
    parameters = finalise_parameters(parameters, telstate_l0,
                                     opts.servers, opts.server_id - 1, rfi_file)
    parameters_to_telstate(parameters, telstate_cal, opts.l0_name)

    nant = len(parameters['antennas'])
    # number of baselines (may include autocorrelations)
    nbl = len(parameters['bls_ordering'])
    npols = len(parameters['bls_pol_ordering'])
    n_chans = len(parameters['channel_freqs'])

    logger.info('Pipeline system input parameters')
    logger.info('   - antennas: %s', nant)
    logger.info('   - number of channels: %d (of %d)', n_chans, n_chans_all)
    logger.info('   - number of polarisation products: %s', npols)

    if nant < 4:
        # if we have less than four antennas, no katsdpcal necessary
        logger.info('Only %d antenna(s) present - stopping katsdpcal', nant)
        return

    # buffer needs to include:
    #   with shape (time, channel, pol, baseline):
    #   - visibilities, type complex64 (8 bytes)
    #   - flags, type uint8 (1 byte)
    #   - excision bitmask (1/8 byte)
    #   - weights, type float32 (4 bytes)
    #   time, shape(time), type float64 (8 bytes)
    # plus minimal extra for scan transition indices
    scale_factor = 8. + 1. + 4. + 0.125  # vis + flags + weights + excision
    time_factor = 8. + 0.1  # time + 0.1 for good measure (indiced)
    array_length = opts.buffer_maxsize/((scale_factor*n_chans*npols*nbl) + time_factor)
    array_length = np.int(np.ceil(array_length))
    logger.info('Buffer size : %f GB', opts.buffer_maxsize / 1e9)
    logger.info('Total slots in buffer : %d', array_length)

    # Set up empty buffers
    buffer_shape = [array_length, n_chans, npols, nbl]
    buffers = create_buffer_arrays(buffer_shape, not opts.threading)

    logger.info('Receiving L0 data from %s via %s',
                endpoint.endpoints_to_str(opts.l0_spead),
                'default interface' if opts.l0_interface is None else opts.l0_interface)
    l0_interface_address = katsdpservices.get_interface_address(opts.l0_interface)
    if opts.flags_spead is not None:
        logger.info('Sending L1 flags to %s via %s',
                    endpoint.endpoints_to_str(opts.flags_spead),
                    'default interface' if opts.flags_interface is None else opts.flags_interface)
    else:
        logger.info('L1 flags not being sent')

    # Suppress SIGINT, so that the children inherit SIG_IGN. This ensures that
    # pressing Ctrl-C in a terminal will only deliver SIGINT to the parent.
    # There is a small window here where pressing Ctrl-C will have no effect at
    # all, which could be fixed in Python 3 with signal.pthread_sigmask.
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    server = create_server(not opts.threading, opts.host, opts.port, buffers,
                           opts.l0_name, opts.l0_spead, l0_interface_address,
                           opts.flags_streams, opts.clock_ratio,
                           telstate_cal, parameters, opts.report_path, log_path, full_log,
                           opts.dask_diagnostics, opts.pipeline_profile, opts.workers,
                           opts.max_scans)
    with server:
        await server.start()

        # TODO: set up aiomonitor

        # Now install the signal handlers (which won't be inherited).
        loop = asyncio.get_event_loop()

        async def shutdown(server, force):
            await server.shutdown(force=force)
            # If we were shut down by a katcp request, give a bit of time for the reply
            # to be sent, just to avoid getting warnings.
            await asyncio.sleep(0.01)
            await server.stop()

        def signal_handler(signum, force):
            loop.remove_signal_handler(signum)
            # If we were asked to do a graceful shutdown, the next attempt should
            # be to try a hard kill, before going back to the default handler.
            if not force:
                loop.add_signal_handler(signum, signal_handler, signum, True)
            asyncio.ensure_future(shutdown(server, force))

        loop.add_signal_handler(signal.SIGTERM, signal_handler, signal.SIGTERM, True)
        loop.add_signal_handler(signal.SIGINT, signal_handler, signal.SIGINT, False)
        logger.info('katsdpcal started')

        # Run until shutdown
        await server.join()
        logger.info('Server stopped')


async def main():
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

    await run(opts, log_path=log_path, full_log=log_name)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
