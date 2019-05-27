#!/usr/bin/env python3
# ----------------------------------------------------------
# Simulate the data stream from a file

import logging
import asyncio

from katsdpcal.simulator import SimData
from katsdptelstate import endpoint
from katsdpservices import ArgumentParser, setup_logging


logger = logging.getLogger(__name__)


def parse_opts():
    parser = ArgumentParser(description='Simulate SPEAD data stream from file')
    parser.add_argument(
        '--l0-spead', type=endpoint.endpoint_parser(7200), default='127.0.0.1:7200',
        help='endpoint to send L0 SPEAD stream (including multicast IPs). '
             '[<ip>][:port]. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument('--file', type=str, help='File for simulated data (H5 or MS)')
    parser.add_argument(
        '--l0-rate', type=float, default=5e7, metavar='BYTES/S',
        help='Simulated L0 SPEAD rate. For laptops, recommend rate of 5e7. Default: 5e7')
    parser.add_argument(
        '--max-scans', type=int, default=None,
        help='Number of scans to transmit. Default: all')
    parser.add_argument(
        '--server', type=endpoint.endpoint_parser(2048), default='localhost:2048',
        help='Address of cal katcp server. Default: %(default)s', metavar='ENDPOINT')
    parser.add_argument(
        '--bchan', type=int, default=0, help='First channel to take from file')
    parser.add_argument(
        '--echan', type=int, default=None, help='Last channel to take from file')
    parser.set_defaults(telstate='localhost')
    return parser.parse_args()


async def main():
    setup_logging()
    opts = parse_opts()

    logger.info("Use TS set up by sim_ts.py and run_cal.py scripts.")
    telstate = opts.telstate

    simdata = SimData.factory(opts.file, opts.server, bchan=opts.bchan, echan=opts.echan)
    async with simdata:
        logger.info("Issuing capture-init")
        await simdata.capture_init()
        logger.info("TX: start.")
        simdata.data_to_spead(telstate, opts.l0_spead, opts.l0_rate, max_scans=opts.max_scans)
        logger.info("TX: ended.")
        logger.info("Issuing capture-done")
        await simdata.capture_done()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()
