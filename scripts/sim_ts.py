#!/usr/bin/env python
# ----------------------------------------------------------
# Simulate the Telescope State from a file

import os
import logging

from katsdpcal import pipelineprocs, param_dir
from katsdpcal.simulator import SimData
from katsdpservices import ArgumentParser, setup_logging


logger = logging.getLogger(__name__)


def parse_opts():
    parser = ArgumentParser(description = 'Simulate Telescope State from h5 or MS file')    
    parser.add_argument('--file', type=str, help='H5 or MS file for simulated data')
    parser.add_argument('--bchan', type=int, default=0, help='First channel to take from file')
    parser.add_argument('--echan', type=int, default=None, help='Last channel to take from file')
    parser.set_defaults(telstate='localhost')
    return parser.parse_args()


def main():
    setup_logging()
    opts = parse_opts()
    telstate = opts.telstate

    logging.info("Opening file %s", opts.file)
    simdata = SimData(opts.file, bchan=opts.bchan, echan=opts.echan)

    logging.info("Clearing telescope state")
    telstate.clear()

    logging.info("Setting values in telescope state")
    simdata.setup_telstate(telstate)


if __name__ == '__main__':
    main()
