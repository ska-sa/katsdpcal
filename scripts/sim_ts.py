#!/usr/bin/env python
# ----------------------------------------------------------
# Simulate the Telescope State from a file

from katsdpcal import pipelineprocs, conf_dir, param_file
from katsdpcal.simulator import init_simdata
from katsdptelstate import ArgumentParser
import os

def parse_opts():
    parser = ArgumentParser(description = 'Simulate Telescope State from h5 or MS file')    
    parser.add_argument('--file', type=str, help='H5 or MS file for simulated data')
    parser.add_argument('--parameters', type=str, default=os.path.join(conf_dir,param_file), help='Default pipeline parameter file (will be over written by TelescopeState. [default: {0}]'.format(param_file,))
    parser.set_defaults(telstate='localhost')
    return parser.parse_args()

opts = parse_opts()
ts = opts.telstate

print "Use parameters from parameter file to initialise TS."
pipelineprocs.clear_ts(ts)
pipelineprocs.ts_from_file(ts, opts.parameters)

print "Open file"
simdata = init_simdata(opts.file)

print "Add to and override TS data from simulator."
simdata.setup_ts(ts)
print "Done."
