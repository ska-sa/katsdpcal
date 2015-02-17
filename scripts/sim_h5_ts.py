#!/usr/bin/env python
# ----------------------------------------------------------
# H5 file to use for simulation
#   simulation of data and Teselcope Model (TM)

from katcal import parameters
from katcal.simulator import SimData
from katsdptelstate.telescope_state import TelescopeState
from katsdptelstate import endpoint, ArgumentParser

def parse_opts():
    parser = ArgumentParser(description = 'Simulate Telescope State H5 file')    
    parser.add_argument('--h5file', type=str, help='H5 file for simulated data')
    parser.set_defaults(telstate='localhost')
    return parser.parse_args()

opts = parse_opts()
simdata = SimData(opts.h5file)

print "Use parameters from parameter file to initialise TS."
parameters.init_ts(opts.telstate)
print "Add and override with TS data from simulator."
simdata.setup_ts(opts.telstate)
print "Done."

