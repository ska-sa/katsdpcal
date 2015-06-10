#!/usr/bin/env python
# ----------------------------------------------------------
# H5 file to use for simulation
#   simulation of data and Teselcope Model (TM)

from katsdpcal import parameters
from katsdpcal.simulator import SimData
from katsdptelstate.telescope_state import TelescopeState
from katsdptelstate import endpoint, ArgumentParser

def parse_opts():
    parser = ArgumentParser(description = 'Simulate Telescope State H5 file')    
    parser.add_argument('--h5file', type=str, help='H5 file for simulated data')
    parser.set_defaults(telstate='localhost')
    return parser.parse_args()

opts = parse_opts()
ts = opts.telstate

print "Use parameters from parameter file to initialise TS."
parameters.init_ts(ts)

print "Open H5 file using appropriate reference antenna for sensor reference."
simdata = SimData(opts.h5file,ts.cal_refant)

print "Add and override TS data from simulator."
simdata.setup_ts(ts)
print "Done."

