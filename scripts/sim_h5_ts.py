#!/usr/bin/env python
# ----------------------------------------------------------
# H5 file to use for simulation
#   simulation of data and Teselcope Model (TM)

import optparse

from katcal import parameters
from katcal.simulator import SimData
from katsdptelstate.telescope_state import TelescopeState

def get_args():
    parser = optparse.OptionParser(usage="%prog [options] <filename.h5>", description="Simulate telescope state from H5 file.")
    parser.add_option("--ts-db", default=1, type="int", help="Telescope state database number. Default: 1")
    parser.add_option("--ts-ip", default="127.0.0.1", help="Telescope state ip address. Default: 127.0.0.1")
    options, args = parser.parse_args()
    if len(args) < 1 or not args[0].endswith(".h5"):
        parser.error("Please provide an h5 filename as an argument.")
    return args[0], options

filename, options = get_args()
simdata = SimData(filename)

print "Create TS."
ts = TelescopeState(host=options.ts_ip,db=options.ts_db)
print "Use parameters from parameter file to initialise TS."
parameters.init_ts(ts)
print "Add and override with TS data from simulator."
simdata.setup_ts(ts)
print "Done."

