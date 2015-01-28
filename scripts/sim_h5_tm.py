
import optparse

from katcal.simulator import SimData
from katcal import parameters

import sys

from katsdptelstate.telescope_state import TelescopeState

# ----------------------------------------------------------
# H5 file to use for simulation
#   simulation of data and Teselcope Model (TM)

parser = optparse.OptionParser(usage="%prog [options] <filename.h5>", description='Simulate telescope state from H5 file.')
parser.add_option("--ts_db", default=1, type="int", help="Telescope state database number. default: 1")
parser.add_option("--ts_ip", default="127.0.0.1", help="Telescope state ip address. default: 127.0.0.1")
(options, args) = parser.parse_args()

if len(args) < 1 or not args[0].endswith(".h5"):
    print "Please provide an H% filename as argument"
    sys.exit(1)
        
file_name = args[0]
simdata = SimData(file_name)

# create TM
ts = TelescopeState(host=options.ts_ip,db=options.ts_db)
# use parameters from parameter file to initialise TM
parameters.init_ts(ts)
# add and override with TM data from simulator 
simdata.setup_ts(ts)