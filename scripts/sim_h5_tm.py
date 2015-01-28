
import optparse

from katcal.simulator import SimData
from katcal import parameters

import sys

from katcal.telescope_model import TelescopeModel

# ----------------------------------------------------------
# H5 file to use for simulation
#   simulation of data and Teselcope Model (TM)

parser = optparse.OptionParser(usage="%prog [options] <filename.h5>", description='Run MeerKAT calibration pipeline on H5 file')
(options, args) = parser.parse_args()

if len(args) < 1 or not args[0].endswith(".h5"):
    print "Please provide an H% filename as argument"
    sys.exit(1)
        
file_name = args[0]
simdata = SimData(file_name)

# create TM
tm = TelescopeModel(host='127.0.0.1',db=1)
# use parameters from parameter file to initialise TM
parameters.init_tm(tm)
# add and override with TM data from simulator 
simdata.setup_TM(tm)