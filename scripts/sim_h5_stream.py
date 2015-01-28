
import optparse
import pickle
import sys

from katcal import parameters
from katcal.simulator import SimData
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

# use TM set up by sim_h5_tm.py and run_cal.py scripts
tm = TelescopeModel(host='127.0.0.1',db=1)
    
# select data to transmit    
BCHAN = tm['bchan']
ECHAN = tm['echan']      
simdata.select(channels=slice(BCHAN,ECHAN))

# transmit data
simdata.h5toSPEAD(tm,8890)
print 'TX: ended.'

