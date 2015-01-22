
import optparse
import pickle
import sys

from katcal import parameters
from katcal.simulator import SimData

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

# set up Telescope model
TMfile = 'TM.pickle'
params = parameters.set_params()
TM = simdata.setup_TM(TMfile,params)
    
# select data to transmit    
BCHAN = TM['bchan']
ECHAN = TM['echan']       
simdata.select(channels=slice(BCHAN,ECHAN))

# transmit data
simdata.h5toSPEAD(8890)
print 'TX: ended.'

