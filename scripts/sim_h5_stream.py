
import optparse
import sys

from katcal.simulator import SimData
from katcal.telescope_model import TelescopeModel

# ----------------------------------------------------------
# H5 file to use for simulation
#   simulation of data and Teselcope Model (TM)

parser = optparse.OptionParser(usage="%prog [options] <filename.h5>", description='Simulate SPEAD data stream from H5 file.')
parser.add_option("--tm_db", default=1, type="int", help="Telescope model database number. default: 1")
parser.add_option("--tm_ip", default="127.0.0.1", help="Telescope model ip address. default: 127.0.0.1")
(options, args) = parser.parse_args()

if len(args) < 1 or not args[0].endswith(".h5"):
    print "Please provide an H% filename as argument"
    sys.exit(1)
        
file_name = args[0]
simdata = SimData(file_name)

# use TM set up by sim_h5_tm.py and run_cal.py scripts
tm = TelescopeModel(host=options.tm_ip,db=options.tm_db)
    
# select data to transmit         
simdata.select(channels=slice(tm.bchan,tm.echan))

# transmit data
simdata.h5toSPEAD(tm,8890)
print 'TX: ended.'

