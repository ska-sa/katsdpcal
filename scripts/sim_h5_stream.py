
import optparse
import sys

from katcal.simulator import SimData

from katsdptelstate.telescope_state import TelescopeState

# ----------------------------------------------------------
# H5 file to use for simulation
#   simulation of data and Teselcope Model (TM)

parser = optparse.OptionParser(usage="%prog [options] <filename.h5>", description='Simulate SPEAD data stream from H5 file.')
parser.add_option("--ts_db", default=1, type="int", help="Telescope state database number. default: 1")
parser.add_option("--ts_ip", default="127.0.0.1", help="Telescope state ip address. default: 127.0.0.1")
(options, args) = parser.parse_args()

if len(args) < 1 or not args[0].endswith(".h5"):
    print "Please provide an H% filename as argument"
    sys.exit(1)
        
file_name = args[0]
simdata = SimData(file_name)

# use TM set up by sim_h5_ts.py and run_cal.py scripts
ts = TelescopeState(host=options.ts_ip,db=options.ts_db)
    
# select data to transmit         
simdata.select(channels=slice(ts.bchan,ts.echan))

# transmit data
simdata.h5toSPEAD(ts,8890)
print 'TX: ended.'

