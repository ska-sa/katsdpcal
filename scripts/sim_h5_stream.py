#!/usr/bin/env python
# ----------------------------------------------------------
# H5 file to use for simulation
#   simulation of data and Teselcope State (TS)


import optparse

from katcal.simulator import SimData
from katsdptelstate.telescope_state import TelescopeState

def get_args():
    parser = optparse.OptionParser(usage="%prog [options] <filename.h5>", description='Simulate SPEAD data stream from H5 file.')
    parser.add_option("--ts-db", default=1, type="int", help="Telescope state database number. default: 1")
    parser.add_option("--ts-ip", default="127.0.0.1", help="Telescope state ip address. default: 127.0.0.1")
    parser.add_option("--l0-spectral-spead", default="127.0.0.1:8890", help="destination host:port for spectral L0 input. default: 127.0.0.1:8890")
    (options, args) = parser.parse_args()
    if len(args) < 1 or not args[0].endswith(".h5"):
        parser.error("Please provide a .h5 filename as argument")
    return args[0], options

filename, options = get_args()
simdata = SimData(filename)

print "Use TS set up by sim_h5_ts.py and run_cal.py scripts."
ts = TelescopeState(endpoint=options.ts_ip,db=options.ts_db)

print "Selecting data to transmit. Slice using ts values."
simdata.select(channels=slice(ts.bchan,ts.echan))

print "TX: start."
l0_ip, l0_port = options.l0_spectral_spead.split(':')
simdata.h5toSPEAD(ts,int(l0_port),l0_ip) 
print "TX: ended."

