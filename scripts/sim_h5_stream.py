#!/usr/bin/env python
# ----------------------------------------------------------
# H5 file to use for simulation
#   simulation of data and Teselcope State (TS)

from katcal.simulator import SimData
from katsdptelstate.telescope_state import TelescopeState
from katsdptelstate import endpoint, ArgumentParser

def parse_opts():
    parser = ArgumentParser(description = 'Simulate SPEAD data stream from H5 file')    
    parser.add_argument('--l0-spectral-spead', type=endpoint.endpoint_parser(7200), default='127.0.0.1:7200', 
            help='endpoints to listen for L0 SPEAD stream (including multicast IPs). [<ip>[+<count>]][:port]. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument('--h5file', type=str, help='H5 file for simulated data')
    parser.add_argument('--spead-rate', type=float, default=1e9, help='SPEAD rate. For laptops, recommend rate of 1e7. Default: 1e9')
    parser.add_argument('--max-scans', type=int, default=0, help='Number of scans to transmit. Default: all')
    parser.set_defaults(telstate='localhost')
    return parser.parse_args()

opts = parse_opts()

print "Use TS set up by sim_h5_ts.py and run_cal.py scripts."
ts = opts.telstate

simdata = SimData(opts.h5file,ts.cal_refant)

print "Selecting data to transmit. Slice using ts values."
simdata.select(channels=slice(ts.cal_bchan,ts.cal_echan))

print "TX: start."
max_scans = opts.max_scans if not opts.max_scans == 0 else None
simdata.h5toSPEAD(ts,opts.l0_spectral_spead,opts.spead_rate,max_scans=max_scans) 
print "TX: ended."

