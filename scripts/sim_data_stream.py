#!/usr/bin/env python
# ----------------------------------------------------------
# Simulate the data stream from a file

from katsdpcal.simulator import init_simdata
from katsdptelstate.telescope_state import TelescopeState
from katsdptelstate import endpoint, ArgumentParser

def parse_opts():
    parser = ArgumentParser(description = 'Simulate SPEAD data stream from H5 file')    
    parser.add_argument('--l0-spectral-spead', type=endpoint.endpoint_parser(7200), default='127.0.0.1:7200', 
            help='endpoints to listen for L0 SPEAD stream (including multicast IPs). [<ip>[+<count>]][:port]. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument('--file', type=str, help='File for simulated data (H5 or MS)')
    parser.add_argument('--l0-rate', type=float, default=5e7, help='Simulated L0 SPEAD rate. For laptops, recommend rate of 5e7. Default: 5e7')
    parser.add_argument('--max-scans', type=int, default=0, help='Number of scans to transmit. Default: all')
    parser.set_defaults(telstate='localhost')
    return parser.parse_args()

opts = parse_opts()

print "Use TS set up by sim_ts.py and run_cal.py scripts."
ts = opts.telstate

simdata = init_simdata(opts.file)

print "Selecting data to transmit. Slice using ts values."
simdata.select(channels=slice(ts.cal_bchan,ts.cal_echan))

print "TX: start."
max_scans = opts.max_scans if not opts.max_scans == 0 else None
simdata.datatoSPEAD(ts,opts.l0_spectral_spead,opts.l0_rate,max_scans=max_scans)
print "TX: ended."

