#!/usr/bin/env python
# ----------------------------------------------------------
# Simulate the Telescope State from a file

from katsdpcal import pipelineprocs, conf_dir
from katsdpcal.simulator import init_simdata
from katsdptelstate import ArgumentParser
import os

def parse_opts():
    parser = ArgumentParser(description = 'Simulate Telescope State from h5 or MS file')    
    parser.add_argument('--file', type=str, help='H5 or MS file for simulated data')
    parser.add_argument('--parameters', type=str, default='', help='Default pipeline parameter file (will be over written by TelescopeState.')
    parser.set_defaults(telstate='localhost')
    return parser.parse_args()

opts = parse_opts()
ts = opts.telstate

print "Clear TS."
pipelineprocs.clear_ts(ts)

print "Open file"
simdata = init_simdata(opts.file)

print "Add to and override TS data from simulator."
simdata.setup_ts(ts)

print "Use parameters from parameter file."
param_file = opts.parameters
if param_file == '':
    if ts.cbf_n_chans == 4096:
        param_filename = 'pipeline_parameters_meerkat_ar1_4k.txt'
        param_file = os.path.join(conf_dir,param_filename)
        print 'Parameter file for 4k mode: {0}'.format(param_file,)
    else:
        param_filename = 'pipeline_parameters_meerkat_ar1_32k.txt'
        param_file = os.path.join(conf_dir,param_filename)
        print 'Parameter file for 32k mode: {0}'.format(param_file,)
pipelineprocs.ts_from_file(ts, param_file)

print "Done."

