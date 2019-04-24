#!/usr/bin/env python
# ----------------------------------------------------------
# Simulate the data stream from a file

from katsdpcal.simulator import init_simdata
from katsdpcal import calprocs

from argparse import ArgumentParser
import numpy as np
import time

def average_timer(n,func,*args,**kwargs):
    beg_ts = time.time()
    for ni in range(n):
        func(*args,**kwargs)
    end_ts = time.time()
    print("{0: .4f}".format((end_ts - beg_ts)/n,))

def parse_opts():
    parser = ArgumentParser(description='Benchmark StEFcal implementation')

    parser.add_argument('-n', '--niter', type=int, default=100, help='Number of iterations for timer.')
    parser.add_argument('--file', type=str, default='', help='File for simulated data (H5).')
    parser.add_argument('--nants', type=int, default=7, help='Number of antennas for simulated array (if no file is provided).')
    parser.add_argument('--nchans', type=int, default=1, help='Number of channels for simulated array (if no file is provided).')
    parser.add_argument('--noise', action='store_true', help='Add noise to simulated data (if no file is provided).')
    parser.set_defaults(noise=False)

    parser.set_defaults(telstate='localhost')
    return parser.parse_args()

opts = parse_opts()

if opts.file == '':
    # if we are not provided with a file, simulate interferometer data
    print()
    print('Data: simulated {0} antenna array'.format(opts.nants,))

    vis_av, bls_lookup, gains = calprocs.fake_vis(opts.nants, noise=False)
    if opts.nchans > 1: vis_av = np.repeat(vis_av[:,np.newaxis],opts.nchans,axis=1).T

else:
    # if we are provided with a file, extract data  and metadata from the file
    print()
    print('Data: open file {0}'.format(opts.file,))
    simdata = init_simdata(opts.file)

    print("Data: use HH pol only")
    simdata.select(pol='hh')
    print("Data: use 100 timestamps only")
    vis = simdata.vis[0:100]

    print("Data: average over time")
    vis_av = np.mean(vis,axis=0)
    print("Data: shape {0}".format(vis_av.shape,))
    print()

    # get data parameters for solver
    antlist = ','.join([a.name for a in simdata.ants])
    bls_lookup = calprocs.get_bls_lookup(antlist,simdata.corr_products)

# numter of iterations for the timer
niter = opts.niter
print("Elapsed time (average over {0} iterations):\n{1}".format(niter, '='*43))

average_timer(niter, calprocs.g_fit, vis_av, bls_lookup, conv_thresh=0.01)
