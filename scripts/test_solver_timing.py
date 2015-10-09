#!/usr/bin/env python
# ----------------------------------------------------------
# Simulate the data stream from a file

from katsdpcal.simulator import SimData
from katsdpcal import calprocs

from argparse import ArgumentParser
import numpy as np
import time

ALGORITHMS = ['adi','adi_nonnumpy','schwardt','adi_schwardt']

def average_timer(n,func,*args,**kwargs):
    beg_ts = time.time()
    for ni in range(n):
        func(*args,**kwargs)
    end_ts = time.time()
    print("{0: .4f}".format((end_ts - beg_ts)/n,))

def parse_opts():
    parser = ArgumentParser(description = 'Simulate SPEAD data stream from H5 file')    

    parser.add_argument('--file', type=str, help='File for simulated data (H5)')
    parser.add_argument('-n', '--niter', type=int, default=1, help='Numer of iterations for timer.')
    parser.set_defaults(telstate='localhost')
    return parser.parse_args()

opts = parse_opts()

print
print 'Data: open file {0}'.format(opts.file,)
simdata = SimData(opts.file)

print "Data: use HH pol only"
simdata.select(pol='hh')
print "Data: use 100 timestamps only"
vis = simdata.vis[0:100]

print "Data: average over time"
vis_av = np.mean(vis,axis=0)
print "Data: shape {0}".format(vis_av.shape,)
print

# get data parameters for solver
antlist = ','.join([a.name for a in simdata.ants])
bls_lookup = calprocs.get_bls_lookup(antlist,simdata.corr_products)

# numter of iterations for the timer
niter = opts.niter
print("Elapsed time (average over {0} iterations):\n{1}".format(niter, '='*41))

# time each algorithm
for algorithm in ALGORITHMS:
    print 'Algorithm {0: <14}:'.format(algorithm,),
    average_timer(niter, calprocs.g_fit, vis_av, bls_lookup, algorithm=algorithm, conv_thresh=0.01)
