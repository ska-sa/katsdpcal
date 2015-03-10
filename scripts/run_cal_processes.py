#! /usr/bin/env python
import numpy as np
import threading
import time
import os

import multiprocessing
from ctypes import c_float, c_ubyte, c_double, c_int

from katcal.control_threads import accumulator_thread, pipeline_thread

from katcal.simulator import SimData
from katcal import parameters

from katsdptelstate.telescope_state import TelescopeState
from katsdptelstate import endpoint, ArgumentParser

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def parse_opts():
    parser = ArgumentParser(description = 'Set up and wait for a spead stream to run the pipeline.')    
    parser.add_argument('--num-buffers', type=int, default=2, help='Specify the number of data buffers to use. default: 2')
    parser.add_argument('--buffer-maxsize', type=float, default=1000e6, help='The amount of memory (in bytes?) to allocate to each buffer. default: 1e9')    
    parser.add_argument('--l0-spectral-spead', type=endpoint.endpoint_list_parser(7200, single_port=True), default=':7200', help='endpoints to listen for L0 SPEAD stream (including multicast IPs). [<ip>[+<count>]][:port]. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument('--l1-spectral-spead', type=endpoint.endpoint_parser(7202), default='127.0.0.1:7202', help='destination for spectral L1 output. [default=%(default)s]', metavar='ENDPOINT')
    parser.set_defaults(telstate='localhost')
    return parser.parse_args()

def all_alive(process_list):
    """
    Check if all of the process in process list are alive and return True
    if they are or False if they are not.
    
    Inputs
    ======
    process_list:  list of threading:Thread: objects
    """

    alive = True
    for process in process_list:
        alive = alive and process.is_alive()
    return alive
            
def create_buffer_arrays(buffer_shape):
    """
    Create empty buffer record using specified dimensions
    """
    data={}
    array_length = buffer_shape[0]
    buffer_size = reduce(lambda x, y: x*y, buffer_shape)
    data['vis'] = multiprocessing.RawArray(c_float, buffer_size*2) # times two for complex
    data['flags'] = multiprocessing.RawArray(c_ubyte, buffer_size)
    data['weights'] = multiprocessing.RawArray(c_float, buffer_size)
    data['times'] = multiprocessing.RawArray(c_double, array_length)
    # assume max 1000 scans!
    data['track_start_indices'] =  multiprocessing.sharedctypes.RawArray(c_int, 1000) 
    
#    data['vis'] = np.empty([array_length,nchan,npol,nbl],dtype=np.complex64)
#    data['flags'] = np.empty([array_length,nchan,npol,nbl],dtype=np.uint8)
#    data['weights'] = np.empty([array_length,nchan,npol,nbl],dtype=np.float64)
#    data['times'] = np.empty([array_length],dtype=np.float)
#    data['track_start_indices'] = []
    return data

def run_threads(ts, num_buffers=2, buffer_maxsize=1000e6, 
           l0_endpoint=':7200', l1_endpoint='127.0.0.1:7202'):
    """
    Start the pipeline using 'num_buffers' buffers, each of size 'buffer_maxsize'.
    This will instantiate num_buffers + 1 threads; a thread for each pipeline and an
    extra accumulator thread the reads data from the spead stream into each buffer
    seen by the pipeline.
    
    Inputs
    ======
    ts: TelescopeState
        The telescope state, default: 'localhost' database 0
    num_buffers: int
        The number of buffers to use- this will create a pipeline thread for each buffer
        and an extra accumulator thread to read the spead stream.
    buffer_maxsize: float
        The maximum size of the buffer. Memory for each buffer will be allocated at first
        and then populated by the accumulator from the spead stream.
    l0_endpoint: endpoint
        Endpoint to listen to for L0 stream, default: ':7200'
    l1_endpoint: endpoint
        Destination endpoint for L1 stream, default: '127.0.0.1:7202'
    """ 

    # extract data shape parameters from TS
    nchan = ts.cbf_n_chans
    npol = 4
    nant = ts.cbf_n_ants
    # number of baselines includes autocorrelations
    nbl = nant*(nant+1)/2
    
    # buffer needs to include:
    #   visibilities, shape(time,channel,baseline,pol), type complex64 (8 bytes)
    #   flags, shape(time,channel,baseline,pol), type int8 (? confirm)
    #   weights, shape(time,channel,baseline,pol), type int8 (? confirm)
    #   time, shape(time), type float64 (8 bytes)
    # plus minimal extra for scan transition indices
    scale_factor = 8. + 1. + 1.  # vis + flags + weights
    time_factor = 8.
    array_length = buffer_maxsize/((scale_factor*nchan*npol*nbl) + time_factor)
    array_length = np.int(np.ceil(array_length))
    logger.info('Max length of buffer array : {0}'.format(array_length,))
    
    # Set up empty buffers
    buffer_shape = [array_length,nchan,npol,nbl]
    buffers = [create_buffer_arrays(buffer_shape) for i in range(num_buffers)]

    # set up conditions for the buffers
    scan_accumulator_conditions = [multiprocessing.Condition() for i in range(num_buffers)]
    
    # Set up the accumulator
    accumulator = accumulator_thread(buffers, buffer_shape, scan_accumulator_conditions, l0_endpoint, ts)

    #Set up the pipelines (one per buffer)
    pipelines = [pipeline_thread(buffers[i], buffer_shape, scan_accumulator_conditions[i], i, l1_endpoint, ts) for i in range(num_buffers)]
    
    #Start the pipeline threads
    map(lambda x: x.start(), pipelines)
    # might need delay here for pipeline threads to aquire conditions then wait?
    #time.sleep(5.)
    #Start the accumulator thread
    accumulator.start()

    try:
        while all_alive([accumulator]+pipelines):

            time.sleep(.1)
    except (KeyboardInterrupt, SystemExit):
        print '\nReceived keyboard interrupt! Quitting threads.\n'
        #Stop pipelines first so they recieve correct signal before accumulator acquires the condition
        map(lambda x: x.stop(), pipelines)
        accumulator.stop()
    except:
        print '\nUnknown error\n'
        map(lambda x: x.stop(), pipelines)
        accumulator.stop()
        
    accumulator.join()
    print "Accumulator Stopped"

    map(lambda x: x.join(), pipelines)
    print "Pipelines Stopped"


if __name__ == '__main__':
    
    opts = parse_opts()

    # short weit to give me time to start up the simulated spead stream
    # time.sleep(5.)

    run_threads(opts.telstate, num_buffers=opts.num_buffers, buffer_maxsize=opts.buffer_maxsize, 
           l0_endpoint=opts.l0_spectral_spead[0], l1_endpoint=opts.l1_spectral_spead)
