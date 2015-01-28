
#! /usr/bin/env python
import numpy as np
import threading
import time
import os

from katcal.control_threads import accumulator_thread, pipeline_thread

import optparse

from katcal.simulator import SimData
from katcal import parameters

from katsdptelstate.telescope_state import TelescopeState

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from optparse import OptionParser

def parse_args():
    usage = "%s [options]"%os.path.basename(__file__)
    description = "Set up and wait for a spead stream to run the pipeline."
    parser= OptionParser( usage=usage, description=description)
    parser.add_option("--num_buffers", default=2, type="int", help="Specify the number of data buffers to use. default: 2")
    parser.add_option("--buffer_maxsize", default=1000e6, type="float", help="The amount of memory (in bytes?) to allocate to each buffer. default: 1e9")
    parser.add_option("--spead_port", default=8890, type="int", help="The port on which to listen for the spead stream. default: 8890")
    parser.add_option("--spead_ip", default="localhost", help="The ip to listen for the spead streap. default: localhost")
    parser.add_option("--ts_db", default=1, type="int", help="Telescope state database number. default: 1")
    parser.add_option("--ts_ip", default="127.0.0.1", help="Telescope state ip address. default: 127.0.0.1")
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
        alive = alive and process.isAlive()
    return alive
            
def create_buffer_arrays(array_length,nchan,nbl,npol):
    """
    Create empty buffer record using specified dimensions
    """
    data={}
    data['vis'] = np.empty([array_length,nchan,nbl,npol],dtype=np.complex64)
    data['flags'] = np.empty([array_length,nchan,nbl,npol],dtype=np.uint8)
    data['weights'] = np.empty([array_length,nchan,nbl,npol],dtype=np.float64)
    data['times'] = np.empty([array_length],dtype=np.float)
    data['track_start_indices'] = []
    return data

def run_threads(num_buffers=2, buffer_maxsize=1000e6, spead_port=8890, spead_ip="localhost", 
                ts_db=1, ts_ip='127.0.0.1'):
    """
    Start the pipeline using 'num_buffers' buffers, each of size 'buffer_maxsize'.
    This will instantiate num_buffers + 1 threads; a thread for each pipeline and an
    extra accumulator thread the reads data from the spead stream into each buffer
    seen by the pipeline.
    Inputs
    ======
    num_buffers: int
        The number of buffers to use- this will create a pipeline thread for each buffer
        and an extra accumulator thread to read the spead stream.
    buffer_maxsize: float
        The maximum size of the buffer. Memory for each buffer will be allocated at first
        and then populated by the accumulator from the spead stream.
    spead_port: int
        The port to read the spead stream from
    spead_ip: string
        The ip to read the spead stream from.
    ts_db : int
        The telescope model database number
    ts_ip : string
        The telescope model ip address
    """ 

    # Parameters which define the size of the array to initialise.
    # Needs to be made generic
    element_size = 8. # 8 bytes in an np.complex64

    # start TM
    ts = TelescopeState(host=ts_ip,db=ts_db)
    nchan = ts.echan - ts.bchan
    # number of baselines includes autocorrelations
    nants = ts.num_ants
    nbl = nants*(nants+1)/2
    npol = 4
    
    # ------------------------------------------------------------
    
    array_length = buffer_maxsize/(element_size*nchan*npol*nbl)
    array_length = np.int(np.ceil(array_length))
    logger.info('Max length of buffer array : {0}'.format(array_length,))
    
    # Set up empty buffers
    buffers = [create_buffer_arrays(array_length,nchan,nbl,npol) for i in range(num_buffers)]

    # set up conditions for the buffers
    scan_accumulator_conditions = [threading.Condition() for i in range(num_buffers)]
    
    # Set up the accumulator
    accumulator = accumulator_thread(buffers, scan_accumulator_conditions, spead_port, spead_ip)

    #Set up the pipelines (one per buffer)
    pipelines = [pipeline_thread(buffers[i], scan_accumulator_conditions[i], i) for i in range(num_buffers)]
    
    #Start the pipeline threads
    map(lambda x: x.start(), pipelines)
    # give the pipeline threads a shirt while to aquire conditions then wait
    time.sleep(5.)
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
    
    (options, args) = parse_args()
    print options
    print args

    # short weit to give me time to start up the simulated spead stream
    time.sleep(5.)

    run_threads(num_buffers=options.num_buffers, buffer_maxsize=options.buffer_maxsize, spead_port=options.spead_port, 
               spead_ip=options.spead_ip, ts_db=options.ts_db, ts_ip=options.ts_ip)
