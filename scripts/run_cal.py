
import numpy as np
from katcal.control_threads import accumulator_thread, pipeline_thread
import threading

import optparse

from katcal.simulator import SimData
from katcal import parameters

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import time

PORT = 8890
            
def create_buffer_arrays(array_length,nchan,nbl,npol):
    """
    Create empty buffer arrays using specified dimensions
    """
    vis = np.empty([array_length,nchan,nbl,npol],dtype=np.complex64)
    flags = np.empty([array_length,nchan,nbl,npol],dtype=np.uint8)
    times = np.empty([array_length],dtype=np.float)
    return vis,flags, times
          
def run_threads(h5file):
    
    # data parameters for buffer
    buffer_maxsize = 1000e6 #128.e9
    element_size = 8. # 8 bits in an np.complex64

    # ------------------------------------------------------------
    # for fake meerkat ingest data:    
#    nchan = 32768
#    nbl = 3
#    npol = 4

    # for kat7 simulated data:
    # get data shape from telescope model
    #   for the moment - mock this up from the simulated data
    file_name = h5file
    simdata = SimData(file_name)
    params = parameters.set_params()
    TMfile = 'TM.pickle'
    # use parameters from parameter file as defaults, but override with TMfile
    TM = simdata.setup_TM(TMfile,params)

    nchan = TM['echan'] - TM['bchan']
    # including autocorrelations
    nbl = TM['num_ants']*(TM['num_ants']+1)/2
    npol = 4
    # ------------------------------------------------------------
    
    array_length = buffer_maxsize/(element_size*nchan*npol*nbl)
    array_length = np.int(np.ceil(array_length))
    logger.info('Max length of buffer array : {0}'.format(array_length,))
    
    # create empty buffer arrays
    vis1, flags1, times1 = create_buffer_arrays(array_length,nchan,nbl,npol)
    vis2, flags2, times2 = create_buffer_arrays(array_length,nchan,nbl,npol)
    
    # set up conditions for the two buffers
    scan_accumulator_condition1 = threading.Condition()
    scan_accumulator_condition2 = threading.Condition()
    
    accumulator = accumulator_thread(times1, vis1, flags1, scan_accumulator_condition1,
        times2, vis2, flags2, scan_accumulator_condition2, PORT)
    pipeline1 = pipeline_thread(times1, vis1, flags1, scan_accumulator_condition1, '1')
    pipeline2 = pipeline_thread(times2, vis2, flags2, scan_accumulator_condition2, '2')
    
    try:
        accumulator.start()
        pipeline1.start()
        pipeline2.start()
        while accumulator.isAlive() and pipeline1.isAlive() and pipeline2.isAlive(): 
            # not sure if there is an appreciable cost to this - stole from the web
            accumulator.join(1)  
            pipeline1.join(1) 
            pipeline2.join(1)
    except (KeyboardInterrupt, SystemExit):
        print '\nReceived keyboard interrupt! Quitting threads.\n'
        accumulator.stop()
        # wait for accumulator to release all before stopping pipeline
        time.sleep(0.1)
        pipeline1.stop()
        pipeline2.stop()
    except:
        print '\nUnexpected error! Quitting threads.\n'
        accumulator.stop()
        # wait for accumulator to release all before stopping pipeline
        pipeline1.stop()
        pipeline2.stop()
        
    time.sleep(2.)
    print '***',  accumulator.isAlive(), pipeline1.isAlive(), pipeline2.isAlive()     
    accumulator.join()
    pipeline1.join()
    pipeline2.join()

if __name__ == '__main__':
    
    parser = optparse.OptionParser(usage="%prog [options] <filename.h5>", description='Run MeerKAT calibration pipeline on H5 file')
    (options, args) = parser.parse_args()
    
    run_threads(args[0])
