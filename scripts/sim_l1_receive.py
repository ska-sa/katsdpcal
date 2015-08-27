#!/usr/bin/env python
# ----------------------------------------------------------
# Simulate receiver for L1 data stream

import spead64_48 as spead

from katsdptelstate import endpoint, ArgumentParser
from katsdpcal.simulator import init_simdata

import numpy as np
import shutil
import os.path

def parse_opts():
    parser = ArgumentParser(description = 'Simulate receiver for L1 data stream')    
    parser.add_argument('--l1-spectral-spead', type=endpoint.endpoint_list_parser(7202, single_port=True), default=':7202', 
            help='endpoints to listen for L1 SPEAD stream (including multicast IPs). [<ip>[+<count>]][:port]. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument('--file', type=str, help='File for simulated data (H5 or MS)')
    parser.set_defaults(telstate='localhost')
    return parser.parse_args()

def accumulate_l1(spead_stream, return_data=False):
    """
    Read L1 data from spead stream and accumulate it into a list

    Inputs:
    -------
    spead_stream : SPEAD data stream
    return_data : If True, collect and return the data

    Returns:
    --------
    If return_data True:
    data_times : list of timestamps
    data_vis : list of visibilities
    data_flags : list of flags
    """
    timestamp_prev = 0

    ig = spead.ItemGroup()

    if return_data: data_times, data_vis , data_flags = [], [], []
    # don't do weights for now, as I'm not really using weights
    
    # receive SPEAD stream
    print 'Got heaps: ',
    array_index = 0
    for heap in spead.iterheaps(spead_stream): 
        ig.update(heap)
        
        timestamp = ig['timestamp']
        vis = ig['correlator_data']
        flags = ig['flags']
        weights = ig['weights']

        if return_data:
            data_times.append(timestamp)
            data_vis.append(vis)
            data_flags.append(flags)
        
        # print some values to see all is well
        print array_index, timestamp, vis.shape ,flags.shape, weights.shape,
        print np.round(timestamp-timestamp_prev,2)
        timestamp_prev = timestamp
        array_index += 1    

    if return_data: return data_times, data_vis, data_flags
                
if __name__ == '__main__':
    """
    Recieve an L1 output stream and print some details to confirm all is going well
    Optionally write the L1 data back to the h5 file
    """
    opts = parse_opts() 
    # Initialise spead receiver
    spead_stream = spead.TransportUDPrx(opts.l1_spectral_spead[0].port)
    # recieve stream and accumulate data into arrays
    return_data = True if opts.file else False
    l1_data = accumulate_l1(spead_stream, return_data=return_data)

    # if specified, write the output back to the file
    if opts.file:
        new_file = '{0}_L1.h5'.format(opts.file.split('.')[0],)

        # need some info from the telstate
        ts = opts.telstate

        if not ts.cal_full_l1:
            print 'Only target L1 stream transmitted. Not saving L1 data to file.'
        else:
            if os.path.isfile(new_file):
                print 'WARNING: L1 file {0} already exists. Over writing it.'.format(new_file,)
            shutil.copyfile(opts.file,new_file)

            # set up file to write the data into
            datafile = init_simdata(new_file,mode='r+')

            print 'Writing data to h5 file {0}'.format(new_file)
            datafile.write(ts,l1_data)
