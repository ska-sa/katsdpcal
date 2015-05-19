#!/usr/bin/env python
# ----------------------------------------------------------
# Simulate receiver for L1 data stream

import spead64_48 as spead

from katsdptelstate import endpoint, ArgumentParser

import numpy as np
import shutil
import os.path

def parse_opts():
    parser = ArgumentParser(description = 'Simulate receiver for L1 data stream')    
    parser.add_argument('--l1-spectral-spead', type=endpoint.endpoint_list_parser(7202, single_port=True), default=':7202', 
            help='endpoints to listen for L1 SPEAD stream (including multicast IPs). [<ip>[+<count>]][:port]. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument('--h5file', type=str, help='H5 data to write data to')
    parser.set_defaults(telstate='localhost')
    return parser.parse_args()

def receive_l1(spead_stream, return_data=False):
    """
    Read L1 data from spead stream

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
    Recieved a the L1 output stream and print some details to confirm all is going well
    Optionally write the L1 data back to the h5 file
    """
    opts = parse_opts() 
    # Initialise spead receiver
    spead_stream = spead.TransportUDPrx(opts.l1_spectral_spead[0].port)
    # recieve stream and accumulate data into arrays
    return_data = True if opts.h5file else False
    l1_data = receive_l1(spead_stream, return_data=return_data)
    # need some info from the telstate
    ts = opts.telstate
    if opts.h5file:
        new_file = '{0}_L1.h5'.format(opts.h5file.split('.')[0],)
        if not ts.cal_full_l1:
            print 'Only target L1 stream transmitted. Not saving L1 data to file.'
        elif os.path.isfile(new_file):
            print 'L1 file {0} already exists. Not saving L1 data to file.'.format(new_file,)
        else:
            shutil.copyfile(opts.h5file,new_file)

            import katdal
            f = katdal.open(new_file)

            data_times, data_vis, data_flags = l1_data

            # number of timestamps in collected data
            ti_max = len(data_times)

            vis = np.array(data_vis)
            times = np.array(data_times)
            flags = np.array(data_flags)

            # check for missing timestamps
            if not np.all(f.timestamps[0:ti_max] == times):
                if np.all(f.timestamps[0:ti_max-1] == times[0:-1]):
                    print 'SPEAD error: extra final L1 time stamp. Ignoring last time stamp.'
                    ti_max += -1
                else:
                    raise ValueError('L1 array and h5 array have different timestamps!')

            # get necessary info from telescope state
            corr_products = f.corr_products
            cal_bls_ordering = ts.cal_bls_ordering
            cal_pol_ordering = ts.cal_pol_ordering
            bchan = ts.cal_bchan
            echan = ts.cal_echan

            # pack data into h5 correlation product list
            #    by iterating through h5 correlation product list
            print 'Writing data to h5 file {0}'.format(new_file)
            for i, [ant1, ant2] in enumerate(corr_products):

                # find index of this pair in the cal product array
                antpair = [ant1[:-1],ant2[:-1]]
                cal_indx = cal_bls_ordering.index(antpair)
                # find index of this element in the pol dimension
                polpair = [ant1[-1],ant2[-1]]
                pol_indx = cal_pol_ordering.index(polpair)

                # vis shape is (ntimes, nchan, ncorrprod) for real and imag
                f._vis[0:ti_max,bchan:echan,i,0] = vis[0:ti_max,:,pol_indx,cal_indx].real
                f._vis[0:ti_max,bchan:echan,i,1] = vis[0:ti_max,:,pol_indx,cal_indx].imag

