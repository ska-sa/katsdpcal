#!/usr/bin/env python
# ----------------------------------------------------------
# Simulate receiver for L1 data stream

import spead64_48 as spead

from katsdptelstate import endpoint, ArgumentParser

def parse_opts():
    parser = ArgumentParser(description = 'Simulate receiver for L1 data stream')    
    parser.add_argument('--l1-spectral-spead', type=endpoint.endpoint_list_parser(7202, single_port=True), default=':7202', 
            help='endpoints to listen for L1 SPEAD stream (including multicast IPs). [<ip>[+<count>]][:port]. [default=%(default)s]', metavar='ENDPOINT')
    parser.set_defaults(telstate='localhost')
    return parser.parse_args()

def receive_l1(spead_stream):
    """
    Read L1 data from spead stream
    """

    ig = spead.ItemGroup()
    
    # receive SPEAD stream
    print 'Got heaps: ',
    array_index = 0
    for heap in spead.iterheaps(spead_stream): 
        ig.update(heap)
        
        ts = ig['timestamp']
        vis = ig['correlator_data']
        flags = ig['flags']
        weights = ig['weights']
        
        # print some values to see all is well
        print array_index, ts, vis.shape ,flags.shape, weights.shape
        array_index += 1    
                
if __name__ == '__main__':
    """
    Recieved a single scan L1 output stream
       and print some details to confirm all is going well
    """
    opts = parse_opts() 
    # Initialise spead receiver
    spead_stream = spead.TransportUDPrx(opts.l1_spectral_spead[0].port)
    # recieve stream
    receive_l1(spead_stream)