
import spead64_48 as spead

from optparse import OptionParser
import os

def parse_args():
    usage = "%s [options]"%os.path.basename(__file__)
    description = "Receive L1 SPEAD data."
    parser = OptionParser( usage=usage, description=description)
    parser.add_option("--l1-spectral-spead", default="8891", help="destination port for spectral L1 input. default: 8890")
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
    
    (options, args) = parse_args()  
    # Initialise spead receiver
    spead_stream = spead.TransportUDPrx(int(options.l1_spectral_spead))
    # recieve stream
    receive_l1(spead_stream)