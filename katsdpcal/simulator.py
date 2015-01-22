"""
KAT-7 Simulator for testing MeerKAT calibration pipeline
========================================================

Simulator class for HDF5 files produced by KAT-7 correlator,
for testing of the MeerKAT pipeline.
"""

import katdal
from katdal import H5DataV2
import pickle
import os
import numpy as np
import spead64_48 as spead
import time

#--------------------------------------------------------------------------------------------------
#--- CLASS :  SimData
#--------------------------------------------------------------------------------------------------

class SimData(katdal.H5DataV2):
    
    def __init__(self, h5filename):
        H5DataV2.__init__(self, h5filename)
   
    def write_h5(self,data,corrprod_mask,tmask=None,cmask=None):
        """
        Writes data into h5 file
   
        Parameters
        ----------
        data          : data to write into the file
        corrprod_mask : correlation product mask showing where to write the data
        tmask         : time mask for timestamps to write
        cmask         : channel mask for channels to write
        """
        
        data_indices = np.squeeze(np.where(tmask)) if np.any(tmask) else np.squeeze(np.where(self._time_keep))
        ti, tf = min(data_indices), max(data_indices)+1 
        data_indices = np.squeeze(np.where(cmask)) if np.any(cmask) else np.squeeze(np.where(self._freq_keep))
        ci, cf = min(data_indices), max(data_indices)+1    
        self._vis[ti:tf,ci:cf,corrprod_mask,0] = data.real
        self._vis[ti:tf,ci:cf,corrprod_mask,1] = data.imag
   
    def setup_TM(self,TMfile,params):
        """
        Initialises the Telescope Model, optionally from existing TM pickle.
   
        Parameters
        ----------
        TMfile  : name of TM pickle file to open 
        params  : dictionary of default parameters

        Returns
        ------- 
        TM      : Telescope Model dictionary
        """   

        # initialise with parameter dictionary 
        TM = params
        # update TM with pickle file values
        TM_update = pickle.load(open(TMfile, 'rb')) if os.path.isfile(TMfile) else {}
        for key in TM_update: TM[key] = TM_update[key]

        # empty solutions - start with no solutions
        TM['BP'] = []
        TM['BP_std'] = []
        TM['K'] = []
        TM['K_std'] = []
        TM['G'] = []
        TM['G_std'] = []
        TM['G_times'] = []

        # set siulated TM values from h5 file
        TM['antlist'] = [ant.name for ant in self.ants]
        TM['num_ants'] = len(self.ants)
        TM['num_channels'] = len(self.channels)
        #antdesclist = [ant.description for ant in simdata.ants]
        TM['corr_products'] = self.corr_products
        TM['dump_period'] = self.dump_period
   
        return TM
        
    def h5toSPEAD(self,port):
        """
        Iterates through H5 file and transmits data as a spead stream.
        
        Parameters
        ----------
        port : port to send spead tream to
        
        """
        
        print 'TX: Initializing...'
        tx = spead.Transmitter(spead.TransportUDPtx('127.0.0.1', port))
        
        for scan_ind, scan_state, target in self.scans(): 
            
            # transmit the data from this scan, timestamp by timestamp
            scan_data = self.vis[:]
            scan_flags = self.flags()[:]

            # transmit data
            for i in range(scan_data.shape[0]): # time axis

                tx_time = self.timestamps[i] # time
                tx_vis = scan_data[i,:,:] # visibilities for this time stamp, for specified channel range
                tx_flags = scan_flags[i,:,:] # flags for this time stamp, for specified channel range

                # transmit timestamps, vis, flags and scan state (which stays the same for a scan)
                transmit_ts(tx, tx_time, tx_vis, tx_flags, scan_state)
                time.sleep(0.01)
                
        end_transmit(tx)
                    
def end_transmit(tx):
    """
    Send stop packet to spead stream tx
    
    Parameters
    ----------
    tx       : spead stream
    """
    tx.end()
    
def transmit_ts(tx, tx_time, tx_vis, tx_flags, tx_state):
    """
    Send spead packet containing time, visibility, flags and array state
    
    Parameters
    ----------
    tx       : spead stream
    tx_time  : timestamp, float
    tx_vis   : visibilities, complex array 
    tx_flags : flags, int array
    tx_state : current state of array, string 
               e.g. 'track', 'slew'
    """
    ig = spead.ItemGroup()

    ig.add_item(name='timestamp', description='Timestamp',
        shape=[], fmt=spead.mkfmt(('f',64)),
        init_val=tx_time)

    ig.add_item(name='correlator_data', description='Full visibility array',
        init_val=tx_vis)

    ig.add_item(name='flags', description='Flag array',
        init_val=tx_flags)
        
    ig.add_item(name='state', description='array state',
        init_val=np.array([tx_state]))

    tx.send_heap(ig.get_heap())
    