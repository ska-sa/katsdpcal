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
from random import random

#--------------------------------------------------------------------------------------------------
#--- CLASS :  SimData
#--------------------------------------------------------------------------------------------------

class SimData(katdal.H5DataV2):
    
    def __init__(self, h5filename):
        H5DataV2.__init__(self, h5filename)
   
    def write_h5(self,data,corrprod_mask,tsask=None,cmask=None):
        """
        Writes data into h5 file
   
        Parameters
        ----------
        data          : data to write into the file
        corrprod_mask : correlation product mask showing where to write the data
        tsask         : time mask for timestamps to write
        cmask         : channel mask for channels to write
        """
        
        data_indices = np.squeeze(np.where(tsask)) if np.any(tsask) else np.squeeze(np.where(self._time_keep))
        ti, tf = min(data_indices), max(data_indices)+1 
        data_indices = np.squeeze(np.where(cmask)) if np.any(cmask) else np.squeeze(np.where(self._freq_keep))
        ci, cf = min(data_indices), max(data_indices)+1    
        self._vis[ti:tf,ci:cf,corrprod_mask,0] = data.real
        self._vis[ti:tf,ci:cf,corrprod_mask,1] = data.imag
   
    def setup_ts(self,ts):
        """
        Initialises the Telescope Model, optionally from existing TM pickle.
   
        Parameters
        ----------
        ts : Telescope Model dictionary
        """   
        # set simulated ts values from h5 file
        ts.add('antlist', [ant.name for ant in self.ants])
        ts.add('nant', len(self.ants))
        ts.add('nchan', ts.echan-ts.bchan)
        ts.add('corr_products', self.corr_products)
        ts.add('dump_period', self.dump_period)
        
    def h5toSPEAD(self,ts,port):
        """
        Iterates through H5 file and transmits data as a spead stream.
        
        Parameters
        ----------
        ts   : Telescope State 
        port : port to send spead tream to
        
        """
        
        print 'TX: Initializing...'
        tx = spead.Transmitter(spead.TransportUDPtx('127.0.0.1', port))
        
        for scan_ind, scan_state, target in self.scans(): 
            # update telescope state with scan information
            #   add random offset to time, <= 0.1 seconds, to simulate
            #   slight differences in times of different sensors
            ts.add('target',target.description,ts=self.timestamps[0]+random()*0.1)
            ts.add('tag',target.tags,ts=self.timestamps[0]+random()*0.1)
            ts.add('scan_state',scan_state,ts=self.timestamps[0]+random()*0.1)
            print scan_state
            
            # transmit the data from this scan, timestamp by timestamp
            scan_data = self.vis[:]
            scan_flags = self.flags()[:]
            scan_weights = self.weights()[:]

            # transmit data
            for i in range(scan_data.shape[0]): # time axis

                tx_time = self.timestamps[i] # time
                tx_vis = scan_data[i,:,:] # visibilities for this time stamp, for specified channel range
                tx_flags = scan_flags[i,:,:] # flags for this time stamp, for specified channel range
                tx_weights = scan_weights[i,:,:]
                tx_tags = np.array(target.tags)

                # transmit timestamps, vis, flags and scan state (which stays the same for a scan)
                transmit_ts(tx, tx_time, tx_vis, tx_flags, tx_weights, scan_state, tx_tags)
                # delay so receiver isn't overwhelmed
                time.sleep(0.5)
            
            # insert little delay between scans
            time.sleep(1.5)
                
        end_transmit(tx)
                    
def end_transmit(tx):
    """
    Send stop packet to spead stream tx
    
    Parameters
    ----------
    tx       : spead stream
    """
    tx.end()
    
def transmit_ts(tx, tx_time, tx_vis, tx_flags, tx_weights, tx_state, tx_tags):
    """
    Send spead packet containing time, visibility, flags and array state
    
    Parameters
    ----------
    tx         : spead stream
    tx_time    : timestamp, float
    tx_vis     : visibilities, complex array 
    tx_flags   : flags, int array
    tx_weights : weights, float array
    tx_state   : current state of array, string 
                 e.g. 'track', 'slew'
    tx_tags    : intent tags, string array
    """
    ig = spead.ItemGroup()

    ig.add_item(name='timestamp', description='Timestamp',
        shape=[], fmt=spead.mkfmt(('f',64)),
        init_val=tx_time)

    ig.add_item(name='correlator_data', description='Full visibility array',
        init_val=tx_vis)

    ig.add_item(name='flags', description='Flag array',
        init_val=tx_flags)
        
    ig.add_item(name='weights', description='Weight array',
        init_val=tx_weights)
        
    ig.add_item(name='state', description='array state',
        shape=-1, fmt=spead.mkfmt(('s', 8)), init_val=tx_state)
        
    ig.add_item(name='tags', description='intent tags',
        init_val=tx_tags)

    tx.send_heap(ig.get_heap())
    