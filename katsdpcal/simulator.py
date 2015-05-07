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
    
    def __init__(self, h5filename, refant=''):
        H5DataV2.__init__(self, h5filename, refant)
        # need reference antenna for simulating activity and target sensors
        self.refant = refant
   
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
        Add key value pairs from h5 file to to Telescope State
   
        Parameters
        ----------
        ts : TelescopeState
        """   
        # set simulated ts values from h5 file
        ts.add('sdp_l0_int_time', self.dump_period)
        ts.add('cbf_n_ants', len(self.ants))
        ts.add('cbf_n_chans', ts.cal_echan-ts.cal_bchan)
        ts.add('cbf_bls_ordering', self.corr_products)
        ts.add('cbf_sync_time', 0.0, immutable=True)
        antenna_mask = ','.join([ant.name for ant in self.ants])
        ts.add('antenna_mask', antenna_mask)
        ts.add('experiment_id', self.experiment_id)
        ts.add('config', {'h5_simulator':True})
        
    def h5toSPEAD(self,ts,l0_endpoint,wait_time=0.5,spead_rate=1e9,max_scans=None):
        """
        Iterates through H5 file and transmits data as a spead stream.
        
        Parameters
        ----------

        ts   : Telescope State 
        port : port to send spead tream to
        
        """
        
        print 'TX: Initializing...'
        # rate limit transmission to work on Laura's laptop
        tx = spead.Transmitter(spead.TransportUDPtx(l0_endpoint.host,l0_endpoint.port,rate=spead_rate))

        num_scans = len(self.scan_indices)
        # if the maximum number of scans to transmit has not been specified, set to total number of scans
        if max_scans is None: 
            max_scans = num_scans
        else:
            num_scans = max_scans
        
        for scan_ind, scan_state, target in self.scans(): 
            # update telescope state with scan information
            #   subtract random offset to time, <= 0.1 seconds, to simulate
            #   slight differences in times of different sensors
            ts.add('{0}_target'.format(self.refant,),target.description,ts=self.timestamps[0]-random()*0.1)
            ts.add('{0}_activity'.format(self.refant,),scan_state,ts=self.timestamps[0]-random()*0.1)
            print 'Scan', scan_ind+1, '/', num_scans, ' -- ', 
            print 'timestamps:', len(self.timestamps), ' -- ',
            print scan_state, target.description
            
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

                # transmit timestamps, vis, flags, weights
                transmit_item(tx, tx_time, tx_vis, tx_flags, tx_weights)
                # delay so receiver isn't overwhelmed
                time.sleep(wait_time)

            if scan_ind+1 == max_scans:
                break
                
        end_transmit(tx)
                    
def end_transmit(tx):
    """
    Send stop packet to spead stream tx
    
    Parameters
    ----------
    tx       : spead stream
    """
    tx.end()
    
def transmit_item(tx, tx_time, tx_vis, tx_flags, tx_weights):
    """
    Send spead packet containing time, visibility, flags and array state
    
    Parameters
    ----------
    tx         : spead stream
    tx_time    : timestamp, float
    tx_vis     : visibilities, complex array 
    tx_flags   : flags, int array
    tx_weights : weights, float array
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

    tx.send_heap(ig.get_heap())
    
