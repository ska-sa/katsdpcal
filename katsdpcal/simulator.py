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
import spead2
import time
from random import random

#--------------------------------------------------------------------------------------------------
#--- CLASS :  SimData
#--------------------------------------------------------------------------------------------------

class SimData(katdal.H5DataV2):
    
    def __init__(self, h5filename):
        H5DataV2.__init__(self, h5filename)
        # need reference antenna for simulating activity and target sensors
   
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
        ts.add('cbf_channel_freqs', self.channel_freqs[ts.cal_bchan:ts.cal_echan])
        ts.add('cbf_bls_ordering', self.corr_products)
        ts.add('cbf_sync_time', 0.0, immutable=True)
        antenna_mask = ','.join([ant.name for ant in self.ants])
        ts.add('antenna_mask', antenna_mask)
        ts.add('experiment_id', self.experiment_id)
        ts.add('config', {'h5_simulator':True})
        
    def h5toSPEAD(self,ts,l0_endpoint,spead_rate=1e9,max_scans=None):
        """
        Iterates through H5 file and transmits data as a spead stream.
        
        Parameters
        ----------

        ts   : Telescope State 
        port : port to send spead tream to
        
        """
        
        print 'TX: Initializing...'
        # rate limit transmission to work on Laura's laptop
        config = spead2.send.StreamConfig(max_packet_size=9172, rate=spead_rate)
        tx = spead2.send.UdpStream(spead2.ThreadPool(),l0_endpoint.host,l0_endpoint.port,config)

        num_scans = len(self.scan_indices)
        # if the maximum number of scans to transmit has not been specified, set to total number of scans
        if max_scans is None: 
            max_scans = num_scans
        else:
            num_scans = max_scans

        total_ts, track_ts, slew_ts = 0, 0, 0
        
        for scan_ind, scan_state, target in self.scans(): 
            # update telescope state with scan information
            #   subtract random offset to time, <= 0.1 seconds, to simulate
            #   slight differences in times of different sensors
            for ant in self.ants:
                ts.add('{0}_target'.format(ant.name,),target.description,ts=self.timestamps[0]-random()*0.1)
                ts.add('{0}_activity'.format(ant.name,),scan_state,ts=self.timestamps[0]-random()*0.1)
            print 'Scan', scan_ind+1, '/', num_scans, ' -- ', 
            n_ts = len(self.timestamps)
            print 'timestamps:', n_ts, ' -- ',
            print scan_state, target.description

            # keep track if number of timestamps
            total_ts += n_ts
            if scan_state == 'track': track_ts += n_ts
            if scan_state == 'slew': slew_ts += n_ts
            
            # transmit the data from this scan, timestamp by timestamp
            scan_data = self.vis[:]
            scan_flags = self.flags()[:]
            scan_weights = self.weights()[:]

            # set up ig items, if they have not been set up yet
            if 'correlator_data' not in ig:
                # set up item group with items
                ig.add_item(id=None, name='correlator_data', description="Visibilities",
                     shape=scan_data[0].shape, dtype=scan_vis.dtype)
                ig.add_item(id=None, name='flags', description="Flags for visibilities",
                     shape=scan_flags[0].shape, dtype=scan_flags.dtype)
                # for now, just transmit flags as placeholder for weights
                ig.add_item(id=None, name='weights', description="Weights for visibilities",
                     shape=scan_flags[0].shape, dtype=scan_weights.dtype)
                ig.add_item(id=None, name='timestamp', description="Seconds since sync time",
                     shape=(), dtype=None, format=[('f', 64)])

            # transmit data timestamp by timestamp
            for i in range(scan_data.shape[0]): # time axis
                # transmit timestamps, vis, flags and weights
                ig['correlator_data'].value = scan_data[i,:,:] # visibilities for this time stamp, for specified channel range
                ig['flags'].value = scan_flags[i,:,:] # flags for this time stamp, for specified channel range
                ig['weights'].value = scan_weights[i,:,:]
                ig['timestamp'].value = self.timestamps[i] # time
                tx.send_heap(ig.get_heap())

            if scan_ind+1 == max_scans:
                break

        print 'Track timestamps:', track_ts
        print 'Slew timestamps: ', slew_ts
        print 'Total timestamps:', total_ts
                
        # end transmission
        tx.end()
    
