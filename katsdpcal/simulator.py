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
from pyrap.tables import table

#--------------------------------------------------------------------------------------------------
#--- simdata classes
#--------------------------------------------------------------------------------------------------

def init_simdata(file_name,refant=None):

    try:
        # Is it a katdal H5 file?
        katdal.open(file_name)
        data_class = SimDataH5
    except IOError:
        try:
            # Not an H5 file. Is it an MS?
            table(file_name)
            data_class = SimDataMS
        except RuntimeError:
            # not an MS file either
            print 'File does not exist, or is not of compatible format! (Must be H5 or MS)'
            return

    #----------------------------------------------------------------------------------------------
    class SimData(data_class):
        
        def __init__(self, file_name, refant=None):
            data_class.__init__(self,file_name,refant=refant)
       
        def write(self,data,corrprod_mask,tsask=None,cmask=None):
            """
            Writes data into MS file
       
            Parameters
            ----------
            data          : data to write into the file
            corrprod_mask : correlation product mask showing where to write the data
            tsask         : time mask for timestamps to write
            cmask         : channel mask for channels to write
            """
            
            print '*'
       
        def setup_ts(self,ts):
            """
            Add key value pairs from MS file to to Telescope State
       
            Parameters
            ----------
            ts : TelescopeState
            """   
            # assume integration time is constant over observation, so just take first value

            # get parameters from data file
            parameter_dict = self.get_params()
            # get extra parameters from TS (set at run time)
            parameter_dict['cbf_n_chans'] = ts.cal_echan-ts.cal_bchan
            # check that the minimum necessary prameters are set
            min_keys = ['sdp_l0_int_time', 'antenna_mask', 'cbf_n_ants', 'cbf_n_chans', 'cbf_bls_ordering', 'cbf_sync_time', 'experiment_id', 'experiment_id']
            for key in min_keys: 
                if not key in parameter_dict: raise KeyError('Required parameter {0} not set by simulator.'.format(key,))
            # add parameters to telescope state
            for param in parameter_dict:
                print param, parameter_dict[param]
                ts.add(param, parameter_dict[param])
            
        def XXdatatoSPEAD(self,ts,l0_endpoint,spead_rate=1e9,max_scans=None):
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

            time_ordered_table = t.sort('TIME')
            
            for ts in time_ordered_table.iter('TIME'):
                print '**'
    #---------------------------------------------------------------------------------------------

    return SimData(file_name,refant)

class SimDataMS(table):
    
    def __init__(self, file_name, refant):
        table.__init__(self, file_name)
        self.data_mask = None
        self.file_name = file_name
        self.refant = refant
        self.intent_to_tag = {'CALIBRATE_PHASE,CALIBRATE_AMPLI':'gaincal', 
                              'CALIBRATE_BANDPASS,CALIBRATE_FLUX,CALIBRATE_DELAY':'bpcal',
                              'CALIBRATE_POLARIZATION':'polcal',
                              'TARGET':'target'}

    def write(self,data,corrprod_mask,tsask=None,cmask=None):
        """
        Writes data into MS file
   
        Parameters
        ----------
        data          : data to write into the file
        corrprod_mask : correlation product mask showing where to write the data
        tsask         : time mask for timestamps to write
        cmask         : channel mask for channels to write
        """
        
        print '**'
   
    def get_params(self):
        """
        Add key value pairs from h5 file to to parameter dictionary
   
        Parameters
        ----------
        ts : TelescopeState
        """   

        param_dict = {}

        param_dict['sdp_l0_int_time'] = self.getcol('EXPOSURE')[0]
        antlist = table(self.getkeyword('ANTENNA')).getcol('NAME')
        param_dict['antenna_mask'] = ','.join([ant for ant in antlist])
        param_dict['cbf_n_ants'] = len(antlist)

        time = self.getcol('TIME')
        a1 = self.getcol('ANTENNA1')[time==time[0]]
        a2 = self.getcol('ANTENNA2')[time==time[0]]
        corr_prods = np.array([[antlist[a1i],antlist[a2i]] for a1i,a2i in zip(a1,a2)])

        npols = table(self.getkeyword('POLARIZATION')).getcol('NUM_CORR')
        if npols == 4: 
            pol_order = np.array([['h','h'],['h','v'],['v','h'],['v','v']])
        elif npols == 2:
            pol_order = np.array([['h','h'],['v','v']])
        elif npols == 1:
            pol_order = np.array([['h','h']])
        else:
            raise ValueError('Weird polarisation setup!')

        corr_prods_pol = np.array([[c1+p1,c2+p2] for c1,c2 in corr_prods for p1,p2 in pol_order])

        print corr_prods_pol
        # need pol here too!!
        param_dict['cbf_bls_ordering'] = corr_prods_pol
        param_dict['cbf_sync_time'] = 0.0
        param_dict['experiment_id'] = self.file_name.split('.')[0]
        param_dict['config'] = {'MS_simulator':True}

        return param_dict

    def select(self,**kwargs):

        chan_slice = None if not kwargs.has_key('channels') else kwargs['channels']
        self.data_mask = np.s_[:,chan_slice,...]
        
    def datatoSPEAD(self,ts,l0_endpoint,spead_rate=1e9,max_scans=None):
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

        num_scans = max(self.getcol('SCAN_NUMBER'))
        # if the maximum number of scans to transmit has not been specified, set to total number of scans
        if max_scans is None: 
            max_scans = num_scans
        else:
            num_scans = max_scans

        ordered_table = self.sort('SCAN_NUMBER, TIME, ANTENNA1, ANTENNA2')
        field_names = table(self.getkeyword('FIELD')).getcol('NAME')
        npols = table(self.getkeyword('POLARIZATION')).getcol('NUM_CORR')
        intents = table(self.getkeyword('STATE')).getcol('OBS_MODE')

        total_ts, track_ts, slew_ts = 0, 0, 0

        for scan_ind, tscan in enumerate(ordered_table.iter('SCAN_NUMBER')):
             
            # update telescope state with scan information
            #   subtract random offset to time, <= 0.1 seconds, to simulate
            #   slight differences in times of different sensors
            field_id = set(tscan.getcol('FIELD_ID'))
            if len(field_id) > 1:
                raise ValueError('More than one target in a scan!')
            state_id = set(tscan.getcol('STATE_ID'))
            if len(state_id) > 1:
                raise ValueError('More than one state in a scan!')
            tag = self.intent_to_tag[intents[state_id.pop()]]
           
            target_desc = ', '.join([field_names[field_id.pop()],tag])

            # MS files only have tracks (?)
            scan_state = 'track'

            ts.add('{0}_target'.format(self.refant,),target_desc,ts=tscan.getcol('TIME',startrow=0,nrow=1)[0]-random()*0.1)
            ts.add('{0}_activity'.format(self.refant,),scan_state,ts=tscan.getcol('TIME',startrow=0,nrow=1)[0]-random()*0.1)
            print 'Scan', scan_ind+1, '/', num_scans, ' -- ', 
            n_ts = len(tscan.select('unique TIME'))
            print 'timestamps:', n_ts, ' -- ',
            print scan_state, target_desc

            # keep track of number of timestamps
            total_ts += n_ts
            if scan_state == 'track': track_ts += n_ts
            if scan_state == 'slew': slew_ts += n_ts

            # transmit the data timestamp by timestamp
            for ttime in tscan.iter('TIME'):

                tx_time = ttime.getcol('TIME')[0] # time
                tx_vis = np.hstack(ttime.getcol('DATA')[self.data_mask]) # visibilities for this time stamp, for specified channel range
                tx_flags = np.hstack(ttime.getcol('FLAG')[self.data_mask]) # flags for this time stamp, for specified channel range
                tx_weights = np.zeros_like(tx_flags,dtype=np.float64)

                # transmit timestamps, vis, flags, weights
                transmit_item(tx, tx_time, tx_vis, tx_flags, tx_weights)

            if scan_ind+1 == max_scans:
                break

        print 'Track timestamps:', track_ts
        print 'Slew timestamps: ', slew_ts
        print 'Total timestamps:', total_ts
                
        end_transmit(tx) 

class SimDataH5(katdal.H5DataV2):
    
    def __init__(self, file_name, refant=''):
        H5DataV2.__init__(self, file_name, refant)
        # need reference antenna for simulating activity and target sensor
        self.refant = refant
   
    def write(self,data,corrprod_mask,tsask=None,cmask=None):
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
   
    def get_params(self):
        """
        Add key value pairs from h5 file to to parameter dictionary
   
        Parameters
        ----------
        ts : TelescopeState
        """   

        param_dict = {}

        param_dict['sdp_l0_int_time'] = self.dump_period
        param_dict['cbf_bls_ordering'] = self.corr_products
        param_dict['cbf_sync_time'] = 0.0
        param_dict['cbf_n_ants'] = len(self.ants)
        antenna_mask = ','.join([ant.name for ant in self.ants])
        param_dict['antenna_mask'] = antenna_mask
        param_dict['experiment_id'] = self.experiment_id
        param_dict['config'] = {'h5_simulator':True}

        return param_dict
        
    def datatoSPEAD(self,ts,l0_endpoint,spead_rate=1e9,max_scans=None):
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

        total_ts, track_ts, slew_ts = 0, 0, 0
        
        for scan_ind, scan_state, target in self.scans(): 
            # update telescope state with scan information
            #   subtract random offset to time, <= 0.1 seconds, to simulate
            #   slight differences in times of different sensors
            ts.add('{0}_target'.format(self.refant,),target.description,ts=self.timestamps[0]-random()*0.1)
            ts.add('{0}_activity'.format(self.refant,),scan_state,ts=self.timestamps[0]-random()*0.1)
            print 'Scan', scan_ind+1, '/', num_scans, ' -- ', 
            n_ts = len(self.timestamps)
            print 'timestamps:', n_ts, ' -- ',
            print scan_state, target.description

            # keep track 0f number of timestamps
            total_ts += n_ts
            if scan_state == 'track': track_ts += n_ts
            if scan_state == 'slew': slew_ts += n_ts
            
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

            if scan_ind+1 == max_scans:
                break

        print 'Track timestamps:', track_ts
        print 'Slew timestamps: ', slew_ts
        print 'Total timestamps:', total_ts
                
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
    