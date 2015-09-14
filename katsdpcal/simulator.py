"""
KAT-7 Simulator for testing MeerKAT calibration pipeline
========================================================
Simulator class for HDF5 files produced by KAT-7 correlator,
for testing of the MeerKAT pipeline.
"""

import katdal
from calprocs import get_reordering_nopol

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

def get_file_format(file_name, mode='r'):

    try:
        # Is it a katdal H5 file?
        katdal.open(file_name, mode=mode)
        data_class = SimDataH5
    except IOError:
        try:
            # Not an H5 file. Is it an MS?
            table(file_name)
            data_class = SimDataMS
        except RuntimeError:
            # not an MS file either
            print 'File does not exist, or is not of compatible format! (Must be H5 or MS.)'
            return

    return data_class

def init_simdata(file_name, **kwargs):
    """
    Initialise simulated data class, using either h5 or MS files for simulation.

    Parameters
    ----------
    file_name : name of data file to use for simulation, strong
    """

    mode = kwargs['mode'] if 'mode' in kwargs else 'r'
    data_class = get_file_format(file_name, mode)

    #----------------------------------------------------------------------------------------------
    class SimData(data_class):
        """
        Simulated data class.
        Uses file to simulate MeerKAT pipeline data SPEAD stream and Telescope State.
        Subclasses either SimDataH5 or SimDataMS, depending on whether an H5 or MS file is used for simulation.
        """
        
        def __init__(self, file_name, **kwargs):
            data_class.__init__(self, file_name, **kwargs)
       
        def write(self,ts,data):
            """
            Writes data into file
       
            Parameters
            ----------
            data          : data to write into the file
            corrprod_mask : correlation product mask showing where to write the data
            tmask         : time mask for timestamps to write
            cmask         : channel mask for channels to write
            """

            data_times, data_vis, data_flags = data

            vis = np.array(data_vis)
            times = np.array(data_times)
            flags = np.array(data_flags)
            # number of timestamps in collected data
            ti_max = len(data_times)

            # check for missing timestamps
            if not np.all(self.timestamps[0:ti_max] == times):
                if np.all(self.timestamps[0:ti_max-1] == times[0:-1]):
                    print 'SPEAD error: extra final L1 time stamp. Ignoring last time stamp.'
                    ti_max -= 1
                else:
                    raise ValueError('L1 array and h5 array have different timestamps!')

            # get data format parameters from TS
            cal_bls_ordering = ts.cal_bls_ordering
            cal_pol_ordering = ts.cal_pol_ordering
            bchan = ts.cal_bchan
            echan = ts.cal_echan

            # write the data into the file
            self.write_data(vis,flags,ti_max,cal_bls_ordering,cal_pol_ordering,bchan=bchan,echan=echan)

        def setup_ts(self,ts):
            """
            Add key value pairs from file to to Telescope State
       
            Parameters
            ----------
            ts : TelescopeState
            """   
            # assume integration time is constant over observation, so just take first value

            # get parameters from data file
            parameter_dict = self.get_params()
            # get/edit extra parameters from TS (set at run time)
            parameter_dict['cbf_n_chans'] = ts.cal_echan-ts.cal_bchan
            parameter_dict['cbf_channel_freqs'] = parameter_dict['cbf_channel_freqs'][ts.cal_bchan:ts.cal_echan]
            # check that the minimum necessary prameters are set
            min_keys = ['sdp_l0_int_time', 'antenna_mask', 'cbf_n_ants', 'cbf_n_chans', 'cbf_bls_ordering', 'cbf_sync_time', 'experiment_id', 'experiment_id']
            for key in min_keys: 
                if not key in parameter_dict: raise KeyError('Required parameter {0} not set by simulator.'.format(key,))
            # add parameters to telescope state
            for param in parameter_dict:
                print param, parameter_dict[param]
                ts.add(param, parameter_dict[param])
            
        def datatoSPEAD(self,ts,l0_endpoint,spead_rate=1e9,max_scans=None):
            """
            Iterates through H5 file and transmits data as a SPEAD stream.
            
            Parameters
            ----------
            ts         : Telescope State
            l0_endoint : Endpoint for SPEAD stream
            spead_rate : SPEAD data transmission rate
            max_scans  : Maximum number of scans to transmit
            """
            print 'TX: Initializing...'
            # rate limit transmission to work on Laura's laptop
            tx = spead.Transmitter(spead.TransportUDPtx(l0_endpoint.host,l0_endpoint.port,rate=spead_rate))

            # if the maximum number of scans to transmit has not been specified, set to total number of scans
            if max_scans is None or max_scans > self.num_scans:
                max_scans = self.num_scans

            # transmit data timestamp by timestamp and update telescope state
            self.tx_data(ts,tx,max_scans)
            # end SPEAD transmission
            end_transmit(tx)

    #---------------------------------------------------------------------------------------------
    return SimData(file_name, **kwargs)

#--------------------------------------------------------------------------------------------------
#--- SimDataMS class
#---   simulates pipeline data from MS
#--------------------------------------------------------------------------------------------------

class SimDataMS(table):
    """
    Simulated data class.
    Uses MS file to simulate MeerKAT pipeline data SPEAD stream and Telescope State,
    subclassing pyrap table.

    Parameters
    ----------
    file_name : Name of MS file, string

    Attributes
    ----------
    file_name     : Name of MS file, string
    data_mask     : Mask for selecting data, numpy array
    intent_to_tag : Dictionary of mappings from MS intents to scan intent tags
    num_scans     : Total number of scans in the MS data set
    corr_prods    : Correlation product mapping

    Note
    ----
    MS files for the simulator currently need to be full polarisation and full correlation (including auto-corrs)
    """
    
    def __init__(self, file_name, **kwargs):
        readonly = False if kwargs.get('mode')=='r+' else True
        table.__init__(self, file_name, readonly=readonly)
        self.data_mask = None
        self.corr_prods = None
        self.file_name = file_name
        self.intent_to_tag = {'CALIBRATE_PHASE,CALIBRATE_AMPLI':'gaincal', 
                              'CALIBRATE_BANDPASS,CALIBRATE_FLUX,CALIBRATE_DELAY':'bpcal',
                              'CALIBRATE_BANDPASS,CALIBRATE_FLUX':'bpcal',
                              'CALIBRATE_POLARIZATION':'polcal',
                              'TARGET':'target'}
        self.num_scans = max(self.getcol('SCAN_NUMBER'))
        self.timestamps = np.unique(self.sort('SCAN_NUMBER, TIME, ANTENNA1, ANTENNA2').getcol('TIME'))
        self.ants = table(self.getkeyword('ANTENNA')).getcol('NAME')
        self.corr_products, self.bls_ordering = self.get_corrprods(self.ants)

    def write_data(self,vis,flags,ti_max,cal_bls_ordering,cal_pol_ordering,bchan=1,echan=0):
        """
        Writes data into MS file.
   
        ------------
        """
        ordermask, desired = get_reordering_nopol(self.ants,cal_bls_ordering,output_order_bls=self.bls_ordering)

        pol_num = {'h': 0, 'v': 1}
        pol_types = {'hh': 9, 'vv': 12, 'hv': 10, 'vh': 11}
        pol_type_array = np.array([pol_types[p1+p2] for p1,p2 in  cal_pol_ordering])[np.newaxis, :]
        pol_index_array = np.array([[pol_num[p1],pol_num[p2]] for p1,p2 in  cal_pol_ordering], dtype=np.int32)[np.newaxis, :]
        poltable = table(self.getkeyword('POLARIZATION'),readonly=False)
        poltable.putcol('CORR_TYPE',pol_type_array)
        poltable.putcol('CORR_PRODUCT',pol_index_array)

        ms_sorted = self.sort('TIME')
        # write data timestamp by timestamp
        corrprod_indices = np.array([[self.ants.index(a1), self.ants.index(a2)] for a1, a2 in cal_bls_ordering])

        data = None
        for ti, ms_time in enumerate(ms_sorted.iter('TIME')):
            if data == None: 
                data = np.zeros_like(ms_time.getcol('DATA'))
            data[:,bchan:echan,:] = np.rollaxis(vis[ti],-1,0)[ordermask,...]
            ms_time.putcol('DATA',data)
            # break when we have reached the max timestamp index in the vis data
            if ti == ti_max-1: break
   
    def get_params(self):
        """
        Add key value pairs from h5 file to to parameter dictionary.
   
        Returns
        -------
        param_dict : dictionary of observations parameters
        """
        param_dict = {}

        param_dict['cbf_channel_freqs'] = table(self.getkeyword('SPECTRAL_WINDOW')).getcol('CHAN_FREQ')[0]
        param_dict['sdp_l0_int_time'] = self.getcol('EXPOSURE')[0]
        param_dict['antenna_mask'] = ','.join([ant for ant in self.ants])
        param_dict['cbf_n_ants'] = len(self.ants)

        # need pol here too!!
        param_dict['cbf_bls_ordering'] = self.corr_products
        param_dict['cbf_sync_time'] = 0.0
        param_dict['experiment_id'] = self.file_name.split('.')[0].split('/')[-1]
        param_dict['config'] = {'MS_simulator':True}

        return param_dict

    def get_corrprods(self,antlist):
        """
        Gets correlation product list from MS

        Returns
        -------
        correlation product list, shape (num_baselines, 2)
        """

        time = self.getcol('TIME')
        a1 = self.getcol('ANTENNA1')[time==time[0]]
        a2 = self.getcol('ANTENNA2')[time==time[0]]

        corr_prods_nopol = np.array([[antlist[a1i],antlist[a2i]] for a1i,a2i in zip(a1,a2)])

        npols = table(self.getkeyword('POLARIZATION')).getcol('NUM_CORR')
        if npols == 4: 
            pol_order = np.array([['h','h'],['h','v'],['v','h'],['v','v']])
        elif npols == 2:
            pol_order = np.array([['h','h'],['v','v']])
        elif npols == 1:
            pol_order = np.array([['h','h']])
        else:
            raise ValueError('Weird polarisation setup!')

        corrprods_nopol = np.array([[c1, c2] for c1,c2 in corr_prods_nopol])
        corrprods = np.array([[c1+p1,c2+p2] for c1,c2 in corr_prods_nopol for p1,p2 in pol_order])
        return corrprods, corrprods_nopol

    def set_corrprods(self,corr_prods):
        """
        Allows MS simulator to emulate katdal style correlation product attribute
        """
        self.corr_prods = corr_prods

    def select(self,**kwargs):
        """
        Allows MS simulator to emulate katdal style data selection.
        Currently only selects on channel.
        """
        chan_slice = None if not kwargs.has_key('channels') else kwargs['channels']
        self.data_mask = np.s_[:,chan_slice,...]
        
    def tx_data(self,ts,tx,max_scans):
        """
        Iterates through H5 file and transmits data as a spead stream,
        also updating the telescope state accordingly.
        
        Parameters
        ----------
        ts        : Telescope State
        tx        : SPEAD transmitter
        max_scans : Maximum number of scans to transmit
        """
        total_ts, track_ts, slew_ts = 0, 0, 0

        ordered_table = self.sort('SCAN_NUMBER, TIME, ANTENNA1, ANTENNA2')
        field_names = table(self.getkeyword('FIELD')).getcol('NAME')
        npols = table(self.getkeyword('POLARIZATION')).getcol('NUM_CORR')
        intents = table(self.getkeyword('STATE')).getcol('OBS_MODE')
        antlist = table(self.getkeyword('ANTENNA')).getcol('NAME')

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

            for ant in antlist:
                ts.add('{0}_target'.format(ant,),target_desc,ts=tscan.getcol('TIME',startrow=0,nrow=1)[0]-random()*0.1)
                ts.add('{0}_activity'.format(ant,),scan_state,ts=tscan.getcol('TIME',startrow=0,nrow=1)[0]-random()*0.1)
            print 'Scan', scan_ind+1, '/', max_scans, ' -- ',
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

#--------------------------------------------------------------------------------------------------
#--- SimDataH5 class
#---   simulates pipeline data from H5 file
#--------------------------------------------------------------------------------------------------

class SimDataH5(katdal.H5DataV2):
    """
    Simulated data class.
    Uses H5 file to simulate MeerKAT pipeline data SPEAD stream and Telescope State,
    subclassing katdal H5DataV2.

    Parameters
    ----------
    file_name : Name of MS file, string

    Attributes
    ----------
    num_scans     : Total number of scans in the MS data set
    """
    
    def __init__(self, file_name, **kwargs):
        mode = kwargs['mode'] if 'mode' in kwargs else 'r'
        H5DataV2.__init__(self, file_name, mode=mode)
        self.num_scans = len(self.scan_indices)
   
    def write_data(self,vis,flags,ti_max,cal_bls_ordering,cal_pol_ordering,bchan=1,echan=0):
        """
        Writes data into h5 file.
   
        ------------
        """
        # pack data into h5 correlation product list
        #    by iterating through h5 correlation product list
        for i, [ant1, ant2] in enumerate(self.corr_products):

            # find index of this pair in the cal product array
            antpair = [ant1[:-1],ant2[:-1]]
            cal_indx = cal_bls_ordering.index(antpair)
            # find index of this element in the pol dimension
            polpair = [ant1[-1],ant2[-1]]
            pol_indx = cal_pol_ordering.index(polpair)

            # vis shape is (ntimes, nchan, ncorrprod) for real and imag
            self._vis[0:ti_max,bchan:echan,i,0] = vis[0:ti_max,:,pol_indx,cal_indx].real
            self._vis[0:ti_max,bchan:echan,i,1] = vis[0:ti_max,:,pol_indx,cal_indx].imag
   
    def get_params(self):
        """
        Add key value pairs from h5 file to to parameter dictionary.
   
        Returns
        -------
        param_dict : dictionary of observations parameters
        """
        param_dict = {}

        param_dict['cbf_channel_freqs'] = self.channel_freqs
        param_dict['sdp_l0_int_time'] = self.dump_period
        param_dict['cbf_n_ants'] = len(self.ants)
        param_dict['cbf_bls_ordering'] = self.corr_products
        param_dict['cbf_sync_time'] = 0.0
        antenna_mask = ','.join([ant.name for ant in self.ants])
        param_dict['antenna_mask'] = antenna_mask
        param_dict['experiment_id'] = self.experiment_id
        param_dict['config'] = {'h5_simulator':True}

        return param_dict
        
    def tx_data(self,ts,tx,max_scans):
        """
        Iterates through H5 file and transmits data as a spead stream,
        also updating the telescope state accordingly.
        
        Parameters
        ----------
        ts        : Telescope State
        tx        : SPEAD transmitter
        max_scans : Maximum number of scans to transmit
        """
        total_ts, track_ts, slew_ts = 0, 0, 0

        for scan_ind, scan_state, target in self.scans(): 
            # update telescope state with scan information
            #   subtract random offset to time, <= 0.1 seconds, to simulate
            #   slight differences in times of different sensors
            for ant in self.ants:
                ts.add('{0}_target'.format(ant.name,),target.description,ts=self.timestamps[0]-random()*0.1)
                ts.add('{0}_activity'.format(ant.name,),scan_state,ts=self.timestamps[0]-random()*0.1)
            print 'Scan', scan_ind+1, '/', max_scans, ' -- ',
            n_ts = len(self.timestamps)
            print 'timestamps:', n_ts, ' -- ',
            print scan_state, target.description

            # keep track of number of timestamps
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

#--------------------------------------------------------------------------------------------------
                    
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
    