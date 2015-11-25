"""
KAT-7 Simulator for testing MeerKAT calibration pipeline
========================================================
Simulator class for HDF5 files produced by KAT-7 correlator,
for testing of the MeerKAT pipeline.
"""

from . import table
from . import spead2
from . import send
from .calprocs import get_reordering_nopol

import katdal
import katpoint

import ephem

import pickle
import os
import numpy as np
import time
from random import random

#--------------------------------------------------------------------------------------------------
#--- simdata classes
#--------------------------------------------------------------------------------------------------

def get_file_format(file_name):
    """
    Determine format of file used for simulation (H5 or MS)

    Parameters
    ----------
    file_name : Name of H5 or MS file, string

    Returns
    -------
    data_class : Class for simulated data, SimDataH5 or SimDataMS
    """
    try:
        # Is it a katdal H5 file?
        # open in r+ mode so we don't inadvertantly lock it into readonly mode for later
        katdal.open(file_name, mode='r+')
        data_class = SimDataH5
    except IOError:
        try:
            # Not an H5 file. Is it an MS?
            table(file_name)
            data_class = SimDataMS
        except RuntimeError:
            # not an MS file either
            print 'File does not exist, or is not of compatible format! (Must be H5 or MS.)'
            data_class = None

    return data_class

def init_simdata(file_name, wait=0.1, **kwargs):
    """
    Initialise simulated data class, using either H5 or MS files for simulation.

    Parameters
    ----------
    file_name : name of H5 or MS file, string
    wait      : pause between sending heaps (potentially necessary to rate limit simulator transmit)

    Returns
    -------
    SimData : data simulator, SimData object
    """
    # determine if simulator file is H5 or MS
    data_class = get_file_format(file_name)

    #----------------------------------------------------------------------------------------------
    class SimData(data_class):
        """
        Simulated data class.
        Uses file to simulate MeerKAT pipeline data SPEAD stream and Telescope State.
        Subclasses either SimDataH5 or SimDataMS, depending on whether an H5 or MS file is used for simulation.

        Parameters
        ----------
        file_name : Name of MS file, string
        wait      : pause between sending heaps (potentially necessary to rate limit simulator transmit)

        Attributes
        ----------
        wait      : pause between sending heaps (potentially necessary to rate limit simulator transmit)
        """ 
        def __init__(self, file_name, wait=wait, **kwargs):
            data_class.__init__(self, file_name, **kwargs)
            self.wait = wait

        def setup_ts(self,ts):
            """
            Add key value pairs from file to the Telescope State
       
            Parameters
            ----------
            ts : TelescopeState
            """   
            # get parameters from data file
            parameter_dict = self.get_params()
            # get/edit extra parameters from TS (set at run time)
            if ts.has_key('cal_echan'):
                # if bchan and echan are set, use them to override number of channels
                parameter_dict['cbf_n_chans'] = ts.cal_echan-ts.cal_bchan
            else:
                # else set bchan and echan to be full channel range
                ts.delete('cal_bchan')
                ts.add('cal_bchan',0,immutable=True)
                ts.delete('cal_echan')
                ts.add('cal_echan',parameter_dict['cbf_n_chans'],immutable=True)

            parameter_dict['cbf_channel_freqs'] = parameter_dict['cbf_channel_freqs'][ts.cal_bchan:ts.cal_echan]
            # check that the minimum necessary prameters are set
            min_keys = ['sdp_l0_int_time', 'antenna_mask', 'cbf_n_ants', 'cbf_n_chans', 'cbf_bls_ordering', 'cbf_sync_time', 'experiment_id', 'experiment_id']
            for key in min_keys: 
                if not key in parameter_dict: raise KeyError('Required parameter {0} not set by simulator.'.format(key,))
            # add parameters to telescope state
            for param in parameter_dict:
                print param, parameter_dict[param]
                ts.add(param, parameter_dict[param])
            
        def datatoSPEAD(self,ts,l0_endpoint,spead_rate=5e8,max_scans=None):
            """
            Iterates through file and transmits data as a SPEAD stream.
            
            Parameters
            ----------
            ts         : Telescope State
            l0_endoint : Endpoint for SPEAD stream
            spead_rate : SPEAD data transmission rate
            max_scans  : Maximum number of scans to transmit
            """
            print 'TX: Initializing...'
            # configure SPEAD - may need to rate limit transmission for laptops etc.
            config = send.StreamConfig(max_packet_size=9172, rate=spead_rate)
            tx = send.UdpStream(spead2.ThreadPool(),l0_endpoint.host,l0_endpoint.port,config)

            # if the maximum number of scans to transmit has not been specified, set to total number of scans
            if max_scans is None or max_scans > self.num_scans:
                max_scans = self.num_scans

            # transmit data timestamp by timestamp and update telescope state
            self.tx_data(ts,tx,max_scans)

        def setup_ig(self, ig, correlator_data, flags, weights):
            """
            Initialises data transmit ItemGroup for SPEAD transmit

            Parameters
            ----------
            ig              : ItemGroup
            correlator_data : visibilities, numpy array
            flags           : flags, numpy array
            weights         : weights, numpy array
            """
            ig.add_item(id=None, name='correlator_data', description="Visibilities",
                shape=correlator_data.shape, dtype=correlator_data.dtype)
            ig.add_item(id=None, name='flags', description="Flags for visibilities",
                shape=flags.shape, dtype=flags.dtype)
            ig.add_item(id=None, name='weights', description="Weights for visibilities",
                shape=weights.shape, dtype=weights.dtype)
            ig.add_item(id=None, name='timestamp', description="Seconds since sync time",
                shape=(), dtype=None, format=[('f', 64)])

        def transmit_item(self, tx, ig, timestamp, correlator_data, flags, weights):
            """
            Transmit single SPEAD ItemGroup

            Parameters
            ----------
            tx              : SPEAD stream
            ig              : ItemGroup
            timestamp       : timestamp, float64
            correlator_data : visibilities, numpy array
            flags           : flags, numpy array
            weights         : weights, numpy array
            wait            : pause between sending heaps (potentially necessary to rate limit simulator transmit)
            """
            # transmit vis, flags and weights, timestamp
            ig['correlator_data'].value = correlator_data 
            ig['flags'].value = flags 
            ig['weights'].value = weights 
            ig['timestamp'].value = timestamp 
            # send all of the descriptors with every heap
            tx.send_heap(ig.get_heap(descriptors='all'))
            time.sleep(self.wait)

        def write(self,ts,data):
            """
            Writes data into file
       
            Parameters
            ----------
            ts   : TelescopeState
            data : data to write into the file: [times, visibilities, flags]
            """
            data_times, data_vis, data_flags = data
            vis = np.array(data_vis)
            times = np.array(data_times)
            flags = np.array(data_flags)
            # number of timestamps in collected data
            ti_max = len(data_times)

            # check for missing timestamps
            if not np.all(self.to_ut(self.timestamps[0:ti_max]) == times):
                start_repeat = np.squeeze(np.where(times[ti_max-1]==times))[0]

                if np.all(self.to_ut(self.timestamps[0:start_repeat]) == times[0:start_repeat]):
                    print 'SPEAD error: {0} extra final L1 time stamp(s). Ignoring last time stamp(s).'.format(ti_max-start_repeat-1,)
                    ti_max += -(ti_max-start_repeat-1)
                else:
                    raise ValueError('Received L1 array and data file array have different timestamps!')

            # get data format parameters from TS
            cal_bls_ordering = ts.cal_bls_ordering
            cal_pol_ordering = ts.cal_pol_ordering
            bchan = ts.cal_bchan
            echan = ts.cal_echan

            # write the data into the file
            self.write_data(vis,flags,ti_max,cal_bls_ordering,cal_pol_ordering,bchan=bchan,echan=echan)

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
    file_name     : name of MS file, string
    data_mask     : mask for selecting data, numpy array
    intent_to_tag : dictionary of mappings from MS intents to scan intent tags
    num_scans     : total number of scans in the MS data set
    timestamps    : timestamps contained in the data, numpy array
    ants          : antenna names, list of strings
    corr_products : polarisation-correlation product pairs, numpy array of strings, shape (nbl*npol, 2)
    bls_ordering  : correlation product pairs, numpy array of strings, shape (nbl, 2)


    Note
    ----
    MS files for the simulator currently need to be full polarisation and full correlation (including auto-corrs)
    """
    
    def __init__(self, file_name, **kwargs):
        readonly = False if kwargs.get('mode')=='r+' else True
        table.__init__(self, file_name, readonly=readonly)
        self.file_name = file_name
        self.data_mask = None
        self.intent_to_tag = {'CALIBRATE_PHASE,CALIBRATE_AMPLI':'gaincal', 
                              'CALIBRATE_BANDPASS,CALIBRATE_FLUX,CALIBRATE_DELAY':'bpcal',
                              'CALIBRATE_BANDPASS,CALIBRATE_FLUX':'bpcal',
                              'CALIBRATE_POLARIZATION':'polcal',
                              'TARGET':'target'}
        self.num_scans = max(self.getcol('SCAN_NUMBER'))
        self.timestamps = np.unique(self.sort('SCAN_NUMBER, TIME, ANTENNA1, ANTENNA2').getcol('TIME'))
        self.ants = table(self.getkeyword('ANTENNA')).getcol('NAME')
        self.corr_products, self.bls_ordering = self.get_corrprods(self.ants)

    def to_ut(self, t):
        """
        Converts MJD seconds into Unix time in seconds

        Parameters
        ----------
        t : time in MJD seconds

        Returns
        -------
        Unix time in seconds
        """
        return (t/86400. - 2440587.5 + 2400000.5)*86400.

    def field_ids(self):
        """
        Field IDs in the data set

        Returns
        -------
        List of field IDs
        """
        return range(table(self.getkeyword('FIELD')).nrows())
   
    def get_params(self):
        """
        Add key value pairs from MS file to to parameter dictionary.
   
        Returns
        -------
        param_dict : dictionary of observation parameters
        """
        param_dict = {}

        param_dict['cbf_channel_freqs'] = table(self.getkeyword('SPECTRAL_WINDOW')).getcol('CHAN_FREQ')[0]
        param_dict['cbf_n_chans'] = table(self.getkeyword('SPECTRAL_WINDOW')).getcol('NUM_CHAN')[0]
        param_dict['sdp_l0_int_time'] = self.getcol('EXPOSURE')[0]
        param_dict['antenna_mask'] = ','.join([ant for ant in self.ants])
        param_dict['cbf_n_ants'] = len(self.ants)
        # need polarisation information in the cbf_bls_ordering
        param_dict['cbf_bls_ordering'] = self.corr_products
        param_dict['cbf_sync_time'] = 0.0
        param_dict['experiment_id'] = self.file_name.split('.')[0].split('/')[-1]
        param_dict['config'] = {'MS_simulator': True}

        # antenna descriptions for all antennas
        #  assume the order is the same as self.ants
        antenna_positions = table(self.getkeyword('ANTENNA')).getcol('POSITION')
        antenna_diameters = table(self.getkeyword('ANTENNA')).getcol('DISH_DIAMETER')
        for ant, diam, pos in zip(self.ants, antenna_diameters, antenna_positions):
            longitude, latitude, altitude = katpoint.ecef_to_lla(pos[0],pos[1],pos[2])
            lla_position = ",".join([str(ephem.degrees(longitude)), str(ephem.degrees(latitude)), str(altitude)])
            description = "{0}, {1}, {2}".format(ant, lla_position, diam)
            param_dict['{0}_description'.format(ant,)] = description

        return param_dict

    def get_corrprods(self,antlist):
        """
        Gets correlation product list from MS

        Parameters
        ----------
        antlist : antenna names, list of strings

        Returns
        -------
        corrprods       : polarisation-correlation product pairs, numpy array of strings, shape (nbl*npol, 2)
        corrprods_nopol : correlation product pairs, numpy array of strings, shape (nbl, 2)
        """
        # get baseline ordering for first timestamp
        time = self.getcol('TIME')
        a1 = self.getcol('ANTENNA1')[time==time[0]]
        a2 = self.getcol('ANTENNA2')[time==time[0]]

        # determine antenna name ordering of antennas from MS baseline ordering indices
        corrprods_nopol = np.array([[antlist[a1i],antlist[a2i]] for a1i,a2i in zip(a1,a2)])

        # determine polarisation ordering of the MS data
        npols = table(self.getkeyword('POLARIZATION')).getcol('NUM_CORR')
        if npols == 4: 
            pol_order = np.array([['h','h'],['h','v'],['v','h'],['v','v']])
        elif npols == 2:
            pol_order = np.array([['h','h'],['v','v']])
        elif npols == 1:
            pol_order = np.array([['h','h']])
        else:
            raise ValueError('Weird polarisation setup!')
        # combine antenna and polarisation strongs to get full correlator product ordering
        corrprods = np.array([[c1+p1,c2+p2] for c1,c2 in corrprods_nopol for p1,p2 in pol_order])

        return corrprods, corrprods_nopol

    def select(self,**kwargs):
        """
        Allows MS simulator to emulate katdal style data selection.
        Currently only selects on channel.
        """
        chan_slice = None if not kwargs.has_key('channels') else kwargs['channels']
        self.data_mask = np.s_[:,chan_slice,...]
        
    def tx_data(self,ts,tx,max_scans):
        """
        Iterates through MS file and transmits data as a SPEAD stream,
        also updating the telescope state accordingly.
        
        Parameters
        ----------
        ts        : Telescope State
        tx        : SPEAD transmitter
        max_scans : Maximum number of scans to transmit
        """
        # order the data for transmission
        ordered_table = self.sort('SCAN_NUMBER, TIME, ANTENNA1, ANTENNA2')
        # get metadata information for the telescope state
        field_names = table(self.getkeyword('FIELD')).getcol('NAME')
        positions = table(self.getkeyword('FIELD')).getcol('DELAY_DIR')
        intents = table(self.getkeyword('STATE')).getcol('OBS_MODE')
        antlist = table(self.getkeyword('ANTENNA')).getcol('NAME')
        # set up ItemGroup for transmission
        flavour = spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
        ig = send.ItemGroup(flavour=flavour)

        # send data scan by scan
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
           
            f_id = field_id.pop()
            ra, dec = np.squeeze(positions[f_id])
            radec = ephem.Equatorial(ra, dec)
            target_desc = '{0}, radec {1}, {2}, {3}'.format(field_names[f_id],tag,str(radec.ra),str(radec.dec))

            # MS files only have tracks (?)
            scan_state = 'track'

            scan_time = tscan.getcol('TIME',startrow=0,nrow=1)[0]
            scan_time_ut = self.to_ut(scan_time)
            for ant in antlist:
                ts.add('{0}_target'.format(ant,),target_desc,ts=scan_time_ut-random()*0.1)
                ts.add('{0}_activity'.format(ant,),scan_state,ts=scan_time_ut-random()*0.1)
            print 'Scan', scan_ind+1, '/', max_scans, ' -- ',
            n_ts = len(tscan.select('unique TIME'))
            print 'timestamps:', n_ts, ' -- ',
            print scan_state, target_desc

            # transmit the data timestamp by timestamp
            for ttime in tscan.iter('TIME'):
                # get data to transmit from MS
                tx_time = self.to_ut(ttime.getcol('TIME')[0]) # time
                tx_vis = np.hstack(ttime.getcol('DATA')[self.data_mask]) # visibilities for this time stamp, for specified channel range
                tx_flags = np.hstack(ttime.getcol('FLAG')[self.data_mask]) # flags for this time stamp, for specified channel range
                try:
                    tx_weights = np.hstack(ttime.getcol('WEIGHT_SPECTRUM')[self.data_mask]) # weights for this time stamp, for specified channel range
                except RuntimeError:
                    # WEIGHT_SPECTRUM column doesn't exist: mock up weights as zeros
                    tx_weights = np.zeros_like(tx_flags,dtype=np.float64)

                # on first transmittion, set up item group, using info from first data item
                if 'correlator_data' not in ig:
                    self.setup_ig(ig,tx_vis,tx_flags,tx_weights)
                # transmit timestamps, vis, flags, weights
                self.transmit_item(tx, ig, tx_time, tx_vis, tx_flags, tx_weights)

            if scan_ind+1 == max_scans:
                break

        # end transmission
        tx.send_heap(ig.get_end())

        # MS only has 'track' scans?
        print 'Track timestamps:', scan_ind
        print 'Slew timestamps: ', 0
        print 'Total timestamps:', scan_ind

    def write_data(self,correlator_data,flags,ti_max,cal_bls_ordering,cal_pol_ordering,bchan=1,echan=0):
        """
        Writes data into MS file.
   
        Parameters
        ----------
        correlator_data  : visibilities, numpy array
        flags            : flags, numpy array
        ti_max           : index of highest timestamp of supplies correlator_data and flag arrays
        cal_bls_ordering : baseline ordering of visibility data in the pipleine, list of lists, shape (nbl, 2)
        cal_pol_ordering : polarisation pair ordering of visibility data in the pipleine, list of lists, shape (npol, 2)
        bchan            : start channel to write, integer
        echan            : end channel to write, integer
        """
        # determine re-ordering necessary to munge pipeline visibility ordering back to MS ordering
        ordermask, desired = get_reordering_nopol(self.ants,cal_bls_ordering,output_order_bls=self.bls_ordering)

        # write MS polarisation table 
        pol_num = {'h': 0, 'v': 1}
        pol_types = {'hh': 9, 'vv': 12, 'hv': 10, 'vh': 11}
        pol_type_array = np.array([pol_types[p1+p2] for p1,p2 in  cal_pol_ordering])[np.newaxis, :]
        pol_index_array = np.array([[pol_num[p1],pol_num[p2]] for p1,p2 in  cal_pol_ordering], dtype=np.int32)[np.newaxis, :]
        poltable = table(self.getkeyword('POLARIZATION'),readonly=False)
        poltable.putcol('CORR_TYPE',pol_type_array)
        poltable.putcol('CORR_PRODUCT',pol_index_array)

        # sort data by time for writing to MS
        ms_sorted = self.sort('TIME')
        # write data to MS timestamp by timestamp
        data = None
        for ti, ms_time in enumerate(ms_sorted.iter('TIME')):
            # if we don't yet have a data array, create one
            if data is None: 
                data = np.zeros_like(ms_time.getcol('DATA'))
            data[:,bchan:echan,:] = np.rollaxis(correlator_data[ti][...,ordermask],-1,0)
            # save data to CORRECTED_DATA column of the MS
            ms_time.putcol('CORRECTED_DATA',data)
            # break when we have reached the max timestamp index in the vis data
            if ti == ti_max-1: break

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
        super(SimDataH5, self).__init__(file_name, mode=mode)
        self.num_scans = len(self.scan_indices)

    def close(self):
        """
        Allows H5 simulator to emulate pyrap MS close function.
        (Does nothing)
        """
        pass

    def to_ut(self, t):
        """
        Allows H5 simulator to emulate MS Unix time conversion
        (Conversion unnecessary for H5)

        Parameters
        ----------
        t : Unix time in seconds

        Returns
        -------
        Unix time in seconds
        """
        return t

    def field_ids(self):
        """
        Field IDs in the data set

        Returns
        -------
        List of field IDs
        """
        return self.target_indices

    def get_params(self):
        """
        Add key value pairs from H5 file to to parameter dictionary.
   
        Returns
        -------
        param_dict : dictionary of observation parameters
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

        # antenna descriptions for all antennas
        for ant in self.ants:
            param_dict['{0}_description'.format(ant.name,)] = ant.description

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

        flavour = spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
        ig = send.ItemGroup(flavour=flavour)

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

            # set up item group, ising info from first data item
            if 'correlator_data' not in ig:
                self.setup_ig(ig,scan_data[0],scan_flags[0],scan_flags[0])

            # transmit data timestamp by timestamp
            for i in range(scan_data.shape[0]): # time axis

                # data values to transmit
                tx_time = self.timestamps[i] # timestamp
                tx_vis = scan_data[i,:,:] # visibilities for this time stamp, for specified channel range
                tx_flags = scan_flags[i,:,:] # flags for this time stamp, for specified channel range
                tx_weights = scan_weights[i,:,:]

                # transmit timestamps, vis, flags, weights
                self.transmit_item(tx, ig, tx_time, tx_vis, tx_flags, tx_weights)

            if scan_ind+1 == max_scans:
                break

        # end transmission
        tx.send_heap(ig.get_end())

        print 'Track timestamps:', track_ts
        print 'Slew timestamps: ', slew_ts
        print 'Total timestamps:', total_ts

    def write_data(self,correlator_data,flags,ti_max,cal_bls_ordering,cal_pol_ordering,bchan=1,echan=0):
        """
        Writes data into H5 file.
   
        Parameters
        ----------
        correlator_data  : visibilities, numpy array
        flags            : flags, numpy array
        ti_max           : index of highest timestamp of supplies correlator_data and flag arrays
        cal_bls_ordering : baseline ordering of visibility data in the pipleine, list of lists, shape (nbl, 2)
        cal_pol_ordering : polarisation pair ordering of visibility data in the pipleine, list of lists, shape (npol, 2)
        bchan            : start channel to write, integer
        echan            : end channel to write, integer
        """
        # pack data into H5 correlation product list
        #    by iterating through H5 correlation product list
        for i, [ant1, ant2] in enumerate(self.corr_products):

            # find index of this pair in the cal product array
            antpair = [ant1[:-1],ant2[:-1]]
            cal_indx = cal_bls_ordering.index(antpair)
            # find index of this element in the pol dimension
            polpair = [ant1[-1],ant2[-1]]
            pol_indx = cal_pol_ordering.index(polpair)

            # vis shape is (ntimes, nchan, ncorrprod) for real and imag
            self._vis[0:ti_max,bchan:echan,i,0] = correlator_data[0:ti_max,:,pol_indx,cal_indx].real
            self._vis[0:ti_max,bchan:echan,i,1] = correlator_data[0:ti_max,:,pol_indx,cal_indx].imag



