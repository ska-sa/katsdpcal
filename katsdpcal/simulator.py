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
import katcp

import numpy as np
import time
from random import random
import ephem

# -------------------------------------------------------------------------------------------------
# --- simdata classes
# -------------------------------------------------------------------------------------------------


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
        f = katdal.open(file_name, mode='r+')
        data_class = SimDataH5V3 if f.version == '3.0' else SimDataH5V2
        print data_class
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


def get_antdesc_relative(names, diameters, positions):
    """
    Determine antenna description dictionary, using offsets from a lat-long-alt reference position.

    Returns
    -------
    Antenna description dictionary
    """
    antdesc = {}
    first_ant = True
    for ant, diam, pos in zip(names, diameters, positions):
        if first_ant:
            # set up reference position (this is necessary to preserve
            # precision of antenna positions when converting because of ephem
            # limitation in truncating decimal places when printing strings).
            longitude, latitude, altitude = katpoint.ecef_to_lla(pos[0], pos[1], pos[2])
            longitude_ref = ephem.degrees(str(ephem.degrees(longitude)))
            latitude_ref = ephem.degrees(str(ephem.degrees(latitude)))
            altitude_ref = round(altitude)
            first_ant = False
        # now determine offsets from the reference position to build up full
        # antenna description string
        e, n, u = katpoint.ecef_to_enu(longitude_ref, latitude_ref, altitude_ref,
                                       pos[0], pos[1], pos[2])
        antdesc[ant] = '{0}, {1}, {2}, {3}, {4}, {5} {6} {7}'.format(
            ant, longitude_ref, latitude_ref, altitude_ref, diam, e, n, u)
    return antdesc


def init_simdata(file_name, server=None, wait=0.0, **kwargs):
    """
    Initialise simulated data class, using either H5 or MS files for simulation.

    Parameters
    ----------
    file_name : name of H5 or MS file, string
    server    : katcp server for cal
    wait      : pause between sending heaps (potentially necessary to rate limit simulator transmit)

    Returns
    -------
    SimData : data simulator, SimData object
    """
    # determine if simulator file is H5 or MS
    data_class = get_file_format(file_name)

    # ---------------------------------------------------------------------------------------------
    class SimData(data_class):
        """
        Simulated data class.
        Uses file to simulate MeerKAT pipeline data SPEAD stream and Telescope State.
        Subclasses either SimDataH5 or SimDataMS, depending on whether an H5 or
        MS file is used for simulation.

        Parameters
        ----------
        file_name : str
            Name of MS file, string
        server : :class:`katsdptelstate.endpoint.Endpoint`
            Katcp server for cal
        wait : float
            Pause between sending heaps (potentially necessary to rate-limit
            simulator transmit).

        Attributes
        ----------
        wait : float
            Pause between sending heaps (potentially necessary to rate-limit
            simulator transmit).
        """
        def __init__(self, file_name, server=None, wait=wait, **kwargs):
            data_class.__init__(self, file_name, **kwargs)
            if server is not None:
                self.client = katcp.BlockingClient(server.host, server.port)
                self.client.start()
            else:
                self.client = None
            self.wait = wait

        def capture_init(self):
            if self.client is not None:
                self.client.wait_connected()
                reply, informs = self.client.blocking_request(katcp.Message.request('capture-init'))
                if not reply.reply_ok():
                    raise RuntimeError('capture-init failed: {}'.format(reply.arguments[1]))

        def capture_done(self):
            if self.client is not None:
                self.client.wait_protocol(10)
                reply, informs = self.client.blocking_request(katcp.Message.request('capture-done'))
                if not reply.reply_ok():
                    raise RuntimeError('capture-done failed: {}'.format(reply.arguments[1]))

        def close(self):
            if self.client is not None:
                self.client.stop()
                self.client.join()
                self.client = None

        def clear_ts(self, ts):
            """
            Clear the TS.

            Parameters
            ----------
            ts : Telescope State
            """
            try:
                for key in ts.keys():
                    ts.delete(key)
            except AttributeError:
                # the Telescope State is empty
                pass

        def setup_ts(self, ts):
            """
            Add key value pairs from file to the Telescope State

            Parameters
            ----------
            ts : :class:`~katsdptelstate.TelescopeState`
                Telescope state.
            """
            # get parameters from data file
            parameter_dict = self.get_params()
            # add fake subarray_id to parameter_dict
            parameter_dict['subarray_product_id'] = 'unknown_subarray'

            # get/edit extra parameters from TS (set at run time)

            # simulator channel frequency range
            if 'cal_sim_echan' in ts:
                # if bchan and echan are set, use them to override number of channels
                if ts['cal_sim_echan'] == 0:
                    ts.delete('cal_sim_echan')
                    ts.add('cal_sim_echan', parameter_dict['cbf_n_chans'], immutable=True)

                parameter_dict['cbf_n_chans'] = ts.cal_sim_echan-ts.cal_sim_bchan
            else:
                # else set bchan and echan to be full channel range
                ts.delete('cal_sim_bchan')
                ts.add('cal_sim_bchan', 0, immutable=True)
                ts.delete('cal_sim_echan')
                ts.add('cal_sim_echan', parameter_dict['cbf_n_chans'], immutable=True)

            # use channel freqs to determine parameters cbf_channel_freq and
            # cbf_bandwidth which will be given by the real system
            subset_channel_freqs = \
                parameter_dict['cbf_channel_freqs'][ts.cal_sim_bchan:ts.cal_sim_echan]
            channel_width = np.abs(subset_channel_freqs[1] - subset_channel_freqs[0])
            ts.add('cbf_bandwidth',
                   np.abs(subset_channel_freqs[0]-subset_channel_freqs[-1]) + channel_width,
                   immutable=True)
            ts.add('cbf_center_freq', subset_channel_freqs[len(subset_channel_freqs)/2],
                   immutable=True)
            ts.delete('cbf_channel_freqs')

            # check that the minimum necessary prameters are set
            min_keys = ['sdp_l0_int_time', 'antenna_mask',
                        'cbf_n_ants', 'cbf_n_chans', 'cbf_n_pols', 'sdp_l0_bls_ordering',
                        'cbf_sync_time', 'subarray_product_id']
            for key in min_keys:
                if key not in parameter_dict:
                    raise KeyError('Required parameter {0} not set by simulator.'.format(key,))
            # add parameters to telescope state
            for param in parameter_dict:
                print param, parameter_dict[param]
                ts.add(param, parameter_dict[param])

        def datatoSPEAD(self, ts, l0_endpoint, spead_rate=5e8, max_scans=None):
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
            config = send.StreamConfig(max_packet_size=8972, rate=spead_rate)
            tx = send.UdpStream(spead2.ThreadPool(), l0_endpoint.host, l0_endpoint.port, config)

            # if the maximum number of scans to transmit has not been
            # specified, set to total number of scans
            if max_scans is None or max_scans > self.num_scans:
                max_scans = self.num_scans

            # transmit data timestamp by timestamp and update telescope state
            self.tx_data(ts, tx, max_scans)

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
                        shape=weights.shape, dtype=np.uint8)
            ig.add_item(id=None, name='weights_channel', description='Per-channel scaling for weights',
                        shape=(weights.shape[0],), dtype=np.float32)
            ig.add_item(id=None, name='timestamp', description="Seconds since sync time",
                        shape=(), dtype=None, format=[('f', 64)])

        def setup_obs_params(self, ts, t=None):
            """
            Set up fake obs params

            Parameters
            ----------
            ts : Telescope State
            t  : time for setting parameters, seconds, float, optional
            """

            # fake obs params for now
            ts.add('obs_params', "experiment_id '2016_{0}'".format(int(time.time()),), ts=t)
            ts.add('obs_params', "observer 'AR1'", ts=t)
            ts.add('obs_params', "proposal_id 'PIPELINE-AR1'", ts=t)
            ts.add('obs_params', "project_id 'PIPELINETEST'", ts=t)
            ts.add('obs_params', "sim_file '{0}'".format(self.file_name,), ts=t)

        def transmit_item(self, tx, ig, timestamp, correlator_data, flags, weights):
            """
            Transmit single SPEAD :class:`~spead2.send.ItemGroup`.

            Parameters
            ----------
            tx : :class:`spead2.send.UdpStream`
                SPEAD stream
            ig : :class:`spead2.send.ItemGroup`
                Item group
            timestamp : :class:`np.float64`
                Timestamp
            correlator_data : :class:`np.ndarray`
                Visibilities
            flags : :class:`np.ndarray`
                Flags
            weights : :class:`np.ndarray`
                Weights
            wait : float
                Pause between sending heaps (potentially necessary to
                rate-limit simulator transmit).
            """
            # transmit vis, flags and weights, timestamp
            ig['correlator_data'].value = correlator_data
            ig['flags'].value = flags
            weights_channel = np.require(np.max(weights, axis=1), dtype=np.float32) / np.float32(255)
            # Avoid divide-by-zero issues if all the weights are zero
            weights_channel = np.maximum(weights_channel, np.float32(2.0**-96))
            scaled_weights = np.round(weights / weights_channel[:, np.newaxis] * np.float32(255))
            scaled_weights = scaled_weights.astype(np.uint8)
            ig['weights_channel'].value = weights_channel
            ig['weights'].value = scaled_weights
            ig['timestamp'].value = timestamp
            # send all of the descriptors with every heap
            tx.send_heap(ig.get_heap(descriptors='all'))
            time.sleep(self.wait)

        def write(self, ts, data):
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
                start_repeat = np.squeeze(np.where(times[ti_max-1] == times))[0]

                if np.all(self.to_ut(self.timestamps[0:start_repeat]) == times[0:start_repeat]):
                    print 'SPEAD error: {0} extra final L1 time stamp(s). ' \
                        'Ignoring last time stamp(s).'.format(ti_max-start_repeat-1,)
                    ti_max += -(ti_max-start_repeat-1)
                else:
                    raise ValueError(
                        'Received L1 array and data file array have different timestamps!')

            # get data format parameters from TS
            cal_bls_ordering = ts.cal_bls_ordering
            cal_pol_ordering = ts.cal_pol_ordering
            bchan = ts.cal_sim_bchan
            echan = ts.cal_sim_echan

            # write the data into the file
            self.write_data(vis, flags, ti_max, cal_bls_ordering, cal_pol_ordering,
                            bchan=bchan, echan=echan)

    # ---------------------------------------------------------------------------------------------
    return SimData(file_name, server, **kwargs)

# -------------------------------------------------------------------------------------------------
# --- SimDataMS class
# ---   simulates pipeline data from MS
# -------------------------------------------------------------------------------------------------


class SimDataMS(table):
    """
    Simulated data class.
    Uses MS file to simulate MeerKAT pipeline data SPEAD stream and Telescope State,
    subclassing pyrap table.

    Parameters
    ----------
    file_name : str
        Name of MS file

    Attributes
    ----------
    file_name : str
        name of MS file
    data_mask : :class:`np.ndarray`
        mask for selecting data
    intent_to_tag : dict
        mappings from MS intents to scan intent tags
    num_scans : int
        total number of scans in the MS data set
    timestamps : :class:`np.ndarray`
        timestamps contained in the data
    ants : list of str
        antenna names
    corr_products : :class:`np.ndarray` of str
        polarisation-correlation product pairs, shape (nbl*npol, 2)
    bls_ordering : :class:`np.ndarray` of str
        correlation product pairs, shape (nbl, 2)

    Note
    ----
    MS files for the simulator currently need to be full polarisation and full
    correlation (including auto-corrs).
    """

    def __init__(self, file_name, **kwargs):
        readonly = False if kwargs.get('mode') == 'r+' else True
        table.__init__(self, file_name, readonly=readonly)
        self.file_name = file_name
        self.data_mask = None
        self.intent_to_tag = {
            'CALIBRATE_PHASE,CALIBRATE_AMPLI': 'gaincal',
            'CALIBRATE_BANDPASS,CALIBRATE_FLUX,CALIBRATE_DELAY,CALIBRATE_POLARIZATION':
                'bpcal polcal',
            'CALIBRATE_BANDPASS,CALIBRATE_FLUX,CALIBRATE_DELAY': 'bpcal delaycal',
            'CALIBRATE_BANDPASS,CALIBRATE_FLUX': 'bpcal',
            'CALIBRATE_POLARIZATION': 'polcal',
            'UNKNOWN': 'unknown',
            'TARGET,CALIBRATE_POLARIZATION': 'target polcal',
            'TARGET': 'target'}
        self.num_scans = max(self.getcol('SCAN_NUMBER'))
        self.timestamps = np.unique(
            self.sort('SCAN_NUMBER, TIME, ANTENNA1, ANTENNA2').getcol('TIME'))
        self.ants = table(self.getkeyword('ANTENNA')).getcol('NAME')
        self.corr_products, self.bls_ordering = self.get_corrprods(self.ants)

    def to_ut(self, t):
        """
        Converts MJD seconds into Unix time in seconds

        Parameters
        ----------
        t : float
            time in MJD seconds

        Returns
        -------
        Unix time in seconds
        """
        return (t/86400. - 2440587.5 + 2400000.5)*86400.

    def get_antdesc(self):
        """
        Determine antenna description dictionary, using offsets from a
        lat-long-alt reference position.

        Returns
        -------
        Antenna description dictionary
        """
        positions = table(self.getkeyword('ANTENNA')).getcol('POSITION')
        diameters = table(self.getkeyword('ANTENNA')).getcol('DISH_DIAMETER')
        names = [ant for ant in self.ants]

        # determine an antenna description dictionary using a reference
        # lat-long-alt position end enu offsets
        return get_antdesc_relative(names, diameters, positions)

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
        param_dict : dict
            observation parameters
        """
        param_dict = {}

        param_dict['cbf_channel_freqs'] = table(
            self.getkeyword('SPECTRAL_WINDOW')).getcol('CHAN_FREQ')[0]
        param_dict['cbf_n_chans'] = table(self.getkeyword('SPECTRAL_WINDOW')).getcol('NUM_CHAN')[0]
        param_dict['cbf_n_pols'] = table(self.getkeyword('POLARIZATION')).getcol('NUM_CORR')[0]
        param_dict['sdp_l0_int_time'] = self.getcol('EXPOSURE')[0]
        antenna_names = [ant for ant in self.ants]
        param_dict['antenna_mask'] = ','.join(antenna_names)
        param_dict['cbf_n_ants'] = len(self.ants)
        # need polarisation information in the sdp_l0_bls_ordering
        param_dict['sdp_l0_bls_ordering'] = self.corr_products
        param_dict['cbf_sync_time'] = 0.0
        param_dict['config'] = {'MS_simulator': True}
        # antenna descriptions for all antennas
        antenna_descriptions = self.get_antdesc()
        for antname in antenna_names:
            param_dict['{0}_observer'.format(antname,)] = antenna_descriptions[antname]

        return param_dict

    def get_corrprods(self, antlist):
        """
        Gets correlation product list from MS

        Parameters
        ----------
        antlist : antenna names, list of strings

        Returns
        -------
        corrprods : :class:`np.ndarray` of str
            polarisation-correlation product pairs, shape (nbl*npol, 2)
        corrprods_nopol : :class:`np.ndarray` of str
            correlation product pairs, shape (nbl, 2)
        """
        # get baseline ordering for first timestamp
        time = self.getcol('TIME')
        a1 = self.getcol('ANTENNA1')[time == time[0]]
        a2 = self.getcol('ANTENNA2')[time == time[0]]

        # determine antenna name ordering of antennas from MS baseline ordering indices
        corrprods_nopol = np.array([[antlist[a1i], antlist[a2i]] for a1i, a2i in zip(a1, a2)])

        # determine polarisation ordering of the MS data
        npols = table(self.getkeyword('POLARIZATION')).getcol('NUM_CORR')
        if npols == 4:
            pol_order = np.array([['h', 'h'], ['h', 'v'], ['v', 'h'], ['v', 'v']])
        elif npols == 2:
            pol_order = np.array([['h', 'h'], ['v', 'v']])
        elif npols == 1:
            pol_order = np.array([['h', 'h']])
        else:
            raise ValueError('Weird polarisation setup!')
        # combine antenna and polarisation strongs to get full correlator product ordering
        corrprods = np.array([[c1+p1, c2+p2] for c1, c2 in corrprods_nopol for p1, p2 in pol_order])

        return corrprods, corrprods_nopol

    def select(self, **kwargs):
        """
        Allows MS simulator to emulate katdal style data selection.
        Currently only selects on channel.
        """
        chan_slice = None if 'channels' not in kwargs else kwargs['channels']
        self.data_mask = np.s_[:, chan_slice, ...]

    def tx_data(self, ts, tx, max_scans):
        """
        Iterates through MS file and transmits data as a SPEAD stream,
        also updating the telescope state accordingly.

        Parameters
        ----------
        ts : :class:`katsdptelstate.TelescopeState`
            Telescope State
        tx : :class:`spead2.send.UdpStream`
            SPEAD transmitter
        max_scans : int
            Maximum number of scans to transmit
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

        # fake obs params for now
        self.setup_obs_params(ts, t=self.to_ut(self.getcol('TIME')[0]))

        time_ind = 0
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
            target_desc = '{0}, radec {1}, {2}, {3}'.format(
                field_names[f_id], tag, str(radec.ra), str(radec.dec))

            # MS files only have tracks (?)
            scan_state = 'track'

            scan_time = tscan.getcol('TIME', startrow=0, nrow=1)[0]
            scan_time_ut = self.to_ut(scan_time)
            for ant in antlist:
                ts.add('{0}_target'.format(ant,), target_desc, ts=scan_time_ut-random()*0.1)
                ts.add('{0}_activity'.format(ant,), scan_state, ts=scan_time_ut-random()*0.1)
            ts.add('cbf_target', target_desc, ts=scan_time_ut-random()*0.1)
            print 'Scan', scan_ind+1, '/', max_scans, ' -- ',
            n_ts = len(tscan.select('unique TIME'))
            print 'timestamps:', n_ts, ' -- ',
            print scan_state, target_desc

            # transmit the data timestamp by timestamp
            for ttime in tscan.iter('TIME'):
                # get data to transmit from MS
                tx_time = self.to_ut(ttime.getcol('TIME')[0])  # time
                # visibilities for this time stamp, for specified channel range
                tx_vis = np.hstack(ttime.getcol('DATA')[self.data_mask])
                # flags for this time stamp, for specified channel range
                tx_flags = np.hstack(ttime.getcol('FLAG')[self.data_mask])
                try:
                    # weights for this time stamp, for specified channel range
                    tx_weights = np.hstack(ttime.getcol('WEIGHT_SPECTRUM')[self.data_mask])
                except RuntimeError:
                    # WEIGHT_SPECTRUM column doesn't exist: mock up weights as zeros
                    tx_weights = np.zeros_like(tx_flags, dtype=np.float32)

                # on first transmittion, set up item group, using info from first data item
                if 'correlator_data' not in ig:
                    self.setup_ig(ig, tx_vis, tx_flags, tx_weights)
                # transmit timestamps, vis, flags, weights
                self.transmit_item(tx, ig, tx_time, tx_vis, tx_flags, tx_weights)

                time_ind += 1

            if scan_ind+1 == max_scans:
                break

        # end transmission
        tx.send_heap(ig.get_end())

        # MS only has 'track' scans?
        print 'Track timestamps:', time_ind
        print 'Slew timestamps: ', 0
        print 'Total timestamps:', time_ind

    def write_data(self, correlator_data, flags, ti_max, cal_bls_ordering, cal_pol_ordering,
                   bchan=1, echan=0):
        """
        Writes data into MS file.

        Parameters
        ----------
        correlator_data : :class:`np.ndarray`
            visibilities
        flags : :class:`np.ndarray`
            flags
        ti_max : int
            index of highest timestamp of supplies correlator_data and flag arrays
        cal_bls_ordering : list of lists
            baseline ordering of visibility data in the pipleine, shape (nbl, 2)
        cal_pol_ordering : list of lists
            polarisation pair ordering of visibility data in the pipleine, shape (npol, 2)
        bchan : int
            start channel to write
        echan : int
            end channel to write
        """
        # determine re-ordering necessary to munge pipeline visibility ordering
        # back to MS ordering
        ordermask, desired = get_reordering_nopol(
            self.ants, cal_bls_ordering, output_order_bls=self.bls_ordering)

        # write MS polarisation table
        pol_num = {'h': 0, 'v': 1}
        pol_types = {'hh': 9, 'vv': 12, 'hv': 10, 'vh': 11}
        pol_type_array = np.array([pol_types[p1+p2] for p1, p2 in cal_pol_ordering])[np.newaxis, :]
        pol_index_array = np.array([[pol_num[p1], pol_num[p2]] for p1, p2 in cal_pol_ordering],
                                   dtype=np.int32)[np.newaxis, :]
        poltable = table(self.getkeyword('POLARIZATION'), readonly=False)
        poltable.putcol('CORR_TYPE', pol_type_array)
        poltable.putcol('CORR_PRODUCT', pol_index_array)

        # sort data by time for writing to MS
        ms_sorted = self.sort('TIME')
        # write data to MS timestamp by timestamp
        data = None
        for ti, ms_time in enumerate(ms_sorted.iter('TIME')):
            # if we don't yet have a data array, create one
            if data is None:
                data = np.zeros_like(ms_time.getcol('DATA'))
            data[:, bchan:echan, :] = np.rollaxis(correlator_data[ti][..., ordermask], -1, 0)
            # save data to CORRECTED_DATA column of the MS
            ms_time.putcol('CORRECTED_DATA', data)
            # break when we have reached the max timestamp index in the vis data
            if ti == ti_max-1:
                break

# -------------------------------------------------------------------------------------------------
# --- SimDataH5 class
# ---   simulates pipeline data from H5 file
# -------------------------------------------------------------------------------------------------

# generic functions for H5 simulator, for V2 and V3 katdal H5 files


def h5_get_params(h5data):
    """
    Add key value pairs from H5 file to to parameter dictionary.

    Returns
    -------
    param_dict : dictionary of observation parameters
    """
    param_dict = {}

    param_dict['cbf_channel_freqs'] = h5data.channel_freqs
    param_dict['sdp_l0_int_time'] = h5data.dump_period
    param_dict['cbf_n_ants'] = len(h5data.ants)
    param_dict['cbf_n_chans'] = len(h5data.channels)
    param_dict['sdp_l0_bls_ordering'] = h5data.corr_products

    # determine number of polarisation products
    corrprods_polonly = [[b[0][4], b[1][4]] for b in h5data.corr_products]
    # find unique pol combinations
    unique_pol = []
    for pbl in corrprods_polonly:
        if pbl not in unique_pol:
            unique_pol.append(pbl)
    param_dict['cbf_n_pols'] = len(unique_pol)

    param_dict['cbf_sync_time'] = 0.0
    antenna_mask = ','.join([ant.name for ant in h5data.ants])
    param_dict['antenna_mask'] = antenna_mask
    param_dict['config'] = {'h5_simulator': True}

    # antenna descriptions for all antennas
    for ant in h5data.ants:
        param_dict['{0}_observer'.format(ant.name,)] = ant.description

    return param_dict


def h5_tx_data(h5data, ts, tx, max_scans):
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

    # fake obs params for now
    h5data.setup_obs_params(ts, t=h5data.timestamps[0])

    for scan_ind, scan_state, target in h5data.scans():
        # update telescope state with scan information
        #   subtract random offset to time, <= 0.1 seconds, to simulate
        #   slight differences in times of different sensors
        for ant in h5data.ants:
            ts.add('{0}_target'.format(ant.name,), target.description,
                   ts=h5data.timestamps[0]-random()*0.1)
            ts.add('{0}_activity'.format(ant.name,), scan_state,
                   ts=h5data.timestamps[0]-random()*0.1)
        ts.add('cbf_target', target.description, ts=h5data.timestamps[0] - random()*0.1)
        print 'Scan', scan_ind+1, '/', max_scans, ' -- ',
        n_ts = len(h5data.timestamps)
        print 'timestamps:', n_ts, ' -- ',
        print scan_state, target.description

        # keep track of number of timestamps
        total_ts += n_ts
        if scan_state == 'track':
            track_ts += n_ts
        if scan_state == 'slew':
            slew_ts += n_ts

        # transmit the data from this scan, timestamp by timestamp
        scan_data = h5data.vis[:]
        # Hack to get around h5data.flags returning a bool of selected flags
        flags_indexer = h5data.flags
        flags_indexer.transforms = []
        scan_flags = flags_indexer[:]
        scan_weights = h5data.weights

        # set up item group, using info from first data item
        if 'correlator_data' not in ig:
            h5data.setup_ig(ig, scan_data[0], scan_flags[0], scan_weights[0])

        # transmit data timestamp by timestamp
        for i in range(scan_data.shape[0]):  # time axis

            # data values to transmit
            tx_time = h5data.timestamps[i]  # timestamp
            # visibilities for this time stamp, for specified channel range
            tx_vis = scan_data[i, :, :]
            # flags for this time stamp, for specified channel range
            tx_flags = scan_flags[i, :, :]
            tx_weights = scan_weights[i, :, :]

            # transmit timestamps, vis, flags, weights
            h5data.transmit_item(tx, ig, tx_time, tx_vis, tx_flags, tx_weights)

        if scan_ind+1 == max_scans:
            break

    # end transmission
    tx.send_heap(ig.get_end())

    print 'Track timestamps:', track_ts
    print 'Slew timestamps: ', slew_ts
    print 'Total timestamps:', total_ts


def h5_write_data(h5data, correlator_data, flags, ti_max, cal_bls_ordering, cal_pol_ordering,
                  bchan=1, echan=0):
    """
    Writes data into H5 file.

    Parameters
    ----------
    correlator_data : :class:`np.ndarray`
        visibilities
    flags : :class:`np.ndarray`
        flags
    ti_max : int
        index of highest timestamp of supplies correlator_data and flag arrays
    cal_bls_ordering : list of list
        baseline ordering of visibility data in the pipleine, shape (nbl, 2)
    cal_pol_ordering : list of list
        polarisation pair ordering of visibility data in the pipleine, shape (npol, 2)
    bchan : int
        start channel to write
    echan : int
        end channel to write
    """
    if echan == 0:
        echan = None
    # pack data into H5 correlation product list
    #    by iterating through H5 correlation product list
    for i, [ant1, ant2] in enumerate(h5data.corr_products):

        # find index of this pair in the cal product array
        antpair = [ant1[:-1], ant2[:-1]]
        cal_indx = cal_bls_ordering.index(antpair)
        # find index of this element in the pol dimension
        polpair = [ant1[-1], ant2[-1]]
        pol_indx = cal_pol_ordering.index(polpair)

        # vis shape is (ntimes, nchan, ncorrprod) for real and imag
        h5data._vis[0:ti_max, bchan:echan, i, 0] = \
            correlator_data[0:ti_max, :, pol_indx, cal_indx].real
        h5data._vis[0:ti_max, bchan:echan, i, 1] = \
            correlator_data[0:ti_max, :, pol_indx, cal_indx].imag


# katdal v3 data
class SimDataH5V3(katdal.H5DataV3):
    """
    Simulated data class.
    Uses H5 file to simulate MeerKAT pipeline data SPEAD stream and Telescope
    State, subclassing katdal H5DataV3.

    Parameters
    ----------
    file_name : str
        Name of MS file

    Attributes
    ----------
    num_scans : int
        Total number of scans
    """

    def __init__(self, file_name, **kwargs):
        mode = kwargs['mode'] if 'mode' in kwargs else 'r'
        super(SimDataH5V3, self).__init__(file_name, mode=mode)
        self.num_scans = len(self.scan_indices)
        self.file_name = file_name

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
        t : float
            Unix time in seconds

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
        return h5_get_params(self)

    def tx_data(self, ts, tx, max_scans):
        return h5_tx_data(self, ts, tx, max_scans)

    def write_data(self, correlator_data, flags, ti_max, cal_bls_ordering, cal_pol_ordering,
                   bchan=1, echan=0):
        return h5_write_data(self, correlator_data, flags, ti_max,
                             cal_bls_ordering, cal_pol_ordering, bchan, echan)


# katdal v2 data
class SimDataH5V2(katdal.H5DataV2):
    """
    Simulated data class.
    Uses H5 file to simulate MeerKAT pipeline data SPEAD stream and Telescope
    State, subclassing katdal H5DataV2.

    Parameters
    ----------
    file_name : str
        Name of MS file

    Attributes
    ----------
    num_scans : int
        Total number of scans in the MS data set
    """

    def __init__(self, file_name, **kwargs):
        mode = kwargs['mode'] if 'mode' in kwargs else 'r'
        super(SimDataH5V2, self).__init__(file_name, mode=mode)
        self.num_scans = len(self.scan_indices)
        self.file_name = file_name

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
        t : float
            Unix time in seconds

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
        return h5_get_params(self)

    def tx_data(self, ts, tx, max_scans):
        return h5_tx_data(self, ts, tx, max_scans)

    def write_data(self, correlator_data, flags, ti_max, cal_bls_ordering, cal_pol_ordering,
                   bchan=1, echan=0):
        return h5_write_data(self, correlator_data, flags, ti_max,
                             cal_bls_ordering, cal_pol_ordering, bchan, echan)
