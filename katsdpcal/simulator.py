"""
KAT-7 Simulator for testing MeerKAT calibration pipeline
========================================================
Simulator class for HDF5 files produced by KAT-7 correlator,
for testing of the MeerKAT pipeline.
"""

from __future__ import print_function, division, absolute_import

import logging
import time

from casacore.tables import table
import spead2
from spead2 import send
from .calprocs import get_reordering_nopol

import katdal
from katdal.h5datav3 import FLAG_NAMES
import katpoint
import katcp

import numpy as np
from random import random
import ephem


logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# --- simdata classes
# -------------------------------------------------------------------------------------------------


class WrongFileType(IOError):
    """Could not open the file with the lower-level API"""


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


class SimData(object):
    """Base class for simulated data.

    Subclasses wrap an existing data set to provide a common interface.

    Note that this is a factory class: constructing it will return an instance
    of a subclass.

    Parameters
    ----------
    filename : str
        name of katdal or MS file
    server : :class:`katsdptelstate.endpoint.Endpoint`
        Katcp server for cal
    bchan,echan : int
        Channel range to select out of the file. If `echan` is ``None``, the
        range extends to the last channel.
    """
    def __init__(self, filename, server=None, bchan=0, echan=None):
        if server is not None:
            self.client = katcp.BlockingClient(server.host, server.port)
            self.client.start()
        else:
            self.client = None
        self.filename = filename
        self.bchan = bchan
        self.echan = echan
        self.cbid = None
        # Subclass must provide num_scans

    @classmethod
    def factory(cls, *args, **kwargs):
        for sub_cls in [SimDataKatdal, SimDataMS]:
            try:
                return sub_cls(*args, **kwargs)
            except WrongFileType:
                pass
        raise WrongFileType('File does not exist, or is not of compatible format! '
                            '(Must be katdal or MS.)')

    def capture_init(self):
        if self.client is not None:
            self.client.wait_protocol()
            cbid = '{}'.format(int(time.time()))
            reply, informs = self.client.blocking_request(
                katcp.Message.request('capture-init', cbid))
            if not reply.reply_ok():
                raise RuntimeError('capture-init failed: {}'.format(reply.arguments[1]))
            self.cbid = cbid

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

    def setup_telstate(self, telstate):
        """
        Add key-value pairs from file to the Telescope State. This only sets
        up the immutables needed to get run_cal.py going. Sensors are added
        during transmission.

        Parameters
        ----------
        telstate : :class:`~katsdptelstate.TelescopeState`
            Telescope state.
        """
        # get parameters from subclass
        parameter_dict = self.get_params()
        # check that the minimum necessary parameters are set
        min_keys = ['sdp_l0_int_time', 'sdp_l0_bandwidth', 'sdp_l0_center_freq',
                    'sdp_l0_n_chans', 'sdp_l0_bls_ordering',
                    'sdp_l0_sync_time']
        for key in min_keys:
            if key not in parameter_dict:
                raise KeyError('Required parameter {0} not set by simulator.'.format(key))

        # add fake subarray_id to parameter_dict
        parameter_dict['subarray_product_id'] = 'unknown_subarray'

        # Adjust the frequency parameters for account for bchan, echan
        n_chans = parameter_dict['sdp_l0_n_chans']
        bandwidth = parameter_dict['sdp_l0_bandwidth']
        center_freq = parameter_dict['sdp_l0_center_freq']
        channel_width = bandwidth / n_chans

        bchan = self.bchan
        echan = self.echan if self.echan is not None else n_chans
        if not 0 <= bchan < echan <= n_chans:
            raise ValueError('Invalid channel range {}:{}'.format(bchan, echan))
        center_freq += (bchan + echan - n_chans) / 2 * channel_width
        parameter_dict['sdp_l0_center_freq'] = center_freq
        parameter_dict['sdp_l0_bandwidth'] = bandwidth * (echan - bchan) / n_chans
        parameter_dict['sdp_l0_n_chans'] = echan - bchan

        # fill in counts that depend on others
        parameter_dict['sdp_l0_n_bls'] = len(parameter_dict['sdp_l0_bls_ordering'])
        parameter_dict['sdp_l0_n_chans_per_substream'] = parameter_dict['sdp_l0_n_chans']
        # separate keys without times from those with times
        notime_dict = {key: parameter_dict[key] for key in parameter_dict.keys()
                       if not key.endswith('noise_diode')}
        time_dict = {key: parameter_dict[key] for key in parameter_dict.keys()
                          if key.endswith('noise_diode')}

        # add parameters to telescope state
        for key, value in sorted(notime_dict.items()):
            logger.info('Setting %s = %s', key, value)
            telstate.add(key, value, immutable=True)

        for key, value in sorted(time_dict.items()):
            logger.info('Setting %s', key)
            for idx, times in enumerate(value['timestamp']):
                telstate.add(key, value['value'][idx], ts=times)

    def data_to_spead(self, telstate, l0_endpoint, spead_rate=5e8, max_scans=None):
        """
        Iterates through file and transmits data as a SPEAD stream.

        Parameters
        ----------
        telstate : :class:`katsdptelstate.TelescopeState`
            Telescope State
        l0_endoint : :class:`katsdptelstate.endpoint.Endpoint`
            Endpoint for SPEAD stream
        spead_rate : float
            SPEAD data transmission rate (bytes per second)
        max_scans : int, optional
            Maximum number of scans to transmit
        """
        logging.info('TX: Initializing...')
        # configure SPEAD - may need to rate-limit transmission for laptops etc.
        config = send.StreamConfig(max_packet_size=8872, rate=spead_rate)
        tx = send.UdpStream(spead2.ThreadPool(), l0_endpoint.host, l0_endpoint.port, config)

        # if the maximum number of scans to transmit has not been
        # specified, set to total number of scans
        if max_scans is None or max_scans > self.num_scans:
            max_scans = self.num_scans

        # transmit data timestamp by timestamp and update telescope state
        self.tx_data(telstate, tx, max_scans)

    def setup_ig(self, ig, correlator_data, flags, weights):
        """
        Initialises data transmit ItemGroup for SPEAD transmit

        Parameters
        ----------
        ig : :class:`spead2.ItemGroup`
            Item group to populate
        correlator_data : array-like
            visibilities
        flags : array-like
            flags
        weights : array-like
            weights
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
        ig.add_item(id=None, name='dump_index', description='Index in time',
                    shape=(), dtype=None, format=[('u', 64)])
        ig.add_item(id=0x4103, name='frequency',
                    description="Channel index of first channel in the heap",
                    shape=(), dtype=np.uint32)

    def setup_capture_block(self, telstate, first_timestamp):
        """Set per-capture-block keys in telstate."""
        telstate.add(self.cbid + '_sdp_l0_first_timestamp', first_timestamp, immutable=True)

    def transmit_item(self, tx, ig, dump_index, timestamp, correlator_data, flags, weights):
        """
        Transmit single SPEAD :class:`~spead2.send.ItemGroup`.

        Parameters
        ----------
        tx : :class:`spead2.send.UdpStream`
            SPEAD stream
        ig : :class:`spead2.send.ItemGroup`
            Item group
        dump_index : int
            Index in time
        timestamp : :class:`np.float64`
            Timestamp
        correlator_data : :class:`np.ndarray`
            Visibilities
        flags : :class:`np.ndarray`
            Flags
        weights : :class:`np.ndarray`
            Weights
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
        ig['dump_index'].value = dump_index
        ig['frequency'].value = 0
        # send all of the descriptors with every heap
        tx.send_heap(ig.get_heap(descriptors='all'))


# -------------------------------------------------------------------------------------------------
# --- SimDataMS class
# ---   simulates pipeline data from MS
# -------------------------------------------------------------------------------------------------


class SimDataMS(SimData):
    """
    Simulated data class.
    Uses MS file to simulate MeerKAT pipeline data SPEAD stream and Telescope State,
    subclassing casacore table.

    Parameters
    ----------
    filename : str
        Name of MS file
    server : :class:`katsdptelstate.endpoint.Endpoint`
        Katcp server for cal
    bchan,echan : int, optional
        Channel range to select out of the file. If `echan` is ``None``, the
        range extends to the last channel.
    mode : {'r', 'r+'}, optional
        Either 'r' to open the table read-only or 'r+' to open it read-write

    Attributes
    ----------
    file : :class:casacore.tables.table`
        main table
    filename : str
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

    def __init__(self, filename, server=None, bchan=0, echan=None, mode='r'):
        super(SimDataMS, self).__init__(filename, server, bchan, echan)
        readonly = mode != 'r+'
        try:
            self.file = table(filename, readonly=readonly)
        except RuntimeError as error:
            raise WrongFileType(str(error))
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
        self.num_scans = max(self.file.getcol('SCAN_NUMBER'))
        self.timestamps = np.unique(
            self.file.sort('SCAN_NUMBER, TIME, ANTENNA1, ANTENNA2').getcol('TIME'))
        with self.subtable('ANTENNA') as t:
            self.ants = t.getcol('NAME')
        self.corr_products, self.bls_ordering = self.get_corrprods(self.ants)

    def subtable(self, name, *args, **kwargs):
        if 'ack' not in kwargs:
            kwargs['ack'] = False
        return table(self.file.getkeyword(name), *args, **kwargs)

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
        with self.subtable('ANTENNA') as t:
            positions = t.getcol('POSITION')
            diameters = t.getcol('DISH_DIAMETER')
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
        with self.subtable('FIELD') as t:
            return range(t.nrows())

    def get_params(self):
        """
        Add key value pairs from MS file to to parameter dictionary.

        Returns
        -------
        param_dict : dict
            observation parameters
        """
        param_dict = {}

        with self.subtable('SPECTRAL_WINDOW') as spw:
            row = 0
            chan_freq = spw.getcell('CHAN_FREQ', row)
            n_chans = len(chan_freq)
            bandwidth = spw.getcell('TOTAL_BANDWIDTH', row)
            chan_width = bandwidth / n_chans
            if n_chans > 1:
                # Check that channels are evenly spaced and in increasing order
                if chan_freq[1] < chan_freq[0]:
                    raise ValueError('Only channels in increasing order are supported')
                if not np.allclose(np.diff(chan_freq), chan_width):
                    raise ValueError('Only regularly spaced frequency channels are supported')
        param_dict['sdp_l0_n_chans'] = n_chans
        param_dict['sdp_l0_int_time'] = self.file.getcell('EXPOSURE', 0)
        param_dict['sdp_l0_bls_ordering'] = self.corr_products
        param_dict['sdp_l0_sync_time'] = 0.0
        param_dict['sdp_l0_bandwidth'] = bandwidth
        param_dict['sdp_l0_center_freq'] = chan_freq[n_chans // 2]
        # a dummy sub-band
        param_dict['sub_band'] = 'l'
        # antenna descriptions for all antennas
        antenna_descriptions = self.get_antdesc()
        for antname in self.ants:
            param_dict['{0}_observer'.format(antname)] = antenna_descriptions[antname]
            # a dummy noise diode sensor per antenna
            param_dict['{0}_dig_{1}_band_noise_diode'.format(antname, param_dict['sub_band'])] = \
                np.array([(self.timestamps[0]), (0.0)],
                         dtype=[('timestamp', float), ('value', float)])

        return param_dict

    def get_corrprods(self, antlist):
        """
        Gets correlation product list from MS

        Parameters
        ----------
        antlist : list of str
            antenna names, list of strings

        Returns
        -------
        corrprods : :class:`np.ndarray` of str
            polarisation-correlation product pairs, shape (nbl*npol, 2)
        corrprods_nopol : :class:`np.ndarray` of str
            correlation product pairs, shape (nbl, 2)
        """
        # get baseline ordering for first timestamp
        time = self.file.getcol('TIME')
        a1 = self.file.getcol('ANTENNA1')[time == time[0]]
        a2 = self.file.getcol('ANTENNA2')[time == time[0]]

        # determine antenna name ordering of antennas from MS baseline ordering indices
        corrprods_nopol = np.array([[antlist[a1i], antlist[a2i]] for a1i, a2i in zip(a1, a2)])

        # determine polarisation ordering of the MS data
        with self.subtable('POLARIZATION') as t:
            npols = t.getcol('NUM_CORR')
        if npols == 4:
            bls_pol_order = np.array([['h', 'h'], ['h', 'v'], ['v', 'h'], ['v', 'v']])
        elif npols == 2:
            bls_pol_order = np.array([['h', 'h'], ['v', 'v']])
        elif npols == 1:
            bls_pol_order = np.array([['h', 'h']])
        else:
            raise ValueError('Weird polarisation setup!')
        # combine antenna and polarisation strings to get full correlator product ordering
        corrprods = np.array([[c1+p1, c2+p2] for c1, c2 in corrprods_nopol for p1, p2 in bls_pol_order])

        return corrprods, corrprods_nopol

    def tx_data(self, telstate, tx, max_scans):
        """
        Iterates through MS file and transmits data as a SPEAD stream,
        also updating the telescope state accordingly.

        Parameters
        ----------
        telstate : :class:`katsdptelstate.TelescopeState`
            Telescope State
        tx : :class:`spead2.send.UdpStream`
            SPEAD transmitter
        max_scans : int
            Maximum number of scans to transmit
        """
        # order the data for transmission
        ordered_table = self.file.sort('SCAN_NUMBER, TIME, ANTENNA1, ANTENNA2')
        # get metadata information for the telescope state
        with self.subtable('FIELD') as t:
            field_names = t.getcol('NAME')
            positions = t.getcol('DELAY_DIR')
        with self.subtable('STATE') as t:
            intents = t.getcol('OBS_MODE')
        antlist = self.ants
        # set up ItemGroup for transmission
        flavour = spead2.Flavour(4, 64, 48)
        ig = send.ItemGroup(flavour=flavour)

        self.setup_capture_block(telstate, self.to_ut(self.file.getcell('TIME', 0)))

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

            scan_time = tscan.getcell('TIME', 0)
            scan_time_ut = self.to_ut(scan_time)
            for ant in antlist:
                telstate.add('{0}_target'.format(ant), target_desc, ts=scan_time_ut-random()*0.1)
                telstate.add('{0}_activity'.format(ant), scan_state, ts=scan_time_ut-random()*0.1)
            telstate.add('cbf_target', target_desc, ts=scan_time_ut-random()*0.1)
            n_ts = len(tscan.select('unique TIME'))
            logger.info('Scan %d/%d -- timestamps: %d -- %s %s',
                        scan_ind+1, max_scans, n_ts, scan_state, target_desc)

            # transmit the data timestamp by timestamp
            for ttime in tscan.iter('TIME'):
                # getcolslice takes inclusive end coordinates, hence the - 1
                chan_kw = dict(blc=[self.bchan, 0],
                               trc=[-1 if self.echan is None else self.echan - 1, -1])
                # get data to transmit from MS
                tx_time = self.to_ut(ttime.getcell('TIME', 0))  # time
                # visibilities for this time stamp, for specified channel range
                tx_vis = np.hstack(ttime.getcolslice('DATA', **chan_kw))
                # flags for this time stamp, for specified channel range
                tx_flags = np.hstack(ttime.getcolslice('FLAG', **chan_kw))
                tx_flags = tx_flags * np.uint8(2**FLAG_NAMES.index('ingest_rfi'))
                try:
                    # weights for this time stamp, for specified channel range
                    tx_weights = np.hstack(ttime.getcolslice('WEIGHT_SPECTRUM', **chan_kw))
                except RuntimeError:
                    # WEIGHT_SPECTRUM column doesn't exist: mock up weights as zeros
                    tx_weights = np.zeros_like(tx_flags, dtype=np.float32)

                # on first transmittion, set up item group, using info from first data item
                if 'correlator_data' not in ig:
                    self.setup_ig(ig, tx_vis, tx_flags, tx_weights)
                # transmit timestamps, vis, flags, weights
                self.transmit_item(tx, ig, time_ind, tx_time, tx_vis, tx_flags, tx_weights)

                time_ind += 1

            if scan_ind+1 == max_scans:
                break

        # end transmission
        tx.send_heap(ig.get_end())

        # MS only has 'track' scans?
        logger.info('Track timestamps: %d', time_ind)
        logger.info('Slew timestamps:  %d', 0)
        logger.info('Total timestamps: %d', time_ind)

    def write_data(self, correlator_data, flags, ti_max, cal_bls_ordering, cal_bls_pol_ordering,
                   bchan=0, echan=None):
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
        cal_bls_pol_ordering : list of lists
            polarisation pair ordering of visibility data in the pipeline, shape (npol, 2)
        bchan : int, optional
            start channel to write
        echan : int, optional
            end channel to write
        """
        # determine re-ordering necessary to munge pipeline visibility ordering
        # back to MS ordering
        ordermask, desired = get_reordering_nopol(
            self.ants, cal_bls_ordering, output_order_bls=self.bls_ordering)

        # write MS polarisation table
        pol_num = {'h': 0, 'v': 1}
        pol_types = {'hh': 9, 'vv': 12, 'hv': 10, 'vh': 11}
        pol_type_array = np.array([pol_types[p1+p2] for p1, p2 in cal_bls_pol_ordering])[np.newaxis, :]
        pol_index_array = np.array([[pol_num[p1], pol_num[p2]] for p1, p2 in cal_bls_pol_ordering],
                                   dtype=np.int32)[np.newaxis, :]
        with self.subtable('POLARIZATION', readonly=False) as poltable:
            poltable.putcol('CORR_TYPE', pol_type_array)
            poltable.putcol('CORR_PRODUCT', pol_index_array)

        # sort data by time for writing to MS
        ms_sorted = self.file.sort('TIME')
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

    def close(self):
        self.file.close()
        super(SimDataMS, self).close()


# -------------------------------------------------------------------------------------------------
# --- SimDataKatdal class
# ---   simulates pipeline data from katdal file
# -------------------------------------------------------------------------------------------------


class SimDataKatdal(SimData):
    def __init__(self, filename, server=None, bchan=0, echan=None, mode='r'):
        super(SimDataKatdal, self).__init__(filename, server, bchan, echan)
        try:
            self.file = katdal.open(filename, mode=mode)
        except IOError as error:
            raise WrongFileType(str(error))
        self.file.select(channels=slice(bchan, echan))
        self.num_scans = len(self.file.scan_indices)

    def get_params(self):
        param_dict = {}
        spw = self.file.spectral_windows[self.file.spw]
        if spw.sideband != 1:
            raise ValueError('Lower sideband is not supported')
        # spw will describe the whole band, ignoring bchan:echan. The caller
        # takes care of the adjustment.
        param_dict['sdp_l0_bandwidth'] = spw.channel_width * spw.num_chans
        param_dict['sdp_l0_center_freq'] = spw.centre_freq
        param_dict['sdp_l0_n_chans'] = spw.num_chans
        param_dict['sdp_l0_int_time'] = self.file.dump_period
        param_dict['sdp_l0_bls_ordering'] = self.file.corr_products
        param_dict['sdp_l0_sync_time'] = 0.0
        param_dict['sub_band'] = self.file._get_telstate_attr('sub_band', default='',
                                                              no_unpickle=('l', 's', 'u', 'x'))

        # antenna descriptions for all antennas
        for ant in self.file.ants:
            param_dict['{0}_observer'.format(ant.name)] = ant.description
            param_dict['{0}_dig_{1}_band_noise_diode'.format(ant.name, param_dict['sub_band'])] = \
                       self.file.sensor.get('Antennas/{0}/nd_coupler'.format(ant.name),
                                            extract=False)

        return param_dict

    def tx_data(self, telstate, tx, max_scans):
        """
        Iterates through katdal file and transmits data as a spead stream,
        also updating the telescope state accordingly.

        Parameters
        ----------
        telstate : :class:`katsdptelstate.TelescopeState`
            Telescope State
        tx : :class:`spead2.send.UdpSender`
            SPEAD transmitter
        max_scans : int
            Maximum number of scans to transmit
        """
        total_ts, track_ts, slew_ts = 0, 0, 0

        flavour = spead2.Flavour(4, 64, 48)
        ig = send.ItemGroup(flavour=flavour)

        self.setup_capture_block(telstate, self.file.timestamps[0])

        for scan_ind, scan_state, target in self.file.scans():
            # update telescope state with scan information
            #   subtract random offset to time, <= 0.1 seconds, to simulate
            #   slight differences in times of different sensors
            ts0 = self.file.timestamps[0]    # First timestamp in scan
            for ant in self.file.ants:
                telstate.add('{0}_target'.format(ant.name), target.description,
                             ts=ts0-random()*0.1)
                telstate.add('{0}_activity'.format(ant.name,), scan_state,
                             ts=ts0-random()*0.1)
            telstate.add('cbf_target', target.description, ts=ts0 - random()*0.1)
            n_ts = len(self.file.timestamps)
            logger.info('Scan %d/%d -- timestamps: %d -- %s, %s',
                        scan_ind+1, max_scans, n_ts, scan_state, target.description)

            # keep track of number of timestamps (total_ts is handled separately)
            if scan_state == 'track':
                track_ts += n_ts
            if scan_state == 'slew':
                slew_ts += n_ts

            # transmit the data from this scan, timestamp by timestamp
            scan_data = self.file.vis[:]
            # Hack to get around flags returning a bool of selected flags
            flags_indexer = self.file.flags
            flags_indexer.transforms = []
            scan_flags = flags_indexer[:]
            scan_weights = self.file.weights

            # set up item group, using info from first data item
            if 'correlator_data' not in ig:
                self.setup_ig(ig, scan_data[0], scan_flags[0], scan_weights[0])

            # transmit data timestamp by timestamp
            for i in range(scan_data.shape[0]):  # time axis

                # data values to transmit
                tx_time = self.file.timestamps[i]  # timestamp
                # visibilities for this time stamp, for specified channel range
                tx_vis = scan_data[i, :, :]
                # flags for this time stamp, for specified channel range
                tx_flags = scan_flags[i, :, :]
                tx_weights = scan_weights[i, :, :]

                # transmit timestamps, vis, flags, weights
                self.transmit_item(tx, ig, total_ts, tx_time, tx_vis, tx_flags, tx_weights)
                total_ts += 1

            if scan_ind+1 == max_scans:
                break

        # end transmission
        tx.send_heap(ig.get_end())

        logger.info('Track timestamps: %d', track_ts)
        logger.info('Slew timestamps:  %d', slew_ts)
        logger.info('Total timestamps: %d', total_ts)

    def write_data(self, correlator_data, flags, ti_max, cal_bls_ordering, cal_bls_pol_ordering,
                   bchan=0, echan=None):
        """
        Writes data into katdal file.

        Parameters
        ----------
        correlator_data : :class:`np.ndarray`
            visibilities
        flags : :class:`np.ndarray`
            flags
        ti_max : int
            index of highest timestamp of supplied correlator_data and flag arrays
        cal_bls_ordering : list of list
            baseline ordering of visibility data in the pipeline, shape (nbl, 2)
        cal_bls_pol_ordering : list of list
            polarisation pair ordering of visibility data in the pipleine, shape (npol, 2)
        bchan : int, optional
            start channel to write
        echan : int, optional
            end channel to write
        """
        # pack data into katdal correlation product list
        #    by iterating through katdal correlation product list
        for i, [ant1, ant2] in enumerate(self.file.corr_products):

            # find index of this pair in the cal product array
            antpair = [ant1[:-1], ant2[:-1]]
            cal_indx = cal_bls_ordering.index(antpair)
            # find index of this element in the pol dimension
            polpair = [ant1[-1], ant2[-1]]
            pol_indx = cal_bls_pol_ordering.index(polpair)

            # vis shape is (ntimes, nchan, ncorrprod) for real and imag
            self.file._vis[0:ti_max, bchan:echan, i, 0] = \
                correlator_data[0:ti_max, :, pol_indx, cal_indx].real
            self.file._vis[0:ti_max, bchan:echan, i, 1] = \
                correlator_data[0:ti_max, :, pol_indx, cal_indx].imag
