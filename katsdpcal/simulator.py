"""
Replayer for testing MeerKAT calibration pipeline
=================================================

It generates telstate state information and a SPEAD stream based on an
existing katdal dataset or Measurement Set.
"""

import logging
import time
import asyncio

import spead2.send.asyncio

import katsdpservices
import katdal
import katpoint
import aiokatcp
import async_timeout

import numpy as np
import dask.array as da
from random import random
import ephem


logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# --- simdata classes
# -------------------------------------------------------------------------------------------------


class WrongFileType(IOError):
    """Could not open the file with the lower-level API"""


def get_antdesc_relative(names, diameters, positions):
    """Determine antenna description dictionary.

    This uses offsets from a lat-lon-alt reference position.

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


def get_params(self):
    param_dict = {}
    spw = self.file.spectral_windows[self.file.spw]
    if spw.sideband != 1:
        raise ValueError('Lower sideband is not supported')
    # spw will describe the whole band, ignoring bchan:echan. The caller
    # takes care of the adjustment.

    # katsdpmodel keys
    telstate = self.file.source.telstate
    correlator_stream = telstate.view('sdp_l0')['src_streams'][0]
    f_engine_stream = telstate.view(correlator_stream)['src_streams'][0]
    ins_name = 'instrument_dev_name'
    # create sensor dictionary here and add them directly to the telstate :)

    param_dict['sdp_l0_bandwidth'] = spw.channel_width * spw.num_chans
    param_dict['sdp_l0_center_freq'] = spw.centre_freq
    param_dict['sdp_l0_n_chans'] = spw.num_chans
    param_dict['sdp_l0_int_time'] = self.file.dump_period
    param_dict['sdp_l0_bls_ordering'] = self.file.corr_products
    param_dict['obs_label'] = self.file.sensor['obs_label']
    param_dict['sub_band'] = self.file.spectral_windows[self.file.spw].band.lower()[0]

    telstate_immutables = ['sdp_l0_sync_time', 'sdp_l0_src_streams', 'sdp_l0_stream_type',
                           'chunk_info', 'first_timestamp', 'sub_pool_resources', 'sub_product',
                           'obs_params', 'stream_name', 'capture_block_id',
                           f'{correlator_stream}_{ins_name}', f'{correlator_stream}_int_time',
                           f'{correlator_stream}_n_accs', f'{f_engine_stream}_{ins_name}',
                           'wide_scale_factor_timestamp', 'wide_sync_time']

    for key in telstate_immutables:
        param_dict[key] = telstate[key]

    telstate_mutables = ['cbf_target', 'obs_label', 'obs_activity']
    for key in telstate_mutables:
        param_dict[key] = telstate.get_range(key, st=0)

    band_mask_key = telstate.join(f_engine_stream, 'model', 'band_mask', 'fixed')
    model_keys = [band_mask_key, 'sdp_model_base_url',
                  'model_rfi_mask_fixed', 'sdp_l0_src_streams',
                  telstate.join(correlator_stream, 'src_streams')]
    for key in model_keys:
        param_dict[key] = telstate[key]

    # antenna descriptions and noise diodes for all antennas
    for ant in self.file.ants:
        param_dict['{0}_observer'.format(ant.name)] = ant.description
        nd_name = '{0}_dig_{1}_band_noise_diode'.format(ant.name, param_dict['sub_band'])
        param_dict[nd_name] = telstate.get_range(nd_name, st=0)

        pol_list = ['h', 'v']
        for pol in pol_list:
            voltage_sensor = 'wide_antenna_channelised_voltage_{0}{1}_eq'.format(ant.name, pol)
            param_dict[voltage_sensor] = telstate.get_range(voltage_sensor, st=0)
    return param_dict


def write_data(self, correlator_data, flags, ti_max, cal_bls_ordering, cal_bls_pol_ordering,
               bchan=0, echan=None):
    """Writes data into katdal file.

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


class SimData:
    """Base class for simulated data

    Parameters
    ----------
    filename : str
        name of katdal file
    servers : list of :class:`katsdptelstate.endpoint.Endpoint`
        Katcp server(s) for cal
    bchan,echan : int
        Channel range to select out of the file. If `echan` is ``None``, the
         range extends to the last channel.
    n_substreams : int
        Number of substreams within the SPEAD stream.
    """
    def __init__(self, filename, servers=None, bchan=0, echan=None, n_substreams=1):
        if servers is None:
            servers = []
        if n_substreams is None:
            n_substreams = max(1, len(servers))
        self.clients = [aiokatcp.Client(server.host, server.port) for server in servers]
        self.filename = filename
        self.bchan = bchan
        self.echan = echan
        self.cbid = None
        self.n_substreams = n_substreams
        self._tx_future = None    # Future for heaps in flight

        try:
            self.file = katdal.open(filename, upgrade_flags=False)
        except IOError as error:
            raise WrongFileType(str(error)) from error
        self.file.select(channels=slice(bchan, echan))
        self.num_scans = len(self.file.scan_indices)

    async def tx_data(self, telstate, tx, max_scans):
        """Transmits katdal dataset as a SPEAD stream.

        Iterates through katdal file and transmits data as a SPEAD stream,
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
        ig = spead2.send.ItemGroup(flavour=flavour)

        # subtract sync_time to realign to the observation timing
        sync_time = self.file.source.telstate['sync_time']
        self.setup_capture_block(telstate, (self.file.timestamps - sync_time)[0])
        telstate_cb = telstate.view(self.cbid)

        # include obs params in telstate
        telstate_cb['obs_params'] = self.file.obs_params
        for scan_ind, scan_state, target in self.file.scans():
            # update telescope state with scan information
            #   subtract random offset to time, <= 0.1 seconds, to simulate
            #   slight differences in times of different sensors
            ts0 = self.file.timestamps[0]    # First timestamp in scan
            telstate.add('cbf_target', target.description, ts=ts0 - random()*0.1)
            telstate_cb.add('obs_activity', scan_state, ts=ts0 - random()*0.1)
            n_ts = len(self.file.timestamps)
            logger.info('Scan %d/%d -- timestamps: %d -- %s, %s',
                        scan_ind+1, max_scans, n_ts, scan_state, target.description)

            # keep track of number of timestamps (total_ts is handled separately)
            if scan_state == 'track':
                track_ts += n_ts
            if scan_state == 'slew':
                slew_ts += n_ts

            # Hack to get around flags returning a bool of selected flags
            self.file.flags._transforms = []
            scan_data = self.file.vis.dataset
            scan_flags = self.file.flags.dataset
            scan_weights = self.file.weights.dataset

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
                # If we have Dask datasets, it's time to compute them
                if isinstance(tx_vis, da.Array):
                    tx_vis, tx_flags, tx_weights = da.compute(tx_vis, tx_flags, tx_weights)

                # transmit timestamps, vis, flags, weights
                await self.transmit_item(tx, ig, total_ts, tx_time, tx_vis, tx_flags, tx_weights)
                total_ts += 1

            if scan_ind+1 == max_scans:
                break

        # end transmission
        await self.transmit_end(tx, ig)

        logger.info('Track timestamps: %d', track_ts)
        logger.info('Slew timestamps:  %d', slew_ts)
        logger.info('Total timestamps: %d', total_ts)

    async def capture_init(self):
        cbid = '{}'.format(int(time.time()))
        self.cbid = cbid
        for client in self.clients:
            await client.wait_connected()
            await client.request('capture-init', cbid)

    async def capture_done(self):
        for client in self.clients:
            try:
                with async_timeout.timeout(10):
                    await client.wait_connected()
                    await client.request('capture-done')
            except asyncio.TimeoutError:
                pass

    async def close(self):
        for client in self.clients:
            client.close()
            await client.wait_closed()
        self.clients.clear()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    def setup_telstate(self, telstate):
        """Add key-value pairs from file to the Telescope State.

        This only sets up the immutables needed to get run_cal.py going.
        Sensors are added during transmission.

        Parameters
        ----------
        telstate : :class:`~katsdptelstate.TelescopeState`
            Telescope state.
        """
        # get parameters
        parameter_dict = get_params(self)
        # check that the minimum necessary parameters are set
        min_keys = ['sdp_l0_int_time', 'sdp_l0_bandwidth', 'sdp_l0_center_freq',
                    'sdp_l0_n_chans', 'sdp_l0_bls_ordering',
                    'sdp_l0_sync_time']
        for key in min_keys:
            if key not in parameter_dict:
                raise KeyError('Required parameter {0} not set by simulator.'.format(key))

        parameter_dict['sdp_l0_need_weights_power_scale'] = False
        parameter_dict['sdp_l0_excise'] = True
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
        n_chans = echan - bchan
        if n_chans % self.n_substreams != 0:
            raise ValueError('number of substreams must divide into the number of channels')
        parameter_dict['sdp_l0_n_chans_per_substream'] = n_chans // self.n_substreams

        # separate keys without times from those with times
        sensor_key_suffixes = ('obs_activity', '_eq', 'cbf_target', 'target_activity', 'obs_label',
                               'noise_diode')
        sensor_keys = [k for k in parameter_dict.keys() if k.endswith(sensor_key_suffixes)]

        for k in parameter_dict.keys():
            if k in sensor_keys:
                logger.info('Setting %s', k)
                for v, t in parameter_dict[k]:
                    telstate.add(k, v, ts=t)
            else:
                logger.info('Setting %s = %s', k, parameter_dict[k])
                telstate[k] = parameter_dict[k]

    async def data_to_spead(self, telstate, l0_endpoints, spead_rate=5e8, max_scans=None,
                            interface=None):
        """Iterates through file and transmits data as a SPEAD stream.

        Parameters
        ----------
        telstate : :class:`katsdptelstate.TelescopeState`
            Telescope State
        l0_endpoints : sequence of :class:`katsdptelstate.endpoint.Endpoint`
            Endpoints for SPEAD stream
        spead_rate : float
            SPEAD data transmission rate (bytes per second)
        max_scans : int, optional
            Maximum number of scans to transmit
        interface : str, optional
            Name of the interface on which to transmit data, if it is multicast.
        """
        if self.n_substreams % len(l0_endpoints) != 0:
            raise ValueError('Number of endpoints must divide into number of substreams')
        logging.info('TX: Initializing...')
        # configure SPEAD - may need to rate-limit transmission for laptops etc.
        config = spead2.send.StreamConfig(
            max_packet_size=8872, rate=spead_rate, max_heaps=self.n_substreams)
        if interface:
            interface_address = katsdpservices.get_interface_address(interface)
        else:
            interface_address = ''
        tx = spead2.send.asyncio.UdpStream(
            spead2.ThreadPool(),
            [(l0_endpoint.host, l0_endpoint.port) for l0_endpoint in l0_endpoints],
            config,
            interface_address=interface_address)

        # if the maximum number of scans to transmit has not been
        # specified, set to total number of scans
        if max_scans is None or max_scans > self.num_scans:
            max_scans = self.num_scans

        # transmit data timestamp by timestamp and update telescope state
        await self.tx_data(telstate, tx, max_scans)

    def _substream_slice(self, array, substream):
        """Extract the part of an array appropriate to a specific substream.

        Parameters
        ----------
        array : array-like
            Array to slice. Frequency must be the first axis.
        substream : int
            Index of the substream
        """
        assert array.shape[0] % self.n_substreams == 0
        n_chans_per_substream = array.shape[0] // self.n_substreams
        chan0 = n_chans_per_substream * substream
        chan1 = chan0 + n_chans_per_substream
        return array[chan0 : chan1]

    def setup_ig(self, ig, correlator_data, flags, weights):
        """Initialises data transmit ItemGroup for SPEAD transmit.

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
                    shape=self._substream_slice(correlator_data, 0).shape,
                    dtype=correlator_data.dtype)
        ig.add_item(id=None, name='flags', description="Flags for visibilities",
                    shape=self._substream_slice(flags, 0).shape,
                    dtype=flags.dtype)
        # Note: real ingest sends weights as uint8, but we want to be able to
        # send the same weights we retrieved from the file, so we use float32,
        # and set weights_channel to 1.0.
        weights_substream = self._substream_slice(weights, 0)
        ig.add_item(id=None, name='weights', description="Weights for visibilities",
                    shape=weights_substream.shape,
                    dtype=np.float32)
        ig.add_item(id=None, name='weights_channel', description='Per-channel scaling for weights',
                    shape=(weights_substream.shape[0],), dtype=np.float32)
        ig.add_item(id=None, name='timestamp', description="Seconds since sync time",
                    shape=(), dtype=None, format=[('f', 64)])
        ig.add_item(id=None, name='dump_index', description='Index in time',
                    shape=(), dtype=None, format=[('u', 64)])
        ig.add_item(id=0x4103, name='frequency',
                    description="Channel index of first channel in the heap",
                    shape=(), dtype=np.uint32)

    def setup_capture_block(self, telstate, first_timestamp):
        """Set per-capture-block keys in telstate."""
        key = telstate.join(self.cbid, 'sdp_l0', 'first_timestamp')
        telstate[key] = first_timestamp

    async def transmit_item(self, tx, ig, dump_index, timestamp, correlator_data, flags, weights):
        """Transmit single SPEAD :class:`~spead2.send.ItemGroup` per server.

        This does not actually wait for the item to be sent. It waits for the
        previous set of item groups to be sent. This allows for concurrent data
        preparation and transmission.

        Parameters
        ----------
        tx : :class:`spead2.send.asyncio.UdpStream`
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
        heaps = []
        for i in range(self.n_substreams):
            # transmit vis, flags and weights, timestamp
            ig['correlator_data'].value = self._substream_slice(correlator_data, i)
            ig['flags'].value = self._substream_slice(flags, i)
            substream_weights = self._substream_slice(weights, i)
            ig['weights_channel'].value = np.ones(substream_weights.shape[:-1], np.float32)
            ig['weights'].value = substream_weights
            ig['timestamp'].value = timestamp
            ig['dump_index'].value = dump_index
            ig['frequency'].value = correlator_data.shape[0] // self.n_substreams * i
            # We map endpoints to spead2 substreams, so we need to convert our
            # substream index into an endpoint index.
            endpoint = i // (self.n_substreams // tx.num_substreams)
            # send all of the descriptors with every heap.
            heap = ig.get_heap(descriptors='all')
            heaps.append(spead2.send.HeapReference(heap, substream_index=endpoint))
        if self._tx_future is not None:
            await self._tx_future
            self._tx_future = None
        self._tx_future = tx.async_send_heaps(heaps, spead2.send.GroupMode.ROUND_ROBIN)

    async def transmit_end(self, tx, ig):
        """Send end-of-stream notification to each endpoint."""
        if self._tx_future is not None:
            await self._tx_future
            self._tx_future = None
        for i in range(tx.num_substreams):
            await tx.async_send_heap(ig.get_end(), substream_index=i)
