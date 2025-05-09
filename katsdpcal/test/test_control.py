"""Tests for control module"""

from multiprocessing import Process
import multiprocessing
import multiprocessing.dummy
import tempfile
import shutil
import os
import itertools
from unittest import mock, IsolatedAsyncioTestCase
import asyncio
import json
import datetime

import numba
import numpy as np
import pytest
import scipy.interpolate

import spead2
import aiokatcp
from aiokatcp import FailReply
import async_timeout
import katsdptelstate
from katsdptelstate.endpoint import Endpoint
import katpoint
from katdal.h5datav3 import FLAG_NAMES
from katdal.applycal import complex_interp
from katsdpcalproc import calprocs

from katsdpcal import control, pipelineprocs, param_dir


numba.config.THREADING_LAYER = 'safe'


def get_sent_heaps(send_stream):
    """Extracts the heaps that a :class:`spead2.send.InprocStream` sent."""
    decoder = spead2.recv.Stream(
        spead2.ThreadPool(),
        spead2.recv.StreamConfig(stop_on_stop_item=False)
    )
    send_stream.queues[0].stop()
    decoder.add_inproc_reader(send_stream.queues[0])
    return list(decoder)


def test_shared_empty():
    def sender(a):
        a[:] = np.outer(np.arange(5), np.arange(3))

    a = control.shared_empty((5, 3), np.int32)
    a.fill(0)
    assert a.shape == (5, 3)
    assert a.dtype == np.int32
    p = Process(target=sender, args=(a,))
    p.start()
    p.join()
    expected = np.outer(np.arange(5), np.arange(3))
    np.testing.assert_equal(expected, a)


class PingTask(control.Task):
    """Task class for the test. It receives numbers on a queue, and updates
    a sensor in response.
    """

    def __init__(self, task_class, master_queue, slave_queue):
        super().__init__(task_class, master_queue, 'PingTask')
        self.slave_queue = slave_queue

    def get_sensors(self):
        return [
            aiokatcp.Sensor(int, 'last-value', 'last number received on the queue'),
            aiokatcp.Sensor(int, 'error', 'sensor that is set to error state')
        ]

    def run(self):
        self.sensors['error'].set_value(0, status=aiokatcp.Sensor.Status.ERROR,
                                        timestamp=123456789.0)
        while True:
            number = self.slave_queue.get()
            if number < 0:
                break
            self.sensors['last-value'].set_value(number)
        self.master_queue.put(control.StopEvent())


class BaseTestTask:
    """Tests for :class:`katsdpcal.control.Task`.

    This is a base class, which is subclassed for each process class.
    """

    def setup_method(self):
        self.master_queue = self.module.Queue()
        self.slave_queue = self.module.Queue()

    def _check_reading(self, event, name, value,
                       status=aiokatcp.Sensor.Status.NOMINAL, timestamp=None):
        assert isinstance(event, control.SensorReadingEvent)
        assert event.name == name
        assert event.reading.value == value
        assert event.reading.status == status
        if timestamp is not None:
            assert event.reading.timestamp == timestamp

    def test(self):
        task = PingTask(self.module.Process, self.master_queue, self.slave_queue)
        assert task.daemon is False   # Test the wrapper property
        task.daemon = True       # Ensure it gets killed if the test fails
        assert task.name == 'PingTask'
        task.start()

        event = self.master_queue.get()
        self._check_reading(event, 'error', 0, aiokatcp.Sensor.Status.ERROR, 123456789.0)
        assert task.is_alive()

        self.slave_queue.put(3)
        event = self.master_queue.get()
        self._check_reading(event, 'last-value', 3)

        self.slave_queue.put(-1)   # Stops the slave
        event = self.master_queue.get()
        assert isinstance(event, control.StopEvent)

        task.join()
        assert not task.is_alive()


class TestTaskMultiprocessing(BaseTestTask):
    module = multiprocessing

    def test_terminate(self):
        task = PingTask(self.module.Process, self.master_queue, self.slave_queue)
        task.start()
        task.terminate()
        task.join()


class TestTaskDummy(BaseTestTask):
    module = multiprocessing.dummy


class ServerData:
    """Test data associated with a single simulated cal server"""

    def make_parameters(self, telstate_l0):
        param_file = os.path.join(param_dir, 'pipeline_parameters_meerkat_L_wide.txt')
        parameters = pipelineprocs.parameters_from_file(param_file)
        pipelineprocs.finalise_parameters(parameters, telstate_l0,
                                          self.testcase.n_servers, self.server_id)
        pipelineprocs.parameters_to_telstate(parameters, telstate_l0.root(), 'sdp_l0test')
        return parameters

    def __init__(self, testcase, server_id):
        self.testcase = testcase
        self.server_id = server_id
        self.parameters = self.make_parameters(testcase.telstate_l0)

        self.report_path = tempfile.mkdtemp()
        testcase.addCleanup(shutil.rmtree, self.report_path)
        self.log_path = tempfile.mkdtemp()
        testcase.addCleanup(shutil.rmtree, self.log_path)

        # Time, channels, pols, baselines
        buffer_shape = (60, testcase.n_channels // testcase.n_servers,
                        4, testcase.n_baselines // 4)
        self.buffers = buffers = control.create_buffer_arrays(buffer_shape, False)
        flags_streams = [
            control.FlagsStream(
                name='sdp_l1_flags_test', endpoints=testcase.flags_endpoints[0],
                rate_ratio=64.0, src_stream='sdp_l0test'),
            control.FlagsStream(
                name='sdp_l1_continuum_flags_test', endpoints=testcase.flags_endpoints[1],
                rate_ratio=64.0, src_stream='sdp_l0test_continuum',
                continuum_factor=4)
        ]
        self.server = control.create_server(
            False, '127.0.0.1', 0, buffers,
            'sdp_l0test', testcase.l0_endpoints, None,
            flags_streams, 1.0,
            testcase.telstate_cal, self.parameters, self.report_path, self.log_path, None)
        self.client = None
        self.testcase = testcase

    async def start(self):
        await self.server.start()
        # We can't simply do an addCleanup to stop the server, because the servers
        # need to be shut down together (otherwise the dump alignment code in
        # Accumulator._accumulate will deadlock). Instead, tell the testcase that
        # we require cleanup.
        self.testcase.cleanup_servers.append(self)

        bind_address = self.server.server.sockets[0].getsockname()
        self.client = await aiokatcp.Client.connect(
            bind_address[0], bind_address[1], auto_reconnect=False)
        self.testcase.addAsyncCleanup(self.client.wait_closed)
        self.testcase.addCleanup(self.client.close)

    async def stop_server(self):
        await self.server.shutdown()
        await self.server.stop()


class TestCalDeviceServer(IsolatedAsyncioTestCase):
    """Tests for :class:`katsdpcal.control.CalDeviceServer.

    This does not test the quality of the solutions that are produced, merely
    that they are produced and calibration reports written.
    """

    def populate_telstate_cb(self, telstate, cb='cb'):
        telstate_cb_l0 = telstate.view(telstate.join(cb, 'sdp_l0test'))
        telstate_cb_l0['first_timestamp'] = 100.0
        telstate_cb = telstate.view(cb)
        telstate_cb.add('obs_activity', 'track', ts=0)
        obs_params = {'description' : 'test observation',
                      'proposal_id' : '123_03',
                      'sb_id_code' : '123_0005',
                      'observer' : 'Kim'}
        telstate_cb['obs_params'] = obs_params

    def populate_telstate(self, telstate_l0):
        telstate = telstate_l0.root()
        bls_ordering = []
        # target model must match the model used by the pipeline in order to produce
        # meaningful calibration solutions. The pipeline model is supplied in
        # conf/sky_models/J1331+3030.txt
        target = ('J1331+3030, radec bfcal single_accumulation, 13:31:08.29, +30:30:33.0, '
                  '(0 50e3 0.1823 1.4757 -0.4739 0.0336)')
        ant_bls = []     # Antenna pairs, later expanded to pol pairs
        for a in self.antennas:
            ant_bls.append((a, a))
        for a in self.antennas:
            for b in self.antennas:
                if a < b:
                    ant_bls.append((a, b))
        for a, b in ant_bls:
            bls_ordering.append((a + 'h', b + 'h'))
            bls_ordering.append((a + 'v', b + 'v'))
            bls_ordering.append((a + 'h', b + 'v'))
            bls_ordering.append((a + 'v', b + 'h'))
        telstate['subarray_product_id'] = 'c856M4k'
        telstate['sub_band'] = 'l'
        telstate.add('cbf_target', target, ts=0)
        telstate_l0['int_time'] = 4.0
        telstate_l0['bls_ordering'] = bls_ordering
        telstate_l0['n_bls'] = len(bls_ordering)
        telstate_l0['bandwidth'] = 856000000.0
        telstate_l0['center_freq'] = 1284000000.0
        telstate_l0['n_chans'] = self.n_channels
        telstate_l0['n_chans_per_substream'] = self.n_channels_per_substream
        telstate_l0['sync_time'] = 1400000000.0
        telstate_l0['excise'] = True
        telstate_l0['need_weights_power_scale'] = True
        self.populate_telstate_cb(telstate)
        for antenna in self.antennas:
            # The position is irrelevant for now, so just give all the
            # antennas the same position.
            telstate.add(
                '{}_observer'.format(antenna),
                '{}, -30:42:47.4, 21:26:38.0, 1035.0, 13.5, -351.163669759 384.481835294, '
                '-0:05:44.7 0 0:00:22.6 -0:09:04.2 0:00:11.9 -0:00:12.8 -0:04:03.5 0 0 '
                '-0:01:33.0 0:01:45.6 0 0 0 0 0 -0:00:03.6 -0:00:17.5, 1.22'.format(antenna))

    def add_items(self, ig):
        channels = self.telstate_l0.n_chans_per_substream
        baselines = len(self.telstate_l0.bls_ordering)
        ig.add_item(id=None, name='correlator_data', description="Visibilities",
                    shape=(channels, baselines), dtype=np.complex64)
        ig.add_item(id=None, name='flags', description="Flags for visibilities",
                    shape=(channels, baselines), dtype=np.uint8)
        ig.add_item(id=None, name='weights',
                    description="Detailed weights, to be scaled by weights_channel",
                    shape=(channels, baselines), dtype=np.uint8)
        ig.add_item(id=None, name='weights_channel', description="Coarse (per-channel) weights",
                    shape=(channels,), dtype=np.float32)
        ig.add_item(id=None, name='timestamp', description="Seconds since sync time",
                    shape=(), dtype=None, format=[('f', 64)])
        ig.add_item(id=None, name='dump_index', description='Index in time',
                    shape=(), dtype=None, format=[('u', 64)])
        ig.add_item(id=0x4103, name='frequency',
                    description="Channel index of first channel in the heap",
                    shape=(), dtype=np.uint32)

    def init_item_group(self):
        """Initalise a :class:`spead2.send.ItemGroup` and send it to servers."""
        self.ig = spead2.send.ItemGroup()
        self.add_items(self.ig)
        for endpoint in self.l0_endpoints:
            self.l0_streams[endpoint].send_heap(self.ig.get_heap(descriptors='all'))

    def patch(self, *args, **kwargs):
        patcher = mock.patch(*args, **kwargs)
        mock_obj = patcher.start()
        self.addCleanup(patcher.stop)
        return mock_obj

    def _get_output_stream(self, thread_pool, endpoints, config,
                           *args, **kwargs):
        """Mock implementation of UdpStream that returns an InprocStream instead.

        It stores the stream in self.output_streams, keyed by hostname and port.
        """
        assert len(endpoints) == 1
        key = Endpoint(*endpoints[0])
        assert key not in self.output_streams
        stream = spead2.send.InprocStream(thread_pool, [spead2.InprocQueue()])
        self.output_streams[key] = stream
        return stream

    async def _stop_servers(self):
        """Similar to shutdown_servers, but run as part of cleanup"""
        await asyncio.gather(*[server.stop_server() for server in self.cleanup_servers])

    def setUp(self):
        self.n_channels = 4096
        self.n_substreams = 8    # L0 substreams
        self.n_endpoints = 4     # L0 endpoints
        self.n_servers = 2
        assert self.n_channels % self.n_substreams == 0
        assert self.n_channels % self.n_servers == 0
        self.n_channels_per_substream = self.n_channels // self.n_substreams
        self.antennas = ["m090", "m091", "m092", "m093"]
        self.n_antennas = len(self.antennas)
        self.n_baselines = self.n_antennas * (self.n_antennas + 1) * 2

        self.telstate = katsdptelstate.TelescopeState()
        self.telstate_cal = self.telstate.view('cal')
        self.telstate_l0 = self.telstate.view('sdp_l0test')
        self.populate_telstate(self.telstate_l0)
        self.dump_period = self.telstate_l0.int_time
        telstate_cb_l0 = control.make_telstate_cb(self.telstate_l0, 'cb')
        self.first_dump_ts = self.telstate_l0.sync_time + telstate_cb_l0.first_timestamp

        self.l0_endpoints = [Endpoint('239.102.255.{}'.format(i), 7148)
                             for i in range(self.n_endpoints)]
        substreams_per_endpoint = self.n_substreams // self.n_endpoints
        self.substream_endpoints = [self.l0_endpoints[i // substreams_per_endpoint]
                                    for i in range(self.n_substreams)]
        self.flags_endpoints = [
            [Endpoint('239.102.253.{}'.format(i), 7148) for i in range(self.n_servers)],
            [Endpoint('239.102.254.{}'.format(i), 7148) for i in range(self.n_servers)]
        ]

        self.l0_queues = {endpoint: spead2.InprocQueue() for endpoint in self.l0_endpoints}
        self.l0_streams = {}
        sender_thread_pool = spead2.ThreadPool()
        # For each server we only actually use a single queue (the readers on
        # the other endpoints will just get no data). This ensures that
        # heaps are received in a predictable order and not affected by timing.
        endpoints_per_server = self.n_endpoints // self.n_servers
        for i, endpoint in enumerate(self.l0_endpoints):
            # Compute last endpoint index of the server. We use the last
            # rather than the first as a quick workaround for
            # https://github.com/ska-sa/spead2/issues/40
            base = (i // endpoints_per_server + 1) * endpoints_per_server - 1
            queue = self.l0_queues[self.l0_endpoints[base]]
            stream = spead2.send.InprocStream(sender_thread_pool, [queue])
            stream.set_cnt_sequence(i, self.n_endpoints)
            self.l0_streams[endpoint] = stream
        self.init_item_group()

        # Need a real function to use in the mock, otherwise it doesn't become
        # a bound method.
        def _add_udp_reader(stream, port, max_size=None, buffer_size=None,
                            bind_hostname='', socket=None):
            queue = self.l0_queues[Endpoint(bind_hostname, port)]
            stream.add_inproc_reader(queue)

        self.patch('spead2.recv.asyncio.Stream.add_udp_reader', _add_udp_reader)
        self.output_streams = {}
        self.patch('spead2.send.UdpStream', self._get_output_stream)

        # Trying to run two dask distributed clients in the same process doesn't
        # work so well, so don't try
        self.patch('dask.distributed.LocalCluster')
        self.patch('dask.distributed.Client')

    async def asyncSetUp(self):
        self.cleanup_servers = []
        self.addAsyncCleanup(self._stop_servers)
        self.servers = [ServerData(self, i) for i in range(self.n_servers)]
        for server in self.servers:
            await server.start()

    async def make_request(self, name, *args, timeout=15):
        """Issue a request to all the servers, and check that the result is ok.

        Parameters
        ----------
        name : str
            Request name
        args : list
            Arguments to the request
        timeout : float
            Time limit for the request

        Returns
        -------
        informs : list of lists
            Informs returned with the reply from each server
        """
        with async_timeout.timeout(timeout):
            coros = [server.client.request(name, *args)
                     for server in self.servers]
            results = await asyncio.gather(*coros)
            return [informs for reply, informs in results]

    async def get_sensor(self, name):
        """Retrieves a sensor value and checks that the value is well-defined.

        Returns
        -------
        values : list of str
            The sensor values (per-server), in the string form it is sent in the protocol
        """
        values = []
        informs_list = await self.make_request('sensor-value', name)
        for informs in informs_list:
            assert len(informs) == 1
            assert informs[0].arguments[3] in (b'nominal', b'warn', b'error')
            values.append(informs[0].arguments[4])
        return values

    async def assert_sensor_value(self, name, expected):
        """Retrieves a sensor value and compares its value.

        The returned string is automatically cast to the type of `expected`.
        """
        values = await self.get_sensor(name)
        for i, value in enumerate(values):
            value = type(expected)(value)
            assert value == expected, \
                "Wrong value for {} ({!r} != {!r})".format(name, value, expected)

    async def assert_request_fails(self, msg_re, name, *args):
        """Assert that a request fails, and test the error message against
        a regular expression."""
        for server in self.servers:
            with pytest.raises(FailReply, match=msg_re):
                await server.client.request(name, *args)

    async def test_empty_capture(self):
        """Terminating a capture with no data must succeed and not write a report.

        It must also correctly remove the capture block from capture-block-state.
        """
        await self.make_request('capture-init', 'cb')
        await self.assert_sensor_value('capture-block-state', b'{"cb": "CAPTURING"}')
        for stream in self.l0_streams.values():
            stream.send_heap(self.ig.get_end())
        await self.make_request('capture-done')
        await self.make_request('shutdown')
        for server in self.servers:
            assert os.listdir(server.report_path) == []
        await self.assert_sensor_value('reports-written', 0)
        await self.assert_sensor_value('capture-block-state', b'{}')

    async def test_init_when_capturing(self):
        """capture-init fails when already capturing"""
        for stream in self.l0_streams.values():
            stream.send_heap(self.ig.get_end())
        await self.make_request('capture-init', 'cb')
        await self.assert_request_fails(r'capture already in progress', 'capture-init', 'cb')

    async def test_done_when_not_capturing(self):
        """capture-done fails when not capturing"""
        await self.assert_request_fails(r'no capture in progress', 'capture-done')
        await self.make_request('capture-init', 'cb')
        for stream in self.l0_streams.values():
            stream.send_heap(self.ig.get_end())
        await self.make_request('capture-done')
        await self.assert_request_fails(r'no capture in progress', 'capture-done')

    def make_vis(self, K, G, target, noise=np.array([])):
        """
        Compute visibilities for the supplied target, delays [K] and gains [G]

        Parameters:
        -----------
        K : :class: `np.ndarray`
            delays, real, shape (2, n_ants)
        G : :class: `np.nadarray`
            gains, complex, shape (2, n_ants)
        target : katpoint Target
            target
        Returns:
        --------
        vis : :class: `np.ndarray`
            visibilities(n_freqs, ncorr)
        """
        bandwidth = self.telstate_l0.bandwidth
        # The + bandwidth is to convert to L band
        freqs = np.arange(self.n_channels) / self.n_channels * bandwidth + bandwidth
        # The pipeline models require frequency in MHz
        flux_density = target.flux_density(freqs / 1e6)[:, np.newaxis]
        freqs = freqs[:, np.newaxis]

        bls_ordering = self.telstate_l0.bls_ordering
        ant1 = [self.antennas.index(b[0][:-1]) for b in bls_ordering]
        ant2 = [self.antennas.index(b[1][:-1]) for b in bls_ordering]
        pol1 = ['vh'.index(b[0][-1]) for b in bls_ordering]
        pol2 = ['vh'.index(b[1][-1]) for b in bls_ordering]

        vis = flux_density * np.exp(2j * np.pi * (K[pol1, ant1] - K[pol2, ant2]) * freqs) \
            * (G[pol1, ant1] * G[pol2, ant2].conj())

        if noise.size > 0:
            noiseboth = noise[:, pol1, ant1] + noise[:, pol2, ant2]
            vis += noiseboth
        return vis

    def prepare_heaps(self, n_times, rs=None, vis=None, flags=None,
                      weights=None, weights_channel=None, corruption=False):
        """Generate a list of L0 SPEAD heaps.

        The basic dimensions of the visibility array are `n_times` (T),
        `self.n_channels` (F) and `self.n_baselines` (B), respectively.
        The input data arrays will be broadcast to the specified shape
        if a different shape is provided.

        Parameters
        ----------
        n_times : int
            Number of dumps to produce (T)
        rs : :class:`np.random.RandomState`, optional
            Random generator used to shuffle the heaps of one dump. If
            ``None``, they are not shuffled.
        vis : array of complex64, shape (T, F, B), optional
            Visibilities, default 1.0 + 0.0j
        flags : array of uint8, shape (T, F, B), optional
            Flags, default 0
        weights : array of uint8, shape (T, F, B), optional
            Detailed weights, to be scaled by `weights_channel`, default 1
        weights_channel : array of float32, shape (T, F), optional
            Coarse (per-channel) weights, defaults to a ramp
        corruption : bool, optional
            If True, corrupt some times to check that RFI flagging is working

        Returns
        -------
        heaps : list of (`Endpoint`, `spead2.send.Heap`)
            List of heaps, length `n_times` * `n_substreams`
        """
        def _corrupt(vis, t):
            return vis + np.complex64(1e9j) if corruption and t in (4, 17) else vis

        shape = (n_times, self.n_channels, self.n_baselines)
        # Default data values
        if vis is None:
            vis = np.ones(1, np.complex64)
        if flags is None:
            flags = np.zeros(1, np.uint8)
        if weights is None:
            weights = np.ones(1, np.uint8)
        if weights_channel is None:
            weights_channel = np.arange(1, n_times * self.n_channels + 1,
                                        dtype=np.float32).reshape(n_times, -1)
        # To support large arrays without excessive memory, we use
        # broadcast_to to generate the full-size array with only a
        # select element of backing storage.
        vis = np.broadcast_to(vis, shape)
        flags = np.broadcast_to(flags, shape)
        weights = np.broadcast_to(weights, shape)
        weights_channel = np.broadcast_to(weights_channel, shape[:2])
        # Time of first dump, in seconds since CBF sync time
        ts = self.first_dump_ts - self.telstate_l0.sync_time
        channel_slices = [np.s_[i * self.n_channels_per_substream :
                                (i + 1) * self.n_channels_per_substream]
                          for i in range(self.n_substreams)]
        heaps = []
        for i in range(n_times):
            dump_heaps = []
            for endpoint, s in zip(self.substream_endpoints, channel_slices):
                self.ig['correlator_data'].value = _corrupt(vis[i, s], i)
                self.ig['flags'].value = flags[i, s]
                self.ig['weights'].value = weights[i, s]
                self.ig['weights_channel'].value = weights_channel[i, s]
                self.ig['timestamp'].value = ts
                self.ig['dump_index'].value = i
                # Channel index of first channel in the heap
                self.ig['frequency'].value = np.uint32(s.start)
                dump_heaps.append((endpoint, self.ig.get_heap()))
            if rs is not None:
                rs.shuffle(dump_heaps)
            heaps.extend(dump_heaps)
            ts += self.dump_period
        return heaps

    async def shutdown_servers(self, timeout):
        inform_lists = await self.make_request('shutdown', timeout=timeout)
        for informs in inform_lists:
            progress = [inform.arguments[0] for inform in informs]
            assert progress == [
                b'Accumulator stopped',
                b'Pipeline stopped',
                b'Sender stopped',
                b'ReportWriter stopped'
            ]

    @classmethod
    def metadata_dict(cls, st=None):
        """
        Produce a metadata dictionary
        Parameters:
        -----------
        st : int
            time of first dump
        """
        metadata = {}
        product_type = {}
        product_type['ProductTypeName'] = 'MeerKATReductionProduct'
        product_type['ReductionName'] = 'Calibration Report'
        metadata['ProductType'] = product_type
        # format time as required
        time = datetime.datetime.utcfromtimestamp(st)
        metadata['StartTime'] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        metadata['CaptureBlockId'] = 'cb'
        metadata['Description'] = 'test observation' + ' cal report'
        metadata['ProposalId'] = '123_03'
        metadata['Observer'] = 'Kim'
        metadata['ScheduleBlockIdCode'] = '123_0005'
        return metadata

    def assemble_bandpass(self, telstate_cb_cal, bp_key):
        """
        Assemble a complete bandpass from the parts stored in
        telstate. Check that each part has the expected shape and dtype.

        Parameters:
        -----------
        telstate_cb_cal : :class:`katsdptelstate.TelescopeState`
            telstate view to retrieve bandpass from
        bandpass_key : str
            telstate key of the bandpass
        Returns:
        --------
        bandpass : :class: `np.ndarray`
            bandpass, complex, shape (n_chans, n_pols, n_ants)
        """
        B = []
        for i in range(self.n_servers):
            cal_product_Bn = telstate_cb_cal.get_range(bp_key+'{}'.format(i), st=0)
            assert len(cal_product_Bn) == 1
            Bn, Bn_ts = cal_product_Bn[0]
            assert Bn.dtype == np.complex64
            assert Bn.shape == (self.n_channels // self.n_servers, 2, self.n_antennas)
            B.append(Bn)
        assert bp_key+'{}'.format(self.n_servers) not in telstate_cb_cal
        return np.concatenate(B), Bn_ts

    def interp_B(self, B):
        """
        Linearly interpolate NaN'ed channels in supplied bandbass [B]

        Parameters:
        -----------
        B : :class: `np.ndarray`
            bandpass, complex, shape (n_chans, n_pols, n_ants)
        Returns:
        --------
        B_interp : :class: `np.ndarray`
        """
        n_chans, n_pols, n_ants = B.shape
        B_interp = np.empty((n_chans, n_pols, n_ants), dtype=np.complex64)
        for p in range(n_pols):
            for a in range(n_ants):
                valid = np.isfinite(B[:, p, a])
                if valid.any():
                    B_interp[:, p, a] = complex_interp(
                        np.arange(n_chans), np.arange(n_chans)[valid], B[:, p, a][valid])
        return B_interp

    @classmethod
    def normalise_phase(cls, value, ref):
        """Multiply `value` by an amount that sets `ref` to zero phase."""
        ref_phase = ref / np.abs(ref)
        return value * ref_phase.conj()

    async def test_capture(self, expected_g=1, expected_BG_rtol=1e-2,
                           expected_BCROSS_DIODE_rtol=1e-3,
                           expected_K_rtol=1e-3):
        """Tests the capture with some data, and checks that solutions are
        computed and a report written.
        """
        # Number of dumps spent tracking and awaiting, respectively
        n_track = 25
        n_await = 3
        n_times = n_track + n_await
        rs = np.random.RandomState(seed=1)

        target = katpoint.Target(self.telstate.cbf_target)
        for antenna in self.antennas:
            self.telstate.add('{0}_dig_l_band_noise_diode'.format(antenna),
                              1, ts=self.first_dump_ts - 2 * self.dump_period)
            self.telstate.add('{0}_dig_l_band_noise_diode'.format(antenna),
                              0, ts=self.first_dump_ts + (n_times + 2) * self.dump_period)
        telstate_cb = self.telstate.view('cb')
        telstate_cb.add('obs_activity', 'await_pipeline',
                        ts=self.first_dump_ts + (n_track - 0.5) * self.dump_period)

        K = rs.uniform(-50e-12, 50e-12, (2, self.n_antennas))
        G = rs.uniform(2.0, 4.0, (2, self.n_antennas)) \
            + 1j * rs.uniform(-0.1, 0.1, (2, self.n_antennas))

        vis = self.make_vis(K, G, target)
        flags = np.zeros(vis.shape, np.uint8)
        # Set flag on one channel per baseline, to test the baseline permutation.
        for i in range(flags.shape[1]):
            flags[i, i] = 1 << FLAG_NAMES.index('ingest_rfi')
        weights = rs.uniform(64, 255, vis.shape).astype(np.uint8)
        weights_channel = rs.uniform(1.0, 4.0, (self.n_channels,)).astype(np.float32)

        heaps = self.prepare_heaps(n_times, rs, vis, flags, weights, weights_channel,
                                   corruption=True)
        for endpoint, heap in heaps:
            self.l0_streams[endpoint].send_heap(heap)
        await self.make_request('capture-init', 'cb')
        await asyncio.sleep(1)
        await self.assert_sensor_value('accumulator-capture-active', 1)
        await self.assert_sensor_value('capture-block-state', b'{"cb": "CAPTURING"}')
        for stream in self.l0_streams.values():
            stream.send_heap(self.ig.get_end())
        await self.shutdown_servers(180)
        await self.assert_sensor_value('accumulator-capture-active', 0)
        await self.assert_sensor_value('input-heaps-total',
                                       n_times * self.n_substreams // self.n_servers)
        await self.assert_sensor_value('accumulator-batches', 2)
        await self.assert_sensor_value('accumulator-observations', 1)
        await self.assert_sensor_value('pipeline-last-slots', n_await)
        await self.assert_sensor_value('reports-written', 1)
        # Check that the slot accounting all balances
        await self.assert_sensor_value('slots', 60)
        await self.assert_sensor_value('accumulator-slots', 0)
        await self.assert_sensor_value('pipeline-slots', 0)
        await self.assert_sensor_value('free-slots', 60)
        await self.assert_sensor_value('capture-block-state', b'{}')

        report_last_path = await self.get_sensor('report-last-path')
        for server in self.servers:
            reports = os.listdir(server.report_path)
            assert len(reports) == 1
            report = os.path.join(server.report_path, reports[0])
            assert os.path.isfile(
                os.path.join(report, 'calreport{}.html'.format(server.server_id + 1))
            )
            assert os.path.samefile(report, report_last_path[server.server_id])
            # Check that metadata file is written and correct
            start_of_first_dump = self.first_dump_ts - 0.5 * self.dump_period
            meta_expected = self.metadata_dict(start_of_first_dump)
            meta_expected['Run'] = server.server_id + 1
            meta_file = os.path.join(report, 'metadata.json')
            assert os.path.isfile(meta_file)
            with open(meta_file, 'r') as infile:
                meta_out = json.load(infile)
            assert meta_out == meta_expected

        telstate_cb_cal = control.make_telstate_cb(self.telstate_cal, 'cb')
        cal_product_B_parts = telstate_cb_cal['product_B_parts']
        assert cal_product_B_parts == self.n_servers
        cal_product_EPOINT_parts = telstate_cb_cal['product_EPOINT_parts']
        assert cal_product_EPOINT_parts == self.n_servers
        ret_B, ret_B_ts = self.assemble_bandpass(telstate_cb_cal, 'product_B')

        cal_product_G = telstate_cb_cal.get_range('product_G', st=0)
        assert len(cal_product_G) == expected_g
        ret_G, ret_G_ts = cal_product_G[0]
        assert ret_G.dtype == np.complex64
        assert np.count_nonzero(np.isnan(ret_G)) == 0
        # Scale the returned G by the sqrt of the measured flux density of the model
        ret_F = telstate_cb_cal['measured_flux']
        ret_F_scale = np.sqrt(ret_F.get(target.name, 1.0))
        ret_BG = ret_B * ret_G[np.newaxis, :, :] / ret_F_scale
        BG = np.broadcast_to(G[np.newaxis, :, :], ret_BG.shape)
        # cal puts NaNs in B in the channels for which it applies the static
        # RFI mask, interpolate across these
        ret_BG_interp = self.interp_B(ret_BG)
        np.testing.assert_allclose(np.abs(BG), np.abs(ret_BG_interp), rtol=expected_BG_rtol)
        np.testing.assert_allclose(self.normalise_phase(BG, BG[..., [0]]),
                                   self.normalise_phase(ret_BG_interp, ret_BG_interp[..., [0]]),
                                   rtol=expected_BG_rtol)

        cal_product_K = telstate_cb_cal.get_range('product_K', st=0)
        assert len(cal_product_K) == 1
        ret_K, ret_K_ts = cal_product_K[0]
        assert ret_K.dtype == np.float32
        np.testing.assert_allclose(K - K[:, [0]], ret_K - ret_K[:, [0]], rtol=expected_K_rtol)

        # check SNR products are in telstate
        cal_product_SNR_K = telstate_cb_cal.get_range('product_SNR_K', st=0)
        assert len(cal_product_SNR_K) == 1
        ret_SNR_K, ret_SNR_K_ts = cal_product_SNR_K[0]
        assert ret_SNR_K.dtype == np.float32
        assert ret_SNR_K_ts == ret_K_ts

        for i in range(self.n_servers):
            cal_product_SNR_B = telstate_cb_cal.get_range('product_SNR_B{0}'.format(i))
            assert len(cal_product_SNR_B) == 1
            ret_SNR_B, ret_SNR_B_ts = cal_product_SNR_B[0]
            assert ret_SNR_K.dtype == np.float32
            assert ret_SNR_B_ts == ret_B_ts

        cal_product_SNR_G = telstate_cb_cal.get_range('product_SNR_G', st=0)
        assert len(cal_product_SNR_G) == expected_g
        ret_SNR_G, ret_SNR_G_ts = cal_product_SNR_G[0]
        assert ret_SNR_G.dtype == np.float32
        assert ret_SNR_G_ts == ret_G_ts

        if 'bfcal' in target.tags:
            # Check KCROSS_DIODE
            cal_product_KCROSS_DIODE = telstate_cb_cal.get_range('product_KCROSS_DIODE', st=0)
            assert len(cal_product_KCROSS_DIODE) == 1
            ret_KCROSS_DIODE, ret_KCROSS_DIODE_ts = cal_product_KCROSS_DIODE[0]
            assert ret_KCROSS_DIODE.dtype == np.float32
            np.testing.assert_allclose(K - K[1] - (ret_K - ret_K[1]),
                                       ret_KCROSS_DIODE, rtol=expected_K_rtol)
            # Check BCROSS_DIODE
            ret_BCROSS_DIODE, ret_BCROSS_DIODE_ts = self.assemble_bandpass(telstate_cb_cal,
                                                                           'product_BCROSS_DIODE')
            ret_BCROSS_DIODE_interp = self.interp_B(ret_BCROSS_DIODE)
            np.testing.assert_allclose(np.ones(ret_BCROSS_DIODE.shape),
                                       np.abs(ret_BCROSS_DIODE_interp), rtol=1e-6)
            BG_angle = np.angle(BG)
            ret_BG_interp_angle = np.angle(ret_BG_interp)
            np.testing.assert_allclose(BG_angle - BG_angle[:, [1], :]
                                       - (ret_BG_interp_angle - ret_BG_interp_angle[:, [1], :]),
                                       np.angle(ret_BCROSS_DIODE_interp),
                                       rtol=expected_BCROSS_DIODE_rtol)

            # Check BCROSS_DIODE_SKY
            ret_DIODE_SKY, ret_DIODE_SKY_ts = self.assemble_bandpass(telstate_cb_cal,
                                                                     'product_BCROSS_DIODE_SKY')
            ret_DIODE_SKY_interp = self.interp_B(ret_DIODE_SKY)
            np.testing.assert_allclose(np.ones(ret_BCROSS_DIODE.shape),
                                       np.abs(ret_DIODE_SKY_interp), rtol=1e-3)

            bcross_sky_spline = self.telstate_cal.get('bcross_sky_spline')
            bandwidth = self.telstate_l0.bandwidth
            freqs = np.arange(self.n_channels) / self.n_channels * bandwidth + bandwidth
            spline_angle = np.float32(scipy.interpolate.splev(freqs/1e6, bcross_sky_spline))

            DIODE_SKY = np.nanmedian(ret_BCROSS_DIODE, axis=-1, keepdims=True)
            DIODE_SKY_angle = np.angle(DIODE_SKY, deg=True)
            # Spline is 'hv' gain, vis pol axes ordering is 'hv, vh'
            # Correction therefore divides first pol by 'hv' spline
            DIODE_SKY_angle[:, 0, :] -= spline_angle[:, np.newaxis]
            DIODE_SKY_angle = np.broadcast_to(DIODE_SKY_angle, ret_BCROSS_DIODE.shape)
            np.testing.assert_allclose(DIODE_SKY_angle,
                                       np.angle(ret_DIODE_SKY, deg=True), rtol=1e-3)

        if 'polcal' in target.tags:
            cal_product_KCROSS = telstate_cb_cal.get_range('product_KCROSS', st=0)
            assert len(cal_product_KCROSS) == 1
            ret_KCROSS, ret_KCROSS_ts = cal_product_KCROSS[0]
            assert ret_KCROSS.dtype == np.float32
            KCROSS = K - K[1] - (ret_K - ret_K[1])
            np.testing.assert_allclose(np.mean(KCROSS, axis=1)[..., np.newaxis],
                                       ret_KCROSS, rtol=1e-3)

        # Check that flags were transmitted
        assert (
            set(self.output_streams.keys()) ==
            set(self.flags_endpoints[0] + self.flags_endpoints[1])
        )
        continuum_factors = [1, 4]
        # Time of first dump, in seconds since CBF sync time
        first_ts = self.first_dump_ts - self.telstate_l0.sync_time
        for stream_idx, continuum_factor in enumerate(continuum_factors):
            for i, endpoint in enumerate(self.flags_endpoints[stream_idx]):
                heaps = get_sent_heaps(self.output_streams[endpoint])
                assert len(heaps) == n_times + 2  # 2 extra for start and end heaps == n_times + 2
                for j, heap in enumerate(heaps[1:-1]):
                    items = spead2.ItemGroup()
                    items.update(heap)
                    ts = items['timestamp'].value
                    assert pytest.approx(ts, abs=1e-7) == (
                        first_ts + j * self.dump_period
                    )
                    idx = items['dump_index'].value
                    assert idx == j
                    assert items['frequency'].value == (
                        i * self.n_channels // self.n_servers // continuum_factor
                    )
                    out_flags = items['flags'].value
                    # Mask out the ones that get changed by cal
                    mask = 1 << FLAG_NAMES.index('cal_rfi')
                    expected = flags[self.servers[i].parameters['channel_slice']]
                    expected = calprocs.wavg_flags_f(expected, continuum_factor, expected, axis=0)
                    np.testing.assert_array_equal(out_flags & ~mask, expected)
        # Validate the flag information in telstate. We'll just validate the
        # continuum version, since that's the trickier case.
        ts_flags = self.telstate.root().view('sdp_l1_continuum_flags_test')
        assert ts_flags['center_freq'] == 1284313476.5625  # Computed by hand
        assert ts_flags['n_chans'] == 1024
        assert ts_flags['n_chans_per_substream'] == 512
        for key in ['bandwidth', 'n_bls', 'bls_ordering', 'sync_time', 'int_time', 'excise']:
            assert ts_flags[key] == self.telstate_l0[key]

    async def test_capture_separate_tags(self):
        # Change the target to one with different tags
        target = ('J1331+3030, radec delaycal gaincal bpcal polcal single_accumulation, '
                  '13:31:08.29, +30:30:33.0, (0 50e3 0.1823 1.4757 -0.4739 0.0336)')
        self.telstate.add('cbf_target', target, ts=0.001)
        await self.test_capture(expected_g=2)

    async def test_capture_complex_models(self):
        # Change the target to one with a complex model,
        # (i.e. one with sources not located at the phase centre)
        target = ('J0408-6545, radec bfcal, '
                  '4:8:20.38, -65:45:09.09, (0 50e3 -113.518 110.7809 -35.2682 3.7012)')
        self.telstate.add('cbf_target', target, ts=0.002)
        # Relax the tolerances as the visibilities are generated using
        # the model given by the target string,
        # but calibration is performed using the full sky model.
        await self.test_capture(expected_BG_rtol=5e-2, expected_BCROSS_DIODE_rtol=1e-2,
                                expected_K_rtol=3e-3)

    async def test_reference_pointing_capture(self):
        """Test the reference pointing feature with some data.

        Also checks that solutions are computed and checks dtypes and shapes.
        """
        # Number of dumps spent tracking, slewing and awaiting
        n_track = 16
        n_slew = 8
        n_await = 4
        # Number of pointings
        n_pointing = 8
        n_times = n_track + n_slew + n_await
        rs = np.random.RandomState(seed=1)

        # Create target with pointingcal tag
        target = ('J1331+3030, radec pointingcal, '
                  '13:31:08.29, +30:30:33.0, (0 50e3 0.1823 1.4757 -0.4739 0.0336)')
        self.telstate.add('cbf_target', target, ts=0.003)
        target = katpoint.Target(self.telstate.cbf_target)
        telstate_cb = self.telstate.view('cb')
        start_time = self.first_dump_ts - 0.5 * self.dump_period

        # Adding track, track, slew obs activities for each offset
        n_activity = 0
        for pointing in range(n_pointing):
            telstate_cb.add('obs_activity', 'track',
                            ts=start_time + n_activity * self.dump_period)
            telstate_cb.add('obs_activity', 'slew',
                            ts=start_time + (n_activity + 2) * self.dump_period)
            n_activity += 3
        # Always track before await_pipeline, otherwise it is ignored :-)
        telstate_cb.add('obs_activity', 'track',
                        ts=start_time + (n_track + n_slew) * self.dump_period)
        telstate_cb.add('obs_activity', 'await_pipeline',
                        ts=start_time + (n_track + n_slew + 1) * self.dump_period)

        # Creating antenna objects
        ants = [self.telstate[a+"_observer"] for a in self.antennas]
        ant_objects = [katpoint.Antenna(a) for a in ants]

        max_extent = 1.0
        # Build up sequence of pointing offsets running linearly in x and y directions
        scan = np.linspace(-max_extent, max_extent, n_pointing // 2)
        offsets_along_x = np.c_[scan, np.zeros_like(scan)]
        offsets_along_y = np.c_[np.zeros_like(scan), scan]
        offsets = np.r_[offsets_along_y, offsets_along_x]
        # Add atmospheric conditions
        self.telstate.add('anc_air_pressure', 901.8, ts=0)
        self.telstate.add('anc_air_temperature', 15.9, ts=0)
        self.telstate.add('anc_air_relative_humidity', 22.3, ts=0)

        # Updating pos_actual_scan_azim/elev every 0.5 seconds (4 times per dump)
        pos_sensor_period = 0.5
        num_pos_updates = int(np.round(self.dump_period / pos_sensor_period))
        for ant in ant_objects:
            ts = start_time
            for offset in offsets:
                for i in range(num_pos_updates * (n_track + n_slew) // n_pointing):
                    # update pos_actual_scan 8 times per dump period, ie. every 0.5 seconds
                    azel = target.plane_to_sphere(
                        katpoint.deg2rad(offset[0]),
                        katpoint.deg2rad(offset[1]),
                        antenna=ant,
                        timestamp=ts,
                    )
                    azel = [katpoint.rad2deg(katpoint.wrap_angle(azel[0])),
                            katpoint.rad2deg(azel[1])]
                    self.telstate.add(ant.name + '_pos_actual_scan_azim', azel[0], ts=ts)
                    self.telstate.add(ant.name + '_pos_actual_scan_elev', azel[1], ts=ts)
                    ts = ts + (self.dump_period/8)

        # Creating antenna delays and gains
        K = rs.uniform(-50e-12, 50e-12, (2, self.n_antennas))
        G = (rs.uniform(2.0, 4.0, (2, self.n_antennas))
             + 1j * rs.uniform(-0.1, 0.1, (2, self.n_antennas)))

        # Making visibilities and preparing + sending heaps
        vis = self.make_vis(K, G, target)
        heaps = self.prepare_heaps(n_times=n_times, rs=rs, vis=vis)
        for endpoint, heap in heaps:
            self.l0_streams[endpoint].send_heap(heap)
        await self.make_request('capture-init', 'cb')
        await asyncio.sleep(1)

        for stream in self.l0_streams.values():
            stream.send_heap(self.ig.get_end())
        await self.shutdown_servers(180)

        telstate_cb_cal = control.make_telstate_cb(self.telstate_cal, 'cb')
        # Asserting dtypes, shape of cal product
        if 'pointingcal' in target.tags:
            num_chunks = telstate_cb_cal['param_epoint_freq_chunks']
            for i in range(self.n_servers):
                cal_product_EPOINTn = telstate_cb_cal.get_range('product_EPOINT{}'.format(i), st=0)
                assert len(cal_product_EPOINTn) == 1
                ret_EPOINTn, ret_EPOINTn_ts = cal_product_EPOINTn[0]
                assert ret_EPOINTn.dtype == np.float32
                assert ret_EPOINTn.shape == (num_chunks // self.n_servers, 2, self.n_antennas, 5)
            assert 'product_EPOINT{}'.format(self.n_servers) not in telstate_cb_cal

    async def test_set_refant(self):
        """Tests the capture with a noisy antenna, and checks that the reference antenna is
         not set to the noisiest antenna. Also checks that a new refant is selected for a new
         capture block if the old one is flagged
        """
        n_times = 25
        rs = np.random.RandomState(seed=1)

        target = katpoint.Target(self.telstate.cbf_target)
        for antenna in self.antennas:
            self.telstate.add('{0}_dig_l_band_noise_diode'.format(antenna),
                              1, ts=self.first_dump_ts - 2 * self.dump_period)
            self.telstate.add('{0}_dig_l_band_noise_diode'.format(antenna),
                              0, ts=self.first_dump_ts + (n_times + 2) * self.dump_period)

        K = rs.uniform(-50e-12, 50e-12, (2, self.n_antennas))
        G = rs.uniform(2.0, 4.0, (2, self.n_antennas)) \
            + 1j * rs.uniform(-0.1, 0.1, (2, self.n_antennas))

        vis = self.make_vis(K, G, target)

        # Add noise per antenna
        var = rs.uniform(0, 2, self.n_antennas)
        # ensure one antenna is noisier than the others
        var[0] += 4.0
        rs.shuffle(var)

        worst_index = np.argmax(var)
        scale = np.array([(var), (var)])
        noise = rs.normal(np.zeros((2, self.n_antennas)), scale, (vis.shape[0], 2, self.n_antennas))
        vis = self.make_vis(K, G, target, noise)
        flags = np.zeros(vis.shape, np.uint8)

        # Set flag on one channel per baseline, to test the baseline permutation.
        for i in range(flags.shape[1]):
            flags[i, i] = 1 << FLAG_NAMES.index('ingest_rfi')
        weights = rs.uniform(64, 255, vis.shape).astype(np.uint8)
        weights_channel = rs.uniform(1.0, 4.0, (self.n_channels,)).astype(np.float32)

        heaps = self.prepare_heaps(n_times, rs, vis, flags, weights, weights_channel,
                                   corruption=True)
        for endpoint, heap in heaps:
            self.l0_streams[endpoint].send_heap(heap)
        await self.make_request('capture-init', 'cb')
        await self.wait_for_heaps(n_times * self.n_substreams, 60)
        for stream in self.l0_streams.values():
            stream.send_heap(self.ig.get_end())
        await self.make_request('capture-done')
        # The pipeline has finished running when 'reports-written' increments
        await self.wait_for_sensor('reports-written', [b'1'] * self.n_servers, 240)

        # Check the pipeline did not select the noisy antenna as the refant
        telstate_cb_cal = control.make_telstate_cb(self.telstate_cal, 'cb')
        refant_name = telstate_cb_cal['refant']
        assert refant_name != self.antennas[worst_index]
        await self.assert_sensor_value('pipeline-reference-antenna', refant_name.encode())

        # Refresh ItemGroup and send it to servers.
        self.init_item_group()
        # Set up a new capture block in telstate
        self.populate_telstate_cb(self.telstate, 'cb2')
        # A new target for a new CB
        self.telstate.add('cbf_target', self.telstate.cbf_target, ts=0.02)

        # flag the refant selected in the previous capture block
        bls_ordering = self.telstate_l0.bls_ordering
        ant1 = [self.antennas.index(b[0][:-1]) for b in bls_ordering]
        ant2 = [self.antennas.index(b[1][:-1]) for b in bls_ordering]
        refant_index_cb = self.antennas.index(refant_name)
        flag_refant = np.where((np.array(ant1) == refant_index_cb) |
                               (np.array(ant2) == refant_index_cb), 1, 0).astype(np.uint8)
        flag_refant = np.broadcast_to(flag_refant, flags.shape)

        heaps = self.prepare_heaps(n_times, rs, vis, flag_refant, weights, weights_channel,
                                   corruption=True)
        for endpoint, heap in heaps:
            self.l0_streams[endpoint].send_heap(heap)
        await self.make_request('capture-init', 'cb2')
        await self.wait_for_heaps(n_times * self.n_substreams, 60)
        for stream in self.l0_streams.values():
            stream.send_heap(self.ig.get_end())

        # The pipeline has finished running when 'reports-written' increments
        await self.wait_for_sensor('reports-written', [b'2'] * self.n_servers, 240)

        # Check the pipeline did not select the now flagged antenna as the refant
        telstate_cb_cal = control.make_telstate_cb(self.telstate_cal, 'cb2')
        refant_name = telstate_cb_cal['refant']
        refant_index_cb2 = self.antennas.index(refant_name)
        assert refant_name != self.antennas[refant_index_cb]
        await self.assert_sensor_value('pipeline-reference-antenna', refant_name.encode())
        # Check the pipeline params have been updated to reflect the current and past refants
        pp = [serv.server.pipeline.parameters for serv in self.servers]
        for params in pp:
            assert params['refant_index_prev'] == refant_index_cb
            assert params['refant_index'] == refant_index_cb2

        await self.shutdown_servers(180)

    async def wait_for_heaps(self, num_heaps, timeout):
        """Wait until `num_heaps` have been delivered to the accumulator or `timeout` in secs."""
        for i in range(timeout):
            await asyncio.sleep(1)
            heaps = await self.get_sensor('input-heaps-total')
            total_heaps = sum(int(x) for x in heaps)
            if total_heaps == num_heaps:
                print('all heaps received')
                break
            print('waiting {} ({}/{} received)'.format(i, total_heaps, num_heaps))
        else:
            raise RuntimeError('Timed out waiting for the heaps to be received')

    async def test_buffer_wrap(self):
        """Test capture with more heaps than buffer slots, to check that it handles
        wrapping around the end of the buffer.
        """
        rs = np.random.RandomState(seed=1)
        n_times = 130
        for endpoint, heap in self.prepare_heaps(n_times, rs):
            self.l0_streams[endpoint].send_heap(heap)
        # Add a target change at an uneven time, so that the batches won't
        # neatly align with the buffer end. We also have to fake a slew to make
        # it work, since the batcher assumes that target cannot change without
        # an activity change (TODO: it probably shouldn't assume this).
        target = 'dummy, radec target, 13:30:00.00, +30:30:00.0'
        slew_start = self.telstate_l0.sync_time + 12.5 * self.dump_period
        slew_end = slew_start + 2 * self.dump_period
        self.telstate.add('cbf_target', target, ts=slew_start)
        telstate_cb = self.telstate.view('cb')
        telstate_cb.add('obs_activity', 'slew', ts=slew_start)
        telstate_cb.add('obs_activity', 'track', ts=slew_end)
        # Start the capture
        await self.make_request('capture-init', 'cb')
        # Wait until all the heaps have been delivered, timing out eventually.
        # This will take a while because it needs to allow the pipeline to run.
        await self.wait_for_heaps(n_times * self.n_substreams, 240)
        for stream in self.l0_streams.values():
            stream.send_heap(self.ig.get_end())
        await self.shutdown_servers(90)

    async def test_out_of_order(self):
        """A heap received from the past should be processed (if possible).

        Missing heaps are filled with data_lost.
        """
        # We want to prevent the pipeline fiddling with data in place.
        telstate_cb = self.telstate.view('cb')
        telstate_cb.add('obs_activity', 'slew', ts=1.0)
        n_times = 7
        # Each element is actually an (endpoint, heap) pair
        heaps = self.prepare_heaps(n_times)
        # Drop some heaps and delay others
        early_heaps = []
        late_heaps = []
        for heap, (t, s) in zip(heaps, itertools.product(range(n_times), range(self.n_substreams))):
            if t == 2 or (t == 4 and s == 2) or (t == 6 and s < self.n_substreams // 2):
                continue    # drop these completely
            elif s == 3:
                late_heaps.append(heap)
            else:
                early_heaps.append(heap)
        heaps = early_heaps + late_heaps
        heaps_expected = [0] * self.n_servers
        n_substreams_per_server = self.n_substreams // self.n_servers
        for endpoint, heap in heaps:
            self.l0_streams[endpoint].send_heap(heap)
            server_id = self.substream_endpoints.index(endpoint) // n_substreams_per_server
            heaps_expected[server_id] += 1
        # Run the capture
        await self.make_request('capture-init', 'cb')
        await asyncio.sleep(1)
        await self.make_request('shutdown', timeout=60)
        # Check that all heaps were accepted
        heaps_received = [int(x) for x in await self.get_sensor('input-heaps-total')]
        assert heaps_received == heaps_expected
        # Check that they were written to the right places and that timestamps are correct
        for t in range(n_times):
            for s in range(self.n_substreams):
                server_id = s // n_substreams_per_server
                s_rel = s % n_substreams_per_server
                buffers = self.servers[server_id].buffers
                channel_slice = np.s_[s_rel * self.n_channels_per_substream :
                                      (s_rel + 1) * self.n_channels_per_substream]
                channel0 = self.servers[server_id].parameters['channel_slice'].start
                channel0 += channel_slice.start
                flags = buffers['flags'][t, channel_slice]
                if t == 2 or (t == 4 and s == 2) or (t == 6 and s < self.n_substreams // 2):
                    np.testing.assert_equal(flags, 2 ** control.FLAG_NAMES.index('data_lost'))
                else:
                    np.testing.assert_equal(flags, 0)
                    # Check that the heap was written in the correct position
                    weights = buffers['weights'][t, channel_slice]
                    expected = np.arange(self.n_channels_per_substream, dtype=np.float32)
                    expected += t * self.n_channels + channel0 + 1
                    expected = expected[..., np.newaxis, np.newaxis]  # Add pol, baseline axes
                    expected = np.broadcast_to(expected, weights.shape)
                    np.testing.assert_equal(weights, expected)
            assert buffers['dump_indices'][t] == t
            assert buffers['times'][t] == self.first_dump_ts + self.dump_period * t

    async def test_weights_power_scale(self):
        """Test the application of need_weights_power_scale"""
        n_times = 2
        # This is the same as the default provided by prepare_heaps, but we
        # make it explicit so that we can use it to compute expected values.
        weights_channel = np.arange(1, n_times * self.n_channels + 1,
                                    dtype=np.float32).reshape(n_times, -1)
        vis = np.ones((n_times, self.n_channels, self.n_baselines), np.complex64)
        bls_ordering = self.telstate_l0['bls_ordering']
        # Adjust the autocorrelation power of some inputs
        vis[1, 100, bls_ordering.index(('m091h', 'm091h'))] = 4.0
        vis[1, 100, bls_ordering.index(('m092v', 'm092v'))] = 8.0
        heaps = self.prepare_heaps(n_times, vis=vis, weights_channel=weights_channel)

        # Compute expected weights
        ordering = calprocs.get_reordering(self.antennas, bls_ordering)[0]
        new_bls_ordering = np.array(bls_ordering)[ordering].reshape(4, -1, 2)
        expected = np.ones((n_times, self.n_channels, 4, new_bls_ordering.shape[1]), np.float32)
        expected *= weights_channel[:, :, np.newaxis, np.newaxis]
        for i in range(4):
            for j in range(new_bls_ordering.shape[1]):
                scale = 1.0
                for inp in new_bls_ordering[i, j]:
                    if inp == 'm091h':
                        scale /= 4.0
                    elif inp == 'm092v':
                        scale /= 8.0
                expected[1, 100, i, j] *= scale

        # Send the data and capture it
        for endpoint, heap in heaps:
            self.l0_streams[endpoint].send_heap(heap)
        await self.make_request('capture-init', 'cb')
        await asyncio.sleep(1)
        await self.make_request('shutdown', timeout=90)
        # Reassemble the buffered data from the individual servers
        actual = np.zeros_like(expected)
        for server in self.servers:
            channel_slice = server.parameters['channel_slice']
            actual[:, channel_slice, :] = server.buffers['weights'][:n_times]
        # First just compare the interesting part, so that test failures
        # are easier to diagnose.
        np.testing.assert_allclose(expected[1, 100], actual[1, 100], rtol=1e-4)
        np.testing.assert_allclose(expected, actual, rtol=1e-4)

    async def wait_for_sensor(self, sensor, value, timeout):
        """Wait `timeout` seconds for `sensor` to have `value`."""
        for i in range(timeout):
            await asyncio.sleep(1)
            rw = await self.get_sensor(sensor)
            if rw == value:
                break
        else:
            raise RuntimeError('Timed out waiting for %s to be %s' % (sensor, value))

    async def test_reset_solution_stores(self):
        """Test that the solution stores are reset between calls to capture_init"""
        # Force pipeline to reset the solution stores
        for serv in self.servers:
            serv.server.pipeline.parameters['reset_solution_stores'] = True
        n_times = 5
        start_time = self.first_dump_ts
        end_time = start_time + n_times * self.dump_period
        target = ('J1331+3030, radec delaycal bpcal gaincal, 13:31:08.29, +30:30:33.0, '
                  '(0 50e3 0.1823 1.4757 -0.4739 0.0336)')
        self.telstate.add('cbf_target', target, ts=0.01)
        heaps = self.prepare_heaps(n_times)
        for endpoint, heap in heaps:
            self.l0_streams[endpoint].send_heap(heap)
        await self.make_request('capture-init', 'cb')
        await self.wait_for_heaps(n_times * self.n_substreams, 60)
        for stream in self.l0_streams.values():
            stream.send_heap(self.ig.get_end())
        await self.make_request('capture-done')
        # The pipeline has finished running when 'reports-written' increments
        await self.wait_for_sensor('reports-written', [b'1'] * self.n_servers, 240)

        # Check that the solution stores have the solutions for the expected target
        ss = [serv.server.pipeline.solution_stores for serv in self.servers]
        for serv_store in ss:
            assert serv_store['B'].latest.target == 'J1331+3030'
            assert serv_store['K'].latest.target == 'J1331+3030'
            assert serv_store['G'].has_target('J1331+3030')
            assert serv_store['G_FLUX'].has_target('J1331+3030')

        # Refresh ItemGroup and send it to servers.
        self.init_item_group()

        # Set up a new capture block in telstate
        self.populate_telstate_cb(self.telstate, 'cb2')

        # A new target for a new CB
        # Only use gaincal tag so we can check the stores have been reset
        target = ('J1331+3030_2, radec gaincal, 13:31:08.29, +30:30:33.0, '
                  '(0 50e3 0.1823 1.4757 -0.4739 0.0336)')
        self.telstate.add('cbf_target', target, ts=0.02)
        heaps = self.prepare_heaps(n_times)
        for endpoint, heap in heaps:
            self.l0_streams[endpoint].send_heap(heap)
        await self.make_request('capture-init', 'cb2')
        await self.wait_for_heaps(n_times * self.n_substreams, 60)
        for stream in self.l0_streams.values():
            stream.send_heap(self.ig.get_end())
        # The pipeline has finished running when 'reports-written' increments
        await self.wait_for_sensor('reports-written', [b'2'] * self.n_servers, 240)

        # Check the solution stores only contain solutions from the new CB
        ss = [serv.server.pipeline.solution_stores for serv in self.servers]
        for serv_store in ss:
            assert serv_store['B'].latest is None
            assert serv_store['K'].latest is None
            assert serv_store['G'].has_target('J1331+3030_2')
            assert not serv_store['G'].has_target('J1331+3030')
            # There should now be no values in the G_FLUX store for this CB
            G_FLUX = serv_store['G_FLUX'].get_range(start_time, end_time)
            assert G_FLUX.values.size == 0

        # Check that 'cb' has 'measured_flux' in telstate for 'J1331+3030' only
        telstate_cb_cal = control.make_telstate_cb(self.telstate_cal, 'cb')
        assert 'J1331+3030' in telstate_cb_cal.get('measured_flux')
        assert 'J1331+3030_2' not in telstate_cb_cal.get('measured_flux')

        # Check that 'cb2' has no 'measured_flux' targets in telstate
        telstate_cb2_cal = control.make_telstate_cb(self.telstate_cal, 'cb2')
        assert telstate_cb2_cal.get('measured_flux') == {}

        await self.shutdown_servers(180)

    async def test_run_pipeline_exception(self):
        rs = np.random.RandomState(seed=1)
        with mock.patch.object(control.Pipeline, 'run_pipeline', side_effect=ZeroDivisionError):
            await self.assert_sensor_value('pipeline-exceptions', 0)
            for endpoint, heap in self.prepare_heaps(n_times=5, rs=rs):
                self.l0_streams[endpoint].send_heap(heap)
            await self.make_request('capture-init', 'cb')
            await asyncio.sleep(1)
            await self.assert_sensor_value('capture-block-state', b'{"cb": "CAPTURING"}')
            for stream in self.l0_streams.values():
                stream.send_heap(self.ig.get_end())
            await self.shutdown_servers(60)
            await self.assert_sensor_value('pipeline-exceptions', 1)
            await self.assert_sensor_value('capture-block-state', b'{}')

    async def test_flush_pipeline_exception(self):
        rs = np.random.RandomState(seed=1)
        with mock.patch.object(control.Pipeline, 'flush_pipeline', side_effect=ZeroDivisionError):
            await self.assert_sensor_value('pipeline-exceptions', 0)

            # Target must be altered to only include the `pointingcal` tag. This is done to avoid
            # triggering the`check_applied_gain_sensor` check. This check will fail/cause an
            # exception as it requires a more comprehensive set of keys to construct the `sdp.vis`
            # stream, which are not included in the test telstate here
            target = ('J1331+3030_2, radec pointingcal, 13:31:08.29, +30:30:33.0, '
                      '(0 50e3 0.1823 1.4757 -0.4739 0.0336)')
            self.telstate.add('cbf_target', target, ts=0.003)
            for endpoint, heap in self.prepare_heaps(n_times=5, rs=rs):
                self.l0_streams[endpoint].send_heap(heap)
            telstate_cb = self.telstate.view('cb')
            telstate_cb.add('obs_activity', 'await_pipeline',
                            ts=self.first_dump_ts + 3.5 * self.dump_period)
            await self.make_request('capture-init', 'cb')
            await asyncio.sleep(1)
            await self.assert_sensor_value('capture-block-state', b'{"cb": "CAPTURING"}')
            for stream in self.l0_streams.values():
                stream.send_heap(self.ig.get_end())
            await self.shutdown_servers(60)
            await self.assert_sensor_value('pipeline-exceptions', 1)
            await self.assert_sensor_value('capture-block-state', b'{}')
