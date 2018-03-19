"""Tests for control module"""

from __future__ import print_function, division, absolute_import
import unittest
from multiprocessing import Process
import multiprocessing
import multiprocessing.dummy
import tempfile
import shutil
import functools
import os
import itertools

import numpy as np
from nose.tools import (
    assert_equal, assert_is_instance, assert_in, assert_not_in, assert_false, assert_true,
    assert_regexp_matches, assert_almost_equal,
    nottest)
import mock

import tornado.gen
from tornado.platform.asyncio import AsyncIOMainLoop
import trollius
from trollius import From, Return

import spead2
import katcp
import katsdptelstate
from katsdptelstate.endpoint import Endpoint
from katsdpservices.asyncio import to_tornado_future
import katpoint
from katdal.h5datav3 import FLAG_NAMES

from katsdpcal import control, pipelineprocs, param_dir, rfi_dir


def test_shared_empty():
    def sender(a):
        a[:] = np.outer(np.arange(5), np.arange(3))

    a = control.shared_empty((5, 3), np.int32)
    a.fill(0)
    assert_equal((5, 3), a.shape)
    assert_equal(np.int32, a.dtype)
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
        super(PingTask, self).__init__(task_class, master_queue, 'PingTask')
        self.slave_queue = slave_queue

    def get_sensors(self):
        return [
            katcp.Sensor.integer('last-value', 'last number received on the queue'),
            katcp.Sensor.integer('error', 'sensor that is set to error state')
        ]

    def run(self):
        self.sensors['error'].set_value(0, status=katcp.Sensor.ERROR, timestamp=123456789.0)
        while True:
            number = self.slave_queue.get()
            if number < 0:
                break
            self.sensors['last-value'].set_value(number)
        self.master_queue.put(control.StopEvent())


class BaseTestTask(object):
    """Tests for :class:`katsdpcal.control.Task`.

    This is a base class, which is subclassed for each process class.
    """
    def setup(self):
        self.master_queue = self.module.Queue()
        self.slave_queue = self.module.Queue()

    def _check_reading(self, event, name, value, status=katcp.Sensor.NOMINAL, timestamp=None):
        assert_is_instance(event, control.SensorReadingEvent)
        assert_equal(name, event.name)
        assert_equal(value, event.reading.value)
        assert_equal(status, event.reading.status)
        if timestamp is not None:
            assert_equal(timestamp, event.reading.timestamp)

    def test(self):
        task = PingTask(self.module.Process, self.master_queue, self.slave_queue)
        assert_equal(False, task.daemon)   # Test the wrapper property
        task.daemon = True       # Ensure it gets killed if the test fails
        assert_equal('PingTask', task.name)
        task.start()

        event = self.master_queue.get()
        self._check_reading(event, 'error', 0, katcp.Sensor.ERROR, 123456789.0)
        assert_true(task.is_alive())

        self.slave_queue.put(3)
        event = self.master_queue.get()
        self._check_reading(event, 'last-value', 3)

        self.slave_queue.put(-1)   # Stops the slave
        event = self.master_queue.get()
        assert_is_instance(event, control.StopEvent)

        task.join()
        assert_false(task.is_alive())


class TestTaskMultiprocessing(BaseTestTask):
    module = multiprocessing

    def test_terminate(self):
        task = PingTask(self.module.Process, self.master_queue, self.slave_queue)
        task.start()
        task.terminate()
        task.join()


class TestTaskDummy(BaseTestTask):
    module = multiprocessing.dummy


@nottest
def async_test(func):
    """Decorator to run a test inside the Tornado event loop"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return tornado.ioloop.IOLoop.current().run_sync(lambda: func(*args, **kwargs))
    return wrapper


class MockRecvData(object):
    """Holds heaps to feed MockRecvStream.

    It has an ordered collection of heaps, each associated with a
    UDP endpoint. When requested, it provides the first one that matches
    a given set of endpoints.
    """
    def __init__(self):
        self._heaps = []
        self._waiters = []    # Each entry is (endpoints, future) pair
        self._thread_pool = spead2.ThreadPool()

    def send_heap(self, endpoint, heap):
        if heap is not None:
            # Convert from send heap to receive heap
            encoder = spead2.send.BytesStream(self._thread_pool)
            encoder.send_heap(heap)
            raw = encoder.getvalue()
            decoder = spead2.recv.Stream(self._thread_pool)
            decoder.stop_on_stop_item = False
            decoder.add_buffer_reader(raw)
            heap = decoder.get()
        for i, waiter in enumerate(self._waiters):
            if endpoint in waiter[0] and not waiter[1].done():
                waiter[1].set_result(heap)
                del self._waiters[i]
                break
        else:
            self._heaps.append((endpoint, heap))

    @trollius.coroutine
    def get(self, endpoints):
        for i in range(len(self._heaps)):
            if self._heaps[i][0] in endpoints:
                heap = self._heaps[i][1]
                del self._heaps[i]
                raise Return(heap)
        # Not found, so wait for it
        future = trollius.Future()
        self._waiters.append((endpoints, future))
        raise Return((yield From(future)))


class MockRecvStream(mock.MagicMock):
    """Mock replacement for :class:`spead2.recv.trollius.Stream`.

    It has a queue of heaps that it yields to the caller. If the queue is
    empty, it blocks until a new item is added or :meth:`stop` is called.
    """
    def __init__(self, data, *args, **kwargs):
        super(MockRecvStream, self).__init__(*args, **kwargs)
        self._data = data
        self._stop_received = False
        self._endpoints = set()

    # Make child mocks use the basic MagicMock, not this class
    def _get_child_mock(self, **kwargs):
        return mock.MagicMock(**kwargs)

    @trollius.coroutine
    def get(self):
        if self._stop_received:
            raise spead2.Stopped()
        heap = yield From(self._data.get(self._endpoints))
        if heap is None:
            # Special value added by stop
            self._stop_received = True
            raise spead2.Stopped()
        raise Return(heap)

    def add_udp_reader(self, port, bind_hostname, buffer_size=None):
        self._endpoints.add(Endpoint(bind_hostname, port))

    def stop(self):
        assert self._endpoints, "can't stop without an endpoint"
        self._data.send_heap(next(iter(self._endpoints)), None)


class ServerData(object):
    """Test data associated with a single simulated cal server"""
    def make_parameters(self, telstate_l0):
        param_file = os.path.join(param_dir, 'pipeline_parameters_meerkat_L_4k.txt')
        rfi_file = os.path.join(rfi_dir, 'rfi_mask.pickle')
        parameters = pipelineprocs.parameters_from_file(param_file)
        pipelineprocs.finalise_parameters(parameters, telstate_l0,
                                          self.testcase.n_servers, self.server_id, rfi_file)
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
        self.server = control.create_server(
            False, 'localhost', 0, buffers,
            'sdp_l0test', testcase.l0_endpoints, None,
            'sdp_l1_flags_test', testcase.flags_endpoints, None, 64.0,
            testcase.telstate_cal, self.parameters, self.report_path, self.log_path, None)
        self.server.start()
        testcase.addCleanup(testcase.ioloop.run_sync, self.stop_server)

        bind_address = self.server.bind_address
        self.client = katcp.AsyncClient(bind_address[0], bind_address[1], timeout=15)
        self.client.set_ioloop(testcase.ioloop)
        self.client.start()
        testcase.addCleanup(self.client.stop)
        testcase.addCleanup(self.client.disconnect)
        testcase.ioloop.run_sync(self.client.until_protocol)

    @tornado.gen.coroutine
    def stop_server(self):
        yield to_tornado_future(self.server.shutdown())
        self.server.stop()


class TestCalDeviceServer(unittest.TestCase):
    """Tests for :class:`katsdpcal.control.CalDeviceServer.

    This does not test the quality of the solutions that are produced, merely
    that they are produced and calibration reports written.
    """

    def patch(self, *args, **kwargs):
        patcher = mock.patch(*args, **kwargs)
        mock_obj = patcher.start()
        self.addCleanup(patcher.stop)
        return mock_obj

    def populate_telstate(self, telstate_l0):
        telstate = telstate_l0.root()
        bls_ordering = []
        target = ('3C286, radec bfcal single_accumulation, 13:31:08.29, +30:30:33.0, '
                  '(800.0 43200.0 0.956 0.584 -0.1644)')
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
        telstate.add('subarray_product_id', 'c856M4k', immutable=True)
        telstate.add('sub_band', 'l', immutable=True)
        telstate_l0.add('int_time', 4.0, immutable=True)
        telstate_l0.add('bls_ordering', bls_ordering, immutable=True)
        telstate_l0.add('n_bls', len(bls_ordering), immutable=True)
        telstate_l0.add('bandwidth', 856000000.0, immutable=True)
        telstate_l0.add('center_freq', 1712000000.0, immutable=True)
        telstate_l0.add('n_chans', self.n_channels, immutable=True)
        telstate_l0.add('n_chans_per_substream', self.n_channels_per_substream, immutable=True)
        telstate_l0.add('sync_time', 1400000000.0, immutable=True)
        telstate_cb_l0 = telstate.view(telstate.SEPARATOR.join(('cb', 'sdp_l0test')))
        telstate_cb_l0.add('first_timestamp', 100.0, immutable=True)
        for antenna in self.antennas:
            telstate.add('{}_activity'.format(antenna), 'track', ts=0)
            telstate.add('{}_target'.format(antenna), target, ts=0)
            # The position is irrelevant for now, so just give all the
            # antennas the same position.
            telstate.add(
                '{}_observer'.format(antenna),
                '{}, -30:42:47.4, 21:26:38.0, 1035.0, 13.5, -351.163669759 384.481835294, '
                '-0:05:44.7 0 0:00:22.6 -0:09:04.2 0:00:11.9 -0:00:12.8 -0:04:03.5 0 0 '
                '-0:01:33.0 0:01:45.6 0 0 0 0 0 -0:00:03.6 -0:00:17.5, 1.22'.format(antenna))

    def add_items(self, ig):
        channels = self.telstate.sdp_l0test_n_chans_per_substream
        baselines = len(self.telstate.sdp_l0test_bls_ordering)
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

    def _get_input_stream(self, *args, **kwargs):
        """Mock implementation of :class:`spead2.recv.Stream` that returns a MockRecvStream."""
        return MockRecvStream(self.input_data, spec=self._dummy_stream)

    def _get_output_stream(self, thread_pool, hostname, port, config,
                           *args, **kwargs):
        """Mock implementation of UdpStream that returns a ByteStream instead.

        It stores it in self.output_streams, keyed by hostname and port.
        """
        key = Endpoint(hostname, port)
        assert_not_in(key, self.output_streams)
        self.output_streams[key] = spead2.send.BytesStream(thread_pool)
        return self.output_streams[key]

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

        self.ioloop = AsyncIOMainLoop()
        self.ioloop.install()
        self.addCleanup(tornado.ioloop.IOLoop.clear_instance)

        self.l0_endpoints = [Endpoint('239.102.255.{}'.format(i), 7148)
                             for i in range(self.n_endpoints)]
        substreams_per_endpoint = self.n_substreams // self.n_endpoints
        self.substream_endpoints = [self.l0_endpoints[i // substreams_per_endpoint]
                                    for i in range(self.n_substreams)]
        self.flags_endpoints = [Endpoint('239.102.254.{}'.format(i), 7148)
                                for i in range(self.n_servers)]

        self.ig = spead2.send.ItemGroup()
        self.add_items(self.ig)
        self.input_data = MockRecvData()
        for endpoint in self.l0_endpoints:
            self.input_data.send_heap(endpoint, self.ig.get_heap(descriptors='all'))
        # Create a real stream just so that we can use it as a spec later
        self._dummy_stream = spead2.recv.trollius.Stream(spead2.ThreadPool())
        self._dummy_stream.stop()
        self.patch('spead2.recv.trollius.Stream',
                   side_effect=self._get_input_stream)
        self.output_streams = {}
        self.patch('spead2.send.UdpStream', side_effect=self._get_output_stream)

        # Trying to run two dask distributed clients in the same process doesn't
        # work so well, so don't try
        self.patch('dask.distributed.LocalCluster')
        self.patch('dask.distributed.Client')

        self.servers = [ServerData(self, i) for i in range(self.n_servers)]

    @tornado.gen.coroutine
    def make_request(self, name, *args, **kwargs):
        """Issue a request to all the servers, and check that the result is ok.

        Parameters
        ----------
        name : str
            Request name
        args : list
            Arguments to the request
        kwargs : dict
            Arguments to ``future_request``

        Returns
        -------
        informs : list of lists
            Informs returned with the reply from each server
        """
        futures = [server.client.future_request(katcp.Message.request(name, *args), **kwargs)
                   for server in self.servers]
        results = yield futures
        all_informs = []
        for reply, informs in results:
            assert_true(reply.reply_ok(), str(reply))
            all_informs.append(informs)
        raise tornado.gen.Return(all_informs)

    @tornado.gen.coroutine
    def get_sensor(self, name):
        """Retrieves a sensor value and checks that the value is well-defined.

        Returns
        -------
        values : list of str
            The sensor values (per-server), in the string form it is sent in the protocol
        """
        values = []
        informs_list = yield self.make_request('sensor-value', name)
        for informs in informs_list:
            assert_equal(1, len(informs))
            assert_in(informs[0].arguments[3], ('nominal', 'warn', 'error'))
            values.append(informs[0].arguments[4])
        raise tornado.gen.Return(values)

    @tornado.gen.coroutine
    def assert_sensor_value(self, name, expected):
        """Retrieves a sensor value and compares its value.

        The returned string is automatically cast to the type of `expected`.
        """
        values = yield self.get_sensor(name)
        for i, value in enumerate(values):
            value = type(expected)(value)
            assert_equal(expected, value,
                         "Wrong value for {} ({!r} != {!r})".format(name, expected, value))

    @tornado.gen.coroutine
    def assert_request_fails(self, msg_re, name, *args):
        """Assert that a request fails, and test the error message against
        a regular expression."""
        for server in self.servers:
            reply, informs = yield server.client.future_request(
                katcp.Message.request(name, *args))
            assert_equal(2, len(reply.arguments))
            assert_equal('fail', reply.arguments[0])
            assert_regexp_matches(reply.arguments[1], msg_re)

    @async_test
    @tornado.gen.coroutine
    def test_empty_capture(self):
        """Terminating a capture with no data must succeed and not write a report.

        It must also correctly remove the capture block from capture-block-state.
        """
        yield self.make_request('capture-init', 'cb')
        yield self.assert_sensor_value('capture-block-state', '{"cb": "CAPTURING"}')
        for endpoint in self.l0_endpoints:
            self.input_data.send_heap(endpoint, self.ig.get_end())
        yield self.make_request('capture-done')
        yield self.make_request('shutdown')
        for server in self.servers:
            assert_equal([], os.listdir(server.report_path))
        yield self.assert_sensor_value('reports-written', 0)
        yield self.assert_sensor_value('capture-block-state', '{}')

    @async_test
    @tornado.gen.coroutine
    def test_init_when_capturing(self):
        """capture-init fails when already capturing"""
        for endpoint in self.l0_endpoints:
            self.input_data.send_heap(endpoint, self.ig.get_end())
        yield self.make_request('capture-init', 'cb')
        yield self.assert_request_fails(r'capture already in progress', 'capture-init', 'cb')

    @async_test
    @tornado.gen.coroutine
    def test_done_when_not_capturing(self):
        """capture-done fails when not capturing"""
        yield self.assert_request_fails(r'no capture in progress', 'capture-done')
        yield self.make_request('capture-init', 'cb')
        for endpoint in self.l0_endpoints:
            self.input_data.send_heap(endpoint, self.ig.get_end())
        yield self.make_request('capture-done')
        yield self.assert_request_fails(r'no capture in progress', 'capture-done')

    @classmethod
    def normalise_phase(cls, value, ref):
        """Multiply `value` by an amount that sets `ref` to zero phase."""
        ref_phase = ref / np.abs(ref)
        return value * ref_phase.conj()

    @tornado.gen.coroutine
    def shutdown_servers(self, timeout):
        inform_lists = yield self.make_request('shutdown', timeout=timeout)
        for informs in inform_lists:
            progress = [inform.arguments[0] for inform in informs]
            assert_equal(['Accumulator stopped',
                          'Pipeline stopped',
                          'Sender stopped',
                          'ReportWriter stopped'], progress)

    @async_test
    @tornado.gen.coroutine
    def test_capture(self, expected_g=1):
        """Tests the capture with some data, and checks that solutions are
        computed and a report written.
        """
        first_ts = ts = 100.0
        n_times = 25
        corrupt_times = (4, 17)
        rs = np.random.RandomState(seed=1)

        bandwidth = self.telstate.sdp_l0test_bandwidth
        target = katpoint.Target(self.telstate.m090_target)
        # The + bandwidth is to convert to L band
        freqs = np.arange(self.n_channels) / self.n_channels * bandwidth + bandwidth
        flux_density = target.flux_density(freqs / 1e6)[:, np.newaxis]
        freqs = freqs[:, np.newaxis]
        for antenna in self.antennas:
            self.telstate.add('{0}_dig_l_band_noise_diode'.format(antenna),
                              0, ts)
        bls_ordering = self.telstate.sdp_l0test_bls_ordering
        ant1 = [self.antennas.index(b[0][:-1]) for b in bls_ordering]
        ant2 = [self.antennas.index(b[1][:-1]) for b in bls_ordering]
        pol1 = ['vh'.index(b[0][-1]) for b in bls_ordering]
        pol2 = ['vh'.index(b[1][-1]) for b in bls_ordering]
        K = rs.uniform(-50e-12, 50e-12, (2, self.n_antennas))
        G = rs.uniform(2.0, 4.0, (2, self.n_antennas)) \
            + 1j * rs.uniform(-0.1, 0.1, (2, self.n_antennas))

        vis = flux_density * np.exp(2j * np.pi * (K[pol1, ant1] - K[pol2, ant2]) * freqs) \
            * (G[pol1, ant1] * G[pol2, ant2].conj())
        corrupted_vis = vis + 1e9j
        flags = np.zeros(vis.shape, np.uint8)
        # Set flag on one channel per baseline, to test the baseline permutation.
        for i in range(flags.shape[1]):
            flags[i, i] = 1 << FLAG_NAMES.index('ingest_rfi')
        weights = rs.uniform(64, 255, vis.shape).astype(np.uint8)
        weights_channel = rs.uniform(1.0, 4.0, (self.n_channels,)).astype(np.float32)

        channel_slices = [np.s_[i * self.n_channels_per_substream
                                : (i+1) * self.n_channels_per_substream]
                          for i in range(self.n_substreams)]
        for i in range(n_times):
            # Corrupt some times, to check that the RFI flagging is working
            dump_heaps = []
            for endpoint, s in zip(self.substream_endpoints, channel_slices):
                self.ig['correlator_data'].value = \
                    corrupted_vis[s] if i in corrupt_times else vis[s]
                self.ig['flags'].value = flags[s]
                self.ig['weights'].value = weights[s]
                self.ig['weights_channel'].value = weights_channel[s]
                self.ig['timestamp'].value = ts
                self.ig['dump_index'].value = i
                self.ig['frequency'].value = np.uint32(s.start)
                dump_heaps.append((endpoint, self.ig.get_heap()))
            rs.shuffle(dump_heaps)
            for endpoint, heap in dump_heaps:
                self.input_data.send_heap(endpoint, heap)
            ts += self.telstate.sdp_l0test_int_time
        yield self.make_request('capture-init', 'cb')
        yield tornado.gen.sleep(1)
        yield self.assert_sensor_value('accumulator-capture-active', 1)
        yield self.assert_sensor_value('capture-block-state', '{"cb": "CAPTURING"}')
        for endpoint in self.l0_endpoints:
            self.input_data.send_heap(endpoint, self.ig.get_end())
        yield self.shutdown_servers(180)
        yield self.assert_sensor_value('accumulator-capture-active', 0)
        yield self.assert_sensor_value('input-heaps-total',
                                       n_times * self.n_substreams // self.n_servers)
        yield self.assert_sensor_value('accumulator-batches', 1)
        yield self.assert_sensor_value('accumulator-observations', 1)
        yield self.assert_sensor_value('pipeline-last-slots', n_times)
        yield self.assert_sensor_value('reports-written', 1)
        # Check that the slot accounting all balances
        yield self.assert_sensor_value('slots', 60)
        yield self.assert_sensor_value('accumulator-slots', 0)
        yield self.assert_sensor_value('pipeline-slots', 0)
        yield self.assert_sensor_value('free-slots', 60)
        yield self.assert_sensor_value('capture-block-state', '{}')

        report_last_path = yield self.get_sensor('report-last-path')
        for server in self.servers:
            reports = os.listdir(server.report_path)
            assert_equal(1, len(reports))
            report = os.path.join(server.report_path, reports[0])
            assert_true(os.path.isfile(os.path.join(report, 'calreport.html')))
            assert_true(os.path.samefile(report, report_last_path[server.server_id]))

        telstate_cb = control.make_telstate_cb(self.telstate_cal, 'cb')
        cal_product_B_parts = telstate_cb['product_B_parts']
        assert_equal(self.n_servers, cal_product_B_parts)
        ret_B = []
        for i in range(self.n_servers):
            cal_product_Bn = telstate_cb.get_range('product_B{}'.format(i), st=0)
            assert_equal(1, len(cal_product_Bn))
            ret_Bn, ret_Bn_ts = cal_product_Bn[0]
            assert_equal(np.complex64, ret_Bn.dtype)
            assert_equal((self.n_channels // self.n_servers, 2, self.n_antennas), ret_Bn.shape)
            ret_B.append(ret_Bn)
        assert_not_in('product_B{}'.format(self.n_servers), telstate_cb)
        ret_B = np.concatenate(ret_B)

        cal_product_G = telstate_cb.get_range('product_G', st=0)
        assert_equal(expected_g, len(cal_product_G))
        ret_G, ret_G_ts = cal_product_G[0]
        assert_equal(np.complex64, ret_G.dtype)
        assert_equal(0, np.count_nonzero(np.isnan(ret_G)))
        ret_BG = ret_B * ret_G[np.newaxis, :, :]
        BG = np.broadcast_to(G[np.newaxis, :, :], ret_BG.shape)
        # TODO: enable when fixed.
        # This test won't work yet:
        # - cal puts NaNs in B in the channels for which it applies the static
        #   RFI mask, instead of interpolating
        # np.testing.assert_allclose(np.abs(BG), np.abs(ret_BG), rtol=1e-3)
        # np.testing.assert_allclose(self.normalise_phase(BG, BG[:, :, [0]]),
        #                            self.normalise_phase(ret_BG, ret_BG[:, :, [0]]),
        #                            rtol=1e-3)

        cal_product_K = telstate_cb.get_range('product_K', st=0)
        assert_equal(1, len(cal_product_K))
        ret_K, ret_K_ts = cal_product_K[0]
        assert_equal(np.float32, ret_K.dtype)
        np.testing.assert_allclose(K - K[:, [0]], ret_K - ret_K[:, [0]], rtol=1e-3)

        # Check that flags were transmitted
        assert_equal(set(self.output_streams.keys()), set(self.flags_endpoints))
        for i, endpoint in enumerate(self.flags_endpoints):
            decoder = spead2.recv.Stream(spead2.ThreadPool())
            decoder.stop_on_stop_item = False
            decoder.add_buffer_reader(self.output_streams[endpoint].getvalue())
            heaps = list(decoder)
            assert_equal(n_times + 2, len(heaps))   # 2 extra for start and end heaps
            for j, heap in enumerate(heaps[1:-1]):
                items = spead2.ItemGroup()
                items.update(heap)
                ts = items['timestamp'].value
                assert_almost_equal(first_ts + j * self.telstate.sdp_l0test_int_time, ts)
                idx = items['dump_index'].value
                assert_equal(j, idx)
                assert_equal(i * self.n_channels // self.n_servers, items['frequency'].value)
                out_flags = items['flags'].value
                # Mask out the ones that get changed by cal
                mask = (1 << FLAG_NAMES.index('static')) | (1 << FLAG_NAMES.index('cal_rfi'))
                expected = flags[self.servers[i].parameters['channel_slice']]
                np.testing.assert_array_equal(out_flags & ~mask, expected)

    def test_capture_separate_tags(self):
        # Change the target to one with different tags
        target = ('3C286, radec delaycal gaincal bpcal kcrosscal single_accumulation, '
                  '13:31:08.29, +30:30:33.0, (800.0 43200.0 0.956 0.584 -0.1644)')
        for antenna in self.antennas:
            self.telstate.add('{}_target'.format(antenna), target, ts=0.001)
        self.test_capture(expected_g=2)

    def prepare_heaps(self, rs, n_times):
        """Produce a list of heaps with arbitrary data.

        Parameters
        ----------
        rs : :class:`numpy.random.RandomState`
            Random generator used to shuffle the heaps of one dump. If
            ``None``, they are not shuffled.
        n_times : int
            Number of dumps
        """
        vis = np.ones((self.n_channels, self.n_baselines), np.complex64)
        weights = np.ones(vis.shape, np.uint8)
        flags = np.zeros(vis.shape, np.uint8)
        ts = 100.0
        channel_slices = [np.s_[i * self.n_channels_per_substream
                                : (i+1) * self.n_channels_per_substream]
                          for i in range(self.n_substreams)]
        heaps = []
        for i in range(n_times):
            dump_heaps = []
            # Create a value with a pattern, to help in tests that check that
            # heaps are written to the correct slots.
            weights_channel = np.arange(self.n_channels, dtype=np.float32) + i * self.n_channels + 1
            for endpoint, s in zip(self.substream_endpoints, channel_slices):
                self.ig['correlator_data'].value = vis[s]
                self.ig['flags'].value = flags[s]
                self.ig['weights'].value = weights[s]
                self.ig['weights_channel'].value = weights_channel[s]
                self.ig['timestamp'].value = ts
                self.ig['dump_index'].value = i
                self.ig['frequency'].value = np.uint32(s.start)
                dump_heaps.append((endpoint, self.ig.get_heap()))
            if rs is not None:
                rs.shuffle(dump_heaps)
            heaps.extend(dump_heaps)
            ts += self.telstate.sdp_l0test_int_time
        return heaps

    @async_test
    @tornado.gen.coroutine
    def test_buffer_wrap(self):
        """Test capture with more heaps than buffer slots, to check that it handles
        wrapping around the end of the buffer.
        """
        rs = np.random.RandomState(seed=1)
        n_times = 130
        for endpoint, heap in self.prepare_heaps(rs, n_times):
            self.input_data.send_heap(endpoint, heap)
        # Add a target change at an uneven time, so that the batches won't
        # neatly align with the buffer end. We also have to fake a slew to make
        # it work, since the batcher assumes that target cannot change without
        # an activity change (TODO: it probably shouldn't assume this).
        target = 'dummy, radec target, 13:30:00.00, +30:30:00.0'
        slew_start = self.telstate.sdp_l0test_sync_time + 12.5 * self.telstate.sdp_l0test_int_time
        slew_end = slew_start + 2 * self.telstate.sdp_l0test_int_time
        for antenna in self.antennas:
            self.telstate.add('{}_target'.format(antenna), target, ts=slew_start)
            self.telstate.add('{}_activity'.format(antenna), 'slew', ts=slew_start)
            self.telstate.add('{}_activity'.format(antenna), 'track', ts=slew_end)
        # Start the capture
        yield self.make_request('capture-init', 'cb')
        # Wait until all the heaps have been delivered, timing out eventually.
        # This will take a while because it needs to allow the pipeline to run.
        for i in range(240):
            yield tornado.gen.sleep(1)
            heaps = (yield self.get_sensor('input-heaps-total'))
            total_heaps = sum(int(x) for x in heaps)
            if total_heaps == n_times * self.n_substreams:
                print('all heaps received')
                break
            print('waiting {} ({}/{} received)'.format(i, total_heaps, n_times * self.n_substreams))
        else:
            raise RuntimeError('Timed out waiting for the heaps to be received')
        for endpoint in self.l0_endpoints:
            self.input_data.send_heap(endpoint, self.ig.get_end())
        yield self.shutdown_servers(60)

    @async_test
    @tornado.gen.coroutine
    def test_out_of_order(self):
        """A heap received from the past should be processed (if possible).

        Missing heaps are filled with data_lost.
        """
        # We want to prevent the pipeline fiddling with data in place.
        for antenna in self.antennas:
            self.telstate.add('{}_activity'.format(antenna), 'slew', ts=1.0)

        n_times = 7
        # Each element is actually an (endpoint, heap) pair
        heaps = self.prepare_heaps(None, n_times)
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
            self.input_data.send_heap(endpoint, heap)
            server_id = self.substream_endpoints.index(endpoint) // n_substreams_per_server
            heaps_expected[server_id] += 1
        # Run the capture
        yield self.make_request('capture-init', 'cb')
        yield tornado.gen.sleep(1)
        yield self.make_request('shutdown', timeout=60)
        # Check that all heaps were accepted
        heaps_received = [int(x) for x in (yield self.get_sensor('input-heaps-total'))]
        assert_equal(heaps_expected, heaps_received)
        # Check that they were written to the right places and that timestamps are correct
        for t in range(n_times):
            for s in range(self.n_substreams):
                server_id = s // n_substreams_per_server
                s_rel = s % n_substreams_per_server
                buffers = self.servers[server_id].buffers
                channel_slice = np.s_[s_rel * self.n_channels_per_substream
                                      : (s_rel+1) * self.n_channels_per_substream]
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
            assert_equal(buffers['dump_indices'][t], t)
            assert_equal(buffers['times'][t], 1400000100.0 + 4 * t)

    @async_test
    @tornado.gen.coroutine
    def test_pipeline_exception(self):
        with mock.patch.object(control.Pipeline, 'run_pipeline', side_effect=ZeroDivisionError):
            yield self.assert_sensor_value('pipeline-exceptions', 0)
            for endpoint, heap in self.prepare_heaps(np.random.RandomState(seed=1), 5):
                self.input_data.send_heap(endpoint, heap)
            yield self.make_request('capture-init', 'cb')
            yield tornado.gen.sleep(1)
            yield self.assert_sensor_value('capture-block-state', '{"cb": "CAPTURING"}')
            for endpoint in self.l0_endpoints:
                self.input_data.send_heap(endpoint, self.ig.get_end())
            yield self.shutdown_servers(60)
            yield self.assert_sensor_value('pipeline-exceptions', 1)
            yield self.assert_sensor_value('capture-block-state', '{}')
