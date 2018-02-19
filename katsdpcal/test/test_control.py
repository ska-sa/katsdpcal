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
import glob
import copy
from collections import deque

import numpy as np
from nose.tools import (
    assert_equal, assert_is_instance, assert_in, assert_false, assert_true,
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


class MockRecvStream(mock.MagicMock):
    """Mock replacement for :class:`spead2.recv.trollius.Stream`.

    It has a queue of heaps that it yields to the caller. If the queue is
    empty, it blocks until a new item is added or :meth:`stop` is called.
    """
    def __init__(self, *args, **kwargs):
        super(MockRecvStream, self).__init__(*args, **kwargs)
        self._queue = trollius.Queue()
        self._stop_received = False
        self._thread_pool = spead2.ThreadPool()

    # Make child mocks use the basic MagicMock, not this class
    def _get_child_mock(self, **kwargs):
        return mock.MagicMock(**kwargs)

    @trollius.coroutine
    def get(self):
        if self._stop_received:
            raise spead2.Stopped()
        heap = yield From(self._queue.get())
        if heap is None:
            # Special value added by stop
            self._stop_received = True
            raise spead2.Stopped()
        raise Return(heap)

    def send_heap(self, heap):
        # Convert from send heap to receive heap
        encoder = spead2.send.BytesStream(self._thread_pool)
        encoder.send_heap(heap)
        raw = encoder.getvalue()
        decoder = spead2.recv.Stream(self._thread_pool)
        decoder.stop_on_stop_item = False
        decoder.add_buffer_reader(raw)
        heap = decoder.get()
        self._queue.put_nowait(heap)

    def stop(self):
        self._queue.put_nowait(None)


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

    def populate_telstate(self, telstate):
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
        telstate.clear()     # Prevent state leaks
        telstate.add('subarray_product_id', 'c856M4k', immutable=True)
        telstate.add('sdp_l0test_int_time', 4.0, immutable=True)
        telstate.add('sdp_l0test_bls_ordering', bls_ordering, immutable=True)
        telstate.add('sdp_l0test_n_bls', len(bls_ordering), immutable=True)
        telstate.add('sdp_l0test_bandwidth', 856000000.0, immutable=True)
        telstate.add('sdp_l0test_center_freq', 1712000000.0, immutable=True)
        telstate.add('sdp_l0test_n_chans', self.n_channels, immutable=True)
        telstate.add('sdp_l0test_n_chans_per_substream', self.n_channels_per_substream, immutable=True)
        telstate.add('sdp_l0test_sync_time', 1400000000.0, immutable=True)
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
        param_file = os.path.join(param_dir, 'pipeline_parameters_meerkat_ar1_4k.txt')
        rfi_file = os.path.join(rfi_dir, 'rfi_mask.pickle')
        pipelineprocs.ts_from_file(telstate, param_file, rfi_file)
        pipelineprocs.setup_ts(telstate, self.antennas)

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

    @tornado.gen.coroutine
    def stop_server(self):
        yield to_tornado_future(self.server.shutdown())
        self.server.stop()

    def setUp(self):
        self.n_channels = 4096
        self.n_substreams = 4
        self.n_endpoints = 2
        assert self.n_channels % self.n_substreams == 0
        self.n_channels_per_substream = self.n_channels // self.n_substreams
        self.antennas = ["m090", "m091", "m092", "m093"]
        self.n_antennas = len(self.antennas)
        self.n_baselines = self.n_antennas * (self.n_antennas + 1) * 2

        self.telstate = katsdptelstate.TelescopeState()
        self.populate_telstate(self.telstate)
        self.ioloop = AsyncIOMainLoop()
        self.ioloop.install()
        self.addCleanup(tornado.ioloop.IOLoop.clear_instance)

        self.ig = spead2.send.ItemGroup()
        self.add_items(self.ig)
        # Create a real stream just so that we can use it as a spec
        dummy_stream = spead2.recv.trollius.Stream(spead2.ThreadPool())
        dummy_stream.stop()
        self.stream = MockRecvStream(spec=dummy_stream)
        self.stream.send_heap(self.ig.get_heap())
        self.patch('spead2.recv.trollius.Stream', return_value=self.stream)
        self.udp_stream = spead2.send.BytesStream(spead2.ThreadPool())
        self.patch('spead2.send.UdpStream', return_value=self.udp_stream)

        self.report_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.report_path)
        self.log_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.log_path)

        # Time, channels, pols, baselines
        buffer_shape = (40, self.n_channels, 4, self.n_baselines // 4)
        buffers = control.create_buffer_arrays(buffer_shape, False)
        self.server = control.create_server(
            False, 'localhost', 0, buffers,
            'sdp_l0test',
            [Endpoint('239.102.255.{}'.format(i), 7148) for i in range(self.n_endpoints)], None,
            'sdp_l1_flags_test', Endpoint('239.102.255.2', 7148), None, 64.0,
            self.telstate, self.report_path, self.log_path, None)
        self.server.start()
        self.addCleanup(self.ioloop.run_sync, self.stop_server)

        bind_address = self.server.bind_address
        self.client = katcp.AsyncClient(bind_address[0], bind_address[1], timeout=15)
        self.client.set_ioloop(self.ioloop)
        self.client.start()
        self.addCleanup(self.client.stop)
        self.addCleanup(self.client.disconnect)
        self.ioloop.run_sync(self.client.until_protocol)

    @tornado.gen.coroutine
    def make_request(self, name, *args, **kwargs):
        """Issue a request to the server, and check that the result is an ok.

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
        informs : list
            Informs returned with the reply
        """
        reply, informs = yield self.client.future_request(
            katcp.Message.request(name, *args), **kwargs)
        assert_true(reply.reply_ok(), str(reply))
        raise tornado.gen.Return(informs)

    @tornado.gen.coroutine
    def get_sensor(self, name):
        """Retrieves a sensor value and checks that the value is well-defined.

        Returns
        -------
        value : str
            The sensor value, in the string form it is sent in the protocol
        """
        informs = yield self.make_request('sensor-value', name)
        assert_equal(1, len(informs))
        assert_in(informs[0].arguments[3], ('nominal', 'warn', 'error'))
        raise tornado.gen.Return(informs[0].arguments[4])

    @tornado.gen.coroutine
    def assert_request_fails(self, msg_re, name, *args):
        """Assert that a request fails, and test the error message against
        a regular expression."""
        reply, informs = yield self.client.future_request(katcp.Message.request(name, *args))
        assert_equal(2, len(reply.arguments))
        assert_equal('fail', reply.arguments[0])
        assert_regexp_matches(reply.arguments[1], msg_re)

    @async_test
    @tornado.gen.coroutine
    def test_empty_capture(self):
        """Terminating a capture with no data must succeed and not write a report.

        It must also correctly remove the capture block from capture-block-state.
        """
        yield self.make_request('capture-init', 'empty_cb')
        state = yield self.get_sensor('capture-block-state')
        assert_equal('{"empty_cb": "CAPTURING"}', state)
        for i in range(self.n_endpoints):
            self.stream.send_heap(self.ig.get_end())
        yield self.make_request('capture-done')
        yield self.make_request('shutdown')
        assert_equal([], os.listdir(self.report_path))
        reports_written = yield self.get_sensor('reports-written')
        assert_equal(0, int(reports_written))
        state = yield self.get_sensor('capture-block-state')
        assert_equal('{}', state)

    @async_test
    @tornado.gen.coroutine
    def test_init_when_capturing(self):
        """capture-init fails when already capturing"""
        for i in range(self.n_endpoints):
            self.stream.send_heap(self.ig.get_end())
        yield self.make_request('capture-init')
        yield self.assert_request_fails(r'capture already in progress', 'capture-init')

    @async_test
    @tornado.gen.coroutine
    def test_done_when_not_capturing(self):
        """capture-done fails when not capturing"""
        yield self.assert_request_fails(r'no capture in progress', 'capture-done')
        yield self.make_request('capture-init')
        for i in range(self.n_endpoints):
            self.stream.send_heap(self.ig.get_end())
        yield self.make_request('capture-done')
        yield self.assert_request_fails(r'no capture in progress', 'capture-done')

    @classmethod
    def normalise_phase(cls, value, ref):
        """Multiply `value` by an amount that sets `ref` to zero phase."""
        ref_phase = ref / np.abs(ref)
        return value * ref_phase.conj()

    def _check_stopped(self, progress):
        assert_equal(['Accumulator stopped',
                      'Pipeline stopped',
                      'Sender stopped',
                      'ReportWriter stopped'], progress)

    @async_test
    @tornado.gen.coroutine
    def test_capture(self):
        """Tests the capture with some data, and checks that solutions are
        computed and a report written.
        """
        first_ts = ts = 100
        n_times = 10
        corrupt_times = (3, 7)
        rs = np.random.RandomState(seed=1)

        bandwidth = self.telstate.sdp_l0test_bandwidth
        target = katpoint.Target(self.telstate.m090_target)
        # The + bandwidth is to convert to L band
        freqs = np.arange(self.n_channels) / self.n_channels * bandwidth + bandwidth
        flux_density = target.flux_density(freqs / 1e6)[:, np.newaxis]
        freqs = freqs[:, np.newaxis]
        for antenna in self.antennas:
            self.telstate.add('{0}_dig_l_band_noise_diode'.format(antenna),
                              0, ts, immutable=False)
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
            for s in channel_slices:
                self.ig['correlator_data'].value = corrupted_vis[s] if i in (3, 7) else vis[s]
                self.ig['flags'].value = flags[s]
                self.ig['weights'].value = weights[s]
                self.ig['weights_channel'].value = weights_channel[s]
                self.ig['timestamp'].value = ts
                self.ig['dump_index'].value = i
                self.ig['frequency'].value = np.uint32(s.start)
                dump_heaps.append(self.ig.get_heap())
            rs.shuffle(dump_heaps)
            for heap in dump_heaps:
                self.stream.send_heap(heap)
            ts += self.telstate.sdp_l0test_int_time
        yield self.make_request('capture-init')
        yield tornado.gen.sleep(1)
        assert_equal(1, int((yield self.get_sensor('accumulator-capture-active'))))
        assert_equal('{"cal_cb_0": "CAPTURING"}', (yield self.get_sensor('capture-block-state')))
        for i in range(self.n_endpoints):
            self.stream.send_heap(self.ig.get_end())
        informs = yield self.make_request('shutdown', timeout=180)
        progress = [inform.arguments[0] for inform in informs]
        self._check_stopped(progress)
        assert_equal(0, int((yield self.get_sensor('accumulator-capture-active'))))
        assert_equal(n_times * self.n_substreams,
                     int((yield self.get_sensor('accumulator-input-heaps'))))
        assert_equal(1, int((yield self.get_sensor('accumulator-batches'))))
        assert_equal(1, int((yield self.get_sensor('accumulator-observations'))))
        assert_equal(n_times, int((yield self.get_sensor('pipeline-last-slots'))))
        assert_equal(1, int((yield self.get_sensor('reports-written'))))
        # Check that the slot accounting all balances
        assert_equal(40, int((yield self.get_sensor('slots'))))
        assert_equal(0, int((yield self.get_sensor('accumulator-slots'))))
        assert_equal(0, int((yield self.get_sensor('pipeline-slots'))))
        assert_equal(40, int((yield self.get_sensor('free-slots'))))
        assert_equal('{}', (yield self.get_sensor('capture-block-state')))

        reports = os.listdir(self.report_path)
        assert_equal(1, len(reports))
        report = os.path.join(self.report_path, reports[0])
        report_files = glob.glob(os.path.join(report, 'calreport_*.html'))
        assert_equal(1, len(report_files))
        assert_true(os.path.samefile(report, (yield self.get_sensor('report-last-path'))))

        cal_product_B = self.telstate.get_range('cal_product_B', st=0)
        assert_equal(1, len(cal_product_B))
        ret_B, ret_B_ts = cal_product_B[0]
        assert_equal(np.complex64, ret_B.dtype)

        cal_product_G = self.telstate.get_range('cal_product_G', st=0)
        assert_equal(1, len(cal_product_G))
        ret_G, ret_G_ts = cal_product_G[0]
        assert_equal(np.complex64, ret_G.dtype)
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

        cal_product_K = self.telstate.get_range('cal_product_K', st=0)
        assert_equal(1, len(cal_product_K))
        ret_K, ret_K_ts = cal_product_K[0]
        assert_equal(np.float32, ret_K.dtype)
        np.testing.assert_allclose(K - K[:, [0]], ret_K - ret_K[:, [0]], rtol=1e-3)

        # Check that flags were transmitted
        decoder = spead2.recv.Stream(spead2.ThreadPool())
        decoder.stop_on_stop_item = False
        decoder.add_buffer_reader(self.udp_stream.getvalue())
        heaps = list(decoder)
        assert_equal(n_times + 2, len(heaps))   # 2 extra for start and end heaps
        for i, heap in enumerate(heaps[1:-1]):
            items = spead2.ItemGroup()
            items.update(heap)
            ts = items['timestamp'].value
            assert_almost_equal(first_ts + i * self.telstate.sdp_l0test_int_time, ts)
            idx = items['dump_index'].value
            assert_equal(i, idx)
            out_flags = items['flags'].value
            # Mask out the ones that get changed by cal
            mask = (1 << FLAG_NAMES.index('static')) | (1 << FLAG_NAMES.index('cal_rfi'))
            np.testing.assert_array_equal(out_flags & ~mask, flags)

    def prepare_heaps(self, rs, n_times):
        """Produce a list of heaps with arbitrary data."""
        vis = np.ones((self.n_channels, self.n_baselines), np.complex64)
        weights = np.ones(vis.shape, np.uint8)
        weights_channel = np.ones((self.n_channels,), np.float32)
        flags = np.zeros(vis.shape, np.uint8)
        ts = 0.0
        channel_slices = [np.s_[i * self.n_channels_per_substream
                                : (i+1) * self.n_channels_per_substream]
                          for i in range(self.n_substreams)]
        heaps = []
        for i in range(n_times):
            dump_heaps = []
            for s in channel_slices:
                self.ig['correlator_data'].value = vis[s]
                self.ig['flags'].value = flags[s]
                self.ig['weights'].value = weights[s]
                self.ig['weights_channel'].value = weights_channel[s]
                self.ig['timestamp'].value = ts
                self.ig['dump_index'].value = i
                self.ig['frequency'].value = np.uint32(s.start)
                dump_heaps.append(self.ig.get_heap())
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
        n_times = 90
        for heap in self.prepare_heaps(rs, n_times):
            self.stream.send_heap(heap)
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
        yield self.make_request('capture-init')
        # Wait until all the heaps have been delivered, timing out eventually.
        # This will take a while because it needs to allow the pipeline to run.
        for i in range(180):
            yield tornado.gen.sleep(0.5)
            heaps = int((yield self.get_sensor('accumulator-input-heaps')))
            if heaps == n_times * self.n_substreams:
                print('all heaps received')
                break
            print('waiting {} ({}/{} received)'.format(i, heaps, n_times * self.n_substreams))
        else:
            raise RuntimeError('Timed out waiting for the heaps to be received')
        for i in range(self.n_endpoints):
            self.stream.send_heap(self.ig.get_end())
        informs = yield self.make_request('shutdown', timeout=60)
        progress = [inform.arguments[0] for inform in informs]
        self._check_stopped(progress)

    @async_test
    @tornado.gen.coroutine
    def test_out_of_order(self):
        """A heap received from the past should be processed (if possible)."""
        rs = np.random.RandomState(seed=1)
        n_times = 4
        heaps = self.prepare_heaps(rs, n_times)
        # Delay one heap to the end
        victim = 3
        heaps = heaps[: victim] + heaps[victim + 1 :] + heaps[victim : victim + 1]
        for heap in heaps:
            self.stream.send_heap(heap)
        # Run the capture
        yield self.make_request('capture-init')
        yield tornado.gen.sleep(1)
        informs = yield self.make_request('shutdown', timeout=60)
        # Check that all heaps were accepted
        assert_equal(n_times * self.n_substreams,
                     int((yield self.get_sensor('accumulator-input-heaps'))))

    @async_test
    @tornado.gen.coroutine
    def test_pipeline_exception(self):
        with mock.patch.object(control.Pipeline, 'run_pipeline', side_effect=ZeroDivisionError):
            assert_equal(0, int((yield self.get_sensor('pipeline-exceptions'))))
            for heap in self.prepare_heaps(np.random.RandomState(seed=1), 5):
                self.stream.send_heap(heap)
            yield self.make_request('capture-init', 'testcb')
            yield tornado.gen.sleep(1)
            assert_equal('{"testcb": "CAPTURING"}', (yield self.get_sensor('capture-block-state')))
            for i in range(self.n_endpoints):
                self.stream.send_heap(self.ig.get_end())
            yield self.make_request('shutdown')
            informs = yield self.make_request('shutdown', timeout=60)
            progress = [inform.arguments[0] for inform in informs]
            self._check_stopped(progress)
            assert_equal(1, int((yield self.get_sensor('pipeline-exceptions'))))
            assert_equal('{}', (yield self.get_sensor('capture-block-state')))
