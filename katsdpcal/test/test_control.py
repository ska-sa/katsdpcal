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
from collections import deque

import numpy as np
from nose.tools import (
    assert_equal, assert_is_instance, assert_in, assert_false, assert_true, assert_regexp_matches,
    nottest)
import mock

import tornado.gen
from tornado.platform.asyncio import AsyncIOMainLoop
import trollius

import spead2
import katcp
import katsdptelstate
from katsdptelstate.endpoint import Endpoint
from katsdpservices.asyncio import to_tornado_future

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


class MockStream(mock.MagicMock):
    """Mock replacement for :class:`spead2.recv.trollius.Stream`.

    It has a list of pseudo-heaps that it yields to the caller. A "pseudo-heap"
    is either:
    - ``None``, which causes the stream to be stopped
    - A future, which is returned as is (up to the caller to resolve the future).
    - anything else yielded as-is (wrapped in a future)

    If the list runs out, it returns a future that can only be resolved by
    calling :meth:`stop`.
    """
    def __init__(self, heaps):
        super(MockStream, self).__init__()
        self._heaps = heaps
        self._last_future = None
        self._stopped = False

    # Make child mocks use the basic MagicMock, not this class
    def _get_child_mock(self, **kwargs):
        return mock.MagicMock(**kwargs)

    def get(self):
        if self._stopped:
            raise spead2.Stopped()
        elif not self._heaps:
            future = trollius.Future()
        else:
            value = self._heaps.popleft()
            if value is None:
                self._stopped = True
                raise spead2.Stopped()
            elif isinstance(value, trollius.Future):
                future = value
            else:
                future = trollius.Future()
                future.set_result(value)
        self._last_future = future
        return future

    def stop(self):
        self._stopped = True
        self._heaps = deque()
        if self._last_future is not None and not self._last_future.done():
            self._last_future.set_exception(spead2.Stopped())


def mock_item_group_update(self, heap):
    """Mock replacement for :meth:`spead2.ItemGroup.update`.

    Instead of a real heap, it takes either
    - a dict replacement values instead, or
    - a callable which is passed the item group.
    """
    if isinstance(heap, dict):
        for key, value in heap.iteritems():
            self[key].value = value
        return heap
    else:
        return heap(self)


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
        antennas = ['m090', 'm091', 'm092', 'm093']
        bls_ordering = []
        target = ('PKS 0408-65 | J0408-6545, radec bfcal single_accumulation, '
                  '4:08:20.38, -65:45:09.1, (800.0 8400.0 -3.708 3.807 -0.7202)')
        ant_bls = []     # Antenna pairs, later expanded to pol pairs
        for a in antennas:
            ant_bls.append((a, a))
        for a in antennas:
            for b in antennas:
                if a < b:
                    ant_bls.append((a, b))
        for a, b in ant_bls:
            bls_ordering.append((a + 'h', b + 'h'))
            bls_ordering.append((a + 'v', b + 'v'))
            bls_ordering.append((a + 'h', b + 'v'))
            bls_ordering.append((a + 'v', b + 'h'))
        telstate.clear()     # Prevent state leaks
        telstate.add('sdp_l0_int_time', 4.0, immutable=True)
        telstate.add('antenna_mask', ','.join(antennas), immutable=True)
        telstate.add('cbf_n_ants', len(antennas), immutable=True)
        telstate.add('cbf_n_chans', 4096, immutable=True)
        telstate.add('cbf_n_pols', 4, immutable=True)
        telstate.add('cbf_center_freq', 428000000.0, immutable=True)
        telstate.add('cbf_bandwidth', 856000000.0, immutable=True)
        telstate.add('sdp_l0_bls_ordering', bls_ordering, immutable=True)
        telstate.add('cbf_sync_time', 1400000000.0, immutable=True)
        telstate.add('subarray_product_id', 'c856M4k', immutable=True)
        for antenna in antennas:
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
        pipelineprocs.setup_ts(telstate)

    def add_items(self, ig):
        channels = self.telstate.cbf_n_chans
        baselines = len(self.telstate.sdp_l0_bls_ordering)
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
        ig.add_item(id=0x4103, name='frequency',
                    description="Channel index of first channel in the heap",
                    shape=(), dtype=np.uint32)
        return {}

    @tornado.gen.coroutine
    def stop_server(self):
        yield to_tornado_future(self.server.shutdown())
        self.server.stop()

    def setUp(self):
        self.telstate = katsdptelstate.TelescopeState()
        self.populate_telstate(self.telstate)
        self.ioloop = AsyncIOMainLoop()
        self.ioloop.install()
        self.addCleanup(tornado.ioloop.IOLoop.clear_instance)

        self.heaps = deque()
        self.heaps.append(self.add_items)
        self.patch('spead2.recv.trollius.Stream', return_value=MockStream(self.heaps))
        self.patch('spead2.ItemGroup.update', mock_item_group_update)

        self.report_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.report_path)
        self.log_path = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.log_path)

        buffer_shape = (20, 4096, 4, 10)  # Time, channels, pols, baselines
        num_buffers = 2
        buffers = [control.create_buffer_arrays(buffer_shape, False) for i in range(num_buffers)]
        self.server = control.create_server(
            False, 'localhost', 0, buffers,
            Endpoint('239.102.255.1', 7148), None,
            None, 0, None, self.telstate,
            self.report_path, self.log_path, None)
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
    def make_request(self, name, *args):
        """Issue a request to the server, and check that the result is an ok.

        Parameters
        ----------
        name : str
            Request name
        args : list
            Arguments to the request

        Returns
        -------
        informs : list
            Informs returned with the reply
        """
        reply, informs = yield self.client.future_request(katcp.Message.request(name, *args))
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
        """Terminating a capture with no data must succeed and not write a report."""
        yield self.make_request('capture-init')
        yield self.make_request('capture-done')
        yield self.make_request('shutdown')
        assert_equal([], os.listdir(self.report_path))
        reports_written = yield self.get_sensor('reports-written')
        assert_equal(0, int(reports_written))

    @async_test
    @tornado.gen.coroutine
    def test_init_when_capturing(self):
        """capture-init fails when already capturing"""
        yield self.make_request('capture-init')
        yield self.assert_request_fails(r'capture already in progress', 'capture-init')

    @async_test
    @tornado.gen.coroutine
    def test_done_when_not_capturing(self):
        """capture-done fails when not capturing"""
        yield self.assert_request_fails(r'no capture in progress', 'capture-done')

    @async_test
    @tornado.gen.coroutine
    def test_capture(self):
        """Tests the capture with some data, and checks that solutions are
        computed and a report written.
        """
        channels = self.telstate.cbf_n_chans
        baselines = len(self.telstate.sdp_l0_bls_ordering)
        ts = 100
        for i in range(10):
            shape = (channels, baselines)
            vis = np.ones(shape, np.complex64)
            flags = np.zeros(shape, np.uint8)
            weights = np.ones(shape, np.uint8)
            weights_channel = np.ones(channels, np.float32)
            self.heaps.append({
                'correlator_data': vis,
                'flags': flags,
                'weights': weights,
                'weights_channel': weights_channel,
                'timestamp': ts
            })
            ts += self.telstate.sdp_l0_int_time
        yield self.make_request('capture-init')
        yield tornado.gen.sleep(1)
        assert_equal(1, int((yield self.get_sensor('accumulator-capture-active'))))
        informs = yield self.make_request('shutdown')
        progress = [inform.arguments[0] for inform in informs]
        assert_equal(['Accumulator stopped',
                      'Pipelines stopped',
                      'Report writer stopped'], progress)
        assert_equal(0, int((yield self.get_sensor('accumulator-capture-active'))))
        assert_equal(10, int((yield self.get_sensor('accumulator-input-heaps'))))
        assert_equal(1, int((yield self.get_sensor('accumulator-batches'))))
        assert_equal(1, int((yield self.get_sensor('accumulator-observations'))))
        assert_equal(10, int((yield self.get_sensor('pipeline-last-slots'))))
        assert_equal(1, int((yield self.get_sensor('reports-written'))))
        # TODO: carefully construct artificial data and check that the results match
        assert_in('cal_product_B', self.telstate)
        assert_in('cal_product_G', self.telstate)
        assert_in('cal_product_K', self.telstate)
        reports = os.listdir(self.report_path)
        assert_equal(1, len(reports))
        report = os.path.join(self.report_path, reports[0])
        report_files = glob.glob(os.path.join(report, 'calreport_*.html'))
        assert_equal(1, len(report_files))
        assert_true(os.path.samefile(report, (yield self.get_sensor('report-last-path'))))
