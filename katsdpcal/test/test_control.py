"""Tests for control module"""

from multiprocessing import Process
import multiprocessing
import multiprocessing.dummy
from nose.tools import assert_equal, assert_is_instance, assert_false, assert_true
import numpy as np
import katcp
import katsdpcal.control as control


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
        assert_equal(False, task.daemon)
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
