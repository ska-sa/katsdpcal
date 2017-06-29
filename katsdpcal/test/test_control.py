"""Tests for control module"""

from multiprocessing import Process
from nose.tools import assert_equal
import numpy as np
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
