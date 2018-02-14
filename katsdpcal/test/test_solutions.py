"""Tests for :mod:`katsdpcal.solutions`"""

import unittest

import numpy as np

from ..solutions import CalSolution, CalSolutionStore, CalSolutionStoreLatest


class TestCalSolutionStoreLatest(unittest.TestCase):
    test_cls = CalSolutionStoreLatest

    def setUp(self):
        self.sol1 = CalSolution('G', np.arange(10), 1234.0)
        self.sol2 = CalSolution('G', np.arange(10, 20), 2345.0)
        self.sol3 = CalSolution('G', np.arange(30, 40), 3456.0)
        self.solK = CalSolution('K', np.arange(30, 40), 3456.0)
        self.store = self.test_cls('G')

    def test_latest_empty(self):
        self.assertIsNone(self.store.latest)

    def test_add_wrong_type(self):
        with self.assertRaises(ValueError):
            self.store.add(self.solK)

    def test_keep_latest(self):
        self.store.add(self.sol2)
        self.assertIs(self.store.latest, self.sol2)
        self.store.add(self.sol1)    # Earlier, should be ignored
        self.assertIs(self.store.latest, self.sol2)
        self.store.add(self.sol3)    # Later, should be kept
        self.assertIs(self.store.latest, self.sol3)

    def test_get_range(self):
        with self.assertRaises(NotImplementedError):
            self.store.get_range(1234.0, 2345.0)


class TestCalSolutionStore(TestCalSolutionStoreLatest):
    test_cls = CalSolutionStore

    def test_get_range(self):
        self.store.add(self.sol2)
        self.store.add(self.sol1)
        self.store.add(self.sol3)
        soln = self.store.get_range(1234.0, 2345.0)
        self.assertEqual(soln.soltype, 'G')
        np.testing.assert_array_equal(soln.values, np.arange(20).reshape(2, 10))
        np.testing.assert_array_equal(soln.times, [1234.0, 2345.0])
