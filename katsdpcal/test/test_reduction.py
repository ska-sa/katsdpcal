"""Tests for :mod:`katsdpcal.reduction`"""

import unittest
import katsdptelstate
import concurrent.futures
import numpy as np

from ..solutions import CalSolution, CalSolutions, CalSolutionStore, CalSolutionStoreLatest
from .. import reduction


class DummyError(Exception):
    pass


class TestSharedSolve(unittest.TestCase):
    def setUp(self):
        self.telstate = katsdptelstate.TelescopeState()
        self.telstate.clear()
        self.n_servers = 4
        self.executor = concurrent.futures.ThreadPoolExecutor(self.n_servers)
        self.server_chans = 1024
        self.bchan = 1100
        self.echan = 1300
        self._seq = 0
        self.parameters = [
            {
                'product_names': {
                    'G': 'product_G',
                    'K': 'product_K',
                    'KCROSS': 'product_KCROSS',
                    'B': 'product_B{}'.format(i)
                },
                'channel_freqs': np.arange(self.server_chans)  # only length matters
            } for i in range(self.n_servers)]
        self.solution_stores = {
            'K': CalSolutionStoreLatest('K'),
            'B': CalSolutionStoreLatest('B'),
            'G': CalSolutionStore('G')
        }

    def tearDown(self):
        self.executor.shutdown()

    def call_futures(self, name, bchan, echan, solver, *args, **kwargs):
        """Run shared_solve and return a future from each server"""

        kwargs['_seq'] = self._seq
        self._seq += 1
        solution_store = self.solution_stores[name] if name else None
        return [self.executor.submit(
            reduction.shared_solve,
            self.telstate, self.parameters[i], solution_store,
            bchan - i * self.server_chans, echan - i * self.server_chans,
            solver, *args, **kwargs) for i in range(self.n_servers)]

    def call(self, name, bchan, echan, solver, *args, **kwargs):
        futures = self.call_futures(name, bchan, echan, solver, *args, **kwargs)
        return [future.result(timeout=5) for future in futures]

    def _test_cal_solution(self, name):
        def solver(bchan, echan):
            values = np.arange(123)
            values[0] = bchan
            values[1] = echan
            return CalSolution(name or 'K', values, 12345.5)

        results = self.call(name, self.bchan, self.echan, solver)
        expected = np.arange(123)
        expected[0] = self.bchan % self.server_chans
        expected[1] = self.echan % self.server_chans
        for i in range(self.n_servers):
            self.assertIsInstance(results[i], CalSolution)
            self.assertEqual(results[i].soltype, 'K')
            np.testing.assert_array_equal(results[i].values, expected)
            self.assertEqual(results[i].time, 12345.5)

    def test_cal_solution_named(self):
        self._test_cal_solution('K')

    def test_cal_solution_anonymous(self):
        self._test_cal_solution(None)

    def _test_cal_solutions(self, name):
        def solver(bchan, echan):
            values = np.arange(128).reshape(2, 4, -1)
            values[0, 0, 0] = bchan
            values[1, 1, 1] = echan
            times = np.array([23456.5, 34567.5])
            return CalSolutions('G', values, times)

        results = self.call(name, self.bchan, self.echan, solver)
        expected = np.arange(128).reshape(2, 4, -1)
        expected[0, 0, 0] = self.bchan % self.server_chans
        expected[1, 1, 1] = self.echan % self.server_chans
        for i in range(self.n_servers):
            self.assertIsInstance(results[i], CalSolutions)
            self.assertEqual(results[i].soltype, 'G')
            np.testing.assert_array_equal(results[i].values, expected)
            np.testing.assert_array_equal(results[i].times, [23456.5, 34567.5])

    def test_cal_solutions_named(self):
        self._test_cal_solutions('G')

    def test_cal_solutions_anonymous(self):
        self._test_cal_solutions(None)

    def test_exception(self):
        def solver(bchan, echan):
            raise DummyError('CRASH')

        futures = self.call_futures('G', self.bchan, self.echan, solver)
        for i in range(self.n_servers):
            with self.assertRaises(DummyError):
                futures[i].result(timeout=5)

    def _test_int(self, name):
        def solver(bchan, echan):
            return bchan

        results = self.call(name, self.bchan, self.echan, solver)
        for i in range(self.n_servers):
            expected = self.bchan % self.server_chans
            self.assertEqual(results[i], expected)

    def test_int_anonymous(self):
        self._test_int(None)

    def test_int_named(self):
        self._test_int('K')
