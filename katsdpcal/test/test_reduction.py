"""Tests for :mod:`katsdpcal.reduction`"""

import unittest
from unittest.mock import patch
from unittest.mock import call
import katsdptelstate
import katpoint
from katsdpcalproc import pointing, test_pointing
from unittest import mock
import concurrent.futures
import numpy as np
import time

from katsdpcalproc.solutions import (
    CalSolution,
    CalSolutions,
    CalSolutionStore,
    CalSolutionStoreLatest,
)
from .. import reduction


class DummyError(Exception):
    pass


ANT_DESCRIPTIONS = [
    """m000,
        -30:42:39.8,
        21:26:38.0,
        1086.6,
        15,
        -8.264 -207 8.6 212.6 212.6 1,
        0:04:20.6 0 0:01:14.2 0:02:58.5 0:00:05.1 0:00:00.4
                          0:20:04.1 -0:00:34.5 0 0 -0:03:10.0 ,1.22""",
    """m001,
        -30:42:39.8,
        21:26:38.0,
        1086.6,
        15,
        -8.264 -207 8.6 212.6 212.6 1,
        0:04:15.6 0 0:01:09.2 0:01:58.5 0:00:05.1 0:00:00.4
                          0:16:04.1 -0:00:34.5 0 0 -0:03:10.0 ,1.22""",
    """m002,
        -30:42:39.8,
        21:26:38.0,
        1086.6,
        15,
        -8.264 -207 8.6 212.6 212.6 1,
        0:04:20.6 0 0:01:14.2 0:02:58.5 0:00:05.1 0:00:00.4
                          0:20:04.1 -0:00:34.5 0 0 -0:03:10.0 ,1.22""",
]


class TestSharedSolve(unittest.TestCase):
    def setUp(self):
        self.telstate = katsdptelstate.TelescopeState()
        self.telstate.clear()
        self.n_servers = 4
        self.executor = concurrent.futures.ThreadPoolExecutor(self.n_servers)
        self.server_chans = 1024
        self.pols = ["h"]
        self.bchan = 1100
        self.echan = 1300
        self.num_chunks = 8
        self._seq = 0
        self.ants = [katpoint.Antenna(ant) for ant in ANT_DESCRIPTIONS]
        self.target = katpoint.Target(
            body="J1939-6342, radec bfcal single_accumulation, 19:39:25.03, -63:42:45.6"
        )
        self.offsets = test_pointing._pointing_offsets(max_extent=1.0, num_pointings=8)
        self.telstate.add("cbf_target", self.target, ts=0)
        self.telstate.add("anc_air_pressure", 897, ts=0)
        self.telstate.add("anc_air_temperature", 14.7, ts=0)
        self.telstate.add("anc_air_relative_humidity", 29.1, ts=0)
        for ant in self.ants:
            self.telstate.add(ant.name + "_pos_actual_scan_azim", -158.29920978, ts=0)
            self.telstate.add(ant.name + "_pos_actual_scan_elev", 51.66647631, ts=0)
        self.parameters = [
            {
                "product_names": {
                    "G": "product_G",
                    "K": "product_K",
                    "KCROSS": "product_KCROSS",
                    "B": "product_B{}".format(i),
                    "EPOINT": "product_EPOINT{}".format(i),
                },
                "channel_freqs": np.linspace(self.bchan, self.echan, self.server_chans),
            }
            for i in range(self.n_servers)
        ]
        self.solution_stores = {
            "K": CalSolutionStoreLatest("K"),
            "B": CalSolutionStoreLatest("B"),
            "G": CalSolutionStore("G"),
            "B_POINTING": CalSolutionStore("B"),
        }
        self.bp_gains = test_pointing.generate_bp_gains(
            self.offsets, self.ants, self.channel_freqs, self.pols
        )
        self.bp_gains[:, :, :, 1] = np.nan
        # Have to add bp gains per offset/ timestamp to mimic pipeline and get correct shape at the end
        for i in range(len(self.bp_gains)):
            self.bp_sol = CalSolution("B", self.bp_gains[i], i + 1, None)
            self.solution_stores["B_POINTING"].add(self.bp_sol)
        self.times = self.solution_stores["B_POINTING"].get_range(0, time.time()).times

    def tearDown(self):
        self.executor.shutdown()

    def call_futures(self, name, bchan, echan, solver, *args, **kwargs):
        """Run shared_solve and return a future from each server"""

        kwargs["_seq"] = self._seq
        self._seq += 1
        solution_store = self.solution_stores[name] if name else None
        return [
            self.executor.submit(
                reduction.shared_solve,
                self.telstate,
                self.parameters[i],
                solution_store,
                bchan - i * self.server_chans,
                echan - i * self.server_chans,
                solver,
                *args,
                **kwargs
            )
            for i in range(self.n_servers)
        ]

    def call(self, name, bchan, echan, solver, *args, **kwargs):
        futures = self.call_futures(name, bchan, echan, solver, *args, **kwargs)
        return [future.result(timeout=5) for future in futures]

    def _test_cal_solution(self, name):
        def solver(bchan, echan):
            values = np.arange(123)
            values[0] = bchan
            values[1] = echan
            return CalSolution(name or "K", values, 12345.5, "Test")

        results = self.call(name, self.bchan, self.echan, solver)
        expected = np.arange(123)
        expected[0] = self.bchan % self.server_chans
        expected[1] = self.echan % self.server_chans
        for i in range(self.n_servers):
            self.assertIsInstance(results[i], CalSolution)
            self.assertEqual(results[i].soltype, "K")
            np.testing.assert_array_equal(results[i].values, expected)
            self.assertEqual(results[i].time, 12345.5)
            self.assertEqual(results[i].target, "Test")

    def test_cal_solution_named(self):
        self._test_cal_solution("K")

    def test_cal_solution_anonymous(self):
        self._test_cal_solution(None)

    def _test_cal_solutions(self, name):
        def solver(bchan, echan):
            values = np.arange(128).reshape(2, 4, -1)
            values[0, 0, 0] = bchan
            values[1, 1, 1] = echan
            times = np.array([23456.5, 34567.5])
            return CalSolutions("G", values, times)

        results = self.call(name, self.bchan, self.echan, solver)
        expected = np.arange(128).reshape(2, 4, -1)
        expected[0, 0, 0] = self.bchan % self.server_chans
        expected[1, 1, 1] = self.echan % self.server_chans
        for i in range(self.n_servers):
            self.assertIsInstance(results[i], CalSolutions)
            self.assertEqual(results[i].soltype, "G")
            np.testing.assert_array_equal(results[i].values, expected)
            np.testing.assert_array_equal(results[i].times, [23456.5, 34567.5])

    def test_cal_solutions_named(self):
        self._test_cal_solutions("G")

    def test_cal_solutions_anonymous(self):
        self._test_cal_solutions(None)

    def test_exception(self):
        def solver(bchan, echan):
            raise DummyError("CRASH")

        futures = self.call_futures("G", self.bchan, self.echan, solver)
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
        self._test_int("K")

    def test_missing_ants(self):
        expected_message = "Skipping m001, no beam available"

        with patch("reduction.logger") as mock_logger:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        reduction.flush_pipeline,
                        self.telstate,
                        self.parameters[i],
                        self.solution_stores,
                    )
                    for i in range(self.n_servers)
                ]
                concurrent.futures.wait(futures)

            matching_calls = [
                call
                for call in mock_logger.info.call_args_list
                if call.args and call.args[0] == expected_message
            ]

            self.assertEqual(len(matching_calls), self.n_servers)