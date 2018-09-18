"""Tests for :mod:`katsdpcal.pipelineprocs`"""

from __future__ import print_function, division, absolute_import
import argparse
import unittest

import numpy as np

import katsdptelstate
import katpoint
import mock

from .. import pipelineprocs


class TestArgparseParameters(unittest.TestCase):
    def test_basic(self):
        parser = argparse.ArgumentParser()
        pipelineprocs.register_argparse_parameters(parser)
        argv = ['--rfi-windows-freq=1,2,4,8', '--g-solint=1']
        args = parser.parse_args(argv)
        parameters = pipelineprocs.parameters_from_argparse(args)
        self.assertEqual([1, 2, 4, 8], parameters['rfi_windows_freq'])
        self.assertEqual(1.0, parameters['g_solint'])
        # Non-specified arguments must not appear at all
        self.assertNotIn('g_bchan', parameters)

    def test_empty_list(self):
        parser = argparse.ArgumentParser()
        pipelineprocs.register_argparse_parameters(parser)
        argv = ['--rfi-windows-freq=']
        args = parser.parse_args(argv)
        parameters = pipelineprocs.parameters_from_argparse(args)
        self.assertEqual([], parameters['rfi_windows_freq'])


class TestFinaliseParameters(unittest.TestCase):
    def setUp(self):
        # These parameters are based on pipeline_parameters_meerkat_L_4k.txt
        self.parameters = {
            'k_solint': 5.0,
            'k_chan_sample': 1,
            'k_bchan': 2250,
            'k_echan': 2450,
            'kcross_chanave': 1,
            'bp_solint': 5.0,
            'g_solint': 5.0,
            'g_bchan': 2250,
            'g_echan': 2450,
            'rfi_calib_nsigma': 4.5,
            'rfi_targ_nsigma': 7.0,
            'rfi_windows_freq': [1, 2, 4, 8],
            'rfi_average_freq': 1,
            'rfi_targ_spike_width_freq': 20.0,
            'rfi_calib_spike_width_freq': 10.0,
            'rfi_spike_width_time': 100.0,
            'rfi_extend_freq': 3,
            'rfi_freq_chunks': 8
        }

        self.antenna_names = ['m001', 'm002', 'm004', 'm006']
        self.antennas = [
            # The position is irrelevant for now, so just give all the
            # antennas the same position.
            katpoint.Antenna(
                '{}, -30:42:47.4, 21:26:38.0, 1035.0, 13.5, -351.163669759 384.481835294, '
                '-0:05:44.7 0 0:00:22.6 -0:09:04.2 0:00:11.9 -0:00:12.8 -0:04:03.5 0 0 '
                '-0:01:33.0 0:01:45.6 0 0 0 0 0 -0:00:03.6 -0:00:17.5, 1.22'.format(name))
            for name in self.antenna_names]

        ant_bls = []     # Antenna pairs, later expanded to pol pairs
        for a in self.antenna_names:
            ant_bls.append((a, a))
        for a in self.antenna_names:
            for b in self.antenna_names:
                if a < b:
                    ant_bls.append((a, b))
        bls_ordering = []
        for a, b in ant_bls:
            bls_ordering.append((a + 'h', b + 'h'))
            bls_ordering.append((a + 'v', b + 'v'))
            bls_ordering.append((a + 'h', b + 'v'))
            bls_ordering.append((a + 'v', b + 'h'))

        self.telstate = katsdptelstate.TelescopeState()
        self.telstate.clear()
        self.telstate_l0 = self.telstate.view('sdp_l0test')
        self.telstate_l0.add('n_chans', 4096, immutable=True)
        self.telstate_l0.add('bandwidth', 856000000.0, immutable=True)
        self.telstate_l0.add('center_freq', 1284000000.0, immutable=True)
        self.telstate_l0.add('bls_ordering', bls_ordering, immutable=True)
        for antenna in self.antennas:
            self.telstate.add(antenna.name + '_observer', antenna.description, immutable=True)

    def test_normal(self):
        parameters = pipelineprocs.finalise_parameters(
            self.parameters, self.telstate_l0, 4, 2, None)
        self.assertEqual(None, parameters['refant_index'])
        self.assertEqual(self.antenna_names, parameters['antenna_names'])
        self.assertEqual(self.antennas, parameters['antennas'])
        expected_freqs = np.arange(1024) / 4096 * 856000000.0 + 1284000000.0
        expected_freqs_all = np.arange(4096) / 4096 * 856000000.0 + 856000000.0
        np.testing.assert_allclose(expected_freqs, parameters['channel_freqs'])
        np.testing.assert_allclose(expected_freqs_all, parameters['channel_freqs_all'])
        self.assertEqual(slice(2048, 3072), parameters['channel_slice'])
        np.testing.assert_array_equal(np.zeros(1024, np.bool_), parameters['rfi_mask'])
        # Check the channel indices are offset
        self.assertEqual(202, parameters['k_bchan'])
        self.assertEqual(402, parameters['k_echan'])
        self.assertEqual(202, parameters['g_bchan'])
        self.assertEqual(402, parameters['g_echan'])
        self.assertEqual(2, parameters['rfi_freq_chunks'])
        # bls_ordering, pol_ordering, bls_lookup get tested elsewhere

    def test_rfi_mask(self):
        # Create a random RFI mask. Randomness makes it highly likely that a
        # shift will be detected.
        rs = np.random.RandomState(seed=1)
        mask = rs.rand(4096) < 0.5
        with mock.patch('__builtin__.open'):   # To suppress trying to open a real file
            with mock.patch('pickle.load', return_value=mask.copy()):
                parameters = pipelineprocs.finalise_parameters(
                    self.parameters, self.telstate_l0, 4, 2, rfi_filename='my_rfi_file')
        np.testing.assert_array_equal(mask[2048:3072], parameters['rfi_mask'])

    def test_bad_rfi_mask(self):
        mask = np.zeros(4097, np.bool_) < 0.5  # Wrong number of channels
        with mock.patch('__builtin__.open'):   # To suppress trying to open a real file
            with mock.patch('pickle.load', return_value=mask.copy()):
                with self.assertRaises(ValueError):
                    pipelineprocs.finalise_parameters(
                        self.parameters, self.telstate_l0, 4, 2, rfi_filename='my_rfi_file')

    def test_invalid_server_id(self):
        with self.assertRaises(ValueError):
            pipelineprocs.finalise_parameters(self.parameters, self.telstate_l0, 4, 5, None)

    def test_invalid_server_count(self):
        with self.assertRaises(ValueError):
            pipelineprocs.finalise_parameters(self.parameters, self.telstate_l0, 3, 0, None)

    def test_missing_parameter(self):
        del self.parameters['k_bchan']
        with self.assertRaises(ValueError):
            pipelineprocs.finalise_parameters(self.parameters, self.telstate_l0, 4, 2, None)

    def test_bad_channel_range(self):
        self.parameters['k_echan'] = 2200
        with self.assertRaises(ValueError):
            pipelineprocs.finalise_parameters(self.parameters, self.telstate_l0, 4, 2, None)

    def test_channel_range_spans_servers(self):
        self.parameters['k_echan'] = 3073
        with self.assertRaises(ValueError):
            pipelineprocs.finalise_parameters(self.parameters, self.telstate_l0, 4, 2, None)

    def test_unexpected_parameters(self):
        self.parameters['foo'] = 'bar'
        with self.assertRaises(ValueError):
            pipelineprocs.finalise_parameters(self.parameters, self.telstate_l0, 4, 2, None)
