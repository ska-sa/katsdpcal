"""Tests for :mod:`katsdpcal.pipelineprocs`"""

import os
import argparse
import unittest
import tempfile
import shutil
from unittest import mock

import numpy as np

import katsdptelstate
import katpoint

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
        # These parameters are based on pipeline_parameters_meerkat_L.txt
        self.parameters = {
            'k_solint': 5.0,
            'k_chan_sample': 1,
            'k_bchan': 1326.200,
            'k_echan': 1368.012,
            'kcross_chanave': 1,
            'bp_solint': 5.0,
            'g_solint': 5.0,
            'g_bchan': 1326.200,
            'g_echan': 1368.012,
            'rfi_calib_nsigma': 4.5,
            'rfi_targ_nsigma': 7.0,
            'rfi_windows_freq': [1, 2, 4, 8],
            'rfi_average_freq': 104491,
            'rfi_targ_spike_width_freq': 4179687,
            'rfi_calib_spike_width_freq': 2089843,
            'rfi_spike_width_time': 100.0,
            'rfi_extend_freq': 391845,
            'rfi_freq_chunks': 8
        }

        self.antennas = [
            katpoint.Antenna(
                'm001, -30:42:47.4, 21:26:38.0, 1035.0, 13.5, -351.163669759 384.481835294, '
                '-0:05:44.7 0 0:00:22.6 -0:09:04.2 0:00:11.9 -0:00:12.8 -0:04:03.5 0 0 '
                '-0:01:33.0 0:01:45.6 0 0 0 0 0 -0:00:03.6 -0:00:17.5, 1.22'),
            # this antenna has a baseline larger than 1000m to the other antennas
            katpoint.Antenna(
                'm002, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, 9.3645 -1304.462 7.632 '
                '5872.543 5873.042 1.0, 0:06:11.7 0 -0:00:43.1 0:04:09.0 0:00:40.6 '
                '0:00:04.9 0:09:00.9 0:01:31.7, 1.22'),
            katpoint.Antenna(
                'm004, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -123.624 -252.946 1.113 '
                '5870.119 5870.721 1.0, 1:03:04.1 0 0:00:43.8 0:00:44.8 -0:00:29.0 '
                '-0:00:14.0 -0:08:42.9 0:01:15.1, 1.22'),
            katpoint.Antenna(
                'm006, -30:42:39.8, 21:26:38.0, 1035.0, 13.5, -18.2315 -295.428 1.788 '
                '5880.189 5880.611 1.0, 1:58:23.5 0 0:00:23.3 0:01:44.4 -0:00:16.9 '
                '-0:00:09.1 -0:07:41.5 0:01:19.5, 1.22')]
        self.antenna_names = [a.name for a in self.antennas]

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
        self.telstate_l0['n_chans'] = 4096
        self.telstate_l0['bandwidth'] = 856000000.0
        self.telstate_l0['center_freq'] = 1284000000.0
        self.telstate_l0['bls_ordering'] = bls_ordering
        for antenna in self.antennas:
            self.telstate[self.telstate.join(antenna.name, 'observer')] = antenna.description
        self.expected_freqs_all = np.arange(4096) / 4096 * 856000000.0 + 856000000.0

    def test_normal(self):
        parameters = pipelineprocs.finalise_parameters(
            self.parameters, self.telstate_l0, 4, 2, None)
        self.assertEqual(None, parameters['refant_index'])
        self.assertEqual(self.antenna_names, parameters['antenna_names'])
        self.assertEqual(self.antennas, parameters['antennas'])
        expected_freqs = np.arange(1024) / 4096 * 856000000.0 + 1284000000.0
        np.testing.assert_allclose(expected_freqs, parameters['channel_freqs'])
        np.testing.assert_allclose(self.expected_freqs_all, parameters['channel_freqs_all'])
        self.assertEqual(slice(2048, 3072), parameters['channel_slice'])
        np.testing.assert_array_equal(np.zeros((1, 1024, 1, 10), np.bool_), parameters['rfi_mask'])
        # Check the channel indices are offset
        self.assertEqual(202, parameters['k_bchan'])
        self.assertEqual(402, parameters['k_echan'])
        self.assertEqual(202, parameters['g_bchan'])
        self.assertEqual(402, parameters['g_echan'])
        self.assertEqual(2, parameters['rfi_freq_chunks'])
        # bls_ordering, pol_ordering, bls_lookup get tested elsewhere

    def test_rfi_mask(self):
        # Create a set of random RFI ranges as a mask.
        # Randomness makes it highly likely that a shift will be detected
        rs = np.random.RandomState(seed=1)
        random_freqs = np.ceil(rs.rand(8) * 856) + 856
        rfi_ranges = np.sort(random_freqs).reshape((4, 2))
        with mock.patch('builtins.open'):  # To suppress trying to open a real file
            with mock.patch('numpy.loadtxt', return_value=rfi_ranges):
                parameters = pipelineprocs.finalise_parameters(
                    self.parameters, self.telstate_l0, 4, 2, rfi_filename='my_rfi_file')

        channel_mask = np.zeros((4096,), np.bool_)
        for r in rfi_ranges:
            idx = np.where((self.expected_freqs_all < r[1] * 1e6) &
                           (self.expected_freqs_all >= r[0] * 1e6))[0]
            channel_mask[idx] = 1

        # Check mask is False on long baselines and auto-correlations
        mask = channel_mask[np.newaxis, :, np.newaxis, np.newaxis]
        mask = np.broadcast_to(mask, (1, 4096, 1, 10))
        bl_mask = mask.copy()
        bls_lookup = parameters['bls_lookup']
        long_bls = np.where((bls_lookup[:, 0] == 1) ^ (bls_lookup[:, 1] == 1))[0]
        bl_mask[..., long_bls] = False
        auto_bls = np.where((bls_lookup[:, 0] == bls_lookup[:, 1]))[0]
        bl_mask[..., auto_bls] = False
        np.testing.assert_array_equal(bl_mask[:, 2048:3072], parameters['rfi_mask'])

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
        self.parameters['k_echan'] = 1315765625  # chan 2200 in L-band 4K
        with self.assertRaises(ValueError):
            pipelineprocs.finalise_parameters(self.parameters, self.telstate_l0, 4, 2, None)

    def test_channel_range_spans_servers(self):
        self.parameters['k_echan'] = 1498208985  # chan 3073 in L-band 4K
        with self.assertRaises(ValueError):
            pipelineprocs.finalise_parameters(self.parameters, self.telstate_l0, 4, 2, None)

    def test_unexpected_parameters(self):
        self.parameters['foo'] = 'bar'
        with self.assertRaises(ValueError):
            pipelineprocs.finalise_parameters(self.parameters, self.telstate_l0, 4, 2, None)


class TestCreateModel(unittest.TestCase):
    """Tests for 'katsdpcal.pipelineprocs.get_model'"""
    def setUp(self):
        self.lsm_dir = tempfile.mkdtemp()
        self.file_model = ('S0, radec, 08:25:26.87, -50:10:38.49, (800 1712 5 6 7 8)')

        self.target = katpoint.Target('cal | alias, radec target, '
                                      '08:25:26.87, -50:10:38.49, '
                                      '(800.0 8400.0 1 2 3)')

        self.addCleanup(shutil.rmtree, self.lsm_dir)

        # create file whose name doesn't match target
        fname = os.path.join(self.lsm_dir, 'not.txt')
        f = open(fname, 'w')
        f.close()

    def test_one_file(self):
        """If one model file matching the target name/alias exists, use it as a model"""
        # create model file with name that matches target name
        for name in ['cal', 'alias']:
            fname = os.path.join(self.lsm_dir, name + '.txt')
            with open(fname, 'w') as f:
                print(self.file_model, file=f)

            expected_params = ['S0, radec, 8:25:26.87, -50:10:38.5, (800.0 1712.0'
                               ' 5.0 6.0 7.0 8.0)']

            model_params, model_file = pipelineprocs.get_model(self.target, self.lsm_dir)
            self.assertEqual(fname, model_file)
            np.testing.assert_equal(model_params, expected_params)

            os.remove(fname)

    def test_nofile_model(self):
        """If no matching model file exists, use the target model"""
        expected_params = [self.target.description]

        model_params, model_file = pipelineprocs.get_model(self.target, self.lsm_dir)
        self.assertEqual([], model_file)
        np.testing.assert_equal(model_params, expected_params)

    def test_nomodel(self):
        """If no model file and no target model, return None"""
        target = katpoint.Target('cal | cal_alias, radec target, 08:25:26.87, -50:10:38.49')
        model_params, model_file = pipelineprocs.get_model(target, self.lsm_dir)
        self.assertEqual([], model_file)
        self.assertEqual(None, model_params)

    def test_two_files(self):
        """If two files match target name, confirm it returns model from one of them"""
        for name in ['cal', 'alias']:
            fname = os.path.join(self.lsm_dir, name + '.txt')
            with open(fname, 'w') as f:
                print(self.file_model, file=f)

        expected_params = ['S0, radec, 8:25:26.87, -50:10:38.5, (800.0 1712.0'
                           ' 5.0 6.0 7.0 8.0)']
        model_params, model_file = pipelineprocs.get_model(self.target, self.lsm_dir)
        np.testing.assert_equal(model_params, expected_params)
