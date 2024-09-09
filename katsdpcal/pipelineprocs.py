"""
Pipeline procedures for MeerKAT calibration pipeline
====================================================
"""

import glob    # for model files
import logging
import argparse
import math

import attr
import numpy as np
import astropy.units as u
import requests

import katpoint

import katsdpmodels.fetch.requests
import katsdpmodels.band_mask
import katsdpmodels.rfi_mask

from collections import OrderedDict
from katsdpcalproc import calprocs

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# --- General pipeline interactions
# -------------------------------------------------------------------------------------------------

BAND_MAP = {'u': 'UHF', 'l': 'L', 's': 'S', 'x': 'X'}


class Converter:
    """Converts parameters between representations.

    A parameter starts with a requested value, which is directly given by the
    user e.g. in a config file, and which is "global" (applies to the whole
    band, not the part handled by a server). It is transformed into a "local"
    (server-specific) representation for `parameters` dictionaries. From there,
    it is turned back into a global representation that is stored in the
    telescope state.
    """
    @staticmethod
    def to_local(value, parameters):
        return value

    @staticmethod
    def to_telstate(value, parameters):
        return value


class ChannelConverter(Converter):
    @staticmethod
    def to_local(value, parameters):
        return value - parameters['channel_slice'].start

    @staticmethod
    def to_telstate(value, parameters):
        return value + parameters['channel_slice'].start


class FreqChunksConverter(Converter):
    @staticmethod
    def to_local(value, parameters):
        # Round up to ensure we always have a positive value
        return int(math.ceil(value / parameters['servers']))

    @staticmethod
    def to_telstate(value, parameters):
        return value * parameters['servers']


class AttrConverter(Converter):
    """Converts to telstate by taking an attribute of the value."""
    def __init__(self, field):
        self._field = field

    def to_telstate(self, value, parameters):
        return getattr(value, self._field)


@attr.s
class Parameter:
    name = attr.ib()
    help = attr.ib()
    type = attr.ib()
    metavar = attr.ib(default=None)
    default = attr.ib(default=None)
    telstate = attr.ib(default=None)   # Set to true or false to override default, or a name
    converter = attr.ib(default=Converter)


def comma_list(type_):
    def convert(value):
        if value == '':
            return []
        parts = value.split(',')
        return [type_(part.strip()) for part in parts]

    return convert


# Parameters that the user can set directly, in units of channels
USER_PARAMS_CHANS = [
    # delay calibration
    Parameter('k_solint', 'nominal pre-k g solution interval, seconds', float),
    Parameter('k_chan_sample', 'sample every nth channel for pre-K BP soln', int),
    Parameter('k_bchan', 'first channel for K fit, (channel no)', int, converter=ChannelConverter),
    Parameter('k_echan', 'last channel for K fit, (channel no)', int, converter=ChannelConverter),
    Parameter('kcross_chanave', 'number of channels to average together to kcross solution', int),
    # bandpass calibration
    Parameter('bp_solint', 'nominal pre-bp g solution interval, seconds', float),
    # gain calibration
    Parameter('g_solint', 'nominal g solution interval, seconds', float),
    Parameter('g_bchan', 'first channel for g fit, (channel no)', int,
              converter=ChannelConverter),
    Parameter('g_echan', 'last channel for g fit, (channel no)', int,
              converter=ChannelConverter),
    # Flagging
    Parameter('rfi_calib_nsigma', 'number of sigma to reject outliers for calibrators', float),
    Parameter('rfi_targ_nsigma', 'number of sigma to reject outliers for targets', float),
    Parameter('rfi_average_freq',
              'amount to average in frequency before flagging, (channels)', int),
    Parameter('rfi_windows_freq',
              'size of windows for SumThreshold on original, unaveraged data, (channels)',
              comma_list(int)),
    Parameter('rfi_targ_spike_width_freq',
              '1sigma frequency width of smoothing Gaussian on final target, (channels)', int),
    Parameter('rfi_calib_spike_width_freq',
              '1sigma frequency width of smoothing Gaussian on calibrators, (channels)', int),
    Parameter('rfi_spike_width_time',
              '1sigma time width of smoothing Gaussian (in seconds)', float),
    Parameter('rfi_extend_freq', 'convolution width in frequency to extend flags, (channels)', int),
    Parameter('rfi_freq_chunks', 'no of chunks to divide band into when estimating noise', int,
              converter=FreqChunksConverter),
    Parameter('array_position', 'antenna object for the array centre', katpoint.Antenna,
              converter=AttrConverter('description')),
    Parameter('bcross_sky_knots', 'knots for bcross_sky spline, in MHz', comma_list(float),
              telstate=False),
    Parameter('bcross_sky_coefs', 'coefs for bcross_sky spline fit across frequency',
              comma_list(float), telstate=False),
    Parameter('bcross_sky_k', 'degree of spline fit for bcross_sky', int, telstate=False)
]


# Parameters that the user can set directly, in units of Hz/MHz
USER_PARAMS_FREQS = [
    Parameter('k_bfreq', 'start frequency for k fit per subband, (MHz)', comma_list(float)),
    Parameter('k_efreq', 'stop frequency for k fit per subband, (MHz)', comma_list(float)),
    Parameter('g_bfreq', 'start frequency for g fit per subband, (MHz)', comma_list(float)),
    Parameter('g_efreq', 'stop frequency for g fit per subband, (MHz)', comma_list(float)),
    Parameter('subband_bfreq', 'start frequency range corresponding to parameters'
              ' supplied per frequency subband, (MHz)', comma_list(float), telstate=False),
    Parameter('rfi_average_hz', 'amount to average in frequency before flagging, (Hz)', float),
    Parameter('rfi_windows_post_average',
              'size of windows for SumThreshold on frequency averaged data, (channels)',
              comma_list(int)),
    Parameter('rfi_targ_spike_width_hz',
              '1sigma frequency width of smoothing Gaussian on final target, (Hz)', float),
    Parameter('rfi_calib_spike_width_hz',
              '1sigma frequency width of smoothing Gaussian on calibrators, (Hz)', float),
    Parameter('rfi_extend_hz', 'convolution width in frequency to extend flags, (Hz)', float)
]


# Parameters that the user cannot set directly (the type is not used)
COMPUTED_PARAMETERS = [
    Parameter('refant_index', 'index of refant in antennas', int),
    Parameter('antenna_names', 'antenna names', list, telstate='antlist'),
    Parameter('antennas', 'antenna objects', list),
    Parameter('bls_ordering', 'list of baselines', list, telstate=True),
    Parameter('pol_ordering', 'list of polarisations', list, telstate=True),
    Parameter('bls_pol_ordering', 'list of polarisation products', list),
    Parameter('bls_lookup', 'list of baselines as indices into antennas', list),
    Parameter('channel_freqs', 'frequency of each channel in Hz, for this server', np.ndarray),
    Parameter('channel_freqs_all', 'frequency of each channel in Hz, for all servers', np.ndarray),
    Parameter('channel_slice', 'portion of channels handled by this server', slice),
    Parameter('product_names', 'names to use in telstate for solutions', dict),
    Parameter('product_B_parts', 'number of separate keys forming bandpass solution', int,
              telstate=True),
    Parameter('servers', 'number of parallel servers', int),
    Parameter('server_id', 'identity of this server (zero-based)', int),
    Parameter('bcross_sky_spline', 'spline fit to bcross_sky across frequency (in MHz)', tuple,
              telstate=True),
    Parameter('reset_solution_stores', 'reset the solution stores between capture blocks',
              bool, telstate=True)
]


def parameters_from_file(filename):
    """Load a set of parameters from a config file, returning a dict.

    The file has the format :samp:`{key}: {value}`. Hashes (#) introduce
    comments.
    """
    rows = np.loadtxt(filename, delimiter=':', dtype=str, comments='#')
    raw_params = {key.strip(): value.strip() for (key, value) in rows}
    param_dict = {}
    for parameter in USER_PARAMS_FREQS + USER_PARAMS_CHANS:
        if parameter.name in raw_params:
            param_dict[parameter.name] = parameter.type(raw_params[parameter.name])
            del raw_params[parameter.name]
    if raw_params:
        raise ValueError('Unknown parameters ' + ', '.join(raw_params.keys()))
    return param_dict


def parameters_from_argparse(namespace):
    """Extracts those parameters that are present in an argparse namespace"""
    param_dict = {}
    for parameter in USER_PARAMS_CHANS + USER_PARAMS_FREQS:
        if parameter.name in vars(namespace):
            param_dict[parameter.name] = getattr(namespace, parameter.name)
    return param_dict


def register_argparse_parameters(parser):
    """Add command-line arguments corresponding to parameters"""
    for parameter in USER_PARAMS_CHANS + USER_PARAMS_FREQS:
        # Note: does NOT set default=. Defaults are resolved only after
        # processing both command-line arguments and config files.
        parser.add_argument('--' + parameter.name.replace('_', '-'),
                            help=parameter.help,
                            type=parameter.type,
                            default=argparse.SUPPRESS,
                            metavar=parameter.metavar)


def _check_blank_freqwin(parameters):
    blank_freqwin = []
    for prefix in ['k', 'g']:
        try:
            if parameters[prefix + '_bfreq'] == parameters[prefix + '_efreq'] == []:
                blank_freqwin.append(prefix)
        # in case solution windows are provided per channel, e.g 'g_bchan'
        except KeyError:
            pass
    return blank_freqwin


def _get_rfi_mask(telstate_l0):
    with katsdpmodels.fetch.requests.TelescopeStateFetcher(telstate_l0) as fetcher:
        rfi_mask_model_key = telstate_l0.join('model', 'rfi_mask', 'fixed')
        try:
            rfi_mask_model = fetcher.get(rfi_mask_model_key,
                                         katsdpmodels.rfi_mask.RFIMask)
            return rfi_mask_model
        except (requests.ConnectionError, katsdpmodels.models.ModelError) as exc:
            logger.warning('Failed to load rfi_mask model: %s', exc)
            return None


def _get_band_mask(telstate_l0):
    with katsdpmodels.fetch.requests.TelescopeStateFetcher(telstate_l0) as fetcher:
        correlator_stream = telstate_l0['src_streams'][0]
        f_engine_stream = telstate_l0.view(correlator_stream, exclusive=True)['src_streams'][0]
        telstate_cbf = telstate_l0.view(f_engine_stream, exclusive=True)

        band_mask_model_key = telstate_l0.join('model', 'band_mask', 'fixed')
        try:
            band_mask_model = fetcher.get(band_mask_model_key,
                                          katsdpmodels.band_mask.BandMask,
                                          telstate=telstate_cbf)
            return band_mask_model
        except (requests.ConnectionError, katsdpmodels.models.ModelError) as exc:
            logger.warning('Failed to load band_mask model: %s', exc)
            return None


def get_static_mask(telstate_l0, channel_freqs, length=100.0):
    """Get the static mask for the given frequencies and baseline length.

    Parameters:
    -----------
    telstate_l0 : :class:`katsdptelstate.TelescopeState`
        Telescope state with a view of the L0 attributes
    channel_freqs : :class:`~astropy.units.Quantity`
        frequencies
    length, optional : float
        baseline length in m
    """
    rfi_mask = _get_rfi_mask(telstate_l0)
    band_mask = _get_band_mask(telstate_l0)

    bandwidth = telstate_l0['bandwidth']
    channel_width = bandwidth / telstate_l0['n_chans']
    if rfi_mask is None:
        static_mask = np.zeros((1, len(channel_freqs)))
    else:
        static_mask = rfi_mask.is_masked(channel_freqs, length * u.m, channel_width * u.Hz)

    if band_mask is not None:
        center = telstate_l0['center_freq']
        band_spw = katsdpmodels.band_mask.SpectralWindow(bandwidth * u.Hz, center * u.Hz)
        mask = band_mask.is_masked(band_spw, channel_freqs, channel_width * u.Hz)
        static_mask |= mask
    return static_mask


def _consecutive(index, stepsize=1):
    """Partition index into list of arrays of consecutive indices.

    From https://stackoverflow.com/a/7353335
    """
    return np.split(index, np.where(np.diff(index) != stepsize)[0]+1)


def parameters_for_blank_freqwin(parameters, channel_freqs, static_mask, prefix):
    """Set the solution interval parameters (e.g k_bfreq) for the solutions in
    `prefix` by selecting an unmasked region of the spectrum.

    Set the solution interval parameters (e.g. k_bfreq). The interval is selected by
    finding the largest interval of channels which is not masked by the static mask
    and fits within a single server. A default number of channels is selected from
    the center of this interval. If the interval is narrower than the default
    width the whole interval is used.

    Parameters
    ----------
    parameters : dict
        Dictionary mapping parameter names from :const:`USER_PARAMS_CHANS` or
        :const:`USER_PARAMS_FREQS`
    telstate_l0 : :class:`katsdptelstate.TelescopeState`
        Telescope state with a view of the L0 attributes
    static_mask : :class:`np.ndarray`
        static channel mask
    prefix : list of str
        list of solution types
    """

    n_chans = len(channel_freqs)
    servers = parameters['servers']

    chan_window = []
    # For each server get the windows of consecutive unmasked channels
    # Iterate in reverse to select the lower frequency range in the case of a tie
    # between servers, for steep spectrum gain calibrators this might result in a
    # tiny bit more signal.
    for server_id in range(servers - 1, -1, -1):
        channel_slice = slice(n_chans * server_id // servers,
                              n_chans * (server_id + 1) // servers)
        index_unmasked = np.where(~static_mask[channel_slice])[0]
        con_unmasked = _consecutive(index_unmasked + n_chans * server_id // servers)
        # Update the window selection if it is larger than the previously selected window.
        for window in con_unmasked:
            if len(window) >= len(chan_window):
                chan_window = window

    # This default gain window size for narrowband (chan_width = 3.265 kHz * 6400 = 20.9 MHz) is
    # half the size used in wideband (41.8 MHz) to allow it to fit within a single server
    # in the default 32k, 4 server case
    max_chans_per_server = len(channel_freqs) // servers
    default_width_chans = min(6400, max_chans_per_server)
    len_window = len(chan_window)
    if len_window >= default_width_chans:
        bchan = chan_window[len_window // 2] - default_width_chans // 2
        echan = chan_window[len_window // 2] + default_width_chans // 2

    # Select the full interval if it is narrower than the default of 6400 chans
    else:
        bchan = chan_window[0]
        echan = chan_window[-1]
        actual_sol_bandwidth = (channel_freqs[echan] - channel_freqs[bchan]).to(u.MHz)
        default_sol_bandwidth = (channel_freqs[default_width_chans] - channel_freqs[0]).to(u.MHz)
        logger.warning("The selected gain window is narrower (%.3f MHz)"
                       " than the default (%.3f MHz)", actual_sol_bandwidth.value,
                       default_sol_bandwidth.value)

    # Parameters are in units of MHz
    for p in prefix:
        parameters[p + '_bfreq'] = [channel_freqs[bchan].to_value(u.MHz)]
        parameters[p + '_efreq'] = [channel_freqs[echan].to_value(u.MHz)]
        logger.info('The %s solution interval is set to %.3f - %.3f MHz',
                    p, channel_freqs[bchan].to_value(u.MHz), channel_freqs[echan].to_value(u.MHz))


def finalise_parameters(parameters, telstate_l0, servers, server_id):
    """Set the defaults and computed parameters in `parameters`.

    On input, `parameters` contains keys from :const:`USER_PARAMS_CHANS` or
    :const:`USER_PARAMS_FREQS`. Keys from `USER_PARAMS_CHANS` can be missing if there
    is a default or there is an equivalent frequency based parameter in `USER_PARAMS_FREQS`.
    On return, parameters in :const:`COMPUTED_PARAMETERS` are filled in and
    parameters given in :const:`USER_PARAMS_FREQS` are converted to channel based parameters
    in :const:`USER_PARAMS_CHANS`

    Parameters
    ----------
    parameters : dict
        Dictionary mapping parameter names from :const:`USER_PARAMS_CHANS` or
        :const:`USER_PARAMS_FREQS`
    telstate_l0 : :class:`katsdptelstate.TelescopeState`
        Telescope state with a view of the L0 attributes
    servers : int
        Number of cooperating servers
    server_id : int
        Number of this server amongst `servers` (0-based)

    Raises
    ------
    ValueError
        - if some element of `parameters` is not set and there is no default
        - if `servers` doesn't divide into the number of channels
        - if any unknown parameters are set in `parameters`
        - if `server_id` is out of range
        - if a channel range for a solver crosses server boundaries
    """
    n_chans = telstate_l0['n_chans']
    if not 0 <= server_id < servers:
        raise ValueError('Server ID {} is out of range [0, {})'.format(server_id, servers))
    if n_chans % servers != 0:
        raise ValueError('Number of channels ({}) is not a multiple of number of servers ({})'
                         .format(n_chans, servers))
    center_freq = telstate_l0['center_freq']
    bandwidth = telstate_l0['bandwidth']
    channel_freqs = (center_freq + (bandwidth / n_chans) * (np.arange(n_chans) - n_chans // 2)) \
        * u.Hz
    channel_slice = slice(n_chans * server_id // servers,
                          n_chans * (server_id + 1) // servers)
    parameters['channel_freqs_all'] = channel_freqs.value
    parameters['channel_freqs'] = channel_freqs.value[channel_slice]
    parameters['channel_slice'] = channel_slice
    parameters['servers'] = servers
    parameters['server_id'] = server_id
    parameters['reset_solution_stores'] = True

    baselines = telstate_l0['bls_ordering']
    ants = set()
    for a, b in baselines:
        ants.add(a[:-1])
        ants.add(b[:-1])
    antenna_names = sorted(ants)
    _, bls_ordering, bls_pol_ordering = calprocs.get_reordering(antenna_names,
                                                                telstate_l0['bls_ordering'])
    antennas = [katpoint.Antenna(telstate_l0['{0}_observer'.format(ant)]) for ant in antenna_names]
    parameters['antenna_names'] = antenna_names
    parameters['antennas'] = antennas
    parameters['bls_ordering'] = bls_ordering
    parameters['bls_pol_ordering'] = bls_pol_ordering
    parameters['pol_ordering'] = [p[0] for p in bls_pol_ordering if p[0] == p[1]]
    parameters['bls_lookup'] = calprocs.get_bls_lookup(antenna_names, bls_ordering)

    # array_position can be set by user, but if not specified we need to
    # get the default from one of the antennas.
    if 'array_position' not in USER_PARAMS_CHANS:
        parameters['array_position'] = katpoint.Antenna(
            'array_position', *antennas[0].ref_position_wgs84)

    # check whether the solution windows are blank/not set for the k and g solutions
    # select a window if it is not provided. This is to support narrowband where the
    # frequency interval is variable.
    blank_freqwin = _check_blank_freqwin(parameters)

    if any(blank_freqwin):
        # select missing solution interval
        static_mask = get_static_mask(telstate_l0, channel_freqs)
        parameters_for_blank_freqwin(parameters, channel_freqs, static_mask, blank_freqwin)

    # select appropriate parameters for the given frequency range
    parameters_for_freq(parameters, channel_freqs)

    # Convert frequency parameters from Hz/MHz to channels
    parameters_to_channels(parameters, channel_freqs)

    # Set the defaults. Needs to be done after dealing with array_position
    # but before interpreting preferred_refants.
    for parameter in USER_PARAMS_CHANS:
        name = parameter.name
        if name not in parameters:
            if parameter.default is None:
                raise ValueError('No value specified for ' + name)
            elif isinstance(parameter.default, attr.Factory):
                parameters[name] = parameter.default.factory()
            else:
                parameters[name] = parameter.default

    # Convert all parameters to server-local values
    global_parameters = parameters.copy()
    for parameter in USER_PARAMS_CHANS:
        name = parameter.name
        parameters[name] = parameter.converter.to_local(parameters[name], global_parameters)

    parameters['refant_index'] = None
    for prefix in ['k', 'g']:
        bchan = parameters[prefix + '_bchan']
        echan = parameters[prefix + '_echan']
        if echan <= bchan:
            raise ValueError('{0}_echan <= {0}_bchan ({1} <= {2})'
                             .format(prefix, bchan, echan))
        if bchan // (n_chans // servers) != (echan - 1) // (n_chans // servers):
            raise ValueError('{} channel range [{}, {}) spans multiple servers'
                             .format(prefix, bchan, echan))

    parameters['product_names'] = {
        'G': 'product_G',
        'SNR_G': 'product_SNR_G',
        'K': 'product_K',
        'SNR_K': 'product_SNR_K',
        'KCROSS': 'product_KCROSS',
        'KCROSS_DIODE': 'product_KCROSS_DIODE',
        'B': 'product_B{}'.format(server_id),
        'SNR_B': 'product_SNR_B{}'.format(server_id),
        'BCROSS_DIODE': 'product_BCROSS_DIODE{}'.format(server_id),
        'BCROSS_DIODE_SKY': 'product_BCROSS_DIODE_SKY{}'.format(server_id)
    }
    parameters['product_B_parts'] = servers

    # Convert spline knots, coefs and degrees into spline tuple to store in telstate
    parameters['bcross_sky_spline'] = (np.array(parameters['bcross_sky_knots']),
                                       np.array(parameters['bcross_sky_coefs']),
                                       parameters['bcross_sky_k'])

    # Sanity check: make sure we didn't set any parameters for which we don't
    # have a description.
    valid_parameters = set(parameter.name for parameter in
                           USER_PARAMS_CHANS + COMPUTED_PARAMETERS)
    for key in parameters:
        if key not in valid_parameters:
            raise ValueError('Unexpected parameter {}'.format(key))
    return parameters


def parameters_to_channels(parameters, channel_freqs):
    """Convert certain parameters from units of Hz/MHz to channels

    If a channel based parameter has already been supplied use it, else
    calculate it from the frequency based parameter.

    Parameters
    ----------
    parameters : dict
        Dictionary mapping parameter names from :const:`USER_PARAMS_CHANS` +
        :const:`USER_PARAMS_FREQS`
    channel_freqs : :class:`~astropy.units.Quantity`
        frequency of channels

    Raises
    ------
    ValueError
        - if a parameter is not set using either the channel or frequency
          convention
    """
    # dict to convert channel based parameter names to freq based names
    # order is NB because converting some keys to channels requires other
    # parameters to already be available in channels.
    chans_to_freq = OrderedDict()
    for param in USER_PARAMS_CHANS:
        if param.name.endswith('chan'):
            chans_to_freq.update({param.name : param.name.replace('chan', 'freq')})
        elif param.name.endswith('freq'):
            chans_to_freq.update({param.name : param.name.replace('freq', 'hz')})
    chans_to_freq.update({'rfi_windows_freq' : 'rfi_windows_post_average'})
    # check all parameters are provided as either Hz/MHz or chans, remove duplicate params
    for chan, freq in chans_to_freq.items():
        if chan in parameters:
            if freq in parameters:
                del parameters[freq]
                logger.info('Parameters %s and %s both set, using %s', chan, freq, chan)
        else:
            if freq not in parameters:
                raise ValueError('Parameters {} and {} both not set'.format(chan, freq))

    # convert frequency params to channel params
    chan_width = channel_freqs[1] - channel_freqs[0]
    for chan, freq in chans_to_freq.items():
        if freq in parameters:
            if freq.endswith('bfreq'):
                bfreq_hz = parameters[freq] * u.MHz
                parameters[chan] = channel_freqs.searchsorted(bfreq_hz)
            elif freq.endswith('efreq'):
                efreq_hz = parameters[freq] * u.MHz
                parameters[chan] = channel_freqs.searchsorted(efreq_hz) - 1
            elif freq in ['rfi_average_hz',
                          'rfi_targ_spike_width_hz',
                          'rfi_calib_spike_width_hz']:
                parameters[chan] = int(np.ceil(parameters[freq] / chan_width.value))
            elif freq == 'rfi_extend_hz':
                # 'rfi_extend_freq' must be odd no of channels
                odd_no_chans = np.ceil(parameters[freq] / chan_width.value) // 2 * 2 + 1
                parameters[chan] = int(max(3, odd_no_chans))
            elif freq == 'rfi_windows_post_average':
                # flagger expects threshold window sizes unaveraged channel widths
                pre_avg_windows = [int(w * parameters['rfi_average_freq'])
                                   for w in parameters[freq]]
                pre_avg_windows[0] = 1
                parameters[chan] = pre_avg_windows
            del parameters[freq]


def parameters_for_freq(parameters, channel_freqs):
    """Select the appropriate parameters for the given frequency setup.

    If solution interval parameters (e.g. k_bfreq) are supplied as a list,
    then select the appropriate value for the frequency setup of the observation
    """
    subband_keys = ['g_bfreq', 'g_efreq', 'k_bfreq', 'k_efreq']
    for key in subband_keys:
        if key not in parameters:
            continue
        if len(parameters[key]) > 1:
            try:
                subband_idx = max([i for i, s_freq in enumerate(parameters['subband_bfreq'])
                                   if min(channel_freqs) >= s_freq * u.MHz])
                parameters[key] = parameters[key][subband_idx]
            except KeyError:
                logger.error("If '%s' is a list of values,"
                             " a 'subband_bfreq' parameter is required", key)
                raise
        elif len(parameters[key]) == 1:
            parameters[key] = float(parameters[key][0])

    # remove parameter as it is no longer required
    if 'subband_bfreq' in parameters:
        del parameters['subband_bfreq']


def parameters_to_telstate(parameters, telstate_cal, l0_name):
    """Take certain parameters and store them in telstate for the benefit of consumers.

    As part of this process, local server values are converted back to global.

    The `telstate_cal` should be a view in the cal_name namespace.
    """
    for parameter in USER_PARAMS_CHANS:
        # Put them in unless explicitly set to False
        if parameter.telstate is None or parameter.telstate:
            if isinstance(parameter.telstate, str):
                key = parameter.telstate
            else:
                key = 'param_' + parameter.name
            value = parameter.converter.to_telstate(parameters[parameter.name], parameters)
            telstate_cal[key] = value
    for parameter in COMPUTED_PARAMETERS:
        # Only put them in if explicitly set to True
        if parameter.telstate:
            if isinstance(parameter.telstate, str):
                key = parameter.telstate
            else:
                key = parameter.name
            value = parameter.converter.to_telstate(parameters[parameter.name], parameters)
            telstate_cal[key] = value

    telstate_cal['stream_type'] = 'sdp.cal'
    # Transfer some keys from L0 stream to cal "stream", to help consumers compute
    # frequencies.
    telstate_l0 = telstate_cal.root().view(l0_name)
    for key in ['bandwidth', 'n_chans', 'center_freq']:
        telstate_cal[key] = telstate_l0[key]
    # Add the L0 stream name too, so that any other information can be found there.
    telstate_cal['src_streams'] = [l0_name]


def get_model(target, lsm_dir_list=[], sub_band='l'):
    """Get a sky model from a text file if one exists.

    The name of the text file must incorporate the name of the source.
    If multiple model files exist select the one whose suffix matches
    the band name. If no text file is found, return the target sky model.

    Parameters
    ----------
    target : :class:`katpoint.Target`
        target
    lsm_dir_list : list
        search path for the source model txt file
    sub_band : str
        sub_band name to search for in model txt file name

    Returns
    -------
    model_components : list of str
        list of katpoint description strings of sources
        in the sky model
    model_file : str
        name of model component file used
    """
    if not isinstance(lsm_dir_list, list):
        lsm_dir_list = [lsm_dir_list]
    # default to check the current directory first
    lsm_dir_list.append('.')

    allnames = [target.name] + target.aliases
    model_file = []
    # iterate through the list from the end so the model from the earliest
    # directory in the list is used
    for lsm_dir in reversed(lsm_dir_list):
        model_list = []
        # iterate over all aliases
        for name in allnames:
            model_list += glob.glob('{0}/*{1}*.txt'.format(glob.os.path.abspath(lsm_dir), name))

        if len(model_list) == 1:
            model_file = model_list[0]
        elif len(model_list) > 1:
            # if there is more than one model file for the source IN THE SAME
            # DIRECTORY use the one that matches the sub_band of the observation

            band_list = [m for m in model_list if m.endswith('_{0}.txt'.format(BAND_MAP[sub_band]))]
            if band_list:
                if len(band_list) > 1:
                    logger.warning(
                        'More than one possible sky model file for'
                        ' %s: %s, using %s', target.name, band_list, band_list[0])
                model_file = band_list[0]
            else:
                logger.warning(
                    'More than one possible sky model file for'
                    ' %s: %s, using %s', target.name, model_list, model_list[0])
                model_file = model_list[0]

    if model_file == []:
        if target.flux_model is not None:
            model_components = [target.description]
        else:
            model_components = None

    else:
        model_components = []
        with open(model_file) as file:
            for line in file:
                li = line.strip()
                if li and not li.startswith('#'):
                    model_components.append(li)

    return model_components, model_file
