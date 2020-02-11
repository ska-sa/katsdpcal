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

import katpoint

from collections import OrderedDict
from . import calprocs

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# --- General pipeline interactions
# -------------------------------------------------------------------------------------------------


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
              converter=AttrConverter('description'))
]


# Parameters that the user can set directly, in units of Hz/MHz
USER_PARAMS_FREQS = [
    Parameter('k_bfreq', 'start frequency for k fit, (MHz)', float),
    Parameter('k_efreq', 'stop frequency for k fit, (MHz)', float),
    Parameter('g_bfreq', 'start frequency for g fit, (MHz)', float),
    Parameter('g_efreq', 'stop frequency for g fit, (MHz)', float),
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
    Parameter('rfi_mask', 'boolean array of channels to mask', np.ndarray),
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
    Parameter('reset_solution_stores', 'reset the solution stores between capture blocks',
              bool, telstate=True)
]


def parameters_from_file(filename):
    """Load a set of parameters from a config file, returning a dict.

    The file has the format :samp:`{key}: {value}`. Hashes (#) introduce
    comments.
    """
    rows = np.loadtxt(filename, delimiter=':', dtype=np.str, comments='#')
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


def finalise_parameters(parameters, telstate_l0, servers, server_id, rfi_filename=None):
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
    rfi_filename : str
        Filename containing a pickled RFI mask

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
    channel_freqs = center_freq + (bandwidth / n_chans) * (np.arange(n_chans) - n_chans // 2)
    channel_slice = slice(n_chans * server_id // servers,
                          n_chans * (server_id + 1) // servers)
    parameters['channel_freqs_all'] = channel_freqs
    parameters['channel_freqs'] = channel_freqs[channel_slice]
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
    if rfi_filename is not None:
        mask_ranges = np.loadtxt(rfi_filename, comments='#', delimiter=',')
        rfi_mask = np.zeros((n_chans,), np.bool_)
        for r in mask_ranges:
            idx = np.where((channel_freqs < r[1] * 1e6) & (channel_freqs >= r[0] * 1e6))[0]
            rfi_mask[idx] = 1
        parameters['rfi_mask'] = rfi_mask
    else:
        parameters['rfi_mask'] = np.zeros((n_chans,), np.bool_)
    # Only use static channel mask on baselines greater than 1 and less than 1000 meters
    bl_mask = get_baseline_mask(parameters['bls_lookup'], antennas, (1, 1000))
    rfi_mask_shape = (1, n_chans, 1, len(parameters['bls_lookup']))
    rfi_mask = np.zeros(rfi_mask_shape, np.bool_)
    channel_mask = parameters['rfi_mask'][np.newaxis, :, np.newaxis, np.newaxis]
    rfi_mask[..., bl_mask] |= channel_mask
    # Mask edges of the band
    edge_chan = np.int(0.05 * n_chans)
    rfi_mask[:, :edge_chan] = True
    rfi_mask[:, -edge_chan:] = True
    parameters['rfi_mask'] = rfi_mask[:, channel_slice, :, :]

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
        'BCROSS_DIODE': 'product_BCROSS_DIODE{}'.format(server_id)
    }
    parameters['product_B_parts'] = servers

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
    channel_freqs : np.ndarray,
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
                bfreq_hz = parameters[freq] * 1e6
                parameters[chan] = channel_freqs.searchsorted(bfreq_hz)
            elif freq.endswith('efreq'):
                efreq_hz = parameters[freq] * 1e6
                parameters[chan] = channel_freqs.searchsorted(efreq_hz) - 1
            elif freq in ['rfi_average_hz',
                          'rfi_targ_spike_width_hz',
                          'rfi_calib_spike_width_hz']:
                parameters[chan] = int(np.ceil(parameters[freq] / chan_width))
            elif freq == 'rfi_extend_hz':
                # 'rfi_extend_freq' must be odd no of channels
                odd_no_chans = np.ceil(parameters[freq] / chan_width) // 2 * 2 + 1
                parameters[chan] = int(max(3, odd_no_chans))
            elif freq == 'rfi_windows_post_average':
                # flagger expects threshold window sizes unaveraged channel widths
                pre_avg_windows = [int(w * parameters['rfi_average_freq'])
                                   for w in parameters[freq]]
                pre_avg_windows[0] = 1
                parameters[chan] = pre_avg_windows
            del parameters[freq]


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


def get_baseline_mask(bls_lookup, ants, limits):
    """Mask indicating baselines with a length within `limits`.

    Compute a mask of the same length as bls_lookup that indicates
    whether the baseline length of the given correlation product is within
    the given limits (in meters).

    Parameters
    ----------
    bls_lookup : :class:`np.ndarray`, int(n_corrs, 2)
        indices of antenna pairs in each correlation product
    ants: list of :class:`katpoint.Antenna`
        antennas in bls_lookup
    limits : tuple
        (lower_limit, upper_limit) in meters
    """
    baseline_mask = np.zeros(bls_lookup.shape[0], dtype=np.bool)

    for prod, baseline in enumerate(bls_lookup):
        bl_vector = ants[baseline[0]].baseline_toward(ants[baseline[1]])
        bl_length = np.linalg.norm(bl_vector)
        if (bl_length >= limits[0]) & (bl_length < limits[1]):
            baseline_mask[prod] = True
    return baseline_mask


def get_model(target, lsm_dir_list=[]):
    """Get a sky model from a text file if one exists.

    The name of the text file must incorporate the name of the source.
    If no text file is found, return the target sky model.

    Parameters
    ----------
    target : :class:`katpoint.Target`
        target
    lsm_dir_list : list
        search path for the source model txt file

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
            # if there are more than one model files for the source IN THE SAME
            # DIRECTORY warn the user and use the first model
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
        with open(model_file) as file:
            model_cat = katpoint.Catalogue(file)
            model_components = [m.description for m in model_cat]
    return model_components, model_file
