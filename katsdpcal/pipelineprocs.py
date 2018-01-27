"""
Pipeline procedures for MeerKAT calibration pipeline
====================================================
"""

import glob    # for model files
import pickle
import logging
import argparse

import attr
import numpy as np

import katpoint

from . import calprocs


logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------------------------------
# --- General pipeline interactions
# -------------------------------------------------------------------------------------------------


@attr.s
class Parameter(object):
    name = attr.ib()
    help = attr.ib()
    type = attr.ib()
    metavar = attr.ib(default=None)
    default = attr.ib(default=None)
    telstate = attr.ib(default=True)    # Whether to store into telstate


def comma_list(type_):
    def convert(value):
        if value == '':
            return []
        parts = value.split(',')
        return [type_(part.strip()) for part in parts]

    return convert


USER_PARAMETERS = [
    Parameter('preferred_refants', 'preferred reference antennas', comma_list(str),
              default=attr.Factory(list)),
    # delay calibration
    Parameter('k_solint', 'nominal pre-k g solution interval, seconds', float),
    Parameter('k_chan_sample', 'sample every nth channel for pre-K BP soln', int),
    Parameter('k_bchan', 'first channel for K fit', int),
    Parameter('k_echan', 'last channel for K fit', int),
    Parameter('kcross_chanave', 'number of channels to average together to kcross solution', int),
    # bandpass calibration
    Parameter('bp_solint', 'nominal pre-bp g solution interval, seconds', float),
    # gain calibration
    Parameter('g_solint', 'nominal g solution interval, seconds', float),
    Parameter('g_bchan', 'first channel for g fit', int),
    Parameter('g_echan', 'last channel for g fit', int),
    # Flagging
    Parameter('rfi_calib_nsigma', 'number of sigma to reject outliers for calibrators', float),
    Parameter('rfi_targ_nsigma', 'number of sigma to reject outliers for targets', float),
    Parameter('rfi_windows_freq', 'size of windows for SumThreshold', comma_list(int)),
    Parameter('rfi_average_freq', 'amount to average in frequency before flagging', int),
    Parameter('rfi_targ_spike_width_freq',
              '1sigma frequency width of smoothing Gaussian on final target (in channels)', float),
    Parameter('rfi_calib_spike_width_freq',
              '1sigma frequency width of smoothing Gaussian on calibrators (in channels)', float),
    Parameter('rfi_spike_width_time',
              '1sigma time width of smoothing Gaussian (in seconds)', float),
    Parameter('rfi_extend_freq', 'convolution width in frequency to extend flags', int),
    Parameter('array_position', 'antenna object for the array centre', katpoint.Antenna)
]

# Parameters that the user cannot set directly (the type is not used)
COMPUTED_PARAMETERS = [
    Parameter('rfi_mask', 'boolean array of channels to mask', np.ndarray),
    Parameter('refant', 'selected reference antenna', katpoint.Antenna),
    Parameter('antlist', 'antenna names', list),    # TODO: eliminate, use antennas?
    Parameter('antennas', 'antenna objects', list),
    Parameter('bls_ordering', 'list of baselines', list),
    Parameter('pol_ordering', 'list of polarisations', list),
    Parameter('bls_lookup', 'list of baselines as indices into antlist', list),  # TODO: eliminate?
    Parameter('channel_freqs', 'frequency of each channel in Hz', np.ndarray)
]


def parameters_from_file(filename):
    """Load a set of parameters from a config file, returning a dict.

    The file has the format :samp:`{key}: {value}`. Hashes (#) introduce
    comments.
    """
    rows = np.loadtxt(filename, delimiter=':', dtype=np.str, comments='#')
    raw_params = {key.strip(): value.strip() for (key, value) in rows}
    param_dict = {}
    for parameter in USER_PARAMETERS:
        if parameter.name in raw_params:
            param_dict[parameter.name] = parameter.type(raw_params[parameter.name])
            del raw_params[parameter.name]
    if raw_params:
        raise ValueError('Unknown parameters ' + ', '.join(raw_params.keys()))
    return param_dict


def parameters_from_argparse(namespace):
    """Extracts those parameters that are present in an argparse namespace"""
    param_dict = {}
    for parameter in USER_PARAMETERS:
        if parameter.name in vars(namespace):
            param_dict[parameter.name] = getattr(namespace, parameter.name)
    return param_dict


def register_argparse_parameters(parser):
    """Add command-line arguments corresponding to parameters"""
    for parameter in USER_PARAMETERS:
        # Note: does NOT set default=. Defaults are resolved only after
        # processing both command-line arguments and config files.
        parser.add_argument('--' + parameter.name.replace('_', '-'),
                            help=parameter.help,
                            type=parameter.type,
                            default=argparse.SUPPRESS,
                            metavar=parameter.metavar)


def finalise_parameters(parameters, telstate_l0, rfi_filename=None):
    """Set the defaults and computed parameters in `parameters`.

    On input, `parameters` contains the keys in :const:`USER_PARAMETERS`. Keys
    may be missing if there is a default. On return, those in
    :const:`COMPUTED_PARAMETERS` are filled in too.

    It chooses the first antenna in `preferred_refants` that is in the antlist,
    or the first antenna from antlist if there are no matches.

    Parameters
    ----------
    parameters : dict
        Dictionary mapping parameter names from :const:`USER_PARAMETERS`
    telstate_l0 : :class:`katsdptelstate.TelescopeState`
        Telescope state with a view of the L0 attributes
    rfi_filename : str
        Filename containing a pickled RFI mask

    Raises
    ------
    ValueError
        if some element of `parameters` is not set and there is no default
    """
    baselines = telstate_l0['bls_ordering']
    ants = set()
    for a, b in baselines:
        ants.add(a[:-1])
        ants.add(b[:-1])
    antlist = sorted(ants)
    _, bls_ordering, pol_ordering = calprocs.get_reordering(antlist, telstate_l0['bls_ordering'])
    parameters['antlist'] = antlist
    parameters['antennas'] = [katpoint.Antenna(telstate_l0['{0}_observer'.format(ant)])
                              for ant in antlist]
    parameters['bls_ordering'] = bls_ordering
    parameters['pol_ordering'] = pol_ordering
    # TODO: eliminate, compute when needed?
    parameters['bls_lookup'] = calprocs.get_bls_lookup(antlist, bls_ordering)

    n_chans = telstate_l0['n_chans']
    center_freq = telstate_l0['center_freq']
    bandwidth = telstate_l0['bandwidth']
    parameters['channel_freqs'] = center_freq \
        + (bandwidth / n_chans) * (np.arange(n_chans) - n_chans / 2)

    # array_position can be set by user, but if not specified we need to
    # get the default from one of the antennas.
    if 'array_position' not in USER_PARAMETERS:
        parameters['array_position'] = katpoint.Antenna(
            'array_position', *parameters['antennas'][0].ref_position_wgs84)

    # Set the defaults. Needs to be done after dealing with array_position
    # but before interpreting preferred_refants.
    for parameter in USER_PARAMETERS:
        if parameter.name not in parameters:
            if parameter.default is None:
                raise ValueError('No value specified for ' + parameter.name)
            elif isinstance(parameter.default, attr.Factory):
                parameters[parameter.name] = parameter.default.factory()
            else:
                parameters[parameter.name] = parameter.default

    refant = None
    for ant in parameters['preferred_refants']:
        if ant in ants:
            refant = next(antenna for antenna in parameters['antennas'] if antenna.name == ant)
            break
    if refant is None:
        logger.warning('No antennas from the antenna mask in the preferred antenna list')
        refant = parameters['antennas'][0]
        logger.info('Reference antenna set to %s', refant.name)
    parameters['refant'] = refant
    if rfi_filename is not None:
        parameters['rfi_mask'] = pickle.load(open(rfi_filename))
    else:
        parameters['rfi_mask'] = None

    # Sanity check: make sure we didn't set any parameters for which we don't
    # have a description.
    valid_parameters = set(parameter.name for parameter in USER_PARAMETERS + COMPUTED_PARAMETERS)
    for key in parameters:
        if key not in valid_parameters:
            raise ValueError('Unexpected parameter {}'.format(key))

    return parameters


def get_model(name, lsm_dir_list=[]):
    """
    Get a sky model from a text file.
    The name of the text file must incorporate the name of the source.

    Parameters
    ----------
    name : str
        name of source
    lsm_dir_list : list
        search path for the source model txt file

    Returns
    -------
    model_components : :class:`numpy.recarray`
        sky model component parameters
    model_file : str
        name of model component file used
    """
    if not isinstance(lsm_dir_list, list):
        lsm_dir_list = [lsm_dir_list]
    # default to check the current directory first
    lsm_dir_list.append('.')

    # iterate through the list from the end so the model from the earliest
    # directory in the list is used
    model_file = []
    for lsm_dir in reversed(lsm_dir_list):
        model_file_list = glob.glob('{0}/*{1}*.txt'.format(glob.os.path.abspath(lsm_dir), name))
        # ignore tilde ~ backup files
        model_file_list = [f for f in model_file_list if f[-1] != '~']

        if len(model_file_list) == 1:
            model_file = model_file_list[0]
        elif len(model_file_list) > 1:
            # if there are more than one model files for the source IN THE SAME
            # DIRECTORY, raise an error
            raise ValueError(
                'More than one possible sky model file for {0}: {1}'.format(name, model_file_list))

    # if there is not model file, return model components as None
    if model_file == []:
        model_components = None
    else:
        model_dtype = [('tag', 'S4'), ('name', 'S16'),
                       ('RA', 'S24'), ('dRA', 'S8'), ('DEC', 'S24'), ('dDEC', 'S8'),
                       ('a0', 'f16'), ('a1', 'f16'), ('a2', 'f16'), ('a3', 'f16'),
                       ('fq', 'f16'), ('fu', 'f16'), ('fv', 'f16')]
        model_components = np.genfromtxt(model_file, delimiter=',', dtype=model_dtype)
    return model_components, model_file
