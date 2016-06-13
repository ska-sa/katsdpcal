"""
Pipeline procedures for MeerKAT calibration pipeline
====================================================
"""

import numpy as np

# for model files
import glob

import logging
logger = logging.getLogger(__name__)

#--------------------------------------------------------------------------------------------------
#--- Telescope State interactions
#--------------------------------------------------------------------------------------------------

def clear_ts(ts):
    """
    Clear the TS.

    Inputs
    ======
    ts: Telescope State
    """
    try:
        for key in ts.keys(): ts.delete(key)
    except AttributeError:
        # the Telescope State is empty
        pass

def init_ts(ts, param_dict, clear=False):
    """
    Initialises the telescope state from parameter dictionary.
    Parameters from the parameter dictionary are only added to the TS the
    the parameter is not already in the TS.

    Inputs
    ======
    ts : Telescope State
    param_dict : dictionary of parameters
    clear : clear ts before initialising
    """ 

    if clear:
        # start with empty Telescope State
        clear_ts(ts)

    # populate ts with parameters 
    #   parameter only added if it is missing from the TS
    for key in param_dict.keys(): 
        if key not in ts: ts.add(key, param_dict[key])

def ts_from_file(ts, filename):
    """
    Initialises the telescope state from parameter file

    Inputs
    ======
    ts : Telescope State
    filename : parameter file

    Notes
    =====
    * Parameter file uses colons (:) for delimiters
    * Parameter file uses hashes (#) for comments
    * Missing parameters are set to empty strings ''
    """
    param_list = np.genfromtxt(filename,delimiter=':',dtype=np.str, comments='#', missing_values='')
    param_dict = {}
    for key, value in param_list:
        try:
            # integer?
            param_value = int(value)                
        except ValueError:
            try:
                # float?
                param_value = float(value)
            except ValueError:
                # keep as string, strip whitespace
                param_value = value.strip()

        param_dict[key.strip()] = param_value

    init_ts(ts, param_dict)

def setup_ts(ts):
    """
    Set up the telescope state for pipeline use.
    In general, the calibration parameters are mutable.
    Only subarray characteristics (e.g. antenna_mask) are immutable.
    Inputs
    ======
    ts : Telescope State

    Notes
    =====
    Assumed starting ts entries:
    antenna_mask

    Assumed ending ts entries
    antenna_mask          - list or csv string of antennas present in the data, immutable
    cal_antlist           - list of antennas present in the data
    cal_preferred_refants - ordered list of refant preference
    cal_refant            - reference antenna
    """  

    # ensure that antenna_mask is list of strings, not single csv string
    if isinstance(ts.antenna_mask, str):
        antlist = [ant.strip() for ant in ts.antenna_mask.split(',')]
        ts.delete('antenna_mask')
        ts.add('antenna_mask',antlist,immutable=True)
    # cal_antlist
    #   this should not be pre-set (determine from antenna_mask, which is pre-set)
    ts.add('cal_antlist',ts.antenna_mask)

    # cal_preferred_refants
    if 'cal_preferred_refants' not in ts:
        ts.add('cal_preferred_refants',ts.cal_antlist,)
    else:
        # reduce the preferred antenna list to only antennas present in can_antlist
        preferred = [ant for ant in ts.cal_preferred_refants if ant in ts.cal_antlist]
        if preferred != ts.cal_preferred_refants:
            if preferred == []:
                ts.delete('cal_preferred_refants')
                ts.add('cal_preferred_refants',ts.cal_antlist)
            else:
                ts.delete('cal_preferred_refants')
                ts.add('cal_preferred_refants',preferred)

    # cal_refant
    if 'cal_refant' not in ts:
        ts.add('cal_refant',ts.cal_preferred_refants[0]) 
    else:
        if ts.cal_refant not in ts.cal_antlist:
            ts.delete('cal_refant')
            ts.add('cal_refant',ts.cal_preferred_refants[0])

    # temporary fix:
    #     the parameter files are surrently set up for 4k mode use.
    #     if we have 32k channels, scale the solution channel ranges accordingly
    if ts.cbf_n_chans > 4096: #i.e. 32k mode
        key_list = ['cal_k_bchan','cal_k_echan','cal_g_bchan','cal_g_echan']
        for k in key_list:
            cal_value = ts[k]
            ts.delete(k)
            ts.add(k,cal_value*8)

def get_model(name, lsm_dir_list = []):
    """
    Get a sky model from a text file.
    The name of the text file must incorporate the name of the source.

    Inputs
    ======
    name : name of source, string
    lsm_dir : directory containing the source model txt file

    Returns
    =======
    model_components : numpy recarray of sky model component parameters
    """
    if not isinstance(lsm_dir_list,list): lsm_dir_list = [lsm_dir_list]
    # default to check the current directory first
    lsm_dir_list.append('.')

    # iterate through the list from the end so the model from the earliest directory in the list is used
    model_file = []
    for lsm_dir in reversed(lsm_dir_list):
        model_file_list = glob.glob('{0}/*{1}*.txt'.format(lsm_dir,name))
        # ignore tilde ~ backup files
        model_file_list = [f for f in model_file_list if f[-1]!='~']

        if len(model_file_list) == 1:
            model_file = model_file_list[0]
        elif len(model_file_list) > 1:
            # if there are more than one model files for the source IN THE SAME DIRECTORY, raise an error
            raise ValueError('More than one possible sky model file for {0}: {1}'.format(name, model_file_list))

    # if there is not model file, return None
    if model_file == []: return None

    model_dtype = [('tag','S4'),('name','S16'),('RA','S24'),('dRA','S8'),('DEC','S24'),('dDEC','S8'),
        ('a0','f16'),('a1','f16'),('a2','f16'),('a3','f16'),('fq','f16'),('fu','f16'),('fv','f16')]
    model_components = np.genfromtxt(model_file,delimiter=',',dtype=model_dtype)
    return model_components

