"""
Pipeline procedures for MeerKAT calibration pipeline
====================================================
"""

import numpy as np

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
    ts: Telescope State
    param_dict : dictionary of parameters
    clear : clear ts before initialising
    """ 

    if clear:
        # start with empty Telescope State
        clear_ts(ts)

    # populate ts with parameters 
    #   parameter only added if it is missing from the TS
    for key in param_dict.keys(): 
        if key not in ts: ts.add(key, param_dict[key], immutable=True)

def ts_from_file(ts, filename):
    """
    Initialises the telescope state from parameter file

    Inputs
    ======
    ts: Telescope State
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

    Inputs
    ======
    ts: Telescope State

    Notes
    =====
    Assumed starting ts entries:
    antenna_mask

    Assumed ending ts entries (all immutable)
    antenna_mask          - list or csv string of antennas present in the data
    cal_antlist           - list of antennas present in the data
    cal_preferred_refants - ordered list of refant preference
    cal_refant            - reference antenna
    """  

    # cal_antlist
    #   this should not be pre-set (determine from antenna_mask, which is pre-set)
    antlist = [ant.strip() for ant in ts.antenna_mask.split(',')] if isinstance(ts.antenna_mask, str) else ts.antenna_mask
    ts.add('cal_antlist',antlist,immutable=True)

    # cal_preferred_refants
    if 'cal_preferred_refants' not in ts:
        ts.add('cal_preferred_refants',ts.cal_antlist,immutable=True)
    else:
        # reduce the preferred antenna list to only antennas present in can_antlist
        preferred = [ant for ant in ts.cal_preferred_refants if ant in ts.cal_antlist]
        if preferred != ts.cal_preferred_refants:
            if preferred == []:
                ts.delete('cal_preferred_refants')
                ts.add('cal_preferred_refants',ts.cal_antlist,immutable=True)
            else:
                ts.delete('cal_preferred_refants')
                ts.add('cal_preferred_refants',preferred,immutable=True)

    # cal_refant
    if 'cal_refant' not in ts:
        ts.add('cal_refant',ts.cal_preferred_refants[0]) 
    else:
        if ts.cal_refant not in ts.cal_antlist:
            ts.delete('cal_refant')
            ts.add('cal_refant',ts.cal_preferred_refants[0])

