"""
katsdpcal
======

Calibration pipeline package for MeerKAT.
"""

# Config file location
from pkg_resources import resource_filename
conf_dir = resource_filename(__name__, 'conf')

# Default parameter file
param_file = 'pipeline_parameters_kat7.txt'

# force module to import pyrap before spead
# (bug fix)
try:
    from pyrap.tables import table
except:
    print "Pyrap not found. Can't use MS simulator."
    # fake table
    class table:
        pass

import spead2
from spead2 import recv
from spead2 import send
