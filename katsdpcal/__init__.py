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
