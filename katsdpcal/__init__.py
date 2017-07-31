"""
katsdpcal
======

Calibration pipeline package for MeerKAT.
"""

# Config file location
from pkg_resources import resource_filename
param_dir = resource_filename(__name__, 'conf/pipeline_params')
lsm_dir = resource_filename(__name__, 'conf/sky_models')
rfi_dir = resource_filename(__name__, 'conf/rfi_masks')

# BEGIN VERSION CHECK
# Get package version when locally imported from repo or via -e develop install
try:
    import katversion as _katversion
except ImportError:
    import time as _time
    __version__ = "0.0+unknown.{}".format(_time.strftime('%Y%m%d%H%M'))
else:
    __version__ = _katversion.get_version(__path__[0])
# END VERSION CHECK
