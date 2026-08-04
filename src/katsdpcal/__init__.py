"""
katsdpcal
======

Calibration pipeline package for MeerKAT.
"""

# Config file location
import sys

if sys.version_info >= (3, 9):
    import importlib.resources as importlib_resources
else:
    import importlib_resources
resources = importlib_resources.files("katsdpcal")
param_dir = resources.joinpath("conf", "pipeline_params")
lsm_dir = resources.joinpath("conf", "sky_models")
docutils_dir = resources.joinpath("conf", "docutil_style")

# BEGIN VERSION CHECK
# Get package version when locally imported from repo or via -e develop install
try:
    from katsdpcal._version import version as __version__
except ImportError:
    import time as _time
    __version__ = "0.0+unknown.{}".format(_time.strftime('%Y%m%d%H%M'))
else:
    __version__ = _katversion.get_version(__path__[0])
# END VERSION CHECK
