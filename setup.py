#!/usr/bin/env python
from setuptools import setup, find_packages

setup (
    name = "katsdpcal",
    description = "MeerKAT calibration pipeline",
    author = "Laura Richter",
    author_email = "laura@ska.ac.za",
    packages = find_packages(),
    package_data={'': ['conf/*/*']},
    include_package_data = True,
    scripts = [
        "scripts/reduction_script.py",
        "scripts/run_cal.py",
        "scripts/run_katsdpcal_sim.py",
        "scripts/sim_l1_receive.py",
        "scripts/sim_ts.py",
        "scripts/sim_data_stream.py",
        "scripts/create_test_data.py"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Astronomy",
    ],
    platforms = [ "OS Independent" ],
    keywords="kat kat7 meerkat ska",
    zip_safe = False,
    setup_requires = ["katversion"],
    use_katversion = True
)
