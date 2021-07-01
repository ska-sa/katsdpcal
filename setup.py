#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="katsdpcal",
    description="MeerKAT calibration pipeline",
    maintainer="MeerKAT SDP Team",
    maintainer_email="sdpdev+katsdpcal@ska.ac.za",
    packages=find_packages(),
    package_data={'': ['conf/*/*']},
    include_package_data=True,
    scripts=[
        "scripts/run_cal.py",
        "scripts/run_katsdpcal_sim.py",
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
    platforms=["OS Independent"],
    keywords="kat kat7 meerkat ska",
    zip_safe=False,
    python_requires=">=3.5",
    install_requires=[
        "numpy>=1.15", "scipy>=0.17", "numba>=0.49.0",
        "dask[array,distributed]>=1.1.0", "distributed>=2.2.0", "bokeh",
        "attrs", "sortedcontainers",
        "aiokatcp", "astropy", "async_timeout",
        "katpoint", "katdal", "katsdpmodels[requests]", "katsdptelstate",
        "katsdpservices[argparse,aiomonitor]", "katsdpsigproc", "spead2>=3.0.0",
        "docutils", "matplotlib>=2", "jsonschema"
    ],
    tests_require=["nose", "asynctest"],
    use_katversion=True
)
