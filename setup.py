#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="katsdpcal",
    description="MeerKAT calibration pipeline",
    maintainer="Ruby van Rooyen",
    maintainer_email="ruby@ska.ac.za",
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
    setup_requires=["katversion"],
    python_requires=">=3.5",
    install_requires=[
        "numpy>=1.15", "scipy>=0.17", "numba>=0.19.0",
        "dask[array,distributed]>=0.17.0", "distributed>=1.12.0", "bokeh",
        "attrs", "sortedcontainers",
        "aiokatcp", "katpoint", "katdal", "katsdptelstate", "katsdpservices[argparse]",
        "katsdpsigproc", "spead2>=1.8.0", "docutils", "matplotlib>=2",
        "jsonschema"
    ],
    tests_require=["nose", "async_timeout", "asynctest"],
    use_katversion=True
)
