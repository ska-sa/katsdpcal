"""
katcal
======

Calibration pipeline package for MeerKAT.
"""

import logging
logging.basicConfig(filename='pipeline.log',
                    format='%(asctime)s %(name)-24s %(levelname)-8s %(message)s',
                    datefmt='%d-%m-%y %H:%M',)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
