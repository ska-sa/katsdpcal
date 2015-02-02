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

# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# set format for console use
formatter = logging.Formatter('%(asctime)s %(name)-24s %(levelname)-8s %(message)s')
formatter.datefmt='%d-%m %H:%M'
# tell the handler to use this format
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)