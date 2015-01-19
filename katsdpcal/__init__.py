"""
katcal
======

Calibration pipeline package for MeerKAT.
"""

import logging
logging.basicConfig(filename='pipeline.log')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


#LOG_FILENAME='/var/log/celery_workflowmgr/celery_workflowmgr.log'
#logging.basicConfig(filename=LOG_FILENAME,level=logging.INFO)