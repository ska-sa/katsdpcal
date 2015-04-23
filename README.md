katsdpresearch: katcal
======================

Preliminary calibration node code.

Instructions for running code with h5 simulator:

* start a redis server 

* run the h5 Telescope State simulator:
  
 > python sim_h5_ts.py --telstate 127.0.0.1:6379 --h5file \<h5file.h5\>

* run the L1 spead receive simulator (will only work on a single scan at a time):

 > python sim_l1_receive.py 

* run the pipeline controller:

 > python run_cal.py   

* run the h5 data stream:

 > python sim_h5_stream.py --telstate 127.0.0.1:6379 --h5file \<h5file.h5\>
