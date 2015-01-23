katsdpresearch: katcal
======================

Preliminary calibration node code.

Instructions for running code with h5 simulator:

* Start a redis server 

* run the TM simulator:
  
 > python sim_h5_tm.py   \<h5file.h5\>

* start the pipeline controller:

 > python run_cal.py   \<h5file.h5\>

* start the h5 data stread:
 > python sim_h5_stream.py   \<h5file.py\>
