katsdpresearch: katcal
======================

Preliminary calibration node code.

Instructions for running code with h5 simulator:

* Start a redis server 

* run the h5 TM simulator:
  
 > python sim_h5_tm.py   \<h5file.h5\>

* run the pipeline controller:

 > python run_cal.py   

* run the h5 data stream:
 > python sim_h5_stream.py   \<h5file.h5\>
