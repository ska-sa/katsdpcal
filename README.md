# katsdpresearch: katsdpcal

Preliminary calibration node code.

## Dependencies

For katsdpcal:

* redis (2.8.19+)
* PySPEAD
* katsdptelstate
* rst2pdf

For katsdpcal simulator:

* katdal
* tmux
* tmuxp (0.8.1+)

## Simulator

The h5 simulator can be run manually, or using a shortcut script. See the help of the various scripts to see what parameters are available and their meanings.

Note: The recommended SPEAD rates for laptops are L0: 0.2e7; L1: 1.2e7; And for Laura's server L0: 0.4e7; L1: 5e7

### Manual simulator

1. start a redis server 

2. run the h5 Telescope State simulator:
  
 > sim_h5_ts.py --telstate 127.0.0.1:6379 --h5file \<h5file.h5\>

3. run the L1 spead receive simulator (will only work on a single scan at a time):

 > sim_l1_receive.py 

4. run the pipeline controller:

 > run_cal.py   

5. run the h5 data stream:

 > sim_h5_stream.py --telstate 127.0.0.1:6379 --h5file \<h5file.h5\>
 
### Shortcut simulator

 > run_katsdpcal_sim.py --telstate 127.0.0.1:6379 --h5file \<h5file.h5\> --l0-rate 0.2e7 --l1-rate 1.2e7 --max-scans=7 --keep-sessions
 
The shortcut simulator runs each of the five commands above in separate tmux sessions, named redis, sim_ts, l1_receiver, pipeline and sim_data respectively.
 
