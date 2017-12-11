# katsdpresearch: katsdpcal

Preliminary calibration node code.

## Dependencies

For katsdpcal:

* redis (2.8.19+)
* PySPEAD
* katsdptelstate

For katsdpcal simulator:

* katdal
* tmux
* tmuxp (0.8.1+)

## Simulator

The simulator can be run manually, or using a shortcut script. See the help of the various scripts to see what parameters are available and their meanings. The simulator uses either an H5 or MS file as the data source.

Note: The recommended SPEAD rates for laptops are L0: 0.2e7; L1: 1.2e7; And for Laura's server L0: 0.4e7; L1: 5e7

### Manual simulator

1. start a redis server 

2. run the h5 Telescope State simulator:
  
 > sim_ts.py --telstate 127.0.0.1:6379 --file \<file.h5 or file.ms\>

3. run the pipeline controller:

 > run_cal.py   

4. run the h5 data stream:

 > sim_data_stream.py --telstate 127.0.0.1:6379 --file \<file.h5 or file.ms\>
 
### Shortcut simulator

 > run_katsdpcal_sim.py --telstate 127.0.0.1:6379 --file \<file.h5 or file.ms\> --l0-rate 0.2e7 --max-scans=7 --keep-sessions
 
The shortcut simulator runs each of the five commands above in separate tmux sessions, named redis, sim_ts, pipeline and sim_data respectively.
 
