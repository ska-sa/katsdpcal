# katsdpcal

Calibration node code.

## Dependencies

Refer to setup.py, or just run `pip install -e .` You will need to install
katsdpservices, katsdpsigproc and katsdptelstate separately from Github. You
will also need a redis server (2.8.19+).

## Simulator

The simulator can be run manually, or using a shortcut script. See the help of
the various scripts to see what parameters are available and their meanings.
The simulator uses either an H5 or MS file as the data source.

### Manual simulator

1. Start a redis server

2. Run the h5 Telescope State simulator:

 > sim_ts.py --telstate 127.0.0.1:6379 --file \<file.h5 or file.ms\>

3. Run the pipeline controller:

 > run_cal.py --telstate 127.0.0.1:6379

4. Run the h5 data stream:

 > sim_data_stream.py --telstate 127.0.0.1:6379 --file \<file.h5 or file.ms\>

You can pass `--max-scans` to restrict the number of scans to replay from a large file.

### Shortcut simulator

This additionally requires

* tmux
* tmuxp (0.8.1+)

 > run_katsdpcal_sim.py --telstate 127.0.0.1:6379 --file \<file.h5 or file.ms\> --max-scans=7 --keep-sessions

The shortcut simulator runs each of the five commands above in separate tmux
sessions, named redis, sim\_ts, pipeline and sim\_data respectively.
