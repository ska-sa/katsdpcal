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

       sim_ts.py --telstate 127.0.0.1:6379 --file <file.rdb/h5/ms>

3. Run the pipeline controller:

       run_cal.py --telstate 127.0.0.1:6379

4. Run the h5 data stream:

       sim_data_stream.py --telstate 127.0.0.1:6379 --file <file.rdb/h5/ms>

You can pass `--max-scans` to restrict the number of scans to replay from a large file.

### Shortcut simulator

This additionally requires

* tmux
* tmuxp (0.8.1+)

      run_katsdpcal_sim.py --telstate 127.0.0.1:6379 --file <file.rdb/h5/ms> --max-scans=7 --keep-sessions

The shortcut simulator runs each of the five commands above in separate tmux
sessions, named redis, sim\_ts, pipeline and sim\_data respectively.

### Multiple pipelines

The multicast groups and the ports for the servers need to be chosen to avoid
conflicts with anything else that happens to the running on the system; the
values given are just examples. The instructions below are for two servers, but
it can scale up to higher numbers.

1. Start a redis server

2. Run the Telescope State simulator:

       sim_ts.py --telstate 127.0.0.1:6379 --file <file.rdb/h5/ms> --substreams 2

3. Run the pipeline controller (in parallel):

       run_cal.py --telstate 127.0.0.1:6379 --l0-spead 239.102.254.0+1:7148 --l0-interface lo \
         --servers 2 -p 2060 --server-id 1
       run_cal.py --telstate 127.0.0.1:6379 --l0-spead 239.102.254.0+1:7148 --l0-interface lo \
         --servers 2 -p 2061 --server-id 2

4. Run the h5 data stream:

       sim_data_stream.py --telstate 127.0.0.1:6379 --file <file.rdb/h5/ms> \
         --l0-spead 239.102.254.0+1:7148 --l0-interface=lo \
         --server localhost:2060,localhost:2061
