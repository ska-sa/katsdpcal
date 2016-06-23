#! /usr/bin/env python
# ---------------------------------------------------------------------------------------
# Runs simulator using tmux
# ---------------------------------------------------------------------------------------
#
# Usage notes:
#
#   * tmux sessions can be attached to using:
#       > tmux attach-session -t <session_name>
#     For the sessions created here, this will be:
#       > tmux attach-session -t redis
#       > tmux attach-session -t sim_ts
#       > tmux attach-session -t pipeline
#       > tmux attach-session -t l1_receiver
#       > tmux attach-session -t sim_data
#
#   * tmux session can be detached from, from the command line within the session, using
#      > tmux detach
#     or using the key-binding Ctrl-b :detach, or Ctrl-b d
#
#   * To scroll up tmux pane history, use Ctrl-b PageUp
#      To exit scroll mode, press q.
#
# ---------------------------------------------------------------------------------------

import tmuxp
import time
from argparse import ArgumentParser
import os.path

def parse_args():
    parser = ArgumentParser(description = 'Run simulated katsdpcal from h5 or MS file')
    parser.add_argument('--keep-sessions', action='store_true', help='Keep any pre-existing tmux sessions. Note: Only use this if pipeline not currently running.')
    parser.set_defaults(keep_sessions=False)
    parser.add_argument('--telstate', type=str, default='127.0.0.1:6379', help='Telescope state endpoint. Default "127.0.0.1:6379"')
    parser.add_argument('--file', type=str, help='Comma separated list of H5 or MS file for simulated data')
    parser.add_argument('--buffer-maxsize', type=float, default=1e9, help='The amount of memory (in bytes) to allocate to each buffer. default: 1e9')
    parser.add_argument('--no-auto', action='store_true', help='Pipeline data DOESNT include autocorrelations [default: False (autocorrelations included)]')
    parser.set_defaults(no_auto=False)
    parser.add_argument('--max-scans', type=int, default=0, help='Number of scans to transmit. Default: all')
    parser.add_argument('--l0-rate', type=float, default=5e7, help='Simulated L0 SPEAD rate. For laptops, recommend rate of 0.2e7. Default: 0.4e7')
    parser.add_argument('--l1-rate', type=float, default=5e7, help='L1 SPEAD transmission rate. For laptops, recommend rate of 1.2e7. Default: 5e7')
    parser.add_argument('--parameter-file', type=str, default='', help='Default pipeline parameter file (will be over written by TelescopeState.')
    parser.add_argument('--report-path', type=str, default=os.path.abspath('.'), help='Path under which to save pipeline report. [default: current directory]')
    parser.add_argument('--log-path', type=str, default=os.path.abspath('.'), help='Path under which to save pipeline logs. [default: current directory]')
    parser.add_argument('--notthreading', action='store_false', help='Use threading to control pipeline and accumulator [default: False (to use multiprocessing)]')
    parser.set_defaults(threading=True)
    return parser.parse_args()

def create_pane(sname,tmserver,keep_session=False):
    """
    Create tmux session and return pane object, for single window single pane

    Inputs
    ------
    sname : name for tmux session, string

    Returns
    -------
    tmux pane object
    """
    # kill session if it already exists, unless we are keeping pre-existing sessions
    if not keep_session:
        try:
            tmserver.kill_session(sname)
            print 'killed session {},'.format(sname,),
        except tmuxp.exc.TmuxpException:
            print 'session {} did not exist,'.format(sname,),
    # start new session
    try:
        tmserver.new_session(sname)
        print 'created session {}'.format(sname,)
    except tmuxp.exc.TmuxSessionExists:
        print 'session {} already exists'.format(sname,)		
    # get pane
    session = tmserver.findWhere({"session_name":sname})
    return session.windows[0].panes[0]

if __name__ == '__main__':
    opts = parse_args()

    file_list = opts.file.split(',')
    # Get full path of first h5 file
    #   Workaround for issue in tmux sessions where relative paths
    #   are not parsed correctly by h5py.File
    first_file = file_list[0]
    first_file_fullpath=os.path.abspath(first_file)

    # create tmux server
    tmserver = tmuxp.Server()

    # start redis-server in tmux pane
    redis_pane = create_pane('redis',tmserver,keep_session=opts.keep_sessions)
    redis_pane.cmd('send-keys','redis-server')
    redis_pane.enter()

    # set up TS in tmux pane
    #  we use the parameter file to initialise  the telescope state for the simulator
    #  ( in the real system we expect the TS to be initialised, and use the parameter file as defaults
    #  for parameter missing from the TS)
    param_string = '--parameter-file {0}'.format(opts.parameter_file,) if opts.parameter_file != '' else ''
    sim_ts_pane = create_pane('sim_ts',tmserver,keep_session=opts.keep_sessions)
    sim_ts_pane.cmd('send-keys','sim_ts.py --telstate {0} --file {1} {2}'.format(opts.telstate, first_file_fullpath, param_string))
    sim_ts_pane.enter()

    # wait a few seconds for TS to be set up
    time.sleep(5.0)

    # start pipeline running in tmux pane
    threading_option = '--notthreading' if opts.notthreading else ''
    no_auto = '--no-auto' if opts.no_auto else ''
    pipeline_pane = create_pane('pipeline',tmserver,keep_session=opts.keep_sessions)
    pipeline_pane.cmd('send-keys','run_cal.py --telstate {0} --buffer-maxsize {1} \
        --l1-rate {2} {3} --report-path {4} --log-path {5} {6} {7}'.format(opts.telstate, opts.buffer_maxsize,
        opts.l1_rate, param_string, opts.report_path, opts.log_path, threading_option, no_auto))
    pipeline_pane.enter()

    # start L1 receiver in tmux pane
    image = '--image' if (opts.max_scans == 0) else ''
    l1_pane = create_pane('l1_receiver',tmserver,keep_session=opts.keep_sessions)
    for f in file_list:
        # Get full path of h5 file
        file_fullpath=os.path.abspath(f)
        l1_pane.cmd('send-keys','sim_l1_receive.py --telstate {0} --file {1} {2}; '.format(opts.telstate, file_fullpath, image))
    l1_pane.enter()

    # wait a couple of seconds to start data flowing
    #   time for setting up the pipeline and L1 receiver (setting parameters, creating buffers, etc)
    #   simulator testing is often done on Laura's laptop, which can need a few seconds here if the buffers are ~> 1G
    time.sleep(5.0)

    # start data flow in tmux pane
    #   wait 60 seconds from the end of one data transmission to the starrt of the next
    sim_data_pane = create_pane('sim_data',tmserver,keep_session=opts.keep_sessions)
    for f in file_list:
        # Get full path of h5 file
        file_fullpath=os.path.abspath(f)
        sim_data_pane.cmd('send-keys','sim_data_stream.py --telstate {0} --file {1} --l0-rate {2} \
            --max-scans {3}; sleep 60. ; '.format(opts.telstate, file_fullpath, opts.l0_rate, opts.max_scans))
    sim_data_pane.enter()



