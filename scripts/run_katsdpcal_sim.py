#!/usr/bin/env python3
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

import libtmux
import time
from argparse import ArgumentParser
import os.path


def parse_args():
    parser = ArgumentParser(description='Run simulated katsdpcal from h5 or MS file')
    parser.add_argument(
        '--keep-sessions', action='store_true',
        help='Keep any pre-existing tmux sessions. '
             'Note: Only use this if pipeline not currently running.')
    parser.set_defaults(keep_sessions=False)
    parser.add_argument(
        '--telstate', type=str, default='127.0.0.1:6379',
        help='Telescope state endpoint. Default "127.0.0.1:6379"')
    parser.add_argument(
        '--file', type=str,
        help='Comma separated list of H5 or MS file for simulated data')
    parser.add_argument(
        '--buffer-maxsize', type=float, default=1e9,
        help='The amount of memory (in bytes) to allocate to each buffer. default: 1e9')
    parser.add_argument(
        '--no-auto', action='store_true',
        help='Pipeline data DOESNT include autocorrelations '
             '[default: False (autocorrelations included)]')
    parser.set_defaults(no_auto=False)
    parser.add_argument(
        '--max-scans', type=int, default=0,
        help='Number of scans to transmit. Default: all')
    parser.add_argument(
        '--l0-rate', type=float, default=5e7,
        help='Simulated L0 SPEAD rate. Default: %(default)s')
    parser.add_argument(
        '--parameter-file', type=str, default='', help='Default pipeline parameter file.')
    parser.add_argument(
        '--report-path', type=str, default=os.path.abspath('.'),
        help='Path under which to save pipeline report. [default: current directory]')
    parser.add_argument(
        '--log-path', type=str, default=os.path.abspath('.'),
        help='Path under which to save pipeline logs. [default: current directory]')
    parser.add_argument(
        '--threading', action='store_true',
        help='Use threading to control pipeline and accumulator [default: use multiprocessing]')
    parser.set_defaults(threading=False)
    return parser.parse_args()


def create_pane(sname, tmserver, keep_session=False):
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
            print('killed session {},'.format(sname), end=' ')
        except libtmux.exc.LibTmuxException:
            print('session {} did not exist,'.format(sname), end=' ')
    # start new session
    try:
        tmserver.new_session(sname)
        print('created session {}'.format(sname))
    except libtmux.exc.TmuxSessionExists:
        print('session {} already exists'.format(sname))
    # get pane
    session = tmserver.find_where({"session_name": sname})
    return session.windows[0].panes[0]


if __name__ == '__main__':
    opts = parse_args()

    file_list = opts.file.split(',')
    # Get full path of first h5 file
    #   Workaround for issue in tmux sessions where relative paths
    #   are not parsed correctly by h5py.File
    first_file = file_list[0]
    first_file_fullpath = os.path.abspath(first_file)

    # create tmux server
    tmserver = libtmux.Server()

    # start redis-server in tmux pane
    redis_pane = create_pane('redis', tmserver, keep_session=opts.keep_sessions)
    redis_pane.cmd('send-keys', 'redis-server')
    redis_pane.enter()

    # set up TS in tmux pane
    sim_ts_pane = create_pane('sim_ts', tmserver, keep_session=opts.keep_sessions)
    sim_ts_pane.cmd('send-keys', 'sim_ts.py --telstate {0} --file {1}'
                    .format(opts.telstate, first_file_fullpath))
    sim_ts_pane.enter()

    # wait a few seconds for TS to be set up
    time.sleep(10.0)

    # start pipeline running in tmux pane
    threading_option = '--threading' if opts.threading else ''
    no_auto = '--no-auto' if opts.no_auto else ''
    pipeline_pane = create_pane('pipeline', tmserver, keep_session=opts.keep_sessions)
    pipeline_pane.cmd(
        'send-keys',
        'run_cal.py --telstate {} --buffer-maxsize {} --report-path {} --log-path {} {} {}'
        .format(opts.telstate, opts.buffer_maxsize,
                opts.report_path, opts.log_path, threading_option, no_auto))
    pipeline_pane.enter()

    # wait a couple of seconds to start data flowing
    #   time for setting up the pipeline and L1 receiver (setting parameters, creating buffers, etc)
    #   simulator testing is often done on Laura's laptop, which can need a few
    #   seconds here if the buffers are ~> 1G
    time.sleep(5.0)

    # start data flow in tmux pane
    #   wait 60 seconds from the end of one data transmission to the starrt of the next
    sim_data_pane = create_pane('sim_data', tmserver, keep_session=opts.keep_sessions)
    for f in file_list:
        # Get full path of h5 file
        file_fullpath = os.path.abspath(f)
        sim_data_pane.cmd(
            'send-keys', 'sim_data_stream.py --telstate {0} --file {1} --l0-rate {2} '
            '--max-scans {3}; sleep 60. ; '
            .format(opts.telstate, file_fullpath, opts.l0_rate, opts.max_scans))
    sim_data_pane.enter()
