
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

def parse_args():
    parser = ArgumentParser(description = 'Run simulated katcal from h5 file')
    parser.add_argument('--keep-sessions', action='store_true', help='Keep any pre-existing tmux sessions. Note: Only use this if pipeline not currently running.')
    parser.set_defaults(keep_sessions=False)
    parser.add_argument('--telstate', type=str, default='127.0.0.1:6379', help='Telescope state endpoint. Default "127.0.0.1:6379"')
    parser.add_argument('--h5file', type=str, default='~/data/1427381884.h5', help='H5 file for simulated data')
    parser.add_argument('--buffer-maxsize', type=float, default=1e9, help='The amount of memory (in bytes?) to allocate to each buffer. default: 1e9')
    parser.add_argument('--max-scans', type=int, default=0, help='Number of scans to transmit. Default: all')
    parser.add_argument('--l0-rate', type=float, default=5e7, help='Simulated L0 SPEAD rate. For laptops, recommend rate of 5e7. Default: 5e7')
    parser.add_argument('--l1-rate', type=float, default=5e7, help='L1 SPEAD transmission rate. For laptops, recommend rate of 5e7. Default: 5e7')
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

	KATCAL_DIR = '$PYTHONPATH'

	# create tmux server
	tmserver = tmuxp.Server()

	# start redis-server in tmux pane
	redis_pane = create_pane('redis',tmserver,keep_session=opts.keep_sessions)
	redis_pane.cmd('send-keys','redis-server')
	redis_pane.enter()

	# set up TS in tmux pane
	sim_ts_pane = create_pane('sim_ts',tmserver,keep_session=opts.keep_sessions)
	sim_ts_pane.cmd('send-keys','cd {}'.format(KATCAL_DIR,))
	sim_ts_pane.enter()
	sim_ts_pane.cmd('send-keys','python scripts/sim_h5_ts.py --telstate {0} --h5file {1}'.format(opts.telstate, opts.h5file))
	sim_ts_pane.enter()

	# start pipeline running in tmux pane
	pipeline_pane = create_pane('pipeline',tmserver,keep_session=opts.keep_sessions)
	pipeline_pane.cmd('send-keys','cd {}'.format(KATCAL_DIR,))
	pipeline_pane.enter()
	pipeline_pane.cmd('send-keys','python scripts/run_cal.py --telstate {0} --buffer-maxsize {1} \
		--l1-rate {2}'.format(opts.telstate, opts.buffer_maxsize, opts.l1_rate))
	pipeline_pane.enter()

	# start L1 receiver in tmux pane
	l1_pane = create_pane('l1_receiver',tmserver,keep_session=opts.keep_sessions)
	l1_pane.cmd('send-keys','cd {}'.format(KATCAL_DIR,))
	l1_pane.enter()
	l1_pane.cmd('send-keys','python scripts/sim_l1_receive.py --telstate {0} --h5file {1}'.format(opts.telstate, opts.h5file))
	l1_pane.enter()

	# wait a couple of seconds to start data flowing
	time.sleep(2.0)

	# start data flow in tmux pane
	sim_data_pane = create_pane('sim_data',tmserver,keep_session=opts.keep_sessions)
	sim_data_pane.cmd('send-keys','cd {}'.format(KATCAL_DIR,))
	sim_data_pane.enter()
	sim_data_pane.cmd('send-keys','python scripts/sim_h5_stream.py --telstate {0} --h5file {1} --l0-rate {2} \
          --max-scans {3}'.format(opts.telstate, opts.h5file, opts.l0_rate, opts.max_scans))
	sim_data_pane.enter()