import spead2
import spead2.recv.trollius
import spead2.send

import katcp
from katcp.kattypes import request, return_reply

from .reduction import pipeline
from .report import make_cal_report
from . import calprocs

import numpy as np
import time
import mmap
import threading
import os
import shutil
from collections import Counter
import trollius
from trollius import From, Return
import tornado.gen
import katsdpservices.asyncio
import concurrent.futures

import logging
logger = logging.getLogger(__name__)


class ObservationEndEvent(object):
    """An observation has finished upstream"""
    def __init__(self, index, start_time, end_time):
        self.index = index
        self.start_time = start_time
        self.end_time = end_time


class StopEvent(object):
    """Graceful shutdown requested"""


class BufferReadyEvent(object):
    """Indicates to the pipeline that the buffer is ready for it."""


def shared_empty(shape, dtype):
    """
    Allocate a numpy array from shared memory. The contents are undefined.

    .. note:: This only works on UNIX-like systems, not Windows.
    """
    dtype = np.dtype(dtype)
    items = int(np.product(shape))
    n_bytes = items * dtype.itemsize
    raw = mmap.mmap(-1, n_bytes, mmap.MAP_SHARED)
    array = np.frombuffer(raw, dtype)
    array.shape = shape
    return array


# ---------------------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------------------

class Accumulator(object):
    """Manages accumulation of L0 data into buffers"""

    def __init__(self, control_method, buffers, buffer_shape,
                 accum_pipeline_queues, pipeline_accum_sems,
                 l0_endpoint, l0_interface_address, telstate):
        self.buffers = buffers
        self.telstate = telstate
        self.l0_endpoint = l0_endpoint
        self.l0_interface_address = l0_interface_address
        self.accum_pipeline_queues = accum_pipeline_queues
        self.pipeline_accum_sems = pipeline_accum_sems
        self.num_buffers = len(buffers)

        # Extract useful parameters from telescope state
        self.cbf_sync_time = self.telstate.cbf_sync_time
        self.sdp_l0_int_time = self.telstate.sdp_l0_int_time
        self.set_ordering_parameters()

        self.name = 'Accumulator'
        self._rx = None
        self._run_future = None
        # First and last timestamps in observation
        self._obs_start = None
        self._obs_end = None
        # Unique number given to each observation
        self._index = 0

        # Get data shape
        self.buffer_shape = buffer_shape
        self.max_length = buffer_shape[0]
        self.nchan = buffer_shape[1]
        self.npol = buffer_shape[2]
        self.nbl = buffer_shape[3]

        # Allocate storage and thread pool for receiver
        # Main data is 10 bytes per entry: 8 for vis, 1 for flags, 1 for weights.
        # Then there are per-channel weights (4 bytes each).
        heap_size = self.nchan * self.npol * self.nbl * 10 + self.nchan * 4
        self._thread_pool = spead2.ThreadPool()
        self._memory_pool = spead2.MemoryPool(heap_size, heap_size + 4096, 4, 4)

        # Thread for doing blocking waits, to avoid stalling the asyncio event loop
        self._executor = concurrent.futures.ThreadPoolExecutor(1)

    @property
    def capturing(self):
        return self._rx is not None

    @trollius.coroutine
    def _run_observation(self, index):
        """Runs for a single observation i.e., until a stop heap is received."""
        try:
            rx = self._rx
            # Increment between buffers, filling and releasing iteratively
            # Initialise current buffer counter
            current_buffer = -1
            obs_stopped = False
            while not obs_stopped:
                # Increment the current buffer
                current_buffer = (current_buffer + 1) % self.num_buffers
                # ------------------------------------------------------------
                # Loop through the buffers and send data to pipeline task when
                # accumulation terminate conditions are met.

                logger.info('waiting for pipeline_accum_sems[%d]', current_buffer)
                yield From(trollius.get_event_loop().run_in_executor(
                    self._executor, self.pipeline_accum_sems[current_buffer].acquire))
                logger.info('pipeline_accum_sems[%d] acquired by %s',
                            current_buffer, self.name)

                # accumulate data scan by scan into buffer arrays
                logger.info('max buffer length %d', self.max_length)
                logger.info('accumulating into buffer %d', current_buffer)
                max_ind, obs_stopped = yield From(self.accumulate(rx, current_buffer))
                logger.info('Accumulated {0} timestamps'.format(max_ind+1))

                # awaken pipeline task that is waiting for the buffer
                self.accum_pipeline_queues[current_buffer].put(BufferReadyEvent())
                logger.info('accum_pipeline_queues[%d] updated by %s', current_buffer, self.name)

            # Tell the pipelines that the observation ended, but only if there
            # was something to work on.
            if self._obs_end is not None:
                for q in self.accum_pipeline_queues:
                    q.put(ObservationEndEvent(index, self._obs_start, self._obs_end))
            else:
                logger.info(' --- no data flowed ---')
            logger.info('Observation ended')
        except trollius.CancelledError:
            logger.info('Observation cancelled')
        except Exception as error:
            logger.error('Exception in capture: %s', error, exc_info=True)
        finally:
            rx.stop()

    def capture_init(self):
        assert self._rx is None, "observation already running"
        assert self._run_future is None
        logger.info('===========================')
        logger.info('   Starting new observation')
        self._obs_start = None
        self._obs_end = None
        # Initialise SPEAD receiver
        logger.info('Initializing SPEAD receiver')
        rx = spead2.recv.trollius.Stream(
            self._thread_pool, bug_compat=spead2.BUG_COMPAT_PYSPEAD_0_5_2,
            max_heaps=2, ring_heaps=1)
        rx.set_memory_allocator(self._memory_pool)
        rx.set_memcpy(spead2.MEMCPY_NONTEMPORAL)
        if self.l0_interface_address is not None:
            rx.add_udp_reader(self.l0_endpoint.host, self.l0_endpoint.port,
                              interface_address=self.l0_interface_address)
        else:
            rx.add_udp_reader(self.l0_endpoint.port, bind_hostname=self.l0_endpoint.host)
        logger.info('reader added')
        self._rx = rx
        self._run_future = trollius.ensure_future(self._run_observation(self._index))
        self._index += 1

    @trollius.coroutine
    def capture_done(self):
        assert self._rx is not None, "observation not running"
        self._rx.stop()
        future = self._run_future
        yield From(future)
        # Protect against another observation having started while we waited to
        # be woken up again.
        if self._run_future is future:
            logger.info('Joined with _run_observation')
            self._run_future = None
            self._rx = None

    @trollius.coroutine
    def stop(self, force=False):
        """Shuts down the accumulator.

        If `force` is true, this assumes that the pipelines have already been
        terminated, and it does not try to wake them up; otherwise it sends
        them stop events.
        """
        if force:
            # Ensure that the semaphores aren't blocked, even if the pipelines
            # have been terminated without releasing the buffers.
            for sem in self.pipeline_accum_sems:
                sem.release()
        if self._run_future is not None:
            yield From(self.capture_done())
        if not force:
            for q in self.accum_pipeline_queues:
                q.put(StopEvent())
        self._executor.shutdown()

    def set_ordering_parameters(self):
        # determine re-ordering necessary to convert from supplied bls
        # ordering to desired bls ordering
        antlist = self.telstate.cal_antlist
        self.ordering, bls_order, pol_order = \
            calprocs.get_reordering(antlist, self.telstate.sdp_l0_bls_ordering)
        # determine lookup list for baselines
        bls_lookup = calprocs.get_bls_lookup(antlist, bls_order)
        # save these to the TS for use in the pipeline/elsewhere
        self.telstate.add('cal_bls_ordering', bls_order)
        self.telstate.add('cal_pol_ordering', pol_order)
        self.telstate.add('cal_bls_lookup', bls_lookup)

    @classmethod
    def _update_buffer(cls, out, l0, ordering):
        """Copy values from an item group to the accumulation buffer.

        The input has a single dimension representing both baseline and
        polarisation, while the output has separate dimensions. There can
        be an arbitrary permutation (given by `ordering`) of the
        pol-baselines.

        This is equivalent to

        .. code:: python
            out[:] = l0[:, ordering].reshape(out.shape)

        but more efficient as it does not construct a temporary array.

        It is required that the output can be reshaped to collapse the
        pol and baseline dimensions. Being C-contiguous is sufficient for
        this.

        Parameters
        ----------
        out : :class:`np.ndarray`
            Output array, shape (nchans, npols, nbls)
        l0 : :class:`np.ndarray`
            Input array, shape (nchans, npols * nbls)
        ordering:
            Indices into l0's last dimension to permute them before
            reshaping into separate polarisation and baseline dimensions.
        """
        # Assign to .shape instead of using reshape so that an exception
        # is raised if a view cannot be created (see np.reshape).
        out_view = out.view()
        out_view.shape = (out.shape[0], out.shape[1] * out.shape[2])
        np.take(l0, ordering, axis=1, out=out_view)

    @trollius.coroutine
    def accumulate(self, rx, buffer_index):
        """
        Accumulates spead data into arrays
           till **TBD** metadata indicates scan has stopped, or
           till array reaches max buffer size

        SPEAD item groups contain:
           correlator_data
           flags
           weights
           weights_channel
           timestamp

        Returns
        -------
        array_index : int
            Last filled position in buffer
        obs_stopped : bool
            Whether the return was due to stream stopping
        """

        start_flag = True
        array_index = -1

        prev_activity = 'none'
        prev_activity_time = 0.
        prev_target_tags = 'none'
        prev_target_name = 'none'

        # set data buffer for writing
        data_buffer = self.buffers[buffer_index]

        # get names of activity and target TS keys, using TS reference antenna
        target_key = '{0}_target'.format(self.telstate.cal_refant,)
        activity_key = '{0}_activity'.format(self.telstate.cal_refant,)

        obs_stopped = False

        # receive SPEAD stream
        ig = spead2.ItemGroup()

        logger.info('waiting to start accumulating data')
        while True:
            try:
                heap = yield From(rx.get())
            except spead2.Stopped:
                obs_stopped = True
                break
            ig.update(heap)
            if len(ig.keys()) < 1:
                logger.info('==== empty heap received ====')
                continue

            # get activity and target tag from TS
            data_ts = ig['timestamp'].value + self.cbf_sync_time
            if self._obs_start is None:
                self._obs_start = data_ts - 0.5 * self.sdp_l0_int_time
            self._obs_end = data_ts + 0.5 * self.sdp_l0_int_time
            activity_full = []
            try:
                activity_full = self.telstate.get_range(
                    activity_key, et=data_ts, include_previous=True)
            except KeyError:
                pass
            if not activity_full:
                logger.info('no activity recorded for reference antenna {0} - ignoring dump'.format(
                    self.telstate.cal_refant))
                continue
            activity, activity_time = activity_full[0]

            # if this is the first scan of the batch, set up some values
            if start_flag:
                unsync_start_time = ig['timestamp'].value
                prev_activity_time = activity_time
                logger.info('accumulating data from targets:')

            # get target time from TS, if it is present (if it isn't present, set to unknown)
            try:
                target = self.telstate.get_range(target_key, et=data_ts,
                                                 include_previous=True)[0][0]
                if target == '':
                    target = 'unknown'
            except KeyError:
                logger.warning(
                    'target description {0} absent from telescope state'.format(target_key))
                target = 'unknown'
            # extract name and tags from target description string
            target_split = target.split(',')
            target_name = target_split[0]
            target_tags = target_split[1] if len(target_split) > 1 else 'unknown'
            if (target_name != prev_target_name) or start_flag:
                # update source list if necessary
                try:
                    target_list = self.telstate.get_range(
                        'cal_info_sources', st=0, return_format='recarray')['value']
                except KeyError:
                    target_list = []
                if target_name not in target_list:
                    self.telstate.add('cal_info_sources', target_name, ts=data_ts)

            # print name of target and activity type, if activity has
            # changed or start of accumulator
            if start_flag or (activity_time != prev_activity_time):
                logger.info(' - {0} ({1})'.format(target_name, activity))
            start_flag = False

            # increment the index indicating the position of the data in the buffer
            array_index += 1
            # reshape data and put into relevent arrays
            self._update_buffer(data_buffer['vis'][array_index],
                                ig['correlator_data'].value, self.ordering)
            self._update_buffer(data_buffer['flags'][array_index],
                                ig['flags'].value, self.ordering)
            weights_channel = ig['weights_channel'].value[:, np.newaxis]
            weights = ig['weights'].value
            self._update_buffer(data_buffer['weights'][array_index],
                                weights * weights_channel, self.ordering)
            data_buffer['times'][array_index] = data_ts

            # break if activity has changed (i.e. the activity time has changed)
            #   unless previous scan was a target, in which case accumulate
            #   subsequent gain scan too
            # ********** THIS BREAKING NEEDS TO BE THOUGHT THROUGH CAREFULLY **********
            ignore_states = ['slew', 'stop', 'unknown']
            if (activity_time != prev_activity_time) \
                    and not np.any([ignore in prev_activity for ignore in ignore_states]) \
                    and ('unknown' not in target_tags) \
                    and ('target' not in prev_target_tags):
                logger.info('Accumulation break - transition')
                break
            # beamformer special case
            if (activity_time != prev_activity_time) \
                    and ('single_accumulation' in prev_target_tags):
                logger.info('Accumulation break - single scan accumulation')
                break

            # this is a temporary mock up of a natural break in the data stream
            # will ultimately be provided by some sort of sensor
            duration = ig['timestamp'].value - unsync_start_time
            if duration > 2000000:
                logger.info('Accumulate break due to duration')
                break
            # end accumulation if maximum array size has been accumulated
            if array_index >= self.max_length - 1:
                logger.info('Accumulate break - buffer size limit')
                break

            prev_activity = activity
            prev_activity_time = activity_time
            prev_target_tags = target_tags
            prev_target_name = target_name

        data_buffer['max_index'][0] = array_index
        logger.info('Accumulation ended')
        raise Return((array_index, obs_stopped))


# ---------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------

def init_pipeline_control(control_method, control_task, *args, **kwargs):

    class pipeline_control(control_task):
        """
        Task (Process or Thread) which runs pipeline
        """

        def __init__(self, control_method, data, data_shape,
                     accum_pipeline_queue, pipeline_accum_sem, pipeline_report_queue,
                     pipenum, l1_endpoint, l1_level, l1_rate, telstate):
            control_task.__init__(self)
            self.data = data
            self.accum_pipeline_queue = accum_pipeline_queue
            self.pipeline_accum_sem = pipeline_accum_sem
            self.pipeline_report_queue = pipeline_report_queue
            self.name = 'Pipeline_' + str(pipenum)
            self.telstate = telstate
            self.data_shape = data_shape
            self.l1_level = l1_level
            self.l1_rate = l1_rate
            self.l1_endpoint = l1_endpoint

        def run(self):
            """
            Task (Process or Thread) run method. Runs pipeline
            """

            # run until stop event received
            try:
                while True:
                    logger.info('waiting for next event (%s)', self.name)
                    event = self.accum_pipeline_queue.get()
                    if isinstance(event, BufferReadyEvent):
                        logger.info('buffer acquired by %s', self.name)
                        # run the pipeline
                        self.run_pipeline()
                        # release condition after pipeline run finished
                        self.pipeline_accum_sem.release()
                        logger.info('pipeline_accum_sem release by %s', self.name)
                    elif isinstance(event, ObservationEndEvent):
                        self.pipeline_report_queue.put(event)
                    elif isinstance(event, StopEvent):
                        logger.info('stop received by %s', self.name)
                        break
                    else:
                        logger.error('unknown event type %r by %s', event, self.name)
            finally:
                self.pipeline_report_queue.put(StopEvent())

        def run_pipeline(self):
            # run pipeline calibration, if more than zero timestamps accumulated
            target_slices = pipeline(self.data, self.telstate, task_name=self.name) \
                if (self.data['max_index'][0] > 0) else []

            # send data to L1 SPEAD if necessary
            if self.l1_level != 0:
                config = spead2.send.StreamConfig(max_packet_size=8972, rate=self.l1_rate)
                tx = spead2.send.UdpStream(spead2.ThreadPool(), self.l1_endpoint.host,
                                           self.l1_endpoint.port, config)
                logger.info('   Transmit L1 data')
                # for streaming all of the data (not target only),
                # use the highest index in the buffer that is filled with data
                transmit_slices = [slice(0, self.data['max_index'][0] + 1)] \
                    if self.l1_level == 2 else target_slices
                self.data_to_SPEAD(transmit_slices, tx)
                logger.info('   End transmit of L1 data')

        def data_to_SPEAD(self, target_slices, tx):
            """
            Sends data to SPEAD stream

            Inputs:
            target_slices : list of slices
                slices for target scans in the data buffer
            """
            # create SPEAD item group
            flavour = spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
            ig = spead2.send.ItemGroup(flavour=flavour)
            # set up item group with items
            ig.add_item(id=None, name='correlator_data', description="Visibilities",
                        shape=self.data['vis'][0].shape, dtype=self.data['vis'][0].dtype)
            ig.add_item(id=None, name='flags', description="Flags for visibilities",
                        shape=self.data['flags'][0].shape, dtype=self.data['flags'][0].dtype)
            # for now, just transmit flags as placeholder for weights
            ig.add_item(id=None, name='weights', description="Weights for visibilities",
                        shape=self.data['flags'][0].shape, dtype=self.data['flags'][0].dtype)
            ig.add_item(id=None, name='timestamp', description="Seconds since sync time",
                        shape=(), dtype=None, format=[('f', 64)])

            # transmit data
            for data_slice in target_slices:
                # get data for this scan, from the slice
                scan_vis = self.data['vis'][data_slice]
                scan_flags = self.data['flags'][data_slice]
                # for now, just transmit flags as placeholder for weights
                scan_weights = self.data['flags'][data_slice]
                scan_times = self.data['times'][data_slice]

                # transmit data timestamp by timestamp
                for i in range(len(scan_times)):  # time axis
                    # transmit timestamps, vis, flags and weights
                    ig['correlator_data'].value = scan_vis[i]
                    ig['flags'].value = scan_flags[i]
                    ig['weights'].value = scan_weights[i]
                    ig['timestamp'].value = scan_times[i]
                    tx.send_heap(ig.get_heap())

    return pipeline_control(control_method, *args, **kwargs)


# ---------------------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------------------

def write_report(obs_start, obs_end, telstate,
                 l1_endpoint, l1_level, report_path, log_path, full_log):
    now = time.time()
    # get subarray ID
    subarray_id = telstate.get('subarray_product_id', 'unknown_subarray')
    # get observation name
    try:
        obs_params = telstate.get_range('obs_params', st=0, et=obs_end,
                                        return_format='recarray')
        obs_keys = obs_params['value']
        obs_times = obs_params['time']
        # choose most recent experiment id (last entry in the list), if
        # there are more than one
        experiment_id_string = [x for x in obs_keys if 'experiment_id' in x][-1]
        experiment_id = eval(experiment_id_string.split()[-1])
    except (TypeError, KeyError, AttributeError):
        # TypeError, KeyError because this isn't properly implemented yet
        # AttributeError in case this key isnt in the telstate for whatever reason
        experiment_id = '{0}_unknown_project'.format(int(now))

    # make directory for this observation, for logs and report
    if not report_path:
        report_path = '.'
    report_path = os.path.abspath(report_path)
    obs_dir = '{0}/{1}_{2}_{3}'.format(
        report_path, int(now), subarray_id, experiment_id)
    current_obs_dir = '{0}-current'.format(obs_dir)
    try:
        os.mkdir(current_obs_dir)
    except OSError:
        logger.warning('Experiment ID directory {} already exists'.format(current_obs_dir))

    # create pipeline report (very basic at the moment)
    try:
        make_cal_report(telstate, current_obs_dir, experiment_id, st=obs_start, et=obs_end)
    except Exception as e:
        logger.info('Report generation failed: %s', e, exc_info=True)

    if l1_level != 0:
        # send L1 stop transmission
        #   wait for a couple of secs before ending transmission, because
        #   it's a separate kernel socket and hence unordered with respect
        #   to the sockets used by the pipelines (TODO: share a socket).
        time.sleep(2.0)
        end_transmit(l1_endpoint.host, l1_endpoint.port)
        logger.info('L1 stream ended')

    logger.info('   Observation ended')
    logger.info('===========================')

    if full_log is not None:
        shutil.copy('{0}/{1}'.format(log_path, full_log),
                    '{0}/{1}'.format(current_obs_dir, full_log))

    # change report and log directory to final name for archiving
    shutil.move(current_obs_dir, obs_dir)
    logger.info('Moved observation report to %s', obs_dir)


def report_writer(pipeline_report_queue, telstate, num_pipelines,
                  l1_endpoint, l1_level,
                  report_path, log_path, full_log):
    remain = num_pipelines            # Number of pipelines still running
    observation_hits = Counter()      # Number of pipelines finished with each observation
    while True:
        event = pipeline_report_queue.get()
        if isinstance(event, StopEvent):
            remain -= 1
            if remain == 0:
                break
        elif isinstance(event, ObservationEndEvent):
            observation_hits[event.index] += 1
            if observation_hits[event.index] == num_pipelines:
                logger.info('Starting report number %d', event.index)
                write_report(event.start_time, event.end_time, telstate,
                             l1_endpoint, l1_level, report_path, log_path, full_log)
                del observation_hits[event.index]
        else:
            logger.error('unknown event type %r', event)
    logger.info('Last pipeline has finished, exiting')


def init_report_writer(control_method, control_task, *args, **kwargs):
    return control_task(target=report_writer, name='report_writer',
                        args=args, kwargs=kwargs)

# ---------------------------------------------------------------------------------------
# SPEAD helper functions
# ---------------------------------------------------------------------------------------


def end_transmit(host, port):
    """
    Send stop packet to spead stream tx

    Parameters
    ----------
    host : str
        host to transmit to
    port : int
        port to transmit to
    """
    config = spead2.send.StreamConfig(max_packet_size=8972)
    tx = spead2.send.UdpStream(spead2.ThreadPool(), host, port, config)

    flavour = spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
    heap = spead2.send.Heap(flavour)
    heap.add_end()

    tx.send_heap(heap)


# ---------------------------------------------------------------------------------------
# Device server
# ---------------------------------------------------------------------------------------

class CalDeviceServer(katcp.server.AsyncDeviceServer):
    def __init__(self, accumulator, *args, **kwargs):
        self.accumulator = accumulator
        super(CalDeviceServer, self).__init__(*args, **kwargs)

    def setup_sensors(self):
        pass    # TODO

    @request()
    @return_reply()
    def request_capture_init(self, msg):
        """Start an observation"""
        if self.accumulator.capturing:
            return ('fail', 'capture already in progress')
        self.accumulator.capture_init()
        return ('ok',)

    @request()
    @return_reply()
    @tornado.gen.coroutine
    def request_capture_done(self, msg):
        """Stop the current observation"""
        if not self.accumulator.capturing:
            raise tornado.gen.Return(('fail', 'no capture in progress'))
        yield katsdpservices.asyncio.to_tornado_future(self.accumulator.capture_done())
        raise tornado.gen.Return(('ok',))
