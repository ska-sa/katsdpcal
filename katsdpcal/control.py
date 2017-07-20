import time
import mmap
import os
import shutil
import logging

import spead2
import spead2.recv.trollius
import spead2.send

import katcp
from katcp.kattypes import request, return_reply, concurrent_reply, Bool

import numpy as np
import trollius
from trollius import From, Return
import tornado.gen
from katsdpservices.asyncio import to_tornado_future
import concurrent.futures

import katsdpcal
from .reduction import pipeline
from .report import make_cal_report
from . import calprocs


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
    def __init__(self, buffer_index):
        self.buffer_index = buffer_index


class SensorReadingEvent(object):
    """An update to a sensor sent to the master"""
    def __init__(self, name, reading):
        self.name = name
        self.reading = reading


class QueueObserver(object):
    """katcp Sensor observer that forwards updates to a queue"""
    def __init__(self, queue):
        self._queue = queue

    def update(self, sensor, reading):
        self._queue.put(SensorReadingEvent(sensor.name, reading))


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


class Task(object):
    """Base class for tasks (threads or processes).

    It manages katcp sensors that are sent back to the master process over a
    :class:`multiprocessing.Queue`. It is intended to be subclassed to provide
    :meth:`get_sensors` and :meth:`run` methods.

    Parameters
    ----------
    task_class : type
        Either :class:`multiprocessing.Process` or an equivalent class such as
        :class:`multiprocessing.dummy.Process`.
    master_queue : :class:`multiprocessing.Queue`
        Queue for sending sensor updates to the master.
    name : str, optional
        Name for the task

    Attributes
    ----------
    master_queue : :class:`multiprocessing.Queue`
        Queue passed to the constructor
    sensors : dict
        Dictionary of :class:`katcp.Sensor`s. This is only guaranteed to be
        present inside the child process.
    """

    def __init__(self, task_class, master_queue, name=None):
        self.master_queue = master_queue
        self._process = task_class(target=_run_task, name=name, args=(self,))
        self.sensors = None
        # Expose assorted methods from the base class
        for key in ['start', 'terminate', 'join', 'name', 'is_alive']:
            if hasattr(self._process, key):
                setattr(self, key, getattr(self._process, key))

    def _run(self):
        sensors = self.get_sensors()
        observer = QueueObserver(self.master_queue)
        for sensor in sensors:
            sensor.attach(observer)
        self.sensors = {sensor.name: sensor for sensor in sensors}
        self.run()

    def get_sensors(self):
        """Get list of katcp sensors.

        The sensors should be instantiated when this function is called, not
        cached.
        """
        return []

    @property
    def daemon(self):
        return self._process.daemon

    @daemon.setter
    def daemon(self, value):
        self._process.daemon = value


def _run_task(task):
    """Free function wrapping the Task runner. It needs to be free because
    bound instancemethods can't be pickled for multiprocessing.
    """
    task._run()


# ---------------------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------------------

class Accumulator(object):
    """Manages accumulation of L0 data into buffers"""

    def __init__(self, buffers,
                 accum_pipeline_queue, pipeline_accum_sems,
                 l0_endpoint, l0_interface_address, telstate):
        self.buffers = buffers
        self.telstate = telstate
        self.l0_endpoint = l0_endpoint
        self.l0_interface_address = l0_interface_address
        self.accum_pipeline_queue = accum_pipeline_queue
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
        # Next buffer index for accumulation
        self._current_buffer = -1

        # Get data shape
        buffer_shape = buffers[0]['vis'].shape
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

        # Sensors for the katcp server to report
        sensors = [
            katcp.Sensor.boolean(
                'accumulator-capture-active',
                'whether an observation is in progress',
                default=False, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.integer(
                'accumulator-observations',
                'number of observations completed by the accumulator',
                default=0, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.integer(
                'accumulator-batches',
                'number of batches completed by the accumulator',
                default=0, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.integer(
                'accumulator-input-heaps',
                'number of L0 heaps received',
                default=0, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.float(
                'accumulator-buffer-filled',
                'fraction of buffer that the current accumulation has written to',
                params=(0.0, 1.0),
                default=0.0, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.float(
                'accumulator-last-wait',
                'time the accumulator had to wait for a free buffer',
                unit='s')
        ]
        self.sensors = {sensor.name: sensor for sensor in sensors}

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
            obs_stopped = False
            batches_sensor = self.sensors['accumulator-batches']
            wait_sensor = self.sensors['accumulator-last-wait']
            while not obs_stopped:
                # Increment the current buffer
                self._current_buffer = (self._current_buffer + 1) % self.num_buffers
                # ------------------------------------------------------------
                # Loop through the buffers and send data to pipeline task when
                # accumulation terminate conditions are met.

                logger.info('waiting for pipeline_accum_sems[%d]', self._current_buffer)
                loop = trollius.get_event_loop()
                now = loop.time()
                yield From(loop.run_in_executor(
                    self._executor, self.pipeline_accum_sems[self._current_buffer].acquire))
                elapsed = loop.time() - now
                logger.info('pipeline_accum_sems[%d] acquired by %s (%.3fs)',
                            self._current_buffer, self.name, elapsed)
                wait_status = katcp.Sensor.NOMINAL if elapsed < 1.0 else katcp.Sensor.WARN
                wait_sensor.set_value(elapsed, status=wait_status)

                # accumulate data scan by scan into buffer arrays
                logger.info('max buffer length %d', self.max_length)
                logger.info('accumulating into buffer %d', self._current_buffer)
                max_ind, obs_stopped = yield From(self.accumulate(rx, self._current_buffer))
                logger.info('Accumulated %d timestamps', max_ind+1)
                batches_sensor.set_value(batches_sensor.value() + 1)

                # pass the buffer to the pipeline
                self.accum_pipeline_queue.put(BufferReadyEvent(self._current_buffer))
                logger.info('accum_pipeline_queue updated by %s', self.name)

            # Tell the pipeline that the observation ended, but only if there
            # was something to work on.
            if self._obs_end is not None:
                self.accum_pipeline_queue.put(
                    ObservationEndEvent(index, self._obs_start, self._obs_end))
                obs_sensor = self.sensors['accumulator-observations']
                obs_sensor.set_value(obs_sensor.value() + 1)
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
        self.sensors['accumulator-capture-active'].set_value(True)
        self.sensors['accumulator-input-heaps'].set_value(0)

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
            self.sensors['accumulator-capture-active'].set_value(False)

    @trollius.coroutine
    def stop(self, force=False):
        """Shuts down the accumulator.

        If `force` is true, this assumes that the pipeline has already been
        terminated, and it does not try to wake it up; otherwise it sends
        it a stop event.
        """
        if force:
            # Ensure that the semaphores aren't blocked, even if the pipeline
            # has been terminated without releasing the buffers.
            for sem in self.pipeline_accum_sems:
                sem.release()
        if self._run_future is not None:
            yield From(self.capture_done())
        if not force:
            if self.accum_pipeline_queue is not None:
                self.accum_pipeline_queue.put(StopEvent())
        self.accum_pipeline_queue = None    # Make safe for concurrent calls to stop
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def set_ordering_parameters(self):
        # determine re-ordering necessary to convert from supplied bls
        # ordering to desired bls ordering
        antlist = self.telstate.cal_antlist
        self.ordering, bls_order, pol_order = \
            calprocs.get_reordering(antlist, self.telstate.sdp_l0_bls_ordering)
        # determine lookup list for baselines
        bls_lookup = calprocs.get_bls_lookup(antlist, bls_order)
        # save these to the telescope state for use in the pipeline/elsewhere
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
        Accumulates spead data into arrays, until accumulation end condition is reached:
         * case 1 -- actitivy change (unless gain cal following target)
         * case 2 -- beamformer phase up ended
         * case 3 -- buffer capacity limit reached
         * case 4 -- time limit reached (may be replaced later?)

        SPEAD item groups contain:
           correlator_data
           flags
           weights
           weights_channel
           timestamp

        Parameters
        ----------
        rx : :class:`spead2.recv.trollius.Stream`
            Receiver for L0 stream
        buffer_index : int
            Index into :attr:`buffers` for current buffer

        Returns
        -------
        array_index : int
            Last filled position in buffer
        obs_stopped : bool
            Whether the return was due to stream stopping
        """

        start_flag = True
        array_index = -1
        fill_sensor = self.sensors['accumulator-buffer-filled']
        fill_sensor.set_value(0.0)
        heaps_sensor = self.sensors['accumulator-input-heaps']

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
            updated = ig.update(heap)
            if not updated:
                logger.info('==== empty heap received ====')
                continue
            have_items = True
            for key in ('timestamp', 'correlator_data', 'flags', 'weights', 'weights_channel'):
                if key not in updated:
                    logger.warn('heap received without %s', key)
                    have_items = False
                    break
            if not have_items:
                continue

            # get activity and target tag from telescope state
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

            # get target time from telescope state, if it is present (if it
            # isn't present, set to unknown)
            try:
                target = self.telstate.get_range(target_key, et=data_ts,
                                                 include_previous=True)[0][0]
                if target == '':
                    target = 'unknown'
            except KeyError:
                logger.warning('target description %s absent from telescope state', target_key)
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
                logger.info(' - %s (%s)', target_name, activity)
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
            fill_sensor.set_value((array_index + 1.0) / self.max_length)
            heaps_sensor.set_value(heaps_sensor.value() + 1)

            # **************** ACCUMULATOR BREAK CONDITIONS ****************
            # ********** THIS BREAKING NEEDS TO BE THOUGHT THROUGH CAREFULLY **********
            # CASE 1 -- break if activity has changed (i.e. the activity time has changed)
            #   unless previous scan was a target, in which case accumulate
            #   subsequent gain scan too
            ignore_states = ['slew', 'stop', 'unknown']
            if (activity_time != prev_activity_time) \
                    and not np.any([ignore in prev_activity for ignore in ignore_states]) \
                    and ('unknown' not in target_tags) \
                    and ('target' not in prev_target_tags):
                logger.info('Accumulation break - transition %s -> %s', prev_activity, activity)
                break

            # CASE 2 -- beamformer special case
            if (activity_time != prev_activity_time) \
                    and ('single_accumulation' in prev_target_tags):
                logger.info('Accumulation break - single scan accumulation')
                break

            # CASE 3 -- end accumulation if maximum array size has been accumulated
            if array_index >= self.max_length - 1:
                logger.warn('Accumulate break - buffer size limit %d', self.max_length)
                break

            # CASE 4 -- temporary mock up of a natural break in the data stream
            # may ultimately be provided by some sort of sensor?
            duration = ig['timestamp'].value - unsync_start_time
            if duration > 2000000:
                logger.warn('Accumulate break due to duration')
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

class Pipeline(Task):
    """
    Task (Process or Thread) which runs pipeline
    """

    def __init__(self, task_class, data,
                 accum_pipeline_queue, pipeline_accum_sems, pipeline_report_queue, master_queue,
                 l1_endpoint, l1_level, l1_rate, telstate):
        super(Pipeline, self).__init__(task_class, master_queue, 'Pipeline')
        self.data = data
        self.accum_pipeline_queue = accum_pipeline_queue
        self.pipeline_accum_sems = pipeline_accum_sems
        self.pipeline_report_queue = pipeline_report_queue
        self.telstate = telstate
        self.l1_level = l1_level
        self.l1_rate = l1_rate
        self.l1_endpoint = l1_endpoint

    def get_sensors(self):
        return [
            katcp.Sensor.float(
                'pipeline-last-time',
                'time taken to process the most recent buffer',
                unit='s'),
            katcp.Sensor.integer(
                'pipeline-last-slots',
                'number of slots filled in the most recent buffer')
        ]

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
                    logger.info('buffer %d acquired by %s', event.buffer_index, self.name)
                    # run the pipeline
                    start_time = time.time()
                    data = self.data[event.buffer_index]
                    self.run_pipeline(data)
                    end_time = time.time()
                    elapsed = end_time - start_time
                    self.sensors['pipeline-last-time'].set_value(elapsed, timestamp=end_time)
                    self.sensors['pipeline-last-slots'].set_value(
                        data['max_index'][0] + 1, timestamp=end_time)
                    # release condition after pipeline run finished
                    self.pipeline_accum_sems[event.buffer_index].release()
                    logger.info('pipeline_accum_sems[%d] release by %s',
                                event.buffer_index, self.name)
                elif isinstance(event, ObservationEndEvent):
                    self.pipeline_report_queue.put(event)
                elif isinstance(event, StopEvent):
                    logger.info('stop received by %s', self.name)
                    break
                else:
                    logger.error('unknown event type %r by %s', event, self.name)
        finally:
            self.pipeline_report_queue.put(StopEvent())

    def run_pipeline(self, data):
        # run pipeline calibration, if more than zero timestamps accumulated
        target_slices = pipeline(data, self.telstate, task_name=self.name) \
            if (data['max_index'][0] > 0) else []

        # send data to L1 SPEAD if necessary
        if self.l1_level != 0:
            config = spead2.send.StreamConfig(max_packet_size=8972, rate=self.l1_rate)
            tx = spead2.send.UdpStream(spead2.ThreadPool(), self.l1_endpoint.host,
                                       self.l1_endpoint.port, config)
            logger.info('   Transmit L1 data')
            # for streaming all of the data (not target only),
            # use the highest index in the buffer that is filled with data
            transmit_slices = [slice(0, data['max_index'][0] + 1)] \
                if self.l1_level == 2 else target_slices
            self.data_to_spead(transmit_slices, tx)
            logger.info('   End transmit of L1 data')

    def data_to_spead(self, target_slices, tx):
        """
        Transmits data as SPEAD stream

        Parameters
        ----------
        target_slices : list of slices
            slices for target scans in the data buffer
        tx : :class:`spead2.send.UdpStream'
            SPEAD transmitter
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


# ---------------------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------------------

class ReportWriter(Task):
    def __init__(self, task_class, pipeline_report_queue, master_queue,
                 telstate, l1_endpoint, l1_level,
                 report_path, log_path, full_log):
        super(ReportWriter, self).__init__(task_class, master_queue, 'ReportWriter')
        if not report_path:
            report_path = '.'
        report_path = os.path.abspath(report_path)
        self.pipeline_report_queue = pipeline_report_queue
        self.telstate = telstate
        self.l1_endpoint = l1_endpoint
        self.l1_level = l1_level
        self.report_path = report_path
        self.log_path = log_path
        self.full_log = full_log
        # get subarray ID
        self.subarray_id = self.telstate.get('subarray_product_id', 'unknown_subarray')

    def get_sensors(self):
        return [
            katcp.Sensor.integer(
                'reports-written', 'Number of calibration reports written',
                default=0, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.float(
                'report-last-time', 'Elapsed time to generate most recent report',
                unit='s'),
            katcp.Sensor.string(
                'report-last-path', 'Directory containing the most recent report')
        ]

    def write_report(self, obs_start, obs_end):
        now = time.time()
        # get observation name
        try:
            obs_params = self.telstate.get_range('obs_params', st=0, et=obs_end,
                                                 return_format='recarray')
            obs_keys = obs_params['value']
            # choose most recent experiment id (last entry in the list), if
            # there are more than one
            experiment_id_string = [x for x in obs_keys if 'experiment_id' in x][-1]
            experiment_id = eval(experiment_id_string.split()[-1])
        except (TypeError, KeyError, AttributeError):
            # TypeError, KeyError because this isn't properly implemented yet
            # AttributeError in case this key isnt in the telstate for whatever reason
            experiment_id = '{0}_unknown_project'.format(int(now))

        # make directory for this observation, for logs and report
        obs_dir = '{0}/{1}_{2}_{3}'.format(
            self.report_path, int(now), self.subarray_id, experiment_id)
        current_obs_dir = '{0}-current'.format(obs_dir)
        try:
            os.mkdir(current_obs_dir)
        except OSError:
            logger.warning('Experiment ID directory %s already exists', current_obs_dir)

        # create pipeline report (very basic at the moment)
        try:
            make_cal_report(self.telstate, current_obs_dir, experiment_id, st=obs_start, et=obs_end)
        except Exception as error:
            logger.warn('Report generation failed: %s', error, exc_info=True)

        if self.l1_level != 0:
            # send L1 stop transmission
            #   wait for a couple of secs before ending transmission, because
            #   it's a separate kernel socket and hence unordered with respect
            #   to the socket used by the pipeline (TODO: move this into the
            #   pipeline).
            time.sleep(2.0)
            end_transmit(self.l1_endpoint.host, self.l1_endpoint.port)
            logger.info('L1 stream ended')

        logger.info('   Observation ended')
        logger.info('===========================')

        if self.full_log is not None:
            shutil.copy('{0}/{1}'.format(self.log_path, self.full_log),
                        '{0}/{1}'.format(current_obs_dir, self.full_log))

        # change report and log directory to final name for archiving
        shutil.move(current_obs_dir, obs_dir)
        logger.info('Moved observation report to %s', obs_dir)
        return obs_dir

    def run(self):
        reports_sensor = self.sensors['reports-written']
        report_time_sensor = self.sensors['report-last-time']
        report_path_sensor = self.sensors['report-last-path']
        while True:
            event = self.pipeline_report_queue.get()
            if isinstance(event, StopEvent):
                break
            elif isinstance(event, ObservationEndEvent):
                logger.info('Starting report number %d', event.index)
                start_time = time.time()
                obs_dir = self.write_report(event.start_time, event.end_time)
                end_time = time.time()
                reports_sensor.set_value(reports_sensor.value() + 1, timestamp=end_time)
                report_time_sensor.set_value(end_time - start_time, timestamp=end_time)
                report_path_sensor.set_value(obs_dir, timestamp=end_time)
            else:
                logger.error('unknown event type %r', event)
        logger.info('Pipeline has finished, exiting')


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
    VERSION_INFO = ('katsdpcal-api', 1, 0)
    BUILD_INFO = ('katsdpcal',) + tuple(katsdpcal.__version__.split('.', 1)) + ('',)

    def __init__(self, accumulator, pipeline, report_writer, master_queue, *args, **kwargs):
        self.accumulator = accumulator
        self.pipeline = pipeline
        self.report_writer = report_writer
        self.master_queue = master_queue
        self._stopping = False
        super(CalDeviceServer, self).__init__(*args, **kwargs)

    def setup_sensors(self):
        for sensor in self.accumulator.sensors.values():
            self.add_sensor(sensor)
        for sensor in self.pipeline.get_sensors():
            self.add_sensor(sensor)
        for sensor in self.report_writer.get_sensors():
            self.add_sensor(sensor)

    def start(self):
        self._run_queue_task = trollius.ensure_future(self._run_queue())
        super(CalDeviceServer, self).start()

    @trollius.coroutine
    def join(self):
        yield From(self._run_queue_task)

    @request()
    @return_reply()
    def request_capture_init(self, msg):
        """Start an observation"""
        if self.accumulator.capturing:
            return ('fail', 'capture already in progress')
        if self._stopping:
            return ('fail', 'server is shutting down')
        self.accumulator.capture_init()
        return ('ok',)

    @request()
    @return_reply()
    @concurrent_reply
    @tornado.gen.coroutine
    def request_capture_done(self, msg):
        """Stop the current observation"""
        if not self.accumulator.capturing:
            raise tornado.gen.Return(('fail', 'no capture in progress'))
        yield to_tornado_future(self.accumulator.capture_done())
        raise tornado.gen.Return(('ok',))

    @trollius.coroutine
    def shutdown(self, force=False, conn=None):
        """Shut down the server.

        This is a potentially long-running operation, particularly if `force`
        is false. While it is running, no new capture sessions can be started.

        If `conn` is given, it is updated with progress of the shutdown.
        """
        def progress(msg):
            logger.info(msg)
            if conn is not None:
                conn.inform(msg)
        loop = trollius.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            self._stopping = True
            if force:
                logger.warn('Forced shutdown - data may be lost')
            else:
                logger.info('Shutting down gracefully')
            if not force:
                yield From(self.accumulator.stop())
                progress('Accumulator stopped')
                yield From(loop.run_in_executor(executor, self.pipeline.join))
                progress('Pipeline stopped')
                yield From(loop.run_in_executor(executor, self.report_writer.join))
                progress('Report writer stopped')
            elif hasattr(self.report_writer, 'terminate'):
                # Kill off all the tasks. This is done in reverse order, to avoid
                # triggering a report writing only to kill it half-way.
                # TODO: this may need to become semi-graceful at some point, to avoid
                # corrupting an in-progress report.
                for task in [self.report_writer, self.pipeline]:
                    task.terminate()
                    yield From(loop.run_in_executor(executor, task.join))
                yield From(self.accumulator.stop(force=True))
            else:
                logger.warn('Cannot force kill tasks, because they are threads')
        self.master_queue.put(StopEvent())
        # Wait until all pending sensor updates have been applied
        yield From(self.join())

    @request(Bool(optional=True, default=False))
    @return_reply()
    @concurrent_reply
    @tornado.gen.coroutine
    def request_shutdown(self, req, force=False):
        """Shut down the server.

        This is a potentially long-running operation, particularly if `force`
        is false. While it is running, no new capture sessions can be started.
        It is possible to make concurrent requests while this request is in
        progress, but it will not be possible to start new observations.

        This does not directly stop the server, but it does cause
        :meth:`_run_queue` to exit, which causes :file:`run_cal.py` to shut down.

        Parameters
        ----------
        force : bool, optional
            If true, terminate processes immediately rather than waiting for
            them to finish pending work. This can cause data loss!
        """
        yield to_tornado_future(trollius.ensure_future(self.shutdown(force, req)))
        raise tornado.gen.Return(('ok',))

    @trollius.coroutine
    def _run_queue(self):
        """Process all events sent to the master queue, until stopped by :meth:`shutdown`."""
        loop = trollius.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            while True:
                event = yield From(loop.run_in_executor(executor, self.master_queue.get))
                if isinstance(event, StopEvent):
                    break
                elif isinstance(event, SensorReadingEvent):
                    try:
                        sensor = self.get_sensor(event.name)
                    except ValueError:
                        logger.warn('Received update for unknown sensor %s', event.name)
                    else:
                        sensor.set(event.reading.timestamp,
                                   event.reading.status,
                                   event.reading.value)
                else:
                    logger.warn('Unknown event %r', event)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # When used as a context manager, a server will ensure its child
        # processes are killed.
        for task in [self.report_writer, self.pipeline]:
            if task.is_alive() and hasattr(task, 'terminate'):
                task.terminate()


def create_buffer_arrays(buffer_shape, use_multiprocessing=True):
    """
    Create empty buffer record using specified dimensions
    """
    if use_multiprocessing:
        factory = shared_empty
    else:
        factory = np.empty
    data = {}
    data['vis'] = factory(buffer_shape, dtype=np.complex64)
    data['flags'] = factory(buffer_shape, dtype=np.uint8)
    data['weights'] = factory(buffer_shape, dtype=np.float32)
    data['times'] = factory(buffer_shape[0], dtype=np.float)
    data['max_index'] = factory([1], dtype=np.int32)
    data['max_index'][0] = 0
    return data


def create_server(use_multiprocessing, host, port, buffers,
                  l0_endpoint, l0_interface_address,
                  l1_endpoint, l1_level, l1_rate, telstate,
                  report_path, log_path, full_log):
    # threading or multiprocessing imports
    if use_multiprocessing:
        logger.info("Using multiprocessing")
        import multiprocessing
    else:
        logger.info("Using threading")
        import multiprocessing.dummy as multiprocessing

    num_buffers = len(buffers)
    # set up inter-task synchronisation primitives.
    # passed events to indicate buffer transfer, end-of-observation, or stop
    accum_pipeline_queue = multiprocessing.Queue()
    # signalled when the pipeline is finished with a buffer
    pipeline_accum_sems = [multiprocessing.Semaphore(value=1) for i in range(num_buffers)]
    # signalled by pipelines when they shut down or finish an observation
    pipeline_report_queue = multiprocessing.Queue()
    # other tasks send up sensor updates
    master_queue = multiprocessing.Queue()

    # Set up the pipelines (one per buffer)
    pipeline = Pipeline(
        multiprocessing.Process, buffers,
        accum_pipeline_queue, pipeline_accum_sems, pipeline_report_queue, master_queue,
        l1_endpoint, l1_level, l1_rate, telstate)
    # Set up the report writer
    report_writer = ReportWriter(
        multiprocessing.Process, pipeline_report_queue, master_queue, telstate,
        l1_endpoint, l1_level, report_path, log_path, full_log)

    # Start the child tasks.
    running_tasks = []
    try:
        for task in [report_writer, pipeline]:
            if not use_multiprocessing:
                task.daemon = True    # Make sure it doesn't prevent process exit
            task.start()
            running_tasks.append(task)

        # Set up the accumulator. This is done after the other processes are
        # started, because it creates a ThreadPoolExecutor, and threads and fork()
        # don't play nicely together.
        accumulator = Accumulator(buffers,
                                  accum_pipeline_queue, pipeline_accum_sems,
                                  l0_endpoint, l0_interface_address, telstate)
        return CalDeviceServer(accumulator, pipeline, report_writer, master_queue, host, port)
    except Exception:
        for task in running_tasks:
            if hasattr(task, 'terminate'):
                task.terminate()
                task.join()
        raise
