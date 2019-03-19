import time
import mmap
import os
import shutil
import logging
from collections import deque, namedtuple, Counter
import multiprocessing
import multiprocessing.dummy
import cProfile
import json

import spead2
import spead2.recv
import spead2.recv.trollius
import spead2.send

import katcp
from katcp.kattypes import request, return_reply, concurrent_reply, Bool, Str
from katdal.h5datav3 import FLAG_NAMES
import katdal.datasources
from katdal import SpectralWindow

import attr
import enum
import numpy as np
import dask.array as da
import dask.diagnostics
import dask.distributed
import trollius
from trollius import From, Return
import tornado.gen
from katsdpservices.asyncio import to_tornado_future
import concurrent.futures

import katsdpcal
from .reduction import pipeline
from .report import make_cal_report
from . import calprocs, calprocs_dask
from . import solutions


logger = logging.getLogger(__name__)


class State(enum.Enum):
    """State of a single capture block"""
    CAPTURING = 1         # capture-init has been called, but not capture-done
    PROCESSING = 2        # capture-done has been called, but still in the pipeline
    REPORTING = 3         # generating the report
    DEAD = 4              # completely finished


class EnumEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, enum.Enum):
            return obj.name
        return json.JSONEncoder.default(self, obj)


class ObservationEndEvent(object):
    """An observation has finished upstream"""
    def __init__(self, capture_block_id, start_time, end_time):
        self.capture_block_id = capture_block_id
        self.start_time = start_time
        self.end_time = end_time


class ObservationStateEvent(object):
    """An observation has changed state.

    This is sent from each component to the master queue to update the
    katcp sensor.
    """
    def __init__(self, capture_block_id, state):
        self.capture_block_id = capture_block_id
        self.state = state


class StopEvent(object):
    """Graceful shutdown requested"""


class BufferReadyEvent(object):
    """Transfers ownership of buffer slots."""
    def __init__(self, capture_block_id, slots):
        self.capture_block_id = capture_block_id
        self.slots = slots


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


ActivityState = namedtuple('ActivityState',
                           ['activity', 'activity_time', 'target_name', 'target_tags'])


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


def _inc_sensor(sensor, delta, status=katcp.Sensor.NOMINAL, timestamp=None):
    """Increment sensor value by `delta`."""
    sensor.set_value(sensor.value() + delta, status, timestamp)


def _slots_slices(slots):
    """Compresses a list of slot positions to a list of ranges (given as slices).

    This is a generator that yields the slices

    Example
    -------
    >>> list(_slots_slices([2, 3, 4, 6, 7, 8, 0, 1]))
    [slice(2, 5, None), slice(6, 9, None), slice(0, 2, None)]
    """
    start = None
    end = None
    for slot in slots:
        if end is not None and slot != end:
            yield slice(start, end)
            start = end = None
        if start is None:
            start = slot
        end = slot + 1
    if end is not None:
        yield slice(start, end)


def _sum_corr(sum_corr, new_corr, limit=None):
    """
    Combines a dictionary of corrected data produced by the pipeline into a dictionary
    which aggregates the corrected data produced throughout the observation.

    Parameters
    ----------
    sum_corr : dict of lists
        lists of combined data from pipeline
    corr_data : dict of lists
        lists contain most recent data produced by pipeline
    limit : int, optional
        If given, truncate the arrays at this many elements

    Returns
    -------
    dict of lists
        for all keys except 't_flags', output dictionary list contains all the elements
        of the two input dictionary lists on a per key basis.
        For key 't_flags', the output dictionary contains a single element which is the sum
        of the sum_corr['t_flags'] and new_corr['t_flags']
        For keys containing '_g_spec', the output dictionary is a list containing the weighted
        average of the current pipeline outputs and the previous outputs.
    """
    # list of keys which don't append data on a per scan basis
    keylist = ['t_flags']
    if sum_corr:
        for key in new_corr.keys():
            # sum the per scan sum of flags
            if key is 't_flags':
                sum_corr[key] += new_corr[key]
                sum_corr[key] = [sum(sum_corr[key])]
                del new_corr[key]

            # take the weighted average over all the gain calibrated scans
            elif key.endswith('_g_spec'):
                sum_corr[key] += new_corr[key]
                wavg = zip(*sum_corr[key])
                vis, flags, weights = [da.stack(a) for a in wavg]
                vis, flags, weights = calprocs_dask.wavg_full(vis, flags, weights, threshold=1)
                sum_corr[key] = [(vis, flags, weights)]
                del new_corr[key]
                # add this key to the list
                keylist.append(key)

            else:
                sum_corr[key] += new_corr[key]
                del new_corr[key]

    # if input dictionary is empty set it to pipeline output dictionary
    else:
        sum_corr = new_corr

    if limit is not None:
        if len(sum_corr['targets']) > limit:
            # find the time of the scan at the limit
            last_time = sum_corr['targets'][limit][1]
            # truncate arrays with times beyond the limit
            for key in sum_corr.keys():
                if key not in keylist:
                    vals, times = zip(*sum_corr[key])
                    time_limit = next((i for i, t in enumerate(times) if t >= last_time), None)
                    if time_limit is not None:
                        del sum_corr[key][time_limit:]

    return sum_corr


def make_telstate_cb(telstate, capture_block_id):
    """Create a telstate view that is capture-block specific.

    This adds two views to the base telstate:

    - <`capture_block_id`>_<first prefix of `telstate`>
    - <`capture_block_id`>

    The first, more specific, namespace will receive updated keys while the
    second namespace provides read access to the main capture block namespace
    for keys like obs sensors.

    Parameters
    ----------
    telstate : :class:`katsdptelstate.TelescopeState`
        Base telescope state
    capture_block_id : str
        Capture block ID

    Returns
    -------
    telstate_cb : :class:`katsdptelstate.TelescopeState`
        Telescope state with `capture_block_id` prepended to the first prefix
    """
    prefix = telstate.SEPARATOR.join([capture_block_id, telstate.prefixes[0]])
    return telstate.view(capture_block_id).view(prefix)


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
    profile_file : str, optional
        Output filename for a cProfile profile of the :meth:`run` method

    Attributes
    ----------
    master_queue : :class:`multiprocessing.Queue`
        Queue passed to the constructor
    sensors : dict
        Dictionary of :class:`katcp.Sensor`s. This is only guaranteed to be
        present inside the child process.
    """

    def __init__(self, task_class, master_queue, name=None, profile_file=None):
        self.master_queue = master_queue
        self.profile_file = profile_file
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
        try:
            if self.profile_file is not None:
                profile = cProfile.Profile()
                profile.enable()
            self.run()
        finally:
            if self.profile_file is not None:
                profile.create_stats()
                profile.dump_stats(self.profile_file)
                logger.info('Wrote profile to %s', self.profile_file)

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


class Accumulator(object):
    """Manages accumulation of L0 data into buffers"""

    def __init__(self, buffers, accum_pipeline_queue, master_queue,
                 l0_name, l0_endpoints, l0_interface_address, telstate_cal, parameters):
        self.buffers = buffers
        self.telstate = telstate_cal.root()
        self.telstate_cal = telstate_cal
        self.telstate_cb_cal = None    # Capture block-specific view
        self.l0_name = l0_name
        self.l0_interface_address = l0_interface_address
        self.accum_pipeline_queue = accum_pipeline_queue
        self.master_queue = master_queue

        # Extract useful parameters from telescope state
        self.telstate_l0 = self.telstate.view(l0_name)
        self.parameters = parameters
        self.int_time = self.telstate_l0['int_time']
        self.sync_time = self.telstate_l0['sync_time']
        self.l0_excise = self.telstate_l0['excise']
        self.need_weights_power_scale = self.telstate_l0.get('need_weights_power_scale', False)
        self.set_ordering_parameters()

        self.name = 'Accumulator'
        self._rx = None
        self._run_future = None
        self._reset_capture_state()

        # Get data shape
        buffer_shape = buffers['vis'].shape
        self.max_length = buffer_shape[0] // 2   # Ensures at least double buffering
        self.nslots = buffer_shape[0]
        self.nchan = buffer_shape[1]
        self.npol = buffer_shape[2]
        self.nbl = buffer_shape[3]

        # Free space tracking
        self._free_slots = deque(range(buffer_shape[0]))
        self._slots_cond = trollius.Condition()  # Signalled when new slots are available
        # Set if stop(force=True) is called, to abort waiting for an available slot
        self._force_stopping = False
        # Enforce only one capture_done at a time
        self._done_lock = trollius.Lock()

        # Allocate storage and thread pool for receiver
        # Main data is 10 bytes per entry: 8 for vis, 1 for flags, 1 for weights.
        # Then there are per-channel weights (4 bytes each).
        stream_n_chans = self.telstate_l0['n_chans']
        stream_n_bls = self.telstate_l0['n_bls']
        stream_n_chans_per_substream = self.telstate_l0['n_chans_per_substream']
        self.n_substreams = stream_n_chans // stream_n_chans_per_substream
        heap_size = (stream_n_chans_per_substream * stream_n_bls * 10
                     + stream_n_chans_per_substream * 4)
        self._thread_pool = spead2.ThreadPool()
        self._memory_pool = spead2.MemoryPool(heap_size, heap_size + 4096,
                                              4 * self.n_substreams, 4 * self.n_substreams)

        if stream_n_chans % len(l0_endpoints):
            raise ValueError('Number of channels ({}) not a multiple of number of endpoints ({})'
                             .format(stream_n_chans, len(l0_endpoints)))
        self.l0_endpoints = []
        for i, endpoint in enumerate(l0_endpoints):
            start = i * stream_n_chans // len(l0_endpoints)
            stop = (i + 1) * stream_n_chans // len(l0_endpoints)
            if (start < parameters['channel_slice'].stop
                    and stop > parameters['channel_slice'].start):
                self.l0_endpoints.append(endpoint)

        # Sensors for the katcp server to report
        sensors = [
            katcp.Sensor.boolean(
                'accumulator-capture-active',
                'whether an observation is in progress (prometheus: gauge)',
                default=False, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.integer(
                'accumulator-observations',
                'number of observations completed by the accumulator (prometheus: counter)',
                default=0, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.integer(
                'accumulator-batches',
                'number of batches completed by the accumulator (prometheus: counter)',
                default=0, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.integer(
                'input-bytes-total',
                'number of bytes of L0 data received (prometheus: counter)',
                default=0, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.integer(
                'input-heaps-total',
                'number of L0 heaps received (prometheus: counter)',
                default=0, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.integer(
                'input-incomplete-heaps-total',
                'number of incomplete L0 heaps received (prometheus: counter)',
                default=0, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.integer(
                'input-too-old-heaps-total',
                'number of L0 heaps rejected because they are too late (prometheus: counter)',
                default=0, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.integer(
                'slots',
                'total number of buffer slots (prometheus: gauge)',
                default=self.nslots, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.integer(
                'accumulator-slots',
                'number of buffer slots the current accumulation has written to '
                '(prometheus: gauge)',
                default=0, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.integer(
                'free-slots',
                'number of unused buffer slots (prometheus: gauge)',
                default=self.nslots, initial_status=katcp.Sensor.NOMINAL),
            # pipeline-slots gives information about the pipeline, but is
            # produced in the accumulator because the pipeline doesn't get
            # interrupted when more work is added to it.
            katcp.Sensor.integer(
                'pipeline-slots',
                'number of buffer slots in use by the pipeline (prometheus: gauge)',
                default=0, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.float(
                'accumulator-last-wait',
                'time the accumulator had to wait for a free buffer (prometheus: gauge)',
                unit='s')
        ]
        self.sensors = {sensor.name: sensor for sensor in sensors}

    @property
    def capturing(self):
        return self._rx is not None

    @trollius.coroutine
    def _next_slot(self):
        wait_sensor = self.sensors['accumulator-last-wait']
        with (yield From(self._slots_cond)):
            if self._force_stopping:
                raise Return(None)
            elif self._free_slots:
                wait_sensor.set_value(0.0)
            else:
                logger.warn('no slots available - waiting for pipeline to return buffers')
                loop = trollius.get_event_loop()
                now = loop.time()
                while not self._force_stopping and not self._free_slots:
                    yield From(self._slots_cond.wait())
                if self._force_stopping:
                    raise Return(None)
                elapsed = loop.time() - now
                logger.info('slot acquired')
                wait_sensor.set_value(elapsed, status=katcp.Sensor.WARN)
            slot = self._free_slots.popleft()
            now = time.time()
            status = katcp.Sensor.WARN if not self._free_slots else katcp.Sensor.NOMINAL
            _inc_sensor(self.sensors['free-slots'], -1, status, timestamp=now)
            _inc_sensor(self.sensors['accumulator-slots'], 1, timestamp=now)
            # Mark all flags as data_lost, so that any that aren't overwritten
            # by data will have this value.
            self.buffers['flags'][slot].fill(np.uint8(2 ** FLAG_NAMES.index('data_lost')))
            raise Return(slot)

    @trollius.coroutine
    def buffer_free(self, event):
        """Return slots to the free list.

        Parameters
        ----------
        event : :class:`BufferReadyEvent`
            Event listing the slots that are now available
        """
        if event.slots:
            with (yield From(self._slots_cond)):
                self._free_slots.extend(event.slots)
                now = time.time()
                _inc_sensor(self.sensors['free-slots'], len(event.slots), timestamp=now)
                _inc_sensor(self.sensors['pipeline-slots'], -len(event.slots), timestamp=now)
                self._slots_cond.notify()

    @trollius.coroutine
    def _run_observation(self, capture_block_id):
        """Runs for a single observation i.e., until a stop heap is received."""
        try:
            self._reset_capture_state(capture_block_id)
            yield From(self._accumulate(capture_block_id))
            # Tell the pipeline that the observation ended, but only if there
            # was something to work on.
            if self._obs_end is not None:
                self.master_queue.put(ObservationStateEvent(capture_block_id, State.PROCESSING))
                self.accum_pipeline_queue.put(
                    ObservationEndEvent(capture_block_id, self._obs_start, self._obs_end))
                _inc_sensor(self.sensors['accumulator-observations'], 1)
            else:
                logger.info(' --- no data flowed ---')
                # Send it twice, since the master expects it from both flag
                # sender and report writer.
                for i in range(2):
                    self.master_queue.put(ObservationEndEvent(capture_block_id, None, None))
            logger.info('Observation %s ended', capture_block_id)
        except trollius.CancelledError:
            logger.info('Observation %s cancelled', capture_block_id)
        except Exception as error:
            logger.error('Exception in capture: %s', error, exc_info=True)
        finally:
            self._rx.stop()

    def capture_init(self, capture_block_id):
        assert self._rx is None, "observation already running"
        assert self._run_future is None, "inconsistent state"
        logger.info('===========================')
        logger.info('   Starting new observation')
        # Prepend the CBID to the cal_name to form a new namespace
        self.telstate_cb_cal = make_telstate_cb(self.telstate_cal, capture_block_id)
        # Initialise SPEAD receiver
        logger.info('Initializing SPEAD receiver')
        rx = spead2.recv.trollius.Stream(
            self._thread_pool,
            max_heaps=2 * self.n_substreams, ring_heaps=self.n_substreams,
            contiguous_only=False)
        rx.set_memory_allocator(self._memory_pool)
        rx.set_memcpy(spead2.MEMCPY_NONTEMPORAL)
        rx.stop_on_stop_item = False
        for l0_endpoint in self.l0_endpoints:
            if self.l0_interface_address is not None:
                rx.add_udp_reader(l0_endpoint.host, l0_endpoint.port,
                                  interface_address=self.l0_interface_address,
                                  buffer_size=64 * 1024**2)
            else:
                rx.add_udp_reader(l0_endpoint.port, bind_hostname=l0_endpoint.host,
                                  buffer_size=64 * 1024**2)
        logger.info('reader added')
        self._rx = rx
        self._run_future = trollius.ensure_future(self._run_observation(capture_block_id))
        self.sensors['accumulator-capture-active'].set_value(True)
        self.sensors['input-bytes-total'].set_value(0)
        self.sensors['input-heaps-total'].set_value(0)
        self.sensors['input-incomplete-heaps-total'].set_value(0)
        self.sensors['input-too-old-heaps-total'].set_value(0)

    @trollius.coroutine
    def capture_done(self):
        assert self._rx is not None, "observation not running"
        assert self._run_future is not None, "inconsistent state"
        with (yield From(self._done_lock)):
            future = self._run_future
            # Give it a chance to stop on its own (from stop heaps)
            logger.info('Waiting for capture to finish (5s timeout)...')
            done, _ = yield From(trollius.wait([self._run_future], timeout=5))
            if future not in done:
                logger.info('Stopping receiver')
                self._rx.stop()
                logger.info('Waiting for capture to finish...')
                yield From(self._run_future)
            logger.info('Joined with _run_observation')
            self._run_future = None
            self._rx = None
            self.telstate_cb_cal = None
            self.sensors['accumulator-capture-active'].set_value(False)

    @trollius.coroutine
    def stop(self, force=False):
        """Shuts down the accumulator.

        If `force` is true, this assumes that the pipeline has already been
        terminated, and it does not try to wake it up; otherwise it sends
        it a stop event.
        """
        if force:
            self._force_stopping = True
            with (yield From(self._slots_cond)):
                # Interrupts wait for free slot, if any
                self._slots_cond.notify()

        if self._run_future is not None:
            yield From(self.capture_done())

        if not force and self.accum_pipeline_queue is not None:
            self.accum_pipeline_queue.put(StopEvent())
        self.accum_pipeline_queue = None    # Make safe for concurrent calls to stop
        if self._thread_pool is not None:
            self._thread_pool.stop()
            self._thread_pool = None

    def set_ordering_parameters(self):
        # determine re-ordering necessary to convert from supplied bls
        # ordering to desired bls ordering
        antenna_names = self.parameters['antenna_names']
        bls_ordering = self.telstate_l0['bls_ordering']
        self.ordering = calprocs.get_reordering(antenna_names, bls_ordering)[0]
        self.weights_power_scale_params = katdal.datasources.corrprod_to_autocorr(bls_ordering)

    def _reset_capture_state(self, capture_block_id=None):
        self._capture_block_id = capture_block_id
        # First and last timestamps in observation
        self._obs_start = None
        self._obs_end = None
        self._state = None
        self._first_timestamp = None
        self._last_idx = -1                # Last dump index that has a slot
        # List of slots that have been filled in for this batch
        self._slots = []
        # Look up slot by dump index
        self._slot_for_index = {}

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

    def _flush_slots(self):
        now = time.time()
        logger.info('Accumulated %d timestamps', len(self._slots))
        _inc_sensor(self.sensors['accumulator-batches'], 1, timestamp=now)

        # pass the buffer to the pipeline
        if self._slots:
            self.accum_pipeline_queue.put(BufferReadyEvent(self._capture_block_id, self._slots))
            logger.info('accum_pipeline_queue updated by %s', self.name)
            _inc_sensor(self.sensors['pipeline-slots'], len(self._slots), timestamp=now)
            _inc_sensor(self.sensors['accumulator-slots'], -len(self._slots), timestamp=now)

        self._slots = []
        self._slot_for_index.clear()
        self._state = None

    @trollius.coroutine
    def _next_heap(self, rx, ig):
        """Retrieve the next usable heap from `rx` and apply it to `ig`.

        Returns
        -------
        dict
            Keys that were updated in `ig`, or ``None`` for a stop heap

        Raises
        ------
        spead2.Stopped
            if the stream stopped
        """
        while True:
            heap = yield From(rx.get())
            if heap.is_end_of_stream():
                raise Return({})
            if isinstance(heap, spead2.recv.IncompleteHeap):
                logger.debug('dropped incomplete heap %d (%d/%d bytes of payload)',
                             heap.cnt, heap.received_length, heap.heap_length)
                _inc_sensor(self.sensors['input-incomplete-heaps-total'], 1,
                            status=katcp.Sensor.WARN)
                continue
            updated = ig.update(heap)
            if not updated:
                logger.info('==== empty heap received ====')
                continue
            have_items = True
            for key in ('dump_index', 'frequency',
                        'correlator_data', 'flags', 'weights', 'weights_channel'):
                if key not in updated:
                    logger.warn('heap received without %s', key)
                    have_items = False
                    break
            if not have_items:
                continue
            raise Return(updated)

    def _get_activity_state(self, data_ts):
        """Extract telescope state information about current activity.

        Parameters
        ----------
        data_ts : float
            Timestamp (UNIX time) for the query.

        Returns
        -------
        :class:`ActivityState`
            Current state, or ``None`` if no activity was recorded
        """
        try:
            activity, activity_time = self.telstate_cb_cal.get_range(
                'obs_activity', et=data_ts, include_previous=True)[0]
        except (KeyError, IndexError):
            logger.info('No obs_activity found in telstate - ignoring dump')
            return None
        # Get target from telescope state if present (otherwise 'unknown')
        try:
            target = self.telstate.get_range('cbf_target', et=data_ts,
                                             include_previous=True)[0][0]
        except (KeyError, IndexError):
            logger.warning('No cbf_target found in telstate')
            target = ''
        # Extract name and tags from target description string
        # XXX Rather use katpoint at some stage
        target_split = target.split(',')
        target_name = target_split[0] if target else 'unknown'
        target_tags = target_split[1] if len(target_split) > 1 else 'unknown'
        return ActivityState(activity, activity_time, target_name, target_tags)

    def _is_break(self, old, new, slots):
        """Determine whether to break batches between `old` and `new`:
         * case 1 -- activity change (unless gain cal following target)
         * case 2 -- beamformer phase up ended
         * case 3 -- buffer capacity limit reached

        Parameters
        ----------
        old, new : :class:`ActivityState`
            Encapsulated activity sensors for the previous and next dump
        slots : list
            Already accumulated slots (including `old` but not `new`)

        Returns
        -------
        bool
            Whether to insert a break between `old` and `new`
        """
        # **************** ACCUMULATOR BREAK CONDITIONS ****************
        # ********** THIS BREAKING NEEDS TO BE THOUGHT THROUGH CAREFULLY **********
        ignore_states = ['slew', 'stop', 'unknown']
        if new is not None and old is not None:
            # CASE 1 -- break if activity has changed (i.e. the activity time has changed)
            #   unless previous scan was a target, in which case accumulate
            #   subsequent gain scan too
            if (new.activity_time != old.activity_time) \
                    and not np.any([ignore in old.activity for ignore in ignore_states]) \
                    and ('unknown' not in new.target_tags) \
                    and ('target' not in old.target_tags):
                logger.info('Accumulation break - transition %s -> %s', old.activity, new.activity)
                return True

            # CASE 2 -- beamformer special case
            if (new.activity_time != old.activity_time) \
                    and ('single_accumulation' in old.target_tags):
                logger.info('Accumulation break - single scan accumulation')
                return True

        # CASE 3 -- end accumulation if maximum array size has been accumulated
        if len(slots) >= self.max_length:
            logger.warn('Accumulate break - buffer size limit %d', self.max_length)
            return True

        return False

    @trollius.coroutine
    def _ensure_slots(self, cur_idx):
        """Add new slots until there is one for `cur_idx`.

        This assumes that ``self._last_idx`` is less than `cur_idx`.

        Returns
        -------
        bool
            True if successful, False if we were interrupted by a force stop
        """
        for idx in range(self._last_idx + 1, cur_idx + 1):
            data_ts = self._first_timestamp + idx * self.int_time + self.sync_time
            if self._obs_start is None:
                self._obs_start = data_ts - 0.5 * self.int_time
            self._obs_end = data_ts + 0.5 * self.int_time

            # get activity and target tag from telescope state
            new_state = self._get_activity_state(data_ts)

            # if this is the first heap of the batch, log a header
            if self._state is None:
                logger.info('accumulating data from targets:')

            # flush a batch if necessary
            if self._is_break(self._state, new_state, self._slots):
                self._flush_slots()

            # print name of target and activity type on changes (and start of batch)
            if new_state is not None and self._state != new_state:
                logger.info(' - %s (%s)', new_state.target_name, new_state.activity)

            # Obtain a slot to copy to
            slot = yield From(self._next_slot())
            if slot is None:
                logger.info('Accumulation interrupted while waiting for a slot')
                raise Return(False)
            self._slots.append(slot)
            self._slot_for_index[idx] = slot
            self.buffers['times'][slot] = data_ts
            self.buffers['dump_indices'][slot] = idx
            self._state = new_state
            self._last_idx = idx
        raise Return(True)

    @trollius.coroutine
    def _accumulate(self, capture_block_id):
        """
        Accumulate SPEAD heaps into arrays and send batches to the pipeline.

        This does the main work of :meth:`_run_observation`, which just wraps
        it to handle cleanup at the end.

        SPEAD item groups contain:
           correlator_data
           flags
           weights
           weights_channel
           dump_index
        """

        rx = self._rx
        ig = spead2.ItemGroup()
        n_stop = 0                   # Number of stop heaps received
        telstate_cb_l0 = make_telstate_cb(self.telstate_l0, capture_block_id)

        # receive SPEAD stream
        logger.info('waiting to start accumulating data')
        while True:
            try:
                updated = yield From(self._next_heap(rx, ig))
            except spead2.Stopped:
                break
            if not updated:   # stop heap was received
                n_stop += 1
                if n_stop == len(self.l0_endpoints):
                    rx.stop()
                    break
                else:
                    continue

            if self._first_timestamp is None:
                self._first_timestamp = telstate_cb_l0['first_timestamp']
            data_idx = int(ig['dump_index'].value)  # Convert from np.uint64, which behaves oddly
            if data_idx < self._last_idx:
                try:
                    slot = self._slot_for_index[data_idx]
                    logger.info('Dump index went backwards (%d < %d), but managed to accept it',
                                data_idx, self._last_idx)
                except KeyError:
                    logger.warning('Dump index went backwards (%d < %d), skipping heap',
                                   data_idx, self._last_idx)
                    _inc_sensor(self.sensors['input-too-old-heaps-total'], 1,
                                status=katcp.Sensor.WARN)
                    continue
            else:
                # Create slots for all entries we haven't seen yet
                if not (yield From(self._ensure_slots(data_idx))):
                    break
                slot = self._slots[-1]

            channel0 = ig['frequency'].value
            # Range of channels provided by the heap (from full L0 range)
            src_range = slice(channel0, channel0 + ig['flags'].shape[0])
            # Range of channels in the buffer (from full L0 range)
            trg_range = self.parameters['channel_slice']
            # Intersection of the two
            common_range = slice(max(src_range.start, trg_range.start),
                                 min(src_range.stop, trg_range.stop))
            if common_range.start < common_range.stop:
                # Compute slice to apply to src/trg to get the common part
                src_subset = slice(common_range.start - src_range.start,
                                   common_range.stop - src_range.start)
                trg_subset = slice(common_range.start - trg_range.start,
                                   common_range.stop - trg_range.start)
                # reshape data and put into relevant arrays
                vis = ig['correlator_data'].value[src_subset]
                flags = ig['flags'].value[src_subset]
                weights_channel = ig['weights_channel'].value[src_subset, np.newaxis]
                weights = ig['weights'].value[src_subset] * weights_channel
                if self.need_weights_power_scale:
                    # weight_power_scale expects a time axis, hence newaxis
                    weights = weights[np.newaxis, ...]
                    katdal.datasources.weight_power_scale(
                        vis[np.newaxis, ...], weights, *self.weights_power_scale_params,
                        out=weights)
                    weights = weights[0]
                self._update_buffer(self.buffers['vis'][slot, trg_subset],
                                    vis, self.ordering)
                self._update_buffer(self.buffers['flags'][slot, trg_subset],
                                    flags, self.ordering)
                self._update_buffer(self.buffers['weights'][slot, trg_subset],
                                    weights, self.ordering)
                # Excise bits are computed from the original (L0) bls ordering,
                # not the cal ordering, because that's the order it's needed in
                # the flag sender.
                if self.l0_excise:
                    self.buffers['excise'][slot, trg_subset] = \
                        np.packbits(ig['flags'].value[src_subset], axis=-1)
                else:
                    self.buffers['excise'][slot, trg_subset] = 0
            heap_nbytes = 0
            for field in ['correlator_data', 'flags', 'weights', 'weights_channel']:
                heap_nbytes += ig[field].value.nbytes
            now = time.time()
            _inc_sensor(self.sensors['input-bytes-total'], heap_nbytes, timestamp=now)
            _inc_sensor(self.sensors['input-heaps-total'], 1, timestamp=now)

        # Need to ensure that all parallel cal servers get the same number of
        # dumps, and hence the same batches. Record the last index of each,
        # exchange, and take the largest.
        max_idx = self._last_idx
        loop = trollius.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            server_id = self.parameters['server_id']
            n_servers = self.parameters['servers']
            self.telstate_cb_cal.add('last_dump_index{}'.format(server_id),
                                     self._last_idx, immutable=True)
            for i in range(n_servers):
                if i != server_id:
                    key = 'last_dump_index{}'.format(i)
                    logger.debug('Waiting for %s', key)
                    yield From(loop.run_in_executor(
                        executor, self.telstate_cb_cal.wait_key, key))
                    other_last_idx = self.telstate_cb_cal[key]
                    logger.debug('Got %s = %d', key, other_last_idx)
                    max_idx = max(max_idx, other_last_idx)
        if max_idx > self._last_idx:
            logger.info('Adding %d extra slots to align end of observation',
                        max_idx - self._last_idx)
            yield From(self._ensure_slots(max_idx))

        # Flush out the final batch
        self._flush_slots()
        logger.info('Accumulation ended')


class Pipeline(Task):
    """
    Task (Process or Thread) which runs pipeline
    """

    def __init__(self, task_class, buffers,
                 accum_pipeline_queue, pipeline_sender_queue, pipeline_report_queue, master_queue,
                 l0_name, telstate_cal, parameters,
                 diagnostics=None, profile_file=None, num_workers=None):
        super(Pipeline, self).__init__(task_class, master_queue, 'Pipeline', profile_file)
        self.buffers = buffers
        self.accum_pipeline_queue = accum_pipeline_queue
        self.pipeline_sender_queue = pipeline_sender_queue
        self.pipeline_report_queue = pipeline_report_queue
        self.telstate_cal = telstate_cal
        self.parameters = parameters
        self.l0_name = l0_name
        self.diagnostics = diagnostics
        if num_workers is None:
            # Leave a core free to avoid starving the accumulator
            num_workers = max(1, multiprocessing.cpu_count() - 1)
        self.num_workers = num_workers
        self._reset_solution_stores()

    def _reset_solution_stores(self):
        self.solution_stores = {
            'K': solutions.CalSolutionStoreLatest('K'),
            'KCROSS': solutions.CalSolutionStoreLatest('KCROSS'),
            'KCROSS_DIODE': solutions.CalSolutionStoreLatest('KCROSS_DIODE'),
            'B': solutions.CalSolutionStoreLatest('B'),
            'G': solutions.CalSolutionStore('G')
        }

    def get_sensors(self):
        return [
            katcp.Sensor.float(
                'pipeline-last-time',
                'time taken to process the most recent buffer (prometheus: gauge)',
                unit='s'),
            katcp.Sensor.integer(
                'pipeline-last-slots',
                'number of slots filled in the most recent buffer (prometheus: gauge)'),
            katcp.Sensor.boolean(
                'pipeline-active',
                'whether pipeline is currently computing (prometheus: gauge)',
                default=False, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.integer(
                'pipeline-exceptions',
                'number of times the pipeline threw an exception (prometheus: counter)',
                default=0, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.float(
                'pipeline-start-flag-fraction-auto-pol',
                'Starting flag fraction prior to RFI detection: auto-pol (prometheus: Gauge)'),
            katcp.Sensor.float(
                'pipeline-start-flag-fraction-cross-pol',
                'Starting flag fraction prior to RFI detection: cross-pol (prometheus: Gauge)'),
            katcp.Sensor.float(
                'pipeline-final-flag-fraction-auto-pol',
                'Final flag fraction post RFI detection: auto-pol (prometheus: Gauge)'),
            katcp.Sensor.float(
                'pipeline-final-flag-fraction-cross-pol',
                'Final flag fraction post RFI detection: cross-pol (prometheus: Gauge)')
        ]

    def run(self):
        """Task (Process or Thread) run method. Runs pipeline

        This is a wrapper around :meth:`_run` which just handles the
        diagnostics option.
        """
        cluster = dask.distributed.LocalCluster(
            n_workers=1, threads_per_worker=self.num_workers,
            processes=False, memory_limit=0, diagnostics_port=self.diagnostics)
        with cluster, dask.distributed.Client(cluster):
            self._run_impl()

    def _run_impl(self):
        """
        Real implementation of :meth:`_run`.

        Note: do not call this `_run`, since that is a method of the base class.
        """
        # run until stop event received
        try:
            while True:
                logger.info('waiting for next event (%s)', self.name)
                event = self.accum_pipeline_queue.get()
                if isinstance(event, BufferReadyEvent):
                    logger.info('buffer with %d slots acquired by %s',
                                len(event.slots), self.name)
                    start_time = time.time()
                    self.sensors['pipeline-active'].set_value(True, timestamp=start_time)
                    # set up dask arrays around the chosen slots
                    data = {'times': self.buffers['times'][event.slots],
                            'dump_indices': self.buffers['dump_indices'][event.slots]}
                    slices = list(_slots_slices(event.slots))
                    for key in ('vis', 'flags', 'weights'):
                        buffer = self.buffers[key]
                        parts = [da.from_array(buffer[s], chunks=buffer[s].shape, name=False)
                                 for s in slices]
                        data[key] = da.concatenate(parts, axis=0)
                    # run the pipeline
                    error = False
                    try:
                        self.run_pipeline(event.capture_block_id, data)
                    except Exception:
                        logger.exception('Exception in pipeline')
                        error = True
                    end_time = time.time()
                    elapsed = end_time - start_time
                    self.sensors['pipeline-last-time'].set_value(elapsed, timestamp=end_time)
                    self.sensors['pipeline-last-slots'].set_value(
                        len(event.slots), timestamp=end_time)
                    self.sensors['pipeline-active'].set_value(False, timestamp=end_time)
                    if error:
                        _inc_sensor(self.sensors['pipeline-exceptions'], 1,
                                    status=katcp.Sensor.ERROR,
                                    timestamp=end_time)
                    # transmit flags after pipeline is finished
                    self.pipeline_sender_queue.put(event)
                    logger.info('buffer with %d slots released by %s for transmission',
                                len(event.slots), self.name)
                elif isinstance(event, ObservationEndEvent):
                    self.master_queue.put(
                        ObservationStateEvent(event.capture_block_id, State.REPORTING))
                    self.pipeline_sender_queue.put(event)
                    self.pipeline_report_queue.put(event)
                elif isinstance(event, StopEvent):
                    logger.info('stop received by %s', self.name)
                    break
                else:
                    logger.error('unknown event type %r by %s', event, self.name)
        finally:
            self.pipeline_sender_queue.put(StopEvent())
            self.pipeline_report_queue.put(StopEvent())

    def run_pipeline(self, capture_block_id, data):
        # run pipeline calibration
        telstate_cb_cal = make_telstate_cb(self.telstate_cal, capture_block_id)
        target_slices, avg_corr = pipeline(data, telstate_cb_cal, self.parameters,
                                           self.solution_stores, self.l0_name, self.sensors)
        # put corrected data into pipeline_report_queue
        self.pipeline_report_queue.put(avg_corr)


@attr.s
class FlagsStream(object):
    """Configuration information about a single L1 flag stream.

    Parameters
    ----------
    name : str
        Name of the stream
    endpoints : list of :class:`.katsdptelstate.endpoint.Endpoint`
        Destination multicast endpoints for transmission
    src_stream : str
        Corresponding L0 stream. This is set in telstate, and need not match
        the stream being fed into the accumulator (and will not if
        continuum_factor is greater than 1).
    interface_address : str
        IP address from which to transmit
    rate_ratio : float
        Speed to send flags, relative to incoming rate
    continuum_factor : int
        Factor by which to combine flags on the spectral axis. It must divide
        the number of channels.
    """
    name = attr.ib()
    endpoints = attr.ib()
    src_stream = attr.ib()
    rate_ratio = attr.ib()
    interface_address = attr.ib(default=None)
    continuum_factor = attr.ib(default=1)


class Transmitter(object):
    """State for a single flags stream held by :class:`Sender`"""
    def __init__(self, l0_name, l0_attr, flags_stream, clock_ratio, telstate_cal, parameters):
        self.flags_stream = flags_stream
        n_endpoints = len(flags_stream.endpoints)
        self._n_servers = n_servers = parameters['servers']
        self._server_id = parameters['server_id']
        if n_endpoints != n_servers:
            raise ValueError(
                'Number of flags endpoints ({}) not equal to number of servers ({})'
                .format(n_endpoints, n_servers))
        self.endpoint = flags_stream.endpoints[parameters['server_id']]

        telstate = telstate_cal.root()
        n_chans = l0_attr['n_chans'] // n_servers
        n_bls = len(l0_attr['bls_ordering'])
        channel_slice = parameters['channel_slice']
        self.rate = n_chans * n_bls / float(l0_attr['int_time']) * flags_stream.rate_ratio
        self.rate = self.rate / clock_ratio if clock_ratio else 0.0
        if n_chans % flags_stream.continuum_factor != 0:
            raise ValueError('Continuum factor {} does not divide into server channels {}'
                             .format(flags_stream.continuum_factor, n_chans))
        out_chans = n_chans // flags_stream.continuum_factor

        # create SPEAD item group
        flavour = spead2.Flavour(4, 64, 48)
        ig = spead2.send.ItemGroup(flavour=flavour)
        # set up item group with items
        ig.add_item(id=None, name='flags', description="Flags for visibilities",
                    shape=(out_chans, n_bls), dtype=None, format=[('u', 8)])
        ig.add_item(id=None, name='timestamp', description="Seconds since sync time",
                    shape=(), dtype=None, format=[('f', 64)])
        ig.add_item(id=None, name='dump_index', description='Index in time',
                    shape=(), dtype=None, format=[('u', 64)])
        ig.add_item(id=0x4103, name='frequency',
                    description="Channel index of first channel in the heap",
                    shape=(), dtype=np.uint32,
                    value=channel_slice.start // flags_stream.continuum_factor)
        ig.add_item(id=None, name='capture_block_id', description='SDP capture block ID',
                    shape=(None,), dtype=None, format=[('c', 8)])
        self._ig = ig

        cal_name = telstate_cal.prefixes[0][:-1]
        self.telstate_flags = telstate.view(flags_stream.name)
        # The flags stream is mostly the same shape/layout as the L0 stream,
        # with the exception of channelisation.
        for key in ['bandwidth', 'int_time', 'sync_time', 'excise', 'n_bls', 'bls_ordering']:
            self.telstate_flags.add(key, l0_attr[key], immutable=True)
        old_spw = SpectralWindow(centre_freq=l0_attr['center_freq'],
                                 bandwidth=l0_attr['bandwidth'],
                                 num_chans=l0_attr['n_chans'],
                                 channel_width=None,    # Computed from bandwidth
                                 sideband=1)
        new_spw = old_spw.rechannelise(l0_attr['n_chans'] // flags_stream.continuum_factor)
        self.telstate_flags.add('center_freq', new_spw.centre_freq, immutable=True)
        self.telstate_flags.add('n_chans', new_spw.num_chans, immutable=True)
        self.telstate_flags.add('n_chans_per_substream', out_chans, immutable=True)
        self.telstate_flags.add('src_streams', [flags_stream.src_stream], immutable=True)
        self.telstate_flags.add('stream_type', 'sdp.flags', immutable=True)
        self.telstate_flags.add('calibrations_applied', [cal_name], immutable=True)

    def prepare(self):
        """Do further setup in the child process."""
        config = spead2.send.StreamConfig(max_packet_size=8872, rate=self.rate)
        self._tx = spead2.send.UdpStream(
            spead2.ThreadPool(), self.endpoint.host, self.endpoint.port,
            config, ttl=1, interface_address=self.flags_stream.interface_address or '')
        self._tx.set_cnt_sequence(self._server_id, self._n_servers)

    def start_capture_block(self, cbid, first_timestamp):
        """Send a start-of-stream heap, and initialise the capture block ID"""
        self._tx.send_heap(self._ig.get_start())
        self._ig['capture_block_id'].value = cbid
        telstate_cb_flags = make_telstate_cb(self.telstate_flags, cbid)
        telstate_cb_flags.add('first_timestamp', first_timestamp, immutable=True)

    def end_capture_block(self):
        """Send an end-of-stream heap that includes capture block ID"""
        cbid_item = self._ig['capture_block_id']
        heap = self._ig.get_end()
        heap.add_descriptor(cbid_item)
        heap.add_item(cbid_item)
        self._tx.send_heap(heap)

    def send(self, idx, timestamp, flags, excise):
        """Transmit flag data"""
        if self.flags_stream.continuum_factor != 1:
            flags = calprocs.wavg_flags_f(
                flags, self.flags_stream.continuum_factor, excise, axis=0)
        self._ig['flags'].value = flags
        self._ig['timestamp'].value = timestamp
        self._ig['dump_index'].value = idx
        self._tx.send_heap(self._ig.get_heap(data='all', descriptors='all'))


class Sender(Task):
    def __init__(self, task_class, buffers,
                 pipeline_sender_queue, master_queue,
                 l0_name, flags_streams, clock_ratio, telstate_cal, parameters):
        super(Sender, self).__init__(task_class, master_queue, 'Sender')
        self.telstate_l0 = telstate_cal.root().view(l0_name)
        l0_attr = {key: self.telstate_l0[key]
                   for key in ['n_bls', 'bls_ordering', 'int_time', 'sync_time', 'excise',
                               'bandwidth', 'center_freq', 'n_chans']}
        n_servers = parameters['servers']
        self._transmitters = [Transmitter(l0_name, l0_attr, flags_stream, clock_ratio,
                                          telstate_cal, parameters)
                              for flags_stream in flags_streams]
        self.int_time = self.telstate_l0['int_time']
        self.n_chans = self.telstate_l0['n_chans'] // n_servers
        l0_bls = np.asarray(self.telstate_l0['bls_ordering'])
        self.n_bls = len(l0_bls)

        self.buffers = buffers
        self.pipeline_sender_queue = pipeline_sender_queue
        # Compute the permutation to get back to L0 ordering. get_reordering gives
        # the inverse of what is needed.
        rev_ordering = calprocs.get_reordering(parameters['antenna_names'], l0_bls)[0]
        self.ordering = np.full(self.n_bls, -1)
        for i, idx in enumerate(rev_ordering):
            self.ordering[idx] = i
        if np.any(self.ordering < 0):
            raise RuntimeError('accumulator discards some baselines')

    def get_sensors(self):
        return [
            katcp.Sensor.integer(
                'output-bytes-total',
                'bytes written to the flags L1 stream (prometheus: counter)',
                default=0, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.integer(
                'output-heaps-total',
                'heaps written to the flags L1 stream (prometheus: counter)',
                default=0, initial_status=katcp.Sensor.NOMINAL)
        ]

    def run(self):
        nt = len(self._transmitters)
        started = False
        out_flags = np.zeros((self.n_chans, self.n_bls), np.uint8)
        for transmitter in self._transmitters:
            transmitter.prepare()
        while True:
            event = self.pipeline_sender_queue.get()
            if isinstance(event, StopEvent):
                break
            elif isinstance(event, BufferReadyEvent):
                if not self._transmitters:
                    self.master_queue.put(event)
                else:
                    logger.info('starting transmission of %d slots', len(event.slots))
                    if not started:
                        cbid = event.capture_block_id
                        telstate_cb_l0 = make_telstate_cb(self.telstate_l0, cbid)
                        first_timestamp = telstate_cb_l0['first_timestamp']
                        for transmitter in self._transmitters:
                            transmitter.start_capture_block(cbid, first_timestamp)
                        started = True
                    for slot in event.slots:
                        flags = self.buffers['flags'][slot]
                        # Flatten the pol and baseline dimensions
                        flags.shape = (self.n_chans, self.n_bls)
                        # Permute into the same order as the L0 stream
                        np.take(flags, self.ordering, axis=1, out=out_flags)
                        excise = np.unpackbits(self.buffers['excise'][slot], axis=1)
                        # unpack_bits will always produce a multiple of 8 bits,
                        # even if the original had fewer. So we need to trim.
                        excise = excise[:, :self.n_bls]
                        idx = self.buffers['dump_indices'][slot]
                        timestamp = first_timestamp + idx * self.int_time
                        for transmitter in self._transmitters:
                            # TODO: send these in parallel
                            transmitter.send(idx, timestamp, out_flags, excise)
                        now = time.time()
                        _inc_sensor(self.sensors['output-heaps-total'], nt, timestamp=now)
                        _inc_sensor(self.sensors['output-bytes-total'], out_flags.nbytes * nt,
                                    timestamp=now)
                        self.master_queue.put(BufferReadyEvent(event.capture_block_id, [slot]))
                    logger.info('finished transmission of %d slots', len(event.slots))
            elif isinstance(event, ObservationEndEvent):
                if started:
                    for transmitter in self._transmitters:
                        transmitter.end_capture_block()
                    started = False
                self.master_queue.put(event)
        if started:
            for transmitter in self._transmitters:
                transmitter.end_capture_block()


class ReportWriter(Task):
    def __init__(self, task_class, pipeline_report_queue, master_queue,
                 l0_name, telstate_cal, parameters,
                 report_path, log_path, full_log, max_scans):
        super(ReportWriter, self).__init__(task_class, master_queue, 'ReportWriter')
        if not report_path:
            report_path = '.'
        report_path = os.path.abspath(report_path)
        self.pipeline_report_queue = pipeline_report_queue
        self.telstate = telstate_cal.root()
        self.telstate_cal = telstate_cal
        self.l0_name = l0_name
        self.parameters = parameters
        self.report_path = report_path
        self.log_path = log_path
        self.full_log = full_log
        self.max_scans = max_scans
        # get subarray ID
        self.subarray_id = self.telstate.get('subarray_product_id', 'unknown_subarray')

    def get_sensors(self):
        return [
            katcp.Sensor.integer(
                'reports-written',
                'Number of calibration reports written (prometheus: counter)',
                default=0, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.float(
                'report-last-time',
                'Elapsed time to generate most recent report (prometheus: gauge)',
                unit='s'),
            katcp.Sensor.boolean(
                'report-active',
                'Whether the report writer is active (prometheus: gauge)',
                default=False, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.string(
                'report-last-path',
                'Directory containing the most recent report'),
            katcp.Sensor.integer(
                'report-scans-received',
                'Number of scan summaries received by report writer (prometheus: counter)',
                default=0, initial_status=katcp.Sensor.NOMINAL),
            katcp.Sensor.integer(
                'report-scans-buffered',
                'Number of scan summaries held in report writer buffer (prometheus: gauge)',
                default=0, initial_status=katcp.Sensor.NOMINAL)
        ]

    def write_report(self, telstate_cal, capture_block_id, obs_start, obs_end, av_corr):
        # make directory for this capture block, for logs and report
        telstate_cb_cal = make_telstate_cb(self.telstate_cal, capture_block_id)
        base_name = '{}_{}_calreport_{}.{}'.format(
            capture_block_id, self.l0_name, self.telstate_cal.prefixes[0][:-1],
            self.parameters['server_id'] + 1)
        report_dir = os.path.join(self.report_path, base_name)
        current_report_dir = report_dir + '-current'
        try:
            os.mkdir(current_report_dir)
        except OSError:
            logger.warning('Report directory %s already exists', current_report_dir)

        # create pipeline report
        try:
            make_cal_report(telstate_cb_cal, capture_block_id, self.l0_name, self.parameters,
                            current_report_dir, av_corr,
                            st=obs_start, et=obs_end)
        except Exception as error:
            logger.warn('Report generation failed: %s', error, exc_info=True)

        logger.info('   Observation ended')
        logger.info('===========================')

        if self.full_log is not None:
            shutil.copy('{0}/{1}'.format(self.log_path, self.full_log),
                        '{0}/{1}'.format(current_report_dir, self.full_log))

        # change report and log directory to final name for archiving
        os.rename(current_report_dir, report_dir)
        logger.info('Moved observation report to %s', report_dir)
        return report_dir

    def run(self):
        reports_sensor = self.sensors['reports-written']
        report_time_sensor = self.sensors['report-last-time']
        report_active_sensor = self.sensors['report-active']
        report_path_sensor = self.sensors['report-last-path']
        report_scans_received_sensor = self.sensors['report-scans-received']
        report_scans_buffered_sensor = self.sensors['report-scans-buffered']
        # Set initial value of averaged corrected data
        av_corr = {}
        report_scans_buffered_sensor.set_value(0)

        while True:
            event = self.pipeline_report_queue.get()
            if isinstance(event, StopEvent):
                break
            if isinstance(event, dict):
                # if corrected data is not empty, aggregate with previous corrected data output
                logger.info('Corrected Data is in the queue')
                if event['targets']:
                    now = time.time()
                    _inc_sensor(report_scans_received_sensor, len(event['targets']), timestamp=now)
                    av_corr = _sum_corr(av_corr, event, self.max_scans)
                    scans_buffered = len(av_corr['targets'])
                    report_scans_buffered_sensor.set_value(
                        scans_buffered,
                        status=(katcp.Sensor.NOMINAL if scans_buffered < self.max_scans
                                else katcp.Sensor.WARN),
                        timestamp=now)

            elif isinstance(event, ObservationEndEvent):
                try:
                    logger.info('Starting report on %s', event.capture_block_id)
                    start_time = time.time()
                    report_active_sensor.set_value(True, timestamp=start_time)
                    obs_dir = self.write_report(
                        self.telstate_cal, event.capture_block_id,
                        event.start_time, event.end_time, av_corr)
                    end_time = time.time()
                    av_corr = {}
                    report_scans_buffered_sensor.set_value(0, timestamp=end_time)
                    _inc_sensor(reports_sensor, 1, timestamp=end_time)
                    report_time_sensor.set_value(end_time - start_time, timestamp=end_time)
                    report_path_sensor.set_value(obs_dir, timestamp=end_time)
                    report_active_sensor.set_value(False, timestamp=end_time)
                finally:
                    self.master_queue.put(event)
            else:
                logger.error('unknown event type %r', event)
        logger.info('Report writer has finished, exiting')


class CalDeviceServer(katcp.server.AsyncDeviceServer):
    VERSION_INFO = ('katsdpcal-api', 1, 0)
    BUILD_INFO = ('katsdpcal',) + tuple(katsdpcal.__version__.split('.', 1)) + ('',)

    def __init__(self, accumulator, pipeline, sender, report_writer, master_queue,
                 *args, **kwargs):
        self.accumulator = accumulator
        self.pipeline = pipeline
        self.sender = sender
        self.report_writer = report_writer
        self.children = [pipeline, sender, report_writer]
        self.master_queue = master_queue
        self._stopping = False
        self._capture_block_state = {}
        # Each capture block needs to be marked done twice: once from
        # Sender, once from ReportWriter.
        self._capture_block_ends = Counter()
        self._capture_block_state_sensor = katcp.Sensor.string(
            'capture-block-state',
            'JSON dict with the state of each capture block')
        self._capture_block_state_sensor.set_value('{}')
        super(CalDeviceServer, self).__init__(*args, **kwargs)

    def setup_sensors(self):
        for sensor in self.accumulator.sensors.values():
            self.add_sensor(sensor)
        for child in self.children:
            for sensor in child.get_sensors():
                self.add_sensor(sensor)
        self.add_sensor(self._capture_block_state_sensor)

    def _set_capture_block_state(self, capture_block_id, state):
        if state == State.DEAD:
            # Remove if present
            self._capture_block_state.pop(capture_block_id, None)
        else:
            self._capture_block_state[capture_block_id] = state
        dumped = json.dumps(self._capture_block_state, sort_keys=True, cls=EnumEncoder)
        self._capture_block_state_sensor.set_value(dumped)

    def start(self):
        self._run_queue_task = trollius.ensure_future(self._run_queue())
        super(CalDeviceServer, self).start()

    @trollius.coroutine
    def join(self):
        yield From(self._run_queue_task)

    @request(Str())
    @return_reply()
    def request_capture_init(self, msg, capture_block_id):
        """Start an observation"""
        if self.accumulator.capturing:
            return ('fail', 'capture already in progress')
        if self._stopping:
            return ('fail', 'server is shutting down')
        if capture_block_id in self._capture_block_state:
            return ('fail', 'capture block ID {} is already active'.format(capture_block_id))
        self._set_capture_block_state(capture_block_id, State.CAPTURING)
        self.accumulator.capture_init(capture_block_id)
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
                if self._capture_block_state:
                    logger.warn('Forced shutdown with active capture blocks - data may be lost')
                else:
                    logger.info('Forced shutdown, no active capture blocks')
            else:
                logger.info('Shutting down gracefully')
            if not force:
                yield From(self.accumulator.stop())
                progress('Accumulator stopped')
                for task in self.children:
                    yield From(loop.run_in_executor(executor, task.join))
                    progress('{} stopped'.format(task.name))
            elif hasattr(self.report_writer, 'terminate'):
                # Kill off all the tasks. This is done in reverse order, to avoid
                # triggering a report writing only to kill it half-way.
                # TODO: this may need to become semi-graceful at some point, to avoid
                # corrupting an in-progress report.
                for task in reversed(self.children):
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
        if self._stopping and not force:
            raise tornado.gen.Return(('fail', 'server is already shutting down'))
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
                elif isinstance(event, BufferReadyEvent):
                    yield From(self.accumulator.buffer_free(event))
                elif isinstance(event, SensorReadingEvent):
                    try:
                        sensor = self.get_sensor(event.name)
                    except ValueError:
                        logger.warn('Received update for unknown sensor %s', event.name)
                    else:
                        sensor.set(event.reading.timestamp,
                                   event.reading.status,
                                   event.reading.value)
                elif isinstance(event, ObservationStateEvent):
                    self._set_capture_block_state(event.capture_block_id, event.state)
                elif isinstance(event, ObservationEndEvent):
                    self._capture_block_ends[event.capture_block_id] += 1
                    if self._capture_block_ends[event.capture_block_id] == 2:
                        # Both sender and pipeline have finished
                        del self._capture_block_ends[event.capture_block_id]
                        self._set_capture_block_state(event.capture_block_id, State.DEAD)
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
    # The excise buffer is in the shape of the L0 stream, and packs
    # each flag into one bit.
    excise_shape = (buffer_shape[0], buffer_shape[1],
                    (buffer_shape[2] * buffer_shape[3] + 7) // 8)
    data = {}
    data['vis'] = factory(buffer_shape, dtype=np.complex64)
    data['flags'] = factory(buffer_shape, dtype=np.uint8)
    data['excise'] = factory(excise_shape, dtype=np.uint8)
    data['weights'] = factory(buffer_shape, dtype=np.float32)
    data['times'] = factory(buffer_shape[0], dtype=np.float)
    data['dump_indices'] = factory(buffer_shape[0], dtype=np.uint64)
    return data


def create_server(use_multiprocessing, host, port, buffers,
                  l0_name, l0_endpoints, l0_interface_address,
                  flags_streams, clock_ratio, telstate_cal, parameters,
                  report_path, log_path, full_log,
                  diagnostics=None, pipeline_profile_file=None, num_workers=None,
                  max_scans=None):
    # threading or multiprocessing imports
    if use_multiprocessing:
        logger.info("Using multiprocessing")
        module = multiprocessing
    else:
        logger.info("Using threading")
        module = multiprocessing.dummy

    # set up inter-task synchronisation primitives.
    accum_pipeline_queue = module.Queue()
    pipeline_sender_queue = module.Queue()
    pipeline_report_queue = module.Queue()
    master_queue = module.Queue()

    # Set up the pipeline
    pipeline = Pipeline(
        module.Process, buffers,
        accum_pipeline_queue, pipeline_sender_queue, pipeline_report_queue, master_queue,
        l0_name, telstate_cal, parameters, diagnostics, pipeline_profile_file, num_workers)
    # Set up the sender
    sender = Sender(
        module.Process, buffers, pipeline_sender_queue, master_queue, l0_name,
        flags_streams, clock_ratio, telstate_cal, parameters)
    # Set up the report writer
    report_writer = ReportWriter(
        module.Process, pipeline_report_queue, master_queue, l0_name, telstate_cal, parameters,
        report_path, log_path, full_log, max_scans)

    # Start the child tasks.
    running_tasks = []
    try:
        for task in [report_writer, sender, pipeline]:
            if not use_multiprocessing:
                task.daemon = True    # Make sure it doesn't prevent process exit
            task.start()
            running_tasks.append(task)

        # Set up the accumulator. This is done after the other processes are
        # started, because it creates a ThreadPoolExecutor, and threads and fork()
        # don't play nicely together.
        accumulator = Accumulator(buffers, accum_pipeline_queue, master_queue,
                                  l0_name, l0_endpoints, l0_interface_address,
                                  telstate_cal, parameters)
        return CalDeviceServer(accumulator, pipeline, sender, report_writer,
                               master_queue, host, port)
    except Exception:
        for task in running_tasks:
            if hasattr(task, 'terminate'):
                task.terminate()
                task.join()
        raise
