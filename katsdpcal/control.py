from . import spead2
from . import send
from . import recv

from .reduction import pipeline
from . import calprocs

import numpy as np
import time
import mmap
import threading

import logging
logger = logging.getLogger(__name__)


class TaskLoggingAdapter(logging.LoggerAdapter):
    """
    This example adapter expects the passed in dict-like object to have a
    'connid' key, whose value in brackets is prepended to the log message.
    """
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['connid'], msg), kwargs


class StopEvent(object):
    """Gracefully shutdown requested"""


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

def init_accumulator_control(control_method, control_task, buffers, buffer_shape,
                             accum_pipeline_queues, pipeline_accum_sems,
                             l0_endpoint, l0_interface_address, telstate):

    class accumulator_control(control_task):
        """
        Task (Process or Thread) which accumutates data from SPEAD into numpy arrays
        """

        def __init__(self, control_method, buffers, buffer_shape,
                     accum_pipeline_queues, pipeline_accum_sems,
                     l0_endpoint, l0_interface_address, telstate):
            control_task.__init__(self)

            self.buffers = buffers
            self.telstate = telstate
            self.l0_endpoint = l0_endpoint
            self.l0_interface_address = l0_interface_address
            self.accum_pipeline_queues = accum_pipeline_queues
            self.pipeline_accum_sems = pipeline_accum_sems
            self.num_buffers = len(buffers)

            self.name = 'Accumulator'
            self._stop = control_method.Event()
            self._obsend = False

            # flag for switching capture to the alternate buffer
            self._switch_buffer = False

            # Get data shape
            self.buffer_shape = buffer_shape
            self.max_length = buffer_shape[0]
            self.nchan = buffer_shape[1]
            self.npol = buffer_shape[2]
            self.nbl = buffer_shape[3]

            # baseline ordering, to use in re-ordering the data
            #   will be set when data starts to flow
            self.ordering = None

            # set up logging adapter for the task
            self.accumulator_logger = TaskLoggingAdapter(logger, {'connid': self.name})

        def run(self):
            """
             Task (Process or Thread) run method. Append random vis to the vis list
            at random time.
            """
            # Initialise SPEAD receiver
            self.accumulator_logger.info('Initializing SPEAD receiver')
            rx = recv.Stream(spead2.ThreadPool(), bug_compat=spead2.BUG_COMPAT_PYSPEAD_0_5_2,
                             max_heaps=2, ring_heaps=1)
            self.accumulator_logger.info('Starting stopper thread')
            stop_thread = threading.Thread(target=self._stop_rx, name='stopper', args=(rx,))
            stop_thread.start()
            try:
                # Main data is 10 bytes per entry: 8 for vis, 1 for flags, 1 for weights.
                # Then there are per-channel weights (4 bytes each).
                heap_size = self.nchan * self.npol * self.nbl * 10 + self.nchan * 4
                rx.set_memory_allocator(spead2.MemoryPool(heap_size, heap_size + 4096, 4, 4))
                rx.set_memcpy(spead2.MEMCPY_NONTEMPORAL)
                if self.l0_interface_address is not None:
                    rx.add_udp_reader(self.l0_endpoint.host, self.l0_endpoint.port,
                                      interface_address=self.l0_interface_address)
                else:
                    rx.add_udp_reader(self.l0_endpoint.port, bind_hostname=self.l0_endpoint.host)

                # Increment between buffers, filling and releasing iteratively
                # Initialise current buffer counter
                current_buffer = -1
                while not self._stop.is_set() and not self._obsend:
                    # Increment the current buffer
                    current_buffer = (current_buffer + 1) % self.num_buffers
                    # ------------------------------------------------------------
                    # Loop through the buffers and send data to pipeline task when
                    # accumulation terminate conditions are met.

                    self.pipeline_accum_sems[current_buffer].acquire()
                    self.accumulator_logger.info('pipeline_accum_sems[%d] acquired by %s',
                                                 current_buffer, self.name)

                    # accumulate data scan by scan into buffer arrays
                    self.accumulator_logger.info('max buffer length %d', self.max_length)
                    self.accumulator_logger.info('accumulating into buffer %d', current_buffer)
                    max_ind = self.accumulate(rx, current_buffer)
                    self.accumulator_logger.info('Accumulated {0} timestamps'.format(max_ind+1,))

                    # awaken pipeline task that is waiting for the buffer
                    self.accum_pipeline_queues[current_buffer].put(BufferReadyEvent())
                    self.accumulator_logger.info(
                        'accum_pipeline_queues[%d] updated by %s', current_buffer, self.name)
            finally:
                self._stop.set()   # Ensure that the stopper thread can shut down
                stop_thread.join()
                rx.stop()
                for q in self.accum_pipeline_queues:
                    q.put(StopEvent())

        def _stop_rx(self, rx):
            """Function run on a separate thread which stops the receiver when
            the master asks us to stop.
            """
            self._stop.wait()
            rx.stop()

        def stop(self):
            """Called by the master to do a graceful stop on the accumulator."""
            self._stop.set()

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

        def accumulate(self, rx, buffer_index):
            """
            Accumulates spead data into arrays
               till **TBD** metadata indicates scan has stopped, or
               till array reaches max buffer size

            SPEAD item groups contain:
               correlator_data
               flags
               weights
               timestamp
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

            obs_end_flag = True

            # sensor parameters which will be set from telescope state when data starts to flow
            cbf_sync_time = None
            data_ts = None

            # receive SPEAD stream
            ig = spead2.ItemGroup()

            self.accumulator_logger.info('waiting to start accumulating data')
            for heap in rx:
                ig.update(heap)
                if len(ig.keys()) < 1:
                    self.accumulator_logger.info('==== empty stop packet received ====')
                    continue

                # get sync time from TS, if it is present (if it isn't present,
                # don't process this dump further)
                if cbf_sync_time is None:
                    if 'cbf_sync_time' in self.telstate:
                        cbf_sync_time = self.telstate.cbf_sync_time
                        self.accumulator_logger.info(' - set cbf_sync_time')
                    else:
                        self.accumulator_logger.warning(
                            'cbf_sync_time absent from telescope state - ignoring dump')
                        continue

                # get activity and target tag from TS
                data_ts = ig['timestamp'].value + cbf_sync_time
                activity_full = []
                if activity_key in self.telstate:
                    activity_full = self.telstate.get_range(
                        activity_key, et=data_ts, include_previous=True)
                if not activity_full:
                    self.accumulator_logger.info(
                        'no activity recorded for reference antenna {0} - ignoring dump'.format(
                            self.telstate.cal_refant,))
                    continue
                activity, activity_time = activity_full[0]

                # if this is the first scan of the observation, set up some values
                if start_flag:
                    unsync_start_time = ig['timestamp'].value
                    prev_activity_time = activity_time

                    # when data starts to flow, set the baseline ordering
                    # parameters for re-ordering the data
                    self.set_ordering_parameters()
                    self.accumulator_logger.info(' - set pipeline data ordering parameters')

                    self.accumulator_logger.info('accumulating data from targets:')

                # get target time from TS, if it is present (if it isn't present, set to unknown)
                if target_key in self.telstate:
                    target = self.telstate.get_range(target_key, et=data_ts,
                                                     include_previous=True)[0][0]
                    if target == '':
                        target = 'unknown'
                else:
                    self.accumulator_logger.warning(
                        'target description {0} absent from telescope state'.format(target_key))
                    target = 'unknown'
                # extract name and tags from target description string
                target_split = target.split(',')
                target_name = target_split[0]
                target_tags = target_split[1] if len(target_split) > 1 else 'unknown'
                if (target_name != prev_target_name) or start_flag:
                    # update source list if necessary
                    target_list = self.telstate.get_range(
                        'cal_info_sources', st=0, return_format='recarray')['value'] \
                        if 'cal_info_sources' in self.telstate else []
                    if target_name not in target_list:
                        self.telstate.add('cal_info_sources', target_name, ts=data_ts)

                # print name of target and activity type, if activity has
                # changed or start of accumulator
                if start_flag or (activity_time != prev_activity_time):
                    self.accumulator_logger.info(' - {0} ({1})'.format(target_name, activity))
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
                    self.accumulator_logger.info('Accumulation break - transition')
                    obs_end_flag = False
                    break
                # beamformer special case
                if (activity_time != prev_activity_time) \
                        and ('single_accumulation' in prev_target_tags):
                    self.accumulator_logger.info('Accumulation break - single scan accumulation')
                    obs_end_flag = False
                    break

                # this is a temporary mock up of a natural break in the data stream
                # will ultimately be provided by some sort of sensor
                duration = ig['timestamp'].value - unsync_start_time
                if duration > 2000000:
                    self.accumulator_logger.info('Accumulate break due to duration')
                    obs_end_flag = False
                    break
                # end accumulation if maximum array size has been accumulated
                if array_index >= self.max_length - 1:
                    self.accumulator_logger.info('Accumulate break - buffer size limit')
                    obs_end_flag = False
                    break

                prev_activity = activity
                prev_activity_time = activity_time
                prev_target_tags = target_tags
                prev_target_name = target_name

            # if we exited the loop because it was the end of the SPEAD transmission
            if obs_end_flag:
                if data_ts is not None:
                    end_time = data_ts
                else:
                    # no data_ts variable because no data has flowed
                    self.accumulator_logger.info(' --- no data flowed ---')
                    end_time = time.time()
                self.telstate.add('cal_obs_end_time', end_time, ts=end_time)
                self.accumulator_logger.info('Observation ended')
                self._obsend = True

            data_buffer['max_index'][0] = array_index
            self.accumulator_logger.info('Accumulation ended')

            return array_index

    return accumulator_control(control_method, buffers, buffer_shape,
                               accum_pipeline_queues, pipeline_accum_sems,
                               l0_endpoint, l0_interface_address, telstate)


# ---------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------

def init_pipeline_control(
        control_method, control_task, data, data_shape,
        accum_pipeline_queue, pipeline_accum_sem, pipenum, l1_endpoint,
        l1_level, l1_rate, telstate):

    class pipeline_control(control_task):
        """
        Task (Process or Thread) which runs pipeline
        """

        def __init__(self, control_method, data, data_shape,
                     accum_pipeline_queue, pipeline_accum_sem,
                     pipenum, l1_endpoint, l1_level, l1_rate, telstate):
            control_task.__init__(self)
            self.data = data
            self.accum_pipeline_queue = accum_pipeline_queue
            self.pipeline_accum_sem = pipeline_accum_sem
            self.name = 'Pipeline_' + str(pipenum)
            self.telstate = telstate
            self.data_shape = data_shape
            self.l1_level = l1_level
            self.l1_rate = l1_rate
            self.l1_endpoint = l1_endpoint

            # set up logging adapter for the task
            self.pipeline_logger = TaskLoggingAdapter(logger, {'connid': self.name})

        def run(self):
            """
            Task (Process or Thread) run method. Runs pipeline
            """

            # run until stop event received
            while True:
                self.pipeline_logger.info('waiting for next event (%s)', self.name)
                event = self.accum_pipeline_queue.get()
                if isinstance(event, BufferReadyEvent):
                    self.pipeline_logger.info('buffer acquired by %s', self.name)
                    # run the pipeline
                    self.run_pipeline()
                    # release condition after pipeline run finished
                    self.pipeline_accum_sem.release()
                    self.pipeline_logger.info('pipeline_accum_sem release by %s', self.name)
                elif isinstance(event, StopEvent):
                    self.pipeline_logger.info('stop received by %s', self.name)
                    break
                else:
                    self.pipeline_logger.error('unknown event type %r by %s', event, self.name)

        def run_pipeline(self):
            # run pipeline calibration, if more than zero timestamps accumulated
            target_slices = pipeline(self.data, self.telstate, task_name=self.name) \
                if (self.data['max_index'][0] > 0) else []

            # send data to L1 SPEAD if necessary
            if self.l1_level != 0:
                config = send.StreamConfig(max_packet_size=8972, rate=self.l1_rate)
                tx = send.UdpStream(spead2.ThreadPool(), self.l1_endpoint.host,
                                    self.l1_endpoint.port, config)
                self.pipeline_logger.info('   Transmit L1 data')
                # for streaming all of the data (not target only),
                # use the highest index in the buffer that is filled with data
                transmit_slices = [slice(0, self.data['max_index'][0] + 1)] \
                    if self.l1_level == 2 else target_slices
                self.data_to_SPEAD(transmit_slices, tx)
                self.pipeline_logger.info('   End transmit of L1 data')

        def data_to_SPEAD(self, target_slices, tx):
            """
            Sends data to SPEAD stream

            Inputs:
            target_slices : list of slices
                slices for target scans in the data buffer
            """
            # create SPEAD item group
            flavour = spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
            ig = send.ItemGroup(flavour=flavour)
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

    return pipeline_control(control_method, data, data_shape,
                            accum_pipeline_queue, pipeline_accum_sem,
                            pipenum, l1_endpoint, l1_level, l1_rate, telstate)

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
    config = send.StreamConfig(max_packet_size=8972)
    tx = send.UdpStream(spead2.ThreadPool(), host, port, config)

    flavour = spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
    heap = send.Heap(flavour)
    heap.add_end()

    tx.send_heap(heap)
