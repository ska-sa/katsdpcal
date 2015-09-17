import spead2
import spead2.send
import spead2.recv

from katsdpcal.reduction import pipeline
from katsdpcal import calprocs
from katsdptelstate.telescope_state import TelescopeState

import socket
import numpy as np
import time

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TaskLoggingAdapter(logging.LoggerAdapter):
    """
    This example adapter expects the passed in dict-like object to have a
    'connid' key, whose value in brackets is prepended to the log message.
    """
    def process(self, msg, kwargs):
        return '[%s] %s' % (self.extra['connid'], msg), kwargs

# ---------------------------------------------------------------------------------------
# Accumulator
# ---------------------------------------------------------------------------------------

def init_accumulator_control(control_method, control_task, buffers, buffer_shape, scan_accumulator_conditions, l0_endpoint, telstate):

    class accumulator_control(control_task):
        """
        Task (Process or Thread) which accumutates data from SPEAD into numpy arrays
        """

        def __init__(self, control_method, buffers, buffer_shape, scan_accumulator_conditions, l0_endpoint, telstate):
            control_task.__init__(self)

            self.buffers = buffers
            self.telstate = telstate
            self.l0_endpoint = l0_endpoint
            self.scan_accumulator_conditions = scan_accumulator_conditions
            self.num_buffers = len(buffers)

            self.name = 'Accumulator_task'
            self._stop = control_method.Event()
            self._obsend = control_method.Event()

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

            # set up multicask sockets
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if self.l0_endpoint.multicast_subscribe(sock):
                self.accumulator_logger.info("Subscribing to multicast address {0}".format(self.l0_endpoint.host))

        def run(self):
            """
             Task (Process or Thread) run method. Append random vis to the vis list
            at random time.
            """
            # if we are usig multiprocessing, view ctypes array in numpy format
            if 'multiprocessing' in str(control_method): self.buffers_to_numpy()

            # Initialise SPEAD receiver
            self.accumulator_logger.info('Initializing SPEAD receiver')
            rx = spead2.recv.Stream(spead2.ThreadPool(), bug_compat=spead2.BUG_COMPAT_PYSPEAD_0_5_2)
            rx.add_udp_reader(self.l0_endpoint.port, max_size=9172, buffer_size=0)

            # Increment between buffers, filling and releasing iteratively
            # Initialise current buffer counter
            current_buffer=-1
            while not self._stop.is_set():
                #Increment the current buffer
                current_buffer = (current_buffer+1)%self.num_buffers
                # ------------------------------------------------------------
                # Loop through the buffers and send data to pipeline task when accumulation terminate conditions are met.

                self.scan_accumulator_conditions[current_buffer].acquire()
                self.accumulator_logger.info('scan_accumulator_condition %d acquired by %s' %(current_buffer, self.name,))

                # accumulate data scan by scan into buffer arrays
                self.accumulator_logger.info('max buffer length %d' %(self.max_length,))
                self.accumulator_logger.info('accumulating data into buffer %d...' %(current_buffer,))
                max_ind = self.accumulate(rx, current_buffer)
                self.accumulator_logger.info('Accumulated {0} timestamps'.format(max_ind+1,))

                # awaken pipeline task that was waiting for condition lock
                self.scan_accumulator_conditions[current_buffer].notify()
                self.accumulator_logger.info('scan_accumulator_condition %d notification sent by %s' %(current_buffer, self.name,))
                # release pipeline task that was waiting for condition lock
                self.scan_accumulator_conditions[current_buffer].release()
                self.accumulator_logger.info('scan_accumulator_condition %d released by %s' %(current_buffer, self.name,))

                time.sleep(0.2)

        def stop_release(self):
            # stop accumulator
            self.stop()
            # close off scan_accumulator_conditions
            #  - necessary for closing pipeline task which may be waiting on condition
            for scan_accumulator in self.scan_accumulator_conditions:
                scan_accumulator.acquire()
                scan_accumulator.notify()
                scan_accumulator.release()

        def stop(self):
            # set stop event
            self._stop.set()
            # stop SPEAD stream receival manually if the observation is still running
            if not self.obs_finished(): self.capture_stop()

        def obs_finished(self):
            return self._obsend.is_set()

        def stopped(self):
            return self._stop.is_set()

        def capture_stop(self):
            """
            Send stop packed to force shut down of SPEAD receiver
            """
            self.accumulator_logger.info('stop SPEAD receiver')
            # send the stop only to the local receiver
            end_transmit('127.0.0.1',self.l0_endpoint.port)

        def buffers_to_numpy(self):
            """
            View ctype data buffers as numpy arrays
            """

            for current_buffer in self.buffers:
                current_buffer['vis'] = np.frombuffer(current_buffer['vis'], dtype=np.float32).view(np.complex64)
                current_buffer['vis'].shape = self.buffer_shape
                current_buffer['flags'] = np.frombuffer(current_buffer['flags'], dtype=np.uint8)
                current_buffer['flags'].shape = self.buffer_shape
                current_buffer['weights'] = np.frombuffer(current_buffer['weights'], dtype=np.float32)
                current_buffer['weights'].shape = self.buffer_shape
                current_buffer['times'] = np.frombuffer(current_buffer['times'], dtype=np.float64)
                current_buffer['max_index'] = np.frombuffer(current_buffer['max_index'], dtype=np.int32)

        def set_ordering_parameters(self):
            # determine re-ordering necessary to convert from supplied bls ordering to desired bls ordering
            self.ordering, bls_order, pol_order = calprocs.get_reordering(self.telstate.antenna_mask,self.telstate.cbf_bls_ordering)
            # determine lookup list for baselines
            bls_lookup = calprocs.get_bls_lookup(self.telstate.antenna_mask, bls_order)
            # save these to the TS for use in the pipeline/elsewhere
            self.telstate.add('cal_bls_ordering',bls_order)
            self.telstate.add('cal_pol_ordering',pol_order)
            self.telstate.add('cal_bls_lookup',bls_lookup)

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
            prev_tags = 'none'

            # set data buffer for writing
            data_buffer = self.buffers[buffer_index]

            # get names of activity and target TS keys, using TS reference antenna
            target_key = '{0}_target'.format(self.telstate.cal_refant,)
            activity_key = '{0}_activity'.format(self.telstate.cal_refant,)
            # sync time provided by the cbf
            cbf_sync_time = self.telstate.cbf_sync_time if self.telstate.has_key('cbf_sync_time') else 0.0

            obs_end_flag = True

            # receive SPEAD stream
            ig = spead2.ItemGroup()
            for heap in rx:
                ig.update(heap)
                array_index += 1

                # if this is the first scan of the observation, set up some values
                if start_flag:
                    start_time = ig['timestamp'].value + cbf_sync_time
                    start_flag = False

                    # when data starts to flow, set the baseline ordering parameters for re-ordering the data
                    self.set_ordering_parameters()

                # get activity and target tag from TS
                data_ts = ig['timestamp'].value + cbf_sync_time
                activity = self.telstate.get_range(activity_key,et=data_ts,include_previous=True)[0][0]
                target = self.telstate.get_range(target_key,et=data_ts,include_previous=True)[0][0]

                # reshape data and put into relevent arrays
                data_buffer['vis'][array_index,:,:,:] = ig['correlator_data'].value[:,self.ordering].reshape([self.nchan,self.npol,self.nbl])
                data_buffer['flags'][array_index,:,:,:] = ig['flags'].value[:,self.ordering].reshape([self.nchan,self.npol,self.nbl])
                # if we are not receiving weights in the SPEAD stream, fake them
                if 'weights' in ig.keys():
                    data_buffer['weights'][array_index,:,:,:] = ig['weights'].value[:,self.ordering].reshape([self.nchan,self.npol,self.nbl])
                else:
                    data_buffer['weights'][array_index,:,:,:] = np.empty([self.nchan,self.npol,self.nbl],dtype=np.float32)
                data_buffer['times'][array_index] = data_ts

                # break if this scan is a slew that follows a track
                #   unless previous scan was a target, in which case accumulate subsequent gain scan too
                # ********** THIS BREAKING NEEDS TO BE THOUGHT THROUGH CAREFULLY **********
                if ('slew' in activity and 'track' in prev_activity) and 'target' not in prev_tags:
                    self.accumulator_logger.info('Accumulate break due to transition')
                    obs_end_flag = False
                    break

                # this is a temporary mock up of a natural break in the data stream
                # will ultimately be provided by some sort of sensor
                duration = (ig['timestamp'].value+ cbf_sync_time)-start_time
                if duration>2000000:
                    self.accumulator_logger.info('Accumulate break due to duration')
                    obs_end_flag = False
                    break
                # end accumulation if maximum array size has been accumulated
                if array_index >= self.max_length - 1:
                    self.accumulator_logger.info('Accumulate break due to buffer size limit')
                    obs_end_flag = False
                    break

                prev_activity = activity
                # extract tags from target description string
                prev_tags = target.split(',')[1]

            # if we exited the loop because it was the end of the SPEAD transmission
            if obs_end_flag:
                self.accumulator_logger.info('Observation ended')
                self._obsend.set()

            if 'multiprocessing' in str(control_method):
                # multiprocessing case
                data_buffer['max_index'][0] = array_index
            else:
                # threading case
                data_buffer['max_index'] = np.atleast_1d(np.array(array_index))

            self.accumulator_logger.info('Accumulation ended')

            return array_index

    return accumulator_control(control_method, buffers, buffer_shape, scan_accumulator_conditions, l0_endpoint, telstate)

# ---------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------

def init_pipeline_control(control_method, control_task, data, data_shape, scan_accumulator_condition, pipenum, l1_endpoint, \
        l1_rate,  telstate):

    class pipeline_control(control_task):
        """
        Task (Process or Thread) which runs pipeline
        """

        def __init__(self, control_method, data, data_shape, scan_accumulator_condition, pipenum, l1_endpoint, l1_rate, telstate):
            control_task.__init__(self)
            self.data = data
            self.scan_accumulator_condition = scan_accumulator_condition
            self.name = 'Pipeline_task_'+str(pipenum)
            self._stop = control_method.Event()
            self.telstate = telstate
            self.data_shape = data_shape
            self.full_l1 = telstate.cal_full_l1
            self.l1_rate = l1_rate
            self.l1_endpoint = l1_endpoint

            # set up logging adapter for the task
            self.pipeline_logger = TaskLoggingAdapter(logger, {'connid': self.name})

        def run(self):
            """
            Task (Process or Thread) run method. Runs pipeline
            """

            # run until stop is set
            while not self._stop.is_set():
                # acquire condition on data
                self.pipeline_logger.info('scan_accumulator_condition acquire by %s' %(self.name,))
                self.scan_accumulator_condition.acquire()

                # release lock and wait for notify from accumulator
                self.pipeline_logger.info('scan_accumulator_condition release and wait by %s' %(self.name,))
                self.scan_accumulator_condition.wait()

                if not self._stop.is_set():
                    # after notify from accumulator, condition lock re-aquired
                    self.pipeline_logger.info('scan_accumulator_condition acquire by %s' %(self.name,))
                    # run the pipeline
                    self.pipeline_logger.info('Pipeline run start on accumulated data')

                    # if we are usig multiprocessing, view ctypes array in numpy format
                    if 'multiprocessing' in str(control_method): self.data_to_numpy()

                    # run the pipeline
                    self.run_pipeline()

                # release condition after pipeline run finished
                self.scan_accumulator_condition.release()
                self.pipeline_logger.info('scan_accumulator_condition release by %s' %(self.name,))

        def stop(self):
            self._stop.set()

        def stopped(self):
            return self._stop.is_set()

        def data_to_numpy(self):
            """
            Convert data buffer from ctypes to numpy arrays
            """
            # note - this sometimes causes a harmless (but irritating) PEP 3118 runtime warning
            self.data['vis'] = np.ctypeslib.as_array(self.data['vis']).view(np.complex64)
            self.data['vis'].shape = self.data_shape

            self.data['flags'] = np.ctypeslib.as_array(self.data['flags'])
            self.data['flags'].shape = self.data_shape

            self.data['weights'] = np.ctypeslib.as_array(self.data['weights'])
            self.data['weights'].shape = self.data_shape

            self.data['times'] = np.ctypeslib.as_array(self.data['times'])
            self.data['max_index'] = np.ctypeslib.as_array(self.data['max_index'])

        def run_pipeline(self):
            # run pipeline calibration
            target_slices = pipeline(self.data,self.telstate,task_name=self.name)

            # send data to L1 SPEAD if necessary
            config = spead2.send.StreamConfig(max_packet_size=9172, rate=self.l1_rate)
            tx = spead2.send.UdpStream(spead2.ThreadPool(),self.l1_endpoint.host,self.l1_endpoint.port,config, buffer_size=0)
            if self.full_l1 or target_scans != []:
                self.pipeline_logger.info('Transmit L1 data')
                self.data_to_SPEAD(target_slices, tx)
                self.pipeline_logger.info('End transmit of L1 data')

        def data_to_SPEAD(self, target_slices, tx):
            """
            Sends data to SPEAD stream

            Inputs:
            target_slices : list of slices
                slices for target scans in the data buffer
            """
            if self.full_l1:
                # for streaming all of the data (not target only), 
                # use the highest index in the buffer that is filled with data
                target_slices = [slice(0,self.data['max_index']+1)]

            # create up SPEAD item group
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
                for i in range(len(scan_times)): # time axis
                    # transmit timestamps, vis, flags and weights
                    ig['correlator_data'].value = scan_vis[i]
                    ig['flags'].value = scan_flags[i]
                    ig['weights'].value = scan_weights[i]
                    ig['timestamp'].value = scan_times[i]
                    tx.send_heap(ig.get_heap())

    return pipeline_control(control_method, data, data_shape, scan_accumulator_condition, pipenum, l1_endpoint, l1_rate, telstate)

# ---------------------------------------------------------------------------------------
# SPEAD helper functions
# ---------------------------------------------------------------------------------------

def end_transmit(host,port):
    """
    Send stop packet to spead stream tx

    Parameters
    ----------
    spead_endpoint : endpoint to transmit to
    """
    config = spead2.send.StreamConfig(max_packet_size=9172)
    tx = spead2.send.UdpStream(spead2.ThreadPool(),host,port,config)

    flavour = spead2.Flavour(4, 64, 48, spead2.BUG_COMPAT_PYSPEAD_0_5_2)
    heap = spead2.send.Heap(flavour)
    heap.add_end()

    tx.send_heap(heap)



