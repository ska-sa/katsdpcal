import spead64_48 as spead
import time

from katcal.reduction import pipeline
from katcal import calprocs
from katsdptelstate.telescope_state import TelescopeState

import logging
import socket
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import numpy as np

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
        Task (Process or Thread) which accumutates data from spead into numpy arrays
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
        
            # flag for switching capture to the alternate buffer
            self._switch_buffer = False

            # Get data shape
            self.buffer_shape = buffer_shape
            self.max_length = buffer_shape[0]
            self.nchan = buffer_shape[1]
            self.npol = buffer_shape[2]
            self.nbl = buffer_shape[3]

            # set up logging adapter for the task
            self.accumulator_logger = TaskLoggingAdapter(logger, {'connid': self.name})
        
        def run(self):
            """
             Task (Process or Thread) run method. Append random vis to the vis list
            at random time.
            """
            # Initialise SPEAD stream
            self.accumulator_logger.info('Initializing SPEAD receiver')

            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if self.l0_endpoint.multicast_subscribe(sock):
                logger.info("Subscribing to multicast address {0}".format(self.l0_endpoint.host))

            spead_stream = spead.TransportUDPrx(self.l0_endpoint.port)
        
            # determine re-ordering necessary to convert from supplied bls ordering to desired bls ordering
            ordering, bls_order, pol_order = calprocs.get_reordering(self.telstate.antenna_mask,self.telstate.cbf_bls_ordering)  
            # determine lookup list for baselines
            bls_lookup = calprocs.get_bls_lookup(self.telstate.antenna_mask, bls_order)  
            # save these to the TS for use in the pipeline/elsewhere  
            self.telstate.add('cal_bls_ordering',bls_order)
            self.telstate.add('cal_pol_ordering',pol_order)  
            self.telstate.add('cal_bls_lookup',bls_lookup)
          
            # if we are usig multiprocessing, view ctypes array in numpy format
            if 'multiprocessing' in str(control_method): self.buffers_to_numpy()
        
            # Increment between buffers, filling and releasing iteratively
            #Initialise current buffer counter
            current_buffer=-1
            while not self._stop.is_set():
                #Increment the current buffer
                current_buffer = (current_buffer+1)%self.num_buffers
                # ------------------------------------------------------------
                # Loop through the buffers and send data to pipeline task when accumulation terminate conditions are met.

                self.scan_accumulator_conditions[current_buffer].acquire()
                self.accumulator_logger.info('scan_accumulator_condition %d acquired by %s' %(current_buffer, self.name,))
            
                # accumulate data scan by scan into buffer arrays
                buffer_size = self.accumulate(spead_stream, self.buffers[current_buffer], ordering)
        
                # awaken pipeline task that was waiting for condition lock
                self.scan_accumulator_conditions[current_buffer].notify()
                self.accumulator_logger.info('scan_accumulator_condition %d notification sent by %s' %(current_buffer, self.name,))
                # release pipeline task that was waiting for condition lock
                self.scan_accumulator_conditions[current_buffer].release()
                self.accumulator_logger.info('scan_accumulator_condition %d released by %s' %(current_buffer, self.name,))

                time.sleep(0.5)
   
        def stop(self):        
            # set stop event
            self._stop.set()
            # stop SPEAD stream recieval
            self.capture_stop()
        
            # close off scan_accumulator_conditions
            #  - necessary for closing pipeline task which may be waiting on condition
            for scan_accumulator in self.scan_accumulator_conditions:
                scan_accumulator.acquire()
                scan_accumulator.notify()
                scan_accumulator.release()

        def stopped(self):
            return self._stop.isSet()
        
        def capture_stop(self):
            """
            Send stop packed to force shut down of SPEAD receiver
            """
            print 'sending stop packet'
            tx = spead.Transmitter(spead.TransportUDPtx(self.l0_endpoint.host,self.l0_endpoint.port))
            tx.end()
        
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
                current_buffer['track_start_indices'] = np.frombuffer(current_buffer['track_start_indices'], dtype=np.int32)
        
        def accumulate(self, spead_stream, data_buffer, ordering):
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

            ig = spead.ItemGroup()
        
            start_flag = True
            array_index = -1
            track_start_indices = []
        
            prev_activity = 'none'
            prev_tags = 'none'
        
            # get names of activity and target TS keys, using TS reference antenna
            target_key = '{0}_target'.format(self.telstate.cal_refant,)
            activity_key = '{0}_activity'.format(self.telstate.cal_refant,)
        
            # receive SPEAD stream
            print 'Got heaps: ',
            for heap in spead.iterheaps(spead_stream): 
                ig.update(heap)
                print array_index, 
                array_index += 1
            
                # get activity and target tag from TS
                activity = self.telstate[activity_key]
                target = self.telstate[target_key]
            
                # accumulate list of track start time indices in the array
                #   for use in the pipeline, to index each track easily 
                if 'track' in activity and not 'track' in prev_activity:
                    track_start_indices.append(array_index)
                
                # break if this scan is a slew that follows a track
                #   unless previous scan was a target, in which case accumulate subsequent gain scan too
                # ********** THIS BREAKING NEEDS TO BE THOUGHT THROUGH CAREFULLY ********** 
                if ('slew' in activity and 'track' in prev_activity) and 'target' not in prev_tags:
                    self.accumulator_logger.info('Accumulate break due to transition')
                    break

                if start_flag: 
                    start_time = ig['timestamp'] 
                    start_flag = False

                # reshape data and put into relevent arrays            
                data_buffer['vis'][array_index,:,:,:] = ig['correlator_data'][:,ordering].reshape([self.nchan,self.npol,self.nbl])
                data_buffer['flags'][array_index,:,:,:] = ig['flags'][:,ordering].reshape([self.nchan,self.npol,self.nbl])
                data_buffer['weights'][array_index,:,:,:] = ig['weights'][:,ordering].reshape([self.nchan,self.npol,self.nbl]) 
                data_buffer['times'][array_index] = ig['timestamp']

                # this is a temporary mock up of a natural break in the data stream
                # will ultimately be provided by some sort of sensor
                duration = ig['timestamp']-start_time
                if duration>2000000: 
                    self.accumulator_logger.info('Accumulate break due to duration')
                    break
                # end accumulation if maximum array size has been accumulated
                if array_index >= self.max_length - 1: 
                    self.accumulator_logger.info('Accumulate break due to buffer size limit')
                    break
                
                prev_activity = activity
                # extract tags from target description string
                prev_tags = target.split(',')[1]

            track_start_indices.append(array_index)                
            track_start_indices.append(-1)
            if 'multiprocessing' in str(control_method):
                # multiprocessing case
                data_buffer['track_start_indices'][0:len(track_start_indices)] = track_start_indices
            else:
                # threading case
                data_buffer['track_start_indices'] = np.array(track_start_indices)
    
            return array_index
    
    return accumulator_control(control_method, buffers, buffer_shape, scan_accumulator_conditions, l0_endpoint, telstate)
        
# ---------------------------------------------------------------------------------------
# Pipeline 
# ---------------------------------------------------------------------------------------
               
def init_pipeline_control(control_method, control_task, data, data_shape, scan_accumulator_condition, pipenum, l1_endpoint, telstate): 
               
    class pipeline_control(control_task):
        """
        Task (Process or Thread) which runs pipeline
        """

        def __init__(self, control_method, data, data_shape, scan_accumulator_condition, pipenum, l1_endpoint, telstate):
            control_task.__init__(self)
            self.data = data
            self.scan_accumulator_condition = scan_accumulator_condition
            self.name = 'Pipeline_task_'+str(pipenum)
            self._stop = control_method.Event()
            self.telstate = telstate
            self.l1_endpoint = l1_endpoint
            self.data_shape = data_shape
        
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
            return self._stop.isSet()
        
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
            self.data['track_start_indices'] = np.ctypeslib.as_array(self.data['track_start_indices'])        
        
        def run_pipeline(self):    
            # run pipeline calibration
            calibrated_data = pipeline(self.data,self.telstate,task_name=self.name)
            # if target data was calibated in the pipeline, send to L1 spead
            if calibrated_data is not None:
                self.pipeline_logger.info('Transmit L1 data')
                data_to_SPEAD(calibrated_data,self.l1_endpoint)
            
    return pipeline_control(control_method, data, data_shape, scan_accumulator_condition, pipenum, l1_endpoint, telstate)
        
# ---------------------------------------------------------------------------------------
# SPEAD transmission
# ---------------------------------------------------------------------------------------
        
def data_to_SPEAD(data,spead_endpoint):
    """
    Sends data to SPEAD stream
    
    data:
       list of: vis, flags, weights, times
    """
    
    tx = spead.Transmitter(spead.TransportUDPtx(spead_endpoint.host,spead_endpoint.port))

    # transmit data
    for i in range(len(data[-1])): # time axis

        tx_vis = data[0][i]
        tx_flags = data[1][i]
        tx_weights = data[2][i]
        tx_time = data[3][i]

        # transmit timestamps, vis, flags and weights
        transmit_item(tx, tx_time, tx_vis, tx_flags, tx_weights)
        # delay so receiver isn't overwhelmed
        time.sleep(0.05)
            
    end_transmit(tx)
    
def end_transmit(tx):
    """
    Send stop packet to spead stream tx
    
    Parameters
    ----------
    tx       : spead stream
    """
    tx.end()
    
def transmit_item(tx, tx_time, tx_vis, tx_flags, tx_weights):
    """
    Send spead packet containing time, visibility, flags and array state
    
    Parameters
    ----------
    tx         : spead stream
    tx_time    : timestamp, float
    tx_vis     : visibilities, complex array 
    tx_flags   : flags, int array
    tx_weights : weights, float array
    """
    ig = spead.ItemGroup()

    ig.add_item(name='timestamp', description='Timestamp',
        shape=[], fmt=spead.mkfmt(('f',64)),
        init_val=tx_time)

    ig.add_item(name='correlator_data', description='Full visibility array',
        init_val=tx_vis)

    ig.add_item(name='flags', description='Flag array',
        init_val=tx_flags)
        
    ig.add_item(name='weights', description='Weight array',
        init_val=tx_weights)

    tx.send_heap(ig.get_heap())
    