import spead64_48 as spead
import threading
import time

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class accumulator_thread(threading.Thread):
    """
    Thread which accumutates data from spead into numpy arrays
    """

    def __init__(self, buffers, scan_accumulator_conditions, spead_port, spead_ip):
        threading.Thread.__init__(self)
        
        self.buffers = buffers
        self.spead_port = spead_port
        self.spead_ip = spead_ip
        self.scan_accumulator_conditions = scan_accumulator_conditions        
        self.num_buffers = len(buffers) 

        self.name = 'Accumulator_thread'
        self._stop = threading.Event()
        
        # flag for switching capture to the alternate buffer
        self._switch_buffer = False

        #Get data shape
        self.nchan = buffers[0]['vis'].shape[1]
        self.nbl = buffers[0]['vis'].shape[2]
        self.npol = buffers[0]['vis'].shape[3]
        
    def run(self):
        """
        Thread run method. Append random vis to the vis list
        at random time.
        """
        # Initialise SPEAD stream
        logger.info('RX: Initializing...')
        spead_stream = spead.TransportUDPrx(self.spead_port)

        #Initialise current buffer counter
        current_buffer=-1
        while not self._stop.isSet():
            #Increment the current buffer
            current_buffer = (current_buffer+1)%self.num_buffers
            # ------------------------------------------------------------
            # Loop through the buffers and send data to pipeline thread when accumulation terminate conditions are met.

            self.scan_accumulator_conditions[current_buffer].acquire()
            logger.debug('scan_accumulator_condition %d acquired by %s' %(current_buffer, self.name,))
            print 'scan_accumulator_condition %d acquired by %s' %(current_buffer, self.name,)
            
            # accumulate data scan by scan into buffer arrays
            buffer_size = self.accumulate(spead_stream, self.buffers[current_buffer])
            #time.sleep(1)
            
            # release buffer array for use in pipeline
            print 'scan_accumulator_condition %d notified by %s' % (current_buffer,self.name)
            self.scan_accumulator_conditions[current_buffer].notify()
            print 'scan_accumulator_condition %d released by %s' % (current_buffer,self.name)
            self.scan_accumulator_conditions[current_buffer].release()
            print '$1'
            #print 'times - acc ', self.times1.shape
            time.sleep(0.5)
   
    def stop(self):        
        # set stop event
        self._stop.set()
        # stop SPEAD stream recieval
        self.capture_stop()
        
        # close off scan_accumulator_conditions
        #  - necessary for closing pipeline thread which may be waiting on condition
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
        tx = spead.Transmitter(spead.TransportUDPtx(self.spead_ip,self.spead_port))
        tx.end()
        
    def accumulate(self, spead_stream, data_buffer):
        '''
        Accumulates spead data into arrays
           till **TBD** metadata indicates scan has stopped, or
           till array reaches max buffer size 
        '''

        ig = spead.ItemGroup()
        start_flag = True
        max_length = data_buffer['times'].shape[0]
        array_index = -1
        # receive SPEAD stream
        print 'Got heaps: ',
        for heap in spead.iterheaps(spead_stream): 
            ig.update(heap)
            print ig.heap_cnt, 
            array_index += 1
            if start_flag: 
                start_time = ig['timestamp'] 
                start_flag = False
    
            # reshape data and put into relevent arrays
            data_buffer['vis'][array_index,:,:,:] = ig['correlator_data'].reshape([self.nchan,self.nbl,self.npol])  
            data_buffer['flags'][array_index,:,:,:] = ig['flags'].reshape([self.nchan,self.nbl,self.npol])   
            data_buffer['times'][array_index] = ig['timestamp']

            # this is a temporary mock up of a natural break in the data stream
            # will ultimately be provided by some sort of sensor
            duration = ig['timestamp']-start_time
            if duration>15: break
            # end accumulation if maximum array size has been accumulated
            if array_index >= max_length - 1: break
    
        print
        return array_index
               
class pipeline_thread(threading.Thread):
    """
    Thread which runs pipeline
    """

    def __init__(self, data, scan_accumulator_condition, pipenum):
        threading.Thread.__init__(self)
        self.data = data
        self.scan_accumulator_condition = scan_accumulator_condition
        self.name = 'Pipeline_thread_'+str(pipenum)
        self._stop = threading.Event()
    
    def run(self):
        """
        Thread run method. Runs pipeline
        """
        
        # run until stop is set   
        while not self._stop.isSet():
            
            # acquire condition on data
            self.scan_accumulator_condition.acquire()
            print 'scan_accumulator_condition acquired by %s' % self.name

            # run the pipeline - mock up for now
            run_pipeline(self.data)
            
            # then wait for next condition on data
            print 'scan_accumulator_condition wait by %s' % self.name
            self.scan_accumulator_condition.wait()
            print 'condition released by %s' % self.name
            
            self.scan_accumulator_condition.release()
        
    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()
        
        
def run_pipeline(data):
    print 'pipeline! ', data['times'][0:5], data['times'].shape, data['vis'][3,0]
