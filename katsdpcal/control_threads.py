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

    def __init__(self, times1, vis1, flags1, scan_accumulator_condition1,
          times2, vis2, flags2, scan_accumulator_condition2, port):
        threading.Thread.__init__(self)
        
        self.vis1 = vis1
        self.times1 = times1
        self.flags1 = flags1
        self.vis2 = vis2
        self.times2 = times2
        self.flags2 = flags2
        
        self.port = port
        
        self.scan_accumulator_condition1 = scan_accumulator_condition1
        self.scan_accumulator_condition2 = scan_accumulator_condition2        
        
        self.name = 'Accumulator_thread'
        self._stop = threading.Event()
        
        # flag for switching capture to the alternate buffer
        self._switch_buffer = False
        
        # assume vis1 and vis 2 have same shape
        self.array_length = self.vis1.shape[0]
        self.nchan = self.vis1.shape[1]
        self.nbl = self.vis1.shape[2]
        self.npol = self.vis1.shape[3]
    
    def run(self):
        """
        Thread run method. Append random vis to the vis list
        at random time.
        """
        # Initialise SPEAD stream
        logger.info('RX: Initializing...')
        spead_stream = spead.TransportUDPrx(self.port)
        
        # accumulate data into the arrays
        array_index1 = -1
        array_index2 = -1
        
        #with self.scan_accumulator_condition:
        while True:
          while not self._stop.isSet() and not self._switch_buffer:
            
            
            # ------------------------------------------------------------
            # first accumulator 
            
            array_index1 += 1

            self.scan_accumulator_condition1.acquire()
            logger.debug('scan_accumulator_condition1 acquired by {0}'.format(self.name,))
            print 'scan_accumulator_condition1 acquired by {0}'.format(self.name,)
            
            # accumulate data scan by scan into buffer arrays
            array_index1 = self.accumulate(spead_stream, array_index1, self.array_length, self.vis1, self.flags1, self.times1)
            #time.sleep(1)
            
            if True: #array_index > 30:
                #self._switch_buffer = True
                # release buffer array for use in pipeline
                print 'scan_accumulator_condition1 notified by %s' % self.name
                self.scan_accumulator_condition1.notify()
                print 'scan_accumulator_condition1 released by %s' % self.name
                self.scan_accumulator_condition1.release()
            print '$1'
            print 'times - acc ', self.times1.shape
            time.sleep(0.5)
            
            # ------------------------------------------------------------
            # second accumulator 
            
            array_index2 += 1

            self.scan_accumulator_condition2.acquire()
            logger.debug('scan_accumulator_condition2 acquired by {0}'.format(self.name,))
            print 'scan_accumulator_condition2 acquired by {0}'.format(self.name,)
            
            # accumulate data scan by scan into buffer arrays
            array_index1 = self.accumulate(spead_stream, array_index2, self.array_length, self.vis2, self.flags2, self.times2)
            #time.sleep(1)
            
            if True: #array_index > 30:
                #self._switch_buffer = True
                # release buffer array for use in pipeline
                print 'scan_accumulator_condition1 notified by %s' % self.name
                self.scan_accumulator_condition2.notify()
                print 'scan_accumulator_condition1 released by %s' % self.name
                self.scan_accumulator_condition2.release()
            print '$1'
            print 'times - acc ', self.times2.shape
            time.sleep(0.5)
 
    def stop(self):        
        # set stop event
        self._stop.set()
        # stop SPEAD stream recieval
        self.capture_stop()
        
        # close off scan_accumulator_condition1 and 2 on data
        #  - necessary for closing pipeline thread which may be waiting on condition
        self.scan_accumulator_condition1.acquire()
        self.scan_accumulator_condition1.notify()
        self.scan_accumulator_condition1.release()
        self.scan_accumulator_condition2.acquire()
        self.scan_accumulator_condition2.notify()
        self.scan_accumulator_condition2.release()
        
    def stopped(self):
        return self._stop.isSet()
        
    def capture_stop(self):
        """
        Send stop packed to force shut down of SPEAD receiver
        """
        print 'sending stop packet'
        tx = spead.Transmitter(spead.TransportUDPtx('localhost',self.port))
        tx.end()
        
    def accumulate(self, spead_stream, array_index, max_length, vis, flags, times):
        '''
        Accumulates spead data into arrays
           till **TBD** metadata indicates scan has stopped, or
           till array reaches max buffer size 
        '''

        ig = spead.ItemGroup()
        start_flag = True
    
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
            vis[array_index,:,:,:] = ig['correlator_data'].reshape([self.nchan,self.nbl,self.npol])  
            flags[array_index,:,:,:] = ig['flags'].reshape([self.nchan,self.nbl,self.npol])   
            times[array_index] = ig['timestamp']

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

    def __init__(self, times, vis, flags, scan_accumulator_condition, pipenum):
        threading.Thread.__init__(self)
        self.vis = vis
        self.times = times
        self.flags = flags
        self.scan_accumulator_condition = scan_accumulator_condition
        self.name = 'Pipeline_thread_'+pipenum
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
            run_pipeline(self.vis,self.flags,self.times)
            
            # then wait for next condition on data
            print 'scan_accumulator_condition wait by %s' % self.name
            self.scan_accumulator_condition.wait()
            print 'condition released by %s' % self.name
            
            self.scan_accumulator_condition.release()
        
    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()
        
        
def run_pipeline(vis,flags,times):
    print 'pipeline! ', times[0:5], times.shape, vis[3,0]
