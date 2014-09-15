
import numpy as np

#--------------------------------------------------------------------------------------------------
#--- CLASS :  CalSolution
#--------------------------------------------------------------------------------------------------

class Solution(object):
   
   def __init__(self, soltype, solvalues, times, solint, corrprod_lookup, **kwargs):
      self.soltype = soltype
      self.values = solvalues
      # start times of each solution
      self.times = times
      # solint - the nominal time interval solutions were calculated over
      self.solint = solint
      self.corrprod_lookup = corrprod_lookup
      # kwargs may include, for example, ???
      for key, value in kwargs.items():
         setattr(self, key, value)
        
   def simple_interpolate(self, num_dumps, **kwargs):
      dump_period = kwargs['dump_period']
      dumps_per_solint = int(self.solint/np.round(dump_period,3))
      interpSolution = self
      interpSolution.values = np.repeat(self.values,dumps_per_solint,axis=0)[0:num_dumps]
      return interpSolution
      
   def inf_interpolate(self, num_dumps):
      interpSolution = self
      interpSolution.values = np.repeat(np.expand_dims(self.values,axis=0),num_dumps,axis=0)
      return interpSolution
      
   def _apply(self, data, solns, chans=None):
      """
      Applies calibration solutions.
      Must already be interpolated to either full time or full frequency.
   
      Parameters
      ----------
      data     : array of complex, shape(num_times, num_chans, baseline)
      chans    :

      Returns
      -------
      """     
      for cp in range(len(self.corrprod_lookup)):
         if len(solns.shape) < 3:
            data[:,:,cp] /= np.expand_dims(solns[...,self.corrprod_lookup[cp][0]]*(solns[...,self.corrprod_lookup[cp][1]].conj()),axis=1)
         else:
            data[:,:,cp] /= solns[...,self.corrprod_lookup[cp][0]]*(solns[...,self.corrprod_lookup[cp][1]].conj())

      return data
        
class CalSolution(Solution):
   
   def __init__(self, soltype, solvalues, times, solint, corrprod_lookup, **kwargs):
      super(CalSolution, self).__init__(soltype, solvalues, times, solint, corrprod_lookup, **kwargs)
      
   def interpolate(self, num_dumps, **kwargs):
      # set up more complex interpolation methods later
      if self.soltype is 'G': 
         return self.simple_interpolate(num_dumps, **kwargs)         
      if self.soltype is 'K': 
         return self.inf_interpolate(num_dumps)
      if self.soltype is 'B': 
         return self.inf_interpolate(num_dumps)
      
   def apply(self, data, chans=None):
      # set up more complex interpolation methods later
      if self.soltype is 'G': 
         return self._apply(data, self.values)    
      if self.soltype is 'K': 
         # want dimensions num_times x num_chans x num_ants
         g_from_k = np.zeros([data.shape[0],data.shape[1],self.values.shape[-1]],dtype=np.complex)
         for c in chans:
            g_from_k[:,c,:] = np.exp(1.0j*self.values*c)
         return self._apply(data, g_from_k)
      if self.soltype is 'B': 
         return self._apply(data, self.values)

      return data
      
      
      
      
            
      
        
   