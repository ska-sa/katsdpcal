
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
        
   def simple_interpolate(self, dump_period, num_dumps):
      dumps_per_solint = int(self.solint/np.round(dump_period,3))
      interpSolution = self
      interpSolution.values = np.repeat(self.values,dumps_per_solint,axis=0)[0:num_dumps]
      return interpSolution
        
class CalSolution(Solution):
   
   def __init__(self, soltype, solvalues, times, solint, corrprod_lookup, **kwargs):
      super(CalSolution, self).__init__(soltype, solvalues, times, solint, corrprod_lookup, **kwargs)
      
   def interpolate(self, dump_period, num_dumps):
      # set up more complex interpolation methods later
      if self.soltype is 'G': 
         return self.simple_interpolate(dump_period, num_dumps)
      
   def apply(self, data, chans=None):
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
         data[:,:,cp] /= np.atleast_2d(self.values[:,self.corrprod_lookup[cp][0]]*(self.values[:,self.corrprod_lookup[cp][1]].conj())).T

      return data
      
      
      
      
            
      
        
   