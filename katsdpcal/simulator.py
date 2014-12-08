"""Simulator class for HDF5 files produced by KAT-7 correlator,
   for testing of the MeerKAT pipeline."""

import katdal
from katdal import H5DataV2
import pickle
import os
import numpy as np

#--------------------------------------------------------------------------------------------------
#--- CLASS :  SimData
#--------------------------------------------------------------------------------------------------

class SimData(katdal.H5DataV2):
    
    def __init__(self, h5filename):
        H5DataV2.__init__(self, h5filename)
   
    def write_h5(self,data,corrprod_mask,tmask=None,cmask=None):
        """
        Writes data into h5 file
   
        Parameters
        ----------
        data          : data to write into the file
        corrprod_mask : correlation product mask showing where to write the data
        tmask         : time mask for timestamps to write
        cmask         : channel mask for channels to write
        """
        
        data_indices = np.squeeze(np.where(tmask)) if np.any(tmask) else np.squeeze(np.where(self._time_keep))
        ti, tf = min(data_indices), max(data_indices)+1 
        data_indices = np.squeeze(np.where(cmask)) if np.any(cmask) else np.squeeze(np.where(self._freq_keep))
        ci, cf = min(data_indices), max(data_indices)+1    
        self._vis[ti:tf,ci:cf,corrprod_mask,0] = data.real
        self._vis[ti:tf,ci:cf,corrprod_mask,1] = data.imag
   
    def setup_TM(self,TMfile,params):
        """
        Initialises the Telescope Model, optionally from existing TM pickle.
   
        Parameters
        ----------
        TMfile  : name of TM pickle file to open 
        params  : dictionary of default parameters

        Returns
        ------- 
        TM      : Telescope Model dictionary
        """   

        # initialise with parameter dictionary 
        TM = params
        # update TM with pickle file values
        TM_update = pickle.load(open(TMfile, 'rb')) if os.path.isfile(TMfile) else {}
        for key in TM_update: TM[key] = TM_update[key]

        # empty solutions - start with no solutions
        TM['BP'] = []
        TM['K'] = []
        TM['G'] = []
        TM['G_times'] = []

        # set siulated TM values from h5 file
        TM['antlist'] = [ant.name for ant in self.ants]
        TM['num_ants'] = len(self.ants)
        TM['num_channels'] = len(self.channels)
        #antdesclist = [ant.description for ant in simdata.ants]
        TM['corr_products'] = self.corr_products
        TM['dump_period'] = self.dump_period
   
        return TM
