
import katdal
import pickle
import os
import numpy as np

def get_h5_simdata(h5filename):
   """
   Opens h5 file to extract data and TM ekements from
   
   Parameters
   ----------
   h5filename : name of h5 file to open 

   Returns
   ------- 
   katdal object
   """
   return katdal.open(h5filename)
   
def write_h5_simdata(simdata,data,corrprod_mask):
   """
   Writes data into h5 file
   
   Parameters
   ----------
   simdata       : h5 file to write data into (katdal object)
   data          : data to write into the file
   corrprod_mask : correlation product mask showing where to write the data
   """
   data_indices = np.squeeze(np.where(simdata._time_keep))
   ti, tf = min(data_indices), max(data_indices)+1   
   data_indices = np.squeeze(np.where(simdata._freq_keep))
   ci, cf = min(data_indices), max(data_indices)+1    
   simdata._vis[ti:tf,ci:cf,corrprod_mask,0] = data.real
   simdata._vis[ti:tf,ci:cf,corrprod_mask,1] = data.imag
   
def setup_TM(TMfile,simdata):
   """
   Initialises the Telescope Model, optionally from existing TM pickle.
   
   Parameters
   ----------
   TMfile  : name of TM pickle file to open 
   simdata : katdal object containing simulated data and metadata 

   Returns
   ------- 
   TM      : Telescope Model dictionary
   """   

   TM = pickle.load(open(TMfile, 'rb')) if os.path.isfile(TMfile) else {}

   # empty solutions - start with no solutions
   TM['BP'] = []
   TM['K'] = []
   TM['G'] = []

   # set siulated TM values from h5 file
   TM['antlist'] = [ant.name for ant in simdata.ants]
   TM['num_ants'] = len(simdata.ants)
   TM['num_channels'] = len(simdata.channels)
   #antdesclist = [ant.description for ant in simdata.ants]
   TM['corr_products'] = simdata.corr_products
   TM['dump_period'] = simdata.dump_period
   
   return TM
