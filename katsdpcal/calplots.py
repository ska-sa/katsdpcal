
import matplotlib.pylab as plt
import numpy as np

#PLOT_COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] 

def plt_v_chan(data,axes,plotnum=0,chans=None,ylabelplus=''):
   """
   Plots data versus channel
  
   Parameters
   ----------
   data    : array of complex, shape(num_chans, num_ants)
   plotnum : location of plot on axes
   chans   : channel numbers, shape(num_chans)
   ylabelplus : additional y-label text, type(string)
   """

   if len(axes.shape) > 1:
      axes_ = axes[plotnum]
   elif len(axes.shape) == 1:
      axes_ = axes
   # raise error here for more axes?

   if not chans: chans = np.arange(data.shape[-2])

   # plot amplitude
   axes_[0].plot(chans,np.abs(data),'.') #,color=PLOT_COLORS[plotnum])
   axes_[0].set_xlim([0,max(chans)])
   axes_[0].set_ylabel('Amplitude'+ylabelplus)
        
   # plot phase
   axes_[1].plot(chans,360.*np.angle(data)/(2.*np.pi),'.')
   axes_[1].set_xlim([0,max(chans)])
   axes_[1].set_ylabel('Phase'+ylabelplus)

   axes_[0].set_xlabel('Channels') 
   axes_[1].set_xlabel('Channels') 

def plot_bp_data(data,chans=None,plotavg=False): 
   """
   Plots bandpass data versus channel
  
   Parameters
   ----------
   data    : array of complex, shape(num_times, num_chans, num_ants)
   chans   : channel numbers, shape(num_chans)
   plotavg : plot additional panel of the average of data over time
   """   
   # just label channels from zero if channel numbers not supplied
   if not chans: chans = np.arange(data.shape[-2])

   if plotavg:
      nrows, ncols = 2,2 
   else:
      nrows, ncols = 1,2 
   fig, axes = plt.subplots(nrows,ncols,figsize=(14.0*ncols,3.5*nrows))

   tlist = np.arange(data.shape[0])
   for ti in tlist:
      plt_v_chan(data[ti],axes,plotnum=0)
        
   if plotavg: plt_v_chan(np.nanmean(data,axis=0),axes,plotnum=1,ylabelplus=' (Avg)')

   plt.show()
    
def plot_bp_solns(data,chans=None):
   """
   Plots bandpass solutions
   
   Parameters
   ----------
   data  : array of complex, shape(num_chans,num_ants)
   chans : channel numbers, shape(num_chans)
   """
   # just label channels from zero if channel numbers not supplied
   if not chans: chans = np.arange(data.shape[-2])

   nrows, ncols = 1,2 
   fig, axes = plt.subplots(nrows,ncols,figsize=(14.0*ncols,3.5*nrows))
   plt_v_chan(data,axes,plotnum=0)

   plt.show()
   
def plot_g_solns(data):
   """
   Plots gain solutions
   
   Parameters
   ----------
   data  : array of complex, shape(num_times,num_ants)
   """
   nrows, ncols = 1,2 
   fig, axes = plt.subplots(nrows,ncols,figsize=(14.0*ncols,3.5*nrows))

   # plot amplitude
   axes[0].plot(np.abs(data),'.')
   axes[0].set_ylabel('Amplitude')
        
   # plot phase
   axes[1].plot(360.*np.angle(data)/(2.*np.pi),'.')
   axes[1].set_ylabel('Phase')

   axes[0].set_xlabel('Time') 
   axes[1].set_xlabel('Time') 

   plt.show()
