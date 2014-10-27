
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
   axes_[0].plot(chans,np.abs(data),'.-') #,color=PLOT_COLORS[plotnum])
   axes_[0].set_xlim([0,max(chans)])
   axes_[0].set_ylabel('Amplitude'+ylabelplus)
        
   # plot phase
   axes_[1].plot(chans,360.*np.angle(data)/(2.*np.pi),'.-')
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
   
def plot_bp_soln_list(bplist,chans=None):
   """
   Plots bandpass solutions
   
   Parameters
   ----------
   data  : array of complex, shape(num_chans,num_ants)
   chans : channel numbers, shape(num_chans)
   """
   # just label channels from zero if channel numbers not supplied
   if not chans: chans = np.arange(bplist[0].shape[-2])

   nrows, ncols = 1,2 
   fig, axes = plt.subplots(nrows,ncols,figsize=(14.0*ncols,3.5*nrows))
   for bp in bplist:
      plt_v_chan(bp,axes,plotnum=0)

   plt.show()
   
def plot_g_solns(times,data):
   """
   Plots gain solutions
   
   Parameters
   ----------
   data  : array of complex, shape(num_times,num_ants)
   """
   nrows, ncols = 1,2 
   fig, axes = plt.subplots(nrows,ncols,figsize=(14.0*ncols,4.0*nrows))
   
   times = np.array(times) - times[0]
   data = np.array(data)

   # plot amplitude
   axes[0].plot(times/60.,np.abs(data),'.-')
   axes[0].set_ylabel('Amplitude')
        
   # plot phase
   axes[1].plot(times/60.,360.*np.angle(data)/(2.*np.pi),'.-')
   axes[1].set_ylabel('Phase')

   axes[0].set_xlabel('Time / [min]') 
   axes[1].set_xlabel('Time / [min]') 

   plt.show()

def plot_waterfall(visdata,contrast=0.01,flags=None,channel_freqs=None):
   """
   Make a waterfall plot from visdata- with an option to plot flags
   and show the frequency axis in MHz.
        
   Parameters
   ----------
   visdata       : array (ntimestamps,nchannels) of floats
   contrast      : percentage of maximum and minimum data values to remove from lookup table
   flags         : array of boolean with same shape as visdata
   channel_freqs : array (nchannels) of frequencies represented by each channel
   """

   fig = plt.figure(figsize=(8.3,11.7))
   kwargs={'aspect' : 'auto', 'origin' : 'lower', 'interpolation' : 'none'}
   if channel_freqs: kwargs['extent'] = (channel_freqs[0],channel_freqs[1], -0.5, data.shape[0] - 0.5)
   image=plt.imshow(visdata,**kwargs)
   image.set_cmap('Rainbow')
   #flagimage=plt.imshow(flags[:,:,0],**kwargs)
   #Make an array of RGBA data for the flags (initialize to alpha=0)
   if flags:
      plotflags = np.zeros(flags.shape[0:2]+(4,))
      plotflags[:,:,0] = 1.0
      plotflags[:,:,3] = flags[:,:,0]
      plt.imshow(plotflags,**kwargs)

   ampsort=np.sort(visdata,axis=None)
   arrayremove = int(len(ampsort)*contrast)
   lowcut,highcut = ampsort[arrayremove],ampsort[-(arrayremove+1)]
   image.norm.vmin = lowcut
   image.norm.vmax = highcut
   plt.show()
   plt.close(fig)