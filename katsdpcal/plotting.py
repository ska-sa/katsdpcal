import matplotlib.pylab as plt
import numpy as np

# for multiple page pdf plotting
from matplotlib.backends.backend_pdf import PdfPages

#PLOT_COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k'] 
    
def flush_plots(fig_list,report_name='cal_report.pdf'):
    """
    Plots accumulated figures to pdf document and screen
    
    Parameters
    ----------
    fig_list    : list of matplotlib figures to plot
    report_name : name of pdf to save
    """
    
    # create multi-page pdf document for report
    pdf_pages = PdfPages(report_name)
    # plot each figure to a separate page 
    #  (till we decide on a better way to do the reporting)
    for fig in fig_list:
        pdf_pages.savefig(fig)
    pdf_pages.close()
    
    # also plot figures to screen
    plt.show()

def plot_data_v_chan(data,axes,plotnum=0,chans=None,ylabelplus=''):
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
    fig, axes = plt.subplots(nrows,ncols,figsize=(18.0*ncols,5.0*nrows))

    tlist = np.arange(data.shape[0])
    for ti in tlist:
        plot_data_v_chan(data[ti],axes,plotnum=0)
        
    if plotavg: plot_data_v_chan(np.nanmean(data,axis=0),axes,plotnum=1,ylabelplus=' (Avg)')

    return fig
    
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
    fig, axes = plt.subplots(nrows,ncols,figsize=(18.0*ncols,5.0*nrows))
    plot_data_v_chan(data,axes,plotnum=0)

    return fig
   
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
    fig, axes = plt.subplots(nrows,ncols,figsize=(18.0*ncols,5.0*nrows))
    for bp in bplist:
        plot_data_v_chan(bp,axes,plotnum=0)

    return fig
   
def plot_g_solns(times,data):
    """
    Plots gain solutions
   
    Parameters
    ----------
    data   : array of complex, shape(num_times,num_ants)
    """
    nrows, ncols = 1,2 
    fig, axes = plt.subplots(nrows,ncols,figsize=(18.0*ncols,5.0*nrows))
   
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

    return fig
    
def plot_g_solns_with_errors(times,data,stddev):
    """
    Plots gain solutions and colour fill ranges for amplitude errors
   
    Parameters
    ----------
    data   : array of complex, shape(num_times,num_ants)
    stddev : array of real, shape(num_times,num_ants)
    """
    
    nrows, ncols = 2,2 
    fig, axes = plt.subplots(nrows,ncols,figsize=(18.0*ncols,5.0*nrows))
   
    times = np.array(times) - times[0]
    data = np.array(data)

    # plot amplitude
    amp = np.abs(data)
    amp_max = amp + stddev
    amp_min = amp - stddev
    
    axes[0,0].plot(times/60.,amp,'.-')
    #axes[0,0].fill_between(times/60.,amp_min,amp_max,alpha=0.1)
    for y in zip(amp_min.T,amp_max.T):
        y0, y1 = y
        axes[0,0].fill_between(times/60.,y0,y1,alpha=0.1,color='k')
    axes[0,0].set_ylabel('Amplitude')
        
    # plot phase
    axes[0,1].plot(times/60.,360.*np.angle(data)/(2.*np.pi),'.-')
    axes[0,1].set_ylabel('Phase')

    axes[0,0].set_xlabel('Time / [min]') 
    axes[0,1].set_xlabel('Time / [min]') 
    
    axes[1,0].plot(times/60.,stddev,'.-')

    return fig

def plot_waterfall(visdata,contrast=0.01,flags=None,channel_freqs=None,dump_timestamps=None):
    """
    Make a waterfall plot from visdata- with an option to plot flags
    and show the frequency axis in MHz and dump in utc seconds if provided.
        
    Parameters
    ----------
    visdata         : array (ntimestamps,nchannels) of floats (amp or phase).
    contrast        : percentage of maximum and minimum data values to remove from lookup table
    flags           : array of boolean with same shape as visdata
    channel_freqs   : array (nchannels) of frequencies represented by each channel
    dump_timestamps : array (ntimestamps) of timestamps represented by each dump
    """

    fig = plt.figure(figsize=(8.3,11.7))
    kwargs={'aspect' : 'auto', 'origin' : 'lower', 'interpolation' : 'none'}
    #Defaults
    kwargs['extent'] = [-0.5, visdata.shape[1] - 0.5, -0.5, visdata.shape[0] -0.5]
    plt.xlabel('Channel number')
    plt.ylabel('Dump number')
    #Change defaults if frequencies or times specified.
    if channel_freqs is not None: 
        kwargs['extent'][0],kwargs['extent'][1] = channel_freqs[0],channel_freqs[-1]
        #reverse the data if the frequencies are in descending order
        if channel_freqs[1]-channel_freqs[0] < 0:
            visdata=visdata[:,::-1]
            kwargs['extent'][0],kwargs['extent'][1] = channel_freqs[-1]/1e6,channel_freqs[0]/1e6
            plt.xlabel('Frequency (MHz)')
    if dump_timestamps is not None:
        kwargs['extent'][2],kwargs['extent'][3] = 0,dump_timestamps[-1]-dump_timestamps[0]
        plt.ylabel('Time (UTC seconds)')
    image=plt.imshow(visdata,**kwargs)
    image.set_cmap('Greys')
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

def plot_RFI_mask(pltobj,extra=None,channelwidth=1e6):
    """
    Plot the frequencies of know rfi satellites on a spectrum

    Parameters
    ----------
    plotobj       : the matplotlib plot object to plot the RFI onto
    extra         : the locations of extra masks to plot
    channelwidth  : the width of the mask per channel
    """

    pltobj.axvspan(1674e6,1677e6, alpha=0.3, color='grey')#Meteosat
    pltobj.axvspan(1667e6,1667e6, alpha=0.3, color='grey')#Fengun
    pltobj.axvspan(1682e6,1682e6, alpha=0.3, color='grey')#Meteosat
    pltobj.axvspan(1685e6,1687e6, alpha=0.3, color='grey')#Meteosat
    pltobj.axvspan(1687e6,1687e6, alpha=0.3, color='grey')#Fengun
    pltobj.axvspan(1690e6,1690e6, alpha=0.3, color='grey')#Meteosat
    pltobj.axvspan(1699e6,1699e6, alpha=0.3, color='grey')#Meteosat
    pltobj.axvspan(1702e6,1702e6, alpha=0.3, color='grey')#Fengyun
    pltobj.axvspan(1705e6,1706e6, alpha=0.3, color='grey')#Meteosat
    pltobj.axvspan(1709e6,1709e6, alpha=0.3, color='grey')#Fengun
    pltobj.axvspan(1501e6,1570e6, alpha=0.3, color='blue')#Inmarsat
    pltobj.axvspan(1496e6,1585e6, alpha=0.3, color='blue')#Inmarsat
    pltobj.axvspan(1574e6,1576e6, alpha=0.3, color='blue')#Inmarsat
    pltobj.axvspan(1509e6,1572e6, alpha=0.3, color='blue')#Inmarsat
    pltobj.axvspan(1574e6,1575e6, alpha=0.3, color='blue')#Inmarsat
    pltobj.axvspan(1512e6,1570e6, alpha=0.3, color='blue')#Thuraya
    pltobj.axvspan(1450e6,1498e6, alpha=0.3, color='red')#Afristar
    pltobj.axvspan(1652e6,1694e6, alpha=0.2, color='red')#Afristar
    pltobj.axvspan(1542e6,1543e6, alpha=0.3, color='cyan')#Express AM1
    pltobj.axvspan(1554e6,1554e6, alpha=0.3, color='cyan')#Express AM 44
    pltobj.axvspan(1190e6,1215e6, alpha=0.3, color='green')#Galileo
    pltobj.axvspan(1260e6,1300e6, alpha=0.3, color='green')#Galileo
    pltobj.axvspan(1559e6,1591e6, alpha=0.3, color='green')#Galileo
    pltobj.axvspan(1544e6,1545e6, alpha=0.3, color='green')#Galileo
    pltobj.axvspan(1190e6,1217e6, alpha=0.3, color='green')#Beidou
    pltobj.axvspan(1258e6,1278e6, alpha=0.3, color='green')#Beidou
    pltobj.axvspan(1559e6,1563e6, alpha=0.3, color='green')#Beidou  
    pltobj.axvspan(1555e6,1596e6, alpha=0.3, color='green')#GPS L1  1555 -> 1596 
    pltobj.axvspan(1207e6,1238e6, alpha=0.3, color='green')#GPS L2  1207 -> 1188 
    pltobj.axvspan(1378e6,1384e6, alpha=0.3, color='green')#GPS L3  
    pltobj.axvspan(1588e6,1615e6, alpha=0.3, color='green')#GLONASS  1588 -> 1615 L1
    pltobj.axvspan(1232e6,1259e6, alpha=0.3, color='green')#GLONASS  1232 -> 1259 L2
    pltobj.axvspan(1616e6,1630e6, alpha=0.3, color='grey')#IRIDIUM
    if not extra is None:
        for i in xrange(extra.shape[0]):
                pltobj.axvspan(extra[i]-channelwidth/2,extra[i]+channelwidth/2, alpha=0.7, color='Maroon')