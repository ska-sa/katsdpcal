import numpy as np
import time

# use Agg backend for when the pipeline is run without an X1 connection
from matplotlib import use
use('Agg', warn=False)

import matplotlib.pylab as plt
from matplotlib.ticker import FuncFormatter

# for multiple page pdf plotting
from matplotlib.backends.backend_pdf import PdfPages

# PLOT_COLORS = ['r', 'g', 'b', 'c', 'm', 'y', 'k']


def flush_plots(fig_list, report_name='cal_report.pdf'):
    """
    Plots accumulated figures to pdf document and screen

    Parameters
    ----------
    fig_list : list
        list of matplotlib figures to plot
    report_name : str
        name of pdf to save
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


def plot_data_v_chan(data, axes, plotnum=0, chans=None, ylabelplus=''):
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

    if not chans:
        chans = np.arange(data.shape[-2])

    # plot amplitude
    axes_[0].plot(chans, np.abs(data), '.-')  # ,color=PLOT_COLORS[plotnum])
    axes_[0].set_xlim([0, max(chans)])
    axes_[0].set_ylabel('Amplitude' + ylabelplus)

    # plot phase
    axes_[1].plot(chans, 360. * np.angle(data) / (2. * np.pi), '.')
    axes_[1].set_xlim([0, max(chans)])
    axes_[1].set_ylabel('Phase' + ylabelplus)

    axes_[1].set_xlabel('Channels')


def plot_bp_data(data, chans=None, plotavg=False):
    """
    Plots bandpass data versus channel

    Parameters
    ----------
    data    : array of complex, shape(num_times, num_chans, num_pol, num_ants)
    chans   : channel numbers, shape(num_chans)
    plotavg : plot additional panel of the average of data over time
    """
    # just label channels from zero if channel numbers not supplied
    if not chans:
        chans = np.arange(data.shape[-2])

    if plotavg:
        nrows, ncols = 2, 2
    else:
        nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(18.0 * ncols, 5.0 * nrows))

    tlist = np.arange(data.shape[0])
    for ti in tlist:
        plot_data_v_chan(data[ti], axes, plotnum=0)

    if plotavg:
        plot_data_v_chan(np.nanmean(data, axis=0), axes, plotnum=1, ylabelplus=' (Avg)')

    # debug
    # plt.savefig('tst.png')

    return fig


def plot_bp_solns(data, chans=None):
    """
    Plots bandpass solutions

    Parameters
    ----------
    data  : array of complex, shape(num_chans,num_pols,num_ants)
    chans : channel numbers, shape(num_chans)
    """
    # just label channels from zero if channel numbers not supplied
    if not chans:
        chans = np.arange(data.shape[-2])

    npols = data.shape[-2]
    nrows, ncols = npols, 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(10.0 * ncols, 3.0 * nrows))
    for p in range(npols):
        plot_data_v_chan(data[:, p, :], axes, plotnum=p, ylabelplus=' - POL ' + str(p))

    return fig


def plot_bp_soln_list(bplist, chans=None):
    """
    Plots bandpass solutions

    Parameters
    ----------
    data  : array of complex, shape(num_chans,num_ants)
    chans : channel numbers, shape(num_chans)
    """
    # just label channels from zero if channel numbers not supplied
    if not chans:
        chans = np.arange(bplist[0].shape[-2])

    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(18.0 * ncols, 5.0 * nrows))
    for bp in bplist:
        plot_data_v_chan(bp, axes, plotnum=0)

    return fig


def plot_g_solns(times, data):
    """
    Plots gain solutions

    Parameters
    ----------
    data   : array of complex, shape(num_times,num_ants)
    """
    nrows, ncols = 1, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.0 * ncols, 3.0 * nrows))

    times = np.array(times) - times[0]
    data = np.array(data)

    # plot amplitude
    axes[0].plot(times / 60., np.abs(data), '.-')
    axes[0].set_ylabel('Amplitude')

    # plot phase
    axes[1].plot(times / 60., 360. * np.angle(data) / (2. * np.pi), '.-')
    axes[1].set_ylabel('Phase')

    axes[0].set_xlabel('Time / [min]')
    axes[1].set_xlabel('Time / [min]')

    return fig


def plot_g_solns_with_errors(times, data, stddev):
    """
    Plots gain solutions and colour fill ranges for amplitude errors

    Parameters
    ----------
    data   : array of complex, shape(num_times, num_ants)
    stddev : array of real, shape(num_times, num_ants)
    """

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(18.0 * ncols, 5.0 * nrows))

    times = np.array(times) - times[0]
    data = np.array(data)

    # plot amplitude
    amp = np.abs(data)
    amp_max = amp + stddev
    amp_min = amp - stddev

    axes[0, 0].plot(times / 60., amp, '.-')
    # axes[0, 0].fill_between(times/60.,amp_min,amp_max,alpha=0.1)
    for y in zip(amp_min.T, amp_max.T):
        y0, y1 = y
        axes[0, 0].fill_between(times / 60., y0, y1, alpha=0.1, color='k')
    axes[0, 0].set_ylabel('Amplitude')

    # plot phase
    axes[0, 1].plot(times / 60., 360. * np.angle(data) / (2. * np.pi), '.-')
    axes[0, 1].set_ylabel('Phase')

    axes[0, 0].set_xlabel('Time / [min]')
    axes[0, 1].set_xlabel('Time / [min]')

    axes[1, 0].plot(times / 60., stddev, '.-')

    return fig


def plot_g_solns_legend(times, data, antlist=None, t_zero=None):
    """
    Plots gain solutions

    Parameters
    ----------
    data   : array of complex, shape(num_times,num_pols,num_ants)
    """
    npols = data.shape[-2]
    nrows, ncols = npols, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.0*ncols, 3.0*nrows))
    if t_zero is None:
        t_zero = times[0]
    t = time.strftime("%Y %x %X", time.gmtime(t_zero))
    times = np.array(times) - t_zero
    data = np.array(data)

    fig, axes = plt.subplots(nrows, ncols, figsize=(10.0*ncols, 4.0*nrows),
                             squeeze=False, sharey='col')
    for p in range(npols):
        # plot amplitude
        p1 = axes[p, 0].plot(times / 60., np.abs(data[:, p, :]), '.-')
        axes[p, 0].set_ylabel('Amplitude'+' Pol_{0}'.format(p))

        # plot phase
        axes[p, 1].plot(times / 60., 360. * np.angle(data[:, p, :]) / (2.*np.pi), '.-')
        axes[p, 1].set_ylabel('Phase'+' Pol_{0}'.format(p))

        plt.setp(axes[p, 0].get_xticklabels(), visible=False)
        plt.setp(axes[p, 1].get_xticklabels(), visible=False)

    # for the last row, add in xticklabels and xlabels
    l_p = npols-1
    plt.setp(axes[l_p, 0].get_xticklabels(), visible=True)
    plt.setp(axes[l_p, 1].get_xticklabels(), visible=True)
    axes[l_p, 0].set_xlabel('Time since {0} (UTC) [min]'.format(t))
    axes[l_p, 1].set_xlabel('Time since {0} (UTC) [min]'.format(t))

    if antlist is not None:
        axes[0, 1].legend(p1, antlist, bbox_to_anchor=(1.0, 1.0), loc="upper left", frameon=False)

    return fig


def flags_bl_v_chan(data, chan, freq_range=None):
    """
    Make a waterfall plot of flagged data in Channels vs Baselines

    Parameters
    ----------
    data    : array of real, shape(num_chans, num_pol, num_baselines)
    freq_range : list of start and stop frequencies of the array, optional
    """
    npols = data.shape[-2]
    nbls = data.shape[-1]
    ncols = npols
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.0 * ncols, 4 * nrows), squeeze=False)
    for p in range(npols):
        im = axes[0, p].imshow(data[:, p, :].transpose(), extent=(chan[0],chan[-1],0,nbls), aspect='auto', origin='lower')
        axes[0, p].set_ylabel('Pol {0} Baselines'.format(p))
        axes[0, p].set_xlabel('Channels')
    # Add colorbar
    cax = fig.add_axes([0.92, 0.12, 0.02, 0.75])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('% time flagged')

    if freq_range is not None:
        for ax in axes.flatten()[0:2]:
            add_freq_axis(ax,chan_range=[chan[0],chan[-1]],freq_range=freq_range)
    return fig


def flags_t_v_chan(data, chan, freq_range=None):
    """
    Make a waterfall plot of flagged data in channels vs time

    Parameters
    ----------
    data    : array of real, shape(num_times, num_chans, num_pol)
    freq_range : list of start and stop frequencies of the array, optional
    
    """
    npols = data.shape[-1]
    nscans = data.shape[0]
    ncols = npols
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.0 * ncols, 5 * nrows), squeeze=False)
    for p in range(npols):
        im = axes[0, p].imshow(data[..., p], extent=(chan[0],chan[-1],0,nscans), aspect='auto', origin='lower')
        axes[0, p].set_ylabel('Pol {0}  Scans'.format(p))
        axes[0, p].set_xlabel('Channels')
    # Add colorbar
    cax = fig.add_axes([0.92, 0.12, 0.02, 0.75])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('% baselines flagged')

    if freq_range is not None:
        for ax in axes.flatten()[0:2]:
            add_freq_axis(ax,chan_range=[chan[0],chan[-1]],freq_range=freq_range)  
    return fig


def plot_el_v_time(targets, times, elevations, t_zero, title=None):
    """
    Make a plot of elevation vs time for a number of targets

    Parameters
    ----------
    targets : list of names of targets for plot legend
    times    : list of arrays of times for each target
    elevations : list of arrays of elevations for each target
    t_zero : start time of observation for x-label
    title : text for title of the plot, optional
    """
    fig, axes = plt.subplots(1, 1, figsize=(20.0, 4.0))
    if title is not None:
        fig.suptitle(title, y=0.95)

    for idx, target in enumerate(targets):
        axes.plot((times[idx] - t_zero) / 60, np.rad2deg(elevations[idx]), '.', label=target)
    t = time.strftime("%Y %x %X", time.gmtime(t_zero))

    axes.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", frameon=False)
    axes.set_ylabel('Elevation (degrees)')
    axes.set_xlabel('Time (UTC) since {0} (minutes)'.format(t))
    return fig


def plot_corr_uvdist(uvdist, data, freqlist=None, title=None, amp=False):
    """
    Plots Amplitude and Phase vs UVdist

    Parameters
    ----------
    uvdist  : array of real, shape(num_times, num_chans, num_baselines)
    data    : array of complex, shape(num_times,num_chans, num_pol, num_baselines)
    freqlist : list of frequencies for legend, optional
    title : text for title of plot, optional
    """

    npols = data.shape[-2]
    nrows, ncols = npols, 2
    times = data.shape[0]
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.0 * ncols, 4 * nrows),
                             squeeze=False, sharey='col')
    if title is not None:
        fig.suptitle(title, y=0.95)

    for p in range(npols):
        for i in range(times):
            # Transpose the axes to ensure that the color cycles on frequencies not on baseline
            p1 = axes[p, 0].plot(uvdist[i, :, :].transpose(),
                                 np.absolute(data[i, :, p, :]).transpose(), '.')

            if amp:
                axes[p, 1].plot(uvdist[i, :, :].transpose(),
                                 np.absolute(data[i, :, p, :]).transpose(), '.')
            else:
                axes[p, 1].plot(uvdist[i, :, :].transpose(),
                            np.angle(data[i, :, 1, :], deg=True).transpose(), '.')
           
            # Reset color cycle so that channels have the same color
            axes[p, 0].set_prop_cycle(None)
            axes[p, 1].set_prop_cycle(None)
        
        axes[p, 0].set_ylabel('Amplitude' + ' Pol_{0}'.format(p))
        if amp:
            axes[p, 1].set_ylabel('Zoom Amplitude' + ' Pol_{0}'.format(p))
            low_ylim,upper_ylim=amp_range(data)
            axes[p, 1].set_ylim(low_ylim,upper_ylim)
        else:
            axes[p, 1].set_ylabel('Phase' + ' Pol_{0}'.format(p))
        plt.setp(axes[p, 0].get_xticklabels(), visible=False)
        plt.setp(axes[p, 1].get_xticklabels(), visible=False)

    # for the final row, add in xticklabels and xlabel
    l_p = npols - 1
    axes[l_p, 0].set_xlabel('UV distance [wavelength]')
    axes[l_p, 1].set_xlabel('UV distance [wavelength]')
    plt.setp(axes[l_p, 0].get_xticklabels(), visible=True)
    plt.setp(axes[l_p, 1].get_xticklabels(), visible=True)

    if freqlist is not None:
        freqlabel = ['{0} MHz'.format(int(i / 1e6)) for i in freqlist]
        axes[0, 1].legend(p1, freqlabel, bbox_to_anchor=(1.0, 1.0), loc="upper left", frameon=False)
    fig.subplots_adjust(hspace=0.1)
    return fig


def plot_delays(times, data, t_zero=None, antlist=None):
    """Plots delay vs time
       Parameters
       ----------
       times: array of real
           timestamps of delays
       data: array of real
            delays in nanoseconds
       t_zero: start time, optional
       antlist: list of antenna names for legend, optional
    """
    npols = data.shape[-2]
    nrows, ncols = 1, npols
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.0 * ncols, 4.0 * nrows), squeeze=False)

    if t_zero is None:
        t_zero = times[0]
    t = time.strftime("%Y %x %X", time.gmtime(t_zero))

    for p in range(npols):
        p1 = axes[0, p].plot((times - t_zero) / 60, data[:, p, :], marker='.', ls='dotted')
        axes[0, p].set_xlabel('Time since {0} (UTC) [minutes]'.format(t))
        axes[0, p].set_ylabel('Delays Pol {0} [ns]'.format(p))

    if antlist is not None:
        axes[0, 1].legend(p1, antlist, bbox_to_anchor=(1.0, 1.0), loc="upper left", frameon=False)

    return fig


def plot_spec(data, chan, antlist=None, freq_range=None, title=None, amp=False):
    """ Plots spectrum of corrected data
    Parameters
    ----------
    data : :class:`np.ndarray`
        plot data, shape(num_chans, num_pol, num_ant/num_bl)
    chan : : class:`np.ndarray`
        channel numbers for x-axis
    antlist : list of str
        list of antenna/baseline names for plot legend, optional  
    freq_range : list 
        start and stop frequencies of the array, optional
    title : str
        plot title, optional 
    amp : bool
        plot only amplitudes if True, else plot amplitude and phase, optional 
    """
    npols = data.shape[-2]
    nrows, ncols = npols, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.0 * ncols, 4.0 * nrows),
                             squeeze=False, sharey='col')
    if title is not None:
        fig.suptitle(title, y=0.95)

    for p in range(npols):
        #plot full range amplitude plots
        p1 = axes[p, 0].plot(chan, np.absolute(data[..., p, :]), '.')
        axes[p, 0].set_ylabel('Amplitude' + ' Pol_{0}'.format(p))
        plt.setp(axes[p, 0].get_xticklabels(), visible=False)
        if amp:
            #plot limited range amplitude plots
            axes[p,1].plot(chan, np.absolute(data[...,p,:]), '.' )
            axes[p,1].set_ylabel('Zoom Amplitude'+' Pol_{0}'.format(p))
        else:
            #plot phase plots
            axes[p, 1].set_ylabel('Phase' + ' Pol_{0}'.format(p))
            axes[p, 1].plot(chan, np.angle(data[..., p, :], deg=True), '.')
        plt.setp(axes[p, 1].get_xticklabels(), visible=False)

    #set range limit 
    if amp :    
        low_ylim,upper_ylim=amp_range(data)
        axes[p,1].set_ylim(low_ylim,upper_ylim)
    
    # For the last row, add in xticklabels and xlabels
    l_p = npols - 1
    plt.setp(axes[l_p, 0].get_xticklabels(), visible=True)
    plt.setp(axes[l_p, 1].get_xticklabels(), visible=True)
    axes[l_p, 0].set_xlabel('Channels')
    axes[l_p, 1].set_xlabel('Channels')

    if antlist is not None:
        axes[0, 1].legend(p1, antlist, bbox_to_anchor=(1.0, 1.0), loc="upper left", frameon=False)

    # If frequency range supplied, plot a frequency axis for the top row
    if freq_range is not None:
        if len(freq_range) == 2:
            for ax in axes.flatten()[0:2]:
                add_freq_axis(ax,[chan[0],chan[-1]],freq_range)

    fig.subplots_adjust(hspace=0.1)
    return fig


def add_freq_axis(ax, chan_range=[0,4096],freq_range=[900,1400]):
    """ Adds a frequency axis to the top of a given matplotlib Axes
    Parameters
    ----------
    ax : : class: `matplotlib.axes.Axes`
         Axes to add the frequency axis to
    chan_range : list
         start and stop channel numbers  
    freq_range : list
         start and stop frequencies corresponding to the start and stop channel numbers 
     """
    ax_freq = ax.twiny()
    delta_freq=freq_range[1]-freq_range[0]
    delta_chan=chan_range[1]-chan_range[0]
    freq_xlim_0 = ax.get_xlim()[0]*delta_freq/delta_chan+freq_range[0]
    freq_xlim_1 = ax.get_xlim()[1]*delta_freq/delta_chan+freq_range[0]
    ax_freq.set_xlim(freq_xlim_0,freq_xlim_1)
    ax_freq.tick_params(direction='in')
    ax_freq.set_xlabel('Frequency MHz')  

def amp_range(data):
    """ Calculate a limited amplitude range based on the NMAD of the data
    Parameters
    ----------
    data : :class:`np.ndarray`
        plot data, shape(..., num_pol, num_ant/num_bl)
        
    Returns
    -------
    tuple:
       lower limit, upper limit
    """
    npols = data.shape[-2]
    # use 3*NMAD to limit y-range of plots, 
    # the definition used is strictly only correct
    # a gaussian distribution of points 
    low=np.empty(npols)
    upper=np.empty(npols)
    for p in range(npols):
        mag=np.absolute(data[...,p,:][~np.isnan(data[...,p,:])])
        med=np.median(mag)       
        thresh=3*1.4826*np.median(np.abs(mag-med))
        low[p]=med-thresh
        upper[p]=med+thresh
        
    low_lim=min(low)
    low_lim=max(low_lim,0)
    upper_lim=max(upper)
    return low_lim, upper_lim 


def plot_corr_v_time(times, data, plottype='p', antlist=None, t_zero=None, title=None):
    """
    Plots amp/phase versus time

    Parameters
    ----------
    times  : array of real, shape(num_times)
    data   : array of complex, shape(num_times,num_chns, num_pol, num_ants)
    plottype : 'a' to plot amplitude else plot phase, default is phase
    antlist : array if antenna names for plot legend, optional
    t_zero : start time of the observation, optional
    title : text for title of plot, optional
    """
    npols = data.shape[-2]
    nrows, ncols = npols, 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(20.0 * ncols, 3.0 * nrows),
                             squeeze=False, sharey=True)
    if title is not None:
        fig.suptitle(title, y=0.95)

    if t_zero is None:
        t_zero = times[0]
    times = np.array(times) - t_zero
    t = time.strftime("%Y %x %X", time.gmtime(t_zero))

    for p in range(npols):
        data_pol = data[:, :, p, :]
        for chan in range(data_pol.shape[-2]):
            if plottype == 'a':
                p1 = axes[p, 0].plot(times / 60., np.absolute(data_pol[:, chan, :]), '.')
                axes[p, 0].set_ylabel('Amp Pol_{0}'.format(p))
            else:
                p1 = axes[p, 0].plot(times / 60., np.angle(data_pol[:, chan, :], deg=True), '.')
                axes[p, 0].set_ylabel('Phase Pol_{0}'.format(p))

            # Reset the colour cycle, so that all channels have the same plot color
            axes[p, 0].set_prop_cycle(None)
            plt.setp(axes[p, 0].get_xticklabels(), visible=False)

    # For the final row, add in xticklabels and xlabel
    l_p = npols - 1
    plt.setp(axes[l_p, 0].get_xticklabels(), visible=True)
    axes[l_p, 0].set_xlabel('Time since {0} (UTC) [min]'.format(t))

    if antlist is not None:
        axes[0, 0].legend(p1, antlist, bbox_to_anchor=(1.0, 1.0), loc="upper left", frameon=False)

    fig.subplots_adjust(hspace=0.1)
    return fig


def plot_waterfall(visdata, contrast=0.01, flags=None, channel_freqs=None, dump_timestamps=None):
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

    fig = plt.figure(figsize=(8.3, 11.7))
    kwargs = {'aspect': 'auto', 'origin': 'lower', 'interpolation': 'none'}
    # Defaults
    kwargs['extent'] = [-0.5, visdata.shape[1] - 0.5, -0.5, visdata.shape[0] - 0.5]
    plt.xlabel('Channel number')
    plt.ylabel('Dump number')
    # Change defaults if frequencies or times specified.
    if channel_freqs is not None:
        kwargs['extent'][0], kwargs['extent'][1] = channel_freqs[0], channel_freqs[-1]
        # reverse the data if the frequencies are in descending order
        if channel_freqs[1] - channel_freqs[0] < 0:
            visdata = visdata[:, ::-1]
            kwargs['extent'][0] = channel_freqs[-1] / 1e6
            kwargs['extent'][1] = channel_freqs[0] / 1e6
            plt.xlabel('Frequency (MHz)')
    if dump_timestamps is not None:
        kwargs['extent'][2], kwargs['extent'][3] = 0, dump_timestamps[-1] - dump_timestamps[0]
        plt.ylabel('Time (UTC seconds)')
    image = plt.imshow(visdata, **kwargs)
    image.set_cmap('Greys')
    # Make an array of RGBA data for the flags (initialize to alpha=0)
    if flags:
        plotflags = np.zeros(flags.shape[0:2] + (4,))
        plotflags[:, :, 0] = 1.0
        plotflags[:, :, 3] = flags[:, :, 0]
        plt.imshow(plotflags, **kwargs)

    ampsort = np.sort(visdata, axis=None)
    arrayremove = int(len(ampsort) * contrast)
    lowcut, highcut = ampsort[arrayremove], ampsort[-(arrayremove + 1)]
    image.norm.vmin = lowcut
    image.norm.vmax = highcut
    plt.show()
    plt.close(fig)


def plot_RFI_mask(pltobj, extra=None, channelwidth=1e6):
    """
    Plot the frequencies of know rfi satellites on a spectrum

    Parameters
    ----------
    plotobj       : the matplotlib plot object to plot the RFI onto
    extra         : the locations of extra masks to plot
    channelwidth  : the width of the mask per channel
    """

    pltobj.axvspan(1674e6, 1677e6, alpha=0.3, color='grey')  # Meteosat
    pltobj.axvspan(1667e6, 1667e6, alpha=0.3, color='grey')  # Fengun
    pltobj.axvspan(1682e6, 1682e6, alpha=0.3, color='grey')  # Meteosat
    pltobj.axvspan(1685e6, 1687e6, alpha=0.3, color='grey')  # Meteosat
    pltobj.axvspan(1687e6, 1687e6, alpha=0.3, color='grey')  # Fengun
    pltobj.axvspan(1690e6, 1690e6, alpha=0.3, color='grey')  # Meteosat
    pltobj.axvspan(1699e6, 1699e6, alpha=0.3, color='grey')  # Meteosat
    pltobj.axvspan(1702e6, 1702e6, alpha=0.3, color='grey')  # Fengyun
    pltobj.axvspan(1705e6, 1706e6, alpha=0.3, color='grey')  # Meteosat
    pltobj.axvspan(1709e6, 1709e6, alpha=0.3, color='grey')  # Fengun
    pltobj.axvspan(1501e6, 1570e6, alpha=0.3, color='blue')  # Inmarsat
    pltobj.axvspan(1496e6, 1585e6, alpha=0.3, color='blue')  # Inmarsat
    pltobj.axvspan(1574e6, 1576e6, alpha=0.3, color='blue')  # Inmarsat
    pltobj.axvspan(1509e6, 1572e6, alpha=0.3, color='blue')  # Inmarsat
    pltobj.axvspan(1574e6, 1575e6, alpha=0.3, color='blue')  # Inmarsat
    pltobj.axvspan(1512e6, 1570e6, alpha=0.3, color='blue')  # Thuraya
    pltobj.axvspan(1450e6, 1498e6, alpha=0.3, color='red')  # Afristar
    pltobj.axvspan(1652e6, 1694e6, alpha=0.2, color='red')  # Afristar
    pltobj.axvspan(1542e6, 1543e6, alpha=0.3, color='cyan')  # Express AM1
    pltobj.axvspan(1554e6, 1554e6, alpha=0.3, color='cyan')  # Express AM 44
    pltobj.axvspan(1190e6, 1215e6, alpha=0.3, color='green')  # Galileo
    pltobj.axvspan(1260e6, 1300e6, alpha=0.3, color='green')  # Galileo
    pltobj.axvspan(1559e6, 1591e6, alpha=0.3, color='green')  # Galileo
    pltobj.axvspan(1544e6, 1545e6, alpha=0.3, color='green')  # Galileo
    pltobj.axvspan(1190e6, 1217e6, alpha=0.3, color='green')  # Beidou
    pltobj.axvspan(1258e6, 1278e6, alpha=0.3, color='green')  # Beidou
    pltobj.axvspan(1559e6, 1563e6, alpha=0.3, color='green')  # Beidou
    pltobj.axvspan(1555e6, 1596e6, alpha=0.3, color='green')  # GPS L1  1555 -> 1596
    pltobj.axvspan(1207e6, 1238e6, alpha=0.3, color='green')  # GPS L2  1207 -> 1188
    pltobj.axvspan(1378e6, 1384e6, alpha=0.3, color='green')  # GPS L3
    pltobj.axvspan(1588e6, 1615e6, alpha=0.3, color='green')  # GLONASS  1588 -> 1615 L1
    pltobj.axvspan(1232e6, 1259e6, alpha=0.3, color='green')  # GLONASS  1232 -> 1259 L2
    pltobj.axvspan(1616e6, 1630e6, alpha=0.3, color='grey')  # IRIDIUM
    if extra is not None:
        for i in xrange(extra.shape[0]):
            pltobj.axvspan(extra[i] - channelwidth / 2, extra[i] + channelwidth / 2,
                           alpha=0.7, color='Maroon')
