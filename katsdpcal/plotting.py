import datetime

import numpy as np
import matplotlib.dates as md

from cycler import cycler
# use Agg backend for when the pipeline is run without an X1 connection
from matplotlib import use
use('Agg', warn=False)

import matplotlib.pylab as plt     # noqa: E402

# Figure sizes
FIG_X = 10
FIG_Y = 4

# figure colors to cycle through in plots
colors = plt.cm.tab20.colors
plt.rc('axes', prop_cycle=(cycler('color', colors)))


def plot_v_antenna(data, ylabel='', title=None, antenna_names=None, pol=[0, 1]):
    """
    Plots a value vs antenna

    Parameters
    ----------
    data : :class:`np.ndarray`
        real, shape(num_pols,num_ants)
    ylabel : str, optional
        label for y-axis
    title : str, optional
        title for plot
    antenna_names: list of str
        antenna names for legend, optional
    pol : list
        list of polarisation descriptions, optional
    """
    npols = data.shape[-2]
    nants = data.shape[-1]
    fig, axes = plt.subplots(1, figsize=(2 * FIG_X, FIG_Y / 2.0))

    for p in range(npols):
        axes.plot(data[p], '.', label=pol[p])

    axes.set_xticks(np.arange(0, nants))
    if antenna_names is not None:
        # right justify the antenna_names for better alignment of labels
        labels = [a.strip().rjust(12) for a in antenna_names]
        axes.set_xticklabels(labels, rotation='vertical')

    axes.set_xlabel('Antennas')
    axes.set_ylabel(ylabel)
    axes.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", frameon=False)
    if title is not None:
        fig.suptitle(title, y=1.0)
    return fig


def plot_g_solns_legend(times, data, antenna_names=None, pol=[0, 1]):
    """
    Plots gain solutions

    Parameters
    ----------
    times : :class:`np.ndarray`
        real, shape(num_times)
    data : :class:`np.ndarray`
        complex, shape(num_times,num_pols,num_ants)
    antenna_names: list of str
        antenna names for legend, optional
    pol : list
        list of polarisation descriptions, optional
    """
    npols = data.shape[-2]
    nrows, ncols = npols, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * FIG_X, nrows * FIG_Y),
                             squeeze=False, sharey='col')
    # get matplotlib dates and format time axis
    datetimes = [datetime.datetime.utcfromtimestamp(unix_timestamp) for unix_timestamp in times]
    dates = md.date2num(datetimes)
    time_xtick_fmt(axes, [datetimes[0], datetimes[-1]])

    for p in range(npols):
        # plot amplitude
        p1 = axes[p, 0].plot(dates, np.abs(data[:, p, :]), '.-')
        axes[p, 0].set_ylabel('Amplitude Pol_{0}'.format(pol[p]))

        # plot phase
        axes[p, 1].plot(dates, np.angle(data[:, p, :], deg=True), '.-')
        axes[p, 1].set_ylabel('Phase Pol_{0}'.format(pol[p]))

        plt.setp(axes[p, 0].get_xticklabels(), visible=False)
        plt.setp(axes[p, 1].get_xticklabels(), visible=False)

    # for the last row, add in xticklabels and xlabels
    l_p = npols - 1
    for i in range(ncols):
        plt.setp(axes[l_p, i].get_xticklabels(), visible=True)
        time_label(axes[l_p, i], [datetimes[0], datetimes[-1]])

    if antenna_names is not None:
        axes[0, 1].legend(p1, antenna_names, bbox_to_anchor=(1.0, 1.0),
                          loc="upper left", frameon=False)
    return fig


def flags_bl_v_chan(data, chan, uvlist, freq_range=None, pol=[0, 1]):
    """
    Make a waterfall plot of flagged data in Channels vs Baselines

    Parameters
    ----------
    data : :class:`np.ndarray`
        real, shape(num_chans, num_pol, num_baselines)
    chan : :class:`np.ndarray`
        real, shape(num_chans), index numbers of the chan axis.
    uvdist : :class:`np.ndarray`
        real, shape(num_bls), UVdist of each baseline
    freq_range : list
        list of start and stop frequencies of the array, optional
    pol : list
        list of polarisation descriptions, optional
    """
    npols = data.shape[-2]
    nbls = data.shape[-1]
    ncols = npols
    nrows = 1
    # scale the size of the plot by the number of bls, but have a minimum size
    rowsize = max(1, nbls / 1000.0)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * FIG_X, rowsize * FIG_Y),
                             squeeze=False, sharey='row')
    for p in range(npols):
        im = axes[0, p].imshow(data[:, p, :].transpose(), extent=(
            chan[0], chan[-1], 0, nbls), aspect='auto', origin='lower', cmap=plt.cm.jet)
        axes[0, p].set_ylabel('Pol {0} Antenna separation [m]'.format(pol[p]))
        axes[0, p].set_xlabel('Channels')
        bl_labels(axes[0, p], uvlist)
    plt.setp(axes[0, 1].get_yticklabels(), visible=False)

    # Add colorbar
    cax = fig.add_axes([0.92, 0.12, 0.02, 0.75])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('% time flagged')

    if freq_range is not None:
        for ax in axes.flatten()[0:2]:
            add_freq_axis(ax, chan_range=[chan[0], chan[-1]], freq_range=freq_range)
    return fig


def bl_labels(ax, seplist):
    """
    Creates ticklabels for the baseline axis of a plot

    Parameters
    ----------
    ax : : class: `matplotlib.axes.Axes`
        axes to add ticklabels to
    seplist : :class:`np.ndarray`
        real (n_bls) of labels corresponding to baseline positions in ax
    """
    yticks = ax.get_yticks()
    # select only the ticks with valid separations
    valid_yticks = [int(y) for y in yticks if y >= 0 and y < len(seplist)]
    # set the yticks to only appear at places with a valid separation
    ax.set_yticks(valid_yticks)
    # set the labels of the yticks to be the separations
    ax.set_yticklabels(np.int_(seplist[valid_yticks]))


def flags_t_v_chan(data, chan, targets, freq_range=None, pol=[0, 1]):
    """
    Make a waterfall plot of flagged data in channels vs time

    Parameters
    ----------
    data : :class:`np.ndarray`
        complex, shape(num_times, num_chans, num_pol)
    chan : :class:`np.ndarray`
        real, shape(num_chans), index number of chan axis
    targets : list of str
        target names/labels for targets in each scan
    freq_range : list
        start and stop frequencies of the array, optional
    pol : list
        list of polarisation descriptions, optional
    """
    npols = data.shape[-1]
    nscans = data.shape[0]
    ncols = npols
    nrows = 1
    # scale the size of the plot by the number of scans but have a min and max plot size
    rowsize = min(max(1.0, data.shape[0] / 50.0), 10.0)

    fig, axes = plt.subplots(nrows, ncols, figsize=(
        ncols * FIG_X, rowsize * FIG_Y), squeeze=False, sharey='row')
    for p in range(npols):
        im = axes[0, p].imshow(data[..., p], extent=(
            chan[0], chan[-1], 0, nscans), aspect='auto', origin='lower', cmap=plt.cm.jet)
        axes[0, p].set_ylabel('Pol {0}  Scans'.format(pol[p]))
        axes[0, p].set_xlabel('Channels')
    plt.setp(axes[0, 1].get_yticklabels(), visible=False)

    # major tick step
    step = nscans // 25 + 1
    axes[0, 0].set_yticks(np.arange(0, len(targets))[::step]+0.5)
    axes[0, 0].set_yticks(np.arange(0, len(targets))+0.5, minor=True)
    axes[0, 0].set_yticklabels(targets[::step])

    # Add colorbar
    cax = fig.add_axes([0.92, 0.12, 0.02, 0.75])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label('% baselines flagged')

    if freq_range is not None:
        for ax in axes.flatten()[0: 2]:
            add_freq_axis(ax, chan_range=[chan[0], chan[-1]], freq_range=freq_range)
    return fig


def time_xtick_fmt(ax, timerange):
    """
    Format the ticklabels for time axis of a plot

    Parameters
    ----------
    ax : : class: `np.ndarray` of : class: `matplotlib.axes.Axes`
        array of axes whose ticklabels will be formatted
    timerange : list of :class: `datetime.datetime`
        start and stop times of the plot
    """
    # Format the xticklabels to display h:m:s
    xfmt = md.DateFormatter('%H:%M:%S')
    ax_flat = ax.flatten()
    for a in ax_flat:
        # set axis range for plots of 1 point, nb or it will fail
        if timerange[0] == timerange[-1]:
            low = md.date2num(timerange[0] - datetime.timedelta(seconds=10))
            high = md.date2num(timerange[-1] + datetime.timedelta(seconds=10))
        else:
            plotrange = md.date2num(timerange[-1]) - md.date2num(timerange[0])
            low = md.date2num(timerange[0]) - 0.05*plotrange
            high = md.date2num(timerange[-1]) + 0.05*plotrange
        a.set_xlim(low, high)
        a.xaxis.set_major_formatter(xfmt)


def time_label(ax, timerange):
    """
    Format the x-axis labels for time axis of a plot

    Parameters
    ----------
    ax : : class: `matplotlib.axes.Axes`
        axes to add the xaxis labels to
    timerange : list of :class: `datetime.datetime`
        start and stop times of the plot
    """

    if timerange[0].date() == timerange[-1].date():
        datelabel = timerange[0].strftime('%Y-%m-%d')
    else:
        datelabel = timerange[0].strftime('%Y-%m-%d') + ' -- ' + timerange[-1].strftime('%Y-%m-%d')
    # Display the date in the label
    ax.set_xlabel('Times (UTC) \n Date: ' + datelabel)


def plot_el_v_time(targets, times, elevations, title=None):
    """
    Make a plot of elevation vs time for a number of targets

    Parameters
    ----------
    targets : list of str
        names of targets for plot legend
    times : list of :class:`np.ndarray`
        real, times for each target
    elevations : list of :class:`np.ndarray`
        real, elevations for each target
    title : str, optional
        title of the plot
    """
    fig, axes = plt.subplots(1, 1, figsize=(2 * FIG_X, FIG_Y))
    if title is not None:
        fig.suptitle(title, y=0.95)

    t_zero = min([np.min(t) for t in times])
    t_max = max([np.max(t) for t in times])
    t_zero = datetime.datetime.utcfromtimestamp(t_zero)
    t_max = datetime.datetime.utcfromtimestamp(t_max)

    for idx, target in enumerate(targets):
        datetimes = [datetime.datetime.utcfromtimestamp(unix_timestamp)
                     for unix_timestamp in times[idx]]
        dates = md.date2num(datetimes)
        axes.plot(dates, np.rad2deg(elevations[idx]), '.', label=target)

    axes.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", frameon=False)
    axes.set_ylabel('Elevation (degrees)')
    time_xtick_fmt(np.array([axes]), [t_zero, t_max])
    time_label(axes, [t_zero, t_max])

    return fig


def plot_corr_uvdist(uvdist, data, freqlist=None, title=None, amp=False,
                     pol=[0, 1], phase_range=[-180, 180]):
    """
    Plots Amplitude and Phase vs UVdist

    Parameters
    ----------
    uvdist : :class:`np.ndarray`
        real, shape(num_baselines)
    data : :class:`np.ndarray`
        complex, shape(num_times,num_chans, num_pol, num_baselines)
    freqlist : list, optional
        frequencies for legend
    title : str, optional
        title of plot
    pol : list, optional
        list of polarisation descriptions
    phase_range : list, optional
        start and stop phase ranges to plot, optional
    """

    npols = data.shape[-2]
    nrows, ncols = npols, 2
    times = data.shape[0]
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * FIG_X, nrows * FIG_Y),
                             squeeze=False, sharey='col')
    if title is not None:
        fig.suptitle(title, y=0.95)

    for p in range(npols):
        for i in range(times):
            # Transpose the axes to ensure that the color cycles on frequencies not on baseline
            p1 = axes[p, 0].plot(uvdist[i, :, :].transpose(),
                                 np.absolute(data[i, :, p, :]).transpose(), '.', ms='3')

            if amp:
                axes[p, 1].plot(uvdist[i, :, :].transpose(),
                                np.absolute(data[i, :, p, :]).transpose(), '.', ms='3')
            else:
                axes[p, 1].plot(uvdist[i, :, :].transpose(),
                                np.angle(data[i, :, p, :], deg=True).transpose(), '.', ms='3')

            # Reset color cycle so that channels have the same color
            axes[p, 0].set_prop_cycle(None)
            axes[p, 1].set_prop_cycle(None)

        axes[p, 0].set_ylabel('Amplitude Pol_{0}'.format(pol[p]))
        if amp:
            axes[p, 1].set_ylabel('Zoom Amplitude Pol_{0}'.format(pol[p]))
            lim = amp_range(data)
            if not np.isnan(lim).any():
                axes[p, 1].set_ylim(*lim)
        else:
            axes[p, 1].set_ylabel('Phase Pol_{0}'.format(pol[p]))
            axes[p, 1].set_ylim(phase_range[0], phase_range[-1])
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
        axes[0, 1].legend(p1, freqlabel, bbox_to_anchor=(1.0, 1.0),
                          loc="upper left", frameon=False, markerscale=2)
    fig.subplots_adjust(hspace=0.1)
    return fig


def plot_delays(times, data, antenna_names=None, pol=[0, 1]):
    """
    Plots delay vs time

    Parameters
    ----------
    times : :class:`np.ndarray`
        real, timestamps of delays
    data : :class:`np.ndarray`
        real, delays in nanoseconds
    antenna_names : list of str
        antenna names for legend, optional
    pol : list
        list of polarisation descriptions, optional
    """
    npols = data.shape[-2]
    nrows, ncols = 1, npols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * FIG_X, nrows * FIG_Y), squeeze=False)

    datetimes = [datetime.datetime.utcfromtimestamp(unix_timestamp) for unix_timestamp in times]
    dates = md.date2num(datetimes)
    time_xtick_fmt(axes, [datetimes[0], datetimes[-1]])

    for p in range(npols):
        p1 = axes[0, p].plot(dates, data[:, p, :], marker='.', ls='dotted')
        axes[0, p].set_ylabel('Delays Pol {0} [ns]'.format(pol[p]))
        time_label(axes[0, p], [datetimes[0], datetimes[-1]])

    if antenna_names is not None:
        axes[0, npols-1].legend(p1, antenna_names, bbox_to_anchor=(1.0, 1.0),
                                loc="upper left", frameon=False)

    return fig


def plot_phaseonly_spec(data, chan, antenna_names=None, freq_range=None, title=None,
                        pol=[0, 1], phase_range=[-180, 180]):
    """ Plots spectrum of corrected data

    Parameters
    ----------
    data : :class:`np.ndarray`
        complex, shape(num_chans, num_pol, num_ant/num_bl)
    chan : :class:`np.ndarray`
        real, (nchan) channel numbers for x-axis
    antenna_names : list of str
        list of antenna/baseline names for plot legend, optional
    freq_range : list
        start and stop frequencies of the array, optional
    title : str, optional
        plot title
    amp : bool, optional
        plot only amplitudes if True, else plot amplitude and phase
    pol : list, optional
        list of polarisation descriptions
    phase_range : list, optional
        start and stop phase ranges to plot, optional
    """
    npols = data.shape[-2]
    nrows, ncols = 1, npols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * FIG_X, nrows * FIG_Y),
                             squeeze=False, sharey='row')
    if title is not None:
        fig.suptitle(title)

    for p in range(npols):
        # plot full range amplitude plots
        p1 = axes[0, p].plot(chan, np.angle(data[..., p, :], deg=True), '.', ms=1)
        axes[0, p].set_ylim(phase_range[0], phase_range[-1])
        axes[0, p].set_ylabel('Phase Pol_{0}'.format(pol[p]))
        axes[0, p].set_xlabel('Channels')

    if antenna_names is not None:
        axes[0, 1].legend(p1, antenna_names, bbox_to_anchor=(1.0, 1.0),
                          loc="upper left", frameon=False, markerscale=5)

    # If frequency range supplied, plot a frequency axis for the top row
    if freq_range is not None:
        for ax in axes.flatten()[0:2]:
            add_freq_axis(ax, [chan[0], chan[-1]], freq_range)

    fig.subplots_adjust(hspace=0.1)
    return fig


def plot_spec(data, chan, antenna_names=None, freq_range=None, title=None, amp=False,
              pol=[0, 1], phase_range=[-180, 180]):
    """ Plots spectrum of corrected data

    Parameters
    ----------
    data : :class:`np.ndarray`
        complex, shape(num_chans, num_pol, num_ant/num_bl)
    chan : : class:`np.ndarray`
        real, (nchan) channel numbers for x-axis
    antenna_names : list of str
        list of antenna/baseline names for plot legend, optional
    freq_range : list
        start and stop frequencies of the array, optional
    title : str, optional
        plot title
    amp : bool, optional
        plot only amplitudes if True, else plot amplitude and phase
    pol : list, optional
        list of polarisation descriptions
    phase_range : list, optional
        start and stop phase ranges to plot, optional
    """
    npols = data.shape[-2]
    nrows, ncols = npols, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * FIG_X, nrows * FIG_Y),
                             squeeze=False, sharey='col')
    if title is not None:
        fig.suptitle(title, y=0.95)

    for p in range(npols):
        # plot full range amplitude plots
        p1 = axes[p, 0].plot(chan, np.absolute(data[..., p, :]), '.', ms=1)
        axes[p, 0].set_ylabel('Amplitude Pol_{0}'.format(pol[p]))
        plt.setp(axes[p, 0].get_xticklabels(), visible=False)
        if amp:
            # plot limited range amplitude plots
            axes[p, 1].plot(chan, np.absolute(data[..., p, :]), '.', ms=1)
            axes[p, 1].set_ylabel('Zoom Amplitude Pol_{0}'.format(pol[p]))
        else:
            # plot phase plots
            axes[p, 1].set_ylabel('Phase Pol_{0}'.format(pol[p]))
            axes[p, 1].plot(chan, np.angle(data[..., p, :], deg=True), '.', ms=1)
            axes[p, 1].set_ylim(phase_range[0], phase_range[-1])
        plt.setp(axes[p, 1].get_xticklabels(), visible=False)

    # set range limit
    if amp:
        lim = amp_range(data)
        if not np.isnan(lim).any():
                axes[p, 1].set_ylim(*lim)

    # For the last row, add in xticklabels and xlabels
    l_p = npols - 1
    plt.setp(axes[l_p, 0].get_xticklabels(), visible=True)
    plt.setp(axes[l_p, 1].get_xticklabels(), visible=True)
    axes[l_p, 0].set_xlabel('Channels')
    axes[l_p, 1].set_xlabel('Channels')

    if antenna_names is not None:
        axes[0, 1].legend(p1, antenna_names, bbox_to_anchor=(1.0, 1.0),
                          loc="upper left", frameon=False, markerscale=5)

    # If frequency range supplied, plot a frequency axis for the top row
    if freq_range is not None:
        for ax in axes.flatten()[0:2]:
            add_freq_axis(ax, [chan[0], chan[-1]], freq_range)

    fig.subplots_adjust(hspace=0.1)
    return fig


def add_freq_axis(ax, chan_range, freq_range):
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
    delta_freq = freq_range[1] - freq_range[0]
    delta_chan = chan_range[1] - chan_range[0]
    freq_xlim_0 = ax.get_xlim()[0] * delta_freq / delta_chan + freq_range[0]
    freq_xlim_1 = ax.get_xlim()[1] * delta_freq / delta_chan + freq_range[0]
    ax_freq.set_xlim(freq_xlim_0, freq_xlim_1)
    ax_freq.set_xlabel('Frequency MHz')


def amp_range(data):
    """
    Calculate a limited amplitude range based on the NMAD of the data

    Parameters
    ----------
    data : :class:`np.ndarray`
        complex, shape(..., num_pol, num_ant/num_bl)

    Returns
    -------
    lower limit : float
        lower limit for plot
    upper limit : float
        upper limit for plot
    """
    npols = data.shape[-2]
    # use 3*NMAD to limit y-range of plots,
    # the definition used is strictly only correct
    # a gaussian distribution of points
    low = np.empty(npols)
    upper = np.empty(npols)
    for p in range(npols):
        mag = np.absolute(data[..., p, :][~np.isnan(data[..., p, :])])
        med = np.median(mag)
        thresh = 3 * 1.4826 * np.median(np.abs(mag - med))
        low[p] = med - thresh
        upper[p] = med + thresh

    low_lim = min(low)
    low_lim = max(low_lim, 0)
    upper_lim = max(upper)
    return low_lim, upper_lim


def plot_corr_v_time(times, data, plottype='p', antenna_names=None, title=None,
                     pol=[0, 1], phase_range=[-180, 180]):
    """
    Plots amp/phase versus time

    Parameters
    ----------
    times : :class:`np.ndarray`
        real, shape(num_times)
    data : class:`np.ndarray`
        complex, shape(num_times,num_chns, num_pol, num_ants)
    plottype : str
        'a' to plot amplitude else plot phase, default is phase
    antenna_names : :class:`np.ndarray`, optional
        antenna names for plot legend
    title : str, optional
        title of plot
    pol : list, optional
        list of polarisation descriptions
    phase_range : list, optional
        start and stop phase ranges to plot, optional
    """
    npols = data.shape[-2]
    nrows, ncols = npols, 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols * FIG_X, nrows * FIG_Y),
                             squeeze=False, sharey=True)
    if title is not None:
        fig.suptitle(title, y=0.95)

    datetimes = [datetime.datetime.utcfromtimestamp(unix_timestamp) for unix_timestamp in times]
    dates = md.date2num(datetimes)
    time_xtick_fmt(axes, [datetimes[0], datetimes[-1]])

    for p in range(npols):
        data_pol = data[:, :, p, :]
        for chan in range(data_pol.shape[-2]):
            if plottype == 'a':
                p1 = axes[p, 0].plot(dates, np.absolute(data_pol[:, chan, :]), '.')
                axes[p, 0].set_ylabel('Amp Pol_{0}'.format(pol[p]))
            else:
                p1 = axes[p, 0].plot(dates, np.angle(data_pol[:, chan, :], deg=True), '.')
                axes[p, 0].set_ylabel('Phase Pol_{0}'.format(pol[p]))
                axes[p, 0].set_ylim(phase_range[0], phase_range[-1])
            # Reset the colour cycle, so that all channels have the same plot color
            axes[p, 0].set_prop_cycle(None)
            plt.setp(axes[p, 0].get_xticklabels(), visible=False)

    # For the final row, add in xticklabels and xlabel
    l_p = npols - 1
    time_label(axes[l_p, 0], [datetimes[0], datetimes[-1]])
    plt.setp(axes[l_p, 0].get_xticklabels(), visible=True)

    if antenna_names is not None:
        axes[0, 0].legend(p1, antenna_names, bbox_to_anchor=(1.0, 1.0),
                          loc="upper left", frameon=False)

    fig.subplots_adjust(hspace=0.1)
    return fig
