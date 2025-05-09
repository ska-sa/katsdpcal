import datetime

import numpy as np
import matplotlib.dates as md

from cycler import cycler
# use Agg backend for when the pipeline is run without an X1 connection
from matplotlib import use
use('Agg')

import matplotlib.pylab as plt     # noqa: E402

# Figure sizes
FIG_X = 10
FIG_Y = 4

# figure colors to cycle through in plots
colors = plt.cm.tab20.colors
plt.rc('axes', prop_cycle=(cycler('color', colors)))


def plot_v_antenna(data, ylabel='', title=None, antenna_names=None, pol=[0, 1],
                   **plot_kwargs):
    """Plots a value vs antenna.

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
    plot_kwargs : keyword arguments, optional
        additional keyword arguments for plotting function
    """
    npols = data.shape[-2]
    nants = data.shape[-1]
    fig, axes = plt.subplots(1, figsize=(2 * FIG_X, FIG_Y / 2.0))

    for p in range(npols):
        axes.plot(data[p], '.', label=pol[p], **plot_kwargs)

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


def plot_g_solns_legend(times, data, antenna_names=None, pol=[0, 1], **plot_kwargs):
    """Plots gain solutions.

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
    plot_kwargs : keyword arguments, optional
        additional keyword arguments for plotting function
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
        p1 = axes[p, 0].plot(dates, np.abs(data[:, p, :]), '.-', **plot_kwargs)
        axes[p, 0].set_ylabel('Amplitude Pol_{0}'.format(pol[p]))

        # plot phase
        axes[p, 1].plot(dates, np.angle(data[:, p, :], deg=True), '.-', **plot_kwargs)
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


def flags_bl_v_chan(data, chan, uvlist, freq_range=None, pol=[0, 1], **plot_kwargs):
    """Waterfall plot of flagged data in Channels vs Baselines.

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
    plot_kwargs : keyword arguments, optional
        additional keyword arguments for plotting function
    """
    npols = data.shape[-2]
    nbls = data.shape[-1]
    ncols = npols
    nrows = 1
    # scale the size of the plot by the number of bls, but have a minimum size
    rowsize = max(1, nbls / 1000.0)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * FIG_X, rowsize * FIG_Y),
                             squeeze=False, sharey='row', **plot_kwargs)
    for p in range(npols):
        im = axes[0, p].imshow(data[:, p, :].transpose(), extent=(
            chan[0], chan[-1], 0, nbls), aspect='auto', origin='lower',
            cmap=plt.cm.jet, **plot_kwargs)
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
    """Creates ticklabels for the baseline axis of a plot.

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


def flags_t_v_chan(data, chan, targets, freq_range=None, pol=[0, 1], **plot_kwargs):
    """Waterfall plot of flagged data in channels vs time.

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
    plot_kwargs : keyword arguments, optional
        additional keyword arguments for plotting function
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
            chan[0], chan[-1], 0, nscans), aspect='auto', origin='lower',
            cmap=plt.cm.jet, **plot_kwargs)
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
    """Format the ticklabels for time axis of a plot.

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
    """Format the x-axis labels for time axis of a plot.

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


def plot_el_v_time(targets, times, elevations, title=None, **plot_kwargs):
    """Plot of elevation vs time for a number of targets.

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
    plot_kwargs : keyword arguments, optional
        additional keyword arguments for plotting function
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
        axes.plot(dates, np.rad2deg(elevations[idx]), '.', label=target,
                  **plot_kwargs)

    axes.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", frameon=False)
    axes.set_ylabel('Elevation (degrees)')
    time_xtick_fmt(np.array([axes]), [t_zero, t_max])
    time_label(axes, [t_zero, t_max])

    return fig


def plot_corr_uvdist(uvdist, data, freqlist=None, title=None, amp=False,
                     pol=[0, 1], phase_range=[-180, 180], units=None, **plot_kwargs):
    """Plots Amplitude and Phase vs UVdist.

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
    units : str, optional
        amplitude units for y-axis label
    plot_kwargs : keyword arguments, optional
        additional keyword arguments for plotting function
    """
    npols = data.shape[-2]
    nrows, ncols = npols, 2
    times = data.shape[0]
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * FIG_X, nrows * FIG_Y),
                             squeeze=False, sharey='col')
    if title is not None:
        fig.suptitle(title, y=0.95)

    if units is not None:
        unit_str = '({})'.format(units)
    else:
        unit_str = ''

    for p in range(npols):
        for i in range(times):
            # Transpose the axes to ensure that the color cycles on frequencies not on baseline
            p1 = axes[p, 0].plot(uvdist[i, :, :].transpose(),
                                 np.absolute(data[i, :, p, :]).transpose(), '.', ms='3',
                                 **plot_kwargs)

            if amp:
                axes[p, 1].plot(uvdist[i, :, :].transpose(),
                                np.absolute(data[i, :, p, :]).transpose(), '.', ms='3',
                                **plot_kwargs)
            else:
                axes[p, 1].plot(uvdist[i, :, :].transpose(),
                                np.angle(data[i, :, p, :], deg=True).transpose(), '.', ms='3',
                                **plot_kwargs)

            # Reset color cycle so that channels have the same color
            axes[p, 0].set_prop_cycle(None)
            axes[p, 1].set_prop_cycle(None)

        axes[p, 0].set_ylabel('Amplitude Pol_{0} {1}'.format(pol[p], unit_str))
        if amp:
            axes[p, 1].set_ylabel('Zoom Amplitude Pol_{0} {1}'.format(pol[p], unit_str))
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


def plot_delays(times, data, antenna_names=None, pol=[0, 1], **plot_kwargs):
    """Plots delay vs time.

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
    plot_kwargs : keyword arguments, optional
        additional keyword arguments for plotting function
    """
    npols = data.shape[-2]
    nrows, ncols = 1, npols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * FIG_X, nrows * FIG_Y), squeeze=False)

    datetimes = [datetime.datetime.utcfromtimestamp(unix_timestamp) for unix_timestamp in times]
    dates = md.date2num(datetimes)
    time_xtick_fmt(axes, [datetimes[0], datetimes[-1]])

    for p in range(npols):
        p1 = axes[0, p].plot(dates, data[:, p, :], marker='.', ls='dotted', **plot_kwargs)
        axes[0, p].set_ylabel('Delays Pol {0} [ns]'.format(pol[p]))
        time_label(axes[0, p], [datetimes[0], datetimes[-1]])

    if antenna_names is not None:
        axes[0, npols-1].legend(p1, antenna_names, bbox_to_anchor=(1.0, 1.0),
                                loc="upper left", frameon=False)

    return fig


def plot_phaseonly_spec(data, chan, antenna_names=None, freq_range=None, title=None,
                        pol=[0, 1], phase_range=[-180, 180], **plot_kwargs):
    """Plots phase-only spectrum of corrected data.

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
    plot_kwargs : keyword arguments, optional
        additional keyword arguments for plotting function
    """
    npols = data.shape[-2]
    nrows, ncols = 1, npols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * FIG_X, nrows * FIG_Y),
                             squeeze=False, sharey='row')
    if title is not None:
        fig.suptitle(title)

    for p in range(npols):
        # plot full range amplitude plots
        p1 = axes[0, p].plot(chan, np.angle(data[..., p, :], deg=True), '.', ms=1,
                             **plot_kwargs)
        axes[0, p].set_ylim(phase_range[0], phase_range[-1])
        axes[0, p].set_ylabel('Phase Pol_{0}'.format(pol[p]))
        axes[0, p].set_xlabel('Channels')

    if antenna_names is not None:
        axes[0, npols-1].legend(p1, antenna_names, bbox_to_anchor=(1.0, 1.0),
                                loc="upper left", frameon=False, markerscale=5)

    # If frequency range supplied, plot a frequency axis for the top row
    if freq_range is not None:
        for ax in axes.flatten()[0:2]:
            add_freq_axis(ax, [chan[0], chan[-1]], freq_range)

    fig.subplots_adjust(hspace=0.1)
    return fig


def plot_spec(data, chan, antenna_names=None, freq_range=None, title=None, amp=False,
              pol=[0, 1], phase_range=[-180, 180], amp_model=None, units=None, **plot_kwargs):
    """Plots spectrum of corrected data.

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
    amp_model : :class:`np.ndarray`, optional
        real, (num_chans) amplitude model to plot
    units : str, optional
        amplitude units for y-axis label
    plot_kwargs : keyword arguments, optional
        additional keyword arguments for plotting function
    """
    npols = data.shape[-2]
    nrows, ncols = npols, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * FIG_X, nrows * FIG_Y),
                             squeeze=False, sharey='col')
    if title is not None:
        fig.suptitle(title, y=0.95)

    if units is not None:
        unit_str = '({})'.format(units)
    else:
        unit_str = ''

    for p in range(npols):
        # plot full range amplitude plots
        p1 = axes[p, 0].plot(chan, np.absolute(data[..., p, :]), '.', ms=1, **plot_kwargs)
        axes[p, 0].set_ylabel('Amplitude Pol_{0} {1}'.format(pol[p], unit_str))
        plt.setp(axes[p, 0].get_xticklabels(), visible=False)
        if amp:
            # plot limited range amplitude plots
            axes[p, 1].plot(chan, np.absolute(data[..., p, :]), '.', ms=1, **plot_kwargs)
            axes[p, 1].set_ylabel('Zoom Amplitude Pol_{0} {1}'.format(pol[p], unit_str))
        else:
            # plot phase plots
            axes[p, 1].set_ylabel('Phase Pol_{0}'.format(pol[p]))
            axes[p, 1].plot(chan, np.angle(data[..., p, :], deg=True), '.', ms=1,
                            **plot_kwargs)
            axes[p, 1].set_ylim(phase_range[0], phase_range[-1])
        plt.setp(axes[p, 1].get_xticklabels(), visible=False)

    # plot model if provided
    if amp_model is not None:
        for p in range(npols):
            p1 += axes[p, 0].plot(chan, amp_model, '--', color='k', alpha=0.6,  **plot_kwargs)
            if amp:
                axes[p, 1].plot(chan, amp_model, '--', color='k', alpha=0.6, **plot_kwargs)
        antenna_names += ['model']

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


def plot_phase_stability_check(phase_nmad, correlator_freq, title=None,  pol=[0, 1], y_scale=0.5,
                               **plot_kwargs):
    """Plot the Normalised Median Absolute Deviation of the Corrected
       Data Phases vs Frequency.

    These plots will show that if the antenna phase-up  was successful,
    the NMAD phase values will be similar accross frequency channels (RFI Free Regions)
    and conversely if phase up was not successful, the NMAD of the phases will show higher
    deviations across frequency channels. We plot the NMAD Phases for each polarisation
    this further provides information on the variation of phases for different pols.

    Parameters:
    ----------
    phase_nmad : :class: `np.ndarray`
                 real, shape (freq, pol)
    correlator_freq : :class: `np.ndarray`
                       real, shape (freq, )
    pol : list
          list of polarisation desciptions
    """

    npols = len(pol)
    nrows, ncols = npols, 1
    fig, axes = plt.subplots(nrows=npols, ncols=ncols, figsize=(nrows * FIG_X, nrows * FIG_Y),
                             sharex=True)
    axes = axes.flatten()

    if title is not None:
        fig.suptitle(title, y=1)

    for idx, p in enumerate(range(npols)):

        axes[idx].plot(correlator_freq, phase_nmad[:, p], marker='.', ls='',
                       ms='2', label=f'Average Phase NMAD :{np.nanmean(phase_nmad[:, p]): .3f}',
                       **plot_kwargs)
        axes[idx].set_ylabel('Phase NMAD_{0}'.format(pol[p]))
        axes[idx].set_xlabel('Frequency (Mhz)')
        axes[idx].legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", frameon=False)
        axes[idx].grid(color='grey', which='both', lw=0.1)
        axes[idx].set_ylim((0, np.nanstd(phase_nmad[:, p])*5))

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.2)

    return fig


def add_freq_axis(ax, chan_range, freq_range):
    """Adds a frequency axis to the top of a given matplotlib Axes.

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
    """Calculate a limited amplitude range based on the NMAD of the data.

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
                     pol=[0, 1], phase_range=[-180, 180], units=None, **plot_kwargs):
    """Plots amp/phase versus time.

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
    units : str, optional
        amplitude units for y-axis label
    plot_kwargs : keyword arguments, optional
        additional keyword arguments for plotting function
    """
    npols = data.shape[-2]
    nrows, ncols = npols, 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols * FIG_X, nrows * FIG_Y),
                             squeeze=False, sharey=True)
    if title is not None:
        fig.suptitle(title, y=0.95)

    if units is not None:
        unit_str = '({})'.format(units)
    else:
        unit_str = ''

    datetimes = [datetime.datetime.utcfromtimestamp(unix_timestamp) for unix_timestamp in times]
    dates = md.date2num(datetimes)
    time_xtick_fmt(axes, [datetimes[0], datetimes[-1]])

    for p in range(npols):
        data_pol = data[:, :, p, :]
        for chan in range(data_pol.shape[-2]):
            if plottype == 'a':
                p1 = axes[p, 0].plot(dates, np.absolute(data_pol[:, chan, :]), '.', **plot_kwargs)
                axes[p, 0].set_ylabel('Amp Pol_{0} {1}'.format(pol[p], unit_str))
            else:
                p1 = axes[p, 0].plot(dates, np.angle(data_pol[:, chan, :], deg=True), '.',
                                     **plot_kwargs)
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


def plot_snr(drawfunc, times, snr, labels=None, title=None, pol=[0, 1], yscale=1, **kwargs):
    """Plot properties of a list of solution SNRs as function of time.

    Parameters
    ----------
    drawfunc : func
        function which plots to the figure axes
    times : list of lists
        times for each solution SNR
    snr : list of :class:`np.ndarray`
        SNRs for each solution, real (ntimes, npols, nants)
    labels : list of str, optional
        labels for each solution
    title : str
        title of plot
    pol : list, optional
        list of polarisation descriptions
    yscale : float
        scale the default ysize of the plot by yscale
    **kwargs : additional keyword arguments passed to `drawfunc`
    """
    npols = snr[0].shape[-2]
    nrows, ncols = 2, 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * FIG_X, nrows * yscale * FIG_Y),
                             squeeze=False, sharey='col', sharex='col')
    if title is not None:
        fig.suptitle(title, y=0.95)

    # get matplotlib dates and format time axis
    # pad the end of the time axis slightly to allow
    # us to separate the solutions in time
    max_time = max([max(t) for t in times]) + 16
    min_time = min([min(t) for t in times])
    min_datetime = datetime.datetime.utcfromtimestamp(min_time)
    max_datetime = datetime.datetime.utcfromtimestamp(max_time)
    time_xtick_fmt(axes, [min_datetime, max_datetime])

    # use labels if provided
    if labels is not None:
        labellist = labels
    else:
        labellist = [''] * len(times)

    for p in range(npols):
        # need distinct colors and symbols to easily distinguish different solutions
        axes[p, 0].set_prop_cycle(color=['blue', 'purple', 'coral'],
                                  marker=['*', 'x', '.'])

        inc_time = 0
        for t, s, l in zip(times, snr, labellist):
            datetimes = [datetime.datetime.utcfromtimestamp(unix_timestamp + inc_time)
                         for unix_timestamp in t]
            # separate the different solutions times by a few seconds to improve overplotting
            inc_time += 8
            dates = md.date2num(datetimes)
            drawfunc(dates, s[:, p, :], axes[p, 0], pol_label=pol[p], label=l, **kwargs)
            plt.setp(axes[p, 0].get_xticklabels(), visible=False)
        axes[p, 0].yaxis.grid(True, linestyle='-', which='major',
                              color='lightgrey', alpha=0.5)

    # for the final row add in the time labels
    l_p = npols - 1
    plt.setp(axes[l_p, 0].get_xticklabels(), visible=True)
    time_label(axes[l_p, 0], [min_datetime, max_datetime])
    # include legend
    if labels is not None:
        axes[0, 0].legend(bbox_to_anchor=(1.0, 1.0), loc="upper left", frameon=False)

    return fig


def draw_errorplot(times, data, ax, pol_label=0, **kwargs):
    """Calculate and plot the median and IQR of data vs time.

    Parameters
    ----------
    times : list
        times
    data : :class:`np.ndarray`
        real, shape (num_times, num_pols, num_ants)
    ax : : class: `matplotlib.axes.Axes`
        axes on which to plot
    pol_label : label for the pol axis
    **kwargs : additional keyword arguments passed to 'ax.errorbar'
    """
    p25, median, p75 = np.nanpercentile(data, (25, 50, 75), axis=-1)
    low_error = median - p25
    up_error = p75 - median

    error = np.array([low_error, up_error])
    ax.errorbar(times, median, yerr=error, linestyle='None', **kwargs)
    ax.set_ylabel('SNR Pol {0}'.format(pol_label))


def draw_below_thresh(times, snr, ax, pol_label=0, snrthresh=10, **kwargs):
    """Plots number of antennas with snr = NaN or < snrthresh as function of time.

    Parameters
    ----------
    times : list
        timestamps
    snr : :class: `np.ndarray`
        real, shape (num_times, num_pols, num_ants)
    ax : :class: `matplotlib.axes.Axes`
        axes on which to plot
    pol_label : str/float, optional
        polarisation label
    snrthresh : float
        threshold below which to count antennas
    **kwargs : additional keyword arguments passed to `ax.plot`
    """
    snr_nonans = np.where(np.isnan(snr), 0, snr)
    n_low = np.sum(snr_nonans < snrthresh, axis=-1)
    ax.plot(times, n_low, linestyle='', **kwargs)
    ax.set_ylabel('No of ants, Pol {0}'.format(pol_label))
