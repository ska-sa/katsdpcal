import os
import logging
import datetime

from . import plotting
from . import calprocs
from . import calprocs_dask
import numpy as np
import dask.array as da

from docutils.core import publish_file

import matplotlib.pylab as plt
import katpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# No of antennas per plot
ANT_CHUNKS = 16
# No of channels for plots which aren't spectra
PLOT_CHANNELS = 8
# Tag blacklist
TAG_WHITELIST = ['gaincal', 'bfcal', 'delaycal', 'polcal', 'bpcal', 'target']

# --------------------------------------------------------------------------------------------------
# --- CLASS :  rstReport
# --------------------------------------------------------------------------------------------------


class rstReport(file):
    """
    RST style report
    """

    def write_heading(self, heading, symbol):
        heading_len = len(heading)
        self.writeln(symbol * heading_len)
        self.writeln(heading)
        self.writeln(symbol * heading_len)
        self.write('\n')

    def write_heading_0(self, heading):
        self.write_heading(heading, '#')

    def write_heading_1(self, heading):
        self.write_heading(heading, '*')

    def write_heading_2(self, heading):
        self.write_heading(heading, '=')

    def write_heading_3(self, heading):
        self.write_heading(heading, '+')

    def writeln(self, line=None):
        if line is not None:
            self.write(line)
        self.write('\n')


# --------------------------------------------------------------------------------------------------
# --- FUNCTION :  Report writing functions
# --------------------------------------------------------------------------------------------------
def utc_tstr(timestamp, day=False):
    """
    Returns a formatted UTC time string

    Parameters
    ----------
    timestamp : float
         unix_timestamp
    day : bool, optional
         if true don't include the year and
         month in the time string
    Returns
    -------
    str : formatted time string
    """
    time = datetime.datetime.utcfromtimestamp(timestamp)
    time_format = "%Y-%m-%d %H:%M:%S"
    if day:
        time_format = "%d %H:%M:%S"
    time_string = time.strftime(time_format)
    return time_string


def insert_fig(report_path, report, fig, name=None):
    """
    Insert matplotlib figure into report

    Parameters
    ----------
    report : file-like
        report file to write to
    fig : matplotlib figure
    """
    if name is None:
        name = str(fig)
    figname = "{}.png".format(name)
    fig.savefig(os.path.join(report_path, figname), bbox_inches='tight')
    # closing the plot is necessary to relase the memory
    #  (this is a pylab issue)
    plt.close()

    fig_text = \
        '''.. image:: {}
       :align: center
    '''.format(figname,)
    report.writeln()
    report.writeln(fig_text)
    report.writeln()


def write_bullet_if_present(report, table, var_text, var_name, transform=None):
    """
    Write bullet point, if `var_name` is present in `table`

    Parameters
    ----------
    report : file-like
        report file to write to
    table : dict-like
        a dict-like interface (e.g. :class:`~katsdptelstate.TelescopeState`)
    var_text : str
        bullet point description
    var_name : str
        key to look up in `table`
    transform : callable, optional
        transform for applying to value before reporting
    """
    value = table.get(var_name, 'unknown')
    if transform is not None:
        value = transform(value)
    report.writeln('* {0}:  {1}'.format(var_text, value))


def write_summary(report, ts, stream_name, parameters, st=None, et=None):
    """
    Write observation summary information to report

    Parameters
    ----------
    report : file-like
        report file to write to
    ts : :class:`katsdptelstate.TelescopeState`
        telescope state
    stream_name : str
        name of the L0 data stream
    parameters : dict
        Pipeline parameters
    st : float, optional
        start time for reporting parameters, seconds
    et : float, optional
        end time for reporting parameters, seconds
    """
    # write RST style bulletted list

    report.writeln('* Start time:  ' + utc_tstr(st))

    # telescope state values
    telstate_l0 = ts.view(stream_name)
    write_bullet_if_present(report, telstate_l0, 'Int time', 'int_time')
    write_bullet_if_present(report, parameters, 'Channels', 'channel_freqs', transform=len)
    write_bullet_if_present(report, parameters, 'Antennas', 'antenna_names', transform=len)
    write_bullet_if_present(report, parameters, 'Antenna list', 'antenna_names',
                            transform=', '.join)
    report.writeln()

    report.writeln('Source list:')
    report.writeln()
    try:
        target_list = ts.get_range('info_sources', st=0, return_format='recarray')['value']
    except KeyError:
        # key not present
        report.writeln('* Unknown')
    else:
        for target in target_list:
            report.writeln('* {0:s}'.format(target))

    report.writeln()


def write_table_timerow(report, colnames, times, data):
    """
    Write RST style table to report, rows: time, columns: antenna

    Parameters
    ----------
    report : file-like
        report file to write to
    colnames : list of str
        list of column names
    times : list
        list of times (equates to number of rows in the table)
    data
        table data, shape (time, columns)
    """
    # create table header
    header = colnames[:]
    header.insert(0, 'time')

    n_entries = len(header)
    col_width = 30
    col_header = '=' * col_width + ' '

    # write table header
    report.writeln()
    report.writeln(col_header * n_entries)
    report.writeln(" ".join([h.ljust(col_width) for h in header]))
    report.writeln(col_header * n_entries)

    timestrings = [utc_tstr(t, True) for t in times]

    # add each time row to the table
    for t, d in zip(timestrings, data):
        data_string = " ".join(["{:.3f}".format(di.real,).ljust(col_width)
                                for di in np.atleast_1d(d)])
        report.write("{}".format(t,).ljust(col_width + 1))
        report.writeln(data_string)

    # table footer
    report.writeln(col_header * n_entries)
    report.writeln()


def write_table_timecol(report, antenna_names, times, data):
    """
    Write RST style table to report, rows: antenna, columns: time

    Parameters
    ----------
    report : file-like
        report file to write to
    antenna_names : list
        list of antenna names
    times : list
        list of times (equates to number of columns in the table)
    data
        table data, shape (time, antenna)
    """
    n_entries = len(times) + 1
    col_width = 30
    col_header = '=' * col_width + ' '

    # create table header
    timestrings = [utc_tstr(t, day=True) for t in times]
    header = " ".join(["{}".format(t,).ljust(col_width) for t in timestrings])
    header = 'Ant'.ljust(col_width + 1) + header

    # write table header
    report.writeln()
    report.writeln(col_header * n_entries)
    report.writeln(header)
    report.writeln(col_header * n_entries)

    # add each antenna row to the table
    for a, d in zip(antenna_names, data.T):
        data_string = " ".join(["{:.3f}".format(di.real,).ljust(col_width) for di in d])
        report.write(a.ljust(col_width + 1))
        report.writeln(data_string)

    # table footer
    report.writeln(col_header * n_entries)
    report.writeln()


def write_elevation(report, report_path, targets, refant, av_corr):
    """
    Put the elevation vs time plot in the report

    Parameters
    ----------
    report : file-like
        report file to write to
    report_path : str
        path where report is written
    targets : list
        list of unique targets
    refant : :class:`katpoint.Antenna`
        reference antenna
    av_corr : dict
        dictionary containing averaged, corrected data
    """
    ts, el, names = [], [], []
    for cal in targets:
        ts_cal = [ti for ti, t in zip(av_corr['timestamps'],
                                      av_corr['targets']) if t == cal]
        ts_flat = np.array([x for y in ts_cal for x in y])
        el_cal = calc_elevation(refant, ts_flat, cal)

        ts.append(ts_flat)
        names.append(katpoint.Target(cal).name)
        el.append(el_cal)

    plot_title = 'Elevation vs Time for Reference Antenna: {0}'.format(refant.name)
    plot = plotting.plot_el_v_time(
        names, ts, el, title=plot_title)
    insert_fig(report_path, report, plot, name='El_v_time')


def write_flag_summary(report, report_path, av_corr, dist, correlator_freq, pol=[0, 1]):
    """
    Write the RFI summary

    Parameters
    ----------
    report : file-like
        report file to write to
    report_path : str
        path where report is written
    av_corr : dict
        dictionary containing averaged, corrected data
    dist : :class:`np.ndarray`
        real (nbls), separations between antennas for baselines in av_corr
    correlator_freq : :class:`np.ndarray`
        real (nchan) of correlator channel frequencies
    pol : list
        description of polarisation axes, optional
    """
    report.writeln('Percentage of time data is flagged')
    # Flags per scan weighted by length of scan
    n_times = np.array([len(_) for _ in av_corr['timestamps']], dtype=np.float32)
    tw_flags = av_corr['n_flags'] / n_times[:, np.newaxis, np.newaxis, np.newaxis]
    tw_flags_ave = 100 * np.mean(tw_flags, axis=0)

    # Sort baseline by antenna separation
    idx_sort = np.argsort(dist)
    tw_flags_sort = tw_flags_ave[:, :, idx_sort]

    # Get channel index in correlator channels
    n_av_chan = tw_flags.shape[-3]
    idx_chan, freq_chan = get_freq_info(correlator_freq, n_av_chan)
    freq_range = [freq_chan[0], freq_chan[-1]]

    plot = plotting.flags_bl_v_chan(tw_flags_sort, idx_chan, dist[idx_sort], freq_range, pol=pol)
    insert_fig(report_path, report, plot, name='Flags_bl_v_chan')

    report.writeln('Percentage of baselines flagged per scan')
    # Average % of baselines flagged per scan
    bl_flags = 100 * np.mean(tw_flags, axis=3)
    target_names = [katpoint.Target(_).name for _ in av_corr['targets']]
    plot = plotting.flags_t_v_chan(bl_flags, idx_chan, target_names, freq_range, pol=pol)
    insert_fig(report_path, report, plot, name='Flags_s_v_chan')
    report.writeln()


def write_ng_freq(report, report_path, targets, av_corr, ant_idx,
                  refant_index, antenna_names, correlator_freq, pol=[0, 1]):
    """
    Include plots of spectra of calibrators which do
    not have gains applied by the pipeline. Make one plot per calibrator scan.
    The plots will only show baselines to reference antenna.

    Parameters
    ----------
    report : file-like
        report file to write to
    report_path : str
        path where report is written
    targets : list of str
        list of target strings for the targets to plot
    av_corr : dict
        dictionary of averaged corrected data
    ant_idx : :class:`np.ndarray` of int
        int, (n_ants) indices of all baselines to the reference antenna
    refant_index : int
        index of reference antenna
    antenna_names : list
        list of antenna names
    correlator_freq : :class:`np.ndarray`
        real (nchan), correlator channel frequencies
    pol : list
        description of polarisation axes, optional
    """
    if len(targets) > 0:
        report.write_heading_2(
            'Corrected Amp and Phase vs Frequency, delay and bandpass calibrators ')
        report.writeln()
        report.write_heading_3(
            'Baselines to the reference antenna : {0}'.format(antenna_names[refant_index]))

    for cal in targets:
        kat_target = katpoint.Target(cal)
        target_name = kat_target.name
        tags = [t for t in kat_target.tags if t in TAG_WHITELIST]

        # Get baselines to the reference antenna for cal
        av_data, av_flags, av_weights, av_times = select_data(av_corr, [cal])
        logger.info(' Corrected data for {0} shape: {1}'.format(target_name, av_data.shape))
        ant_data = av_data[..., ant_idx]
        # Get channel index in correlator channels
        n_av_chan = ant_data.shape[-3]
        idx_chan, freq_chan = get_freq_info(correlator_freq, n_av_chan)
        freq_range = [freq_chan[0], freq_chan[-1]]

        for ti in range(len(av_times)):
            report.writeln()
            t = utc_tstr(av_times[ti])
            report.writeln('Time : {0}'.format(t))
            plot_title = 'Calibrator: {0} , tags are: {1}'.format(target_name, ', '.join(tags))
            # Only plot 16 antennas per plot
            for idx in range(0, ant_data.shape[-1], ANT_CHUNKS):
                plot = plotting.plot_spec(
                    ant_data[ti, ..., idx : idx + ANT_CHUNKS], idx_chan,
                    antenna_names=antenna_names[idx : idx + ANT_CHUNKS],
                    freq_range=freq_range, title=plot_title, pol=pol)
                insert_fig(report_path, report, plot, name='Corr_v_Freq_{0}_ti_{1}_{2}'.format(
                    target_name.replace(' ', '_'), ti, idx))
                report.writeln()


def write_g_freq(report, report_path, targets, av_corr, antenna_names,
                 cal_bls_lookup, correlator_freq, is_calibrator=True, pol=[0, 1]):
    """
    Include plots of spectra of calibrators which have gains applied
    by the pipeline. Average all scans on each target into a single plot.

    Parameters
    ----------
    report : file-like
        report file to write to
    report_path : str
        path where report is written
    targets : list of str
        list of target strings for targets to plot
    av_corr : dict
        dictionary of averaged corrected data
    antenna_names : list
        list of antenna names
    cal_bls_lookup : :class:`np.ndarray`
        int (nbls x 2), of antenna indices in each baseline
    correlator_freq : :class:`np.ndarray`
        real (ncha), correlator channel frequencies
    is_calibrator: bool, optional
        make plots of amp and phase and label them as calibrator plots if true,
        else plot only amplitudes and label them as target plots
    pol : list
        description of polarisation axes, optional
    """
    if is_calibrator:
        suffix = (' and Phase', 'all gain-calibrated calibrators')
    else:
        suffix = ('', 'all target fields')

    if len(targets) > 0:
        report.write_heading_2('Corrected Amp{0} vs Frequency, {1}'.format(suffix[0], suffix[1]))
        report.writeln()
        report.write_heading_3('All baselines, averaged per antenna')

    # For calibrators with gains applied by the pipeline
    # average per antenna and across scans before plotting.
    for cal in targets:
        kat_target = katpoint.Target(cal)
        target_name = kat_target.name
        tags = [t for t in kat_target.tags if t in TAG_WHITELIST]

        # Average per antenna and across all scans for target cal
        av_data, av_flags, av_weights, av_times = select_data(av_corr, [cal])
        logger.info(' Corrected data for {0} shape: {1}'.format(target_name, av_data.shape))
        av_data, av_flags, av_weights = calprocs.wavg_ant(av_data, av_flags, av_weights,
                                                          ant_array=antenna_names,
                                                          bls_lookup=cal_bls_lookup)
        av_data = calprocs_dask.wavg(da.asarray(av_data),
                                     da.asarray(av_flags),
                                     da.asarray(av_weights))

        # Get channel index in correlator channels
        n_av_chan = av_data.shape[-3]
        idx_chan, freq_chan = get_freq_info(correlator_freq, n_av_chan)
        freq_range = [freq_chan[0], freq_chan[-1]]

        # Set the plot label
        if is_calibrator:
            plot_title = 'Calibrator: {0} , tags are {1}'.format(target_name, ', '.join(tags))
            amp = False
        else:
            plot_title = 'Target: {0}'.format(target_name)
            amp = True
        # Only plot a maximum of 16 antennas per plot
        for idx in range(0, av_data.shape[-1], ANT_CHUNKS):
            data = av_data[..., idx : idx + ANT_CHUNKS].compute()
            plot = plotting.plot_spec(
                data, idx_chan, antenna_names[idx : idx + ANT_CHUNKS],
                freq_range, plot_title, amp=amp, pol=pol)

            insert_fig(report_path, report, plot,
                       name='Corr_v_Freq_{0}_{1}'.format(target_name.replace(" ", "_"), idx))
            report.writeln()


def write_g_time(report, report_path, targets, av_corr, antenna_names, cal_bls_lookup, pol):
    """
    Include plots of amp and phase versus time of all scans of the given targets in report.
    The plots show data averaged per antenna. The data is averaged in frequency to the number
    of channels given by PLOT_CHANNELS

    Parameters
    ----------
    report : file-like
        report file to write to
    report_path : str
        path where report is written
    targets : list of str
        list of target strings for targets to plot
    av_corr : dict
        dictionary of averaged corrected data
    antenna_names : list
        list of antenna names
    cal_bls_lookup : :class:`np.ndarray`
        array of antenna indices in each baseline
    pol : list
        description of polarisation axes, optional
    """
    # Select all scans of calibrators which have gains applied by the pipeline.
    av_data, av_flags, av_weights, av_times = select_data(av_corr, targets)

    if len(av_data) > 0:
        report.write_heading_2(
            'Corrected Phase vs Time, all gain-calibrated calibrators')
        report.writeln()
        report.write_heading_3('All baselines, averaged per antenna')
        report.writeln()

        # Average per antenna
        av_data, av_flags, av_weights = calprocs.wavg_ant(
            av_data, av_flags, av_weights,
            ant_array=antenna_names,
            bls_lookup=cal_bls_lookup)

        # average bandpass into a maximum number of chunks given by PLOT_CHANNELS
        nchan = av_data.shape[-3]
        if nchan >= PLOT_CHANNELS:
            chanav = nchan // PLOT_CHANNELS
            av_data, av_flags, av_weights = calprocs.wavg_full_f(
                av_data, av_flags, av_weights, chanav)

        # insert plots of phase v time
        for idx in range(0, av_data.shape[-1], ANT_CHUNKS):
            plot = plotting.plot_corr_v_time(
                av_times, av_data[..., idx : idx + ANT_CHUNKS],
                antenna_names=antenna_names[idx : idx + ANT_CHUNKS], pol=pol)
            insert_fig(report_path, report, plot, name='Phase_v_Time_{0}'.format(idx))
            report.writeln()

        report.write_heading_2(
            'Corrected Amp vs Time, all gain-calibrated calibrators')
        report.writeln()
        report.write_heading_3('All baselines, averaged per antenna')
        report.writeln()

        # insert plots of amp v time
        for idx in range(0, av_data.shape[-1], ANT_CHUNKS):
            plot = plotting.plot_corr_v_time(av_times,
                                             av_data[..., idx : idx + ANT_CHUNKS], plottype='a',
                                             antenna_names=antenna_names[idx : idx + ANT_CHUNKS],
                                             pol=pol)

            insert_fig(report_path, report, plot, name='Amp_v_Time_{0}'.format(idx))
            report.writeln()


def write_g_uv(report, report_path, targets, av_corr, cal_bls_lookup,
               antennas, cal_array_position, correlator_freq,
               is_calibrator=True, pol=[0, 1]):
    """
    Include plots of amp and phase/amp versus uvdist in report.
    The data is averaged in frequency to to the number
    of channels given by PLOT_CHANNELS

    Parameters
    ----------
    report : file-like
        report file to write to
    report_path : str
        path where report is written
    targets : list of str
        list of target strings for targets to plot
    av_corr : dict
        dictionary of averaged corrected data from which to
        select target data
    cal_bls_lookup : :class:`np.ndarray`
        array of antenna indices in each baseline
    antennas : list of :class:`katpointer.Antennas`
        list of antennas
    cal_array_position : :class:`katpoint.Antenna`
        description string of array position
    correlator_freq : :class:`np.ndarray`
        real (nchan) correlator channel frequencies
    is_calibrator : bool, optional
        make plots of amp and phase and label them as calibrator plots if true,
        else plot only amplitudes and label them as target plots
    pol : list
        description of polarisation axes, optional
    """
    if is_calibrator:
        suffix = (' and Phase', 'all gain-calibrated calibrators')
    else:
        suffix = ('', 'all target fields')

    if len(targets) > 0:
        report.write_heading_2(
            'Amp{0} vs UVdist, {1}'.format(suffix[0], suffix[1]))
        report.write_heading_3('All baselines')

    # Plot the Amp and Phase vs UV distance for all calibrators
    # with gains applied by the pipeline.
    for cal in targets:
        kat_target = katpoint.Target(cal)
        target_name = kat_target.name
        tags = [t for t in kat_target.tags if t in TAG_WHITELIST]
        # Get data for target cal
        av_data, av_flags, av_weights, av_times = select_data(av_corr, [cal])
        # average bandpass into a maximum of 8 chunks
        nchan = av_corr['vis'].shape[-3]
        if nchan >= PLOT_CHANNELS:
            chanav = nchan // PLOT_CHANNELS
            nchan = PLOT_CHANNELS
            av_data, av_flags, av_weights = calprocs.wavg_full_f(av_data,
                                                                 av_flags,
                                                                 av_weights,
                                                                 chanav)
        else:
            chanav = nchan

        # Get channel index in correlator channels
        idx_chan, freq_chan = get_freq_info(correlator_freq, nchan)
        freq_chan = freq_chan * 1e6
        uvdist = calc_uvdist(cal, freq_chan, av_times,
                             cal_bls_lookup, antennas, cal_array_position)

        if is_calibrator:
            plot_title = 'Calibrator {0}, tags are {1}'.format(target_name, ', '.join(tags))
            amp = False
        else:
            plot_title = 'Target {0}'.format(target_name)
            amp = True
        plot = plotting.plot_corr_uvdist(uvdist, av_data, freq_chan,
                                         plot_title, amp=amp, pol=pol)
        insert_fig(report_path, report, plot,
                   name='Corr_v_UVdist_{0}'.format(target_name.replace(" ", "_")))
        report.writeln()


def write_products(report, report_path, ts, parameters,
                   st, et, antenna_names, correlator_freq, pol=[0, 1]):
    """
    Include calibration product plots in the report

    Parameters
    ----------
    report : file-like
        report file to write to
    report_path : str
        path where report is written
    ts : :class:`katsdptelstate.TelescopeState`
        telescope state
    parameters : dict
        pipeline parameters
    st : float
        start time for reporting parameters, seconds
    et : float
        end time for reporting parameters, seconds
    antenna_names : list
        list of antenna names
    correlator_freq : :class:`np.ndarray`
        real (nchans), correlator channel frequencies

    """

    cal_list = ['K', 'KCROSS', 'B', 'G']
    product_names = parameters['product_names']
    solns_exist = any([product_names[cal] in ts for cal in cal_list])
    if not solns_exist:
        logger.info(' - no calibration solutions')

    # delay
    cal = 'K'
    vals, times = get_cal(ts, cal, product_names[cal], st, et)
    if len(times) > 0:
        cal_heading(report, cal, 'Delay', '([ns])')
        write_K(report, report_path, times, vals, antenna_names, pol)

    # ---------------------------------
    # cross pol delay
    cal = 'KCROSS'
    vals, times = get_cal(ts, cal, product_names[cal], st, et)
    if len(times) > 0:
        cal_heading(report, cal, 'Cross polarisation delay', '([ns])')
        # convert delays to nano seconds
        vals = 1e9 * vals
        write_table_timerow(report, [cal], times, vals)

    # ---------------------------------
    # bandpass
    cal = 'B'
    vals, times = get_cal(ts, cal, product_names[cal], st, et)
    if len(times) > 0:
        cal_heading(report, cal, 'Bandpass')
        write_B(report, report_path, times, vals, antenna_names, correlator_freq, pol)

    # ---------------------------------
    # gain
    cal = 'G'
    vals, times = get_cal(ts, cal, product_names[cal], st, et)
    if len(times) > 0:
        cal_heading(report, cal, 'Gain')
        for idx in range(0, vals.shape[-1], ANT_CHUNKS):
            plot = plotting.plot_g_solns_legend(
                times, vals[..., idx : idx + ANT_CHUNKS],
                antenna_names=antenna_names[idx : idx + ANT_CHUNKS], pol=pol)
            insert_fig(report_path, report, plot, name='{0}'.format(cal))


def get_cal(ts, cal, ts_name, st, et):
    """
    Fetch a calibration product from telstate

    Parameters:
    -----------
    ts : :class:`katsdptelstate.TelescopeState`
        telescope state
    cal : str
        string indicating calibration product type
    ts_name : str
        name of the telescope state key holding the cal solution
    st : float
        start time for reporting parameters, seconds
    et : float
        end time for reporting parameters, seconds
    Returns:
    --------
    vals : :class:`np.ndarray`
        values of calibration product
    times : :class:`np.ndarray`
        times of calibration product
    """
    vals, times = [], []
    if ts_name in ts:
        product = ts.get_range(ts_name, st=0, return_format='recarray')
        if len(product['time']) > 0:
            logger.info('Calibration product: {0}'.format(cal,))
            vals = product['value']
            # K shape is n_time, n_pol, n_ant
            times = product['time']
            logger.info('  shape: {0}'.format(vals.shape,))
    return vals, times


def write_K(report, report_path, times, vals, antenna_names, pol=[0, 1]):
    """
    Include table of delays and delay plots in cal report

    Parameters
    ----------
    report : file-like
        report file to write to
    report_path : str
        path where report is written
    times : list
        list of times for delay solutions
    vals : :class:`np.ndarray`
        delay solutions
    antenna_names : list
        list of antenna names
    pol : list
        description of polarisation axes, optional
    """
    # convert delays to nano seconds
    vals = 1e9 * vals
    # iterate through polarisation
    for p in range(vals.shape[-2]):
        report.writeln('**POL {0}**'.format(p,))
        kpol = vals[:, p, :]
        logger.info('  pol{0} shape: {1}'.format(p, kpol.shape))
        write_table_timecol(report, antenna_names, times, kpol)

    for idx in range(0, vals.shape[-1], ANT_CHUNKS):
        plot = plotting.plot_delays(times, vals, antenna_names=antenna_names, pol=pol)
        insert_fig(report_path, report, plot, name='K')


def write_B(report, report_path, times, vals, antenna_names, correlator_freq, pol=[0, 1]):
    """
    Include plots of bandpass solutions at all given times in report

    Parameters
    ----------
    report : file-like
        report file to write to
    report_path : str
        path where report is written
    times : list
        list of times for delay solutions
    vals : array
        bandpass solutions
    antenna_names : list
        list of antenna names
    chanidx : float
        number of channels solutions
    pol : list
        description of polarisation axes, optional
    """
    # B shape is n_time, n_chan, n_pol, n_ant
    freq_range = [correlator_freq[0], correlator_freq[-1]]
    chan_no = np.arange(0, len(correlator_freq))

    for ti in range(len(times)):
        t = utc_tstr(times[ti])
        report.writeln('Time: {}'.format(t,))

        for idx in range(0, vals[ti].shape[-1], ANT_CHUNKS):
            plot = plotting.plot_spec(
                vals[ti, ..., idx : idx + ANT_CHUNKS], chan_no,
                antenna_names=antenna_names[idx : idx + ANT_CHUNKS],
                freq_range=freq_range, pol=pol)
            insert_fig(report_path, report, plot,
                       name='B_ti_{0}_{1}'.format(ti, idx))


def cal_heading(report, cal, prefix, suffix=''):
    """
    Write cal pipeline product headings

    Parameters
    ----------
    report : file-like
        report file to write to
    cal : str
        calibration product
    prefix : str
        description of calibration product
    suffix : str, optional
        units of calibration product
    """
    report.write_heading_2('Calibration product {0}'.format(cal))
    report.writeln('{0} calibration solutions {1}'.format(prefix, suffix))
    report.writeln()


def get_freq_info(correlator_freq, nchan):
    """
    Given nchan averaged channels calculate the channel index (in correlator channels)
    and the frequencies of the averaged channels.

    Parameters
    -----------
    correlator_freq : :class:`np.ndarray`
        array of correlator channel frequencies
    nchan : int
        no of averaged channels
    Returns
    -------
    avchan : :class:`np.ndarray`
        real (nchan) of mean channel indices in correlator channels
    avfreq : :class:`np.ndarray`
        real (nchan) of mean frequencies of averaged channels
    """

    nc_chan = correlator_freq.shape[0]
    chanav = nc_chan // nchan

    index = np.arange(0, nc_chan, chanav)
    # get average chan_no
    avchan = np.add.reduceat(np.arange(0, nc_chan), index, dtype=np.float32)
    avchan /= np.add.reduceat(np.ones(nc_chan), index)
    # get average freq
    avfreq = np.add.reduceat(correlator_freq, index)
    avfreq /= np.add.reduceat(np.ones(nc_chan), index)

    return avchan, avfreq


def select_data(av_corr, t_list):
    """ Select only data corresponding to targets in t_list
    Parameters
    ----------
    av_corr : dict
        dictionary of calibrated data
    t_list : list of str
        list of target strings of targets to select
    Returns
    -------
    data : :class:`np.ndarray`
        complex, visibilities for targets in t_list
    flags : :class:`np.ndarray`
        boolean, flags for targets in t_list
    weights : :class:`np.ndarray`
        real, weights for targets in t_list
    times : :class:`np.ndarray` of times
        real, times for targets in t_list
    """
    idx = [t in t_list for t in av_corr['targets']]
    data = av_corr['vis'][idx]
    flags = av_corr['flags'][idx]
    weights = av_corr['weights'][idx]
    times = av_corr['times'][idx]
    if isinstance(times, np.float64):
        times = np.array([times])

    return data, flags, weights, times


def split_targets(targets):
    """
    Separate targets into three lists containing
    calibrators without gains applied, calibrators
    with gains applied and targets

    Parameters
    -----------
    targets : list
        list of unique targets
    Returns
    -------
    nogain : list
        list of calibrators without gains applied by the pipeline
    gain : list
        list of calibrators with gains applied by the pipeline
    target : list
        list of targets
    """
    nogain, gain, target = [], [], []
    for cal in targets:
        kat_target = katpoint.Target(cal)
        tags = kat_target.tags
        # tags which have gains applied by pipeline
        gaintaglist = ('gaincal', 'bfcal', 'target')
        if not any(x in gaintaglist for x in tags):
            nogain.append(cal)
        elif 'target' in tags:
            target.append(cal)
        else:
            gain.append(cal)
    return nogain, gain, target


def calc_elevation(refant, times, target):
    """
      Calculates elevation versus timestamps for observation targets.
      It calculates the elevation from the target and antenna
      position and it does not reflect the actual pointing of the dish.

      Parameters
      ----------
      refant : str
          the reference antenna
      times : array
          timestamps of scan
      target : str
          target string

      Returns
      -------
      times : :class:`np.ndarray`
          real, (ntimes) timestamps
      elevation : :class:`np.ndarray`
          real, (ntimes) of elevations
      """
    kat_target = katpoint.Target(target)
    elevations = kat_target.azel(times, refant)[1]

    return elevations


def calc_uvdist(target, freq, times, cal_bls_lookup, antennas, cal_array_position):
    """
    Calculate uvdistance in wavelengths

    Parameters
    ----------
    ts : :class:`katsdptelstate.TelescopeState`
        telescope state
    target : str
         target string
    frequencies : :class:`np.ndarray`
         real, (nchan) frequencies
    times : :class:`np.ndarray`
         real, (ntimes) times

    Returns
    -------
    uvdist : :class:`np.ndarray`
        real, (nbls) UV distances in wavelengths
    """
    wl = katpoint.lightspeed / freq
    cross_idx = np.where(cal_bls_lookup[:, 0] !=
                         cal_bls_lookup[:, 1])[0]
    kat_target = katpoint.Target(target)
    u, v, w = calprocs.calc_uvw_wave(kat_target, times, cal_bls_lookup[cross_idx],
                                     antennas, wl, cal_array_position)
    uvdist = np.hypot(u, v)
    return uvdist


def calc_enu_sep(antennas, bls_lookup):
    """
    Calculate baseline separation in meters
    for cross correlations only.

    Parameters
    ----------
    antennas : :class:`katpoint.Antenna`
        antennas
    bls_lookup : :class:`np.ndarray`
        array of indices of antennas in each baseline

    Returns
    -------
    sep: :class:`np.ndarray`
        real (nbls), separations for baselines in bls_lookup
    """

    cross_idx = np.where(bls_lookup[:, 0] != bls_lookup[:, 1])[0]
    bls_lookup = bls_lookup[cross_idx]
    ant1 = [antennas[bls_lookup[i][0]] for i in range(len(bls_lookup))]
    ant2 = [antennas[bls_lookup[i][1]] for i in range(len(bls_lookup))]
    bl = np.empty([len(ant1), 3])

    for i in range(len(bl)):
        enu = ant1[i].baseline_toward(ant2[i])
        bl[i] = enu

    sep = np.linalg.norm(bl, axis=1)

    return sep


def make_cal_report(ts, capture_block_id, stream_name, parameters, report_path, av_corr,
                    st=None, et=None):
    """
    Creates pdf calibration pipeline report (from RST source),
    using data from the Telescope State

    Parameters
    ----------
    ts : :class:`katsdptelstate.TelescopeState`
        telescope state, with prefixes for calname and cbid_calname
    capture_block_id : str
        capture block ID
    stream_name : str
        name of the L0 data stream
    parameters : dict
        Pipeline parameters
    report_path : str
        path where report will be created
    av_corr : dict
        dictionary containing arrays of calibrated data
    st : float, optional
        start time for reporting parameters, seconds
    et : float, optional
        end time for reporting parameters, seconds
    """

    logger.info('Report compiling in directory {0}'.format(report_path))

    # --------------------------------------------------------------------
    # open report file
    report_file = os.path.join(report_path, 'calreport.rst')

    # --------------------------------------------------------------------
    # write heading
    with rstReport(report_file, 'w') as cal_rst:
        cal_rst.write_heading_0('Calibration pipeline report')

        # --------------------------------------------------------------------
        # write observation summary info
        cal_rst.write_heading_1('Observation summary')
        cal_rst.writeln('Capture block: {}'.format(capture_block_id))
        cal_rst.writeln()
        cal_rst.writeln('Stream: {}'.format(stream_name))
        cal_rst.writeln()
        write_summary(cal_rst, ts, stream_name, parameters, st=st, et=et)

        # Plot elevation vs time for reference antenna
        unique_targets = list(set(av_corr['targets']))
        refant_index = parameters['refant_index']
        antennas = parameters['antennas']
        if len(av_corr['targets']) > 0:
            write_elevation(cal_rst, report_path, unique_targets, antennas[refant_index], av_corr)

        # -------------------------------------------------------------------
        # write RFI summary
        cal_rst.write_heading_1('RFI and Flagging summary')

        correlator_freq = parameters['channel_freqs'] / 1e6
        cal_bls_lookup = parameters['bls_lookup']
        pol_order = parameters['pol_ordering']
        pol = [_[0].upper() for _ in pol_order if _[0] == _[1]]
        if len(av_corr['targets']) > 0:
            dist = calc_enu_sep(antennas, cal_bls_lookup)
            write_flag_summary(cal_rst, report_path, av_corr, dist, correlator_freq, pol)
        else:
            logger.info(' - no calibrated data')

        # --------------------------------------------------------------------
        # add cal products to report
        antenna_names = parameters['antenna_names']
        write_products(cal_rst, report_path, ts, parameters,
                       st, et, antenna_names, correlator_freq, pol)
        logger.info('Calibration solution summary')

        # --------------------------------------------------------------------
        # Corrected data : Calibrators
        cal_rst.write_heading_1('Calibrator Summary Plots')

        # Split observed targets into different types of sources,
        # according to their pipeline tags
        nogain, gain, target = split_targets(unique_targets)

        # For calibrators which do not have gains applied by the pipeline
        # plot the baselines to the reference antenna for each timestamp
        # get idx of baselines to refant
        ant_idx = np.where((
            (cal_bls_lookup[:, 0] == refant_index)
            | (cal_bls_lookup[:, 1] == refant_index))
            & ((cal_bls_lookup[:, 0] != cal_bls_lookup[:, 1])))[0]

        write_ng_freq(cal_rst, report_path, nogain, av_corr, ant_idx,
                      refant_index, antenna_names, correlator_freq, pol)
        write_g_freq(cal_rst, report_path, gain, av_corr, antenna_names,
                     cal_bls_lookup, correlator_freq, True, pol)
        write_g_time(cal_rst, report_path, gain, av_corr, antenna_names, cal_bls_lookup, pol)

        cal_array_position = parameters['array_position']
        write_g_uv(cal_rst, report_path, gain, av_corr, cal_bls_lookup,
                   antennas, cal_array_position, correlator_freq, True, pol=pol)

        # --------------------------------------------------------------------
        # Corrected data : Targets
        cal_rst.write_heading_1('Calibrated Target Fields')
        write_g_freq(cal_rst, report_path, target, av_corr, antenna_names,
                     cal_bls_lookup, correlator_freq, False, pol=pol)
        write_g_uv(cal_rst, report_path, target, av_corr, cal_bls_lookup,
                   antennas, cal_array_position, correlator_freq, False, pol=pol)

        cal_rst.writeln()

    # convert to html
    report_file_html = os.path.join(report_path, 'calreport.html')
    publish_file(source_path=report_file, destination_path=report_file_html,
                 writer_name='html')
