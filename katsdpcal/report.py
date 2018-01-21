import os
import time
import logging
import datetime

from . import plotting
from . import calprocs
from . import calprocs_dask
import numpy as np

from docutils.core import publish_file

import matplotlib.pylab as plt
import katpoint

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# No of antennas per plot 
ant_chunks=16 
# No of channels for plots which aren't spectra
plot_channels=8
# Tag blacklist
tag_whitelist=['gain','bfcal','delaycal','polcal','bpcal','target']

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
def utc_tstr(t_stamp, day=False):
    """ 
    Returns a formatted UTC time string 

    Parameters
    ----------
    t_stamp : float
         unix_timestamp
    day : bool, optional 
         if true don't include the year and 
         month in the time string
    Returns
    -------
    str : formatted time string 
    """ 
    time=datetime.datetime.utcfromtimestamp(t_stamp)
    time_format="%Y-%m-%d %H:%M:%S"
    if day:
        time_format="%d %H:%M:%S"        
    time_string=time.strftime(time_format)
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


def write_bullet_if_present(report, ts, var_text, var_name, transform=None):
    """
    Write bullet point, if TescopeState key is present

    Parameters
    ----------
    report : file-like
        report file to write to
    ts : :class:`katsdptelstate.TelescopeState`
        telescope state
    var_text : str
        bullet point description
    var_name : str
        telescope state key
    transform : callable, optional
        transform for applying to TelescopeState value before reporting
    """
    ts_value = ts[var_name] if var_name in ts else 'unknown'
    if transform is not None:
        ts_value = transform(ts_value)
    report.writeln('* {0}:  {1}'.format(var_text, ts_value))


def write_summary(report, ts, stream_name, st=None, et=None):
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
    st : float, optional
        start time for reporting parameters, seconds
    et : float, optional
        end time for reporting parameters, seconds
    """
    # write RST style bulletted list

    report.writeln('* Start time:  '+utc_tstr(st))

    # telescope state values
    write_bullet_if_present(report, ts, 'Int time', stream_name + '_int_time')
    write_bullet_if_present(report, ts, 'Channels', stream_name + '_n_chans')
    write_bullet_if_present(report, ts, 'Antennas', 'cal_antlist', transform=len)
    write_bullet_if_present(report, ts, 'Antenna list', 'cal_antlist')
    report.writeln()

    report.writeln('Source list:')
    report.writeln()
    try:
        target_list = \
            ts.get_range('cal_info_sources', st=st, et=et, return_format='recarray')['value'] \
            if 'cal_info_sources' in ts else []
        for target in target_list:
            report.writeln('* {0:s}'.format(target,))
    except AttributeError:
        # key not present
        report.writeln('* Unknown')

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

    timestrings = [utc_tstr(t,True) for t in times]

    # add each time row to the table
    for t, d in zip(timestrings, data):
        data_string = " ".join(["{:.3f}".format(di.real,).ljust(col_width)
                                for di in np.atleast_1d(d)])
        report.write("{}".format(t,).ljust(col_width + 1))
        report.writeln(data_string)

    # table footer
    report.writeln(col_header * n_entries)
    report.writeln()


def write_table_timecol(report, antennas, times, data):
    """
    Write RST style table to report, rows: antenna, columns: time

    Parameters
    ----------
    report : file-like
        report file to write to
    antennas : str or list
        list of antenna names or single string of comma-separated antenna names
    times : list
        list of times (equates to number of columns in the table)
    data
        table data, shape (time, antenna)
    """
    n_entries = len(times) + 1
    col_width = 30
    col_header = '=' * col_width + ' '

    # create table header
    timestrings = [utc_tstr(t,day = 'True') for t in times]
    header = " ".join(["{}".format(t,).ljust(col_width) for t in timestrings])
    header = 'Ant'.ljust(col_width + 1) + header

    # write table header
    report.writeln()
    report.writeln(col_header * n_entries)
    report.writeln(header)
    report.writeln(col_header * n_entries)

    # add each antenna row to the table
    antlist = antennas if isinstance(antennas, list) else antennas.split(',')
    for a, d in zip(antlist, data.T):
        data_string = " ".join(["{:.3f}".format(di.real,).ljust(col_width) for di in d])
        report.write(a.ljust(col_width + 1))
        report.writeln(data_string)

    # table footer
    report.writeln(col_header * n_entries)
    report.writeln()


def write_elevation(report, report_path, targets, refant_desc, av_corr):
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
    refant_desc : str
         reference antenna description string
    av_corr : dict
         dictionary containing averaged, corrected data
    """
    ts, el, names = [], [], []
    for cal in targets:
        ts_cal = [ti for ti, t in zip(av_corr['t_stamps'],
                                      av_corr['targets']) if t == cal]
        ts_flat = np.array([x for y in ts_cal for x in y])
        el_cal = calc_elevation(refant_desc, ts_flat, cal)

        ts.append(ts_flat)
        names.append(katpoint.Target(cal).name)
        el.append(el_cal)

    refant_name = katpoint.Antenna(refant_desc).name
    plot_title = 'Elevation vs Time for Reference Antenna: {0}'.format(
        refant_name)
    plot = plotting.plot_el_v_time(
        names, ts, el, title=plot_title)
    insert_fig(report_path, report, plot, name='El_v_time')


def write_flag_summary(report, report_path, av_corr, dist, correlator_freq, pol=[0,1]):
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
    dist : array
        array of separations between antennas for baselines in av_corr
    correlator_freq : array
        array of correlator channel frequencies
    pol : list 
        description of polarisation axes, optional 
    """
    report.writeln('Percentage of time data is flagged')
    # Flags per scan weighted by length of scan
    n_times = np.array([len(_) for _ in av_corr['t_stamps']], dtype=np.float32)
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
                  refant_index, antenna_mask, correlator_freq, pol=[0,1]):
    """
    Include plots of spectra of all scans of non gain-calibrated calibrators in report.
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
    ant_idx : array of int
           array of indices of all baselines to the reference antenna
    refant_index : int
           index of reference antenna
    antenna_mask : list
           list of antenna names
    correlator_freq : array
           array of correlator channel frequencies
    pol : list 
        description of polarisation axes, optional 
    """
    if len(targets) > 0:
        report.write_heading_2(
            'Corrected Amp and Phase vs Frequency, delay and bandpass calibrators ')
        report.writeln()
        report.write_heading_3(
            'Baselines to the reference antenna : {0}'.format(antenna_mask[refant_index]))

    for cal in targets:
        kat_target = katpoint.Target(cal)
        target_name = kat_target.name
        tags = [t for t in kat_target.tags if t in tag_whitelist]

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
            for idx in range(0, ant_data.shape[-1], ant_chunks):
                plot = plotting.plot_spec(
                    ant_data[ti, ..., idx: idx + ant_chunks], idx_chan,
                    antlist=antenna_mask[idx: idx + ant_chunks],
                    freq_range=freq_range, title=plot_title, pol=pol)
                insert_fig(report_path, report, plot, name='Corr_v_Freq_{0}_ti_{1}_{2}'.format(
                    target_name.replace(' ', '_'), ti, idx))
                report.writeln()


def write_g_freq(report, report_path, targets, av_corr, antenna_mask,
                 cal_bls_lookup, correlator_freq, amp=False, pol=[0,1]):
    """
    Include plots of spectra of all scans of gain-calibrated calibrators in report.

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
    antenna_mask : list
           list of antenna names
    cal_bls_lookup : :class:`np.ndarray`
           array of antenna indices in each baseline
    correlator_freq : :class:`np.ndarray`
           array of correlator channel frequencies
    amp : bool, optional
           plot only amplitudes if True, else plot amplitude and phase
    pol : list 
        description of polarisation axes, optional 
    """
    if amp:
        suffix = ('', 'all target fields')
    else:
        suffix = (' and Phase', 'all gain-calibrated calibrators')

    if len(targets) > 0:
        report.write_heading_2('Corrected Amp{0} vs Frequency, {1}'.format(suffix[0], suffix[1]))
        report.writeln()
        report.write_heading_3('All baselines, averaged per antenna')

    # For gain-calibrated calibrators average per antenna and across scans and plot
    for cal in targets:
        kat_target = katpoint.Target(cal)
        target_name = kat_target.name
        tags = [t for t in kat_target.tags if t in tag_whitelist]

        # Average per antenna and across all scans for target cal
        av_data, av_flags, av_weights, av_times = select_data(av_corr, [cal])
        logger.info(' Corrected data for {0} shape: {1}'.format(target_name, av_data.shape))
        av_data, av_flags, av_weights = calprocs.wavg_ant(av_data, av_flags, av_weights,
                                                          ant_array=antenna_mask,
                                                          bls_lookup=cal_bls_lookup)
        av_data = calprocs_dask.wavg(av_data, av_flags, av_weights)

        # Get channel index in correlator channels
        n_av_chan = av_data.shape[-3]
        idx_chan, freq_chan = get_freq_info(correlator_freq, n_av_chan)
        freq_range = [freq_chan[0], freq_chan[-1]]

        # Set the plot label
        if amp:
            plot_title = 'Target: {0}'.format(target_name)
        else:
            plot_title = 'Calibrator: {0} , tags are {1}'.format(target_name, ', '.join(tags))
        # Only plot a maximum of 16 antennas per plot
        for idx in range(0, av_data.shape[-1], ant_chunks):
            data = av_data[..., idx: idx + ant_chunks].compute()
            plot = plotting.plot_spec(
                data, idx_chan, antlist=antenna_mask[idx: idx + ant_chunks],
                freq_range=freq_range, title=plot_title, amp=amp, pol=pol)

            insert_fig(report_path, report, plot,
                       name='Corr_v_Freq_{0}_{1}'.format(target_name.replace(" ", "_"), idx))
            report.writeln()


def write_g_time(report, report_path, targets, av_corr, antenna_mask, cal_bls_lookup, pol):
    """
    Include plots of amp and phase versus time of all scans of the given targets in report.
    The plots show data averaged per antenna, with 8 channels averaged in frequency.

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
    antenna_mask : list
           list of antenna names
    cal_bls_lookup : :class:`np.ndarray`
           array of antenna indices in each baseline
    pol : list 
        description of polarisation axes, optional 
    """
    # Select all gain-calibrated calibrator scans
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
            ant_array=antenna_mask,
            bls_lookup=cal_bls_lookup)

        # average bandpass into a maximum of 8 chunks
        nchan = av_data.shape[-3]
        if nchan >= plot_channels:
            chanav = nchan // plot_channels
            av_data, av_flags, av_weights = calprocs_dask.wavg_full_f(
                av_data, av_flags, av_weights, chanav)

        # insert plots of phase v time
        for idx in range(0, av_data.shape[-1], ant_chunks):
            plot = plotting.plot_corr_v_time(
                av_times, av_data[..., idx: idx + ant_chunks],
                antlist=antenna_mask[idx: idx + ant_chunks], pol=pol)
            insert_fig(report_path, report, plot, name='Phase_v_Time_{0}'.format(idx))
            report.writeln()

        report.write_heading_2(
            'Corrected Amp vs Time, all gain-calibrated calibrators')
        report.writeln()
        report.write_heading_3('All baselines, averaged per antenna')
        report.writeln()

        # insert plots of amp v time
        for idx in range(0, av_data.shape[-1], ant_chunks):
            plot = plotting.plot_corr_v_time(av_times,
                                             av_data[..., idx: idx + ant_chunks], plottype='a',
                                             antlist=antenna_mask[idx: idx + ant_chunks], pol=pol)

            insert_fig(report_path, report, plot, name='Amp_v_Time_{0}'.format(idx))
            report.writeln()


def write_g_uv(report, report_path, targets, av_corr, cal_bls_lookup,
               cal_antlist_description, cal_array_position, correlator_freq, amp, pol=[0,1]):
    """
    Include plots of amp and phase/amp versus uvdist in report.
    The plots show 8 channels averaged in frequency.

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
    cal_antlist_description : list
           list of description strings of antennas
    cal_array_position : str
           description string of array position
    correlator_freq : :class:`np.ndarray`
           array of correlator channel frequencies
    amp : bool, optional
           plot only amplitudes if True, else plot amplitude and phase
    pol : list 
        description of polarisation axes, optional 
    """
    if amp:
        suffix = ('', 'all target fields')
    else:
        suffix = (' and Phase', 'all gain-calibrated calibrators')

    if len(targets) > 0:
        report.write_heading_2(
            'Amp{0} vs UVdist, {1}'.format(suffix[0], suffix[1]))
        report.write_heading_3('All baselines')

    # Plot the Amp and Phase vs UV distance for all gain-calibrated calibrators.
    for cal in targets:
        kat_target = katpoint.Target(cal)
        target_name = kat_target.name
        tags = [t for t in kat_target.tags if t in tag_whitelist]
        # Get data for target cal
        av_data, av_flags, av_weights, av_times = select_data(av_corr, [cal])
        # average bandpass into a maximum of 8 chunks
        nchan = av_corr['vis'].shape[-3]
        if nchan >= plot_channels:
            chanav = nchan // plot_channels
            nchan = plot_channels
            av_data, av_flags, av_weights = calprocs_dask.wavg_full_f(av_data,
                                                                      av_flags,
                                                                      av_weights,
                                                                      chanav)
        else:
            chanav = nchan

        # Get channel index in correlator channels
        idx_chan, freq_chan = get_freq_info(correlator_freq, nchan)
        freq_chan = freq_chan * 1e6
        uvdist = calc_uvdist(cal, freq_chan, av_times,
                             cal_bls_lookup, cal_antlist_description, cal_array_position)
        if amp:
            plot_title = 'Target {0}'.format(target_name)
        else:
            plot_title = 'Calibrator {0}, tags are {1}'.format(target_name, ', '.join(tags))
        plot = plotting.plot_corr_uvdist(uvdist, av_data, freq_chan, plot_title, amp, pol=pol)
        insert_fig(report_path, report, plot,
                   name='Corr_v_UVdist_{0}'.format(target_name.replace(" ", "_")))
        report.writeln()


def write_products(report, report_path, ts, st, et, antenna_mask, correlator_freq, pol=[0,1]):
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
    st : float
        start time for reporting parameters, seconds
    et : float
        end time for reporting parameters, seconds
    antenna_mask : list
           list of antenna names
    correlator_freq : :class:`np.ndarray`
           array of correlator channel frequencies

    """

    cal_list = ['K', 'KCROSS', 'B', 'G']
    solns_exist = any(['cal_product_' + cal in ts.keys() for cal in cal_list])
    if not solns_exist:
        logger.info(' - no calibration solutions')

    # delay
    cal = 'K'
    vals, times = get_cal(ts, cal, st, et)
    if len(times) > 0:
        cal_heading(report, cal, 'Delay', '([ns])')
        write_K(report, report_path, times, vals, antenna_mask, pol)

    # ---------------------------------
    # cross pol delay
    cal = 'KCROSS'
    vals, times = get_cal(ts, cal, st, et)
    if len(times) > 0:
        cal_heading(report, cal, 'Cross polarisation delay', '([ns])')
        # convert delays to nano seconds
        vals = 1e9 * vals
        write_table_timerow(report, [cal], times, vals)

    # ---------------------------------
    # bandpass
    cal = 'B'
    vals, times = get_cal(ts, cal, st, et)
    if len(times) > 0:
        cal_heading(report, cal, 'Bandpass')
        write_B(report, report_path, times, vals, antenna_mask, correlator_freq, pol)

    # ---------------------------------
    # gain
    cal = 'G'
    vals, times = get_cal(ts, cal, st, et)
    if len(times) > 0:
        cal_heading(report, cal, 'Gain')
        for idx in range(0, vals.shape[-1], ant_chunks):
            plot = plotting.plot_g_solns_legend(
                times, vals[..., idx: idx + ant_chunks],
                antlist=antenna_mask[idx: idx + ant_chunks], pol=pol)
            insert_fig(report_path, report, plot, name='{0}'.format(cal))


def get_cal(ts, cal, st, et):
    """
    Fetch a calibration product from telstate

    Parameters:
    -----------
    ts : :class:`katsdptelstate.TelescopeState`
        telescope state
    cal: str
         string indicating calibration product type
    st : float
        start time for reporting parameters, seconds
    et : float
        end time for reporting parameters, seconds
    Returns:
    --------
    vals : array of values of calibration product
    times : array of times of calibration product
    """
    cal_product = 'cal_product_' + cal
    vals, times = [], []
    if cal_product in ts:
        product = ts.get_range(cal_product, st=st, et=et, return_format='recarray')
        if len(product['time']) > 0:
            logger.info('Calibration product: {0}'.format(cal,))
            vals = product['value']
            # K shape is n_time, n_pol, n_ant
            times = product['time']
            logger.info('  shape: {0}'.format(vals.shape,))
    return vals, times


def write_K(report, report_path, times, vals, antenna_mask, pol=[0,1]):
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
    antenna_mask : list
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
        write_table_timecol(report, antenna_mask, times, kpol)

    plot = plotting.plot_delays(times, vals, antlist=antenna_mask, pol=pol)
    insert_fig(report_path, report, plot, name='K')


def write_B(report, report_path, times, vals, antenna_mask, correlator_freq, pol=[0,1]):
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
    antenna_mask : list
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

        for idx in range(0, vals[ti].shape[-1], ant_chunks):
            plot = plotting.plot_spec(
                vals[ti, ..., idx: idx + ant_chunks], chan_no,
                antlist=antenna_mask[idx: idx + ant_chunks],
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
    correlator_freq : array
         array of correlator channel frequencies
    nchan : int
         no of averaged channels
    Returns
    -------
    avchan : array
       array of mean channel indices in correlator channels
    avfreq : array
       array of mean frequencies of averaged channels
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
    data : :class:`np.ndarray` of visibilities
    flags : :class:`np.ndarray` of flags
    weights : :class:`np.ndarray` of weights
    times : :class:`np.ndarray` of times

    """
    idx = [t in t_list for t in av_corr['targets']]
    data = av_corr['vis'][idx]
    flags = av_corr['flags'][idx]
    weights = av_corr['weights'][idx]
    times = av_corr['times'][idx]
    if type(times) == np.float64:
        times=np.array([times]) 

    return data, flags, weights, times


def split_targets(targets):
    """
    Separate targets into three lists containing
    non gain-calibrated calibrators, gain-calibrated
    calibrators and targets

    Parameters
    -----------
    targets : list
        list of unique targets
    Returns
    -------
    nogain : list
        list of non gain-calibrated calibrators
    gain : list
        list of gain-calibrated calibrators
    target : list
        list of targets
    """
    nogain, gain, target = [], [], []
    for cal in targets:
        kat_target = katpoint.Target(cal)
        tags = kat_target.tags
        # tags which are gain calibrated by pipeline
        gaintaglist = ('gaincal', 'bfcal', 'target')
        if not any(x in gaintaglist for x in tags):
            nogain.append(cal)
        elif 'target' in tags:
            target.append(cal)
        else:
            gain.append(cal)
    return nogain, gain, target


def calc_elevation(refant_desc, times, target):
    """
      Calculates elevation versus timestamps for observation targets

      Parameters
      ----------
      refant_desc : str
          description string of the reference antenna
      times : array
          timestamps of scan
      target : str
          target string

      Returns
      -------
      times : array
          array of timestamps
      elevation : array
          array of elevations
      """
    kat_refant = katpoint.Antenna(refant_desc)
    kat_target = katpoint.Target(target)
    elevations = kat_target.azel(times, kat_refant)[1]

    return elevations


def calc_uvdist(target, freq, times, cal_bls_lookup, cal_antlist_description, cal_array_position):
    """
    Calculate uvdistance in wavelengths

    Parameters
    ----------
    ts : :class:`katsdptelstate.TelescopeState`
        telescope state
    target : str
         target string
    frequencies : array
         array of frequencies
    times : array
         array of times

    Returns
    -------
    array of UV distances in wavelengths
    """
    wl = katpoint.lightspeed / freq
    cross_idx = np.where(cal_bls_lookup[:, 0] !=
                         cal_bls_lookup[:, 1])[0]
    kat_target = katpoint.Target(target)
    u, v, w = calprocs.calc_uvw_wave(kat_target, times, cal_bls_lookup[cross_idx],
                                     cal_antlist_description, wl, cal_array_position)
    uvdist = np.hypot(u, v)
    return uvdist


def calc_enu_sep(ant_desc, bls_lookup):
    """
    Calculate baseline separation in meters
    for cross correlations only.

    Parameters
    ----------
    ant_desc : str
         antenna description string
    bls_lookup : :class:`np.ndarray`
         array of indices of antennas in each baseline

    Returns
    -------
    array of separations for baselines in bls_lookup
    """

    cross_idx = np.where(bls_lookup[:, 0] != bls_lookup[:, 1])[0]
    bls_lookup = bls_lookup[cross_idx]
    ant1 = [ant_desc[bls_lookup[i][0]] for i in range(len(bls_lookup))]
    ant2 = [ant_desc[bls_lookup[i][1]] for i in range(len(bls_lookup))]
    bl = np.empty([len(ant1), 3])

    for i, _ in enumerate(bl):
        kat_ant1 = katpoint.Antenna(ant1[i])
        kat_ant2 = katpoint.Antenna(ant2[i])
        enu = kat_ant1.baseline_toward(kat_ant2)
        bl[i] = enu

    sep = np.linalg.norm(bl, axis=1)

    return sep


def make_cal_report(ts, stream_name, report_path, av_corr, project_name=None, st=None, et=None):
    """
    Creates pdf calibration pipeline report (from RST source),
    using data from the Telescope State

    Parameters
    ----------
    ts : :class:`katsdptelstate.TelescopeState`
        telescope state
    stream_name : str
        name of the L0 data stream
    report_path : str
        path where report will be created
    av_corr : dict
        dictionary containing arrays of calibrated data
    project_name : str, optional
        ID associated with project
    st : float, optional
        start time for reporting parameters, seconds
    et : float, optional
        end time for reporting parameters, seconds
    """

    if project_name is None:
        project_name = '{0}_unknown_project'.format(time.time())

    if not report_path:
        report_path = '.'
    project_dir = os.path.abspath(report_path)
    logger.info('Report compiling in directory {0}'.format(project_dir))

    # --------------------------------------------------------------------
    # open report file
    report_file = 'calreport_{0}.rst'.format(project_name)
    report_file = os.path.join(project_dir, report_file)

    # --------------------------------------------------------------------
    # write heading
    with rstReport(report_file, 'w') as cal_rst:
        cal_rst.write_heading_0('Calibration pipeline report')

        # --------------------------------------------------------------------
        # write observation summary info
        cal_rst.write_heading_1('Observation summary')
        cal_rst.writeln('Observation: {0:s}'.format(project_name))
        cal_rst.writeln()
        write_summary(cal_rst, ts, stream_name, st=st, et=et)

        # Plot elevation vs time for reference antenna
        unique_targets = list(set(av_corr['targets']))
        refant_index = ts['cal_antlist'].index(ts['cal_refant'])
        ant_desc = ts['cal_antlist_description']
        write_elevation(cal_rst, report_path, unique_targets, ant_desc[refant_index], av_corr)

        # -------------------------------------------------------------------
        # write RFI summary
        cal_rst.write_heading_1('RFI and Flagging summary')

        correlator_freq = ts['cal_channel_freqs'] / 1e6
        cal_bls_lookup = ts['cal_bls_lookup']
        pol_order=ts['cal_pol_ordering']
        pol = [_[0].upper() for _ in pol_order if _[0]==_[1]]
        if len(av_corr['targets']) > 0:
            dist = calc_enu_sep(ant_desc, cal_bls_lookup)
            write_flag_summary(cal_rst, report_path, av_corr, dist, correlator_freq, pol)
        else:
            logger.info(' - no calibrated data')

        # --------------------------------------------------------------------
        # add cal products to report
        antenna_mask = ts.cal_antlist
        write_products(cal_rst, report_path, ts, st, et, antenna_mask, correlator_freq, pol)
        logger.info('Calibration solution summary')

        # --------------------------------------------------------------------
        # Corrected data : Calibrators
        cal_rst.write_heading_1('Calibrator Summary Plots')

        # Split observed targets into different types of sources,
        # according to their pipeline tags
        nogain, gain, target = split_targets(unique_targets)

        # For non gain-calibrated calibrators plot the baselines
        # to the reference antenna for each timestamp
        # get idx of baselines to refant
        ant_idx = np.where((
            (cal_bls_lookup[:, 0] == refant_index)
            | (cal_bls_lookup[:, 1] == refant_index))
            & ((cal_bls_lookup[:, 0] != cal_bls_lookup[:, 1])))[0]

        write_ng_freq(cal_rst, report_path, nogain, av_corr, ant_idx,
                      refant_index, antenna_mask, correlator_freq, pol)
        write_g_freq(cal_rst, report_path, gain, av_corr, antenna_mask,
                     cal_bls_lookup, correlator_freq, amp=False, pol=pol)
        write_g_time(cal_rst, report_path, gain, av_corr, antenna_mask, cal_bls_lookup, pol)

        cal_array_position = ts['cal_array_position']
        write_g_uv(cal_rst, report_path, gain, av_corr, cal_bls_lookup,
                   ant_desc, cal_array_position, correlator_freq, amp=False, pol=pol)

        # --------------------------------------------------------------------
        # Corrected data : Targets
        cal_rst.write_heading_1('Calibrated Target Fields')
        write_g_freq(cal_rst, report_path, target, av_corr, antenna_mask,
                     cal_bls_lookup, correlator_freq, amp=True, pol=pol)
        write_g_uv(cal_rst, report_path, target, av_corr, cal_bls_lookup,
                   ant_desc, cal_array_position, correlator_freq, amp=True, pol=pol)

        cal_rst.writeln()

    # convert to html
    publish_file(source_path=report_file, destination_path=report_file.replace('rst', 'html'),
                 writer_name='html')
