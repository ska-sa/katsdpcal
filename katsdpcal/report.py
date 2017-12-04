import os

import logging

from . import plotting
from . import calprocs
from . import calprocs_dask
import numpy as np

import time

from docutils.core import publish_file

import matplotlib.pylab as plt
import katpoint
from scipy.constants import c as light_speed

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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

    def writeln(self, line=None):
        if line is not None:
            self.write(line)
        self.write('\n')


# --------------------------------------------------------------------------------------------------
# --- FUNCTION :  Report writing functions
# --------------------------------------------------------------------------------------------------

def insert_fig(report_path, report, fig, name=None):
    """
    Insert matplotlib figure into report

    Parameters
    ----------
    report : file-like
        open report file to write to
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
    report.writeln('* {0}:  {1}'.format('Start time', time.strftime("%x %X", time.gmtime(st))))

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

    timestrings = [time.strftime("%d %X", time.gmtime(t)) for t in times]

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
    timestrings = [time.strftime("%d %X", time.gmtime(t)) for t in times]
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


def calc_elevation(ts, stream_name, times, n_times, target):
    """
      Calculates elevation versus timestamps for observation targets

      Parameters
      ----------
      ts : :class:`katsdptelstate.TelescopeState`
          telescope state
      stream_name : str
          name of the L0 data stream
      times : array
          start times
      n_times : array
          number of timestamps
      target : str
          target string

      Returns
      -------
      times : array of timestamps
      elevation : array of elevations
      """
    # get reference antenna description
    refant_index = ts['cal_antlist'].index(ts['cal_refant'])
    kat_refant = katpoint.Antenna(ts['cal_antlist_description'][refant_index])

    timestamps = np.array([])
    elevations = np.array([])

    kattarget = katpoint.Target(target)
    int_time = ts[stream_name + '_int_time']

    for st, nt in zip(times, n_times):
        # calculate timestamps
        t_inc = int_time * (range(nt) - np.median(range(nt))) + st
        # calculate elevations
        el = kattarget.azel(t_inc, kat_refant)[1]
        timestamps = np.append(timestamps, t_inc, axis=0)
        elevations = np.append(elevations, el, axis=0)

    return timestamps, elevations


def calc_uvdist(ts, target, freq, times):
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
    wl = light_speed / freq
    cross_idx = np.where(ts['cal_bls_lookup'][:, 0] !=
                         ts['cal_bls_lookup'][:, 1])[0]
    kattarget = katpoint.Target(target)
    u, v, w = calprocs.calc_uvw_wave(kattarget, times, ts['cal_bls_lookup'][cross_idx],
                                     ts['cal_antlist_description'], wl, ts['cal_array_position'])
    uvdist = np.power(u**2 + v**2, 0.5)
    return uvdist


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
    av_corr : dictionary
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
    cal_rst = rstReport(report_file, 'w')

    # --------------------------------------------------------------------
    # write heading
    cal_rst.write_heading_0('Calibration pipeline report')

    # --------------------------------------------------------------------
    # write observation summary info
    cal_rst.write_heading_1('Observation summary')
    cal_rst.writeln('Observation: {0:s}'.format(project_name))
    cal_rst.writeln()
    write_summary(cal_rst, ts, stream_name, st=st, et=et)

    # Determine frequency channels and ranges for plots
    chanav = int(av_corr['vis'].shape[-3] / 8)
    total_chanav = int(ts['cal_channel_freqs'].shape[0] / 8)
    chan_steps = range(0, ts['cal_channel_freqs'].shape[0], total_chanav)
    av_chan_freq = np.add.reduceat(
        ts['cal_channel_freqs'], chan_steps) / total_chanav
    freq_range = (ts['cal_channel_freqs'][0] / 1e6,
                  ts['cal_channel_freqs'][-1] / 1e6)

    # Plot elevation vs time for one antenna
    if len(av_corr['targets']) > 0:
        unique_targets = list(set(av_corr['targets']))
        timestamps, el, names = [], [], []
        t_zero = av_corr['times'][0]
        for cal in unique_targets:
            cal_idx = [idx for idx, t in enumerate(
                av_corr['targets']) if t == cal]
            t_cal, el_cal = calc_elevation(ts, stream_name,
                                           av_corr['times'][cal_idx],
                                           av_corr['n_times'][cal_idx], cal)
            t_zero = min([t_zero, np.min(t_cal)])
            timestamps.append(t_cal)
            names.append(cal.split(',')[0])
            el.append(el_cal)

        plot_title = 'Elevation vs Time for Reference Antenna: {0}'.format(
            ts['cal_refant'])
        plot = plotting.plot_el_v_time(
            names, timestamps, el, t_zero=t_zero, title=plot_title)
        insert_fig(report_path, cal_rst, plot, name='El_v_time')
    else:
        logger.info(' - no calibrated data')

    # --------------------------------------------------------------------
    # write RFI summary
    cal_rst.write_heading_1('RFI and Flagging summary')
    if len(av_corr['targets']) > 0:

        cal_rst.writeln('Percentage of time data is flagged')
        # Flags per scan weighted by length of scan
        tw_flags = av_corr['flags'] * av_corr['n_times'][:, np.newaxis, np.newaxis, np.newaxis]
        tw_flags = 100 * np.sum(tw_flags, axis=0) / np.sum(av_corr['n_times'], axis=0)
        plot = plotting.flags_bl_v_chan(tw_flags, freq_range=freq_range)
        insert_fig(report_path, cal_rst, plot, name='Flags_bl_v_chan')

        cal_rst.writeln('Percentage of baselines flagged per scan')
        # Average % of baselines flagged per scan
        bl_flags = 100 * np.sum(av_corr['flags'], axis=3) / av_corr['flags'].shape[3]
        plot = plotting.flags_t_v_chan(bl_flags, freq_range=freq_range)
        insert_fig(report_path, cal_rst, plot, name='Flags_s_v_chan')

    cal_rst.writeln()

    # --------------------------------------------------------------------
    # add cal products to report
    antenna_mask = ts.cal_antlist

    logger.info('Calibration solution summary')
    cal_list = ['K', 'KCROSS', 'B', 'G']
    solns_exist = any(['cal_product_' + cal in ts.keys() for cal in cal_list])
    if not solns_exist:
        logger.info(' - no calibration solutions')

    # ---------------------------------
    # delay
    cal = 'K'
    cal_product = 'cal_product_' + cal
    if cal_product in ts:
        product = ts.get_range(cal_product, st=st, et=et, return_format='recarray')
        if len(product['time']) > 0:
            logger.info('Calibration product: {0}'.format(cal,))

            cal_rst.write_heading_1('Calibration product {0:s}'.format(cal,))
            cal_rst.writeln('Delay calibration solutions ([ns])')
            cal_rst.writeln()

            vals = product['value']
            # K shape is n_time, n_pol, n_ant
            times = product['time']

            # convert delays to nano seconds
            vals = 1e9 * vals

            logger.info('  shape: {0}'.format(vals.shape,))

            # iterate through polarisation
            for p in range(vals.shape[-2]):
                cal_rst.writeln('**POL {0}**'.format(p,))
                kpol = vals[:, p, :]
                logger.info('  pol{0} shape: {1}'.format(p, kpol.shape))
                write_table_timecol(cal_rst, antenna_mask, times, kpol)
            plot = plotting.plot_delays(times, vals, antlist=ts['cal_antlist'])
            insert_fig(report_path, cal_rst, plot, name='K')

    # ---------------------------------
    # cross pol delay
    cal = 'KCROSS'
    cal_product = 'cal_product_' + cal
    if cal_product in ts:
        product = ts.get_range(cal_product, st=st, et=et, return_format='recarray')
        if len(product['time']) > 0:
            logger.info('Calibration product: {0}'.format(cal,))

            cal_rst.write_heading_1('Calibration product {0:s}'.format(cal,))
            cal_rst.writeln('Cross polarisation delay calibration solutions ([ns])')
            cal_rst.writeln()

            vals = product['value']
            # K shape is n_time, n_pol, n_ant
            times = product['time']
            logger.info('  shape: {0}'.format(vals.shape,))

            # convert delays to nano seconds
            vals = 1e9 * vals

            write_table_timerow(cal_rst, [cal], times, vals)

    # ---------------------------------
    # bandpass
    cal = 'B'
    cal_product = 'cal_product_' + cal
    if cal_product in ts:
        product = ts.get_range(cal_product, st=st, et=et, return_format='recarray')
        if len(product['time']) > 0:
            logger.info('Calibration product: {0}'.format(cal,))

            cal_rst.write_heading_1('Calibration product {0:s}'.format(cal,))
            cal_rst.writeln('Bandpass calibration solutions')
            cal_rst.writeln()

            vals = product['value']
            # B shape is n_time, n_chan, n_pol, n_ant
            times = product['time']
            logger.info('  shape: {0}'.format(vals.shape,))

            for ti in range(len(times)):
                t = time.strftime("%Y %x %X", time.gmtime(times[ti]))
                cal_rst.writeln('Time: {}'.format(t,))
                ant_chunks = 16
                for idx in range(0, vals[ti].shape[-1], ant_chunks):
                    plot = plotting.plot_spec(
                        vals[ti, ..., idx: idx + ant_chunks],
                        antlist=ts['cal_antlist'][idx: idx + ant_chunks],
                        freq_range=freq_range)
                    insert_fig(report_path, cal_rst, plot,
                               name='{0}_ti_{1}_{2}'.format(cal, str(ti), idx))

    # ---------------------------------
    # gain
    cal = 'G'
    cal_product = 'cal_product_' + cal
    if cal_product in ts:
        product = ts.get_range(cal_product, st=st, et=et, return_format='recarray')
        if len(product['time']) > 0:
            logger.info('Calibration product: {0}'.format(cal,))

            cal_rst.write_heading_1('Calibration product {0:s}'.format(cal,))
            cal_rst.writeln('Gain calibration solutions')
            cal_rst.writeln()

            vals = product['value']
            # G shape is n_time, n_pol, n_ant
            times = product['time']

            logger.info('  shape: {0}'.format(vals.shape,))
            ant_chunks = 16
            for idx in range(0, vals.shape[-1], ant_chunks):
                plot = plotting.plot_g_solns_legend(
                        times, vals[..., idx: idx + ant_chunks],
                        antlist=ts['cal_antlist'][idx: idx + ant_chunks])
                insert_fig(report_path, cal_rst, plot, name='{0}'.format(cal))

    # --------------------------------------------------------------------
    # Corrected data : Calibrators
    cal_rst.write_heading_1('Calibrator Summary Plots')

    if len(av_corr['targets']) > 0:
        # Create lists of different types of targets.
        nogain, gain, target = [], [], []
        for cal in unique_targets:
            tags = cal.split(',')[1].split(' ')
            gaintaglist = ('gaincal', 'bfcal', 'target')
            if not any(x in gaintaglist for x in tags):
                nogain.append(cal)
            elif 'target' in tags:
                target.append(cal)
            else:
                gain.append(cal)

    cal_rst.writeln(
        'Calibrated Amp vs Frequency, delay and bandpass calibrators ')
    cal_rst.writeln()
    # For non-gain calibrated calibrators plot the baselines
    # to the reference antenna for each timestamp,
    for cal in nogain:
        target_name = cal.split(',')[0]
        tags = cal.split(',')[1].split(' ')
        cal_idx = [i for i, t in enumerate(av_corr['targets']) if t == cal]
        logger.info(' Corrected data for {0} shape: {1}'.format(
                    target_name, av_corr['vis'][cal_idx].shape))
        times = av_corr['times'][cal_idx]
        refant_index = ts['cal_antlist'].index(ts['cal_refant'])
        ant_idx = np.where((
            (ts['cal_bls_lookup'][:, 0] == refant_index)
            | (ts['cal_bls_lookup'][:, 1] == refant_index))
            & ((ts['cal_bls_lookup'][:, 0] != ts['cal_bls_lookup'][:, 1])))[0]
        ant_data = av_corr['vis'][cal_idx][..., ant_idx]
        for ti in range(len(times)):
            cal_rst.writeln(
                'Baselines to the reference antenna : {0}'.format(ts['cal_refant']))
            cal_rst.writeln()
            t = time.strftime("%Y %x %X", time.gmtime(times[ti]))
            cal_rst.writeln('Time : {0}'.format(t))
            plot_title = 'Calibrator: {0} , tags are {1}'.format(target_name, tags[2:])
            # Only plot 16 antennas per plot
            ant_chunks = 16
            for idx in range(0, ant_data.shape[-1], ant_chunks):
                plot = plotting.plot_spec(
                    ant_data[ti, ..., idx:idx + ant_chunks],
                    antlist=ts['cal_antlist'][idx:idx + ant_chunks],
                    freq_range=freq_range, title=plot_title)
                insert_fig(report_path, cal_rst, plot, name='Corr_v_Freq_{0}_ti_{1}_{2}'.format(
                    target_name.replace(' ', '_'), ti, idx))
                cal_rst.writeln()

    cal_rst.writeln('Corrected Amp vs Frequency, gain calibrators')
    cal_rst.writeln()
    cal_rst.writeln('All baselines, averaged per antenna')

    # For gain calibrated calibrators average per antenna and across scans and plot
    for cal in gain:
        target_name = cal.split(',')[0]
        tags = cal.split(',')[1].split(' ')
        cal_idx = [i for i, t in enumerate(av_corr['targets']) if t == cal]

        av_data, av_flags, av_weights = calprocs.wavg_ant(
            av_corr['vis'][cal_idx], av_corr['flags'][cal_idx],
            av_corr['weights'][cal_idx], ant_array=ts['cal_antlist'],
            bls_lookup=ts['cal_bls_lookup'])

        av_data = calprocs_dask.wavg(av_data, av_flags, av_weights)
        plot_title = 'Calibrator: {0} , tags are {1}'.format(target_name, tags[2:])
        # Only plot a maximum of 16 antennas per plot
        ant_chunks = 16
        for idx in range(0, av_data.shape[-1], ant_chunks):
            data = av_data[..., idx:idx + ant_chunks]
            plot = plotting.plot_spec(
                data, antlist=ts['cal_antlist'][idx:idx + ant_chunks],
                freq_range=freq_range, title=plot_title)

            insert_fig(report_path, cal_rst, plot,
                       name='Corr_v_Freq_{0}_{1}'.format(target_name.replace(" ", "_"), idx))
            cal_rst.writeln()

    cal_rst.writeln(
        'Corrected Phase vs Time, all gain calibrated calibrators')
    cal_rst.writeln()

    # get index of all gain calibrated calibrator scans, combine these into one plot
    cal_idx = [i for i, t in enumerate(av_corr['targets']) if t in gain]

    # average bandbass into 8 chunks, average per antenna
    av_data, av_flags, av_weights = calprocs.wavg_full_f(
        av_corr['vis'][cal_idx], av_corr['flags'][cal_idx],
        av_corr['weights'][cal_idx], chanav, threshold=0.7)
    av_data, av_flags, av_weights = calprocs.wavg_ant(
        av_data, av_flags, av_weights,
        ant_array=ts['cal_antlist'],
        bls_lookup=ts['cal_bls_lookup'])

    ant_chunks = 16
    for idx in range(0, av_data.shape[-1], ant_chunks):
        plot = plotting.plot_corr_v_time(
            av_corr['times'][cal_idx], av_data[..., idx:idx + ant_chunks],
            antlist=ts['cal_antlist'][idx:idx + ant_chunks], t_zero=t_zero)

        insert_fig(report_path, cal_rst, plot, name='Phase_v_Time_{0}'.format(idx))

    # Plot the Amp and Phase vs time for all gain calibrated calibrators.
    cal_rst.writeln()
    cal_rst.writeln(
        'Corrected Amp vs Time, all gain calibrated calibrators')
    cal_rst.writeln()
    cal_rst.writeln('All baselines, averaged per antenna')
    cal_rst.writeln()

    for idx in range(0, av_data.shape[-1], ant_chunks):
        plot = plotting.plot_corr_v_time(
            av_corr['times'][cal_idx],
            av_data[..., idx:idx + ant_chunks], plottype='a',
            antlist=ts['cal_antlist'][idx:idx + ant_chunks],
            t_zero=t_zero)

        insert_fig(report_path, cal_rst, plot, name='Amp_v_Time_{0}'.format(idx))

    cal_rst.writeln()
    cal_rst.writeln(
        'Amp and Phase vs UVdist, all gain calibrated calibrators')
    cal_rst.writeln()
    cal_rst.writeln('All baselines')
    cal_rst.writeln()

    # Plot the Amp and Phase vs UV distance for all gain calibrated calibrators.
    for cal in gain:
        target_name = cal.split(',')[0]
        tags = cal.split(',')[1].split(' ')
        cal_idx = [i for i, t in enumerate(av_corr['targets']) if t == cal]
        uvdist = calc_uvdist(ts, cal, av_chan_freq, av_corr['times'][cal_idx])
        av_data, av_flags, av_weights = calprocs.wavg_full_f(av_corr['vis'][cal_idx],
                                                             av_corr['flags'][cal_idx],
                                                             av_corr['weights'][cal_idx],
                                                             chanav, threshold=0.7)

        plot_title = 'Calibrator {0}, tags are {1}'.format(target_name, tags[2:])
        plot = plotting.plot_corr_uvdist(uvdist, av_data, av_chan_freq, plot_title)
        insert_fig(report_path, cal_rst, plot,
                   name='Corr_v_UVdist_{0}'.format(target_name.replace(" ", "_")))
        cal_rst.writeln()

    # --------------------------------------------------------------------
    # Corrected data : Target Fields
    cal_rst.write_heading_1('Calibrated Target Fields ')
    cal_rst.writeln()
    cal_rst.writeln('Amp vs Frequency, all target fields')
    cal_rst.writeln()
    cal_rst.writeln('All baselines, averaged per antenna')
    cal_rst.writeln()
    # plot the Amplitude vs Frequency for all target fields.
    for cal in target:
        target_name = cal.split(',')[0]
        cal_idx = [i for i, t in enumerate(av_corr['targets']) if t == cal]
        # average every target per antenna, and across all timestamps.
        av_data, av_flags, av_weights = calprocs.wavg_ant(
            av_corr['vis'][cal_idx], av_corr['flags'][cal_idx],
            av_corr['weights'][cal_idx], ant_array=ts['cal_antlist'],
            bls_lookup=ts['cal_bls_lookup'])

        av_data = calprocs_dask.wavg(av_data, av_flags, av_weights)
        plot_title = 'Field: {0}'.format(target_name)
        # only plot a maximum of 16 antennas per plot
        ant_chunks = 16
        for idx in range(0, av_data.shape[-1], ant_chunks):
            data = av_data[..., idx:idx + ant_chunks]
            plot = plotting.plot_spec_amp(
                data, antlist=ts['cal_antlist'][idx:idx + ant_chunks],
                freq_range=freq_range, title=plot_title)
            insert_fig(report_path, cal_rst, plot,
                       name='Corr_v_Freq_{0}_{1}'.format(target_name.replace(" ", "_"), idx))
            cal_rst.writeln()

    # Plot Amplitude vs UV distance
    cal_rst.writeln('Amp vs UV distance, all target fields')
    cal_rst.writeln()
    cal_rst.writeln('All baselines')
    cal_rst.writeln()
    for cal in target:
        target_name = cal.split(',')[0]
        cal_idx = [i for i, t in enumerate(av_corr['targets']) if t == cal]
        uvdist = calc_uvdist(ts, cal, av_chan_freq, av_corr['times'][cal_idx])
        av_data, av_flags, av_weights = calprocs.wavg_full_f(
            av_corr['vis'][cal_idx], av_corr['flags'][cal_idx],
            av_corr['weights'][cal_idx], chanav, threshold=0.7)
        plot_title = 'Field: {0}'.format(target_name)
        plot = plotting.plot_corr_uvdist_amp(
            uvdist, av_data, freqlist=av_chan_freq, title=plot_title)
        insert_fig(report_path, cal_rst, plot,
                   name='Corr_v_UVdist_{0}'.format(target_name.replace(" ", "_")))
        cal_rst.writeln()

    # --------------------------------------------------------------------
    # close off report
    cal_rst.writeln()
    cal_rst.writeln()
    cal_rst.close()

    # convert to html
    publish_file(source_path=report_file, destination_path=report_file.replace('rst', 'html'),
                 writer_name='html')
