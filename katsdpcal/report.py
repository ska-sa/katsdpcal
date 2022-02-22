import os
import logging
import datetime
import threading

from . import plotting
from . import calprocs
from . import docutils_dir

import numpy as np
import dask.array as da

from docutils.core import publish_file

import astropy.units as u
import astropy.constants as const
import matplotlib.pylab as plt
import katpoint
from katdal.sensordata import to_str
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# No of antennas per plot
ANT_CHUNKS = 16
# Tag blacklist
TAG_WHITELIST = ['gaincal', 'bfcal', 'delaycal', 'polcal', 'bpcal', 'target']

# --------------------------------------------------------------------------------------------------
# --- CLASS :  rstReport
# --------------------------------------------------------------------------------------------------


class rstReport:
    """RST style report."""

    def __init__(self, filename):
        self._file = open(filename, 'w')

    def write(self, text):
        self._file.write(text)

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

    def write_color(self, text, color, width):
        string = ":{0}:`{1}`".format(color, text).ljust(width)
        self.write(string)

    def writeln(self, line=None):
        if line is not None:
            self.write(line)
        self.write('\n')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._file.close()


# --------------------------------------------------------------------------------------------------
# --- FUNCTION :  Report writing functions
# --------------------------------------------------------------------------------------------------
def utc_tstr(timestamp, day=False):
    """Returns a formatted UTC time string.

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
    """Insert matplotlib figure into report.

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
    """Write bullet point, if `var_name` is present in `table`.

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


def _get_units(target, av_corr, flux_cal={}):
    """Determine the amplitude units of a target.

    Parameters
    ----------
    target : :class:`katpoint.Target`
        target whose amplitude units are to be obtained
    av_corr : dict
        dict with model flux densities, keyed by target.name + '_model'
    flux_cal : dict, optional
        dict of flux calibrated targets

    Returns
    -------
    str : units
    """
    # targets with these tags are implicitly flux-calibrated
    # by the pipeline if a flux model is provided
    implicit_tags = ['bpcal', 'bfcal']

    implicit_gain = any([t in target.tags for t in implicit_tags])
    model_key = target.name + '_model'
    is_model = model_key in av_corr
    is_flux = target.name in flux_cal.keys()

    if (implicit_gain and is_model) or is_flux:
        units = 'Jy'
    else:
        units = 'Counts'
    return units


def metadata(ts, capture_block_id, report_path, run, st=None):
    """Create a dictionary with required metadata.

    Parameters
    ----------
    ts : : class:`katsdptelstate.TelescopeState`
        telescope state
    capture_block_id : int
        capture_block_id
    report_path : string
        path where report is written
    run : int
        server index
    st : float, optional
        start time, seconds

    Returns
    -------
    dict
    """
    telstate_cb = ts.root().view(capture_block_id)
    obs_params = to_str(telstate_cb['obs_params'])
    metadata = {}
    product_type = {}
    product_type['ProductTypeName'] = 'MeerKATReductionProduct'
    product_type['ReductionName'] = 'Calibration Report'
    metadata['ProductType'] = product_type
    # format time as required
    time = datetime.datetime.utcfromtimestamp(st)
    metadata['StartTime'] = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata['CaptureBlockId'] = capture_block_id
    metadata['Description'] = obs_params['description'] + ' cal report'
    metadata['ProposalId'] = obs_params['proposal_id']
    metadata['Observer'] = obs_params['observer']
    metadata['ScheduleBlockIdCode'] = obs_params['sb_id_code']
    metadata['Run'] = run

    return metadata


def write_summary(report, ts, stream_name, parameters, targets, st=None, et=None):
    """Write observation summary information to report.

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
    targets : list of str
        Target description strings (without duplicates)
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
    target_names = list(set(katpoint.Target(target).name for target in targets))
    for target in target_names:
        report.writeln('* {0:s}'.format(target))
    if not target_names:
        report.writeln('* Unknown')

    report.writeln()


def write_table_timerow(report, colnames, times, data):
    """Write RST style table to report, rows: time, columns: antenna.

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


def write_table_timecol(report, antenna_names, times, data, ave=False):
    """Write RST style table to report, rows: antenna, columns: time.

    Parameters
    ----------
    report : file-like
        report file to write to
    antenna_names : list
        list of antenna names
    times : list
        list of times (equates to number of columns in the table)
    data : :class:`np.ndarray`
        table data, shape (time, antenna)
    ave : bool
        if True write the median values in each column, else don't
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
        # highlight reference antenna in green
        if 'refant' in a:
            report.write_color(a, 'green', col_width + 1)
            for di in d:
                report.write_color("{:.3f}".format(di.real,), 'green', col_width + 1)
        else:
            report.write(a.ljust(col_width + 1))
            for di in d:
                # highlight NaN solutions in red
                if np.isnan(di):
                    report.write_color("{:.3f}".format(di.real,), 'red', col_width + 1)
                else:
                    report.write(" {:<{}.3f}".format(di, col_width + 1))
        report.writeln()

    if ave:
        report.write("MEDIAN".ljust(col_width + 1))
        for di in data:
            report.write(" {:<{}.3f}".format(np.nanmedian(di), col_width + 1))
        report.writeln()

    # table footer
    report.writeln(col_header * n_entries)
    report.writeln()


def write_flux_cal(report, flux_cal, flux_std, parameters, targets, telstate):
    """Write Flux calibration info in a table.

    Info only gets written if the 'measured_flux' dict in telstate
    isn't empty

    Parameters
    ----------
    report : file-like
        report file to write to
    flux_cal : dict
        target names and their measured flux
    flux_std : dict
        target names and their measured flux errors
    parameters : dict
        pipeline parameters
    targets : list of :class:`katpoint.Target`
        targets in the observation
    telstate : :class:`katsdptelstate.TelescopeState`
        telescope state
    """

    if flux_cal:
        g_bchan = parameters['g_bchan']
        g_echan = parameters['g_echan']
        freqs = parameters['channel_freqs_all'][g_bchan:g_echan + 1] / 1e6

        report.write_heading_2('Flux Calibration')
        flux_desc = '{:.2f} - {:.2f} MHz'.format(freqs[0], freqs[-1])
        report.writeln('Frequency range of quoted flux: **{}**'.format(flux_desc))

        # create table header
        header = ['Calibrator', 'Measured Flux', 'Model Flux']
        n_entries = len(header)
        col_width = 30
        col_header = '=' * col_width + ' '
        header = ' '.join(['{:<{}}'.format(h, col_width) for h in header])

        # write table header
        report.writeln()
        report.writeln(col_header * n_entries)
        report.writeln(header)
        report.writeln(col_header * n_entries)

        for cal in targets:
            name = cal.name
            tags = cal.tags

            # if model flux is available report it
            model_key = name + '_model_flux'
            if model_key in telstate:
                flux_density = telstate.get(model_key)
                model_desc = '{:.3f} Jy'.format(flux_density)

            else:
                model_desc = ''

            if name in flux_cal.keys():
                flux_desc = '{:.3f} +/- {:.3f} Jy'.format(flux_cal[name], flux_std[name])
                for entry in [name, flux_desc, model_desc]:
                    report.write('{:<{}}'.format(entry, col_width + 1))
                report.writeln()
            elif 'bpcal' in tags:
                flux_desc = ''
                for entry in [name, flux_desc, model_desc]:
                    report.write('{:<{}}'.format(entry, col_width + 1))
                report.writeln()

        # write table footer
        report.writeln(col_header * n_entries)
        report.writeln()


def write_elevation(report, report_path, targets, antennas, refant_index, av_corr):
    """Put the elevation vs time plot in the report.

    Parameters
    ----------
    report : file-like
        report file to write to
    report_path : str
        path where report is written
    targets : list
        list of unique targets
    antennas : list of :class:`katpoint.Antenna`
        list of antennas
    refant_index : int
        reference antenna index
    av_corr : dict
        dictionary containing averaged, corrected data
    """
    if refant_index is not None:
        antenna = antennas[refant_index]
    else:
        antenna = antennas[0]

    ts, el, names = [], [], []
    for cal in targets:
        ts_cal = [ti[0] for ti, t in zip(av_corr['timestamps'], av_corr['targets']) if t[0] == cal]
        ts_flat = np.array([x for y in ts_cal for x in y])
        el_cal = calc_elevation(antenna, ts_flat, cal)

        ts.append(ts_flat)
        names.append(katpoint.Target(cal).name)
        el.append(el_cal)

    plot_title = 'Elevation vs Time for Antenna: {0}'.format(antenna.name)
    plot = plotting.plot_el_v_time(names, ts, el, title=plot_title)
    insert_fig(report_path, report, plot, name='El_v_time')


def write_flag_summary(report, report_path, av_corr, dist, correlator_freq, pol=[0, 1]):
    """Write the RFI summary.

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
    n_times = np.array([len(_[0]) for _ in av_corr['timestamps']], dtype=np.float32)
    n_times = np.sum(n_times)
    tw_flags_ave = 100 * av_corr['t_flags'][0] / n_times

    # Sort baseline by antenna separation
    idx_sort = np.argsort(dist)
    tw_flags_sort = tw_flags_ave[:, :, idx_sort]

    # Get channel index in correlator channels
    n_av_chan = tw_flags_ave.shape[-3]
    idx_chan, freq_chan = get_freq_info(correlator_freq, n_av_chan)
    freq_range = [freq_chan[0], freq_chan[-1]]

    # Plot fraction of time flagged vs chan
    plot = plotting.flags_bl_v_chan(tw_flags_sort, idx_chan, dist[idx_sort], freq_range, pol=pol)
    insert_fig(report_path, report, plot, name='Flags_bl_v_chan')

    report.writeln('Percentage of baselines flagged per scan')
    # Average % of baselines flagged per scan
    bl_flags, bl_times = zip(*av_corr['bl_flags'])
    bl_flags = 100 * np.stack(bl_flags)
    target_names = [katpoint.Target(_[0]).name for _ in av_corr['targets']]
    plot = plotting.flags_t_v_chan(bl_flags, idx_chan, target_names, freq_range, pol=pol)
    insert_fig(report_path, report, plot, name='Flags_s_v_chan')
    report.writeln()


def write_hv(report, report_path, targets, av_corr,
             refant_name, antenna_names, correlator_freq,
             auto=True, pol=[0, 1]):
    """Include plots of the delay-corrected phases of the cross-pol data.

    Parameters
    ----------
    report : file-like
        report file to write to
    report_path : str
        path where report is written
    targets: list of :class:`katpoint.Target`
        list of targets with cross pol data
    av_corr : dict
        dictionary of averaged corrected data from which to
        select target data
    refant_name : str
        reference antenna name
    antenna_names : list
        list of antenna names
    correlator_freq : :class:`np.ndarray`
        real (nchan) correlator channel frequencies
    auto : bool, optional
        if True plot phases of auto-corrs else cross-corrs
    pol : list
        description of polarisation axes, optional
    """
    if auto:
        heading3 = 'Cross Hand Auto-correlations, all antennas'
        key = 'auto_cross'
    else:
        heading3 = 'Cross Hand Cross-correlations,\
                    Baselines to the reference antenna : {0}'.format(refant_name)
        key = 'cross'

    report.write_heading_1(
        'Calibrated Cross Hand Phase')
    report.write_heading_2(
        'Corrected Phase vs Frequency')

    for cal in targets:
        kat_target = katpoint.Target(cal)
        target_name = kat_target.name
        tags = [t for t in kat_target.tags if t in TAG_WHITELIST]

        report.write_heading_3(heading3)
        # Get cross hand auto-correlation data
        av_data, av_times = zip(*av_corr['{}_{}'.format(target_name, key)])
        av_data = np.stack(av_data)

        # Get channel index in correlator channels
        n_av_chan = av_data.shape[-3]
        idx_chan, freq_chan = get_freq_info(correlator_freq, n_av_chan)
        freq_range = [freq_chan[0], freq_chan[-1]]

        for ti in range(av_data.shape[0]):
            report.writeln()
            t = utc_tstr(av_times[ti])
            report.writeln('Time : {0}'.format(t))
            for idx in range(0, av_data.shape[-1], ANT_CHUNKS):
                if auto:
                    plot_title = 'Cross Hand Phase vs Frequency'
                    plot_name = 'HV_v_Freq_{0}_{1}'.format(ti, idx)
                else:
                    plot_title = 'Calibrator: {0}, tags are: {1}'.format(
                        target_name, ', '.join(tags))
                    plot_name = 'HV_v_Freq_{0}_ti_{1}_{2}'.format(
                        target_name.replace(' ', '_'), ti, idx)

                data = av_data[ti, ..., idx : idx + ANT_CHUNKS]
                plot = plotting.plot_phaseonly_spec(
                    data, idx_chan, antenna_names[idx : idx + ANT_CHUNKS],
                    freq_range, plot_title, pol=pol)

                insert_fig(report_path, report, plot,
                           name=plot_name)
            report.writeln()


def write_ng_freq(report, report_path, targets, av_corr,
                  refant_name, antenna_names, correlator_freq, pol=[0, 1]):
    """Include plots of spectra of calibrators without applied gains.

    These calibrators do not have gains applied by the pipeline. Make one plot
    per calibrator scan. The plots will only show baselines to reference antenna.

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
    refant_name : str
        name of reference antenna
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
            'Baselines to the reference antenna : {0}'.format(refant_name))

    for cal in targets:
        kat_target = katpoint.Target(cal)
        target_name = kat_target.name
        tags = [t for t in kat_target.tags if t in TAG_WHITELIST]

        # Retrieve visibilities on baselines to the reference antenna
        ant_data, av_times = zip(*av_corr['{}_nog_spec'.format(target_name)])
        ant_data = np.stack(ant_data)
        logger.info(' Corrected data for {0} shape: {1}'.format(target_name, ant_data.shape))

        # Get channel index in correlator channels
        n_av_chan = ant_data.shape[-3]
        idx_chan, freq_chan = get_freq_info(correlator_freq, n_av_chan)
        freq_range = [freq_chan[0], freq_chan[-1]]

        units = _get_units(kat_target, av_corr)

        model_key = kat_target.name + '_model'
        if model_key in av_corr.keys():
            flux_density = np.abs(av_corr[model_key][0])
        else:
            flux_density = None

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
                    freq_range=freq_range, title=plot_title, pol=pol, amp_model=flux_density,
                    units=units)
                insert_fig(report_path, report, plot, name='Corr_v_Freq_{0}_ti_{1}_{2}'.format(
                    target_name.replace(' ', '_'), ti, idx))
                report.writeln()


def write_g_freq(report, report_path, flux_cal, targets, av_corr, antenna_names,
                 cal_bls_lookup, correlator_freq, is_calibrator=True, pol=[0, 1]):
    """Include plots of spectra of calibrators with applied gains.

    Include plots of spectra of calibrators which have gains applied
    by the pipeline. Average all scans on each target into a single plot.

    Parameters
    ----------
    report : file-like
        report file to write to
    report_path : str
        path where report is written
    flux_cal : dict
        target names and their measured flux
    targets : list of str
        targets to plot
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

    for cal in targets:
        kat_target = katpoint.Target(cal)
        target_name = kat_target.name
        tags = [t for t in kat_target.tags if t in TAG_WHITELIST]

        # Get averaged spectrum for gain calibrated targets
        av_data, av_flags, av_weights = da.compute(*av_corr['{0}_g_spec'.format(target_name)][0])
        av_data[av_flags] = np.nan
        # flux calibrate gain calibrators
        if target_name in flux_cal.keys():
            flux = flux_cal[target_name]
            av_data *= flux
        logger.info(' Corrected data for {0} shape: {1}'.format(target_name, av_data.shape))

        # Get channel index in correlator channels
        n_av_chan = av_data.shape[-3]
        idx_chan, freq_chan = get_freq_info(correlator_freq, n_av_chan)
        freq_range = [freq_chan[0], freq_chan[-1]]

        units = _get_units(kat_target, av_corr)
        model_key = kat_target.name + '_model'
        if model_key in av_corr.keys():
            flux_density = np.abs(av_corr[model_key][0])
        else:
            flux_density = None

        # Set the plot label
        if is_calibrator:
            plot_title = 'Calibrator: {0} , tags are {1}'.format(target_name, ', '.join(tags))
            amp = False
        else:
            plot_title = 'Target: {0}'.format(target_name)
            amp = True

        # turn flagged data into NaN's so it doesn't appear in the plots
        av_data[av_data == 0] = np.nan
        # Only plot a maximum of 16 antennas per plot
        for idx in range(0, av_data.shape[-1], ANT_CHUNKS):
            data = av_data[..., idx : idx + ANT_CHUNKS]
            plot = plotting.plot_spec(
                data, idx_chan, antenna_names[idx : idx + ANT_CHUNKS],
                freq_range, plot_title, amp=amp, pol=pol, amp_model=flux_density, units=units)

            insert_fig(report_path, report, plot,
                       name='Corr_v_Freq_{0}_{1}'.format(target_name.replace(" ", "_"), idx))
            report.writeln()


def write_g_time(report, report_path, flux_cal, targets, av_corr,
                 antenna_names, cal_bls_lookup, correlator_freq, pol):
    """Plots of amp and phase versus time of all scans of the given targets.

    The plots show data averaged per antenna.

    Parameters
    ----------
    report : file-like
        report file to write to
    report_path : str
        path where report is written
    flux_cal : dict
        target names and their measured flux
    targets : list of :class: `katpoint.Target`
        targets to plot
    av_corr : dict
        dictionary of averaged corrected data
    antenna_names : list
        list of antenna names
    cal_bls_lookup : :class:`np.ndarray`
        array of antenna indices in each baseline
    correlator_freq : :class:`np.ndarray`
        real (nchan), correlator channel frequencies
    pol : list
        description of polarisation axes, optional
    """
    # Get all scans of calibrators which have gains applied by the pipeline.
    if len(targets) > 0:

        report.write_heading_2(
            'Corrected Phase vs Time, all gain-calibrated calibrators')
        report.writeln()
        report.write_heading_3('All baselines, averaged per antenna')
        report.writeln()

        data, times, units = [], [], []
        for cal in targets:
            kat_target = katpoint.Target(cal)
            target_name = kat_target.name
            av_data, av_times = zip(*av_corr[target_name + '_g_phase'])

            # flux calibrate if possible
            if target_name in flux_cal.keys():
                av_data = [d * flux_cal[target_name] for d in av_data]
            data += av_data
            times += av_times
            units.append(_get_units(kat_target, av_corr, flux_cal))

        if all([unit == 'Jy' for unit in units]):
            units = 'Jy'
        else:
            units = 'arb'

        av_data = np.stack(data, axis=0)
        # insert plots of phase v time
        for idx in range(0, av_data.shape[-1], ANT_CHUNKS):
            plot = plotting.plot_corr_v_time(
                times, av_data[..., idx : idx + ANT_CHUNKS],
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
            plot = plotting.plot_corr_v_time(times,
                                             av_data[..., idx : idx + ANT_CHUNKS], plottype='a',
                                             antenna_names=antenna_names[idx : idx + ANT_CHUNKS],
                                             pol=pol, units=units)

            insert_fig(report_path, report, plot, name='Amp_v_Time_{0}'.format(idx))
            report.writeln()


def write_g_uv(report, report_path, flux_cal, targets, av_corr, cal_bls_lookup,
               antennas, cal_array_position, correlator_freq,
               is_calibrator=True, pol=[0, 1]):
    """Include plots of amp and phase/amp versus uvdist in report.

    The data is averaged in frequency to the number of channels given by
    PLOT_CHANNELS.

    Parameters
    ----------
    report : file-like
        report file to write to
    report_path : str
        path where report is written
    flux_cal : dict
        target names and their measured flux
    targets : list of str
        targets to plot
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

    # Plot vs UV distance for targets with gains applied by the pipeline.
    for cal in targets:
        kat_target = katpoint.Target(cal)
        target_name = kat_target.name
        tags = [t for t in kat_target.tags if t in TAG_WHITELIST]

        # Get averaged data on all baselines
        av_data, av_times = zip(*av_corr['{}_g_bls'.format(target_name)])
        av_data = np.stack(av_data)
        # flux calibrate gain calibrators
        if target_name in flux_cal.keys():
            flux = flux_cal[target_name]
            av_data *= flux
        logger.info(' Corrected data for {0} shape: {1}'.format(target_name, av_data.shape))

        # Get channel index in correlator channels
        nchan = av_data.shape[-3]
        idx_chan, freq_chan = get_freq_info(correlator_freq, nchan)
        freq_chan = freq_chan * 1e6
        uvdist = calc_uvdist(cal, freq_chan, av_times,
                             cal_bls_lookup, antennas, cal_array_position)

        units = _get_units(kat_target, av_corr, flux_cal)
        if is_calibrator:
            plot_title = 'Calibrator {0}, tags are {1}'.format(target_name, ', '.join(tags))
            amp = False
        else:
            plot_title = 'Target {0}'.format(target_name)
            amp = True

        plot = plotting.plot_corr_uvdist(uvdist, av_data, freq_chan,
                                         plot_title, amp=amp, pol=pol, units=units)
        insert_fig(report_path, report, plot,
                   name='Corr_v_UVdist_{0}'.format(target_name.replace(" ", "_")))
        report.writeln()


def write_SNR_products(report, report_path, ts, parameters,
                       st, et, antenna_names, pol=[0, 1]):
    """Include calibration product SNR plots in the report.

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
    pol : list, optional
        description of polarisation axes
    """
    snr_list = ['SNR_K', 'SNR_B', 'SNR_G']
    product_names = parameters['product_names']
    solns_exist = any([product_names[snr] in ts for snr in snr_list])
    if not solns_exist:
        logger.info(' - no solution SNRs')

    else:
        all_times = []
        all_snr = []
        all_labels = []

        for s in snr_list:
            vals, times = get_cal(ts, s, product_names[s], st, et)
            if len(times) > 0:
                all_times.append(times)
                all_snr.append(vals)
                all_labels.append(s)

        report.write_heading_1('Calibration product SNR summary')
        report.writeln()
        report.writeln('Median and IQR of solution SNRs across antennas')

        plot = plotting.plot_snr(plotting.draw_errorplot, all_times, all_snr, all_labels,
                                 pol=pol)
        insert_fig(report_path, report, plot, name='SNR_v_time')

        report.writeln()
        report.writeln('Number of antennas with SNR < 10 (or NaN) solutions:')
        nonan_snr = [np.where(np.isnan(s), 0, s) for s in all_snr]
        low_exist = any([np.any(s < 10) for s in nonan_snr])
        if low_exist:
            plot = plotting.plot_snr(plotting.draw_below_thresh, all_times, all_snr,
                                     all_labels, pol=pol, yscale=0.5)
            insert_fig(report_path, report, plot, name='SNR_low')
        else:
            report.writeln()
            report.writeln('* None')
            report.writeln()


def write_products(report, report_path, ts, parameters,
                   st, et, antenna_names, correlator_freq, pol=[0, 1]):
    """Include calibration product plots in the report.

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
    cal_list = ['K', 'KCROSS', 'KCROSS_DIODE', 'B', 'G']
    product_names = parameters['product_names']
    solns_exist = any([product_names[cal] in ts for cal in cal_list])
    if not solns_exist:
        logger.info(' - no calibration solutions')

    # delay
    cal = 'K'
    vals, times = get_cal(ts, cal, product_names[cal], st, et)
    if len(times) > 0:
        cal_heading(report, cal, 'Delay', '(ns)')
        write_K(report, report_path, times, vals, antenna_names, pol)

    # ---------------------------------
    # cross pol delay
    cal = 'KCROSS'
    vals, times = get_cal(ts, cal, product_names[cal], st, et)
    if len(times) > 0:
        cal_heading(report, cal, 'Cross polarisation delay', '(ns)')
        # convert delays to nano seconds
        vals = 1e9 * vals
        write_table_timecol(report, [cal], times, vals[:, 0, :])

    # ---------------------------------
    # cross pol delay from noise diode
    cal = 'KCROSS_DIODE'
    vals, times = get_cal(ts, cal, product_names[cal], st, et)
    if len(times) > 0:
        cal_heading(report, cal, 'Cross polarisation delay {0}'.format(pol[0] + pol[1]), '(ns)')
        # convert delays to nano seconds
        vals = 1e9 * vals
        write_table_timecol(report, antenna_names, times, vals[:, 0, :], True)
        for idx in range(0, vals.shape[-1], ANT_CHUNKS):
            plot = plotting.plot_delays(times, vals[:, 0:1, idx : idx + ANT_CHUNKS],
                                        antenna_names[idx : idx + ANT_CHUNKS],
                                        pol=[pol[0] + pol[1]])
            insert_fig(report_path, report, plot, name='{0}_{1}'.format(cal, idx))

    # plot number of solutions
        report.writeln()
        report.writeln()
        title = 'Number of slns : {0}'.format(cal)
        no_k_slns = np.sum(~np.isnan(vals), axis=0, dtype=np.uint16)
        plot = plotting.plot_v_antenna(no_k_slns, 'No of slns: {0}'.format(cal), title,
                                       antenna_names, pol)
        insert_fig(report_path, report, plot, name='No_{0}'.format(cal))

    cal = 'BCROSS_DIODE'
    vals, times = get_cal(ts, cal, product_names[cal], st, et)
    if len(times) > 0:
        cal_heading(report, cal, 'Cross polarisation phase {0}'.format(pol[0] + pol[1]))
        write_BCROSS_DIODE(report, report_path, times, vals, antenna_names,
                           correlator_freq, pol)

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
        # summarize bad antennas
        report.writeln('Antennas flagged for all times:')
        report.writeln()
        antenna_labels = write_bad_antennas(report, vals, antenna_names, pol)
        # plot solns
        for idx in range(0, vals.shape[-1], ANT_CHUNKS):
            plot = plotting.plot_g_solns_legend(
                times, vals[..., idx: idx + ANT_CHUNKS],
                antenna_labels[idx: idx + ANT_CHUNKS], pol)
            insert_fig(report_path, report, plot, name='{0}_{1}'.format(cal, idx))

        # plot number of solutions
        report.writeln()
        report.writeln()
        title = 'Number of slns : {0}'.format(cal)
        no_slns = np.sum(~np.isnan(vals), axis=0, dtype=np.uint16)
        plot = plotting.plot_v_antenna(no_slns, 'No of slns: {0}'.format(cal), title,
                                       antenna_names, pol)
        insert_fig(report_path, report, plot, name='No_{0}'.format(cal))


def get_cal(ts, cal, ts_name, st, et):
    """Fetch a calibration product from telstate.

    Parameters
    ----------
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

    Returns
    -------
    vals : :class:`np.ndarray`
        values of calibration product
    times : list
        times of calibration product
    """
    vals, times = np.array([]), []
    if ts_name in ts:
        product = ts.get_range(ts_name, st=0)
        if len(product) > 0:
            logger.info('Calibration product: {0}'.format(cal))
            vals, times = zip(*product)
            vals = np.array(vals)
            logger.info('  shape: {0}'.format(vals.shape))
    return vals, times


def write_K(report, report_path, times, vals, antenna_names, pol=[0, 1]):
    """Include table of delays and delay plots in cal report.

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
        report.writeln('**POL {0}**'.format(pol[p],))
        kpol = vals[:, p, :]
        logger.info('  pol {0} shape: {1}'.format(pol[p], kpol.shape))
        write_table_timecol(report, antenna_names, times, kpol)

    for idx in range(0, vals.shape[-1], ANT_CHUNKS):
        plot = plotting.plot_delays(times, vals[..., idx : idx + ANT_CHUNKS],
                                    antenna_names[idx : idx + ANT_CHUNKS], pol=pol)
        insert_fig(report_path, report, plot, name='K_{0}'.format(idx))

    # plot number of solutions
    report.writeln()
    report.writeln()
    cal = 'K'
    title = 'Number of slns : {0}'.format(cal)
    no_slns = np.sum(~np.isnan(vals), axis=0, dtype=np.uint16)
    plot = plotting.plot_v_antenna(no_slns, 'No of slns: {0}'.format(cal), title,
                                   antenna_names, pol)
    insert_fig(report_path, report, plot, name='No_{0}'.format(cal))


def write_bad_antennas(report, vals, antenna_names, pol=[0, 1]):
    """Write list of bad antennas to the report.

    Write a bulleted list of completely NaN'ed antennas per polarization.
    Return a list with bad-p appended to the antenna-names for NaN'ed antennas

    Parameters
    ----------
    report : file-like
        report file to write to
    vals : array
        solutions
    antenna_names : list
        list of antenna names
    list :
        description of polarisation axes, optional

    Returns
    -------
    list :
        list of antenna_names with bad(pol) appended to flagged antennas
    """
    bad_labels = antenna_names[:]
    # write a list of bad antennas per polarisation, if all/none are flagged label as such
    for p in range(vals.shape[-2]):
        bad_ant = np.all(np.isnan(vals[:, p, :]), axis=0)
        if bad_ant.any():
            if bad_ant.all():
                report.writeln('* {0}: :red:`all`'.format(pol[p]))
            else:
                names = np.asarray(antenna_names)[bad_ant]
                report.writeln('* {0}: {1}'.format(pol[p], ', '.join(names)))
        else:
            report.writeln('* {0}: None'.format(pol[p]))
            report.writeln()

        # modify the antenna labels to indicate the flagged antennas
        for i, bad in enumerate(bad_ant):
            if bad:
                if 'bad' in bad_labels[i]:
                    bad_labels[i] += pol[p]
                else:
                    bad_labels[i] = bad_labels[i].strip() + ', bad {}'.format(pol[p])
    return bad_labels


def write_B(report, report_path, times, vals, antenna_names, correlator_freq, pol=[0, 1]):
    """Include plots of bandpass solutions at all given times in report.

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
    correlator_freq : :class:`np.ndarray`
        array of correlator channel frequencies
    pol : list
        description of polarisation axes, optional
    """
    # B shape is n_time, n_chan, n_pol, n_ant
    freq_range = [correlator_freq[0], correlator_freq[-1]]
    chan_no = np.arange(0, len(correlator_freq))

    for ti in range(len(times)):
        t = utc_tstr(times[ti])
        report.writeln('Time: {}'.format(t,))
        report.writeln()
        # summarize bad antennas
        report.writeln('Antennas flagged for all channels:')
        report.writeln()
        antenna_labels = write_bad_antennas(report, vals[ti], antenna_names, pol)

        # plot slns
        for idx in range(0, vals[ti].shape[-1], ANT_CHUNKS):
            plot = plotting.plot_spec(
                vals[ti, ..., idx : idx + ANT_CHUNKS], chan_no,
                antenna_labels[idx : idx + ANT_CHUNKS],
                freq_range, pol=pol)
            insert_fig(report_path, report, plot,
                       name='B_ti_{0}_{1}'.format(ti, idx))

    # plot number of solutions
    report.writeln()
    report.writeln()
    cal = 'B'
    title = 'Number of slns : {0}'.format(cal)
    b_slns = ~np.all(np.isnan(vals), axis=1)
    no_slns = np.sum(b_slns, axis=0, dtype=np.uint32)
    plot = plotting.plot_v_antenna(no_slns, 'No of slns: {0}'.format(cal), title,
                                   antenna_names, pol)
    insert_fig(report_path, report, plot, name='No_{0}'.format(cal))


def write_BCROSS_DIODE(report, report_path, times, vals, antenna_names, correlator_freq, pol):
    """Include plots of bcross_diode solutions at all given times in report.

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
    correlator_freq : :class:`np.ndarray`
        array of correlator channel frequencies
    pol : list
        description of polarisation axes, optional
    """
    freq_range = [correlator_freq[0], correlator_freq[-1]]
    chan_no = np.arange(0, len(correlator_freq))
    for ti in range(len(times)):
        t = utc_tstr(times[ti])
        report.writeln('Times: {}'.format(t,))
        report.writeln()
        # summarize bad antenn
        report.writeln('Antennas flagged for all channels:')
        report.writeln()
        antenna_labels = write_bad_antennas(report, vals[ti], antenna_names, pol)

        for idx in range(0, vals.shape[-1], ANT_CHUNKS):
            plot = plotting.plot_phaseonly_spec(
                vals[ti, :, 0:1, idx : idx + ANT_CHUNKS], chan_no,
                antenna_labels[idx : idx + ANT_CHUNKS],
                freq_range, pol=[pol[0]+pol[1]])
            insert_fig(report_path, report, plot,
                       name='BCROSS_DIODE_ti_{0}_{1}'.format(ti, idx))

    # plot number of solutions
    report.writeln()
    report.writeln()
    cal = 'BCROSS_DIODE'
    title = 'Number of slns : {}'.format(cal)
    b_slns = ~np.all(np.isnan(vals), axis=1)
    no_slns = np.sum(b_slns, axis=0, dtype=np.uint32)
    plot = plotting.plot_v_antenna(no_slns, 'No of slns: {0}'.format(cal), title,
                                   antenna_names, pol)
    insert_fig(report_path, report, plot, name='No_{0}'.format(cal))


def cal_heading(report, cal, prefix, suffix=''):
    """Write cal pipeline product headings.

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
    """Channel indices and frequencies of averaged frequency channels.

    Given nchan averaged channels calculate the channel index (in correlator
    channels) and the frequencies of the averaged channels.

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


def split_targets(targets):
    """Split list of targets into groups depending on calibration properties.

    Separate targets into four lists containing calibrators without gains
    applied, calibrators with gains applied, polarisation calibrators and targets.

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
    pol : list
        list of calibrators with 'polcal' tag
    target : list
        list of targets
    """
    nogain, gain, pol, target = [], [], [], []
    for cal in targets:
        kat_target = katpoint.Target(cal)
        tags = kat_target.tags
        # tags which have gains applied by pipeline
        gaintaglist = ('gaincal', 'bfcal')
        nogaintaglist = ('bpcal', 'delaycal')
        if any(x in nogaintaglist for x in tags):
            nogain.append(cal)
        if any(x in gaintaglist for x in tags):
            gain.append(cal)
        if 'polcal' in tags:
            pol.append(cal)
        # if a target is a calibrator, don't include it here as it will already be included
        # in the calibrator plots
        if ('target' in tags and
                not any(x in nogaintaglist + gaintaglist for x in tags)):
            target.append(cal)
    return nogain, gain, pol, target


def calc_elevation(refant, times, target):
    """Calculates elevation versus timestamps for observation targets.

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
    azel = kat_target.azel(times, refant)
    elevations = azel.alt.rad

    return elevations


def calc_uvdist(target, freq, times, cal_bls_lookup, antennas, cal_array_position):
    """Calculate uvdistance in wavelengths.

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
    wl = (const.c / (freq * u.Hz)).to_value(u.m)
    cross_idx = np.where(cal_bls_lookup[:, 0] !=
                         cal_bls_lookup[:, 1])[0]
    kat_target = katpoint.Target(target)
    uvw = calprocs.calc_uvw_wave(kat_target, times, cal_bls_lookup[cross_idx],
                                 antennas, wl, cal_array_position)
    uvdist = np.hypot(uvw[0], uvw[1])
    return uvdist


def calc_enu_sep(antennas, bls_lookup):
    """Calculate baseline separation in meters for cross correlations only.

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
        bl[i] = enu.xyz.to_value(u.m)

    sep = np.linalg.norm(bl, axis=1)

    return sep


def refant_antlabels(bls_lookup, refant_index, antenna_names):
    """Antenna names in the baselines to the reference antenna.

    Parameters
    ----------
    bls_lookup : :class:`np.ndarray`
        array of indices of antennas in each baseline
    refant_index : int
        index of reference antenna
    antenna_names : list of str
        list of antenna names

    Returns
    -------
    list of str:
        list of antenna names
    """
    ant_idx = np.where(
        (bls_lookup[:, 0] == refant_index)
        ^ (bls_lookup[:, 1] == refant_index))[0]

    refant_bls = bls_lookup[ant_idx]
    bls_names = []
    for bls in refant_bls:
        for idx in bls:
            if idx != refant_index:
                bls_names.append(antenna_names[idx])
    return bls_names


# This is required as pylab is not multithreading safe, it only required to run
# unit tests which run several servers in a single process. Real use runs the
# servers in separate processes.
_lock = threading.Lock()


def make_cal_report(ts, capture_block_id, stream_name, parameters, report_path, av_corr,
                    st=None, et=None):
    """Create calibration report.

    Creates HTML calibration pipeline report (from RST source), using data from
    the Telescope State.

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
    report_file = os.path.join(report_path, 'calreport{}.rst'.format(parameters['server_id']+1))
    # --------------------------------------------------------------------
    # write heading
    with _lock:
        with rstReport(report_file) as cal_rst:
            cal_rst.write_heading_0('Calibration pipeline report')

            # --------------------------------------------------------------------
            # write observation summary info
            cal_rst.writeln('.. role:: red')
            cal_rst.writeln('.. role:: green')
            cal_rst.writeln()
            cal_rst.write_heading_1('Observation summary')
            cal_rst.writeln('Capture block: {}'.format(capture_block_id))
            cal_rst.writeln()
            cal_rst.writeln('Stream: {}'.format(stream_name))
            cal_rst.writeln()

            antennas = parameters['antennas']
            if av_corr:
                targets, times = zip(*av_corr['targets'])
                unique_targets = list(set(targets))
            else:
                unique_targets = []
            write_summary(cal_rst, ts, stream_name, parameters, unique_targets, st=st, et=et)
            metadata_dict = metadata(ts, capture_block_id, report_path,
                                     parameters['server_id'] + 1, st)

            # get parameters
            correlator_freq = parameters['channel_freqs'] / 1e6
            cal_bls_lookup = parameters['bls_lookup']
            pol = [_[0].upper() for _ in parameters['pol_ordering']]
            antenna_names = list(parameters['antenna_names'])

            # Obtain reference antenna selected by the pipeline
            refant_name = ts.get('refant')
            if refant_name is not None:
                refant_index = antenna_names.index(refant_name)
                parameters['refant'] = refant_name
                parameters['refant_index'] = refant_index

                # label the reference antenna in the list of antennas
                antenna_names[refant_index] += ', refant'
                name_width = len(antenna_names[refant_index])
                antenna_names = [name.ljust(name_width) for name in antenna_names]

            else:
                logger.info(' - no reference antenna')
                refant_index = None

            if av_corr:
                write_elevation(cal_rst, report_path, unique_targets,
                                antennas, refant_index, av_corr)
                # -------------------------------------------------------------------
                # write RFI summary
                cal_rst.write_heading_1('RFI and Flagging summary')
                # Plot flags
                dist = calc_enu_sep(antennas, cal_bls_lookup)
                write_flag_summary(cal_rst, report_path, av_corr, dist, correlator_freq, pol)
            else:
                logger.info(' - no calibrated data')

            # write solution SNR summary
            logger.info('Calibration solution SNR summary')
            write_SNR_products(cal_rst, report_path, ts, parameters,
                               st, et, antenna_names, pol)
            logger.info('Calibration solution summary')
            # add cal products to report
            flux_cal = ts.get('measured_flux')
            flux_std = ts.get('measured_flux_std')

            write_flux_cal(cal_rst, flux_cal, flux_std, parameters, unique_targets, ts)
            write_products(cal_rst, report_path, ts, parameters,
                           st, et, antenna_names, correlator_freq, pol)
            # Corrected data
            if av_corr:
                # Split observed targets into different types of sources,
                # according to their pipeline tags
                nogain, gain, polcal, target = split_targets(unique_targets)
                cal_array_position = parameters['array_position']

                # Calibrator data requires a reference antenna
                if refant_index is not None:
                    # Corrected data : HV delay Noise Diode
                    if any(katpoint.Target(target).name + '_auto_cross'
                           in av_corr for target in gain):
                        write_hv(cal_rst, report_path, gain, av_corr, refant_name, antenna_names,
                                 correlator_freq, pol=[pol[0] + pol[1], pol[1] + pol[0]])
                    # --------------------------------------------------------------------
                    # For calibrators which do not have gains applied by the pipeline
                    # plot the baselines to the reference antenna for each timestamp
                    # get idx of baselines to refant
                    bls_names = refant_antlabels(cal_bls_lookup, refant_index, antenna_names)
                    # Corrected data : HV delay on `polcal`
                    if any(katpoint.Target(target).name + '_cross' in av_corr for target in polcal):
                        write_hv(cal_rst, report_path, polcal, av_corr, refant_name, bls_names,
                                 correlator_freq, auto=False,
                                 pol=[pol[0] + pol[1], pol[1] + pol[0]])

                    # Corrected data : Calibrators
                    cal_rst.write_heading_1('Calibrator Summary Plots')

                    write_ng_freq(cal_rst, report_path, nogain, av_corr,
                                  refant_name, bls_names, correlator_freq, pol)
                    write_g_freq(cal_rst, report_path, flux_cal, gain, av_corr, antenna_names,
                                 cal_bls_lookup, correlator_freq, True, pol)
                    write_g_time(cal_rst, report_path, flux_cal, gain, av_corr, antenna_names,
                                 cal_bls_lookup, correlator_freq, pol)
                    write_g_uv(cal_rst, report_path, flux_cal, gain, av_corr, cal_bls_lookup,
                               antennas, cal_array_position, correlator_freq, True,
                               pol=pol)

                # --------------------------------------------------------------------
                # Corrected data : Targets
                cal_rst.write_heading_1('Calibrated Target Fields')
                write_g_freq(cal_rst, report_path, flux_cal, target, av_corr, antenna_names,
                             cal_bls_lookup, correlator_freq, False, pol=pol)
                write_g_uv(cal_rst, report_path, flux_cal, target, av_corr, cal_bls_lookup,
                           antennas, cal_array_position, correlator_freq, False, pol=pol)

            cal_rst.writeln()

        # convert to html
        stylesheet_path = os.path.join(docutils_dir, 'html4css1.css')
        overrides = {'stylesheet_path': stylesheet_path}
        report_file_html = os.path.join(report_path,
                                        'calreport{}.html'.format(parameters['server_id']+1))
        publish_file(source_path=report_file, destination_path=report_file_html,
                     writer_name='html', settings_overrides=overrides)

        # write metadata file
        metadata_file = os.path.join(report_path, 'metadata.json')
        with open(metadata_file, 'w') as cal_meta:
            json.dump(metadata_dict, cal_meta)
