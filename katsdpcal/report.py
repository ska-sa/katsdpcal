
import os
import shutil

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from . import plotting

import numpy as np
import time

from docutils.core import publish_file

import matplotlib.pylab as plt

#--------------------------------------------------------------------------------------------------
#--- CLASS :  rstReport
#--------------------------------------------------------------------------------------------------

class rstReport(file):
    """
    RST style report
    """

    def write_heading(self,heading,symbol):
        heading_len = len(heading)
        self.writeln(symbol*heading_len)
        self.writeln(heading)
        self.writeln(symbol*heading_len)
        self.write('\n')

    def write_heading_0(self,heading):
        self.write_heading(heading,'#')

    def write_heading_1(self,heading):
        self.write_heading(heading,'*')

    def writeln(self,line=None):
        if line is not None: self.write(line)
        self.write('\n')

#--------------------------------------------------------------------------------------------------
#--- FUNCTION :  Report writing functions
#--------------------------------------------------------------------------------------------------

def insert_fig(report,fig,name=None):
    """
    Insert matplotlib figure into report
    
    Parameters
    ----------
    report : open report file to write to
    fig    : matplitlib figure 
    """
    if (name == None):
        name = str(fig)
    figname = "{}.png".format(name,)
    fig.savefig(figname,bbox_inches='tight')
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

def write_bullet_if_present(report,ts,var_text,var_name,transform=None):
    """
    Write bullet point, if TescopeState key is present

    Parameters
    ----------
    report : report file to write to
    ts     : telescope state
    var_text : bullet point description, string
    var_name : telescope state key, string
    transform : transform for applying to TelescopeState value before reporting, optional
    """
    ts_value = ts[var_name] if ts.has_key(var_name) else 'unknown'
    if transform is not None:
        ts_value = transform(ts_value)
    report.writeln('* {0}:  {1}'.format(var_text,ts_value))

def write_summary(report,ts,st=None,et=None):
    """
    Write observation summary information to report

    Parameters
    ----------
    report : report file to write to
    ts     : telescope state
    st     : start time for reporting parameters, seconds, float, optional
    et     : end time for reporting parameters, seconds, float, optional
    """
    # write RST style bulletted list
    report.writeln('* {0}:  {1}'.format('Start time',time.strftime("%x %X",time.gmtime(st))))

    # telescope state values
    write_bullet_if_present(report,ts,'Int time','sdp_l0_int_time')
    write_bullet_if_present(report,ts,'Channels','cbf_n_chans')
    write_bullet_if_present(report,ts,'Polarisation products','cbf_n_pols')
    write_bullet_if_present(report,ts,'Antennas','cal_antlist',transform=len)
    write_bullet_if_present(report,ts,'Antenna list','cal_antlist')
    report.writeln()

    report.writeln('Source list:')
    report.writeln()
    try:
        target_list = ts.get_range('cal_info_sources',st=st,et=et,return_format='recarray')['value'] if ts.has_key('cal_info_sources') else []
        for target in target_list:
            report.writeln('* {0:s}'.format(target,))
    except AttributeError:
        # key not present
        report.writeln('* Unknown')

    report.writeln()    

def write_table_timerow(report,colnames,times,data):
    """
    Write RST style table to report, rows: time, columns: antenna

    Parameters
    ----------
    report   : report file to write to
    colnames : list of column names, list of string
    times    : list of times (equates to number of rows in the table)
    data     : table data, shape (time, columns)
    """
    # create table header
    header = colnames[:]
    header.insert(0,'time')

    n_entries = len(header)
    col_width = 30
    col_header = '='*col_width+' '

    # write table header
    report.writeln()
    report.writeln(col_header*n_entries)
    report.writeln(" ".join([h.ljust(col_width) for h in header])) 
    report.writeln(col_header*n_entries)

    timestrings = [time.strftime("%d %X",time.gmtime(t)) for t in times]

    # add each time row to the table
    for t, d in zip(timestrings,data):
        data_string = " ".join(["{:.3f}".format(di.real,).ljust(col_width) for di in np.atleast_1d(d)])
        report.write("{}".format(t,).ljust(col_width+1))
        report.writeln(data_string)  

    # table footer
    report.writeln(col_header*n_entries)
    report.writeln()

def write_table_timecol(report,antennas,times,data):
    """
    Write RST style table to report, rows: antenna, columns: time

    Parameters
    ----------
    report : report file to write to
    antennas : list of antenna names, comma separated string, or single string of comma separated antenna names
    times    : list of times (equates to number of columns in the table)
    data     : table data, shape (time, antenna)
    """
    n_entries = len(times) + 1
    col_width = 30
    col_header = '='*col_width+' '

    # create table header
    timestrings = [time.strftime("%d %X",time.gmtime(t)) for t in times]
    header = " ".join(["{}".format(t,).ljust(col_width) for t in timestrings])
    header = 'Ant'.ljust(col_width+1) + header

    # write table header
    report.writeln()
    report.writeln(col_header*n_entries)
    report.writeln(header) 
    report.writeln(col_header*n_entries)

    # add each antenna row to the table
    antlist = antennas if isinstance(antennas, list) else antennas.split(',')
    for a, d in zip(antlist,data.T):
        data_string = " ".join(["{:.3f}".format(di.real,).ljust(col_width) for di in d])
        report.write(a.ljust(col_width+1))
        report.writeln(data_string)  

    # table footer
    report.writeln(col_header*n_entries)
    report.writeln()

def make_cal_report(ts,report_path,project_name=None,st=None,et=None):
    """
    Creates pdf calibration pipeline report (from RST source),
    using data from the Telescope State 
    
    Parameters
    ----------
    ts           : TelescopeState
    report_path  : path where report will be created, string
    project_name : ID associated with project, string, optional
    st           : start time for reporting parameters, seconds, float, optional
    et           : end time for reporting parameters, seconds, float, optional
    """

    if project_name == None:
        project_name = '{0}_unknown_project'.format(time.time())

    if not report_path: report_path = '.'
    project_dir = os.path.abspath(report_path)
    logger.info('Report compiling in directory {0}'.format(project_dir,))
    # change into project directory
    os.chdir(project_dir)
    
    # --------------------------------------------------------------------
    # open report file
    report_file = 'calreport_{0}.rst'.format(project_name,)
    cal_rst = rstReport(report_file, 'w')

    # --------------------------------------------------------------------
    # write heading
    cal_rst.write_heading_0('Calibration pipeline report')

    # --------------------------------------------------------------------
    # write observation summary info
    cal_rst.write_heading_1('Observation summary')
    cal_rst.writeln('Observation: {0:s}'.format(project_name,))
    cal_rst.writeln()
    write_summary(cal_rst,ts,st=st,et=et)

    # --------------------------------------------------------------------
    # write RFI summary
    cal_rst.write_heading_1('RFI summary')
    cal_rst.writeln('Watch this space')
    cal_rst.writeln()

    # --------------------------------------------------------------------    
    # add cal products to report
    antenna_mask = ts.antenna_mask

    logger.info('Calibration solution summary')
    cal_list = ['K','KCROSS','B','G']
    solns_exist = any(['cal_product_'+cal in ts.keys() for cal in cal_list])
    if not solns_exist:
        logger.info(' - no calibration solutions')

    # ---------------------------------
    # delay
    cal = 'K'
    cal_product = 'cal_product_'+cal
    if ts.has_key(cal_product):
        product = ts.get_range(cal_product,st=st,et=et,return_format='recarray')
        if len(product['time']) > 0:
            logger.info('Calibration product: {0}'.format(cal,))

            cal_rst.write_heading_1('Calibration product {0:s}'.format(cal,))
            cal_rst.writeln('Delay calibration solutions ([ns])')
            cal_rst.writeln()

            vals = product['value']
            # K shape is n_time, n_pol, n_ant
            times = product['time']

            # convert delays to nano seconds
            vals = 1e9*vals

            logger.info('  shape: {0}'.format(vals.shape,))

            # iterate through polarisation
            for pi in range(vals.shape[-2]):
                cal_rst.writeln('**POL {0}**'.format(pi,))
                kpol = vals[:,pi,:]
                logger.info('  pol{0} shape: {1}'.format(pi,kpol.shape))
                write_table_timecol(cal_rst,antenna_mask,times,kpol)

    # ---------------------------------
    # cross pol delay
    cal = 'KCROSS'
    cal_product = 'cal_product_'+cal
    if ts.has_key(cal_product):
        product = ts.get_range(cal_product,st=st,et=et,return_format='recarray')
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
            vals = 1e9*vals

            write_table_timerow(cal_rst,[cal],times,vals)

    # ---------------------------------
    # bandpass
    cal = 'B'
    cal_product = 'cal_product_'+cal
    if ts.has_key(cal_product):
        product = ts.get_range(cal_product,st=st,et=et,return_format='recarray')
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
                t = time.strftime("%Y %x %X",time.gmtime(times[ti]))
                cal_rst.writeln('Time: {}'.format(t,))
                plot = plotting.plot_bp_solns(vals[ti])
                insert_fig(cal_rst,plot,name='{0}_{1}'.format(cal,str(ti)))

    # ---------------------------------
    # gain
    cal = 'G'
    cal_product = 'cal_product_'+cal
    if ts.has_key(cal_product):
        product = ts.get_range(cal_product,st=st,et=et,return_format='recarray')
        if len(product['time']) > 0:
            logger.info('Calibration product: {0}'.format(cal,))

            cal_rst.write_heading_1('Calibration product {0:s}'.format(cal,))
            cal_rst.writeln('Gain calibration solutions')
            cal_rst.writeln()

            vals = product['value']
            # G shape is n_time, n_pol, n_ant
            times = product['time']

            logger.info('  shape: {0}'.format(vals.shape,))

            # iterate through polarisation
            for pi in range(vals.shape[-2]):
                cal_rst.writeln('**POL {0}**'.format(pi,))
                gpol = vals[:,pi,:]
                logger.info('  pol{0} shape: {1}'.format(pi,gpol.shape))
                plot = plotting.plot_g_solns(times,gpol)
                insert_fig(cal_rst,plot,name='{0}_P{1}'.format(cal,pi))

    # --------------------------------------------------------------------
    # close off report
    cal_rst.writeln()
    cal_rst.writeln()
    cal_rst.close()

    # convert to html
    publish_file(source_path=report_file,destination_path=report_file.replace('rst','html'),writer_name='html')
