
import os
import shutil

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import plotting

import matplotlib.pylab as plt
import numpy as np
import time


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
    figname = "{}.jpeg".format(name if name else str(fig))
    fig.savefig(figname,bbox_inches='tight')
    fig_text = \
    '''.. image:: {}
       :align: center
    '''.format(figname)
    report.writeln()
    report.writeln(fig_text)
    report.writeln()

def write_summary(report,ts):
    """
    Write observation summary information to report

    Parameters
    ----------
    report : report file to write to
    ts     : telescope state
    """

    # write RST style bulletted list
    report.writeln('* Int time:     {0:f}'.format(ts.sdp_l0_int_time,))
    report.writeln('* Channels:     {0:d}'.format(ts.cbf_n_chans,))
    csv_antlist = ts.antenna_mask.split(',')
    report.writeln('* Antennas:     {0:d}'.format(len(csv_antlist),))
    report.writeln('* Antenna list: {0:s}'.format(ts.antenna_mask,))
    report.writeln()

    report.writeln('Source list:')
    report.writeln()
    for target in ts.get_range('cal_info_sources',st=0,return_format='recarray')['value']:
        report.writeln('* {0:s}'.format(target,))

    report.writeln()    

def write_table_timerow(report,antennas,time,data):
    """
    Write RST style table to report, rows: time, columns: antenna

    Parameters
    ----------
    report   : report file to write to
    antennas : list of antenna names, comma separated string
    time     : list of times (equates to number of rows in the table)
    data     : table data, shape (time, antenna)
    """

    # create table header
    header = antennas.split(',')
    header.insert(0,'time')

    n_entries = len(header)
    col_width = 15
    col_header = '='*col_width+' '

    # write table header
    report.writeln()
    report.writeln(col_header*n_entries)
    report.writeln(" ".join([h.ljust(col_width) for h in header])) 
    report.writeln(col_header*n_entries)

    # add each time row to the table
    for t, d in zip(time,data):
        data_string = " ".join(["{:.4e}".format(abs(di),).ljust(col_width) for di in d])
        report.write("{:.4e}".format(t,).ljust(col_width+1))
        report.writeln(data_string)  

    # table footer
    report.writeln(col_header*n_entries)
    report.writeln()

def write_table_timecol(report,antennas,time,data):
    """
    Write RST style table to report, rows: antenna, columns: time

    Parameters
    ----------
    report : report file to write to
    antennas : list of antenna names, comma separated string
    time     : list of times (equates to number of columns in the table)
    data     : table data, shape (time, antenna)
    """

    n_entries = len(time) + 1
    col_width = 15
    col_header = '='*col_width+' '

    # create table header
    header = " ".join(["{:.4e}".format(t,).ljust(col_width) for t in time])
    header = 'ant'.ljust(col_width+1) + header

    # write table header
    report.writeln()
    report.writeln(col_header*n_entries)
    report.writeln(header) 
    report.writeln(col_header*n_entries)

    # add each antenna row to the table
    antlist = antennas.split(',')
    for a, d in zip(antlist,data.T):
        data_string = " ".join(["{:.4e}".format(abs(di),).ljust(col_width) for di in d])
        report.write(a.ljust(col_width+1))
        report.writeln(data_string)  

    # table footer
    report.writeln(col_header*n_entries)
    report.writeln()

def make_cal_report(ts): 
    """
    Creates pdf calibration pipeline report (from RST source),
    using data from the Telescope State 
    
    Parameters
    ----------
    ts : TelescopeState
    """

    try:
        project_name = ts.experiment_id
    except AttributeError:
        project_name = 'unknown_project'

    # make calibration report directory and move into it
    try:
        os.mkdir(project_name)
    except OSError:
        project_name = project_name+'_0'
        os.mkdir(project_name)

    os.chdir(project_name)
    project_dir = os.getcwd()
    
    # make source directory and move into is
    os.mkdir('calreport_source')
    os.chdir('calreport_source')
    
    # --------------------------------------------------------------------
    # open report file
    report_file = 'calreport.rst'
    cal_rst = rstReport(report_file, 'w')

    # --------------------------------------------------------------------
    # write heading
    cal_rst.write_heading_0('Calibration pipeline report')

    # --------------------------------------------------------------------
    # write observation summary info
    cal_rst.write_heading_1('Observation summary')
    cal_rst.writeln('Observation: {0:s}'.format(project_name,))
    cal_rst.writeln()
    write_summary(cal_rst,ts)

    # --------------------------------------------------------------------
    # write RFI summary
    cal_rst.write_heading_1('RFI summary')
    cal_rst.writeln('Watch this space')
    cal_rst.writeln()

    # --------------------------------------------------------------------    
    # add cal products to report
    antenna_mask = ts.antenna_mask


    # ---------------------------------
    # delay
    cal = 'K'
    cal_product = 'cal_product_'+cal

    if cal_product in ts.keys():
        cal_rst.write_heading_1('Calibration product {0:s}'.format(cal,))
        cal_rst.writeln('Delay calibration solutions')
        cal_rst.writeln()

        product = ts.get_range(cal_product,st=0,return_format='recarray')
        vals = product['value']
        # K shape is n_time, n_pol, n_ant
        times = product['time']

        cal_rst.writeln('**POL 0**')
        write_table_timerow(cal_rst,antenna_mask,times,vals[:,0,:])
        cal_rst.writeln('**POL 1**')
        write_table_timerow(cal_rst,antenna_mask,times,vals[:,1,:])

    # ---------------------------------
    # bandpass
    cal = 'B'
    cal_product = 'cal_product_'+cal

    if cal_product in ts.keys():
        cal_rst.write_heading_1('Calibration product {0:s}'.format(cal,))
        cal_rst.writeln('Bandpass calibration solutions')
        cal_rst.writeln()

        product = ts.get_range(cal_product,st=0,return_format='recarray')
        vals = product['value']
        # B shape is n_time, n_chan, n_pol, n_ant
        times = product['time']

        for ti in range(len(times)):
            t = time.strftime("%Y %x %X",time.gmtime(times[ti]))
            cal_rst.writeln('Time: {}'.format(t,))
            insert_fig(cal_rst,plotting.plot_bp_solns(vals[ti]),name='B_'+str(ti))

    # ---------------------------------
    # gain
    cal = 'G'
    cal_product = 'cal_product_'+cal

    if cal_product in ts.keys():
        cal_rst.write_heading_1('Calibration product {0:s}'.format(cal,))
        cal_rst.writeln('Gain calibration solutions')
        cal_rst.writeln()

        product = ts.get_range(cal_product,st=0,return_format='recarray')
        vals = product['value']
        # G shape is n_time, n_pol, n_ant
        times = product['time']

        cal_rst.writeln('**POL 0**')
        insert_fig(cal_rst,plotting.plot_g_solns(times,vals[:,0,:]),name='G_P0')
        cal_rst.writeln('**POL 1**')
        insert_fig(cal_rst,plotting.plot_g_solns(times,vals[:,1,:]),name='G_P1')

    # --------------------------------------------------------------------
    # close off report
    cal_rst.writeln()
    cal_rst.writeln()
    cal_rst.close()
    
    # convert rst to pdf 
    os.system('rst2pdf {} -s eightpoint'.format(report_file))
    # move to project directory
    shutil.move(report_file.replace('rst','pdf'),project_dir)
    
