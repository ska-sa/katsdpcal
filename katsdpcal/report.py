
import os
import shutil

def get_path_from_env(envvar,filename):
    """
    Gets the path of a file from list of directories in an environment variable
    
    Parameters
    ----------
    envvar   : name of environment variable, string
    filename : name of file to search for, string
    
    Returns
    -------
    path     : path to file <filename>, of None if not found
    """
    # get list of paths in environment varible
    path_list = os.getenv(envvar).split(':')
    # iterate through list and return first path where file is found
    for path in path_list:
        if os.path.isfile(path+'/katcal/'+filename):
            return path
    # return none if environment variable is empty or if file is not found
    return None 
    
def insert_fig_list(report,fig_list):
    """
    Appends images to report
    
    Parameters
    ----------
    report   : open report file to write into
    fig_list : list of matplitlib figures 
    """
    for i, fig in enumerate(fig_list):
        figname = str(i)+'.jpeg'
        fig.savefig(figname,bbox_inches='tight')
        fig_text = \
        '''.. image:: {}
           :align: center
        '''.format(figname)
        report.write(fig_text)
        report.write('\n')

def make_cal_report(project_name,fig_list):
    """
    Creates pdf calibration pipeline report
    Report is built from template and list of figures.
    
    Parameters
    ----------
    project_name : name of observation project
    fig_list     : list of matplitlib figures 
    """
    # make calibration report directory and move into it
    os.mkdir(project_name)
    os.chdir(project_name)
    project_dir = os.getcwd()
    
    # make source directory and move into is
    os.mkdir('calreport_source')
    os.chdir('calreport_source')
    
    # copy template report into directory
    template_file = 'calreport_template.rst'
    report_file = 'calreport.rst'
    template_katcal_path = get_path_from_env('PYTHONPATH',template_file)
    template = '{0}/katcal/{1}'.format(template_katcal_path,template_file)
    shutil.copyfile(template,report_file)

    # open report to append to
    cal_rst = open(report_file, 'a')
    cal_rst.write('Calibration pipeline report for observation {0}\n\n'.format(project_name))
    
    # add images to report
    cal_rst.write('Gain solutions\n\n')
    insert_fig_list(cal_rst,fig_list)
    
    # close off report
    cal_rst.write('\n')
    cal_rst.write('\n')
    cal_rst.close()
    
    # convert rst to pdf 
    os.system('rst2pdf '+report_file)
    # move to project directory
    shutil.move(report_file.replace('rst','pdf'),project_dir)
    
    
    
    
