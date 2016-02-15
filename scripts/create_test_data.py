#!/usr/bin/env python
# ----------------------------------------------------------
# Create sumulated data files from existing KAT-7 h5 files

from pyrap import tables
import numpy as np
import ephem
import glob
import subprocess
import optparse
import os.path

def parse_opts():
    parser = optparse.OptionParser(description = 'Create sumulated data files from KAT-7 h5 file')    
    parser.add_option('-f', '--file', type=str, help='H5 file to use for the simulation.')
    parser.add_option('-n', '--num-scans', type=int, default=0, help='Number of scans to keep in the output MS. Default: all')
    return parser.parse_args()

def h5toms(filename):
    # convert h5 to MS
    proc = subprocess.Popen('h5toms.py -f {0}'.format(filename,), stderr=subprocess.PIPE, shell=True)
    outs, errs = proc.communicate()
    # if the MS file already exists, raise an error
    if 'RuntimeError' in errs:
        error_message = errs.split('\n')[-2]
        raise RuntimeError(error_message)

def get_msname(filename):
    name_base = filename.split('.h5')[0]
    ms_name = glob.glob('{0}*.ms'.format(name_base,))
    if len(ms_name) > 1:
        raise ValueError('Multiple matching MS files?! {0}'.format(ms_name,))
    else:
        ms_name = ms_name[0]
    return ms_name

def extract_scans(msfile, num_scans):
    # extract the number of scans requored (starting at scan 0)
    new_ms = 'TEST_{0}scans.ms'.format(num_scans)
    scan_list = ','.join([str(i) for i in range(opts.num_scans)])
    casa_command = 'casapy -c \"split(vis=\'{0}\',outputvis=\'{1}\',scan=\'{2}\',datacolumn=\'data\')\"'.format(msfile,new_ms,scan_list)
    proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
    outs, errs = proc.communicate()
    # if the MS file already exists, raise an error
    if 'already exists' in errs:
        print 'ms file {0} already exists - using it!'.format(new_ms,)
    return new_ms

# ---------------------------------------------------------------------------------------------------
# SIMPLE POINT
# ------------
# Simple point source in the phase centre, I=10**a0

def create_point_simple(orig_msfile):
    basename = 'TEST0'
    msfile = basename+'.ms'
    a0, a1, a2, a3 = [1.15, 0, 0, 0]

    os.system('cp -r {0} {1}'.format(orig_msfile,msfile))

    t = tables.table(msfile,readonly=False)
    d = t.getcol('DATA')
    new_data = (10**a0) * np.ones_like(d)
    t.putcol('DATA',new_data)
    t.close()

    # change to using source position of 3C123 and name 'TEST0'
    field_table = tables.table(msfile+'/FIELD',readonly=False)
    source_table = tables.table(msfile+'/SOURCE',readonly=False)

    names = ['TEST0']*3
    field_table.putcol('NAME',names)
    source_table.putcol('NAME',names)

    ra_string = '04:37:04.3753'
    dec_string = '29:40:13.819'
    ra = ephem.hours(ra_string)
    dec = ephem.degrees(dec_string)
    positions = np.array([[ra,dec]]*3)
    positions3d = positions[:,np.newaxis,:]

    field_table.putcol('DELAY_DIR',positions3d)
    field_table.putcol('PHASE_DIR',positions3d)
    field_table.putcol('REFERENCE_DIR',positions3d)
    source_table.putcol('DIRECTION',positions3d)

    field_table.close()
    source_table.close()

    # write fake source parameters into a sky model file
    f = open(basename+'.txt','w')
    f.write('# Test data set TEST0\n')
    f.write('# -------------------\n')
    f.write('# Simple point source in the phase centre\n')
    f.write('# a0 = {0}\n'.format(a0,))
    f.write('# a1 = {0}\n'.format(a1,))
    f.write('# a2 = {0}\n'.format(a2,))
    f.write('# a3 = {0}\n'.format(a3,))
    f.write('S0, P, {0}, 0, {1}, 0, {2}, {3}, {4}, {5}, 0, 0, 0'.format(ra_string,dec_string,a0,a1,a2,a3)) 
    f.write('\n')# python will convert \n to os.linesep
    f.close()

# ---------------------------------------------------------------------------------------------------
# FREQEUNCY SLOPE POINT
#---------------------
# Point source in the phase centre, with frequency slope given by the parameters:
# a0 = 1.8077
# a1 = -0.8018
# a2 = -0.1157
# a3 = 0.0
# from calibrator 3C123:
# http://iopscience.iop.org/article/10.1088/0067-0049/204/2/19/pdf;jsessionid=B230B651C501A8F2568F1A2C34523A83.c2.iopscience.cld.iop.org

def create_point_spectral(orig_msfile):
    basename = 'TEST1'
    msfile = basename+'.ms'
    a0, a1, a2, a3 = [1.8077, -0.8018, -0.1157, 0]

    os.system('cp -r {0} {1}'.format(orig_msfile,msfile))

    t = tables.table(msfile,readonly=False)
    d = t.getcol('DATA')

    spw_table = tables.table(msfile+'/SPECTRAL_WINDOW')
    nu = spw_table.getcol('CHAN_FREQ')[0]
    nu_ghz = nu/1.0e9
    spw_table.close()

    S = 10.**(a0 + a1*np.log10(nu_ghz) + a2*(np.log10(nu_ghz)**2.0) + a3*(np.log10(nu_ghz)**3.0))
    new_data = np.ones_like(d) * S[np.newaxis,:,np.newaxis]

    t.putcol('DATA',new_data)
    t.close()

    # change to using source 3C123
    field_table = tables.table(msfile+'/FIELD',readonly=False)
    source_table = tables.table(msfile+'/SOURCE',readonly=False)

    names = ['TEST1']*3
    field_table.putcol('NAME',names)
    source_table.putcol('NAME',names)

    ra_string = '04:37:04.3753'
    dec_string = '29:40:13.819'
    ra = ephem.hours(ra_string)
    dec = ephem.degrees(dec_string)
    positions = np.array([[ra,dec]]*3)
    positions3d = positions[:,np.newaxis,:]

    field_table.putcol('DELAY_DIR',positions3d)
    field_table.putcol('PHASE_DIR',positions3d)
    field_table.putcol('REFERENCE_DIR',positions3d)
    source_table.putcol('DIRECTION',positions3d)

    field_table.close()
    source_table.close()

    # write fake source parameters into a sky model file
    f = open(basename+'.txt','w')
    f.write('# Test data set TEST1\n')
    f.write('# -------------------\n')
    f.write('# Point source in the phase centre, with frequency slope (from 3C123 model)\n')
    f.write('# a0 = {0}\n'.format(a0,))
    f.write('# a1 = {0}\n'.format(a1,))
    f.write('# a2 = {0}\n'.format(a2,))
    f.write('# a3 = {0}\n'.format(a3,))
    f.write('S0, P, {0}, 0, {1}, 0, {2}, {3}, {4}, {5}, 0, 0, 0'.format(ra_string,dec_string,a0,a1,a2,a3)) 
    f.write('\n')# python will convert \n to os.linesep
    f.close()

# ---------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # parse command line options
    (opts, args) = parse_opts()

    msfile = get_msname(opts.file)
    # if the MS file already exists

    # convert h5 file to MS, if MS is not already present
    if not os.path.isdir(msfile):
        h5toms(opts.file)

    if opts.num_scans > 0:
        msfile = extract_scans(msfile, opts.num_scans)

    # create test data sets and accompanying sky model files
    create_point_simple(msfile)
    create_point_spectral(msfile)

