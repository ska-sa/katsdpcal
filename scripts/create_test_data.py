#!/usr/bin/env python
# ----------------------------------------------------------
# Create sumulated data files from existing KAT-7 h5 files

from pyrap import tables
import numpy as np
import ephem
import glob
import subprocess
import os
import optparse
import katpoint
from scipy.constants import c as light_speed

def parse_opts():
    parser = optparse.OptionParser(description = 'Create sumulated data files from KAT-7 h5 file')    
    parser.add_option('-f', '--file', type=str, help='H5 file to use for the simulation.')
    parser.add_option('-n', '--num-scans', type=int, default=0, help='Number of scans to keep in the output MS. Default: all')
    parser.add_option('--image', action='store_true', help='Image the output MS files.') 
    parser.set_defaults(image=False)
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
    elif ms_name == []:
        return None
    else:
        return ms_name[0]

def get_antdesc(names, positions, diameters):
    # get antenna description dictionary
    antdesc = {}
    first_ant = True
    for ant, diam, pos in zip(names, diameters, positions):
        if first_ant:
            # set up reference position (this is necessary to preserve precision of antenna positions when converting
            #  because of ephem limitation in truncating decimal places when printing strings
            longitude, latitude, altitude = katpoint.ecef_to_lla(pos[0],pos[1],pos[2])
            longitude_centre = ephem.degrees(str(ephem.degrees(longitude)))
            latitude_centre = ephem.degrees(str(ephem.degrees(latitude)))
            altitude_centre = round(altitude)
            ref_position = katpoint.Antenna('reference_position, {0}, {1}, {2}, 0.0'.format(longitude_centre,latitude_centre,altitude_centre))
            first_ant = False
        # now determine offsets from the reference position to build up full antenna description string
        e, n, u = katpoint.ecef_to_enu(longitude_centre, latitude_centre, altitude_centre, pos[0],pos[1],pos[2])
        antdesc[ant] = '{0}, {1}, {2}, {3}, {4}, {5} {6} {7}'.format(ant, longitude_centre, latitude_centre, altitude_centre, diam, e, n, u)
    return antdesc

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

def to_ut(t):
    """
    Converts MJD seconds into Unix time in seconds

    Parameters
    ----------
    t : time in MJD seconds

    Returns
    -------
    Unix time in seconds
    """
    return (t/86400. - 2440587.5 + 2400000.5)*86400.

def calc_uvw(phase_centre, wavelength, time, antlist, ant1, ant2,ant_desc):
    """
    Calculate uvw coordinates

    Parameters
    ----------
    phase_centre : katpoint target for phase centre position
    wavelength : wavelengths, single value or array shape(nchans)
    time : times, array of floats, shape(nrows)
    antlist : list of antenna names - used for associating antenna descriptions with an1 and ant2 indices, shape(nant)
    ant1, ant2: array of antenna indices, shape(nrows)

    Returns
    -------
    uvw_wave : uvw coordinates, normalised by wavelength
    """
    uvw = np.array([phase_centre.uvw(katpoint.Antenna(ant_desc[antlist[a1]]), timestamp=to_ut(t), antenna=katpoint.Antenna(ant_desc[antlist[a2]])) for t, a1, a2 in zip(time,ant1,ant2)])
    uvw_wave = np.array([uvw/wl for wl in wavelength])
    return uvw_wave

# ---------------------------------------------------------------------------------------------------
# SIMPLE POINT
# ------------
# Simple point source in the phase centre, I=10**a0

def create_point_simple(orig_msfile,basename='TEST0'):
    msfile = basename+'.ms'
    a0, a1, a2, a3 = [1.15, 0, 0, 0]

    os.system('rm -rf {0}'.format(msfile,))
    print orig_msfile, '88888 ', msfile
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

    # source position
    #   limit the precision of the RA DEC positions because of pyephem precision limitations on printing
    #   (this becomes an issue when creating the target description trings in the simulator)
    ra_string = '04:37:04.4'
    dec_string = '29:40:13.8'
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

    # change uvw coords in the MS file to reflect then new phase centre position
    casa_command = 'casapy -c \"fixvis(vis=\'{0}\',outputvis=\'{1}\',reuse=False)\"'.format(msfile,msfile)
    proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
    outs, errs = proc.communicate()

    # clearcal to initialise CORRECTED column for imaging
    casa_command = 'casapy -c \"clearcal(vis=\'{0}\')\"'.format(msfile,)
    proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
    outs, errs = proc.communicate()

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

def create_point_spectral(orig_msfile,basename='TEST1'):
    msfile = basename+'.ms'
    a0, a1, a2, a3 = [1.8077, -0.8018, -0.1157, 0]

    os.system('rm -rf {0}'.format(msfile,))
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

    # source position
    #   limit the precision of the RA DEC positions because of pyephem precision limitations on printing
    #   (this becomes an issue when creating the target description trings in the simulator)
    ra_string = '04:37:04.4'
    dec_string = '29:40:13.8'
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

    # change uvw coords in the MS file to reflect then new phase centre position
    casa_command = 'casapy -c \"fixvis(vis=\'{0}\',outputvis=\'{1}\',reuse=False)\"'.format(msfile,msfile)
    proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
    outs, errs = proc.communicate()

    # clearcal to initialise CORRECTED column for imaging
    casa_command = 'casapy -c \"clearcal(vis=\'{0}\')\"'.format(msfile,)
    proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
    outs, errs = proc.communicate()

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
# TWO SIMPLE POINTS
#------------------
# Two points: point source in the phase centre and an offset point, both with no frequency slope.
# Source parameters:
#   S0, P, 04:37:04.3753, 0, -19:40:13.819, 0, 1.3, 0, 0, 0, 0, 0, 0
#   S1, P, 04:39:14.3753, 0, -19:40:13.819, 0, 1.15, 0, 0, 0, 0, 0, 0

def create_points_two(orig_msfile,basename='TEST2'):
    msfile = basename+'.ms'

    os.system('rm -rf {0}'.format(msfile,))
    os.system('cp -r {0} {1}'.format(orig_msfile,msfile))

    t = tables.table(msfile,readonly=False)
    d = t.getcol('DATA')
    time = t.getcol('TIME')

    # get antenna info
    ant1 = t.getcol('ANTENNA1')
    ant2 = t.getcol('ANTENNA2')
    ant_table = tables.table(msfile+'/ANTENNA')
    antlist = ant_table.getcol('NAME')
    positions = ant_table.getcol('POSITION')
    diameters = ant_table.getcol('DISH_DIAMETER')
    ant_table.close()

    antenna_descriptions = get_antdesc(antlist, positions, diameters)

    # get wavelength info
    spw_table = tables.table(msfile+'/SPECTRAL_WINDOW')
    nu = spw_table.getcol('CHAN_FREQ')[0]
    wl = light_speed/nu
    spw_table.close()

    # point source parameters
    #   limit the precision of the RA DEC positions because of pyephem precision limitations on printing
    #   (this becomes an issue when creating the target description trings in the simulator)
    source_list = ['S0, P, 04:37:04.4, 0, -19:40:13.8, 0, 1.3, 0, 0, 0, 0, 0, 0',
    'S1, P, 04:39:14.4, 0, -20:00:00.0, 0, 1.15, 0, 0, 0, 0, 0, 0']

    # get phase centre details
    source0_params = source_list[0].split(',')
    ra0_string = source0_params[2].strip()
    dec0_string = source0_params[4].strip()
    ra0 = ephem.hours(ra0_string)
    dec0 = ephem.degrees(dec0_string)

    # set up phase centre as katpoint target
    centre_target = katpoint.Target('{0}, radec target, {1}, {2}'.format(basename,ra0_string,dec0_string))
    # calculate uvw
    uvw = calc_uvw(centre_target,wl,time,antlist,ant1,ant2,antenna_descriptions)
    u = uvw[:,:,0].T
    v = uvw[:,:,1].T
    w = uvw[:,:,2].T

    # write fake source parameters into a sky model file
    f = open(basename+'.txt','w')
    f.write('# Test data set TEST2\n')
    f.write('# -------------------\n')
    f.write('# Two points: point source in the phase centre and an offset point, both with no frequency slope.\n')
    f.write('#\n')

    # calculate visibilities for these sources
    new_data = np.zeros_like(d)
    for source in source_list:
        f.write(source)
        f.write('\n')# python will convert \n to os.linesep

        source_params = source.split(',')

        ra_string = source_params[2].strip()
        dec_string = source_params[4].strip()

        # only have first parameter in this test model (no spectral change)
        a0 = float(source_params[6])

        ra = ephem.hours(ra_string)
        dec = ephem.degrees(dec_string)

        l = np.cos(dec)*np.sin(ra-ra0)
        m = np.sin(dec)*np.cos(dec0)-np.cos(dec)*np.sin(dec0)*np.cos(ra-ra0)

        S = 10.**a0
        new_data += S*np.exp(2.*np.pi*1j*(u*l + v*m + w*(np.sqrt(1-l**2.0-m**2.0)-1.0)))[:,:,np.newaxis]

    t.putcol('DATA',new_data)
    t.close()

    # change source name and position (field is centered on S0)
    field_table = tables.table(msfile+'/FIELD',readonly=False)
    source_table = tables.table(msfile+'/SOURCE',readonly=False)

    names = ['TEST2']*3
    field_table.putcol('NAME',names)
    source_table.putcol('NAME',names)

    positions = np.array([[ra0,dec0]]*3)
    positions3d = positions[:,np.newaxis,:]

    field_table.putcol('DELAY_DIR',positions3d)
    field_table.putcol('PHASE_DIR',positions3d)
    field_table.putcol('REFERENCE_DIR',positions3d)
    source_table.putcol('DIRECTION',positions3d)

    field_table.close()
    source_table.close()

    # change uvw coords in the MS file to reflect then new phase centre position
    casa_command = 'casapy -c \"fixvis(vis=\'{0}\',outputvis=\'{1}\',reuse=False)\"'.format(msfile,msfile)
    proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
    outs, errs = proc.communicate()

    # clearcal to initialise CORRECTED column for imaging
    casa_command = 'casapy -c \"clearcal(vis=\'{0}\')\"'.format(msfile,)
    proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
    outs, errs = proc.communicate()

    # close off model text file
    f.close()

# ---------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    # parse command line options
    (opts, args) = parse_opts()

    msfile = get_msname(opts.file)
    # if the MS file already exists

    # convert h5 file to MS, if MS is not already present
    if not msfile:
        h5toms(opts.file)
        msfile = get_msname(opts.file)

    if opts.num_scans > 0:
        msfile = extract_scans(msfile, opts.num_scans)

    # create test data sets and accompanying sky model files
    basename = 'TEST0'
    create_point_simple(msfile,basename=basename)
    if opts.image:
        casa_command = 'casapy -c \"clean(vis=\'{0}.ms\',imagename=\'{1}_image\',niter=0,cell=\'30arcsec\',imsize=256)\"'.format(basename,basename)
        proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
        outs, errs = proc.communicate()

    basename = 'TEST1'
    create_point_spectral(msfile,basename=basename)
    if opts.image:
        casa_command = 'casapy -c \"clean(vis=\'{0}.ms\',imagename=\'{1}_image\',niter=0,cell=\'30arcsec\',imsize=256)\"'.format(basename,basename)
        proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
        outs, errs = proc.communicate()

    basename = 'TEST2'
    create_points_two(msfile,basename=basename)
    if opts.image:
        casa_command = 'casapy -c \"clean(vis=\'{0}.ms\',imagename=\'{1}_image\',niter=0,cell=\'30arcsec\',imsize=256)\"'.format(basename,basename)
        proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
        outs, errs = proc.communicate()
