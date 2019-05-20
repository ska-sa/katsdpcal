#!/usr/bin/env python3
# ----------------------------------------------------------
# Create sumulated data files from existing KAT-7 h5 files

from casacore import tables
import numpy as np
import ephem
import glob
import subprocess
import os
import optparse
import katpoint
from scipy.constants import c as light_speed
from katsdpcal.calprocs import to_ut
from katsdpcal.simulator import get_antdesc_relative


def parse_opts():
    parser = optparse.OptionParser(description='Create sumulated data files from KAT-7 h5 file')
    parser.add_option('-f', '--file', type=str, help='H5 file to use for the simulation.')
    parser.add_option('-n', '--num-scans', type=int, default=0,
                      help='Number of scans to keep in the output MS. Default: all')
    parser.add_option('--image', action='store_true', help='Image the output MS files.')
    parser.set_defaults(image=False)
    return parser.parse_args()


def h5toms(filename):
    # convert h5 to MS
    proc = subprocess.Popen('h5toms.py -f {0}'.format(filename),
                            stderr=subprocess.PIPE, shell=True)
    outs, errs = proc.communicate()
    # if the MS file already exists, raise an error
    if 'RuntimeError' in errs:
        error_message = errs.split('\n')[-2]
        raise RuntimeError(error_message)


def get_msname(filename):
    # check if there is an MS for the base filename
    name_base = filename.split('.h5')[0]
    ms_name = glob.glob('{0}*.ms'.format(name_base))
    if len(ms_name) > 1:
        raise ValueError('Multiple matching MS files?! {0}'.format(ms_name))
    elif ms_name == []:
        return None
    else:
        return ms_name[0]


def extract_scans(msfile, num_scans):
    # extract the number of scans requored (starting at scan 0)
    new_ms = 'TEST_{0}scans.ms'.format(num_scans)
    scan_list = ','.join([str(i) for i in range(opts.num_scans)])
    casa_command = (
        'casapy -c \"split(vis=\'{0}\',outputvis=\'{1}\',scan=\'{2}\',datacolumn=\'data\')\"'
        .format(msfile, new_ms, scan_list))
    proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
    outs, errs = proc.communicate()
    # if the MS file already exists, raise an error
    if 'already exists' in errs:
        print('ms file {0} already exists - using it!'.format(new_ms))
    return new_ms


def calc_uvw(phase_centre, timestamps, antlist, ant1, ant2, ant_descriptions, refant_ind=0):
    """
    Calculate uvw coordinates

    Parameters
    ----------
    phase_centre
        katpoint target for phase centre position
    timestamps
        times, array of floats, shape(nrows)
    antlist
        list of antenna names - used for associating antenna descriptions with
        an1 and ant2 indices, shape(nant)
    ant1, ant2
        array of antenna indices, shape(nrows)
    antenna_descriptions
        description strings for the antennas, same order as antlist, list of string
    refant_ind
        index of reference antenna in antlist, integer

    Returns
    -------
    uvw
        uvw coordinates numpy array, shape (3, nbl x ntimes)
    """
    # use the lat-long-alt values of one of the antennas as the array reference position
    refant = katpoint.Antenna(ant_descriptions[antlist[refant_ind]])
    array_reference_position = katpoint.Antenna('array_position', *refant.ref_position_wgs84)
    # use the array reference position for the basis
    basis = phase_centre.uvw_basis(timestamp=to_ut(timestamps), antenna=array_reference_position)
    # get enu vector for each row in MS, for each antenna in the baseline pair for that row
    antenna1_uvw = np.empty([3, len(timestamps)])
    antenna2_uvw = np.empty([3, len(timestamps)])
    for i, [a1, a2] in enumerate(zip(ant1, ant2)):
        antenna1 = katpoint.Antenna(ant_descriptions[antlist[a1]])
        enu1 = np.array(antenna1.baseline_toward(array_reference_position))
        antenna1_uvw[..., i] = np.tensordot(basis[..., i], enu1, ([1], [0]))

        antenna2 = katpoint.Antenna(ant_descriptions[antlist[a2]])
        enu2 = np.array(antenna2.baseline_toward(array_reference_position))
        antenna2_uvw[..., i] = np.tensordot(basis[..., i], enu2, ([1], [0]))

    # then subtract the vectors for each antenna to get the baseline vectors
    baseline_uvw = np.empty([3, len(timestamps)])
    for i, [a1, a2] in enumerate(zip(ant1, ant2)):
        baseline_uvw[..., i] = - antenna1_uvw[..., i] + antenna2_uvw[..., i]

    return baseline_uvw


def calc_uvw_wave(phase_centre, timestamps, antlist, ant1, ant2, ant_descriptions,
                  refant_ind=0, wavelengths=None):
    """
    Calculate uvw coordinates

    Parameters
    ----------
    phase_centre
        katpoint target for phase centre position
    timestamps
        times, array of floats, shape(nrows)
    antlist
        list of antenna names - used for associating antenna descriptions with
        an1 and ant2 indices, shape(nant)
    ant1, ant2
        array of antenna indices, shape(nrows)
    antenna_descriptions
        description strings for the antennas, same order as antlist, string
    refant_ind
        index of reference antenna in antlist, integer
    wavelengths
        wavelengths, single value or array shape(nchans)

    Returns
    -------
    uvw_wave
        uvw coordinates normalised by wavelength, shape (3, nbl x ntimes, len(wavelengths))
    """
    uvw = calc_uvw(phase_centre, timestamps, antlist, ant1, ant2, ant_descriptions)
    if wavelengths is None:
        return uvw
    elif np.isscalar(wavelengths):
        return uvw / wavelengths
    else:
        return uvw[:, :, np.newaxis] / wavelengths[np.newaxis, np.newaxis, :]


# ---------------------------------------------------------------------------------------------------
# SIMPLE POINT
# ------------
# Simple point source in the phase centre, I=10**a0


def create_point_simple(orig_msfile, basename='TEST0'):
    msfile = basename+'.ms'
    a0, a1, a2, a3 = [1.15, 0, 0, 0]

    os.system('rm -rf {0}'.format(msfile))
    os.system('cp -r {0} {1}'.format(orig_msfile, msfile))

    t = tables.table(msfile, readonly=False)
    d = t.getcol('DATA')
    new_data = (10**a0) * np.ones_like(d)
    t.putcol('DATA', new_data)
    t.close()

    # change to using source position of 3C123 and name <basename>
    field_table = tables.table(msfile+'/FIELD', readonly=False)
    source_table = tables.table(msfile+'/SOURCE', readonly=False)
    nfields = len(field_table.getcol('NAME'))
    names = [basename]*nfields
    field_table.putcol('NAME', names)
    source_table.putcol('NAME', names)

    # source position
    #   limit the precision of the RA DEC positions because of pyephem
    #   precision limitations on printing (this becomes an issue when creating
    #   the target description trings in the simulator)
    ra_string = '04:37:04.4'
    dec_string = '29:40:13.8'
    ra = ephem.hours(ra_string)
    dec = ephem.degrees(dec_string)
    positions = np.array([[ra, dec]]*nfields)
    positions3d = positions[:, np.newaxis, :]

    field_table.putcol('DELAY_DIR', positions3d)
    field_table.putcol('PHASE_DIR', positions3d)
    field_table.putcol('REFERENCE_DIR', positions3d)
    source_table.putcol('DIRECTION', positions3d)

    field_table.close()
    source_table.close()

    # change uvw coords in the MS file to reflect then new phase centre position
    casa_command = (
        'casapy -c \"fixvis(vis=\'{0}\',outputvis=\'{1}\',reuse=False)\"'
        .format(msfile, msfile))
    proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
    outs, errs = proc.communicate()

    # clearcal to initialise CORRECTED column for imaging
    casa_command = 'casapy -c \"clearcal(vis=\'{0}\')\"'.format(msfile)
    proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
    outs, errs = proc.communicate()

    # write fake source parameters into a sky model file
    f = open(basename+'.txt', 'w')
    heading = 'Test data set {0}'.format(basename)
    underlining = '-'*len(heading)
    f.write('# {0}\n'.format(heading))
    f.write('# {0}\n'.format(underlining))
    f.write('# Simple point source in the phase centre\n')
    f.write('# a0 = {0}\n'.format(a0))
    f.write('# a1 = {0}\n'.format(a1))
    f.write('# a2 = {0}\n'.format(a2))
    f.write('# a3 = {0}\n'.format(a3))
    f.write('S0, P, {0}, 0, {1}, 0, {2}, {3}, {4}, {5}, 0, 0, 0'
            .format(ra_string, dec_string, a0, a1, a2, a3))
    f.write('\n')  # python will convert \n to os.linesep
    f.close()

# ---------------------------------------------------------------------------------------------------
# FREQEUNCY SLOPE POINT
# ---------------------
# Point source in the phase centre, with frequency slope given by the parameters:
# a0 = 1.8077
# a1 = -0.8018
# a2 = -0.1157
# a3 = 0.0
# from calibrator 3C123:
# http://iopscience.iop.org/article/10.1088/0067-0049/204/2/19/pdf;jsessionid=B230B651C501A8F2568F1A2C34523A83.c2.iopscience.cld.iop.org


def create_point_spectral(orig_msfile, basename='TEST1'):
    msfile = basename+'.ms'
    a0, a1, a2, a3 = [1.8077, -0.8018, -0.1157, 0]

    os.system('rm -rf {0}'.format(msfile))
    os.system('cp -r {0} {1}'.format(orig_msfile, msfile))

    t = tables.table(msfile, readonly=False)
    d = t.getcol('DATA')

    spw_table = tables.table(msfile+'/SPECTRAL_WINDOW')
    nu = spw_table.getcol('CHAN_FREQ')[0]
    nu_ghz = nu/1.0e9
    spw_table.close()

    S = 10.**(a0 + a1*np.log10(nu_ghz) + a2*(np.log10(nu_ghz)**2.0) + a3*(np.log10(nu_ghz)**3.0))
    new_data = np.ones_like(d) * S[np.newaxis, :, np.newaxis]

    t.putcol('DATA', new_data)
    t.close()

    # change to using source 3C123
    field_table = tables.table(msfile+'/FIELD', readonly=False)
    source_table = tables.table(msfile+'/SOURCE', readonly=False)
    nfields = len(field_table.getcol('NAME'))
    names = [basename]*nfields
    field_table.putcol('NAME', names)
    source_table.putcol('NAME', names)

    # source position
    #   limit the precision of the RA DEC positions because of pyephem
    #   precision limitations on printing (this becomes an issue when creating
    #   the target description trings in the simulator)
    ra_string = '04:37:04.4'
    dec_string = '29:40:13.8'
    ra = ephem.hours(ra_string)
    dec = ephem.degrees(dec_string)
    positions = np.array([[ra, dec]]*nfields)
    positions3d = positions[:, np.newaxis, :]

    field_table.putcol('DELAY_DIR', positions3d)
    field_table.putcol('PHASE_DIR', positions3d)
    field_table.putcol('REFERENCE_DIR', positions3d)
    source_table.putcol('DIRECTION', positions3d)

    field_table.close()
    source_table.close()

    # change uvw coords in the MS file to reflect then new phase centre position
    casa_command = (
        'casapy -c \"fixvis(vis=\'{0}\',outputvis=\'{1}\',reuse=False)\"'
        .format(msfile, msfile))
    proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
    outs, errs = proc.communicate()

    # clearcal to initialise CORRECTED column for imaging
    casa_command = 'casapy -c \"clearcal(vis=\'{0}\')\"'.format(msfile)
    proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
    outs, errs = proc.communicate()

    # write fake source parameters into a sky model file
    f = open(basename+'.txt', 'w')
    heading = 'Test data set {0}'.format(basename)
    underlining = '-'*len(heading)
    f.write('# {0}\n'.format(heading))
    f.write('# {0}\n'.format(underlining))
    f.write('# Point source in the phase centre, with frequency slope (from 3C123 model)\n')
    f.write('# a0 = {0}\n'.format(a0))
    f.write('# a1 = {0}\n'.format(a1))
    f.write('# a2 = {0}\n'.format(a2))
    f.write('# a3 = {0}\n'.format(a3))
    f.write('S0, P, {0}, 0, {1}, 0, {2}, {3}, {4}, {5}, 0, 0, 0'
            .format(ra_string, dec_string, a0, a1, a2, a3))
    f.write('\n')  # python will convert \n to os.linesep
    f.close()

# ---------------------------------------------------------------------------------------------------
# TWO SIMPLE POINTS
# ------------------
# Two points: point source in the phase centre and an offset point, both with no frequency slope.
# Source parameters:
#   S0, P, 04:37:04.4, 0, -9:40:13.8, 0, 1.3, 0, 0, 0, 0, 0, 0
#   S1, P, 04:39:14.4, 0, -10:00:00.0, 0, 1.15, 0, 0, 0, 0, 0, 0


def create_points_two(orig_msfile, basename='TEST2'):
    msfile = basename+'.ms'

    os.system('rm -rf {0}'.format(msfile))
    os.system('cp -r {0} {1}'.format(orig_msfile, msfile))

    t = tables.table(msfile, readonly=False)
    d = t.getcol('DATA')
    timestamps = t.getcol('TIME')

    # get antenna info
    ant1 = t.getcol('ANTENNA1')
    ant2 = t.getcol('ANTENNA2')
    ant_table = tables.table(msfile+'/ANTENNA')
    antlist = ant_table.getcol('NAME')
    positions = ant_table.getcol('POSITION')
    diameters = ant_table.getcol('DISH_DIAMETER')
    ant_table.close()

    antenna_descriptions = get_antdesc_relative(antlist, diameters, positions)

    # get wavelength info
    spw_table = tables.table(msfile+'/SPECTRAL_WINDOW')
    nu = spw_table.getcol('CHAN_FREQ')[0]
    wl = light_speed/nu
    spw_table.close()

    # point source parameters
    #   limit the precision of the RA DEC positions because of pyephem
    #   precision limitations on printing (this becomes an issue when creating
    #   the target description trings in the simulator)
    source_list = ['S0, P, 04:37:04.4, 0, -9:40:13.8, 0, 1.3, 0, 0, 0, 0, 0, 0',
                   'S1, P, 04:39:14.4, 0, -10:00:00.0, 0, 1.15, 0, 0, 0, 0, 0, 0']

    # get phase centre details
    source0_params = source_list[0].split(',')
    ra0_string = source0_params[2].strip()
    dec0_string = source0_params[4].strip()
    ra0 = ephem.hours(ra0_string)
    dec0 = ephem.degrees(dec0_string)

    # set up phase centre as katpoint target
    centre_target = katpoint.Target('{0}, radec target, {1}, {2}'
                                    .format(basename, ra0_string, dec0_string))
    # calculate uvw
    uvw = calc_uvw(centre_target, timestamps, antlist, ant1, ant2, antenna_descriptions)
    uvw_wave = uvw[:, :, np.newaxis] / wl[np.newaxis, np.newaxis, :]

    # write fake source parameters into a sky model file
    f = open(basename+'.txt', 'w')
    heading = 'Test data set {0}'.format(basename)
    underlining = '-'*len(heading)
    f.write('# {0}\n'.format(heading))
    f.write('# {0}\n'.format(underlining))
    f.write('# Two points: point source in the phase centre and an offset point, '
            'both with no frequency slope.\n')
    f.write('#\n')

    # calculate visibilities for these sources
    new_data = np.zeros_like(d)
    for source in source_list:
        f.write(source)
        f.write('\n')  # python will convert \n to os.linesep

        source_params = source.split(',')

        ra_string = source_params[2].strip()
        dec_string = source_params[4].strip()

        # only have first parameter in this test model (no spectral change)
        a0 = float(source_params[6])

        ra = ephem.hours(ra_string)
        dec = ephem.degrees(dec_string)

        l = np.cos(dec)*np.sin(ra-ra0)     # noqa: E741
        m = np.sin(dec)*np.cos(dec0)-np.cos(dec)*np.sin(dec0)*np.cos(ra-ra0)
        n = np.sqrt(1.0-l**2.0-m**2.0)
        delay = uvw_wave[0]*l + uvw_wave[1]*m + uvw_wave[2]*(n-1.0)

        S = 10.**a0
        new_data += S*np.exp(2.*np.pi*1j*delay)[:, :, np.newaxis]

    t.putcol('DATA', new_data)
    t.putcol('UVW', uvw.T)
    t.close()

    # change source name and position (field is centered on S0)
    field_table = tables.table(msfile+'/FIELD', readonly=False)
    source_table = tables.table(msfile+'/SOURCE', readonly=False)
    nfields = len(field_table.getcol('NAME'))
    names = [basename]*nfields
    field_table.putcol('NAME', names)
    source_table.putcol('NAME', names)

    positions = np.array([[ra0, dec0]]*nfields)
    positions3d = positions[:, np.newaxis, :]

    field_table.putcol('DELAY_DIR', positions3d)
    field_table.putcol('PHASE_DIR', positions3d)
    field_table.putcol('REFERENCE_DIR', positions3d)
    source_table.putcol('DIRECTION', positions3d)

    field_table.close()
    source_table.close()

    # clearcal to initialise CORRECTED column for imaging
    casa_command = 'casapy -c \"clearcal(vis=\'{0}\')\"'.format(msfile)
    proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
    outs, errs = proc.communicate()

    # close off model text file
    f.close()

# ---------------------------------------------------------------------------------------------------
# FIVE FREQUENCY SLOPE POINTS
# ---------------------------
# Five points: point source in the phase centre and four offset points, all with frequency slope.
# Source parameters:
#   S0, P, 04:37:04.4, 0, -9:40:13.8, 0, 1.8077, -0.8018, -0.1157, 0, 0, 0, 0
#   S1, P, 04:39:14.4, 0, -10:00:00.0, 0, 1.45, -0.2, -0.05, 0.003, 0, 0, 0
#   S2, P, 04:38:0.0, 0, -10:10:00.0, 0, 1.15, 0.05, -0.1, -0.02, 0, 0, 0
#   S3, P, 04:39:0.0, 0, -10:15:00.0, 0, 1.03, -0.4, -0.03, 0.001, 0, 0, 0
#   S4, P, 04:36:30.0, 0, -9:30:00.0, 0, 0.85, 0.08, -0.01, -0.04, 0, 0, 0


def create_points_five(orig_msfile, basename='TEST3'):
    msfile = basename+'.ms'

    os.system('rm -rf {0}'.format(msfile))
    os.system('cp -r {0} {1}'.format(orig_msfile, msfile))

    t = tables.table(msfile, readonly=False)
    d = t.getcol('DATA')
    timestamps = t.getcol('TIME')

    # get antenna info
    ant1 = t.getcol('ANTENNA1')
    ant2 = t.getcol('ANTENNA2')
    ant_table = tables.table(msfile+'/ANTENNA')
    antlist = ant_table.getcol('NAME')
    positions = ant_table.getcol('POSITION')
    diameters = ant_table.getcol('DISH_DIAMETER')
    ant_table.close()

    antenna_descriptions = get_antdesc_relative(antlist, diameters, positions)

    # get wavelength info
    spw_table = tables.table(msfile+'/SPECTRAL_WINDOW')
    nu = spw_table.getcol('CHAN_FREQ')[0]
    nu_ghz = nu/1.0e9
    wl = light_speed/nu
    spw_table.close()

    # point source parameters
    #   limit the precision of the RA DEC positions because of pyephem
    #   precision limitations on printing (this becomes an issue when creating
    #   the target description trings in the simulator)
    source_list = ['S0, P, 04:37:04.4, 0, -9:40:13.8, 0, 1.8077, -0.8018, -0.1157, 0, 0, 0, 0',
                   'S1, P, 04:39:14.4, 0, -10:00:00.0, 0, 1.45, -0.2, -0.05, 0.003, 0, 0, 0',
                   'S2, P, 04:38:0.0, 0, -10:10:00.0, 0, 1.15, 0.05, -0.1, -0.02, 0, 0, 0',
                   'S3, P, 04:39:0.0, 0, -10:15:00.0, 0, 1.03, -0.4, -0.03, 0.001, 0, 0, 0',
                   'S4, P, 04:36:30.0, 0, -9:30:00.0, 0, 0.85, 0.08, -0.01, -0.04, 0, 0, 0']

    # get phase centre details
    source0_params = source_list[0].split(',')
    ra0_string = source0_params[2].strip()
    dec0_string = source0_params[4].strip()
    ra0 = ephem.hours(ra0_string)
    dec0 = ephem.degrees(dec0_string)

    # set up phase centre as katpoint target
    centre_target = katpoint.Target('{0}, radec target, {1}, {2}'
                                    .format(basename, ra0_string, dec0_string))
    # calculate uvw
    uvw = calc_uvw(centre_target, timestamps, antlist, ant1, ant2, antenna_descriptions)
    uvw_wave = uvw[:, :, np.newaxis] / wl[np.newaxis, np.newaxis, :]

    # write fake source parameters into a sky model file
    f = open(basename+'.txt', 'w')
    heading = 'Test data set {0}'.format(basename)
    underlining = '-'*len(heading)
    f.write('# {0}\n'.format(heading))
    f.write('# {0}\n'.format(underlining))
    f.write('# Five points: point source in the phase centre and four offset points, '
            'all with frequency slope.\n')
    f.write('#\n')

    # calculate visibilities for these sources
    new_data = np.zeros_like(d)
    for source in source_list:
        f.write(source)
        f.write('\n')  # python will convert \n to os.linesep

        source_params = source.split(',')
        a0, a1, a2, a3 = [float(s) for s in source_params[6:10]]
        log_nu = np.log10(nu_ghz)
        S = 10.**(a0 + a1*log_nu + a2*(log_nu**2.0) + a3*(log_nu**3.0))

        ra_string = source_params[2].strip()
        dec_string = source_params[4].strip()
        ra = ephem.hours(ra_string)
        dec = ephem.degrees(dec_string)
        l = np.cos(dec)*np.sin(ra-ra0)      # noqa: E741
        m = np.sin(dec)*np.cos(dec0)-np.cos(dec)*np.sin(dec0)*np.cos(ra-ra0)
        n = np.sqrt(1.0-l**2.0-m**2.0)
        delay = uvw_wave[0]*l + uvw_wave[1]*m + uvw_wave[2]*(n-1.0)

        # add each source to the data
        new_data += (S*np.exp(2.*np.pi*1j*delay))[:, :, np.newaxis]

    t.putcol('DATA', new_data)
    t.putcol('UVW', uvw.T)
    t.close()

    # change source name and position (field is centered on S0)
    field_table = tables.table(msfile+'/FIELD', readonly=False)
    source_table = tables.table(msfile+'/SOURCE', readonly=False)
    nfields = len(field_table.getcol('NAME'))
    names = [basename]*nfields
    field_table.putcol('NAME', names)
    source_table.putcol('NAME', names)

    positions = np.array([[ra0, dec0]]*nfields)
    positions3d = positions[:, np.newaxis, :]

    field_table.putcol('DELAY_DIR', positions3d)
    field_table.putcol('PHASE_DIR', positions3d)
    field_table.putcol('REFERENCE_DIR', positions3d)
    source_table.putcol('DIRECTION', positions3d)

    field_table.close()
    source_table.close()

    # clearcal to initialise CORRECTED column for imaging
    casa_command = 'casapy -c \"clearcal(vis=\'{0}\')\"'.format(msfile)
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
    create_point_simple(msfile, basename=basename)
    if opts.image:
        # don't clean, just invert (niter=0)
        casa_command = 'casapy -c \"clean(vis=\'{0}.ms\',imagename=\'{1}_image\',niter=0,cell=\'30arcsec\',imsize=256)\"'.format(basename, basename)     # noqa: E501
        proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
        outs, errs = proc.communicate()

    basename = 'TEST1'
    create_point_spectral(msfile, basename=basename)
    if opts.image:
        # don't clean, just invert (niter=0)
        casa_command = 'casapy -c \"clean(vis=\'{0}.ms\',imagename=\'{1}_image\',niter=0,cell=\'30arcsec\',imsize=256)\"'.format(basename, basename)     # noqa: E501
        proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
        outs, errs = proc.communicate()

    basename = 'TEST2'
    create_points_two(msfile, basename=basename)
    if opts.image:
        # clean (niter=100)
        casa_command = 'casapy -c \"clean(vis=\'{0}.ms\',imagename=\'{1}_image\',niter=100,cell=\'30arcsec\',imsize=256)\"'.format(basename, basename)   # noqa: E501
        proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
        outs, errs = proc.communicate()

    basename = 'TEST3'
    create_points_five(msfile, basename=basename)
    if opts.image:
        # clean (niter=100)
        casa_command = 'casapy -c \"clean(vis=\'{0}.ms\',imagename=\'{1}_image\',niter=100,cell=\'30arcsec\',imsize=256)\"'.format(basename, basename)   # noqa: E501
        proc = subprocess.Popen(casa_command, stderr=subprocess.PIPE, shell=True)
        outs, errs = proc.communicate()
