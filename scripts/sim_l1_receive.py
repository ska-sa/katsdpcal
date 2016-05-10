#!/usr/bin/env python
# ----------------------------------------------------------
# Simulate receiver for L1 data stream

from katsdpcal.simulator import init_simdata, get_file_format, SimDataMS
# import spead2 through katsdpcal to enforce import order (pyrap must be imported first)
from katsdpcal import spead2

from katsdptelstate import endpoint, ArgumentParser

import numpy as np
import shutil
import os

def parse_opts():
    parser = ArgumentParser(description = 'Simulate receiver for L1 data stream')    
    parser.add_argument('--l1-spectral-spead', type=endpoint.endpoint_list_parser(7202, single_port=True), default=':7202', 
            help='endpoints to listen for L1 SPEAD stream (including multicast IPs). [<ip>[+<count>]][:port]. [default=%(default)s]', metavar='ENDPOINT')
    parser.add_argument('--file', type=str, help='File for simulated data (H5 or MS)')
    parser.add_argument('--image', action='store_true', help='Image the L1 data [default: False]')
    parser.set_defaults(image=False)
    parser.set_defaults(telstate='localhost')
    return parser.parse_args()

def accumulate_l1(rx, return_data=False):
    """
    Read L1 data from spead stream and accumulate it into a list

    Inputs:
    -------
    rx          : SPEAD data receiver
    return_data : If True, collect and return the data

    Returns:
    --------
    If return_data True:
    data_times : list of timestamps
    data_vis : list of visibilities
    data_flags : list of flags
    """
    timestamp_prev = 0

    if return_data: data_times, data_vis , data_flags = [], [], []
    # don't do weights for now, as I'm not really using weights
    
    print 'Got heaps: ',
    array_index = 0

    # receive SPEAD stream
    ig = spead2.ItemGroup()
    for heap in rx: 
        ig.update(heap)
        
        timestamp = ig['timestamp'].value
        vis = ig['correlator_data'].value
        flags = ig['flags'].value
        weights = ig['weights'].value

        if return_data:
            data_times.append(timestamp)
            data_vis.append(vis)
            data_flags.append(flags)
        
        # print some values to see all is well
        print array_index, timestamp, vis.shape ,flags.shape, weights.shape,
        print np.round(timestamp-timestamp_prev,2)
        timestamp_prev = timestamp
        array_index += 1    

    if return_data: return data_times, data_vis, data_flags
                
if __name__ == '__main__':
    """
    Recieve an L1 output stream and print some details to confirm all is going well
    Optionally write the L1 data back to the MS or H5 file
    """
    opts = parse_opts() 
    # Initialise spead receiver
    rx = spead2.recv.Stream(spead2.ThreadPool(), bug_compat=spead2.BUG_COMPAT_PYSPEAD_0_5_2)
    rx.add_udp_reader(opts.l1_spectral_spead[0].port)
    # recieve stream and accumulate data into arrays

    return_data = True if opts.file else False
    l1_data = accumulate_l1(rx, return_data=return_data)

    # if specified, write the output back to the file
    if opts.file:
        file_base = opts.file.split('.')[0:-1]
        file_base = '.'.join(file_base)
        file_type = opts.file.split('.')[-1]
        new_file = '{0}_L1.{1}'.format(file_base,file_type)

        # need some info from the telstate
        ts = opts.telstate

        # was our simulator using an H5 or MS file?
        file_class = get_file_format(opts.file)

        if not ts.cal_full_l1:
            print 'Only target L1 stream transmitted. Not saving L1 data to file.'
        else:
            if os.path.isfile(new_file) or os.path.isdir(new_file):
                print 'WARNING: L1 data file {0} already exists. Over writing it.'.format(new_file,)
                os.system('rm -rf {0}'.format(new_file,))

            os.system('cp -r {0} {1}'.format(opts.file,new_file))
            # set up file to write the data into
            datafile = init_simdata(new_file,mode='r+')

            print 'Writing data to {0} file {1}'.format(file_type, new_file)
            datafile.write(ts,l1_data)
            # get field IDs before closing the file
            field_id_list = datafile.field_ids()
            datafile.close()

            if opts.image:
                # only image from MS simulator
                if file_class == SimDataMS:
                    bchan = ts.cal_sim_bchan
                    echan = ts.cal_sim_echan-ts.cal_sim_bchan

                    for field_id in field_id_list:
                        # image L0 data
                        if os.path.isfile(new_file) or os.path.isdir(new_file):
                            print 'WARNING: L0 image {0}_F{1}_L0_I* already exists. Over writing it.'.format(file_base,field_id)
                            os.system('rm -rf {0}_F{1}_L0_I*'.format(file_base,field_id))

                        # flag uncalibrated first dump
                        flag_params = 'vis="{0}",mode="quack",quackinterval=1.0,quackmode="beg",action="apply",datacolumn="CORRECTED_DATA"'.format(opts.file,)
                        os.system("casapy -c 'flagdata({0})' ".format(flag_params))

                        # image using casapy
                        clean_params = 'vis="{0}",imagename="{1}_F{2}_L0_I",niter=0,stokes="I",spw="0:{3}~{4}",field="{5}",cell="30arcsec",imsize=[1024,1024],weighting="uniform"'.format(opts.file,file_base,field_id,bchan,echan,field_id)
                        os.system("casapy -c 'clean({0})' ".format(clean_params))

                        # image L1 data
                        if os.path.isfile(new_file) or os.path.isdir(new_file):
                            print 'WARNING: L1 image {0}_F{1}_L1_I* already exists. Over writing it.'.format(file_base,field_id)
                            os.system('rm -rf {0}_F{1}_L1_I*'.format(file_base,field_id))

                        # flag uncalibrated first dump
                        flag_params = 'vis="{0}",mode="quack",quackinterval=1.0,quackmode="beg",action="apply",datacolumn="CORRECTED_DATA"'.format(new_file,)
                        os.system("casapy -c 'flagdata({0})' ".format(flag_params))

                        # image using casapy
                        clean_params = 'vis="{0}",imagename="{1}_F{2}_L1_I",niter=0,stokes="I",spw="0:{3}~{4}",field="{5}",cell="30arcsec",imsize=[1024,1024],weighting="uniform"'.format(new_file,file_base,field_id,bchan,echan,field_id)
                        os.system("casapy -c 'clean({0})' ".format(clean_params))

                else:
                    print 'Simulator didnt use MS file. Can only currently image from MS.'
