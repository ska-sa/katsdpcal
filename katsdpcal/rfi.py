"""
RFI flagging module for MeerKAT calibration pipeline
====================================================

RFI flagger routines for use in the MeerKAT calibration pipeline.
"""

#--------------------------------------------------------------------------------------------------
#--- Below added by Laura
#--------------------------------------------------------------------------------------------------

import numpy as np

import logging
logger = logging.getLogger(__name__)

def no_transform(data):
  return(data)

def threshold_flagging(data,flags,threshold,transform=None):
    """
    Flags data that falls above a specified threshold

    Inputs
    ------
    data : array of numeric
        Visibility data on which to base the flags
    flags : array of boolean or uint8 same shape as data
        Flags for bad data points
    threshold : float
        Treshold to use for flagging
    transform : function
        Transform to apply to the data before thresholding. 
        For example, a function that takes the absolute value, numpy.abs 
    """

    if transform is None: transform = no_transform
    # get flags in bool format to use as a mask
    fg = flags.view(np.bool)

    mean_data = np.average(transform(data[~fg]))
    std_data = np.std(transform(data[~fg]))

    # flag based on the mean and stddev, unless there is no data in the block
    if mean_data:  
        # this is not efficient - calculates the abs offset even for masked values
        # fix later
        flags[np.abs(transform(data)-mean_data)>threshold*std_data] = 1.0

def threshold_avg_flagging(data,flags,thresholds,blocks=None,transform=None):
    """
    Flags data that falls above a specified threshold, iteratively averaging in
    larger time-channel blocks

    Inputs
    ------
    data : array of numeric
        Visibility data on which to base the flags
    flags : array of boolean or uint8 same shape as data
        Flags for bad data points
    thresholds : list of float, shape(N)
        Tresholds to use for tN iterations of flagging
    blocks : list of int, shape(N-1, 2)
        List of block sizes to average over from the second iteration, in the
        form [time_block, channel_block]
    transform : function
        Transform to apply to the data before thresholding. 
        For example, a function that takes the absolute value, numpy.abs 
    """

    thresholds = np.atleast_1d(thresholds)
    blocks = np.atleast_2d(blocks)

    # first flagging iteration
    threshold_flagging(data,flags,thresholds[0],transform=transform)

    for i,t in enumerate(thresholds[1:]):
        # average in blocks
        sx, sy = data.shape[0:2]
        # averaging blocks for time and channel
        tblock, cblock = blocks[i]

        data_av = np.add.reduceat(data*~flags.view(np.bool),range(0,sx,tblock),axis=0)
        av_flags = np.add.reduceat(flags,range(0,sx,tblock),axis=0,dtype=flags.dtype)
        data_av = np.add.reduceat(data_av*~av_flags.view(np.bool),range(0,sy,cblock),axis=1)
        av_flags = np.add.reduceat(av_flags,range(0,sy,cblock),axis=1,dtype=av_flags.dtype)

        # threshold averaged data
        threshold_flagging(data_av,av_flags,t,transform=np.abs)

        # spread flags back out to original size
        flags_spread = av_flags.repeat(tblock,axis=0)[0:sx,:]
        flags = flags_spread.repeat(cblock,axis=1)[:,0:sy]

#--------------------------------------------------------------------------------------------------
#--- Below original module from Tom
#--------------------------------------------------------------------------------------------------

# Library to contain RFI flagging routines for the pipeline
# and other 2-d RFI related functions.
import katdal
import katpoint 

import numpy as np
import scipy.signal as signal
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import math

import h5py
import os


def getbackground_2dspline(data,spike_width):

    """ From a 2-d data array determine a background iteratively by fitting a spline
    and removing data more than a few sigma from the spline """

    # Remove the first and last element in data from the fit.
    y=np.copy(data[1:-1])
    arraysize=y.shape[0]
    x=np.arange(arraysize)

    # Iterate 4 times
    for iteration in range(4):

        # First iteration fits a linear spline with 3 knots.
        if iteration==0:
            npieces=3
        # Second iteration fits a quadratic spline with 10 knots.
        elif iteration==1:
            npieces=10
        # Third and fourth iterations fit a cubic spline with 50 and 75 knots respectively.
        elif iteration>1:
            npieces=iteration*25
        deg=min(iteration+1,3)
        
        # Size of each piece of the spline.
        psize = arraysize/npieces
        firstindex = arraysize%psize + int(psize/2)
        indices = np.trim_zeros(np.arange(firstindex,arraysize,psize))

        # Fit the spline
        thisfit = interpolate.LSQUnivariateSpline(x,y,indices,k=deg)
        
        thisfitted_data=np.asarray(thisfit(x),y.dtype)

        # Subtract the fitted spline from the data
        residual = y-thisfitted_data
        this_std = np.std(residual)

        # Reject data more than 5sigma from the residual. 
        flags = residual > 5*this_std

        # Set rejected data value to the fitted value + 1sigma.
        y[flags] = thisfitted_data[flags] + this_std

    # Final iteration has knots separated by "spike_width".
    npieces = int(y.shape[0]/spike_width)
    psize = (x[-1]+1)/npieces
    firstindex = int((y.shape[0]%psize))
    indices = np.trim_zeros(np.arange(firstindex,arraysize,psize))

    # Get the final background.
    finalfit = interpolate.LSQUnivariateSpline(x,y,indices,k=3)
    thisfitted_data = np.asarray(finalfit(x),y.dtype)
    
    # Insert the original data at the beginning and ends of the data array.
    thisfitted_data = np.append(thisfitted_data,data[-1])
    thisfitted_data = np.insert(thisfitted_data,0,data[0])

    return(thisfitted_data)

def getbackground_median(data,spike_width_freq=5,spike_width_time=5):
    """
    Determine a background using a 2d median filter for 2d arrays or a 1d one for 1d arrays. 
    Just a call to median_filter from scipy.ndimage.
    Parameters
    ----------
    data : a 2d array (time, frequency) of floats
    spike_width_freq (int) : the median filter width in frequency
    spike_width_time (int) : the median filter width in time (ignored for 1d input data)
    """

    if len(data.shape) == 1 : kernel = spike_width_freq
    else : kernel = (spike_width_time,spike_width_freq)
    return ndimage.filters.median_filter(data,size=kernel)



def detect_spikes_sumthreshold(data, blarray=None, spike_width=5, outlier_sigma=11.0, window_size_auto=[1,3,5], window_size_cross=[2,4,8]):
    """FUNCTION :  detect_spikes_sumthreshold
    Given an array "data" from a baseline:
    determine if the data is an auto-correlation or a cross-correlation.
    Get the background in the data using a median filter for auto_correlations or a 
    cubic spline for cross correlations.
    Make an array of flags from the data using the "sumthreshold" method and return
    this array of flags.
    Parameters
    ----------
    data : array-like
        An N-dimensional numpy array containing the data to flag
    blarray : array-like
        An array of baseline labels used to determine if the 
        baseline index is an auto- or a cross- correlation. Should have the same
        shape as the baseline part of the data array.
    spike_width : integer
        The width of the median filter and the gaps between knots in the spline fit.
    outlier_sigma : float
        The number of sigma in the first iteration of the sumthreshold method.
    buffer_size : int
        The number of timestamps in the data array to average.
    window_size_auto : array of ints
        The sizes of the averaging windows in each sumthreshold iteration for auto-correlations.
    window_size_cross : array of ints 
        The sizes of the averaging windows in each sumthreshold iteration for cross-correlations.
    """

    # Kernel size for the median filter.
    kernel_size = 2 * max(int(spike_width), 0) + 1
    #Init Flags
    flags = np.zeros(list(data.shape), dtype=np.uint8)

    for bl_index in range(data.shape[-1]):
        # Extract this baseline from the data
        this_data_buffer = data[:,bl_index]
        
        #Separate the auto-correlations and the cross-correlations
        #auto-correlations use a median filter and cross correlations
        #use a fitted spline.
        if blarray is not None:
            bl_name = bline.bls_ordering[bl_index]
        
        # Check if this is an auto or a cross... (treat as auto if we have no bl-ordering)
        if blarray is None or bl_name[0][:-1] == bl_name[1][:-1]:
            #Auto-Correlation.
            #filtered_data = np.asarray(signal.medfilt(this_data_buffer, kernel_size), this_data_buffer.dtype)
            filtered_data = getbackground_spline(this_data_buffer,kernel_size)
            #Use the auto correlation window function
            #filtered_data = ndimage.grey_opening(this_data_buffer, (kernel_size,))
            window_bl = window_size_auto
            this_sigma = outlier_sigma
        else:
            #Cross-Correlation.
            filtered_data = getbackground_spline(this_data_buffer,kernel_size)
            #Use the cross correlation window function
            window_bl = window_size_cross
            # Can lower the threshold a little (10%) for cross correlations
            this_sigma = outlier_sigma * 0.9
    
        av_dev = (this_data_buffer-filtered_data)

        av_abs_dev = np.abs(av_dev)
            
        # Calculate median absolute deviation (MAD)
        med_abs_dev = np.median(av_abs_dev[av_abs_dev>0])
            
        # Assuming normally distributed deviations, this is a robust estimator of the standard deviation
        estm_stdev = 1.4826 * med_abs_dev
            
        # Identify initial outliers (again based on normal assumption), and replace them with local median
        threshold = this_sigma * estm_stdev
        outliers = np.zeros(data.shape[0],dtype=np.bool)
        # Always flag the first element of the array.
        outliers[0] = True 

        for window in window_bl:
            #Set up 'this_data' from the averaged background subtracted buffer 
            bl_data = av_dev.copy()
                
            #The threshold for this iteration is calculated from the initial threshold
            #using the equation from Offringa (2010).
            # rho=1.3 in the equation seems to work better for KAT-7 than rho=1.5 from AO.
            thisthreshold = threshold / pow(1.2,(math.log(window)/math.log(2.0)))
            #Set already flagged values to be the value of this threshold
            bl_data[outliers] = thisthreshold
                
            #Calculate a rolling average array from the data with a windowsize for this iteration
            weight = np.repeat(1.0, window)/window
            avgarray = np.convolve(bl_data, weight,mode='valid')
                
            #Work out the flags from the convolved data using the current threshold.
            #Flags are padded with zeros to ensure the flag array (derived from the convolved data)
            #has the same dimension as the input data.
            this_flags = (avgarray > thisthreshold)
            #Convolve the flags to be of the same width as the current window.
            convwindow = np.ones(window,dtype=np.bool)
            this_outliers = np.convolve(this_flags,convwindow)
            #"OR" the flags with the flags from the previous iteration.
            outliers = outliers | this_outliers
        flags[:,bl_index] = outliers
    return flags



def detect_spikes_mad(data,blarray=None,spike_width=20,outlier_sigma=3):
    """
    FUNCTION :  detect_spikes_mad
    Given an array "data" from a baseline determine flags using the "median absolute
    deviation" method. The data is median filtered (with a kernel defined by "spike_width")
    to find a background and then rfi spikes are found by finding peaks that are 
    "outlier_sigma" from the median of the absolute values of the background subtracted data.
    Parameters
    ----------
    data : array-like
        An N-dimensional numpy array containing the data to flag
    blarray : CorrProdRef object
        Baseline labels used to determine if the 
        baseline index is an auto- or a cross- correlation.
    spike_width : integer
        The width of the median filter and the gaps between knots in the spline fit.
    outlier_sigma : float
        The number of sigma in the first iteration of the sumthreshold method.
    """    
    flags = np.zeros_like(data, dtype=np.uint8)

    for bl_index in range(data.shape[-1]):
        spectral_data = np.abs(data[:,bl_index])
        kernel_size = 2 * max(int(spike_width), 0) + 1
        # Medfilt now seems to upcast 32-bit floats to doubles - convert it back to floats...
        #filtered_data = np.asarray(signal.medfilt(spectral_data, kernel_size), spectral_data.dtype)     
        filtered_data =  np.median(rolling_window(spectral_data, kernel_size,pad=True),axis=-1)
        # The deviation is measured relative to the local median in the signal
        abs_dev = np.abs(spectral_data - filtered_data)
        # Calculate median absolute deviation (MAD)
        med_abs_dev = np.median(abs_dev[abs_dev>0])
        #med_abs_dev = signal.medfilt(abs_dev, kernel)
        # Assuming normally distributed deviations, this is a robust estimator of the standard deviation
        estm_stdev = 1.4826 * med_abs_dev
        # Identify outliers (again based on normal assumption), and replace them with local median
        #outliers = ( abs_dev > self.n_sigma * estm_stdev)
        #print outliers
        # Identify only positve outliers
        outliers = (spectral_data - filtered_data > outlier_sigma*estm_stdev)
        flags[:,bl_index] = outliers
        # set appropriate flag bit for detected RFI
    
    return flags


def detect_spikes_median(data,blarray=None,spike_width=10,outlier_sigma=11.0):
    """
    FUNCTION :  detect_spikes_median
    Simple RFI flagging through thresholding.
    Given an array "data" from a baseline determine flags using a simple median filter.
    Trivial thresholder that looks for n sigma deviations from the average
    of the supplied frame.
    Parameters
    ----------
    data : array-like
        An N-dimensional numpy array containing the data to flag
    blarray : CorrProdRef object
        Baseline labels used to determine if the 
        baseline index is an auto- or a cross- correlation.
        This is a dummy varaible
    spike_width : integer
        The width of the median filter and the gaps between knots in the spline fit.
    outlier_sigma : float
        The number of sigma in the first iteration of the sumthreshold method.
    """
    flags = np.zeros(list(data.shape), dtype=np.uint8)
    for bl_index in xrange(data.shape[-1]):
        spectral_data = data[...,bl_index]
        #spectral_data = np.atleast_1d(spectral_data)
        kernel_size=spike_width
        filtered_data =  np.median(rolling_window(spectral_data, kernel_size,pad=True),axis=-1)
        # The deviation is measured relative to the local median in the signal
        abs_dev = spectral_data - filtered_data
        outliers = (abs_dev > np.std(abs_dev)*2.3)# TODO outlier_sigma pram
        flags[:,bl_index] = outliers
    return flags

def rolling_window(a, window,axis=-1,pad=False,mode='reflect',**kargs):
    """
     This function produces a rolling window shaped data with the rolled data in the last col
        a      :  n-D array of data  
        window : integer is the window size
        axis   : integer, axis to move the window over
                 default is the last axis.
        pad    : {Boolean} Pad the array to the origanal size
        mode : {str, function} from the function numpy.pad
        One of the following string values or a user supplied function.
        'constant'      Pads with a constant value.
        'edge'          Pads with the edge values of array.
        'linear_ramp'   Pads with the linear ramp between end_value and the
                        array edge value.
        'maximum'       Pads with the maximum value of all or part of the
                        vector along each axis.
        'mean'          Pads with the mean value of all or part of the
                      con  vector along each axis.
        'median'        Pads with the median value of all or part of the
                        vector along each axis.
        'minimum'       Pads with the minimum value of all or part of the
                        vector along each axis.
        'reflect'       Pads with the reflection of the vector mirrored on
                        the first and last values of the vector along each
                        axis.
        'symmetric'     Pads with the reflection of the vector mirrored
                        along the edge of the array.
        'wrap'          Pads with the wrap of the vector along the axis.
                        The first values are used to pad the end and the
                        end values are used to pad the beginning.
        <function>      of the form padding_func(vector, iaxis_pad_width, iaxis, **kwargs)
                        see numpy.pad notes
        **kargs are passed to the function numpy.pad
        
    Returns:
        an array with shape = np.array(a.shape+(window,))
        and the rolled data on the last axis
        
    Example:
        import numpy as np
        data = np.random.normal(loc=1,scale=np.sin(5*np.pi*np.arange(10000).astype(float)/10000.)+1.1, size=10000)
        stddata = rolling_window(data, 400).std(axis=-1)
    """
    if axis == -1 : axis = len(a.shape)-1 
    if pad :
        pad_width = []
        for i in xrange(len(a.shape)):
            if i == axis: 
                pad_width += [(window//2,window//2 -1 +np.mod(window,2))]
            else :  
                pad_width += [(0,0)] 
        a = np.pad(a,pad_width=pad_width,mode=mode,**kargs)
    a1 = np.swapaxes(a,axis,-1) # Move target axis to last axis in array
    shape = a1.shape[:-1] + (a1.shape[-1] - window + 1, window)
    strides = a1.strides + (a1.strides[-1],)
    return np.lib.stride_tricks.as_strided(a1, shape=shape, strides=strides).swapaxes(-2,axis) # Move original axis to 


def detect_spikes_orig(data, axis=0, spike_width=2, outlier_sigma=11.0):
    """
    Detect and Remove outliers from data, replacing them with a local median value.

    The data is median-filtered along the specified axis, and any data values
    that deviate significantly from the local median is is marked as outlier.

    Parameters
    ----------
    data : array-like
        N-dimensional numpy array containing data to clean
    axis : int, optional
        Axis along which to perform median, between 0 and N-1
    spike_width : int, optional
        Spikes with widths up to this limit (in samples) will be removed. A size
        of <= 0 implies no spike removal. The kernel size for the median filter
        will be 2 * spike_width + 1.
    outlier_sigma : float, optional
        Multiple of standard deviation that indicates an outlier

    Returns
    -------
    cleaned_data : array
        N-dimensional numpy array of same shape as original data, with outliers
        removed

    Notes
    -----
    This is very similar to a *Hampel filter*, also known as a *decision-based
    filter* or three-sigma edit rule combined with a Hampel outlier identifier.

    .. todo::

       TODO: Make this more like a Hampel filter by making MAD time-variable too.

    """
    flags = np.zeros(data.shape, dtype='int32')#making sure that the data is already 1-D
    spectral_data = np.atleast_1d(data)
    kernel_size = 2 * max(int(spike_width), 0) + 1
    # Median filter data along the desired axis, with given kernel size
    kernel = np.ones(spectral_data.ndim, dtype='int32')
    kernel[axis] = kernel_size
    # Medfilt now seems to upcast 32-bit floats to doubles - convert it back to floats...
    #filtered_data = np.asarray(signal.medfilt(spectral_data, kernel), spectral_data.dtype)
    filtered_data =  np.median(rolling_window(spectral_data, kernel_size,pad=True),axis=-1)
    
    # The deviation is measured relative to the local median in the signal
    abs_dev = np.abs(spectral_data - filtered_data)
    # Calculate median absolute deviation (MAD)
    med_abs_dev = np.expand_dims(np.median(abs_dev[abs_dev>0],axis), axis)
    #med_abs_dev = signal.medfilt(abs_dev, kernel)
    # Assuming normally distributed deviations, this is a robust estimator of the standard deviation
    estm_stdev = 1.4826 * med_abs_dev
    # Identify outliers (again based on normal assumption), and replace them with local median
    #outliers = ( abs_dev > self.n_sigma * estm_stdev)
    # Identify only positve outliers
    outliers = (spectral_data - filtered_data > outlier_sigma*estm_stdev)
    # assign as my cols to flags as outliers has.
    flags[...] = outliers[...]
    #return flags
    return flags
