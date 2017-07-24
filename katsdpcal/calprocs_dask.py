"""
Calibration procedures for MeerKAT calibration pipeline
=======================================================

Solvers and averagers for use in the MeerKAT calibration pipeline. The
functions in this module generally expect and return :class:`dask.Array`s
rather than numpy arrays.
"""

from functools import wraps

import numpy as np
import dask.array as da

from . import calprocs


def stefcal(rawvis, num_ants, corrprod_lookup, weights=None, ref_ant=0,
            init_gain=None, *args, **kwargs):
    """Solve for antenna gains using StEFCal.

    Refer to :func:`katsdpcal.calprocs.stefcal` for details. This version
    expects a dask array for `rawvis`, and optionally for `weights` and
    `init_gain` as well.
    """
    if weights is None:
        weights = da.ones(1, dtype=rawvis.real.dtype, chunks=1)
    else:
        weights = da.asarray(weights)
    if weights.ndim == 0:
        weights = weights[np.newaxis]

    if init_gain is None:
        init_gain = da.ones(num_ants, dtype=rawvis.dtype, chunks=num_ants)
    else:
        init_gain = da.asarray(init_gain)

    # label the dimensions; the reverse is to match numpy broadcasting rules
    # where the number of dimensions don't match. The final dimension in each
    # case is given a unique label because they do not necessarily match along
    # that dimension.
    rawvis_dims = list(reversed(range(rawvis.ndim)))
    rawvis_dims[-1] = 'i'
    weights_dims = list(reversed(range(weights.ndim)))
    weights_dims[-1] = 'j'
    init_gain_dims = list(reversed(range(init_gain.ndim)))
    init_gain_dims[-1] = 'k'
    out_dims = list(reversed(range(max(rawvis.ndim, weights.ndim, init_gain.ndim))))
    out_dims[-1] = 'l'

    # Determine the output dtype, since the gufunc has two signatures
    if (np.can_cast(rawvis.dtype, np.complex64)
        and np.can_cast(weights.dtype, np.float32)
        and np.can_cast(init_gain, np.complex64)):
        dtype = np.complex64
    else:
        dtype = np.complex128

    def stefcal_wrapper(rawvis, weights, init_gain):
        return calprocs.stefcal(rawvis, num_ants, corrprod_lookup, weights, ref_ant, init_gain,
                                *args, **kwargs)
    return da.atop(stefcal_wrapper, out_dims,
                   rawvis, rawvis_dims, weights, weights_dims, init_gain, init_gain_dims,
                   concatenate=True, new_axes={'l': num_ants}, dtype=dtype)


def _wavg_fallback(data, flags, weights, axis):
    """Default implementation of :func:`wavg`, for cases where the numba
    implementation doesn't match.
    """
    flagged_weights = da.where(calprocs.asbool(flags), weights.dtype.type(0), weights)
    weighted_data = data * flagged_weights
    # Clear the elements that have a nan anywhere
    isnan = da.isnan(weighted_data)
    weighted_data = da.where(isnan, weighted_data.dtype.type(0), weighted_data)
    flagged_weights = da.where(isnan, flagged_weights.dtype.type(0), flagged_weights)
    vis = da.sum(weighted_data, axis=axis) / da.sum(flagged_weights, axis=axis)
    return vis


def wavg(data, flags, weights, times=False, axis=0):
    """
    Perform weighted average of data, applying flags,
    over specified axis

    Parameters
    ----------
    data    : array of complex
    flags   : array of uint8 or boolean
    weights : array of floats
    times   : array of times. If times are given, average times are returned
    axis    : axis to average over

    Returns
    -------
    vis, times : weighted average of data and, optionally, times
    """
    # TODO: reintegrate this with dask
    # if data.ndim == 4 and axis in (0, 1):
    #     sum_shape = data.shape[:axis] + data.shape[axis + 1:]
    #     vis = _wavg(*np.broadcast_arrays(data, flags, weights), axis=axis, sum_shape=sum_shape)
    # else:
    vis = _wavg_fallback(data, flags, weights, axis)
    return vis if times is False else (vis, np.average(times, axis=axis))


def wavg_full(data, flags, weights, threshold=0.3):
    """
    Perform weighted average of data, flags and weights,
    applying flags, over axis 0.

    Parameters
    ----------
    data       : array of complex
    flags      : array of uint8 or boolean
    weights    : array of floats

    Returns
    -------
    av_data    : weighted average of data
    av_flags   : weighted average of flags
    av_weights : weighted average of weights
    """

    bool_flags = calprocs.asbool(flags)
    flagged_weights = da.where(bool_flags, weights.dtype.type(0), weights)
    weighted_data = data * flagged_weights
    # Clear the elements that have a nan anywhere
    isnan = da.isnan(weighted_data)
    weighted_data = da.where(isnan, weighted_data.dtype.type(0), weighted_data)
    flagged_weights = da.where(isnan, flagged_weights.dtype.type(0), flagged_weights)
    av_weights = da.sum(flagged_weights, axis=0)
    av_data = da.sum(weighted_data, axis=0) / av_weights
    n_flags = da.sum(bool_flags, axis=0)
    av_flags = n_flags > flags.shape[0] * threshold
    return av_data, av_flags, av_weights


def wavg_full_t(data, flags, weights, solint, times=None):
    """
    Perform weighted average of data, flags and weights,
    applying flags, over axis 0, for specified
    solution interval increments

    Parameters
    ----------
    data       : array of complex
    flags      : array of boolean
    weights    : array of floats
    solint     : index interval over which to average, integer
    times      : optional array of times to average, array of floats

    Returns
    -------
    av_data    : weighted average of data
    av_flags   : weighted average of flags
    av_weights : weighted average of weights
    av_times   : optional average of times
    """
    # ensure solint is an intager
    solint = np.int(solint)
    inc_array = range(0, data.shape[0], solint)

    av_data = []
    av_flags = []
    av_weights = []
    # TODO: might be more efficient to use reduceat?
    for ti in inc_array:
        w_out = wavg_full(data[ti:ti+solint], flags[ti:ti+solint], weights[ti:ti+solint])
        av_data.append(w_out[0])
        av_flags.append(w_out[1])
        av_weights.append(w_out[2])
    av_data = da.stack(av_data)
    av_flags = da.stack(av_flags)
    av_weights = da.stack(av_weights)

    if np.any(times):
        av_times = np.array([np.average(times[ti:ti+solint], axis=0) for ti in inc_array])
        return av_data, av_flags, av_weights, av_times
    else:
        return av_data, av_flags, av_weights
