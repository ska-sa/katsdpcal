"""
Calibration procedures for MeerKAT calibration pipeline
=======================================================

Solvers and averagers for use in the MeerKAT calibration pipeline. The
functions in this module generally expect and return :class:`dask.Array`s
rather than numpy arrays.
"""

import logging
import operator

import numpy as np
import dask
import dask.array as da
from dask.highlevelgraph import HighLevelGraph

from . import calprocs


logger = logging.getLogger(__name__)


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
    rawvis_dims[-1] = -1
    weights_dims = list(reversed(range(weights.ndim)))
    weights_dims[-1] = -2
    init_gain_dims = list(reversed(range(init_gain.ndim)))
    init_gain_dims[-1] = -3
    out_dims = list(reversed(range(max(rawvis.ndim, weights.ndim, init_gain.ndim))))
    out_dims[-1] = -4

    # Determine the output dtype, since the gufunc has two signatures
    if (np.can_cast(rawvis.dtype, np.complex64)
            and np.can_cast(weights.dtype, np.float32)
            and np.can_cast(init_gain.dtype, np.complex64)):
        dtype = np.complex64
    else:
        dtype = np.complex128

    def stefcal_wrapper(rawvis, weights, init_gain):
        return calprocs.stefcal(rawvis, num_ants, corrprod_lookup, weights, ref_ant, init_gain,
                                *args, **kwargs)
    return da.blockwise(stefcal_wrapper, out_dims,
                        rawvis, rawvis_dims, weights, weights_dims, init_gain, init_gain_dims,
                        concatenate=True, new_axes={-4: num_ants}, dtype=dtype)


def where(condition, x, y):
    """Reimplementation of :func:`da.where` that doesn't suffer from
    https://github.com/dask/dask/issues/2526, and is also faster. It
    may not be as fully featured, however.
    """
    return da.core.elemwise(np.where, condition, x, y)


def divide_weights(weighted_data, weights):
    """Divide weighted_data by weights, suppress divide by zero errors"""
    # Suppress divide by zero warnings by replacing zeros with ones
    # all zero weight data will already be set to zero in weighted_data
    weights_nozero = where(weights == 0, weights.dtype.type(1), weights)
    return weighted_data / weights_nozero


def _wavg_axis(weighted_data, weights, axis=0):
    """Weighted average and weights along an axis"""
    av_weights = da.sum(weights, axis)
    sum_data = da.sum(weighted_data, axis)
    av_data = divide_weights(sum_data, av_weights)
    return av_data, av_weights


def weight_data(data, flags, weights):
    """Return flagged, weighted data and flagged weights.

    Data that are zero, weights that are zero or unfeasibly high
    are all set to zero in the output arrays

    Parameters
    ----------
    data    : array of complex
    flags   : array of uint8 or boolean
    weights : array of floats

    Returns
    -------
    weighted_data : array of complex
    flagged_weights : array of floats
    """
    # Suppress comparison with nan warnings by replacing nans with zeros
    flagged_weights = where(calprocs.asbool(flags) | da.isnan(weights),
                            weights.dtype.type(0), weights)
    weighted_data = data * flagged_weights
    # Clear all invalid elements, ie. nans, zeros and high weights
    # High weights may occur due to certain corner cases when performing excision in ingest.
    # https://skaafrica.atlassian.net/browse/SPR1-291 should ensure these no longer occur, but
    # retain this check to be cautious.
    invalid = (da.isnan(weighted_data) | (weighted_data == 0) |
               (flagged_weights > calprocs.HIGH_WEIGHT))
    weighted_data = where(invalid, weighted_data.dtype.type(0), weighted_data)
    flagged_weights = where(invalid, flagged_weights.dtype.type(0), flagged_weights)
    return weighted_data, flagged_weights


def wavg(data, flags, weights, times=False, axis=0):
    """Perform weighted average of data, applying flags, over specified axis.

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
    weighted_data, flagged_weights = weight_data(data, flags, weights)
    vis, av_weights = _wavg_axis(weighted_data, flagged_weights, axis)
    return vis if times is False else (vis, np.average(times, axis=axis))


def wavg_full(data, flags, weights, axis=0, threshold=0.8):
    """Perform weighted average of data, flags and weights, applying flags, over axis.

    Parameters
    ----------
    data       : array of complex
    flags      : array of uint8 or boolean
    weights    : array of floats
    axis       : int
    threshold  : int

    Returns
    -------
    av_data    : weighted average of data
    av_flags   : weighted average of flags
    av_weights : weighted average of weights
    """
    weighted_data, flagged_weights = weight_data(data, flags, weights)

    av_data, av_weights = _wavg_axis(weighted_data, flagged_weights, axis)
    # Update flags to include all invalid data, ie vis = 0j and weights > 1e15
    updated_flags = flagged_weights == 0
    n_flags = da.sum(updated_flags, axis)

    av_flags = n_flags >= flags.shape[axis] * threshold
    return av_data, av_flags, av_weights


def wavg_full_t(data, flags, weights, solint, times=None, threshold=0.8):
    """Perform weighted average of data, flags and weights, over axis 0.

    This applies flags and uses specified solution interval increments.

    Parameters
    ----------
    data       : array of complex
    flags      : array of boolean
    weights    : array of floats
    solint     : index interval over which to average, integer
    times      : optional array of times to average, array of floats
    threshold  : optional float

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
        w_out = wavg_full(data[ti:ti+solint], flags[ti:ti+solint], weights[ti:ti+solint],
                          threshold=threshold)
        av_data.append(w_out[0])
        av_flags.append(w_out[1])
        av_weights.append(w_out[2])
    av_data = da.stack(av_data)
    av_flags = da.stack(av_flags)
    av_weights = da.stack(av_weights)

    if times is not None:
        av_times = np.array([np.average(times[ti:ti+solint], axis=0) for ti in inc_array])
        return av_data, av_flags, av_weights, av_times
    else:
        return av_data, av_flags, av_weights


def av_blocks(data, blocksize):
    """Calculate the mean in blocks of a fixed size over axis 0.

    Parameters
    ----------
    data : :class:`da.Array`
        data to average
    blocksize : int
        size of blocks to average

    Returns
    -------
    av_data : :class:`da.Array`
    """
    inc = np.arange(0, data.shape[0], blocksize)
    sum_data = []
    for i in inc:
        inc_data = da.sum(data[i:i+blocksize, ...], axis=0, dtype=data.dtype)
        sum_data.append(inc_data)
    av_data = da.stack(sum_data, axis=0) / np.float32(blocksize)
    return av_data


def _align_chunks(chunks, alignment):
    """Compute a new chunking scheme where chunk boundaries are aligned.

    `chunks` must be a dask chunks specification in normalised form.
    `alignment` is a dictionary mapping axes to an alignment factor. The return
    value is a new chunking scheme where all chunk sizes, except possibly the
    last on each axis, is a multiple of that alignment.

    The implementation tries to minimize the cost of rechunking to the new
    scheme, while also minimising the number of chunks. Within each existing
    chunk, the first and last alignment boundaries are split along (which may
    be a no-op where the start/end of the chunk is already aligned).
    """
    out = list(chunks)
    for axis, align in alignment.items():
        sizes = []
        in_pos = 0       # Sum of all processed incoming sizes
        out_pos = 0      # Sum of generated sizes
        for c in chunks[axis]:
            in_end = in_pos + c
            low = (in_pos + align - 1) // align * align    # first aligned point
            if low > out_pos and low <= in_end:
                sizes.append(low - out_pos)
                out_pos = low
            high = in_end // align * align             # last aligned point
            if high > out_pos and high >= in_pos:
                sizes.append(high - out_pos)
                out_pos = high
            in_pos = in_end
        # May be a final unaligned piece
        if out_pos < in_pos:
            sizes.append(in_pos - out_pos)
        out[axis] = tuple(sizes)
    return tuple(out)


def wavg_full_f(data, flags, weights, chanav, threshold=0.8):
    """Perform weighted average of data, flags and weights, over axis -3.

    This applies flags and uses the specified number of channels.

    Parameters
    ----------
    data       : array of complex
    flags      : array of boolean
    weights    : array of floats
    chanav     : number of channels over which to average, integer

    Returns
    -------
    av_data    : weighted average of data
    av_flags   : weighted average of flags
    av_weights : weighted average of weights
    """
    # We rechunk (if needed) to get blocks that are multiples of chanav
    # long, then apply the non-dask version of wavg_full_f per block.
    # This would be simple with da.core.map_blocks, but it doesn't
    # support multiple outputs, so we need to do manual construction
    # of the dask graphs.
    chunks = _align_chunks(data.chunks, {1: chanav})
    out_chunks = list(chunks)
    # Divide by chanav, rounding up
    # use axis -3 for freq, to support cases where time axis has been averaged away
    out_chunks[-3] = tuple((x + chanav - 1) // chanav for x in chunks[-3])
    out_chunks = tuple(out_chunks)

    data = data.rechunk(chunks)
    flags = flags.rechunk(chunks)
    weights = weights.rechunk(chunks)

    token = da.core.tokenize(data, flags, weights, chanav, threshold)
    base_name = 'wavg_full_f-' + token
    keys = list(dask.core.flatten(data.__dask_keys__()))

    base_layer = {
        (base_name,) + key[1:]: (calprocs.wavg_full_f, key,
                                 (flags.name,) + key[1:],
                                 (weights.name,) + key[1:],
                                 chanav, threshold)
        for key in keys
    }
    base_graph = HighLevelGraph.from_collections(base_name, base_layer, [data, flags, weights])

    def sub_array(name, idx, dtype):
        layer = {
            (name,) + key[1:]: (operator.getitem, (base_name,) + key[1:], idx)
            for key in keys
        }
        layers = dict(base_graph.layers)
        layers[name] = layer
        dependencies = dict(base_graph.dependencies)
        dependencies[name] = {base_name}
        dsk = HighLevelGraph(layers, dependencies)
        return da.Array(dsk, name, out_chunks, dtype)

    av_data = sub_array('wavg_full_f-data-' + token, 0, data.dtype)
    av_flags = sub_array('wavg_full_f-flags-' + token, 1, flags.dtype)
    av_weights = sub_array('wavg_full_f-weights-' + token, 2, weights.dtype)
    return av_data, av_flags, av_weights


def wavg_ant(data, flags, weights, ant_array, bls_lookup, threshold=0.8):
    """Perform weighted average of data, flags and weights, over axis -1.

    This applies flags and is done per antenna.

    Parameters
    ----------
    data : :class:`da.Array`
        complex (..., bls)
    flags : :class:`da.Array`
        int (..., bls)
    weights : :class:`da.Array`
        real (..., bls)
    ant_array : :class:`np.ndarray`
        array of strings representing antennas
    bls_lookup : :class:`np.ndarray`
        (bls x 2) array of antennas in each baseline
    threshold : float
        if fraction of flags in the input data array
        exceeds threshold then set output flag to True, else False

    Returns
    -------
    av_data : :class:`da.Array`
        complex (..., n_ant), weighted average of data
    av_flags : :class:`da.Array`
        bool (n_ant), weighted average of flags
    av_weights : :class:`da.Array`
        real (..., n_ant), weighted average of weights
    """
    av_data = []
    av_flags = []
    av_weights = []

    weighted_data, flagged_weights = weight_data(data, flags, weights)
    for ant in range(len(ant_array)):
        # select all correlations with same antenna but ignore autocorrelations
        ant_idx = np.where((bls_lookup[:, 0] == ant)
                           ^ (bls_lookup[:, 1] == ant))[0]

        # conjugate visibilities if antenna is 2nd on the baseline
        ant_data = weighted_data[..., ant_idx]
        ant1 = bls_lookup[ant_idx][:, 0] == ant
        ant1 = np.broadcast_to(ant1, ant_data.shape)
        ant_conj_data = where(ant1, ant_data, da.conj(ant_data))

        ant_ave_data, ant_weights = _wavg_axis(ant_conj_data,
                                               flagged_weights[..., ant_idx],
                                               axis=-1)
        # update flags to include all invalid data
        updated_flags = flagged_weights[..., ant_idx] == 0
        n_flags = da.sum(updated_flags, axis=-1)
        ant_flags = n_flags > ant_idx.shape[0] * threshold

        av_data.append(ant_ave_data)
        av_flags.append(ant_flags)
        av_weights.append(ant_weights)

    av_data = da.stack(av_data, axis=-1)
    av_flags = da.stack(av_flags, axis=-1)
    av_weights = da.stack(av_weights, axis=-1)

    return av_data, av_flags, av_weights


def wavg_t_f(data, flags, weights, nchans):
    """Perform weighted average, over all times and in frequency blocks.

    This averages data, flags and weights, forming a product with nchans
    channels. If nchans is less than the number of channels in the data,
    don't average in frequency.

    Parameters:
    -----------
    data : :class:`da.Array`
        complex (ntimes, nchans, npols, bls)
    flags : :class:`da.Array`
        int (ntimes, nchans, npols, bls)
    weights : :class:`da.Array`
        real (ntimes, nchans, npols, bls)
    nchans : int
        number of channels in averaged product, if data has less channels than nchans,
        don't average

    Returns:
    --------
    av_data : :class:`da.Array`
        complex (..., n_ant), weighted average of data
    av_flags : :class:`da.Array`
        bool (..., n_ant), weighted average of flags
    av_weights : :class:`da.Array`
        real (..., n_ant), weighted average of weights
    """
    # Average over all times, in frequency blocks
    orig_chans = data.shape[1]
    av_data, av_flags, av_weights = wavg_full(data, flags, weights, threshold=1)
    chanav = max(1, orig_chans // nchans)
    if chanav > 1:
        av_data, av_flags, av_weights = wavg_full_f(av_data, av_flags, av_weights,
                                                    chanav, threshold=1)
    return av_data, av_flags, av_weights


def bp_fit(data, weights, corrprod_lookup, bp0=None, refant=0, **kwargs):
    """Fit bandpass to visibility data.

    Parameters
    ----------
    data : array of complex, shape(num_chans, num_pols, baselines)
    weights : array of real, shape(num_chans, num_pols, baselines)
    bp0 : array of complex, shape(num_chans, num_pols, num_ants) or None
    corrprod_lookup : antenna mappings, for first then second antennas in bl pair
    refant : reference antenna

    Returns
    -------
    bpass : Bandpass, shape(num_chans, num_pols, num_ants)
    """
    n_ants = calprocs.ants_from_bllist(corrprod_lookup)

    # -----------------------------------------------------
    # solve for the bandpass over the channel range
    bp = stefcal(data, n_ants, corrprod_lookup, weights, refant, num_iters=100,
                 init_gain=bp0, **kwargs)
    return bp
