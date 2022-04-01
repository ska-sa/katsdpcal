"""
Calibration procedures for MeerKAT calibration pipeline
=======================================================

Solvers and averagers for use in the MeerKAT calibration pipeline.
"""

import ast
import logging

import numpy as np
import scipy.fftpack
import numba

import katpoint

from katdal.applycal import complex_interp

logger = logging.getLogger(__name__)

# threshold for bad/high weights
HIGH_WEIGHT = 1e15
# --------------------------------------------------------------------------------------------------
# --- Modelling procedures
# --------------------------------------------------------------------------------------------------


def radec_to_lm(ra, dec, ra0, dec0):
    """Obtain direction cosines of (ra, dec) relative to (ra0, dec0).

    Parameters
    ----------
    ra : float
        Right Ascension, radians
    dec : float
        Declination, radians
    ra0 : float
        Right Ascention of centre, radians
    dec0 : float
        Declination of centre, radians

    Returns
    -------
    l, m : float
        direction cosines
    """
    l = np.cos(dec)*np.sin(ra - ra0)    # noqa: E741
    m = np.sin(dec)*np.cos(dec0) - np.cos(dec)*np.sin(dec0)*np.cos(ra-ra0)
    return l, m


def list_to_model(model_list, centre_position):
    lmI_list = []

    ra0, dec0 = centre_position

    for ra, dec, I in model_list:
        l, m = radec_to_lm(ra, dec, ra0, dec0)
        lmI_list.append([l, m, I])

    return lmI_list


def to_ut(t):
    """Converts MJD seconds into Unix time in seconds.

    Parameters
    ----------
    t : float
        time in MJD seconds

    Returns
    -------
    float
        Unix time in seconds
    """
    return (t/86400. - 2440587.5 + 2400000.5)*86400.


def calc_uvw_wave(phase_centre, timestamps, corrprod_lookup, antennas,
                  wavelengths=None, array_centre=None):
    """Calculate uvw coordinates per baseline, in wavelengths / lambda.

    Parameters
    ----------
    phase_centre : :class:`katpoint.Target`
        phase centre position
    timestamps : array of float
        times, shape(nrows)
    corrprod_lookup : array
        lookup table of antenna indices for each baseline, shape(nant,2)
    antennas : list of :class:`katpoint.Antenna`
        the antennas, same order as antlist
    wavelengths : array or scalar
        wavelengths, single value or array shape(nchans)
    array_centre : :class:`katpoint.Antenna`
        array centre position

    Returns
    -------
    uvw_wave
        uvw coordinates, normalised by wavelength
    """
    uvw = calc_uvw(phase_centre, timestamps, corrprod_lookup, antennas, array_centre)
    if wavelengths is None:
        return uvw
    elif np.isscalar(wavelengths):
        return uvw/wavelengths
    else:
        return uvw[:, :, np.newaxis, :]/wavelengths[:, np.newaxis]


def calc_uvw(phase_centre, timestamps, corrprod_lookup, antennas, array_centre=None):
    """Calculate uvw coordinates per baseline, in metres.

    Parameters
    ----------
    phase_centre : :class:`katpoint.Target`
        phase centre position
    timestamps : array of float
        times, shape(nrows)
    corrprod_lookup : array
        lookup table of antenna indices for each baseline, shape(nant,2)
    antennas : list of :class:`katpoint.Antenna`
        the antennas, same order as antlist
    array_centre : str
        array centre position

    Returns
    -------
    baseline_uvw
        uvw coordinates
    """
    if array_centre is None:
        # if no array centre position is given, use lat-long-alt of first
        # antenna in the antenna list
        array_centre = katpoint.Antenna('array_position', *antennas[0].ref_position_wgs84)

    # use the array reference position of antenna_uvw
    antenna_uvw = np.array(phase_centre.uvw(antennas, timestamps, array_centre))

    baseline_uvw = np.empty([3, len(timestamps), len(corrprod_lookup)])
    for i, [a1, a2] in enumerate(corrprod_lookup):
        baseline_uvw[..., i] = antenna_uvw[..., a1] - antenna_uvw[..., a2]

    return baseline_uvw

# --------------------------------------------------------------------------------------------------
# --- Solvers
# --------------------------------------------------------------------------------------------------


def get_bl_ant_pairs(corrprod_lookup):
    """Get antenna lists in solver format, from `corrprod_lookup`.

    Inputs:
    -------
    corrprod_lookup : array
        lookup table of antenna indices for each baseline, array shape(nant,2)

    Returns:
    --------
    antlist1, antlist2 : list
        lists of antennas matching the correlation_product lookup table,
        appended with their conjugates (format required by stefcal)
    """
    # NOTE: no longer need hh and vv masks as we re-ordered the data to be
    # ntime x nchan x nbl x npol

    # get antenna number lists for stefcal - need vis then vis.conj (assume
    # constant over an observation)
    # assume same for hh and vv
    antlist1 = np.concatenate((corrprod_lookup[:, 0], corrprod_lookup[:, 1]))
    antlist2 = np.concatenate((corrprod_lookup[:, 1], corrprod_lookup[:, 0]))

    return antlist1, antlist2


@numba.guvectorize(['c8[:], int_[:], int_[:], f4[:], int_[:], c8[:], int_[:], f4[:], c8[:]',
                    'c16[:], int_[:], int_[:], f8[:], int_[:], c16[:], int_[:], f8[:], c16[:]'],
                   '(n),(n),(n),(n),(),(a),(),()->(a)', nopython=True, cache=True)
def _stefcal_gufunc(rawvis, ant1, ant2, weights, ref_ant, init_gain, num_iters, conv_thresh, g):
    ref_ant2 = max(ref_ant[0], 0)
    num_ants = init_gain.shape[0]
    R = np.zeros((num_ants, num_ants), rawvis.dtype)    # Weighted visibility matrix
    M = np.zeros((num_ants, num_ants), weights.dtype)   # Weighted model
    # Convert list of baselines into a covariance matrix
    for i in range(len(ant1)):
        a = ant1[i]
        b = ant2[i]
        if a != b:
            weighted_vis = rawvis[i] * weights[i]
            R[a, b] = weighted_vis
            R[b, a] = np.conj(weighted_vis)
            M[a, b] = weights[i]
            M[b, a] = weights[i]
    g_old = init_gain.copy()

    # Remove completely flagged antennas,
    # where flags are indicated by a weighted visibility of zero
    good_ant = np.ones(num_ants, np.bool_)
    for p in range(num_ants):
        good_ant[p] = ~np.all(R[p] == 0j)

    antlist = np.arange(num_ants)[good_ant]  # list of good antennas to iterate over.
    for n in range(num_iters[0]):
        for p in antlist:
            gn = g.dtype.type(0)
            gd = g.real.dtype.type(0)
            for i in antlist:
                z = g_old[i] * M[p, i]
                gn += R[p, i] * z       # exploiting R[p, i] = R[i, p].conj()
                gd += z.real * z.real + z.imag * z.imag
            g[p] = gn / gd
        # Salvini & Wijnholds tweak gains every *even* iteration but their counter starts at 1
        if n % 2:
            # Check for convergence of relative l_2 norm of change in gain vector
            dnorm2 = g.real.dtype.type(0)
            gnorm2 = g.real.dtype.type(0)
            for i in antlist:
                delta = g[i] - g_old[i]
                dnorm2 += delta.real * delta.real + delta.imag * delta.imag
                gnorm2 += g[i].real * g[i].real + g[i].imag * g[i].imag
            # The sense of this if test is carefully chosen so that if
            # g contains nans, the loop will exit (there is no point
            # continuing, since the nans can only spread.
            if dnorm2 >= conv_thresh[0] * conv_thresh[0] * gnorm2:
                # Avoid getting stuck bouncing between two gain vectors by
                # going halfway in between
                for i in antlist:
                    g[i] = (g[i] + g_old[i]) / 2
            else:
                break
        g_old[:] = g

    # replace gains of completely flagged antennas with NaNs to indicate
    # that no gain was found by stefcal.
    g[~good_ant] = np.nan*1j

    ref = g[ref_ant2]
    g *= np.conj(ref) / np.abs(ref)

    if ref_ant[0] < 0:
        middle_angle = np.median(g.real) - np.median(g.imag) * g.dtype.type(1j)
        middle_angle /= np.abs(middle_angle)
        g *= middle_angle


def stefcal(rawvis, num_ants, corrprod_lookup, weights=None, ref_ant=0,
            init_gain=None, num_iters=30, conv_thresh=0.0001):
    """Solve for antenna gains using StEFCal.

    The observed visibilities are provided in a NumPy array of any shape and
    dimension, as long as the last dimension represents baselines. The gains
    are then solved in parallel for the rest of the dimensions. For example,
    if the *vis* array has shape (T, F, B) containing *T* dumps / timestamps,
    *F* frequency channels and *B* baselines, the resulting gain array will be
    of shape (T, F, num_ants), where *num_ants* is the number of antennas.

    Parameters
    ----------
    rawvis : array of complex, shape (M, ..., N)
        Complex cross-correlations between antennas A and B, assuming *N*
        baselines or antenna pairs on the last dimension
    num_ants : int
        Number of antennas
    corrprod_lookup : numpy array of int, shape (N, 2)
        First and second antenna indices associated with visibilities
    weights : float or array of float, shape (M, ..., N), optional
        Visibility weights (positive real numbers)
    ref_ant : int, optional
        Reference antenna for which phase will be forced to 0.0. Alternatively,
        if *ref_ant* is -1, the median gain phase will be 0.
    init_gain : array of complex, shape(num_ants,), optional
        Initial gain vector (all equal to 1.0 by default)
    num_iters : int, optional
        Number of iterations
    conv_thresh : float, optional
        Convergence threshold (max relative l_2 norm of change in gain vector)

    Returns
    -------
    gains : array of complex, shape (M, ..., num_ants)
        Complex gains per antenna

    Notes
    -----
    The model visibilities are assumed to be 1, implying a point source model.
    The algorithm is iterative but should converge in a small number of
    iterations (10 to 30).

    References
    ----------
    .. [1] Salvini, Wijnholds, "Fast gain calibration in radio astronomy using
       alternating direction implicit methods: Analysis and applications," 2014,
       preprint at `<http://arxiv.org/abs/1410.2101>`_
    """
    if init_gain is None:
        init_gain = np.ones(num_ants, rawvis.dtype)
    if init_gain.shape[-1] != num_ants:
        raise ValueError('initial gains have wrong length {} for number of antennas {}'.format(
            init_gain.shape[-1], num_ants))
    if weights is None or np.isscalar(weights):
        # Any scalar weight is the same as an unweighted problem, and we don't
        # want a double-precision scalar weight to cause the result to be
        # upgraded to double precision.
        weights = rawvis.real.dtype.type(1.0)
    if not all(0 <= x < num_ants for x in corrprod_lookup.flat):
        raise ValueError('invalid antenna index in corrprod_lookup')
    if ref_ant >= num_ants:
        raise ValueError('invalid reference antenna')
    weights = np.broadcast_to(weights, rawvis.shape)
    return _stefcal_gufunc(rawvis, corrprod_lookup[:, 0], corrprod_lookup[:, 1], weights,
                           ref_ant, init_gain, num_iters, conv_thresh)


# --------------------------------------------------------------------------------------------------
# --- Calibration helper functions
# --------------------------------------------------------------------------------------------------

def g_from_K(chans, K):
    g_array = np.ones(K.shape+(len(chans),), dtype=np.complex)
    for i, c in enumerate(chans):
        g_array[:, :, i] = np.exp(1.0j * 2. * np.pi * K * c)
    return g_array


def ants_from_bllist(bllist):
    return len(set([item for sublist in bllist for item in sublist]))


def best_refant(data, corrprod_lookup, chans):
    """Identify antenna that is most suited to be the reference antenna.

    Determine antenna whose FFT has the maximum peak to noise ratio (PNR) by
    taking the median PNR of the FFT over all baselines to each antenna.

    Parameters
    ----------
    data : :class:`np.ndarray`, complex, shape (n_chans, n_pols, n_bls)
        Visibility data
    corrprod_lookup : :class:`np.ndarray`, int, shape (2, n_bls)
        Antenna pairs for selected baselines
    chans : :class:`np.ndarray`
        Channel frequencies in Hz

    Returns
    -------
    best_refant : :class:`np.ndarray`
        Array of indices of antennas in decreasing order of median of PNR over all baselines
    """
    # Detect position of fft peak
    ft_vis = scipy.fftpack.fft(data, axis=0)
    k_arg = np.argmax(np.abs(ft_vis), axis=0)
    # Shift data so that peak of fft is always positioned at the first index of the array
    n_chans, n_pols, n_bls = data.shape
    index = np.array([np.roll(range(n_chans), -n) for n in k_arg.ravel()])
    index = index.T.reshape(ft_vis.shape)
    ft_vis = np.take_along_axis(ft_vis, index, axis=0)

    # Calculate the height of the peak and the standard deviation away from the peak
    peak = np.max(np.abs(ft_vis), axis=0)
    chan_slice = np.s_[len(chans)//2-len(chans)//4:len(chans)//2+len(chans)//4]
    mean = np.mean(np.abs(ft_vis[chan_slice]), axis=0)
    # Add 1e-9 to avoid divide by zero errors
    std = np.std(np.abs(ft_vis[chan_slice]), axis=0) + 1e-9

    # Calculate the median of the pnr for all baselines per antenna
    num_ants = ants_from_bllist(corrprod_lookup)
    med_pnr_ants = np.zeros(num_ants)
    for a in range(num_ants):
        mask = (corrprod_lookup[:, 0] == a) ^ (corrprod_lookup[:, 1] == a)
        # NB: it's important that mask is an np.ndarray here and not a list,
        # due to https://github.com/numpy/numpy/pull/13715
        pnr = (peak[..., mask] - mean[..., mask]) / std[..., mask]
        med_pnr_ants[a] = np.median(pnr)
    return np.argsort(med_pnr_ants)[::-1]


def g_fit(data, weights, corrprod_lookup,  g0=None, refant=0, **kwargs):
    """Fit complex gains to visibility data.

    Parameters
    ----------
    data : visibility data, array of complex, shape(num_sol, num_chans, baseline)
    weights : weight data, array of real, shape(num_sol, num_chans, baseline)
    g0 : array of complex, shape(num_ants) or None
    corrprod_lookup : antenna mappings, for first then second antennas in bl pair
    refant : reference antenna

    Returns
    -------
    g_array : Array of gain solutions, shape(num_sol, num_ants)
    """
    num_ants = ants_from_bllist(corrprod_lookup)
    return stefcal(data, num_ants, corrprod_lookup, weights,
                   ref_ant=refant, init_gain=g0, **kwargs)


def k_fit(data, weights, corrprod_lookup, chans, refant=0, cross=True, chan_sample=1):
    """Fit delay (phase slope across frequency) to visibility data.

    If corrprod_lookup is xc, i.e. all baselines have two different antenna indices,
    it will decompose the delays into solutions per antenna.
    Else it solves for a delay per element in final axis.

    Parameters
    ----------
    data : array of complex, shape (num_chans, num_pols, num_baselines)
        Visibility data (may contain NaNs indicating completely flagged data)
    weights : array of real, shape (num_sol, num_chans, baseline)
        Weight data, must be zero for flagged data and NaNed data
    corrprod_lookup : array of int, shape (num_baselines/num_ant, 2)
        Pairs of antenna indices associated with each baseline
    chans : sequence of float, length num_chans
        Channel frequencies in Hz
    refant : int, optional
        Reference antenna index
    cross: bool, optional
        default assume data are cross-correlations per baseline, else assume data are
        auto-correlations per antenna
    chan_sample : int, optional
        Subsample channels by this amount, i.e. use every n'th channel in fit

    Returns
    -------
    kdelay : array of float, shape (num_pols, num_ants/data.shape[-1])
        Delay solutions per antenna, in seconds (NaN if fit failed)
    """
    # OVER THE TOP NOTE:
    # This solver was written to deal with extreme gains, which can wrap many
    # many times across the solution window, to the extent that solving for a
    # per-channel per-antenna bandpass, then fitting a line to the per-antenna
    # bandpass breaks down. It breaks down because the phase unwrapping is
    # unreliable when the phases (a) wrap so extremely, and (b) are also
    # noisy.
    #
    # This situation was adressed by first doing a coarse delay fit through an
    # FFT, for each antenna to the reference antenna. Then the course delay is
    # applied, and a secondary linear phase fit is performed with the corrected
    # data. With the coarse delays removed there should be little, or minimal
    # wrapping, and the unwrapping prior to the linear fit usually works.
    # BUT:
    # * This algorithm is most effective if the delays are of similar scales.
    #   How to ensure this is to choose a reference antenna that has high delays
    #   on as many baselines as possible.  Perhaps this could be done by looking
    #   through all of the FFT results, grouped by antenna, to find the highest
    #   set of delays?
    # * In the future world where not every delay fit needs to deal with
    #   extreme delay values, maybe we could separate out the
    #   bandpass-linear-phase-fit component of this algorithm, and use just that
    #   for delay solutions we know won't be too extreme. Then this extreme delay
    #  solver could solve for the coarse delay then apply it and call the fine
    #  delay bandpass-linear-phase-fit solver.

    # -----------------------------------------------------
    chans = np.asarray(chans, dtype=np.float32)
    # if channel sampling is specified, thin down the data and channel list
    if chan_sample != 1:
        data = data[::chan_sample, ...]
        weights = weights[::chan_sample, ...]
        chans = chans[::chan_sample]

    # set up parameters
    num_pol = data.shape[-2] if len(data.shape) > 2 else 1
    chan_spacing = chans[1] - chans[0]

    # -----------------------------------------------------
    # NOTE: I know that iterating over polarisation is horrible, but I ran out
    # of time to fix this up I initially just assumed we would always have two
    # polarisations, but shouldn't make that assumption especially for use of
    # calprocs offline on an h5 file
    kdelay = []
    for p in range(num_pol):
        pol_data = data[:, p, :] if len(data.shape) > 2 else data
        pol_weights = weights[:, p, :] if len(weights.shape) > 2 else weights
        # Suppress NaNs by setting them to zero. This masking step broadens
        # the delay peak in Fourier space and potentially introduces spurious
        # sidelobes if severe, like a dirty image suffering from poor uv coverage.
        good_pol_data = np.nan_to_num(pol_data)
        # -----------------------------------------------------
        # FT to find visibility space delays
        # NOTE: This is a bit inefficient at the moment as the FFT for al
        # baselines is calculated but only the baselines to the reference
        # antenna are used for the coarse delay
        # Also note that scipy.fftpack will do a single-precision FFT given
        # single-precision input, unlike np.fft.fft which converts it to
        # double precision.
        # NB: The coarse delay part assumes regularly spaced frequencies in chans
        ft_vis = scipy.fftpack.fft(good_pol_data, axis=0)
        # get index of FT maximum
        k_arg = np.argmax(np.abs(ft_vis), axis=0)
        # calculate vis space K from FT sample frequencies
        vis_k = np.float32(np.fft.fftfreq(ft_vis.shape[0], chan_spacing)[k_arg])

        # test whether data is xc
        if cross:
            # now determine per-antenna K values
            num_ants = ants_from_bllist(corrprod_lookup)
            coarse_k = np.zeros(num_ants, np.float32)
            for ai in range(num_ants):
                k = vis_k[..., (corrprod_lookup == (ai, refant)).all(axis=1)]
                if k.size > 0:
                    coarse_k[ai] = np.squeeze(k)
                k = vis_k[..., (corrprod_lookup == (refant, ai)).all(axis=1)]
                if k.size > 0:
                    coarse_k[ai] = np.squeeze(-1.0 * k)

            # apply coarse K values to the data and solve for bandpass
            # The baseline delay is calculated as delay(ant2) - delay(ant1)
            bl_delays = np.diff(coarse_k[corrprod_lookup])
            good_pol_data *= np.exp(2j * np.pi * np.outer(chans, bl_delays))
            bpass = stefcal(good_pol_data, num_ants, corrprod_lookup, pol_weights,
                            num_iters=100, ref_ant=refant, init_gain=None)

            # set weight for NaNed stefcal outputs to zero
            bpass_weights = (~np.isnan(bpass)).astype(np.float32)

        # else assume solution is ac
        else:
            coarse_k = vis_k
            bpass = pol_data * np.exp(-2j * np.pi * np.outer(chans, coarse_k))
            bpass_weights = pol_weights

        # Find slope of the residual bandpass, per antenna (defaults to NaN)
        delta_k = np.full_like(coarse_k, np.nan)
        for i, (bp, bp_weights) in enumerate(zip(bpass.T, bpass_weights.T)):
            # np.unwrap falls over in the case of bad RFI - robustify this later
            # np.unwrap might not unwrap correctly with zeroed data, thus mask
            # zero values in these routines (but skip if everything is masked)
            valid = bp_weights > 0
            if any(valid):
                bp_phase = np.unwrap(np.angle(bp[valid]), discont=1.9 * np.pi)
                freqs = chans[valid]
                A = np.array([freqs, np.ones(len(freqs))])
                # trick np.linalg.lstsq into performing a weighted least squares (WLS) fit
                # multiply each term in the fit by w to minimize w**2(bp_phase-A.T)**2
                # for a WLS fit set w**2 = bp_weights
                w = np.sqrt(bp_weights[valid])
                delta_k[i] = np.linalg.lstsq((w * A).T, w * bp_phase, None)[0][0] / (2. * np.pi)

        kdelay.append(coarse_k + delta_k)

    return np.atleast_2d(kdelay)


def normalise_complex(x, weights=None, axis=0):
    """Normalisation factor that centers phase on 0 and scales amplitude to 1.

    Calculate a (weighted) normalisation factor for a complex array across the selected
    axis to center the (weighted) phase of data on zero and scale the (weighted) average
    amplitude to one. If the selected axis is all NaN or all zero, the normalisation
    factor is given as 1+0j.

    Parameters
    ----------
    x : :class:`np.ndarray`
        data to be normalised
    weights : :class:`np.ndarray`
        weights of data to be normalised, default is unweighted
    axis : int, optional
        axis to normalise across, default is 0

    Returns
    -------
    :class:`np.ndarray`
        normalisation factor, complex
    """
    # set weights to one, if none supplied
    if weights is None:
        weights = np.ones_like(x, dtype=np.float32)
    # ensure all NaN'ed data has zero weight
    valid_weights = np.where(np.isfinite(x), weights, 0)

    # suppress warnings related to all-NaN and all-zero values on the selected axis
    # by replacing instances of all NaN and/or zero with all ones.
    all_nan = np.all((~np.isfinite(x)) | (x == 0), axis, keepdims=True)
    all_nan = np.broadcast_to(all_nan, x.shape)
    valid_x = np.where(all_nan, 1.0, x)
    valid_weights = np.where(all_nan, 1.0, valid_weights)

    angle = np.angle(valid_x)
    base_angle = np.nanmin(angle, axis, keepdims=True) - np.pi
    # angle relative to base_angle, wrapped to range [0, 2pi], with
    # some data point sitting at pi.
    rel_angle = np.fmod(angle - base_angle, 2 * np.pi)

    sum_weights = np.nansum(valid_weights, axis, keepdims=True)
    mean_angle = np.nansum(rel_angle * valid_weights, axis, keepdims=True) / sum_weights
    mid_angle = mean_angle + base_angle
    centre_rotation = np.exp(-1.0j * mid_angle)

    amp = np.abs(valid_x)
    average_amplitude = np.nansum(amp * valid_weights, axis, keepdims=True) / sum_weights
    norm_factor = centre_rotation / average_amplitude

    return norm_factor


@numba.jit(nopython=True, parallel=True)
def K_ant(uvw, l, m, wl, k_ant):    # noqa: E741
    """Calculate K-Jones term per antenna.

    Calculate the K-Jones term for a point source with the
    given position (l, m) at the given wavelengths.
    The K-Jones term is the geometrical delay in the approximate
    2D Fourier transform relationship between the complex
    visibility and the sky brightness distribution

    Parameters
    ----------
    uvw : :class:`np.ndarray`, real, shape (3, ntimes, nants)
        uvw co-ordinates of antennas
    l : float
        direction cosine, right ascension
    m : float
        direction cosine, declination
    wl : :class:`np.ndarray`, real, shape (nchans)
        wavelengths
    k_ant : :class:`np.ndarray`, complex, shape (ntimes, nchans, nants)
        array to update with K-Jones term

    Returns
    -------
    :class: `np.ndarray`, complex, shape (ntimes, nchans, nants)
        K-Jones term per antenna
    """
    n = np.sqrt(1 - l*l - m*m)
    _, ntimes, nants = uvw.shape
    nchans = wl.shape[0]

    for t in range(ntimes):
        cstep = 128
        cblocks = (nchans + cstep - 1) // cstep
        for cblock in numba.prange(cblocks):
            cstart = cblock * cstep
            cstop = min(nchans, cstart + cstep)
            for c in range(cstart, cstop):
                for a in range(nants):
                    phase = (l * uvw[0, t, a] + m * uvw[1, t, a] + (n-1) * uvw[2, t, a]) / wl[c]
                    k_ant[t, c, a] = np.exp(2j * np.pi * phase)
    return k_ant


@numba.jit(nopython=True, parallel=True)
def add_model_vis(k_ant, ant1, ant2, I, model):    # noqa: E741
    """Add model visibilities to model.

    Calculate model visibilities from the per-antenna K-Jones term and source
    Stokes I flux densities and add them to the model array.

    Parameters
    ----------
    k_ant : :class:`np.ndarray`, complex, shape (ntimes, nchans, nants)
        per-antenna K-Jones term
    ant1 : :class:`np.ndarray`, int, shape (nbls)
        index of first antenna of each baseline pair
    ant2 : :class:`np.ndarray`, int, shape (nbls)
        index of second antenna of each baseline pair
    I : :class:`np.ndarray`, real, shape (nchans)
        Stokes I source flux densities per channel
    model : :class:`np.ndarray`, complex, shape (ntimes, nchans, nbls)
        array to add model visibilities to

    Returns
    -------
    :class: `np.ndarray`, complex, shape (ntimes, nchans, nbls)
        array with model visibilities added to it
    """
    ntimes, nchans, nants = k_ant.shape
    nbls = ant1.shape[0]
    for t in range(ntimes):
        cstep = 128
        cblocks = (nchans + cstep - 1) // cstep
        for cblock in numba.prange(cblocks):
            cstart = cblock * cstep
            cstop = min(nchans, cstart + cstep)
            for c in range(cstart, cstop):
                for b in range(nbls):
                    model[t, c, b] += (k_ant[t, c, ant1[b]]
                                       * np.conj(k_ant[t, c, ant2[b]])
                                       * I[c])
    return model


def interpolate_soln(x, xi, yi):
    """Interpolate over NaNs in the first axis of a cal solution.

    Parameters
    ----------
    x : 1-D sequence of float, length *M*
        The x-coordinates at which to evaluate the interpolated values
    xi : 1-D sequence of float, length *N*
        The x-coordinates of the data points, must be sorted in ascending order
    yi : :class:`np.ndarray`, complex, shape(N, npols, nants)
        solution to interpolate over, first axis must match the length of xi

    Returns
    -------
    y_interp : :class`np.ndarray`
         interpolated solution, shape(M, npols, nants)
    """
    x = np.array(x)
    xi = np.array(xi)

    n, npols, nants = yi.shape
    ni = len(x)
    y_interp = np.zeros((ni, npols, nants), dtype=yi.dtype)
    for p in range(npols):
        for a in range(nants):
            valid = np.isfinite(yi[:, p, a])
            if np.any(valid):
                y_interp[:, p, a] = complex_interp(x,
                                                   xi[valid],
                                                   yi[:, p, a][valid])
            else:
                y_interp[:, p, a] = np.nan
    return y_interp


def asbool(arr):
    """View an array as boolean.

    If possible it simply returns a view, otherwise a copy. It works on both
    dask and numpy arrays.
    """
    if arr.dtype in (np.uint8, np.int8, np.bool_):
        return arr.view(np.bool_)
    else:
        return arr.astype(np.bool_)


def solint_from_nominal(solint, dump_period, num_times):
    """Adjust solution interval to be commensurate with dump period and scan length.

    Given nominal solint, modify it by up to 20% to optimally fit the scan length
    and dump period. Times are assumed to be contiguous.

    Parameters
    ----------
    solint      : nominal solint
    dump_period : dump period of the data
    num_times   : number of time dumps in the scan

    Returns
    -------
    nsolint     : new optimal solint
    dumps_per_solint : number of dump periods in a solution interval
    """
    if dump_period > solint:
        # case of solution interval that is shorter than dump time
        dumps_per_solint = 1
    elif dump_period * num_times < solint:
        # case of solution interval that is longer than scan
        dumps_per_solint = num_times
    else:
        # requested number of dumps per nominal solution interval
        req_dumps_per_solint = solint / dump_period

        # range for searching: nominal solint +-20%
        dumps_per_solint_low = int(np.round(req_dumps_per_solint * 0.8))
        dumps_per_solint_high = int(np.round(req_dumps_per_solint * 1.2))
        solint_check_range = np.arange(dumps_per_solint_low,
                                       dumps_per_solint_high + 1).astype(np.int)

        # compute size of final partial interval (in dumps)
        tail = (num_times - 1) % solint_check_range + 1
        delta = solint_check_range - tail

        # choose a solint to minimise the change in the final fractional solution interval
        solint_index = np.argmin(delta)
        logger.debug('solint index: {0}'.format(solint_index,))
        dumps_per_solint = solint_check_range[solint_index]
    return dumps_per_solint * dump_period, dumps_per_solint


def measure_flux(scaled_solns, unscaled_solns, start_time, end_time):
    """Derive flux of gain calibrators based on ones with known flux models.

    Take the ratio between the median over time of solutions scaled
    to 1.0 Jy (unscaled_solns) per target and the median over time of
    solutions that were scaled by a flux density model (scaled_solns).
    The median of this ratio across all antennas and polarisations
    for each target is the factor which scales the unscaled solutions
    from 1.0 Jy to the estimated flux density of the target. The square
    of the ratio is the measured flux density of the target which is
    logged and returned.

    This method of flux calibration closely follows that used by
    the 'fluxscale' task in CASA and described at:
    https://casa.nrao.edu/docs/TaskRef/fluxscale-task.html

    Parameters
    ----------
    scaled_solns : :class:`CalSolutionStore`
        A solution store containing gains that have been scaled using
        a prior flux density model.
    unscaled_solns : :class:`CalSolutionStore`
        A solution store containing gains that have been scaled to 1.0 Jy.
    start_time, end_time : float
        Use gain solutions found between start and end time.

    Returns
    -------
    measured_flux, measured_flux_std : dict mapping string to float
        Measured flux density and standard error, per target name in unscaled_solns
    """
    scaled_amp = np.abs(scaled_solns.get_range(start_time, end_time).values)
    targets = set([soln.target for soln in unscaled_solns._values])

    # If we don't have solutions from a model we can't do any scaling.
    if len(scaled_amp) == 0:
        return {}, {}

    # Get the median of all of the scaled amplitudes over time
    median_scaled_amp = np.nanmedian(scaled_amp, axis=0)
    measured_flux = {}
    measured_flux_std = {}
    for targ in targets:
        targ_solns = unscaled_solns.get_range(start_time, end_time, target=targ)
        median_targ_amp = np.nanmedian(np.abs(targ_solns.values), axis=0)
        # Ensure the time median of the scaled amplitudes and
        # target amplitudes are the same shape so they can be divided safely
        if median_scaled_amp.shape != median_targ_amp.shape:
            logger.warning('Gains for flux calibrators and %s have different shape. '
                           '%s != %s. Skipping flux calibration on %s.',
                           targ, median_scaled_amp.shape, median_targ_amp.shape, targ)
            continue
        soln_ratio = median_targ_amp / median_scaled_amp
        flux_scale = np.nanmedian(soln_ratio)
        flux_Jy = flux_scale ** 2.0

        sigma_ratio = np.nanstd(soln_ratio)
        # Standard error of the median
        error_ratio = 1.253 * sigma_ratio / np.sqrt(np.count_nonzero(~np.isnan(soln_ratio)))
        error_flux_Jy = 2. * flux_Jy * error_ratio

        logger.info('Measured flux density of %s: %.3f +/- %.3f Janskys'
                    % (targ, flux_Jy, error_flux_Jy))
        measured_flux[targ] = flux_Jy
        measured_flux_std[targ] = error_flux_Jy
    return measured_flux, measured_flux_std


# --------------------------------------------------------------------------------------------------
# --- Baseline ordering
# --------------------------------------------------------------------------------------------------

def get_ant_bls(pol_bls_ordering):
    """Pure antenna list given baseline list with polarisation information.

    E.g. go from
      [['ant0h','ant0h'], ['ant0v','ant0v'], ['ant0h','ant0v'],
       ['ant0v','ant0h'], ['ant1h','ant1h'], ...]
    to
      [['ant0,ant0'], ['ant1','ant1'], ...]
    """
    # get antenna only names from pol-included bls orderding
    ant_bls = np.array([[a1[:-1], a2[:-1]] for a1, a2 in pol_bls_ordering])
    ant_dtype = ant_bls[0, 0].dtype
    # get list without repeats (ie no repeats for pol)
    #     start with array of empty strings, shape (num_baselines x 2)
    bls_ordering = np.empty([len(ant_bls) / 4, 2], dtype=ant_dtype)

    # iterate through baseline list and only include non-repeats
    #   I know this is horribly non-pythonic. Fix later.
    bls_ordering[0] = ant_bls[0]
    bl = 0
    for c in ant_bls:
        if not np.all(bls_ordering[bl] == c):
            bl += 1
            bls_ordering[bl] = c

    return bls_ordering


def get_pol_bls(bls_ordering, pol):
    """Full baseline-pol ordering given baseline ordering and pol ordering.

    Parameters
    ----------
    bls_ordering
        list of correlation products without polarisation information, string shape(nbl,2)
    pol
        list of polarisation pairs, string shape (npol_pair, 2)

    Returns
    -------
    pol_bls_ordering : :class:`np.ndarray`
        correlation products without polarisation information, shape(nbl*4, 2)
    """
    pol_ant_dtype = np.array(bls_ordering[0][0] + 'h').dtype
    nbl = len(bls_ordering)
    pol_bls_ordering = np.empty([nbl * len(pol), 2], dtype=pol_ant_dtype)
    for i, p in enumerate(pol):
        for b, bls in enumerate(bls_ordering):
            pol_bls_ordering[nbl * i + b] = bls[0] + p[0], bls[1] + p[1]
    return pol_bls_ordering


def get_reordering(antlist, bls_ordering, output_order_bls=None, output_order_pol=None):
    """Reordering necessary to change given bls_ordering into desired ordering.

    Parameters
    ----------
    antlist : list of antennas, string shape(nants), or string of csv antenna names
    bls_ordering : list of correlation products, string shape(nbl,2)
    output_order_bls : desired baseline output order, string shape(nbl,2), optional
    output_order_pol : desired polarisation output order, string shape(npolcorr,2), optional

    Returns
    -------
    ordering
        ordering array necessary to change given bls_ordering into desired
        ordering, numpy array shape(nbl*4)
    bls_wanted
        ordering of baselines, without polarisation, list shape(nbl, 2)
    pol_order
        ordering of polarisations, list shape (4, 2)
    """
    # convert antlist to list, if it is a csv string
    if isinstance(antlist, str):
        antlist = antlist.split(',')
    # convert bls_ordering to a list, if it is not a list (e.g. np.ndarray)
    if not isinstance(bls_ordering, list):
        bls_ordering = bls_ordering.tolist()

    # get current antenna list without polarisation
    bls_ordering_nopol = [[b[0][0:4], b[1][0:4]] for b in bls_ordering]
    # find unique elements
    unique_bls = []
    for b in bls_ordering_nopol:
        if b not in unique_bls:
            unique_bls.append(b)

    # determine polarisation ordering
    if output_order_pol is not None:
        pol_order = output_order_pol
    else:
        # get polarisation products from current antenna list
        bls_ordering_polonly = [[b[0][4], b[1][4]] for b in bls_ordering]
        # find unique pol combinations
        unique_pol = []
        for pbl in bls_ordering_polonly:
            if pbl not in unique_pol:
                unique_pol.append(pbl)
        # put parallel polarisations first
        pol_order = []
        if len(bls_ordering_polonly) > 2:
            for pp in unique_pol:
                if pp[0] == pp[1]:
                    pol_order.insert(0, pp)
            pol_order.append([pol_order[0][0], pol_order[1][0]])
            pol_order.append([pol_order[1][0], pol_order[0][0]])
    npol = len(pol_order)

    if output_order_bls is None:
        # default output order is XC then AC
        bls_wanted = [b for b in unique_bls if b[0] != b[1]]

        # number of baselines including autocorrelations
        nants = len(antlist)
        nbl_ac = nants * (nants + 1) / 2

        # add AC to bls list, if necessary
        # assume that AC are either included or not (no partial states)
        if len(bls_ordering) == (nbl_ac * npol):
            # data include XC and AC
            bls_wanted.extend([b for b in unique_bls if b[0] == b[1]])

        bls_pol_wanted = get_pol_bls(bls_wanted, pol_order)
    else:
        bls_wanted = output_order_bls
        bls_pol_wanted = get_pol_bls(output_order_bls, pol_order)
    # note: bls_pol_wanted must be an np array for list equivalence below

    # find ordering necessary to change given bls_ordering into desired ordering
    # note: ordering must be a numpy array to be used for indexing later
    ordering_map = {tuple(bls): i for i, bls in enumerate(bls_ordering)}
    ordering = np.array([ordering_map[tuple(bls)] for bls in bls_pol_wanted])

    # how to use this:
    # print bls_ordering[ordering]
    # print bls_ordering[ordering].reshape([4,nbl,2])
    return ordering, bls_wanted, pol_order


def get_reordering_nopol(antlist, bls_ordering, output_order_bls=None):
    """Reordering necessary to change given bls_ordering into desired ordering.

    Parameters
    ----------
    antlist : list of antennas, string shape(nants), or string of csv antenna names
    bls_ordering : list of correlation products, string shape(nbl,2)
    output_order_bls : desired baseline output order, string shape(nbl,2), optional

    Returns
    -------
    ordering
        ordering array necessary to change given bls_ordering into desired
        ordering, numpy array, shape(nbl*4, 2)
    bls_wanted
        ordering of baselines, without polarisation, numpy array, shape(nbl, 2)
    """
    # convert antlist to list, if it is a csv string
    if isinstance(antlist, str):
        antlist = antlist.split(',')
    # convert bls_ordering to a list, if it is not a list (e.g. np.ndarray)
    if not isinstance(bls_ordering, list):
        bls_ordering = bls_ordering.tolist()

    # find unique elements
    unique_bls = []
    for b in bls_ordering:
        if b not in unique_bls:
            unique_bls.append(b)

    # defermine output ordering:
    if output_order_bls is None:
        # default output order is XC then AC
        bls_wanted = [b for b in unique_bls if b[0] != b[1]]
        # add AC to bls list
        bls_wanted.extend([b for b in unique_bls if b[0] == b[1]])
    else:
        # convert to list in case it is not a list
        bls_wanted = output_order_bls
    # note: bls_wanted must be an np array for list equivalence below
    bls_wanted = np.array(bls_wanted)

    # find ordering necessary to change given bls_ordering into desired ordering
    # note: ordering must be a numpy array to be used for indexing later
    ordering = np.array([np.all(bls_ordering == bls, axis=1).nonzero()[0][0]
                        for bls in bls_wanted])
    # how to use this:
    # print bls_ordering[ordering]
    # print bls_ordering[ordering].reshape([4,nbl,2])
    return ordering, bls_wanted


def get_bls_lookup(antlist, bls_ordering):
    """Correlation product antenna mapping.

    Parameters
    ----------
    antlist : list of antenna names, string shape (nant)
    bls_ordering : list of correlation products, string shape(nbl,2)

    Returns
    -------
    corrprod_lookup : lookup table of antenna indices for each baseline, shape(nbl,2)
    """
    # make polarisation and corr_prod lookup tables (assume this doesn't change
    # over the course of an observaton)
    antlist_index = dict([(antlist[i], i) for i in range(len(antlist))])
    return np.array([[antlist_index[a1[0:4]], antlist_index[a2[0:4]]] for a1, a2 in bls_ordering])


# --------------------------------------------------------------------------------------------------
# --- Simulation
# --------------------------------------------------------------------------------------------------

def fake_vis(shape=(7,), gains=None, noise=None, random_state=None):
    """Create fake point source visibilities, corrupted by given or random gains.

    The final dimension of `shape` corresponds to the number of antennas.
    """
    # create antenna lists
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    nants = shape[-1]

    antlist = list(range(nants))
    list1 = np.array([])
    for a, i in enumerate(range(nants - 1, 0, -1)):
        list1 = np.hstack([list1, np.ones(i) * a])
    list1 = np.hstack([list1, antlist])
    list1 = np.int_(list1)

    list2 = np.array([], dtype=np.int)
    mod_antlist = antlist[1:]
    for i in range(0, len(mod_antlist)):
        list2 = np.hstack([list2, mod_antlist[:]])
        mod_antlist.pop(0)
    list2 = np.hstack([list2, antlist])

    # create fake gains, if gains are not given as input
    if random_state is None:
        random_state = np.random
    if gains is None:
        gains = random_state.random_sample(shape) + 1j * random_state.random_sample(shape)
    else:
        assert shape == gains.shape

    # create fake corrupted visibilities
    nbl = nants * (nants + 1) // 2
    vis = np.ones(tuple(shape[:-1]) + (nbl,), gains.dtype)
    # corrupt vis with gains
    for i, (a, b) in enumerate(zip(list1, list2)):
        vis[..., i] *= gains[..., a] * gains[..., b].conj()
    # if requested, corrupt vis with noise
    if noise is not None:
        vis_noise = random_state.standard_normal(vis.shape) * noise
        vis = vis + vis_noise

    # return useful info
    bl_pair_list = np.column_stack([list1, list2])
    return vis, bl_pair_list, gains

# --------------------------------------------------------------------------------------------------
# --- Averaging
# --------------------------------------------------------------------------------------------------


def divide_weights(weighted_data, weights):
    """Divide weighted_data by weights, suppressing divide-by-zero errors."""
    # Suppress divide by zero warnings by replacing zeros with ones
    # all zero weight data is assumed to be set to zero in weighted_data
    weights_nozero = np.where(weights == 0, weights.dtype.type(1), weights)
    return weighted_data / weights_nozero


def wavg_full_f(data, flags, weights, chanav, threshold=0.8):
    """Perform weighted average over channels of data, flags and weights.

    This applies flags and averages over axis -3, for the specified number of
    channels.

    Parameters
    ----------
    data : :class:`np.ndarray`
        complex (ntimes, nchans, npols, nbls)
    flags : :class:`np.ndarray`
        int (ntimes, nchans, npols, bls)
    weights : :class:`np.ndarray`
        real (ntimes, nchans, npols, bls)
    chanav : int
        number of channels over which to average, integer
    threshold : float
        if fraction of flags in the input data
        exceeds threshold then set output flag to True, else False

    Returns
    -------
    av_data : :class:`np.ndarray`
        complex (..., av_chans, npols, nbls), weighted average of data
    av_flags : :class:`np.ndarray`
        bool (..., av_chans, npols, nbls), weighted average of flags
    av_weights : :class:`np.ndarray`
        real (..., av_chans, npols, nbls), weighted average of weights
    """
    # ensure chanav is an integer
    chanav = np.int(chanav)
    inc_array = list(range(0, data.shape[-3], chanav))

    # suppress any comparison-with-nan errors by replacing them with zeros
    flagged_weights = np.where(asbool(flags) | np.isnan(weights), weights.dtype.type(0), weights)
    weighted_data = data * flagged_weights

    # Clear all invalid elements, ie. nans, zeros and high weights
    invalid = np.isnan(weighted_data) | (weighted_data == 0) | (flagged_weights > HIGH_WEIGHT)
    weighted_data = np.where(invalid, weighted_data.dtype.type(0), weighted_data)
    flagged_weights = np.where(invalid, flagged_weights.dtype.type(0), flagged_weights)

    av_weights = np.add.reduceat(flagged_weights, inc_array, axis=-3)
    av_data = divide_weights(np.add.reduceat(weighted_data, inc_array, axis=-3), av_weights)

    # Update flags to include invalid elements
    n_flags = np.add.reduceat(flagged_weights == 0, inc_array, axis=-3)
    n_samples = np.add.reduceat(np.ones(flagged_weights.shape), inc_array, axis=-3)
    av_flags = n_flags >= n_samples * threshold

    return av_data, av_flags, av_weights


@numba.jit(nopython=True)
def _wavg_flags_f(flags, chanav, excise):
    """Implementation of wavg_flags_f.

    This numba-accelerated version handles only the case where there are
    4 dimensions and we're averaging along axis -3 (which will also be axis 1).
    """
    n_time, n_channels, n_pol, n_baselines = flags.shape
    good = np.empty((n_pol, n_baselines), np.bool_)
    merge_good = np.empty((n_pol, n_baselines), flags.dtype)
    merge_all = np.empty((n_pol, n_baselines), flags.dtype)
    out_channels = n_channels // chanav
    out = np.empty((n_time, out_channels, n_pol, n_baselines), flags.dtype)
    for t in range(n_time):
        for oc in range(out_channels):
            good.fill(False)
            merge_good.fill(0)
            merge_all.fill(0)
            start_channel = oc * chanav
            end_channel = start_channel + chanav
            for ic in range(start_channel, end_channel):
                for p in range(n_pol):
                    for b in range(n_baselines):
                        flag = flags[t, ic, p, b]
                        is_good = not excise[t, ic, p, b]
                        merge_all[p, b] |= flag
                        merge_good[p, b] |= is_good * flag
                        good[p, b] |= is_good
            for p in range(n_pol):
                for b in range(n_baselines):
                    out[t, oc, p, b] = merge_good[p, b] if good[p, b] else merge_all[p, b]
    return out


def wavg_flags_f(flags, chanav, excise, axis):
    """Combine flags across frequencies.

    This function is designed to mirror flag merging done by ingest. For each
    visibility, there is a flag bitmask in `flags` and a corresponding boolean
    in `excise`. The combined flags for a set of visibilities is
    - The bitwise OR of the flags for which the excise bit is clear, if there
      is at least one such; otherwise
    - The bitwise OR of all the flags.

    The implementation currently supports at most one axis before `axis` and at
    most two after it (which caters for the standard time, frequency, pol,
    baseline axis ordering).

    Parameters
    ----------
    flags : :class:`np.ndarray`
        uint8 flag masks
    chanav : int
        number of channels over which to average, integer. It must exactly
        divide into the number of channels.
    excise : class:`np.ndarray`
        boolean corresponding to each element of `flags`
    axis : int
        Axis along which to apply

    Returns
    -------
    av_flags : :class:`np.ndarray`
        uint8 combined flags

    Raises
    ------
    ValueError
        if `flags` and `excise` have different shapes
    ValueError
        if chanav doesn't divide into the length of the chosen axis
    np.AxisError
        if `axis` is invalid
    NotImplementedError
        if there are more than two axes after `axis` or more than one axis
        before it
    """
    flags = np.asarray(flags)
    excise = np.asarray(excise)
    if excise.shape != flags.shape:
        raise ValueError('excise must have the same shape as flags')
    if axis < 0:
        axis += flags.ndim
    if axis < 0 or axis >= flags.ndim:
        raise np.AxisError('axis {} is out of bounds for array of dimension {}'
                           .format(axis, flags.ndim))
    if axis > 1 or flags.ndim - axis > 3:
        raise NotImplementedError('axis {} is not currently supported'.format(axis))
    if flags.shape[axis] % chanav != 0:
        raise ValueError('chanav {} does not divide length {}'
                         .format(chanav, flags.shape[axis]))

    pre = 1 - axis
    post = axis + 3 - flags.ndim
    for i in range(pre):
        flags = flags[np.newaxis]
        excise = excise[np.newaxis]
    for i in range(post):
        # Insert the new axes immediately after channels, rather than at the
        # end, because we want the inner loops to run over longer axes.
        flags = flags[:, :, np.newaxis]
        excise = excise[:, :, np.newaxis]
    out_flags = _wavg_flags_f(flags, chanav, excise)
    # Now reverse the axis removals
    for i in range(post):
        out_flags = out_flags[:, :, 0]
    for i in range(pre):
        out_flags = out_flags[0]
    return out_flags


# --------------------------------------------------------------------------------------------------
# --- General helper functions
# --------------------------------------------------------------------------------------------------

def arcsec_to_rad(angle):
    """Convert angle in arcseconds to angle in radians."""
    return np.deg2rad(angle / 60. / 60.)


# --------------------------------------------------------------------------------------------------
# --- SNR calculations
# --------------------------------------------------------------------------------------------------

def calc_rms(x, weights, axis):
    """Calculate weighted root-mean-square (RMS) over given axis.

    Parameters
    ----------
    x : :class:`np.ndarray`,
        real
    weights : :class:`np.ndarray`,
        real
    axis : int or tuple of int
        axis or axes to sum weighted RMS over

    Returns
    -------
    rms : :class:`np.ndarray`
    """
    # ensure nans have weight zero
    weights_zero = np.where(~np.isfinite(x), 0, weights)

    w_square = x**2 * weights_zero
    sum_w_square = np.nansum(w_square, axis)
    sum_weights = np.sum(weights_zero, axis)
    with np.errstate(divide='ignore', invalid='ignore'):
        rms = np.sqrt(sum_w_square / sum_weights)
    return rms


def poor_antenna_flags(data, weights, bls_lookup, threshold):
    """Flags indicating antennas with noisy phases on a majority of baselines.

    Create an array of flags which flag antennas when >80% of their baselines
    have a phase RMS noise greater than the given threshold (in radians).

    Parameters
    ----------
    data : :class:`np.ndarray`
        complex (ntimes, nchans, npols, nbls)
    weights : :class:`np.ndarray`
        real (ntimes, nchans, npols, nbls)
    bls_lookup : :class:`np.ndarray`
        int (nbls, 2) , antenna pairs in each baseline
    threshold : float
        rms threshold in radians which results in a flag

    Returns
    -------
    ant_flags : :class:`np.ndarray`
        bool (ntimes, nchans, npols, nbls), True if flagged, else False
    """
    ntimes, nchans, npols, nbls = data.shape
    nants = ants_from_bllist(bls_lookup)
    angle = np.float32(np.angle(data))

    ant_flags = np.zeros((ntimes, npols, nbls), dtype=bool)
    for a in range(nants):
        # change sign of phase if antenna is not first in baseline pair
        a_bls = (bls_lookup[:, 0] == a) ^ (bls_lookup[:, 1] == a)
        sign_flip = np.where(bls_lookup[a_bls][:, 0] == a, 1, -1)
        ant_angle = angle[..., a_bls] * sign_flip
        # calculate rms phase of all baselines to antenna a
        rms = calc_rms(ant_angle, weights[..., a_bls], axis=1)

        # flag if the rms of a large number of baselines to the antenna
        # are above threshold, only consider non-NaN baselines
        rms_valid = np.where(~np.isfinite(rms), 0, rms)
        n_bad = np.sum(rms_valid > threshold, axis=-1, keepdims=True)
        n_valid_bls = np.sum(np.isfinite(rms), axis=-1, keepdims=True)
        flag = divide_weights(n_bad, n_valid_bls) > 0.8
        ant_flags[..., a_bls] += flag

    # broadcast flags to shape of data
    ant_flags = ant_flags[:, np.newaxis]
    ant_flags = np.broadcast_to(ant_flags, (ntimes, nchans, npols, nbls))
    return ant_flags


def snr_antenna(data, weights, bls_lookup, flag_ants=None):
    """Signal-to-noise ratio (SNR) estimate per antenna.

    SNR is estimated as the inverse of phase RMS variance per antenna.
    The variance per antenna is calculated by summing the variance over all
    channels and all baselines to the antenna, assuming a mean phase of zero.
    Optionally supply a baseline mask of bad antennas. These bad antennas are
    excluded from the SNR estimate of all other antennas, but included in the
    calculation of their own SNR.

    Parameters
    ----------
    data : :class:`np.ndarray`
        complex (ntimes, nchans, npols, nbls)
    weights : :class:`np.ndarray`
        real (ntimes, nchans, npols, nbls)
    bls_lookup : :class:`np.ndarray`
        int (nbls, 2) , antenna pairs in each baseline
    flag_ants : :class:`np.ndarray`
        bool (ntimes, nchans, npols, nbls), True if flagged, else False

    Returns
    -------
    snr : :class:`np.ndarray`
        real (ntimes, npol, nants)
    """
    nants = ants_from_bllist(bls_lookup)
    ntimes, nchans, npols, nbls = data.shape
    angle = np.float32(np.angle(data))

    if flag_ants is None:
        flag_ants = np.zeros((ntimes, nchans, npols, nbls), dtype=bool)

    snr = np.empty((ntimes, npols, nants), np.float32)
    for a in range(nants):
        # change sign of phase if antenna is not first in baseline pair
        a_bls = (bls_lookup[:, 0] == a) ^ (bls_lookup[:, 1] == a)
        sign_flip = np.where(bls_lookup[a_bls][:, 0] == a, 1, -1)
        angle_ant = angle[..., a_bls] * sign_flip
        weights_ant = weights[..., a_bls]

        # if antenna is flagged use all baselines to it in calculation of its own snr
        # but exclude baselines to it when calculating snr of other antennas.
        # This is necessary because the final SNR sums over all baselines to a given antenna.
        # One very poor baseline can therefore dramatically reduce the overall SNR measurement.
        # This strategy tries to protect against this while preserving a measurement of SNR
        # on the poor antenna.
        bad_mask = np.all(flag_ants[..., a_bls], axis=-1, keepdims=True) ^ flag_ants[..., a_bls]
        weights_ant[bad_mask] = 0

        # calc snr across chans and antenna axis
        rms = calc_rms(angle_ant, weights_ant, axis=(1, -1))
        with np.errstate(divide='ignore', invalid='ignore'):
            snr[..., a] = 1. / rms

    return snr


class FluxDensityModel:
    """A flux density model.

    A flux density model valid over a given frequency interval, specified as parameters of
    a variety of defined flux density polynomials.

    Models can be instantiated by supplying parameters directly or indirectly using a space
    separated description string. Defined polynomials are a Baars polynomial and the log and
    ordinary polynomials defined in the wsclean package.

    For the Baars polynomial the description string contains the minimum frequency and maximum
    frequency and the polynomial coefficients as space separated values, optionally the third
    argument in the string can specify the model type. If this is omitted the default is to assume
    a Baars polynomial. Some examples
    are:
        '(1000 2000 baars_mhz 0.34 0.85)'
        '(1000 2000 0.34 0.85 1)'

    For the wsclean polynomials the description string supplies the minimum frequency, the maximum
    frequency, the model_type as 'wsclean', the stokes I parameter, the coefficients of the
    polynomial (enclosed in square brackets), a 'true' or 'false' string indicating whether a log or
    ordinary polynomial fit is supplied, and a reference frequency in Hz. An example is:
        '(1000 2000 wsclean 30 [0.34 0.85] false 150000000.00)'

    Parameters
    ----------
    min_freq_MHz : float or str
        minimum frequency that flux density model is valid for or
        a description string
    max_freq_MHz : float
        maximum frequency that flux density model is valid for
    coefs : tuple
       coeficients of the flux density polynomial
    model_type : str, optional
       type of polynomial used to calculate flux density, default is a Baars polynomial with
       frequency in MHz
    log : bool
       log or ordinary polynomial if model_type is wsclean, required if model_type is wsclean
    stokes_I : float
       stokes_I parameter in wsclean polynomial, required if model_type is wsclean
    ref_freq : float
       ref_freq in wsclean polynomial, specified in Hz, required if model_type is wsclean
    """
    def __init__(self, min_freq_MHz, max_freq_MHz=None, coefs=None, model_type=None, log=False,
                 stokes_I=None, ref_freq=None):
        # If the first parameter is a description string, extract the relevant flux parameters
        if isinstance(min_freq_MHz, str):
            # Cannot have other parameters if description string is given - this is a safety check
            params = [max_freq_MHz, coefs, model_type, log, stokes_I, ref_freq]
            if any(params):
                raise ValueError("First parameter {} is description string"
                                 "- cannot have other parameters".format(min_freq_MHz))

            # Split description string on spaces
            flux_info = [num for num in min_freq_MHz.strip(' ()').split()]
            if len(flux_info) < 3:
                raise ValueError('Flux density description string {}'
                                 'is invalid'.format(min_freq_MHz))
            # first coefficient can be a string indicating type of flux density polynomial
            # if no string is supplied assume baars_mhz polynomial
            try:
                float(flux_info[2])
            except ValueError:
                self.model_type = flux_info.pop(2)
            else:
                self.model_type = 'baars_mhz'

            try:
                self.min_freq_MHz = float(flux_info[0])
                self.max_freq_MHz = float(flux_info[1])
            except ValueError:
                raise ValueError('Min/max_freq_MHz contains invalid floating point '
                                 'numbers')

            if self.model_type == 'baars_mhz':
                self._set_baars_params(flux_info[2:])

            elif self.model_type == 'wsclean':
                self._set_wsclean_params(flux_info[2:])

        else:
            self.min_freq_MHz = min_freq_MHz
            self.max_freq_MHz = max_freq_MHz
            self.coefs = coefs
            # if no model type supplied assume baars_mhz polynomial
            self.model_type = model_type if model_type else 'baars_mhz'
            self.log = log
            self.stokes_I = stokes_I
            self.ref_freq = ref_freq
            # require some arguments for wsclean model
            if self.model_type == 'wsclean':
                wsclean_params = [log, stokes_I, ref_freq]
                if any(p is None for p in wsclean_params):
                    raise ValueError('A required argument for model_type "{}" '
                                     'hasn\'t been supplied'.format(self.model_type))

        # get flux density function and no of coefs for given model_type
        self._flux_density, max_coefs = self._model_func()
        self.coefs = self._extract_coefs(max_coefs)

    def _set_baars_params(self, coefs):
        try:
            coefs = [float(c) for c in coefs]
        except ValueError:
            raise ValueError('Description string coefficients {} contain invalid floating point '
                             'numbers'.format(' '.join(coefs)))

        self.coefs = tuple(coefs)

    def _set_wsclean_params(self, wsclean_info):
        csv = ','.join(s.capitalize() for s in wsclean_info)
        try:
            stokes_I, coefs, log, ref_freq = ast.literal_eval(csv)
            self.stokes_I = float(stokes_I)
            self.coefs = [float(c) for c in coefs]
            self.log = bool(log)
            self.ref_freq = float(ref_freq)
        except ValueError as err:
            raise ValueError("Invalid wsclean component description"
                             "string '{}'".format(' '.join(wsclean_info))) from err

    def _model_func(self):
        if self.model_type == 'baars_mhz':
            flux_func = self._baars_mhz
            max_coefs = 6
        elif (self.model_type == 'wsclean' and self.log is False):
            flux_func = self._wsc_ord_mhz
            max_coefs = 5
        elif (self.model_type == 'wsclean' and self.log is True):
            flux_func = self._wsc_log_mhz
            max_coefs = 5
        else:
            raise ValueError("Unknown flux density polynomial '%s'" % (self.model_type))
        return flux_func, max_coefs

    def _extract_coefs(self, max_coefs):
        coefs = np.zeros(max_coefs)
        coefs[:min(len(self.coefs), max_coefs)] = self.coefs[:min(len(self.coefs), max_coefs)]
        if len(self.coefs) > max_coefs:
            logger.warning('Received %d coefficients but only expected %d - ignoring the rest',
                           len(self.coefs), max_coefs)
        return coefs

    def _baars_mhz(self, freq_MHz):
        a, b, c, d, e, f = self.coefs
        log10_v = np.log10(freq_MHz)
        log10_S = a + b * log10_v + c * log10_v ** 2 + d * log10_v ** 3 + e * np.exp(f * log10_v)
        return 10 ** log10_S

    def _wsc_ord_mhz(self, freq_MHz):
        a, b, c, d, e = self.coefs
        ref_freq = self.ref_freq / 1e6
        sI = self.stokes_I
        x = freq_MHz / ref_freq - 1
        flux = sI + a * x + b * x ** 2 + c * x ** 3 + d * x ** 4 + e * x ** 5
        return flux

    def _wsc_log_mhz(self, freq_MHz):
        a, b, c, d, e = self.coefs
        ref_freq = self.ref_freq / 1e6
        sI = self.stokes_I
        x = np.log(freq_MHz / ref_freq)
        ln_S = np.log(sI) + a * x + b * x ** 2 + c * x ** 3 + d * x ** 4 + e * x ** 5
        return np.exp(ln_S)

    def flux_density(self, freq_MHz):
        """Calculate Stokes I flux density for given observation frequency.

        Returns nan for frequencies outside the valid range of the model.

        Parameters
        ----------
        freq_MHz : float or sequence of floats
            frequency at which to calculate flux density in MHz
        """

        freq_MHz = np.asarray(freq_MHz)
        flux = self._flux_density(freq_MHz)
        flux = np.asarray(flux)
        flux[freq_MHz < self.min_freq_MHz] = np.nan
        flux[freq_MHz > self.max_freq_MHz] = np.nan
        # use flux.item() to convert 0-d array back to scalar if necessary
        return flux if flux.ndim else flux.item()
