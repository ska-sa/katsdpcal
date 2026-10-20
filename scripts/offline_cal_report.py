#!/usr/bin/env python3
import argparse
import logging
import os
import re
from collections import defaultdict
import numpy as np
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import katdal
import katpoint
from katdal.applycal import complex_interp
from katsdpcal import report, param_dir
from katsdpcal.reduction import check_noise_diode
from katsdpcal.pipelineprocs import parameters_from_file, finalise_parameters
from katsdpcalproc import calprocs, calprocs_dask

# Silencing dask's "large chunk" PerformanceWarning
dask.config.set(**{'array.slicing.split_large_chunks': False})


logger = logging.getLogger(__name__)


def get_telstate_product(telstate, product_key, t1):
    """Fetch the most recent value of a telstate solution product.

    Returns the value of the most recent solution stored of 'product_key' at or
    before 't1'

    Parameters
    ----------
    telstate : :class:'katsdptelstate.TelescopeState'
        telescope state, 'CBID_cal_' namespace
    product_key : str
        telstate key the solution is stored under (e.g.
        merged_params['product_names']['K'])
    t1 : float
        time cutoff to query up to

    Returns
    -------
    :class:'np.ndarray' or None
        the solution value, or None if no solution exists at or before the cutoff
    """
    vals = telstate.get_range(product_key, st=0, et=t1)
    if not vals:
        return None
    return vals[-1][0]


def interp_b_nans(b_gain, channel_freqs):
    """Linearly interpolate over flagged (NaN) channels in a B solution.

    Uses katdal's own 'complex_interp' (applycal.py) to interpolate magnitude
    and phase separately.

    Parameters
    ----------
    b_gain : :class:'np.ndarray'
        complex, shape (nchan, npol, nant), B solution to interpolate
    channel_freqs : :class:'np.ndarray'
        real, shape (nchan,), channel frequencies corresponding to 'b_gain''s
        first axis

    Returns
    -------
    :class:'np.ndarray'
        complex, same shape and dtype as 'b_gain', with each (pol, ant)
        column's NaN channels linearly interpolated wherever that column has
        some, but not all, valid data
    """
    out = b_gain.copy()
    for pol in range(b_gain.shape[1]):
        for ant in range(b_gain.shape[2]):
            col = b_gain[:, pol, ant]
            valid = np.isfinite(col)
            # added 'not valid.all()' to prevent wasted computation,
            # an all valid column has nothing to interpolate.
            if valid.any() and not valid.all():
                out[:, pol, ant] = complex_interp(channel_freqs, channel_freqs[valid], col[valid])
    return out

def apply_parallel_pol(solval, vis, index0=None, index1=None):
    """Apply calibration solutions to parallel-pol (HH/VV) visibilities
       auto- or cross-correlation.

    Mirrors scan.py's Scan._apply(). Pass 'index0'/'index1' for cross-
    correlation (baseline) data, omit them for auto-correlation data.

    Parameters
    ----------
    solval : :class:'np.ndarray'
        complex, (..., nant), multiplicative solution values to divide out
    vis : :class:'np.ndarray'
        complex, (..., nbl) input visibilities for cross-correlation data
        , or (..., nant) for auto-correlation data
    index0, index1 : :class:'np.ndarray', optional
        int, (nbl,), antenna index of each baseline's first/second antenna
        Omit for auto-correlation data, where vis's antenna axis already
        matches that of solval.

    Returns
    -------
    :class:'np.ndarray'
        complex, same shape as 'vis', corrected visibilities
    """
    inv_solval = 1.0 / solval
    if index0 is None and index1 is None:
        correction = inv_solval * inv_solval.conj()
    else:
        correction = inv_solval[..., index0] * inv_solval[..., index1].conj()
    return vis * correction


def apply_cross_pol(solval, vis, index0=None, index1=None):
    """Apply calibration solutions to cross-hand (HV/VH) visibilities
       auto- or cross-correlation.

    Mirrors scan.py's Scan._apply(). Pass 'index0'/'index1' for cross-
    correlation (baseline) data, omit them for auto-correlation data.

    Parameters
    ----------
    solval : :class:'np.ndarray'
        complex, (..., npol, nant), multiplicative solution to divide out
    vis : :class:'np.ndarray'
        complex, (..., npol, nbl) input visibilities for cross-correlation data,
        or (..., npol, nant) for auto-correlation data
    index0, index1 : :class:'np.ndarray', optional
        int, (nbl,), antenna index of each baseline's first/second antenna
        Omit for auto-correlation data, where vis's antenna axis already
        matches that of solval.

    Returns
    -------
    :class:'np.ndarray'
        complex, same shape as 'vis', corrected visibilities
    """
    inv_solval = 1.0 / solval
    flipped = np.flip(inv_solval, axis=-2).conj()
    if index0 is None and index1 is None:
        correction = inv_solval * flipped
    else:
        correction = inv_solval[..., index0] * flipped[..., index1]
    return vis * correction


def valid_allfreq(model_cat, freqs):
    """Check all models in model_cat are valid for the given freqs (in Hz).

    Parameters
    ----------
    model_cat : list of :class:'katsdpcalproc.calprocs.FluxDensityModel'
        flux density models to check
    freqs : :class:'np.ndarray'
        real, frequencies to check coverage for, in Hz

    Returns
    -------
    bool
        True if every model's valid range covers the full span of 'freqs'
    """
    min_valid = [m.min_freq_MHz for m in model_cat]
    max_valid = [m.max_freq_MHz for m in model_cat]
    freqs_MHz = freqs / 1e6
    return max(min_valid) <= min(freqs_MHz) and min(max_valid) >= max(freqs_MHz)



def build_model(target, target_name, timestamps, telstate, antennas, array_position,
                channel_freqs, bls_lookup, max_offset=8.0):
    """Reconstruct a target's own flux model. Mirrors scan.py _create_model.

    Case A: a single source at the phase center -- model is a scalar
    Case B: Complex model requiring calculation via uvw coordinates -- model
            is an array of the same shape as self.vis

    Parameters
    ----------
    target : :class:'katpoint.Target'
        target to build a model for
    target_name : str
        Name of target observed in the scan
    timestamps : :class:'np.ndarray'
        scan timestamps to build the model at (Case B)
    telstate : :class:'katsdptelstate.TelescopeState'
        telescope state, used to fetch 'model_<target_name>'
    antennas : list of :class:'katpoint.Antenna'
        list of antennas
    array_position : :class:'katpoint.Antenna'
        array center position
    channel_freqs : :class:'np.ndarray'
        Array of channel frequencies -- this server's slice
    bls_lookup : :class:'np.ndarray'
        int, (nbls, 2), List of antenna pairs for each baseline (xc_lookup)
    max_offset : float
        The difference in position away from the phase center for a point to
        be considered at the phase center [arcseconds]
        Default: 8 (= meerkat beam)
    Returns
    -------
    :class:'np.ndarray' or None
        complex, shape (ntimes, nchans, 1, nbls), or None if the target has no
        stored model
    """
    try:
        model_params = telstate['model_' + target_name]
    except KeyError:
        return None
    if len(timestamps) == 0:
        return None

    model_targets = [katpoint.Target(','.join(mc.split(',')[:-1])) for mc in model_params]
    model_fluxes  = [calprocs.FluxDensityModel(''.join(mc.split(',')[-1])) for mc in model_params]

    if not valid_allfreq(model_fluxes, channel_freqs) and len(model_params) > 1:
        # If not every component covers the full band -- fall back to just the first source
        model_params = [model_params[0]]
        model_targets = [model_targets[0]]
        model_fluxes = [model_fluxes[0]]

    position_offset = target.separation(model_targets[0], antenna=array_position)

    if len(model_params) == 1 and position_offset < calprocs.arcsec_to_rad(max_offset):
        # Case A:
        if not valid_allfreq(model_fluxes, channel_freqs):
            model = np.ones((1, len(channel_freqs), 1, 1), np.complex64)
        else:
            # reshape to (1, nchan, 1, 1) 
            model = model_fluxes[0].flux_density(channel_freqs / 1.0e6)[np.newaxis, :, np.newaxis, np.newaxis]
            model = np.require(model, dtype=np.float32)
    else:
        # Case B:
        uvw = target.uvw(antennas, timestamps, array_position)
        uvw = np.array(uvw, np.float64)

        ntimes, nchans, nants, nbls = len(timestamps), len(channel_freqs), len(antennas), len(bls_lookup)
        k_ant = np.zeros((ntimes, nchans, nants), np.complex128)
        complexmodel = np.zeros((ntimes, nchans, nbls), np.complex64)
        wl = katpoint.lightspeed / channel_freqs

        for source, flux_model in zip(model_targets, model_fluxes):
            S = flux_model.flux_density(channel_freqs / 1.0e6)
            l, m = target.sphere_to_plane(*source.radec(), projection_type='SIN', coord_system='radec')
            k_ant = calprocs.K_ant(uvw, l, m, wl, k_ant)
            complexmodel = calprocs.add_model_vis(k_ant, bls_lookup[:, 0], bls_lookup[:, 1], S, complexmodel)

        # add a pol axis so it matches Case A's (ntimes, nchans, 1, nbls) shape
        model = complexmodel[:, :, np.newaxis, :]

    return model


def solve_g_flux(vis_autopol, flags_autopol, weights_autopol, model, xc_lookup, refant,
                 bchan, echan):
    """Reconstruct G_FLUX the way the real pipeline does.

    Mirrors reduction.py's BANDPASS section + scan.py's g_sol: divide K,B-corrected
    vis by the target's own flux model, average over the fit channel range and the
    whole scan as one solution interval, then run the same gain fit
    (calprocs.g_fit) the pipeline uses. Only valid for bpcal targets, since
    only they get a G_FLUX saved to solution store online.

    Parameters
    ----------
    vis_autopol : :class:'np.ndarray' or :class:'da.Array'
        complex, (ntime, nchan, npol, nbl), K,B-corrected parallel-hand visibilities
    flags_autopol : :class:'np.ndarray' or :class:'da.Array'
        bool, same shape as 'vis_autopol'
    weights_autopol : :class:'np.ndarray' or :class:'da.Array'
        real, same shape as 'vis_autopol'
    model : :class:'np.ndarray'
        complex, (ntime, nchan, 1, nbl), the target's own flux model
    xc_lookup : :class:'np.ndarray'
        int, (nbl, 2), antenna index pairs for each baseline
    refant : int
        reference antenna index
    bchan, echan : int
        start/end channel of the fit range (merged_params['g_bchan']/['g_echan'])

    Returns
    -------
    :class:'np.ndarray'
        complex, (npol, nant), the reconstructed G_FLUX solution for this scan
    """
    echan = echan if echan else None
    chan_slice = slice(bchan, echan)

    fitvis = vis_autopol[:, chan_slice] / model[:, chan_slice]
    fit_flags = flags_autopol[:, chan_slice]
    fit_weights = weights_autopol[:, chan_slice]

    # average over time (one solution interval spanning the whole scan),
    # then over the channel range -- wavg_full_t (time) then wavg_full(axis=1) (channel)
    solint = fitvis.shape[0]   # take the whole scan as one solution interval
    av_vis, av_flags, av_weights = calprocs_dask.wavg_full_t(fitvis, fit_flags, fit_weights, solint)
    av_vis, av_flags, av_weights = calprocs_dask.wavg_full(av_vis, av_flags, av_weights, axis=1)
    logger.info("Computing G_FLUX gain fit inputs")
    with ProgressBar():
        av_vis, av_weights = da.compute(av_vis, av_weights)

    #Solve for gain
    g_soln = calprocs.g_fit(av_vis, av_weights, xc_lookup, None, refant)
    return g_soln[0]   # shape (npol, nant)


def num_servers(telstate, CBID, product_prefix='product_B'):
    """Determine how many servers were used to processed this observation.

    Solutions such as B, BCROSS_DIODE and SNR_B are written per-server --
    product_B0, product_B1, ... product_B{n-1} for n servers. A 1K/4K
    observation is typically run on a single server (so
    only product_B0 exists); a 32K observation is typically split across 4
    (product_B0..product_B3). Count the number of B solutions in telstate
    to determine how many servers were used.

    Parameters
    ----------
    telstate : :class:'katsdptelstate.TelescopeState'
        telescope state, 'CBID_cal_' namespace
    CBID : str
        capture block ID
    product_prefix : str, optional
        product name prefix to search for (default 'product_B' as it
        is always present if any server ran)

    Returns
    -------
    int
        number of servers found (at least 1)

    Raises
    ------
    ValueError
        if no matching keys are found
    """
    # Look for keys matching the pattern of product B solution name
    pattern = re.compile(re.escape(CBID + '_cal_' + product_prefix) + r'(\d+)$')
    server_ids = []
    for key in telstate.keys():
        m = pattern.search(key)
        # Add server ID to list if found
        if m:
            server_ids.append(int(m.group(1)))
    if not server_ids:
        raise ValueError(f"No '{product_prefix}<N>' keys found in telstate -- "
                         f"can't determine how many servers processed this observation")
    return len(server_ids)


def build_server_merged_params(base_params, telstate, servers, server_id):
    """Build one server's local merged_params, from a fresh copy of base_params.

    'finalise_parameters()' Converts certain parameters (eg. g_bchan,
    g_echan, k_bchan, k_echan, rfi_freq_chunks, epoint_freq_chunks) from 'global'
    (all channels) values to 'local' (server dependant channel range) values.

    Parameters
    ----------
    base_params : dict
        global parameters (for full channel range)
    telstate : :class:'katsdptelstate.TelescopeState'
        telescope state, 'CBID_cal_' namespace
    servers : int
        total number of servers the band is being split across
    server_id : int
        this server's index

    Returns
    -------
    merged_params : dict
        finalised, server-local parameters for this server_id
    xc_lookup : :class:'np.ndarray'
        int, (nbl, 2), antenna index pairs for each cross-correlation baseline
    refant : int
        reference antenna index
    """
    merged_params = dict(base_params)   # fresh copy -- see warning above
    finalise_parameters(merged_params, telstate, servers, server_id)
    if merged_params['refant_index'] is None:
        merged_params['refant_index'] = telstate['cal_antlist'].index(telstate['refant'])

    # strip out auto-correlations just cross-correlation baseline pairs
    xc_mask = np.array([a != b for a, b in merged_params['bls_lookup']])
    xc_lookup = merged_params['bls_lookup'][xc_mask]
    refant = merged_params['refant_index']
    return merged_params, xc_lookup, refant


def gather_model_av_corr(d, telstate, merged_params, xc_lookup, cbf_targets):
    """Build every calibrator's/target's own flux model for this server's band.

    Builds av_corr's '<target>_model' entries for this server's local
    channel_freqs slice, using build_model. A target visited more than once across the
    observation gets its visits averaged into one entry.

    Parameters
    ----------
    d : :class:'katdal.DataSet'
        dataset selected down to relevant server's channel slice
    telstate : :class:'katsdptelstate.TelescopeState'
        telescope state, 'CBID_cal_' namespace
    merged_params : dict
        this server's finalised, local parameters
    xc_lookup : :class:'np.ndarray'
        int, (nbl, 2), antenna index pairs for each cross-correlation baseline
    cbf_targets : list of :class:'katpoint.Target'
        every target in the observation

    Returns
    -------
    dict
        {'<target_name>_model': [model_average_array]} for every target with a
        usable model, for this server's local channel_freqs
    """
    antennas = merged_params['antennas']
    array_position = merged_params['array_position']
    channel_freqs = merged_params['channel_freqs']
    n_chans = len(channel_freqs)   # single server channel count

    model_entries = {}
    # Iterate over scans
    for compscan_index, label, target in d.compscans():
        d.select(scans="track", compscans=compscan_index, corrprods='cross')
        target_name = target.name

        if not any(k in target.tags for k in ('bfcal', 'delaycal', 'polcal', 'bpcal', 'gaincal')):
            continue

        timestamps = d.timestamps[1:].copy()
        model = build_model(target, target_name, timestamps, telstate, antennas,
                            array_position, channel_freqs, xc_lookup)
        if model is None:
            continue

        # Average down to 1024 channels
        chan_sample = n_chans // 1024
        model_average = np.average(model, axis=(0, -2, -1))[::chan_sample]
        model_entries.setdefault(target_name + '_model', []).append(model_average)
    # Average models for targets visited more than once
    return {name: [np.mean(vals, axis=0)] for name, vals in model_entries.items()}


def gather_flag_av_corr(d, cal_tags):
    """Build av_corr's 't_flags'/'bl_flags' entries for this server's channel slice.

    t_flags is a running *count* of flagged samples per freq, pol and
    baseline, summed across the whole observation; bl_flags is the
    fraction flagged per freq and pol, averaged over baseline per scan.

    Parameters
    ----------
    d : :class:'katdal.DataSet'
        dataset, already selected down to relevant server's channel slice
    cal_tags : list of str
        target tags to include (gaincal, target, bfcal, bpcal, delaycal)

    Returns
    -------
    dict
        {'t_flags': [...], 'bl_flags': [...]}
    """
    t_sum_graphs, bl_sum_graphs, meta = [], [], []
    d.select(scans="track")
    for compscan_index, label, target in d.compscans():
        if not any(k in target.tags for k in cal_tags):
            continue
        d.select(scans="track", compscans=compscan_index, corrprods='cross')
        if len(d.timestamps) <= 1:
            continue

        ant_names = [a.name for a in d.ants]
        reorder_idx, bls_ordering, bls_pol_ordering = calprocs.get_reordering(ant_names, d.corr_products)
        flags = d.raw_flags.dataset[1:]
        flags_reordered = flags[:, :, reorder_idx]

        # reshape to separate pol: (t, f, npol, nbl)
        npol = len(set(map(tuple, bls_pol_ordering)))  # either 4 or 2
        nbl = flags_reordered.shape[2] // npol
        flags_4d = flags_reordered.reshape(flags.shape[0], flags.shape[1], npol, nbl)
        flags_autopol = flags_4d[:, :, :2, :]  # HH and VV

        nchans = 1024
        t_sum_flags = da.sum(calprocs.asbool(flags_autopol), axis=0, dtype=np.float32)
        orig_chans = flags_autopol.shape[1]
        chanav = max(1, orig_chans // nchans)
        t_sum_flags = calprocs_dask.av_blocks(t_sum_flags, chanav)

        timestamps = d.timestamps[1:]
        n_times = np.float32(len(timestamps))
        bl_sum_flags = da.mean(t_sum_flags, axis=-1) / n_times

        t_sum_graphs.append(t_sum_flags)
        bl_sum_graphs.append(bl_sum_flags)
        meta.append(np.mean(timestamps))

    t_flags, bl_flags = [], []
    logger.info("Computing t_flags/bl_flags for %d scans", len(t_sum_graphs))
    with ProgressBar():
        t_sum_results, bl_sum_results = da.compute(t_sum_graphs, bl_sum_graphs)
    for t, b, ts in zip(t_sum_results, bl_sum_results, meta):
        t_flags.append(t)
        bl_flags.append((b, ts))
    return {'t_flags': [np.sum(t_flags, axis=0)], 'bl_flags': bl_flags}


def summarise_av_corr(d, telstate, merged_params, xc_lookup, refant, cbf_targets, cal_tags):
    """Build av_corr's '_g_spec'/'_g_bls'/'_g_phase'/'_nog_spec' entries.

    Calibration is applied manually: K, B (and G, except for bpcal/delaycal-
    only scans) using apply_parallel_pol function.

    Note: each solution is reshaped/broadcast to vis's (time, freq, pol, ant)
    axes.

    Parameters
    ----------
    d : :class:'katdal.DataSet'
        dataset, already selected down to relevant server's channel slice
    telstate : :class:'katsdptelstate.TelescopeState'
        telescope state, 'CBID_cal_' namespace
    merged_params : dict
        this server's finalised, local parameters
    xc_lookup : :class:'np.ndarray'
        int, (nbl, 2), antenna index pairs for each cross-correlation baseline
    refant : int
        reference antenna index
    cbf_targets : list of :class:'katpoint.Target'
        every target in the observation's catalogue
    cal_tags : list of str
        target tags that get processed in this function

    Returns
    -------
    dict
        the combined '_g_spec'/'_g_bls'/'_g_phase'/'_nog_spec' entries, with
        '_g_spec' weighted-averaged across scans (mirrors control.py's _sum_corr)
    """
    channel_freqs = merged_params['channel_freqs']
    k_key = merged_params['product_names']['K']
    b_key = merged_params['product_names']['B']
    g_key = merged_params['product_names']['G']

    nogaintag = ['bpcal', 'delaycal']
    gaintag = ['gaincal', 'target', 'bfcal']
    gaincaltag = ['gaincal', 'bfcal']

    # Create empty av_corr lists that this loop could possibly append to
    summarise = {}
    for target in cbf_targets:
        if any(k in target.tags for k in gaintag):
            summarise[target.name + '_g_spec'] = []
            summarise[target.name + '_g_bls'] = []
            if any(k in target.tags for k in gaincaltag):
                summarise[target.name + '_g_phase'] = []
        if any(k in target.tags for k in nogaintag):
            summarise[target.name + '_nog_spec'] = []

    nchans = 1024
    # indices for antennas in cross-correlation baselines
    index0, index1 = xc_lookup[:, 0], xc_lookup[:, 1]

    for compscan_index, label, target in d.compscans():
        if not any(k in target.tags for k in cal_tags):
            continue

        d.select(scans="track", compscans=compscan_index, corrprods='cross')
        t0, t1 = d.timestamps[0], d.timestamps[-1]
        logger.info("scan %d: %s", compscan_index, target.name)

        is_gaintag = any(k in target.tags for k in gaintag)
        is_nogaintag = any(k in target.tags for k in nogaintag)
        is_bpcal = 'bpcal' in target.tags

        k_delay = get_telstate_product(telstate, k_key, t1)
        b_gain = get_telstate_product(telstate, b_key, t1)
        b_gain = interp_b_nans(b_gain, channel_freqs)

        g_val = None
        if is_gaintag:
            g_val = get_telstate_product(telstate, g_key, t1)

        ant_names = [a.name for a in d.ants]
        reorder_idx, bls_ordering, bls_pol_ordering = calprocs.get_reordering(ant_names, d.corr_products)
        flags = d.raw_flags.dataset[1:]
        flags_reordered = flags[:, :, reorder_idx]
        vis = d.vis.dataset[1:]
        vis_reordered = vis[:, :, reorder_idx]
        weights = d.weights.dataset[1:]
        weights_reordered = weights[:, :, reorder_idx]
        timestamps = d.timestamps[1:]

        npol = len(set(map(tuple, bls_pol_ordering)))  # typically 4 or 2
        nbl = flags_reordered.shape[2] // npol
        flags_4d = flags_reordered.reshape(flags.shape[0], flags.shape[1], npol, nbl)
        vis_4d = vis_reordered.reshape(vis.shape[0], vis.shape[1], npol, nbl)
        weights_4d = weights_reordered.reshape(weights.shape[0], weights.shape[1], npol, nbl)
        # vis, flags, weights all HH and VV pol
        vis_autopol = vis_4d[:, :, :2, :]
        flags_autopol = flags_4d[:, :, :2, :]
        weights_autopol = weights_4d[:, :, :2, :]

        # apply K, B manually (and G, except for nogaintag targets)
        # add time, freq axes to k_delay and time, pol, ant axes to channel_freqs
        g_from_k = np.exp(2j * np.pi * k_delay[np.newaxis, np.newaxis, :, :]
                          * channel_freqs[np.newaxis, :, np.newaxis, np.newaxis])
        vis_autopol = apply_parallel_pol(g_from_k, vis_autopol, index0, index1)
        
        # Add time axis to B
        b_broadcast = b_gain[np.newaxis, :, :, :]
        vis_autopol = apply_parallel_pol(b_broadcast, vis_autopol, index0, index1)
        flags_autopol = flags_autopol | np.isnan(vis_autopol)

        if g_val is not None:
            # Add time and freq axes to G
            g_broadcast = g_val[np.newaxis, np.newaxis, :, :]
            vis_autopol = apply_parallel_pol(g_broadcast, vis_autopol, index0, index1)

        if is_bpcal:
            model = build_model(target, target.name, timestamps, telstate,
                                merged_params['antennas'], merged_params['array_position'],
                                channel_freqs, xc_lookup)
            if model is not None:
                g_bchan = merged_params.get('g_bchan', 1)
                g_echan = merged_params.get('g_echan', 0)
                try:
                    g_flux_soln = solve_g_flux(vis_autopol, flags_autopol, weights_autopol,
                                               model, xc_lookup, refant, g_bchan, g_echan)
                    # add same time, freq axes as G
                    g_flux_broadcast = g_flux_soln[np.newaxis, np.newaxis, :, :]
                    vis_autopol = apply_parallel_pol(g_flux_broadcast, vis_autopol,
                                                              index0, index1)
                except Exception as e:
                    print(f"G_FLUX reconstruction failed for {target.name}: {e}")

        if is_nogaintag:
            # NOG_SPEC: K,B-only
            av_vis, av_flags, av_weights = calprocs_dask.wavg_t_f(vis_autopol, flags_autopol, weights_autopol, nchans)
            ant_idx = np.where(((xc_lookup[:, 0] == refant)
                               ^ (xc_lookup[:, 1] == refant)))[0]
            av_vis = av_vis[..., ant_idx]
            av_flags = av_flags[..., ant_idx]

            av_vis[av_flags] = np.nan
            logger.info("Computing _nog_spec for %s", target.name)
            with ProgressBar():
                val = (av_vis.compute(), np.mean(timestamps))
            summarise[target.name + '_nog_spec'].append(val)

        if is_gaintag:
            # G_SPEC: full K,B,G-correction
            av_vis, av_flags, av_weights = calprocs_dask.wavg_t_f(vis_autopol, flags_autopol, weights_autopol, nchans)
            av_vis, av_flags, av_weights = calprocs_dask.wavg_ant(av_vis, av_flags, av_weights,
                                                                  merged_params['antennas'], xc_lookup)
            logger.info("Computing _g_spec for %s", target.name)
            with ProgressBar():
                av_vis, av_flags, av_weights = da.compute(av_vis, av_flags, av_weights)
            summarise[target.name + '_g_spec'].append((av_vis, av_flags, av_weights))

            # G_BLS: same corrected spectrum, just not antenna-averaged and coarser in frequency averaging
            av_vis, av_flags, av_weights = calprocs_dask.wavg_t_f(vis_autopol, flags_autopol, weights_autopol, 8)
            av_vis[av_flags] = np.nan
            logger.info("Computing _g_bls for %s", target.name)
            with ProgressBar():
                val = (av_vis.compute(), np.mean(timestamps))
            summarise[target.name + '_g_bls'].append(val)

            ## G_PHASE: antenna-averaged, coarse-frequency
            if any(k in target.tags for k in gaincaltag):
                av_vis, av_flags, av_weights = calprocs_dask.wavg_t_f(vis_autopol, flags_autopol, weights_autopol, 8)
                av_vis, av_flags, av_weights = calprocs_dask.wavg_ant(av_vis, av_flags, av_weights,
                                                                      merged_params['antennas'], xc_lookup)
                av_vis[av_flags] = np.nan
                logger.info("Computing _g_phase for %s", target.name)
                with ProgressBar():
                    val = (av_vis.compute(), np.mean(timestamps))
                summarise[target.name + '_g_phase'].append(val)

    # Combine each gain target's per-scan '_g_spec' entries into the single weighted-averaged entry av_corr
    summarise_final = summarise.copy()
    for key in summarise.keys():
        if key.endswith('_g_spec') and summarise[key]:
            wavg = list(zip(*summarise[key]))
            vis, flags, weights = [da.stack(a) for a in wavg]
            vis, flags, weights = calprocs_dask.wavg_full(vis, flags, weights, threshold=1)
            summarise_final[key] = [(vis, flags, weights)]
    return summarise_final


def auto_cross_av_corr(d, telstate, telstate_full, merged_params):
    """Build av_corr's auto correlation, cross polarisation entries.

    For bfcal targets observed while the noise diode was firing, applies the
    K, B, G solutions (parallel-hand) plus KCROSS_DIODE/BCROSS_DIODE
    (cross-hand) to the auto-correlation cross-hand (H x V) data, and summarises
    it per antenna.

    Note: each solution is reshaped/broadcast to vis's (time, freq, pol, ant)
    axes.

    Parameters
    ----------
    d : :class:'katdal.DataSet'
        dataset, already selected down to relevant server's channel slice
    telstate : :class:'katsdptelstate.TelescopeState'
        telescope state, 'CBID_cal_' namespace
    telstate_full : :class:'katsdptelstate.TelescopeState'
        root (unviewed) telescope state -- needed for the raw per-antenna noise
        diode sensors, which live outside the cal namespace
    merged_params : dict
        this server's finalised, local parameters

    Returns
    -------
    dict
        {'<target_name>_auto_cross': [(av_vis, mean_timestamp)]}
    """
    nant = len(merged_params['antennas'])
    channel_freqs = merged_params['channel_freqs']
    k_key = merged_params['product_names']['K']
    b_key = merged_params['product_names']['B']
    g_key = merged_params['product_names']['G']
    kcross_key = merged_params['product_names']['KCROSS_DIODE']
    bcross_key = merged_params['product_names']['BCROSS_DIODE']

    result = {}
    d.select(scans="track")
    for compscan_index, label, target in d.compscans():
        target_name = target.name
        if 'bfcal' not in target.tags:
            continue

        d.select(scans="track", compscans=compscan_index)
        if len(d.timestamps) <= 1:
            continue
        t0, t1 = d.timestamps[0], d.timestamps[-1]

        ant_names = [a.name for a in d.ants]
        nd_on = check_noise_diode(telstate_full, ant_names, [t0, t1])
        if not np.any(nd_on):
            continue

        k_delay = get_telstate_product(telstate, k_key, t1)
        b_gain = get_telstate_product(telstate, b_key, t1)
        g_val = get_telstate_product(telstate, g_key, t1)
        if k_delay is None or b_gain is None or g_val is None:
            continue
        logger.info("scan %d: %s", compscan_index, target_name)

        d.select(scans="track", compscans=compscan_index, corrprods='auto')
        auto_bls_wanted = [[a, a] for a in ant_names]
        reorder_idx, bls_ordering, bls_pol_ordering = calprocs.get_reordering(
            ant_names, d.corr_products, output_order_bls=auto_bls_wanted)
        npol = len(sorted(set(map(tuple, bls_pol_ordering))))
        if npol < 4:
            continue

        vis = d.vis.dataset[1:][:, :, reorder_idx]
        flags = d.raw_flags.dataset[1:][:, :, reorder_idx]
        weights = d.weights.dataset[1:][:, :, reorder_idx]
        timestamps = d.timestamps[1:]

        vis_4d = vis.reshape(vis.shape[0], vis.shape[1], npol, nant)
        flags_4d = flags.reshape(flags.shape[0], flags.shape[1], npol, nant)
        weights_4d = weights.reshape(weights.shape[0], weights.shape[1], npol, nant)
        vis_cross = vis_4d[:, :, 2:, :]
        flags_cross = flags_4d[:, :, 2:, :]
        weights_cross = weights_4d[:, :, 2:, :]

        # K:
        # add time, freq axes to k_delay, and time, pol, ant axes to to channel_freqs
        g_from_k = np.exp(2j * np.pi * k_delay[np.newaxis, np.newaxis, :, :]
                          * channel_freqs[np.newaxis, :, np.newaxis, np.newaxis])
        vis_cross = apply_cross_pol(g_from_k, vis_cross)

        # B:
        # add time axis to B 
        b_broadcast = b_gain[np.newaxis, :, :, :]
        vis_cross = apply_cross_pol(b_broadcast, vis_cross)
        flags_cross = flags_cross | np.isnan(vis_cross)   # B can have NaN at invalid channels

        # G:
        # add time,freq axis to G
        g_broadcast = g_val[np.newaxis, np.newaxis, :, :]
        vis_cross = apply_cross_pol(g_broadcast, vis_cross)

        # KCROSS_DIODE, BCROSS_DIODE
        kcross_vals = telstate.get_range(kcross_key, st=t0, et=t1)
        bcross_gain = get_telstate_product(telstate, bcross_key, t1)
        if not kcross_vals or bcross_gain is None:
            continue
        kcross_delay = kcross_vals[-1][0]

        kcross_delay = np.nanmedian(kcross_delay, axis=-1, keepdims=True)
        kcross_delay = np.repeat(kcross_delay, nant, axis=-1)
        # same reshape as K's g_from_k
        g_from_kcross = np.exp(2j * np.pi * kcross_delay[np.newaxis, np.newaxis, :, :]
                               * channel_freqs[np.newaxis, :, np.newaxis, np.newaxis])

        bcross_gain = np.nanmedian(bcross_gain, axis=-1, keepdims=True)
        bcross_gain = np.repeat(bcross_gain, nant, axis=-1)
        # add time axis to BCROSS_DIODE
        bcross_broadcast = bcross_gain[np.newaxis, :, :, :]

        vis_cross = apply_cross_pol(g_from_kcross, vis_cross)
        vis_cross = apply_cross_pol(bcross_broadcast, vis_cross)
        flags_cross = flags_cross | np.isnan(vis_cross)

        av_vis, av_flags, av_weights = calprocs_dask.wavg_t_f(vis_cross, flags_cross, weights_cross, 1024)
        av_vis[av_flags] = np.nan
        logger.info("Computing _auto_cross for %s", target_name)
        with ProgressBar():
            val = (da.compute(av_vis)[0], np.mean(timestamps))
        result.setdefault(target_name + '_auto_cross', []).append(val)

    return result


def gather_targets_timestamps_av_corr(d, cal_tags):
    """Build av_corr's 'targets'/'timestamps' entries.

    Parameters
    ----------
    d : :class:'katdal.DataSet'
        dataset
    cal_tags : list of str
        target tags to include

    Returns
    -------
    dict
        {'targets': [...], 'timestamps': [...]}
    """
    timestamps_by_compscan = []
    targets_by_compscan = []
    d.select(scans="track")
    for compscan_index, label, target in d.compscans():
        if not any(k in target.tags for k in cal_tags):
            continue
        d.select(scans="track", compscans=compscan_index)
        if len(d.timestamps) <= 1:
            continue
        timestamps = d.timestamps[1:].copy()
        mean_timestamp = np.mean(timestamps)
        timestamps_by_compscan.append((timestamps, mean_timestamp))
        targets_by_compscan.append((target, mean_timestamp))   # target is already a katpoint.Target

    return {'timestamps': timestamps_by_compscan, 'targets': targets_by_compscan}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', help="rdb file path, or full archive url (with ?token=...)")
    # Reports are written to the current directory as default
    parser.add_argument('--output-dir', default=os.getcwd())
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
    # Silencing katsdpcal and katsdpcalproc logger
    logging.getLogger('katsdpcal').setLevel(logging.WARNING)
    logging.getLogger('katsdpcalproc').setLevel(logging.ERROR)

    f = args.source
    logger.info("Opening dataset: %s", f)
    # Turn gaincal_flux off because report.py already does scaling (rescaling is automatic with
    # katdal.open unless turned off) and do not use applycal since G_FLUX is never stored in
    # telstate for bpcal target.
    d = katdal.open(f, gaincal_flux=None)

    logger.info("%s", d)

    CBID = d.source.capture_block_id
    stream_name = d.source.stream_name
    telstate_full = d.source.telstate
    telstate = telstate_full.view(CBID + '_cal_')
    n_chans_total = d.spectral_windows[d.spw].num_chans  # full-band channel count (e.g. 32768)

    # Pick the pipeline parameter file matching the observation band/mode
    spw = d.spectral_windows[d.spw]
    band = spw.band
    bandwidth = spw.bandwidth
    narrow54 = 0 < bandwidth <= 54e6
    narrow107 = 54e6 < bandwidth <= 107e6
    modes = {'narrowband': ['narrow_54', 'narrow_107'], 'wideband': 'wide'}
    if narrow54:
        mode = modes['narrowband'][0]
    elif narrow107:
        mode = modes['narrowband'][1]
    else:
        mode = modes['wideband']
    param_filename = 'pipeline_parameters_meerkat_{}_{}.txt'.format(band, mode)
    param_file = os.path.join(param_dir, param_filename)

    file_params = parameters_from_file(param_file)

    # Add whatever parameters the online pipeline actually ran with (telstate's own
    # cal_param_ keys, e.g. g_bchan/g_echan) on top of the file defaults,

    telstate_params = {}
    for key in telstate.keys():
        if 'cal_param' in key:
            telstate_params[key] = telstate[key]
    pipeline_params = {
        key.removeprefix('cal_param_'): value
        for key, value in telstate_params.items()
    }

    # The parameters here are not yet finalised, this step is done per server with
    # katsdpcal finalise_parameters
    base_params = file_params | pipeline_params
    
    # Cluster reports of the same CBID in one directory
    base_path = os.path.join(args.output_dir, f'{CBID}_cal_reports')

    SERVERS = num_servers(telstate, CBID)
    logger.info("Servers found = %d", SERVERS)

    cal_tags = ['gaincal', 'target', 'bfcal', 'bpcal', 'delaycal']
    cbf_targets = d.catalogue.targets

    # targets/timestamps have no frequency axis, built once, reused for every server.
    shared_av_corr = gather_targets_timestamps_av_corr(d, cal_tags)

    for server_id in range(SERVERS):
        logger.info("Building av_corr for server %d ", server_id)

        merged_params, xc_lookup, refant = build_server_merged_params(
            base_params, telstate, SERVERS, server_id)

        # Restrict d to exactly this server's channel slice
        channel_slice = slice(n_chans_total * server_id // SERVERS,
                              n_chans_total * (server_id + 1) // SERVERS)
        d.select(channels=channel_slice)

        av_corr = dict(shared_av_corr)
        av_corr.update(gather_model_av_corr(d, telstate, merged_params, xc_lookup, cbf_targets))
        av_corr.update(gather_flag_av_corr(d, cal_tags))
        av_corr.update(summarise_av_corr(d, telstate, merged_params, xc_lookup,
                                                refant, cbf_targets, cal_tags))
        av_corr.update(auto_cross_av_corr(d, telstate, telstate_full, merged_params))

        av_corr_dd = defaultdict(list, av_corr)
        # give each server its own subdirectory so the figures (and the .rst/.html)
        # never clash and overwrite.
        server_path = os.path.join(base_path, f'{CBID}_{stream_name}_calreport_cal.{server_id + 1}')
        os.makedirs(server_path, exist_ok=True)

        logger.info("Writing report for server %d to %s", server_id + 1, server_path)
        with ProgressBar():
            report.make_cal_report(telstate, CBID, stream_name, merged_params, server_path, av_corr_dd, st=0)
        logger.info("Wrote %s/calreport%d.rst --", server_path, server_id + 1)

    logger.info("Done -- %d report(s) written under %s", SERVERS, base_path)


if __name__ == "__main__":
    main()
