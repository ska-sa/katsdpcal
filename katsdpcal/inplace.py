"""Tools for safely performing dask computations that overwrite their inputs.

Refer to :func:`store_inplace` for details.
"""

from __future__ import print_function, division, absolute_import

import numpy as np
import dask.array as da
import dask.base
import dask.core
import dask.optimize
import dask.array.optimization


class _ArrayDependency(object):
    """An array that a task depends on. To make this object hashable, two
    arrays to be equal if they refer to the same data with the same type,
    shape etc, even if they are different views.

    An "elementwise" dependency is one where the output of the task has the
    same shape as the input and the dependencies are elementwise.
    """
    def __init__(self, array, elementwise):
        self.array = array
        self.elementwise = elementwise

    def __eq__(self, other):
        return (type(self.array) == type(other.array)  # noqa: E721
                and self.array.__array_interface__ == other.array.__array_interface__
                and self.elementwise == other.elementwise)

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        # Would be nice to have a better way to hash __array_interface__, but
        # it contains lists so it would need some recursive mechanism.
        return hash((type(self.array), repr(self.array.__array_interface__), self.elementwise))

    def __getitem__(self, index):
        if self.elementwise:
            return _ArrayDependency(self.array[index], True)
        else:
            return self


class UnsafeInplaceError(Exception):
    """Exception raised when an in-place data hazard is detected"""
    def __init__(self, source_key, target_key):
        self.source_key = source_key
        self.target_key = target_key
        super(UnsafeInplaceError, self).__init__(
            'Data hazard between source key {} and target key {}'.format(source_key, target_key))


def _in_graph(dsk, key):
    try:
        return key in dsk
    except TypeError:
        # key is not hashable
        return False


def _safe_in_place(dsk, source_keys, target_keys):
    """Safety check on :func:`safe_in_place`. It uses the following algorithm:

    1. For each key in the graph, determine a set of :class:`_ArrayDependency`s that
    contribute to it. In most cases this is just all arrays that are reachable
    from that key, but a chain of getters connected to a numpy array is treated
    specially so that only the part of the array that is sliced becomes part of
    the dependency. This is computed non-recursively by walking the graph in
    topological order.

    2. If a source and target corresponding to *different* chunks depend on
    overlapping numpy arrays, the operation is unsafe.
    """
    dependencies = dict((k, dask.optimize.get_dependencies(dsk, k)) for k in dsk)
    # For each key, contains a set of _ArrayDependencys
    arrays = {}
    for k in dask.optimize.toposort(dsk, dependencies=dependencies):
        v = dsk[k]
        if isinstance(v, np.ndarray):
            arrays[k] = set([_ArrayDependency(v, True)])
        elif _in_graph(dsk, v):
            arrays[k] = arrays[v]
        else:
            out = set()
            # Getters can also have length 5 to pass optional arguments. For
            # now we ignore these to avoid dealing with locks. We also exclude
            # more complicated cases where the parameters are not simply a
            # node and a literal expression.
            is_getter = (type(v) is tuple and len(v) == 3
                         and v[0] in da.optimization.GETTERS
                         and _in_graph(dsk, v[1])
                         and not dask.core.has_tasks(dsk, v[2]))
            for dep in dependencies[k]:
                for array_dep in arrays[dep]:
                    if is_getter:
                        out.add(array_dep[v[2]])
                    elif not array_dep.elementwise:
                        out.add(array_dep)
                    else:
                        out.add(_ArrayDependency(array_dep.array, False))
            arrays[k] = out
    for key in target_keys:
        if len(arrays[key]) != 1 or not next(iter(arrays[key])).elementwise:
            raise ValueError('Target key {} does not directly map a numpy array')
    for i, src_key in enumerate(source_keys):
        for j, trg_key in enumerate(target_keys):
            if i != j:
                trg_array = next(iter(arrays[trg_key])).array
                for array_dep in arrays[src_key]:
                    if np.shares_memory(array_dep.array, trg_array):
                        raise UnsafeInplaceError(src_key, trg_key)


def store_inplace(sources, targets, safe=True, **kwargs):
    """Evaluate a dask computation and write results back to the numpy arrays
    backing dask arrays.

    Dask is designed to operate on immutable data: the key for a node in the
    graph is intended to uniquely identify the value. It's possible to create
    tasks that modify the backing storage, but it can potentially create race
    conditions where a value might be replaced either before or after it is
    used. This function provides safety checks that will raise an exception if
    there is a risk of this happening.

    Despite the safety checks, it still requires some user care to be used
    safely:

    - The arrays in `targets` must be backed by numpy arrays, with no
      computations other than slicing. Thus, the dask functions
      :func:`~dask.array.asarray`, :func:`~dask.array.from_array`,
      :func:`~dask.array.concatenate` and :func:`~dask.array.stack` are safe.
    - The target keys must be backed by *distinct* numpy arrays. This is not
      currently checked (although duplicate keys will be detected).
    - When creating a target array with :func:`~dask.array.from_array`,
      ensure that the array has a unique name (e.g., by passing
      ``name=False``).
    - The safety check only applies to the sources and targets passed to this
      function. Any simultaneous use of objects based on the targets is
      invalid, and afterwards any dask objects based on the targets will be
      computed with the overwritten values.

    The safety check is conservative i.e., there may be cases where it will
    throw an exception even though the operation can be proven to be safe.

    Each source is rechunked to match the chunks of the target. In cases where
    the target is backed by a single large numpy array, it may be more
    efficient to construct a new dask wrapper of that numpy array whose
    chunking matches the source.

    Parameters
    ----------
    sources : iterable of :class:`dask.array.Array`
        Values to compute.
    targets : iterable of :class:`dask.array.Array`
        Destinations in which to store the results of computing `sources`, with
        the same length and matching shapes (the dtypes need not match, as long
        as they are assignable).
    safe : bool, optional
        If true (default), raise an exception if the operation is potentially
        unsafe. This can be an expensive operation (quadratic in the number of
        chunks).
    kwargs : dict
        Extra arguments are passed to the scheduler

    Raises
    ------
    UnsafeInplaceError
        if a data hazard is detected
    ValueError
        if the sources and targets have the wrong type or don't match
    """
    def store(target, source):
        target[:] = source

    if isinstance(sources, da.Array):
        sources = [sources]
        targets = [targets]

    if any(not isinstance(s, da.Array) for s in sources):
        raise ValueError('All sources must be instances of da.Array')
    if any(not isinstance(t, da.Array) for t in targets):
        raise ValueError('All targets must be instances of da.Array')

    dsk = {}
    src_keys = []
    trg_keys = []
    out_keys = []
    for source, target in zip(sources, targets):
        if source.shape != target.shape:
            raise ValueError('Source and target have different shapes')
        source_chunked = source.rechunk(target.chunks)
        slices = da.core.slices_from_chunks(target.chunks)
        name = 'store-' + source_chunked.name
        for src_key, trg_key, slc in zip(dask.core.flatten(source_chunked.__dask_keys__()),
                                         dask.core.flatten(target.__dask_keys__()),
                                         slices):
            key = (name,) + src_key[1:]
            dsk[key] = (store, trg_key, src_key)
            src_keys.append(src_key)
            trg_keys.append(trg_key)
            out_keys.append(key)
        dsk.update(source_chunked.dask)
        dsk.update(target.dask)
    if len(set(trg_keys)) < len(trg_keys):
        raise ValueError('The target contains duplicate keys')
    if safe:
        _safe_in_place(dsk, src_keys, trg_keys)
    dask.base.compute_as_if_collection(da.Array, dsk, out_keys)


def _rename(comp, keymap):
    """Compute the replacement for a computation by remapping keys through `keymap`."""
    if _in_graph(keymap, comp):
        return keymap[comp]
    elif dask.core.istask(comp):
        return (comp[0],) + tuple(_rename(c, keymap) for c in comp[1:])
    elif isinstance(comp, list):
        return [_rename(c, keymap) for c in comp]
    else:
        return comp


def _rename_key(key, salt):
    if isinstance(key, str):
        return 'rename-' + dask.base.tokenize([key, salt])
    elif isinstance(key, tuple) and len(key) > 0:
        return (_rename_key(key[0], salt),) + key[1:]
    else:
        raise TypeError('Cannot rename key {!r}'.format(key))


def rename(array, salt=''):
    """Rewrite the graph in a dask array to rename all the nodes.

    This is intended to be used when the backing storage has changed
    underneath, to invalidate any caches.

    Parameters
    ----------
    array : :class:`dask.array.Array`
        Array to rewrite. It is modified in place.
    salt : str, optional
        Value mixed in to the hash function used for renaming. If two arrays
        share keys, then calling this function on those arrays with the same
        salt will cause them to again share keys.
    """
    keymap = {key: _rename_key(key, salt) for key in array.dask}
    array.dask = {keymap[key]: _rename(value, keymap) for (key, value) in array.dask.items()}
    array.name = _rename_key(array.name, salt)
