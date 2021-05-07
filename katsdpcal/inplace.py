"""Tools for safely performing dask computations that overwrite their inputs.

Refer to :func:`store_inplace` for details.
"""

import inspect
import itertools

import numpy as np
import dask.array as da
import dask.base
import dask.core
import dask.optimization
import dask.array.optimization
from dask.blockwise import Blockwise
from dask.highlevelgraph import HighLevelGraph
try:
    from dask.highlevelgraph import MaterializedLayer
except ImportError:
    # Older verisons of dask don't have this. We only use it for an instance
    # check, so a dummy implementation suffices.
    class MaterializedLayer:
        pass


class _ArrayDependency:
    """An array that a task depends on.

    To make this object hashable, two arrays are considered to be equal if they
    refer to the same data with the same type, shape etc, even if they are
    different views.

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
    """Exception raised when an in-place data hazard is detected."""

    def __init__(self, source_key, target_key):
        self.source_key = source_key
        self.target_key = target_key
        super().__init__(
            'Data hazard between source key {} and target key {}'.format(source_key, target_key))


def _in_graph(dsk, key):
    try:
        return key in dsk
    except TypeError:
        # key is not hashable
        return False


def _slice_key(slc):
    """Turn a slice or tuple of slices into a hashable type."""
    if isinstance(slc, slice):
        return (slc.start, slc.stop, slc.step)
    elif isinstance(slc, tuple):
        return tuple(_slice_key(s) for s in slc)
    else:
        raise TypeError(f'Expected slice or tuple of slices, not {type(slc)}')


def _is_getter(dsk, v):
    """Check whether an element of a graph is a getter task."""
    # Getters can also have length 5 to pass optional arguments. For
    # now we ignore these to avoid dealing with locks. We also exclude
    # more complicated cases where the parameters are not simply a
    # node and a literal expression.
    return (type(v) is tuple and len(v) == 3
            and v[0] in da.optimization.GETTERS
            and _in_graph(dsk, v[1])
            and not dask.core.has_tasks(dsk, v[2]))


def _array_get(dsk, key, cache):
    """Obtain a key from a dask graph.

    This is similar to :meth:`dask.core.get`, but it only executes tasks that
    are considered getters. Any other tasks raise :exc:`ValueError`.
    """
    if key in cache:
        return cache[key]
    v = dsk[key]
    if _is_getter(dsk, v):
        array = v[0](_array_get(dsk, v[1], cache), v[2])
    elif _in_graph(dsk, v):
        array = _array_get(dsk, v, cache)
    else:
        array = v

    if not isinstance(array, np.ndarray):
        raise ValueError(f'Key {key} does not refer to an array')
    cache[key] = array
    return array


class _StoreWrapper:
    """Interface for :meth:`dask.array.store`.

    It holds a map from indices to numpy arrays, and when called, sets the data in
    the corresponding array. This is a very limited interface that is only
    intended to work with :meth:`dask.array.store` rather than a general way to
    write to a stitched-together set of numpy arrays.

    Parameters
    ----------
    array : da.Array
        A dask array whose individual chunks refer directly to numpy arrays.
    """

    def __init__(self, array):
        slices = da.core.slices_from_chunks(array.chunks)
        graph = array.__dask_graph__()
        self._slice_map = {}
        cache = {}
        for key, slc in zip(dask.core.flatten(array.__dask_keys__()), slices):
            ndarray = _array_get(graph, key, cache)
            if not isinstance(ndarray, np.ndarray):
                raise ValueError(f'Target key {key} does not directly map a numpy array')
            # Slices are not hashable
            self._slice_map[_slice_key(slc)] = ndarray

    def __setitem__(self, idx, value):
        self._slice_map[_slice_key(idx)][:] = value


def _safe_inplace(sources, targets):
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
    # Create a graph with all the sources and targets in it
    dsk = HighLevelGraph.from_collections('store', {}, sources + targets)
    source_keys = list(itertools.chain.from_iterable(
        dask.core.flatten(source.__dask_keys__())
        for source in sources
    ))
    target_keys = list(itertools.chain.from_iterable(
        dask.core.flatten(target.__dask_keys__())
        for target in targets
    ))
    if len(set(target_keys)) < len(target_keys):
        raise ValueError('The target contains duplicate keys')

    dependencies = dict((k, dask.optimization.get_dependencies(dsk, k)) for k in dsk)
    # For each key, contains a set of _ArrayDependencys
    arrays = {}
    for k in dask.optimization.toposort(dsk, dependencies=dependencies):
        v = dsk[k]
        if isinstance(v, np.ndarray):
            arrays[k] = set([_ArrayDependency(v, True)])
        elif _in_graph(dsk, v):
            arrays[k] = arrays[v]
        else:
            out = set()
            is_getter = _is_getter(dsk, v)
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
            raise ValueError(f'Target key {key} does not directly map a numpy array')
    for i, src_key in enumerate(source_keys):
        for j, trg_key in enumerate(target_keys):
            if i != j:
                trg_array = next(iter(arrays[trg_key])).array
                for array_dep in arrays[src_key]:
                    if np.shares_memory(array_dep.array, trg_array):
                        raise UnsafeInplaceError(src_key, trg_key)


def store_inplace(sources, targets, safe=True, **kwargs):
    """Evaluate a dask computation and store results in the original numpy arrays.

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
    if isinstance(sources, da.Array):
        sources = [sources]
        targets = [targets]

    if any(not isinstance(s, da.Array) for s in sources):
        raise ValueError('All sources must be instances of da.Array')
    if any(not isinstance(t, da.Array) for t in targets):
        raise ValueError('All targets must be instances of da.Array')

    chunked_sources = [
        source.rechunk(target.chunks) for source, target in zip(sources, targets)
    ]
    if safe:
        _safe_inplace(chunked_sources, targets)
    store_wrappers = [_StoreWrapper(target) for target in targets]
    da.store(chunked_sources, store_wrappers, lock=False)


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


def _rename_layer(layer, keymap, salt):
    """Rename a single layer in a :class:`dask.highlevelgraph.HighLevelGraph`."""
    if type(layer) is Blockwise:
        new_indices = tuple(
            (_rename_key(name, salt) if ind is not None else name, ind)
            for name, ind in layer.indices)
        sub_keymap = {key: _rename_key(key, salt) for key in layer.dsk}
        kwargs = {}
        # The available arguments depend on the Dask version.
        sig = inspect.signature(Blockwise)
        for arg_name in ['output_blocks', 'annotations']:
            if arg_name in sig.parameters:
                kwargs[arg_name] = getattr(layer, arg_name)
        if 'io_deps' in sig.parameters:
            kwargs['io_deps'] = {
                _rename_key(key, salt): value for key, value in layer.io_deps.items()
            }
        return Blockwise(
            _rename_key(layer.output, salt),
            layer.output_indices,
            _rename_layer(layer.dsk, sub_keymap, salt),
            new_indices,
            {_rename_key(name, salt): value for name, value in layer.numblocks.items()},
            layer.concatenate,
            layer.new_axes,
            **kwargs)
    elif type(layer) is MaterializedLayer:
        mapping = {keymap[key]: _rename(value, keymap) for (key, value) in layer.mapping.items()}
        return MaterializedLayer(mapping, layer.annotations)
    else:
        return {keymap[key]: _rename(value, keymap) for (key, value) in layer.items()}


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
    dsk = array.__dask_graph__()
    keymap = {key: _rename_key(key, salt) for key in dsk}
    if isinstance(dsk, HighLevelGraph):
        layers = {
            _rename_key(name, salt): _rename_layer(layer, keymap, salt)
            for name, layer in dsk.layers.items()
        }
        dependencies = {
            _rename_key(name, salt): {_rename_key(dep, salt) for dep in deps}
            for name, deps in dsk.dependencies.items()
        }
        array.dask = HighLevelGraph(layers, dependencies)
    else:
        array.dask = _rename_layer(dsk, keymap)
    new_name = _rename_key(array.name, salt)
    try:
        array.name = new_name
    except TypeError:
        # Recent versions of dask (at least since 2021.3.0) require setting
        # _name rather than name.
        array._name = new_name
