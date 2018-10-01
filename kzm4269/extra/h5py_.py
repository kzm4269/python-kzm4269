import h5py
import numpy as np

from kzm4269 import misc

__all__ = [
    'require_extendible_dataset',
    'extend_dataset',
    'iter_chunks',
    'dump_tree',
]


def require_extendible_dataset(h5, name, chunks, dtype=None, axis=0, **kwargs):
    """Open a extendible dataset, creating it if it doesn't exist.

    Parameters
    ----------
    h5 : h5py.Group
    name : str
        Name of the dataset (absolute or relative).
    chunks : tuple of ints
        Chunk shape.
    dtype : np.dtype or str, optional
        Numpy dtype or string.  If omitted, the dtype of the dataset exists or
        dtype('f') will be used.
    axis : int or tuple of ints, optional
        The axis over which to extend the dataset.
    kwargs :
        Optional keyword arguments for `h5py.Group.require_dataset()`.

    Returns
    -------
    dataset : h5py.Dataset
    """
    assert isinstance(h5, h5py.Group)

    chunks = tuple(chunks)

    maxshape = np.array(chunks, dtype=object)
    maxshape[axis] = None
    maxshape = tuple(maxshape)

    shape = np.array(chunks)
    shape[axis] = 0
    shape = tuple(shape)

    if not isinstance(h5.get(name), h5py.Dataset):
        return h5.create_dataset(
            name=name, shape=shape, dtype=dtype,
            chunks=chunks, maxshape=maxshape, **kwargs)

    dataset = h5[name]
    dataset = h5.require_dataset(
        name, shape=dataset.shape,
        dtype=dtype if dtype else dataset.dtype,
        **kwargs)

    if dataset.maxshape != maxshape:
        raise TypeError(
            'Max shapes do not match (existing {})'.format(dataset.shape))

    return dataset


def extend_dataset(h5, name, data, chunk_len=None, axis=0, **kwargs):
    """Extend a HDF5 dataset.

    Parameters
    ----------
    h5 : h5py.Group
    name : str
        Name of the dataset (absolute or relative).  Provide None to make an
        anonymous dataset.
    data : array-like
        Provide data to extend the dataset.
    chunk_len : int, optional
    axis : int, optional
        The axis over which to extend the dataset.
    **kwargs :
        Optional keyword arguments for `h5py.Group.require_dataset()`.
    """
    data = np.asarray(data)

    chunks = list(data.shape)
    if chunk_len:
        chunks[axis] = chunk_len

    if isinstance(h5.get(name), h5py.Dataset):
        dataset = h5[name]
    else:
        dataset = require_extendible_dataset(
            h5=h5, name=name, chunks=tuple(chunks),
            dtype=data.dtype, axis=axis, **kwargs)

    if not data.size:
        return

    shape = list(dataset.shape)
    shape[axis] += data.shape[axis]
    dataset.resize(tuple(shape))

    index = [slice(None)] * data.ndim
    index[axis] = slice(-data.shape[axis], None)
    dataset[tuple(index)] = data


def iter_chunks(dataset, chunk_len=None):
    """Iterate chunks of a dataset of a HDF5 file.

    Parameters
    ----------
    dataset : h5py.Dataset
    chunk_len : int, optional
        If `None` is given, use `dataset.chunks[0]` or `dataset.shape[0]`.
    """
    assert isinstance(dataset, h5py.Dataset)
    if chunk_len is None:
        chunk_len = (dataset.chunks or dataset.shape)[0]
    return (dataset[i:i + chunk_len]
            for i in range(0, dataset.shape[0], chunk_len))


def dump_tree(root, max_depth=None, detail=False):
    def get_label(g):
        label = g.name.split('/')[-1]
        if isinstance(g, h5py.Group):
            label += '/'
        if detail:
            if isinstance(g, h5py.Group):
                label += f' ({len(g)} members)'
            else:
                label += f' (shape={g.shape}, dtype={g.dtype.str!r})'
        return label

    def get_children(g):
        if not isinstance(g, h5py.Group):
            return

        if g.name == root.name:
            depth = 0
        else:
            assert g.name.startswith(root.name)
            depth = g.name[len(root.name):].count('/')

        if not max_depth or depth < max_depth:
            yield from sorted(g.values(), key=get_label)

    return misc.dump_tree(root, label=get_label, children=get_children)
