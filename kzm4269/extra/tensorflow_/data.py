import tensorflow as tf

from kzm4269.extra import h5py_
from kzm4269.extra.tensorflow_.session import require_session

__all__ = [
    'iter_dataset',
    'dataset_from_hdf5',
]


def iter_dataset(src, sess=None):
    if isinstance(src, tf.data.Dataset):
        src = src.make_one_shot_iterator().get_next()

    sess = require_session(sess)
    try:
        while True:
            yield sess.run(src)
    except tf.errors.OutOfRangeError:
        pass


def dataset_from_hdf5(src):
    """Create a `tf.data.Dataset` from a `h5py.Dataset`.

    Parameters
    ----------
    src : h5py.Dataset

    Returns
    -------
    dst : tf.data.Dataset
    """
    dst = tf.data.Dataset.from_generator(
        generator=lambda: ((c,) for c in h5py_.iter_chunks(src)),
        output_types=(tf.as_dtype(src.dtype),),
        output_shapes=((None,) + src.shape[1:],),
    )
    dst = dst.map(lambda t: t)
    dst = dst.apply(tf.contrib.data.unbatch())
    return dst
