import numpy as np
import tensorflow as tf

from .. import numpy as npu

__all__ = [
    'actual_shape',
    'jacobian',
    'log10',
    'diff',
    'rdiff',
    'irdiff',
]


def actual_shape(tensor, axis=None, default=None):
    if axis is None:
        axes = range(len(tensor.shape))
    elif isinstance(axis, int):
        axes = [axis]
    else:
        axes = list(axis)

    dim_values = [tensor.shape[i].value for i in axes]
    if all(v is not None for v in dim_values):
        shape = dim_values
    else:
        if default is None:
            default = tf.shape(tensor)
        shape = [
            default[i] if v is None else v
            for i, v in zip(axes, dim_values)
        ]

    if isinstance(axis, int):
        return shape[0]
    else:
        return tuple(shape)


def jacobian(y, x, name=None):
    with tf.name_scope(name, 'jacobian', [y, x]):
        y = tf.convert_to_tensor(y)
        x = tf.convert_to_tensor(x)
        if len(y.shape) != 2:
            raise ValueError(y)
        if len(x.shape) != 2:
            raise ValueError(x)

        nbatches, y_dim = actual_shape(y)
        x_dim = actual_shape(x, axis=1)

        _, result = tf.while_loop(
            cond=lambda i, _: i < y_dim,
            body=lambda i, res: (
                i + 1,
                res.write(i, tf.gradients(y[:, i], x)[0])
            ),
            loop_vars=(
                tf.constant(0, tf.int32),
                tf.TensorArray(y.dtype, size=y_dim),
            ),
        )

        return tf.reshape(
            tf.transpose(result.stack(), (1, 0, 2)),
            (nbatches, y_dim, x_dim))


def log10(x, name=None):
    with tf.name_scope(name, 'log10', values=[x]):
        return tf.log(x) / tf.log(10.)


def diff(inputs, axis=-1, name=None):
    with tf.name_scope(name, 'diff', [inputs, axis]):
        inputs = tf.convert_to_tensor(inputs)
        ndims = len(inputs.shape)
        inputs0 = inputs[npu.slices(ndims, {axis: np.s_[:-1]})]
        inputs1 = inputs[npu.slices(ndims, {axis: np.s_[1:]})]
        return inputs1 - inputs0


def _diff_recursively(
        y, x=None, axis=-1, name=None, inverse=False, default_name=None):
    with tf.name_scope(name, default_name, [y, x]):
        y = tf.convert_to_tensor(y)
        if x is None:
            x = tf.constant(1, dtype=y.dtype)
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        if len(x.shape) > 0:
            x = tf.reshape(x, npu.slices(len(y.shape), {axis: -1}, default=1))

        y_shape = actual_shape(y)
        y_ndims = len(y_shape)

        if len(x.shape) == 0:
            dx = x
        else:
            dx = diff(x, axis=axis)

        def body(i, res):
            with tf.name_scope('body'):
                tail = res[npu.slices(y_ndims, {axis: np.s_[i - 1:]})]

                head0 = res[npu.slices(y_ndims, {axis: np.s_[:i - 1]})]
                head1 = res[npu.slices(y_ndims, {axis: np.s_[1:i]})]
                if len(x.shape) == 0:
                    dx0 = dx
                else:
                    dx0 = dx[npu.slices(y_ndims, {axis: np.s_[:i - 1]})]

                if inverse:
                    head = head1 - head0 * dx0
                else:
                    head = (head1 - head0) / dx0

                return tf.reshape(
                    tensor=tf.concat([head, tail], axis=axis),
                    shape=y_shape,
                )

        _, result = tf.while_loop(
            cond=lambda i, _: i > 0,
            body=lambda i, res: (i - 1, body(i, res)),
            loop_vars=[tf.constant(y_shape[axis], tf.int32), y],
        )
        return result


def rdiff(y, x=None, axis=-1, name=None):
    return _diff_recursively(
        y, x,
        axis=axis,
        inverse=False,
        default_name='rdiff',
        name=name,
    )


def irdiff(y, x=None, axis=-1, name=None):
    return _diff_recursively(
        y, x,
        axis=axis,
        inverse=True,
        default_name='irdiff',
        name=name,
    )
