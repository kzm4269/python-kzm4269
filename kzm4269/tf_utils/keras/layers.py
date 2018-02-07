import functools as ft

import tensorflow as tf

from ..subgraph import layer_type, StaticGraph


def custom_object_scope():
    return tf.keras.utils.custom_object_scope({
        StaticGraph.__name__: StaticGraph,
    })


def set_trainable(layer_or_model, trainable, recursive=False):
    layer_or_model.trainable = trainable
    if not recursive:
        return

    try:
        sublayers = layer_or_model.layers
    except AttributeError:
        return
    for sublayer in sublayers:
        set_trainable(sublayer, trainable)


def normalized(layer):
    @ft.wraps(layer)
    def wrapper(*args, **kwargs):
        normalization = kwargs.pop(
            'normalization', tf.keras.layers.BatchNormalization)
        if normalization:
            normalization_ = normalization()
        else:
            normalization_ = through()

        activation = kwargs.pop('activation', None)
        if activation:
            activation_ = tf.keras.layers.Activation(activation)
        else:
            activation_ = through()

        if 'use_bias' not in kwargs:
            kwargs['use_bias'] = bool(normalization)

        layer_ = layer(*args, **kwargs)

        def call(x):
            return activation_(normalization_(layer_(x)))

        return call

    return wrapper


def _dropout(x, rate, noise_shape=None, seed=None):
    rate_ = tf.get_variable(
        'rate', shape=(), dtype=tf.keras.backend.floatx(),
        initializer=tf.constant_initializer(rate),
        trainable=False)

    return tf.cond(
        tf.keras.backend.learning_phase(),
        lambda: tf.nn.dropout(x, 1 - rate_, noise_shape, seed),
        lambda: x)


Dropout = layer_type(_dropout, 'dropout')


def _gaussian_noise(x, stddev):
    stddev_ = tf.get_variable(
        'stddev', shape=(), dtype=tf.keras.backend.floatx(),
        initializer=tf.constant_initializer(stddev),
        trainable=False)

    return tf.cond(
        tf.keras.backend.learning_phase(),
        lambda: x + tf.random_normal(
            shape=tf.shape(x), mean=0., stddev=stddev_),
        lambda: x)


GaussianNoise = layer_type(_gaussian_noise, 'gaussian_noise')


def identity(x, name):
    return layer_type(tf.identity)(name=name)(x)


def stack(values, axis=0, name=None):
    if axis >= 0:
        axis += 1
    return layer_type(tf.stack, pack_args=True)(axis=axis, name=name)(values)


def unstack(value, num=None, axis=0, name=None):
    if axis >= 0:
        axis += 1
    return layer_type(tf.unstack)(num=num, axis=axis, name=name)(value)


def subtract(a, b, name=None):
    return layer_type(
        lambda a_, b_: a_ - b_,
        name='subtract')(name=name)([a, b])


# noinspection PyUnusedLocal
def through(*args, **kwargs):
    return lambda x: x
