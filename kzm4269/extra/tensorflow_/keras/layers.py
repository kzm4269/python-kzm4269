import functools as ft

import tensorflow as tf

from kzm4269.extra.tensorflow_.subgraph import unique_layer_name

__all__ = [
    'set_trainable',
    'normalized',
    'unique_layer_name',
]


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
            normalization_ = None

        activation = kwargs.pop('activation', None)
        if activation:
            activation_ = tf.keras.layers.Activation(activation)
        else:
            activation_ = None

        if 'use_bias' not in kwargs:
            kwargs['use_bias'] = not bool(normalization)

        layer_ = layer(*args, **kwargs)

        def call(x):
            y = layer_(x)
            if normalization_:
                y = normalization_(y)
            if activation_:
                y = activation_(y)
            return y

        return call

    return wrapper
