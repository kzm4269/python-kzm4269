import functools as ft

import tensorflow as tf
from tensorflow.python.keras.models import Model


def save_yaml(model, filename):
    with open(filename, 'w') as fp:
        fp.write(model.to_yaml())


def load_model_from_yaml(filename, custom_objects=None) -> Model:
    with open(filename, 'r') as fp:
        return tf.keras.models.model_from_yaml(
            yaml_string=fp.read(),
            custom_objects=custom_objects)


def model_builder(name, custom_objects=None, new_graph=True):
    def decorator(builder):

        @ft.wraps(builder)
        def wrapper(*args, **kwargs):
            if 'name' not in kwargs:
                kwargs['name'] = name

            if new_graph:
                graph = tf.Graph()
                sess = tf.keras.backend.get_session()
                try:
                    with graph.as_default():
                        model = builder(*args, **kwargs)
                        config = model.get_config()
                    return type(model).from_config(config, custom_objects)
                finally:
                    tf.keras.backend.set_session(sess)
            else:
                return builder(*args, **kwargs)

        return wrapper

    return decorator


def walk_layers(model):
    try:
        yield model.name, model
    except AttributeError:
        return

    try:
        layers = model.layers
    except AttributeError:
        layers = ()

    for layer in layers:
        for path, sublayer in walk_layers(layer):
            yield model.name + '/' + path, sublayer


def walk_weights(model):
    for path, layer in walk_layers(model):
        for w in layer.weights:
            yield path, w


def _main():
    from takahara.python.utils.tf_utils.keras.layers import GaussianNoise

    layer = GaussianNoise(0.5)
    layer(tf.keras.layers.Input([1]))
    print(layer.built_layer.weights)


if __name__ == '__main__':
    _main()
