import tensorflow as tf


def save_yaml(model, filename):
    with open(filename, 'w') as fp:
        fp.write(model.to_yaml())


def load_model_from_yaml(filename, **kwargs) -> tf.keras.models.Model:
    with open(filename, 'r') as fp:
        return tf.keras.models.model_from_yaml(
            yaml_string=fp.read(),
            **kwargs,
        )


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
