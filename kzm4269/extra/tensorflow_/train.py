import tensorflow as tf

__all__ = [
    'example',
    'bytes_feature',
    'float_feature',
    'int64_feature',
]


def example(**feature):
    return tf.train.Example(features=tf.train.Features(feature=feature))


def _feature(key, message_type, value):
    return tf.train.Feature(**{key: message_type(value=list(value))})


def bytes_feature(*value):
    return _feature('bytes_list', tf.train.BytesList, value)


def float_feature(*value):
    return _feature('float_list', tf.train.FloatList, value)


def int64_feature(*value):
    return _feature('int64_list', tf.train.Int64List, value)
