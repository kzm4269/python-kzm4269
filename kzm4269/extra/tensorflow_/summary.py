import os
from datetime import datetime

import tensorflow as tf

__all__ = [
    'temp_logdir',
    'temp_writer',
]


def temp_logdir(root=None, name=None):
    if root is None:
        root = os.path.join(os.path.expanduser('~'), 'tmp', 'tensorboard')
    if name is None:
        name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')
    return os.path.join(root, name)


def temp_writer(root=None, name=None):
    return tf.summary.FileWriter(
        logdir=temp_logdir(root, name),
        graph=tf.get_default_graph(),
    )
