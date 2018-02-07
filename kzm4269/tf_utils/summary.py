import os
from datetime import datetime

import tensorflow as tf


def temp_logdir(root=None, name=None):
    if root is None:
        root = os.path.join(os.path.expanduser('~'), 'tmp', 'tensorboard')
    if name is None:
        name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return os.path.join(root, name)


def temp_writer(root=None, name=None):
    return tf.summary.FileWriter(
        temp_logdir(root, name),
        tf.get_default_graph())
