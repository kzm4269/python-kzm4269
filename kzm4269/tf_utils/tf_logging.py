import contextlib

import tensorflow as tf


@contextlib.contextmanager
def temp_logging(level=None):
    """Return a context manager changing and restoring the logging level"""
    if level is None:
        level = tf.logging.INFO
    level_ = tf.logging.get_verbosity()
    try:
        tf.logging.set_verbosity(level)
        yield level
    finally:
        tf.logging.set_verbosity(level_)
