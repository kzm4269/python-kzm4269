import contextlib

import tensorflow as tf

__all__ = [
    'temp_logging',
]


@contextlib.contextmanager
def temp_logging(level=None):
    """Return a context manager changing and restoring the logging level"""
    if level is None:
        level = tf.logging.INFO
    original_level = tf.logging.get_verbosity()
    try:
        tf.logging.set_verbosity(level)
        yield level
    finally:
        tf.logging.set_verbosity(original_level)
