import functools as ft

import tensorflow as tf

TFRecordWriter = tf.python_io.TFRecordWriter

TFRecordGzipWriter = ft.partial(
    tf.python_io.TFRecordWriter,
    options=tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.GZIP,
    ),
)

TFRecordZlibWriter = ft.partial(
    tf.python_io.TFRecordWriter,
    options=tf.python_io.TFRecordOptions(
        compression_type=tf.python_io.TFRecordCompressionType.ZLIB,
    ),
)
