import tensorflow as tf


def require_session(sess=None):
    if sess is None:
        sess = tf.get_default_session()
    if sess is None:
        raise RuntimeError(
            'Require a session. '
            'Try `tf.InteractiveSession()` before call this function.')
    return sess


def debug_session(target='', graph=None, config=None):
    """If config is not provided, choose some reasonable defaults for debugging
    use.
    """
    if not config:
        config = tf.ConfigProto(
            allow_soft_placement=True,
            gpu_options=tf.GPUOptions(
                allow_growth=True,
            ),
        )
    return tf.Session(target=target, graph=graph, config=config)
