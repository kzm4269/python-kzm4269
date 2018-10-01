import os
from contextlib import contextmanager

__all__ = [
    'open_r',
    'open_w',
]


@contextmanager
def open_r(path, open_func=open, mode='r', arg_name='src', **kwargs):
    """Open a file to read.
    If a file object is given, return it as is.
    """
    try:
        fspath = os.fspath(path)
    except TypeError:
        pass
    else:
        with open_func(fspath, mode=mode, **kwargs) as fh:
            yield iter(fh)
        return

    try:
        lines = iter(path)
    except TypeError:
        raise ValueError(
            arg_name +
            ' must be a path-like object, file handle, read function'
            ' or generator.'
        )
    yield lines


@contextmanager
def open_w(path, open_func=open, mode='w', **kwargs):
    """Open a file to write.
    If a file object is given, return it as is.
    """
    try:
        fspath = os.fspath(path)
    except TypeError:
        yield path
    else:
        with open_func(fspath, mode=mode, **kwargs) as fh:
            yield fh
