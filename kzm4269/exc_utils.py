"""Utilities for exception handling"""
from contextlib import contextmanager


@contextmanager
def wrap_exc(of=None, by=None):
    """Convert the types of exceptions.

    Parameters
    ----------
    of : type, optional
        If `None` is given, ignore expected exceptions.
    by : type or tuple of types, optional
        If `None` is given, wrap any exception.

    Examples
    --------
    >>> with wrap_exc(by=ValueError):
    ...     raise Exception('hello')
    Traceback (most recent call last):
    ...
    ValueError: hello
    >>> @wrap_exc(of=TypeError, by=ValueError)
    ... def f():
    ...     raise TypeError()
    ...
    >>> f()
    Traceback (most recent call last):
    ...
    ValueError
    """
    if of is None:
        of = Exception

    try:
        yield
    except of as e:
        if callable(by):
            raise by(e) from e
