import functools as ft
import inspect

__all__ = [
    'nonstrict_partial',
]


def nonstrict_partial(func, *args, **kwargs):
    """Make a function ignore invalid keyword arguments.

    Examples
    --------
    >>> from functools import partial
    >>> f = partial(lambda: None, foo=123)
    >>> f()
    Traceback (most recent call last):
        ...
    TypeError: <lambda>() got an unexpected keyword argument 'foo'
    >>> g = nonstrict_partial(lambda: None, foo=123)
    >>> g()
    >>> g(foo=123)
    Traceback (most recent call last):
        ...
    TypeError: <lambda>() got an unexpected keyword argument 'foo'
    """
    parameters = inspect.signature(func).parameters
    param_kinds = [p.kind for p in parameters.values()]
    if inspect.Parameter.VAR_KEYWORD not in param_kinds:
        kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in parameters
        }
    return ft.partial(func, *args, **kwargs)
