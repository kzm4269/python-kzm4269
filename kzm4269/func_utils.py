"""Utilities for functools"""
import functools as ft
import inspect


def nonstrict_partial(func, *args, **kwargs):
    """ignore invalid kwargs"""
    parameters = inspect.signature(func).parameters
    param_kinds = [p.kind for p in parameters.values()]
    if inspect.Parameter.VAR_KEYWORD not in param_kinds:
        kwargs = {k: v for k, v in kwargs.items() if k in parameters}
    return ft.partial(func, *args, **kwargs)
