"""Utilities for SymPy"""
import functools as ft

import numpy as np
import sympy as sy
from sympy.parsing.sympy_parser import parse_expr


def hstack(matrixes):
    matrixes = iter(matrixes)
    return ft.reduce(lambda a, b: a.row_join(b), matrixes, next(matrixes))


def vstack(matrixes):
    matrixes = iter(matrixes)
    return ft.reduce(lambda a, b: a.col_join(b), matrixes, next(matrixes))


def applyfunc(expr, func, *args, **kwargs):
    try:
        applyfunc_ = expr.applyfunc
    except AttributeError:
        return func(expr, *args, **kwargs)
    return applyfunc_(func, *args, **kwargs)


def subs(expr, *args, **kwargs):
    try:
        subs_ = expr.subs
    except AttributeError:
        return expr
    return subs_(*args, **kwargs)


def simplifier(base=None):
    if base is None:
        return sy.simplify
    elif not base:
        return lambda expr: expr
    elif callable(base):
        return base
    else:
        return sy.simplify


simplify = simplifier()


def recursive(func, is_dict=None, is_seq=None):
    if is_dict is None:
        def is_dict(arg):
            return isinstance(arg, dict)
    if is_seq is None:
        def is_seq(arg):
            try:
                iter(arg)
            except TypeError:
                return False
            return not isinstance(arg, str)

    @ft.wraps(func)
    def wrapper(arg, *args, **kwargs):
        if isinstance(arg, sy.MatrixBase):
            return arg.applyfunc(lambda e: func(e, *args, **kwargs))
        if is_dict(arg):
            return type(arg)(
                (k, wrapper(v, *args, **kwargs))
                for k, v in arg.items())
        elif is_seq(arg):
            return type(arg)([wrapper(v, *args, **kwargs) for v in arg])
        return func(arg, *args, **kwargs)

    return wrapper


def broadcastable(func):
    @ft.wraps(func)
    def wrapper(expr, *args, **kwargs):
        return applyfunc(expr, lambda arg: func(arg, *args, **kwargs))

    return wrapper


def lambdify(args, expr, broadcast=True, **kwargs):
    """
    >>> x, y = sy.symbols('x, y')
    >>> m = sy.Matrix([x, x + y])
    >>> m1 = sy.lambdify([x, y], m, modules='numpy')
    >>> m2 = lambdify([x, y], m, broadcast=True)
    >>> try:
    ...     m1(1, [2, 3])
    ... except ValueError:
    ...     print('failed')
    ...
    failed
    >>> m2(1, [2, 3])
    array([[[1, 1]],
    <BLANKLINE>
           [[3, 4]]])
    """
    kwargs['modules'] = 'numpy'
    if not broadcast or len(args) <= 1:
        return sy.lambdify(args, expr, **kwargs)
    try:
        shape = expr.shape
    except AttributeError:
        return sy.lambdify(args, expr, **kwargs)

    dummy = sy.Dummy()
    f = sy.lambdify(
        args=sy.flatten([dummy, args]),
        expr=expr + sy.ones(*shape) * dummy,
        **kwargs)
    return lambda *x: f(np.zeros(np.broadcast(*x).shape, dtype=int), *x)


def simplify_subs_arg(arg, **kwargs):
    try:
        arg = arg.items()
    except AttributeError:
        pass

    arg = [(k, parse_expr(v) if isinstance(v, str) else v) for k, v in arg]
    arg = [(k, subs(v, arg[i + 1:], **kwargs)) for i, (k, v) in enumerate(arg)]
    return dict(list(dict(arg[::-1]).items())[::-1])


def dict_subs(target, arg, **kwargs):
    if isinstance(target, dict):
        target = target.items()
    if isinstance(arg, dict):
        arg = arg.items()

    return dict((k, subs(v, arg, **kwargs)) for k, v in target)


@broadcastable
def constant_term(expr):
    for term in sy.Add.make_args(sy.expand(expr)):
        try:
            if term.is_Number:
                return term
        except AttributeError:
            return term
    return 0


def _main():
    pass


if __name__ == '__main__':
    _main()
