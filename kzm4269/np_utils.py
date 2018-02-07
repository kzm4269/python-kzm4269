"""Utilities for NumPy"""
import functools as ft
import itertools as it

import numpy as np


def uniform_on_sphere(dim=2, size=None):
    """Generate uniformly distributed points on the surface of the N-dim unit
    sphere.

    Parameters
    ----------
    dim : int, optional
    size : int or tuple of ints, optional

    Returns
    -------
    out : ndarray of floats

    Examples
    --------
    >>> np.random.seed(0)
    >>> p = uniform_on_sphere()
    >>> p
    array([ 0.97522402,  0.22121958])
    >>> np.linalg.norm(p)
    1.0
    >>> p = uniform_on_sphere(dim=3)
    >>> p
    array([ 0.31809241,  0.72829615,  0.60696123])
    >>> np.linalg.norm(p)
    0.99999999999999989
    >>> p = uniform_on_sphere(dim=2, size=3)
    >>> p
    array([[-0.71701063,  0.69706223],
           [-0.82617461, -0.56341416],
           [ 0.9436187 ,  0.33103435]])
    >>> np.linalg.norm(p, axis=-1)
    array([ 1.,  1.,  1.])
    """
    if isinstance(size, int):
        size = size,
    elif not size:
        size = ()
    points = np.random.randn(*size, dim)
    r2 = np.einsum('...i, ...i', points, points)
    points /= np.sqrt(r2)[..., None]
    return points


def random_orthogonal(dim):
    """Return a random orthogonal matrix.

    Parameters
    ----------
    dim : int

    Returns
    -------
    out : ndarray of floats
        Rundom orthogonal matrix with the given dimension.

    Examples
    --------
    >>> np.random.seed(0)
    >>> m = random_orthogonal(3)
    >>> m
    array([[-0.58684003,  0.48538278, -0.6480913 ],
           [-0.74546871, -0.63631164,  0.19845377],
           [-0.316062  ,  0.5995924 ,  0.73525082]])
    >>> m.T @ m - np.eye(3)
    array([[ -4.44089210e-16,   3.86008230e-17,  -9.37643276e-17],
           [  3.86008230e-17,  -2.22044605e-16,   1.25667846e-17],
           [ -9.37643276e-17,   1.25667846e-17,   0.00000000e+00]])
    """
    return np.linalg.qr(np.random.randn(dim, dim))[0]


def random_symetric(eigvals):
    """Return a random real symmetric matrix has specified eigenvalues.

    Parameters
    ----------
    eigvals : (N,) array_like

    Returns
    -------
    out : (N, N) ndarray

    Examples
    --------
    >>> np.random.seed(0)
    >>> m = random_symetric([1, 2, 3])
    >>> m
    array([[ 2.07564112, -0.56608704, -0.66198749],
           [-0.56608704,  1.4836603 , -0.08970104],
           [-0.66198749, -0.08970104,  2.44069858]])
    >>> np.linalg.eigvals(m)
    array([ 1.,  3.,  2.])
    """
    eigvals = np.array(eigvals, copy=False, ndmin=1)
    u = random_orthogonal(eigvals.size)
    return (u * eigvals) @ u.T


def batches(*args, batch_size=None):
    """
    Examples
    --------
    >>> args = np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8, 9, 10]), 11
    >>> for b in batches(*args, batch_size=2):
    ...     print(b)
    (array([0, 1]), array([5, 6]), 11)
    (array([2, 3]), array([7, 8]), 11)
    (array([4]), array([9]), 11)

    >>> for b in batches(1, [2, 3, 4], [5, 6]):
    ...     print(b)
    (1, [2, 3], [5, 6])

    >>> list(batches(1, 2, 3))
    [(1, 2, 3)]
    """
    args = tuple(args)
    if all(np.isscalar(arg) for arg in args):
        return iter([args])
    total = min(len(arg) for arg in args if not np.isscalar(arg))
    if batch_size is None:
        batch_size = total
    return zip(*(
        it.repeat(arg) if np.isscalar(arg)
        else [arg[:total][i:i + batch_size]
              for i in range(0, len(arg), batch_size)]
        for arg in args))


def slices(ndims, replace=None, default=None):
    """
    Examples
    --------
    >>> slices(2)
    (slice(None, None, None), slice(None, None, None))
    >>> slices(2, {0: slice(2, 5)})
    (slice(2, 5, None), slice(None, None, None))
    >>> slices(2, {0: slice(2, 5)}, default=1)
    (slice(2, 5, None), 1)
    """
    if default is None:
        default = np.s_[:]
    ret = [default] * ndims
    if replace:
        for axis, s in replace.items():
            ret[axis] = s
    return tuple(ret)


def rdiff(y, x=None, axis=-1, inverse=False):
    """
    Examples
    --------
    >>> rdiff([3])
    array([3])
    >>> rdiff([2, 3])
    array([1, 3])
    >>> rdiff([-1, 2, 3])
    array([-2,  1,  3])
    >>> rdiff([-2, 1, 3], inverse=True)
    array([-1,  2,  3])
    """
    y = np.copy(y)
    if x is None:
        x = 1
    x = np.asarray(x)
    if x.ndim > 0:
        x = x.reshape(slices(y.ndim, {axis: -1}, default=1))
        dx = np.diff(x, axis=axis)
    else:
        dx = x

    for i in range(y.shape[axis], 0, -1):
        s0 = slices(y.ndim, {axis: np.s_[:i - 1]})
        s1 = slices(y.ndim, {axis: np.s_[1:i]})

        if dx.ndim > 0:
            dx0 = dx[s0]
        else:
            dx0 = dx

        if inverse:
            y[s0] = y[s1] - y[s0] * dx0
        else:
            y[s0] = (y[s1] - y[s0]) / dx0

    return y


irdiff = ft.partial(rdiff, inverse=True)
