import functools as ft
import itertools as it
import os
import re
from contextlib import contextmanager

__all__ = [
    'temp_chdir',
    'find',
    'path_depth',
    'require_dir',
]


def _return_fspath(f):
    generator_iterator = type(_ for _ in ())

    @ft.wraps(f)
    def wrapper(*args, **kwargs):
        path = f(*args, **kwargs)
        if isinstance(path, generator_iterator):
            return (os.fspath(p) for p in path)
        else:
            return os.fspath(path)

    return wrapper


def path_depth(path):
    """Return the number of files or directories in the given path.

    Parameters
    ----------
    path : path_like

    Returns
    -------
    out : int

    Examples
    --------
    >>> path_depth('A/B/C')
    3
    >>> path_depth(''), path_depth('A/'), path_depth('A//B')
    (0, 1, 2)
    >>> path_depth('/'), path_depth('/A')
    (1, 2)
    >>> path_depth('.'), path_depth('./A'), path_depth('A/.')
    (1, 2, 2)
    >>> path_depth('..'), path_depth('../A'), path_depth('A/..')
    (1, 2, 2)
    """
    if not path:
        return 0
    for i in it.count(1):
        dirname, basename = os.path.split(path)
        if not basename:
            path, dirname = dirname, os.path.dirname(dirname)
        if not dirname or dirname == path:
            return i
        path = dirname


@contextmanager
@_return_fspath
def temp_chdir(path):
    """Change the working directory.

    Parameters
    ----------
    path : path_like

    Returns
    -------
    path : str
    """
    backup = os.path.abspath(os.curdir)
    try:
        os.chdir(path)
        yield path
    finally:
        os.chdir(backup)


@_return_fspath
def require_dir(dirpath, basename=None):
    """Make a directory and return the given path as is.

    Parameters
    ----------
    dirpath : path_like
    basename : path_like

    Returns
    -------
    path : str
    """
    os.makedirs(dirpath, exist_ok=True)
    if not os.path.isdir(dirpath):
        raise NotADirectoryError(dirpath)
    return os.path.join(dirpath, basename or '')


def _find(root, **kwargs):
    for parent, _, filenames in os.walk(root, topdown=True, **kwargs):
        yield parent
        yield from (
            os.path.join(parent, filename)
            for filename in filenames
        )


@_return_fspath
def find(root, types=None, name=None, path=None, regex=None,
         maxdepth=None, onerror=None, followlinks=False):
    """Find file paths.

    Parameters
    ----------
    root : path_like
    types : str, optional
        'f':
            Find regular files.
        'd':
            Find directories.
        'l':
            Find symbolic links.
    name : str or callable, optional
        File name of the target or a function which returns true if the target
        file name is given.
    path : str or callable, optional
        Path of the target or a function which returns true if the target path
        is given.
    regex : str, optional
        Regex pattern which matchs target paths.
    maxdepth : int, optional
        Maximum depth from the root path.
    onerror : callable, optional
    followlinks : bool, optional

    Yields
    ------
    path : str
    """
    if isinstance(name, str):
        name = name.__eq__
    if isinstance(path, str):
        path = path.__eq__
    if isinstance(regex, str):
        regex = re.compile(regex).search

    def depth(p):
        return path_depth(p) - path_depth(root)

    for path_ in _find(root, onerror=onerror, followlinks=followlinks):
        if types:
            if 'f' not in types and os.path.isfile(path_):
                continue
            if 'd' not in types and os.path.isdir(path_):
                continue
            if 'l' not in types and os.path.islink(path_):
                continue
        if callable(name) and not name(os.path.basename(path_)):
            continue
        if callable(path) and not path(path_):
            continue
        if callable(regex) and not regex(path_):
            continue
        if maxdepth is not None and depth(path_) > maxdepth:
            break
        yield path_
