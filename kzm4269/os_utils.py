"""Utilities for os and os.path"""
import contextlib
import functools as ft
import itertools as it
import os
import re


def is_path(path):
    try:
        os.fspath(path)
    except TypeError:
        return False
    return True


def return_fspath(f):
    @ft.wraps(f)
    def wrapper(*args, **kwargs):
        return os.fspath(f(*args, **kwargs))

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


@contextlib.contextmanager
def temp_cd(path):
    """Change the working directory .

    Parameters
    ----------
    path : path_like

    Returns
    -------
    path : path_like

    Examples
    --------
    >>> with temp_cd('/home') as path_to_home:
    ...    pass  # do something in /home
    """
    backup = os.path.abspath(os.curdir)
    try:
        os.chdir(path)
        yield path
    finally:
        os.chdir(backup)


def _find(root, **kwargs):
    for dpath, _, filenames in os.walk(root, topdown=True, **kwargs):
        yield dpath
        yield from (os.path.join(dpath, fn) for fn in filenames)


def find(
        root, types=None, name=None, path=None, regex=None,
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
    path : path_like
    """
    if isinstance(name, str):
        name = name.__eq__
    if isinstance(path, str):
        path = path.__eq__
    if isinstance(regex, str):
        regex = re.compile(regex).search

    def depth(p):
        return path_depth(p) - path_depth(root)

    for _path in _find(root, onerror=onerror, followlinks=followlinks):
        if types:
            if 'f' not in types and os.path.isfile(_path):
                continue
            if 'd' not in types and os.path.isdir(_path):
                continue
            if 'l' not in types and os.path.islink(_path):
                continue
        if callable(name) and not name(os.path.basename(_path)):
            continue
        if callable(path) and not path(_path):
            continue
        if callable(regex) and not regex(_path):
            continue
        if maxdepth is not None and depth(_path) > maxdepth:
            break
        yield _path


def require_dir(path):
    """Make a directory and return the given path as is.

    Parameters
    ----------
    path : path_like

    Returns
    -------
    path : path_like
    """
    os.makedirs(path, exist_ok=True)
    if not os.path.isdir(path):
        raise OSError(f'mkdir failed: {path}')
    return path


def require_parents(path):
    """Make the parent directory of the given path and return the path as is.

    Parameters
    ----------
    path : path_like

    Returns
    -------
    path : path_like
    """
    require_dir(os.path.dirname(path))
    return path


def _main():
    for path in find('/tmp', re_name=r'^tmp'):
        print(path)


if __name__ == '__main__':
    _main()
