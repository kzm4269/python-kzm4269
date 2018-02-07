"""Types for argparse.ArgumentParser"""
import os
from argparse import ArgumentTypeError

from kzm4269.exc_utils import wrap_exc


def path_type(
        types=None, modes=None, exists=None, makedirs=None,
        as_abs=None, as_real=None, as_rel=None):
    """Return a callable object parsing a string as a file path.

    Parameters
    ----------
    types : str or sequence of str, optional
        'f':
            Parser accepts only a path to a regular file.
        'd':
            Parser accepts only a path to a directory.
    modes : str or sequence of str, optional
        'r':
            Parser accepts only a path to a readable file.
        'w':
            Parser accepts only a path to a writable file.
        'x':
            Parser accepts only a path to a executable file.
    exists : bool, optional
        True:
            Parser accepts only a path to a existing file.
        False:
            Parser accepts only a path of a non-existing file.
    makedirs : bool, optional
        If true is given, parser creates all directories in the given path.
    as_abs : bool, optional
        If true is given, parser converts the given path to the absolute path.
    as_real : bool, optional
        If true is given, parser converts the given path to the real path.
    as_rel : bool or os.PathLike, optional
        If true is given, parser converts to the relative path from the
        current directory.  If a path is given, parser converts to the
        relative path from the path.

    Returns
    -------
    parse : callable

    Examples
    --------
    >>> import tempfile, pathlib
    >>> tempd = tempfile.TemporaryDirectory()
    >>> os.chdir(tempd.__enter__())
    >>> path_type('d')('foo')
    Traceback (most recent call last):
    ...
    argparse.ArgumentTypeError: No such a file or directory: foo
    >>> path_type('d', makedirs=True)('foo')
    'foo'
    >>> path_type('f')('foo')
    Traceback (most recent call last):
    ...
    argparse.ArgumentTypeError: Is a directory: foo
    >>> path_type('f')('foo/bar')
    Traceback (most recent call last):
    ...
    argparse.ArgumentTypeError: No such a file or directory: foo/bar
    >>> pathlib.Path('foo/bar').touch()
    >>> path_type('f')('foo/bar')
    'foo/bar'
    >>> path_type('f', as_rel='foo')('foo/bar')
    'bar'
    >>> os.chdir(os.path.dirname(tempd.name))
    """
    if types or modes:
        exists = True

    @wrap_exc(ArgumentTypeError, OSError)
    @wrap_exc(RuntimeError, TypeError)
    def parse(path):
        if makedirs:
            os.makedirs(
                path if types and 'd' in types else os.path.dirname(path),
                exist_ok=True)

        if exists is not None:
            if exists and not os.path.exists(path):
                raise FileNotFoundError(f'No such a file or directory: {path}')
            elif not exists and os.path.exists(path):
                raise FileExistsError(f'Already exists: {path}')

        if (not types
                or 'f' in types and os.path.isfile(path)
                or 'd' in types and os.path.isdir(path)):
            pass
        elif 'f' in types and 'd' in types:
            raise FileNotFoundError(f'No such a file or directory: {path}')
        elif 'f' in types:
            raise IsADirectoryError(f'Is a directory: {path}')
        elif 'd' in types:
            raise NotADirectoryError(f'Not a directory: {path}')
        else:
            assert False

        if (not modes
                or 'r' in modes and os.access(path, os.R_OK)
                or 'w' in modes and os.access(path, os.W_OK)
                or 'x' in modes and os.access(path, os.X_OK)):
            pass
        elif 'r' in types:
            raise PermissionError(f'Permission denied: cannot read: {path}')
        elif 'w' in types:
            raise PermissionError(f'Permission denied: cannot write: {path}')
        elif 'x' in types:
            raise PermissionError(f'Permission denied: cannot execute: {path}')

        if as_abs:
            path = os.path.abspath(path)
        if as_real:
            path = os.path.realpath(path)
        if as_rel:
            try:
                start = os.fspath(as_rel)
            except TypeError:
                start = os.curdir
            path = os.path.relpath(path, start=start)
        return path

    return parse


def _main():
    import doctest
    doctest.testmod()


if __name__ == '__main__':
    _main()
