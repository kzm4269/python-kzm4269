import importlib.util
import pkgutil
import re
import sys
import traceback
import warnings
from collections import namedtuple
from urllib.request import urlopen

__all__ = [
    'import_module_from_file_location',
    'import_stdlib',
    'import_submodules',
    'reload_submodules',
    'stdlib_modules',
]


def import_module_from_file_location(name, location=None):
    """Import a module based on a file location."""
    spec = importlib.util.spec_from_file_location(name, location=location)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def import_submodules(
        root, onerror=None, exclude=(), verbose=False, exclude_private=True):
    """Import submodules recursively."""
    root_path = getattr(root, '__path__', None) or ()
    root_name = getattr(root, '__name__', None) or ''

    if callable(exclude):
        exclude = [exclude]
    exclude = [e if callable(e) else e.__eq__ for e in exclude]

    if exclude_private:
        exclude = [re.compile(r'.*\._').match] + exclude

    if verbose:
        submodules = pkgutil.iter_modules(root_path, root_name + '.')
        print([fn for _, fn, _ in submodules], file=sys.stderr)

    for _, fullname, _ in pkgutil.iter_modules(root_path, root_name + '.'):
        if any(e(fullname) for e in exclude):
            if verbose:
                print('# reload({})'.format(fullname), file=sys.stderr)
        else:
            if verbose:
                print('reload({})'.format(fullname), file=sys.stderr)

            try:
                with warnings.catch_warnings():
                    if not verbose:
                        warnings.filterwarnings('ignore')
                    mod = importlib.import_module(fullname)
            except ImportError:
                if onerror is not None:
                    onerror(fullname)
                elif verbose:
                    traceback.print_exc()
            except Exception:
                if onerror is not None:
                    onerror(fullname)
                else:
                    raise
            else:
                import_submodules(
                    root=mod,
                    onerror=onerror,
                    exclude=exclude,
                    verbose=verbose,
                )


def reload_submodules(
        root, onerror=None, exclude=(), verbose=False, exclude_private=False):
    """Reload submodules recursively."""
    root_name = getattr(root, '__name__', '')

    if callable(exclude):
        exclude = [exclude]
    exclude = [e if callable(e) else e.__eq__ for e in exclude]

    if exclude_private:
        exclude = [re.compile(r'.*\._').match] + exclude

    for fullname, submod in sorted(sys.modules.items(), key=lambda kv: kv[0]):
        if fullname == root_name or fullname.startswith(root_name + '.'):
            if any(e(fullname) for e in exclude):
                if verbose:
                    print('# reload({})'.format(fullname), file=sys.stderr)
            else:
                if verbose:
                    print('reload({})'.format(fullname), file=sys.stderr)

                try:
                    with warnings.catch_warnings():
                        if not verbose:
                            warnings.filterwarnings('ignore')
                        importlib.reload(submod)
                except ImportError:
                    if onerror is not None:
                        onerror(fullname)
                    elif verbose:
                        traceback.print_exc()
                except Exception:
                    if onerror is not None:
                        onerror(fullname)
                    else:
                        raise


def stdlib_modules(version=None):
    """List modules of the Python standard library.

    Parameters
    ----------
    version

    Returns
    -------
    List of dictionaries.

    Examples
    --------
    >>> libs = stdlib_modules('2.7')
    >>> for lib in libs[:5]:
    ...     print(lib['name'], ':', lib['description'])
    __builtin__ : The module that provides the built-in namespace.
    __future__ : Future statement definitions
    __main__ : The environment where the top-level script is run.
    _winreg : Routines and objects for manipulating the Windows registry.
    abc : Abstract base classes according to PEP 3119.
    """

    from lxml import html

    if not version:
        version = '%d.%d' % sys.version_info[:2]
    url = 'https://docs.python.org/{}/py-modindex.html'.format(version)
    with urlopen(url) as fp:
        content = html.fromstring(fp.read().decode())

    td_lists = (
        tr.xpath('.//td')
        for tr in content.xpath('//tr[.//code[@class="xref"]]')
    )

    return [
        {
            'name': tds[1].xpath('.//code/text()')[0],
            'note': [
                text.rstrip(':')
                for text in tds[2].xpath('./strong/text()')
            ],
            'os': [
                os_name
                for text in tds[1].xpath('./em/text()')
                for os_name in text[1:-1].split(', ')
            ],
            'description': ''.join(
                text.replace('\n', '')
                for text in tds[2].xpath('./em/text()')
            ),
        }
        for tds in td_lists
    ]


def import_stdlib(as_dict=False):
    """Import all modules of the Python standard library.

    Parameters
    ----------
    as_dict

    Returns
    -------
    Namedtuple or dictionary of the standard modules.

    Examples
    --------
    >>> import sys
    >>> stdlib = import_stdlib()
    >>> stdlib.sys is sys
    True
    >>> stdlib = import_stdlib(as_dict=True)
    >>> stdlib['sys'] is sys
    True
    """

    def _import_stdlib():
        for info in stdlib_modules():
            name = info['name']
            if not name.startswith('_'):
                try:
                    module = __import__(name)
                except ImportError:
                    pass
                else:
                    if '.' not in name:
                        yield name, module

    attr_dict = dict(_import_stdlib())
    if as_dict:
        return attr_dict

    return namedtuple('stdlib', attr_dict)(**attr_dict)
