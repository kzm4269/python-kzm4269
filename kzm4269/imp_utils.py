"""Utilities for module importing"""
import importlib.util
import itertools as it
import pkgutil
import re
import sys
import traceback
import warnings
from collections import namedtuple


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
                print(f'# import {fullname}', file=sys.stderr)
        else:
            if verbose:
                print(f'import {fullname}', file=sys.stderr)

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
                    verbose=verbose)


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
                    print(f'# reload({fullname})', file=sys.stderr)
            else:
                if verbose:
                    print(f'reload({fullname})', file=sys.stderr)

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
    import lxml.html
    import requests

    if not version:
        version = '%d.%d' % sys.version_info[:2]
    url = 'https://docs.python.org/{}/py-modindex.html'.format(version)
    content = lxml.html.fromstring(requests.get(url).content.decode())
    for tr in content.xpath('//tr[.//code[@class="xref"]]'):
        td = tr.xpath('.//td')
        yield dict(
            name=td[1].xpath('.//code')[0].text,
            os=list(set(it.chain(*(
                em.text[1:-1].split(', ')
                for em in td[1].xpath('./em'))))),
            note=[e.text[:-1] for e in td[2].xpath('./strong')],
            description=''.join(
                (em.text or '').replace('\n', ' ')
                for em in td[2].xpath('./em')))


def import_stdlib(version=None, as_dict=False):
    def _import_stdlib():
        for info in stdlib_modules(version):
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

    return namedtuple('stdlib', attr_dict.keys())(**attr_dict)
