"""Utilities for PyYAML"""
import json
import os
import sys

import dateutil.parser
import yaml


class _SafeDumper(yaml.SafeDumper):
    pass


class _SafeLoader(yaml.SafeLoader):
    pass


def represent_singleline_text(dumper, data):
    if len(data.splitlines()) >= 2:
        style = '"'
    elif yaml.dump(dict(_=data), Dumper=_SafeDumper)[4] in ('"', "'"):
        style = "'"
    else:
        style = None
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style=style)


def represent_multiline_text(dumper, data):
    style = '|' if len(data.splitlines()) >= 2 else None
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style=style)


def represent_ordereddict(dumper, data):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())


def represent_isodatetime(dumper, data):
    value = data.isoformat(sep='T')
    return dumper.represent_scalar('tag:yaml.org,2002:timestamp', value)


def represent_path(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', os.fspath(data))


def represent_function(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', repr(data))


def represent_ndarray(dumper, data):
    return dumper.represent_data(data.tolist())


def construct_isodatetime(loader, node):
    value = loader.construct_scalar(node)
    return dateutil.parser.parse(value)


def dump_singleline(data, stream=None, **kwargs):
    class Dumper(kwargs.pop('Dumper', yaml.Dumper)):
        pass

    Dumper.add_representer(str, represent_singleline_text)
    kwargs.update(
        width=sys.maxsize,
        default_flow_style=True,
        explicit_start=False)
    text = yaml.dump(dict(_=data), stream=stream, Dumper=Dumper, **kwargs)
    return text.splitlines()[0][4:-1]


def pprint(data, default=None):
    class _Dumper(yaml.Dumper):
        pass

    _Dumper.add_representer(dict, represent_ordereddict)

    if default is not None:
        data = json.loads(json.dumps(
            data,
            default=default,
            ensure_ascii=False))
    print(yaml.dump(data, width=10000, allow_unicode=True, Dumper=_Dumper))
