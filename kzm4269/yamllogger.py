"""Utilities for logging with PyYAML"""
import abc
import copy
import logging
import textwrap
from datetime import datetime

from dateutil.tz import tzlocal
import yaml

from . import yaml_utils, logging_utils


class YamlFormatterBase(logging.Formatter, metaclass=abc.ABCMeta):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self._required_fields = list(logging_utils.fmt_fields(fmt, style))

    def format(self, record):
        super().format(record)
        record_copy = copy.copy(record)
        for field in self._required_fields:
            setattr(record_copy, field, self.format_field(record, field))
        return self.reformat_message(record, self.formatMessage(record_copy))

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=tzlocal())
        if datefmt:
            return dt.strftime(datefmt)
        return dt

    @abc.abstractmethod
    def format_field(self, record, field):
        pass

    @abc.abstractmethod
    def reformat_message(self, record, message):
        return message


class YamlFormatter(YamlFormatterBase):
    Dumper = yaml.Dumper
    Loader = yaml.Loader

    def __init__(self, fmt=None, datefmt=None, style='%', **kwargs):
        self.Dumper = kwargs.pop('Dumper', self.Dumper)
        self.Loader = kwargs.pop('Loader', self.Loader)
        super().__init__(fmt=fmt, datefmt=datefmt, style=style, **kwargs)

    def format_field(self, record, field):
        return yaml_utils.dump_singleline(
            data=getattr(record, field),
            Dumper=self.Dumper,
            allow_unicode=True)

    def reformat_message(self, record, message):
        return self.dump_yaml(yaml.load(message, Loader=self.Loader))

    def dump_yaml(self, data):
        class Dumper(self.Dumper):
            pass

        Dumper.add_representer(str, yaml_utils.represent_multiline_text)
        return yaml.dump(
            data=data,
            Dumper=Dumper,
            allow_unicode=True,
            explicit_start=True)


class SinglelineYamlFormatter(YamlFormatter):
    def dump_yaml(self, data):
        return '--- ' + yaml_utils.dump_singleline(
            data=data,
            Dumper=self.Dumper,
            allow_unicode=True)


class NonstrictYamlFormatter(YamlFormatter):
    def format_field(self, record, field):
        text = super().format_field(record, field)
        if text[0] == '!' or text[0] == text[-1] == "'":
            return getattr(record, field)
        return text

    def reformat_message(self, record, message):
        if record.exc_text:
            message += '\n' + textwrap.indent(record.exc_text, ' ' * 2)
        return message

    def dump_yaml(self, data):
        assert False


def _main():
    import os
    from pathlib import PurePath

    logging_utils.extend_record_factory(
        cwd=os.getcwd,
        levelchr=lambda r: r.levelname[0],
    )

    class Dumper(yaml.Dumper):
        pass

    class Loader(yaml.Loader):
        pass

    Dumper.add_representer(
        data_type=dict,
        representer=yaml_utils.represent_ordereddict)
    Dumper.add_multi_representer(
        data_type=PurePath,
        representer=yaml_utils.represent_path)
    Dumper.add_representer(
        data_type=datetime,
        representer=yaml_utils.represent_isodatetime)
    Loader.add_constructor(
        tag='tag:yaml.org,2002:timestamp',
        constructor=yaml_utils.construct_isodatetime)

    multiline = YamlFormatter(
        fmt=textwrap.dedent(r"""
            timestamp: %(asctime)s
            logger: %(name)s
            level: %(levelname)s
            source:
                file: %(pathname)s
                line: %(lineno)s
                function: %(funcName)s
                thread: {id: %(thread)s, name: %(threadName)s}
                process: {id: %(process)s, name: %(processName)s}
                workdir: %(cwd)s
            message: %(msg)s
            exception: %(exc_text)s
        """).strip(),
        Dumper=Dumper, Loader=Loader)
    singleline = SinglelineYamlFormatter(
        fmt=textwrap.dedent(r"""
            timestamp: %(asctime)s
            logger: %(name)s
            level: %(levelname)s
            message: %(msg)s
            exception: %(exc_text)s
        """).strip(),
        Dumper=Dumper, Loader=Loader)
    nonstrict = NonstrictYamlFormatter(
        '[%(levelchr)s %(asctime)s %(name)s] %(msg)s',
        datefmt='%H:%M:%S.%f',
        Dumper=Dumper, Loader=Loader)

    logger = logging.getLogger('my_logger')

    handler = logging.StreamHandler()
    logger.addHandler(handler)
    handler.setFormatter(multiline)

    handler = logging.StreamHandler()
    logger.addHandler(handler)
    handler.setFormatter(singleline)

    handler = logging.StreamHandler()
    logger.addHandler(handler)
    handler.setFormatter(nonstrict)

    logger.setLevel(logging.DEBUG)

    logger.debug('hello')
    logger.debug('hello\nworld')
    logger.debug(PurePath('a/b/c'))
    try:
        raise Exception('error!')
    except Exception as e:
        logger.exception(str(e))
    logger.debug({
        'message': 'hello\nworld', 'goo\tgle': 'https://google.com',
        'foo': {'bar': 'baz'},
    })
    logger.warning([1, 2, 3])
    logger.info('hello', extra={'google': 'https://google.com'})
    logger.debug(str(logging))


if __name__ == '__main__':
    _main()
