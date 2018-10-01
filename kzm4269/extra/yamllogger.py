import abc
import copy
import logging
import textwrap
from datetime import datetime

import yaml

from kzm4269.extra import yaml_
from kzm4269.logging_ import fmt_fields


class YamlFormatterBase(logging.Formatter, metaclass=abc.ABCMeta):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt=fmt, datefmt=datefmt, style=style)
        self._required_fields = list(fmt_fields(fmt, style))

    def format(self, record):
        super().format(record)
        record_copy = copy.copy(record)
        for field in self._required_fields:
            setattr(record_copy, field, self.format_field(record, field))
        return self.reformat_message(record, self.formatMessage(record_copy))

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created).astimezone()
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
        return yaml_.dump_singleline(
            data=getattr(record, field),
            Dumper=self.Dumper,
            allow_unicode=True)

    def reformat_message(self, record, message):
        return self.dump_yaml(yaml.load(message, Loader=self.Loader))

    def dump_yaml(self, data):
        class Dumper(self.Dumper):
            pass

        Dumper.add_representer(str, yaml_.represent_multiline_text)
        return yaml.dump(
            data=data,
            Dumper=Dumper,
            allow_unicode=True,
            explicit_start=True)


class SinglelineYamlFormatter(YamlFormatter):
    def dump_yaml(self, data):
        return '--- ' + yaml_.dump_singleline(
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
