"""Utilities for logging"""
import contextlib
import logging
import string
from inspect import signature

from kzm4269 import str_utils


@contextlib.contextmanager
def temp_config(handlers=(), filters=(), level=None, logger=None):
    if not logger:
        logger = logging.getLogger()

    original_level = logger.level
    if level is not None:
        logger.setLevel(level)

    handlers = tuple(handlers)
    for h in handlers:
        logger.addHandler(h)

    filters = tuple(filters)
    for f in filters:
        logger.addFilter(f)

    try:
        yield logger
    finally:
        logger.setLevel(original_level)
        for h in handlers:
            logger.removeHandler(h)
        for f in filters:
            logger.removeFilter(f)


def extend_record_factory(**attr_dict):
    class LogRecord(logging.getLogRecordFactory()):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for k, v in attr_dict.items():
                setattr(self, k, v(self) if signature(v).parameters else v())

    logging.setLogRecordFactory(LogRecord)


def fmt_fields(fmt, style='%'):
    if style == '%':
        return str_utils.parse_printf_style_format(fmt)
    elif style == '{':
        return [p[1] for p in string.Formatter().parse(fmt) if p[1]]
    elif style == '$':
        return [m.group('named') or m.group('braced')
                for m in string.Template(fmt).pattern.finditer(fmt)
                if m.group('named') or m.group('braced')]
    else:
        raise ValueError('unknown style ' + repr(style))
