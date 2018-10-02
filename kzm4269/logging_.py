import logging
import string
from contextlib import contextmanager
from inspect import signature

from kzm4269.string_ import parse_printf_style_format

__all__ = [
    'extend_record_factory',
    'fmt_fields',
    'temp_config',
]


def extend_record_factory(**attr_dict):
    """
    Examples
    --------
    >>> import logging
    >>> import sys
    >>> from datetime import datetime
    >>> extend_record_factory(
    ...     isotime=lambda r: (
    ...         datetime
    ...             .fromtimestamp(r.created)
    ...             .astimezone()
    ...             .isoformat()
    ...     ),
    ...     levelchar=lambda r: (
    ...         'C' if r.levelno >= logging.CRITICAL else
    ...         'F' if r.levelno >= logging.FATAL else
    ...         'E' if r.levelno >= logging.ERROR else
    ...         'W' if r.levelno >= logging.WARNING else
    ...         'I' if r.levelno >= logging.INFO else
    ...         'D' if r.levelno >= logging.DEBUG else
    ...         'N'  # None
    ...     ),
    ...     levelcolor=lambda r: (
    ...         1 if r.levelno >= logging.ERROR else
    ...         3 if r.levelno >= logging.WARNING else
    ...         2 if r.levelno >= logging.INFO else
    ...         5 if r.levelno >= logging.DEBUG else
    ...         0
    ...     ),
    ... )
    >>> logging.basicConfig(
    ...     format='%(levelchar)s %(message)s',
    ...     level=0,
    ...     stream=sys.stdout,
    ... )
    >>> logging.info('message')
    I message
    """

    class LogRecord(logging.getLogRecordFactory()):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            for k, v in attr_dict.items():
                setattr(self, k, v(self) if signature(v).parameters else v())

    logging.setLogRecordFactory(LogRecord)


def fmt_fields(fmt, style='%'):
    """
    Examples
    --------
    >>> fmt_fields('%(asctime)s %(message)s', style='%')
    ['asctime', 'message']
    >>> fmt_fields('{asctime} {message!r}', style='{')
    ['asctime', 'message']
    >>> fmt_fields('$asctime ${message}', style='$')
    ['asctime', 'message']
    """
    if style == '%':
        return parse_printf_style_format(fmt)
    elif style == '{':
        return [
            p[1]
            for p in string.Formatter().parse(fmt)
            if p[1]
        ]
    elif style == '$':
        return [
            m.group('named') or m.group('braced')
            for m in string.Template(fmt).pattern.finditer(fmt)
            if m.group('named') or m.group('braced')
        ]
    else:
        raise ValueError('invalid style ' + repr(style))


@contextmanager
def temp_config(handlers=(), filters=(), level=None, logger=None):
    if not logger:
        logger = logging.getLogger()

    original_level = logger.level
    if level is not None:
        logger.setLevel(level)

    handlers = list(handlers)
    for h in handlers:
        logger.addHandler(h)

    filters = list(filters)
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
