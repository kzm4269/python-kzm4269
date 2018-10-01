import re

__all__ = [
    'parse_printf_style_format',
    'camel_to_snake',
]


def parse_printf_style_format(fmt):
    """Parse the printf-style format string.

    Parameters
    ----------
    fmt : str
        Printf-style format string.

    Returns
    -------
    out : int or list
        Number of arguments or a list of argument names for the format string.

    Examples
    --------
    >>> parse_printf_style_format('%d')
    1
    >>> parse_printf_style_format('%d %s')
    2
    >>> parse_printf_style_format('%(foo)d %(bar)s')
    ['foo', 'bar']
    """
    if not isinstance(fmt, (bytes, str)):
        raise TypeError('got ' + type(fmt).__name__)

    try:
        fmt % ()
    except TypeError as e:
        if e.args[0] == 'not enough arguments for format string':
            values = ()
        elif e.args[0] == 'format requires a mapping':
            values = {}
        else:
            raise
    else:
        return None

    if isinstance(values, tuple):
        while True:
            try:
                fmt % values
            except TypeError as e:
                if e.args[0] == 'not enough arguments for format string':
                    values += (0,)
                else:
                    raise ValueError('invalid format: ' + repr(fmt))
            else:
                return len(values)
    elif isinstance(values, dict):
        while True:
            try:
                fmt % values
            except TypeError as e:
                if e.args[0] == 'not enough arguments for format string':
                    raise ValueError('invalid format: ' + repr(fmt))
                else:
                    raise
            except KeyError as e:
                values[e.args[0]] = 0
            else:
                return list(values.keys())
    else:
        assert False


def camel_to_snake(text):
    """
    Examples
    --------
    >>> camel_to_snake('GeneralDynamicsF16FightingFalcon')
    'general_dynamics_f16_fighting_falcon'
    >>> camel_to_snake('HTTPServer')
    'http_server'
    >>> camel_to_snake('')
    """
    text = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', text)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text).lower()
