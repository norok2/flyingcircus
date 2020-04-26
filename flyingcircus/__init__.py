#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FlyingCircus - Everything you always wanted to have in Python.*
"""

# Copyright (c) Riccardo Metere <rick@metere.it>

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import datetime  # Basic date and time types
import inspect  # Inspect live objects
import os  # Miscellaneous operating system interfaces
import appdirs  # Determine appropriate platform-specific dirs
# import pkg_resources  # Manage package resource (from setuptools module)
import doctest  # Test interactive Python examples

# ======================================================================
# :: External Imports
# import flyingcircus as fc  # Everything you always wanted to have in Python*
# from flyingcircus import msg, dbg, fmt, fmtm, elapsed, report

# ======================================================================
# :: Version
from flyingcircus._version import __version__

# ======================================================================
# :: Project Details
INFO = {
    'name': 'FlyingCircus',
    'author': 'FlyingCircus developers',
    'contrib': (
        'Riccardo Metere <rick@metere.it>',
    ),
    'copyright': 'Copyright (C) 2014-2018',
    'license': 'GNU General Public License version 3 or later (GPLv3+)',
    'notice':
        """
This program is free software and it comes with ABSOLUTELY NO WARRANTY.
It is covered by the GNU General Public License version 3 (GPLv3+).
You are welcome to redistribute it under its terms and conditions.
        """,
    'version': __version__
}

# ======================================================================
# :: Supported Verbosity Levels
VERB_LVL_NAMES = (
    'none', 'lowest', 'lower', 'low', 'medium', 'high', 'higher', 'highest',
    'warning', 'debug')
VERB_LVL = {k: v for v, k in enumerate(VERB_LVL_NAMES)}
D_VERB_LVL = VERB_LVL['lowest']

# ======================================================================
# :: quick and dirty timing facility
_EVENTS = []

# ======================================================================
# Greetings
MY_GREETINGS = r"""
 _____ _       _              ____ _                    
|  ___| |_   _(_)_ __   __ _ / ___(_)_ __ ___ _   _ ___ 
| |_  | | | | | | '_ \ / _` | |   | | '__/ __| | | / __|
|  _| | | |_| | | | | | (_| | |___| | | | (__| |_| \__ \
|_|   |_|\__, |_|_| |_|\__, |\____|_|_|  \___|\__,_|___/
         |___/         |___/                            
"""


# generated with: figlet 'FlyingCircus' -f standard

# :: Causes the greetings to be printed any time the library is loaded.
# print(MY_GREETINGS)


# ======================================================================
def do_nothing_decorator(*_args, **_kws):
    """
    Callable decorator that does nothing.

    Arguments are catched, but ignored.
    This is very useful to provide proxy for decorators that may not be
    defined.

    Args:
        *_args: Positional arguments.
        **_kws: Keyword arguments.

    Returns:
        result (callable): The unmodified callable.
    """

    def wrapper(f):
        return f

    if len(_args) > 0 and not callable(_args[0]) or len(_kws) > 0:
        return wrapper
    elif len(_args) == 0:
        return wrapper
    else:
        return _args[0]


# ======================================================================
# Numba import
try:
    from numba import jit
except ImportError:
    HAS_JIT = False
    jit = do_nothing_decorator
else:
    HAS_JIT = True


# ======================================================================
def valid_index(
        idx,
        size,
        circular=False):
    """
    Return a valid index for an object of given size.

    Args:
        idx (int): The input index.
        size (int): The size of the object to index.
        circular (bool): Use circular normalization.
            If True, just use a modulo operation.
            Otherwise, indices beyond the edges are set to the edges.

    Returns:
        idx (int): The valid index.

    Examples:
        >>> print([(i, valid_index(i, 3)) for i in range(-4, 4)])
        [(-4, 0), (-3, 0), (-2, 1), (-1, 2), (0, 0), (1, 1), (2, 2), (3, 2)]
        >>> print([(i, valid_index(i, 3, True)) for i in range(-4, 4)])
        [(-4, 2), (-3, 0), (-2, 1), (-1, 2), (0, 0), (1, 1), (2, 2), (3, 0)]
    """
    if circular:
        return idx % size
    elif idx < 0 and idx < -size:
        return 0
    elif idx < 0:
        return idx + size
    elif idx >= size:
        return size - 1
    else:
        return idx


# ======================================================================
def find_all(
        text,
        pattern,
        overlap=False,
        first=0,
        last=-1):
    """
    Find all occurrences of the pattern in the text.

    For dense inputs (pattern is more than ~20% of the text), a looping
    comprehension may be faster.

    Args:
        text (str|bytes|bytearray): The input text.
        pattern (str|bytes|bytearray): The pattern to find.
        overlap (bool): Detect overlapping patterns.
        first (int): The first index.
            The index is forced within boundaries.
        last (int): The last index (included).
            The index is forced within boundaries.

    Yields:
        position (int): The position of the next finding.

    Examples:
        >>> list(find_all('0010120123012340123450123456', '0'))
        [0, 1, 3, 6, 10, 15, 21]
        >>> list(find_all('  1 12 123 1234 12345 123456', '0'))
        []
        >>> list(find_all('  1 12 123 1234 12345 123456', '12'))
        [4, 7, 11, 16, 22]
        >>> list(find_all(b'  1 12 123 1234 12345 123456', b'12'))
        [4, 7, 11, 16, 22]
        >>> list(find_all(bytearray(b'  1 12 123 1234 12345 123456'), b'12'))
        [4, 7, 11, 16, 22]
        >>> list(find_all('0123456789', ''))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> list(find_all('', ''))
        []
        >>> list(find_all('', '0123456789'))
        []
        >>> list(find_all(b'0123456789', b''))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> list(find_all(b'', b''))
        []
        >>> list(find_all(b'', b'0123456789'))
        []
        >>> list(find_all('000000000', '000'))
        [0, 3, 6]
        >>> list(find_all('000000000', '000', True))
        [0, 1, 2, 3, 4, 5, 6]
    """
    n = len(text)
    if n > 0:
        first = valid_index(first, n)
        last = valid_index(last, n)
        offset = 1 if overlap else (len(pattern) or 1)
        i = first
        while True:
            i = text.find(pattern, i)
            if 0 <= i <= last:
                yield i
                i += offset
            else:
                break
    else:
        return


# ======================================================================
def nested_delimiters(
        text,
        l_delim,
        r_delim,
        including=True):
    """
    Find matching delimiters in a sequence.

    The delimiters are matched according to nesting level.

    Args:
        text (str|bytes|bytearray): The input text.
        l_delim (str|bytes|bytearray): The left delimiter.
        r_delim (str|bytes|bytearray): The right delimiter.
        including (bool): Include delimiters.

    yields:
        result (tuple[int]): The matching delimiters.

    Examples:
        >>> s = '{a} {b:{c}}'
        >>> list(nested_delimiters(s, '{', '}'))
        [(0, 3, 0), (7, 10, 1), (4, 11, 0)]
        >>> [s[i:j] for i, j, depth in nested_delimiters(s, '{', '}')]
        ['{a}', '{c}', '{b:{c}}']
        >>> [s[i:j] for i, j, d in nested_delimiters(s, '{', '}') if d == 0]
        ['{a}', '{b:{c}}']
        >>> list(nested_delimiters('{a} {b:{c}', '{', '}'))
        Traceback (most recent call last):
            ...
        ValueError: Found `1` unmatched left token(s) `{` (position: 4).
        >>> list(nested_delimiters('{a}} {b:{c}}', '{', '}'))
        Traceback (most recent call last):
            ...
        ValueError: Found `1` unmatched right token(s) `}` (position: 3).
        >>> list(nested_delimiters(s.encode(), b'{', b'}'))
        [(0, 3, 0), (7, 10, 1), (4, 11, 0)]
        >>> list(nested_delimiters(bytearray(s.encode()), b'{', b'}'))
        [(0, 3, 0), (7, 10, 1), (4, 11, 0)]

    See Also:
        - flyingcircus.nested_pairs()
    """
    l_offset = len(l_delim) if including else 0
    r_offset = len(r_delim) if including else 0
    stack = []

    l_tokens = set(find_all(text, l_delim))
    r_tokens = set(find_all(text, r_delim))
    positions = l_tokens.union(r_tokens)
    for pos in sorted(positions):
        if pos in l_tokens:
            stack.append(pos + 1)
        elif pos in r_tokens:
            if len(stack) > 0:
                prev = stack.pop()
                yield (prev - l_offset, pos + r_offset, len(stack))
            else:
                raise ValueError(
                    'Found `{}` unmatched right token(s) `{}` (position: {}).'
                        .format(len(r_tokens) - len(l_tokens), r_delim, pos))
    if len(stack) > 0:
        raise ValueError(
            'Found `{}` unmatched left token(s) `{}` (position: {}).'
                .format(
                len(l_tokens) - len(r_tokens), l_delim, stack.pop() - 1))


# ======================================================================
def safe_format_map(
        text,
        source):
    """
    Perform safe string formatting from a mapping source.

    If a value is missing from source, this is simply ignored, and no
    `KeyError` is raised.

    Args:
        text (str): Text to format.
        source (Mapping|None): The mapping to use as source.
            If None, uses caller's `vars()`.

    Returns:
        result (str): The formatted text.

    See Also:
        - flyingcircus.fmt()
        - flyingcircus.fmtm()

    Examples:
        >>> text = '{a} {b} {c}'
        >>> safe_format_map(text, dict(a='-A-'))
        '-A- {b} {c}'
        >>> safe_format_map(text, dict(b='-B-'))
        '{a} -B- {c}'
        >>> safe_format_map(text, dict(c='-C-'))
        '{a} {b} -C-'

        >>> source = dict(a=4, c=101, d=dict(x='FOO'), e=[1, 2])
        >>> safe_format_map('{b} {f}', source)
        '{b} {f}'
        >>> safe_format_map('{a} {b}', source)
        '4 {b}'
        >>> safe_format_map('{a} {b} {c:5d}', source)
        '4 {b}   101'
        >>> safe_format_map('{a} {b} {c!s}', source)
        '4 {b} 101'
        >>> safe_format_map('{a} {b} {c!s:>{a}s}', source)
        '4 {b}  101'
        >>> safe_format_map('{a} {b} {c:0{a}d}', source)
        '4 {b} 0101'
        >>> safe_format_map('{a} {b} {d[x]}', source)
        '4 {b} FOO'
        >>> safe_format_map('{a} {b} {e.index}', source)  # doctest:+ELLIPSIS
        '4 {b} <built-in method index of list object at ...>'
        >>> safe_format_map('{a} {b} {f[g]}', source)
        '4 {b} {f[g]}'
        >>> safe_format_map('{a} {b} {f.values}', source)
        '4 {b} {f.values}'
        >>> safe_format_map('{a} {b} {e[0]}', source)
        '4 {b} 1'
        >>> safe_format_map('{{a}} {b}', source)
        '{a} {b}'
        >>> safe_format_map('{{a}} {{b}}', source)
        '{a} {b}'
    """
    stack = []
    for i, j, depth in nested_delimiters(text, '{', '}'):
        if depth == 0:
            try:
                replacing = text[i:j].format_map(source)
            except KeyError:
                pass
            else:
                stack.append((i, j, replacing))
    result = ''
    i, j = len(text), 0
    while len(stack) > 0:
        last_i = i
        i, j, replacing = stack.pop()
        result = replacing + text[j:last_i] + result
    if i > 0:
        result = text[0:i] + result
    return result


# ======================================================================
def fmt(
        text,
        *_args,
        **_kws):
    """
    Perform string formatting using `text.format()`.

    Args:
        text (str): Text to format.
        *_args: Positional arguments for `str.format()`.
        **_kws: Keyword arguments for `str.format()`.

    Returns:
        result (str): The formatted text.

    Examples:
        >>> a, b, c = 1, 2, 3
        >>> dd = dict(a=10, b=20, c=30)
        >>> fmt('{a} + {a} = {b}', a=a, b=b)
        '1 + 1 = 2'
        >>> fmt('{a} + {a} = {b}', **dd)
        '10 + 10 = 20'
        >>> fmt('{} + {} = {}', 2, 2, 4)
        '2 + 2 = 4'
        >>> fmt('{b} + {b} = {}', 4, b=2)
        '2 + 2 = 4'

    See Also:
        - flyingcircus.fmtm()
        - flyingcircus.partial_format()
    """
    return text.format(*_args, **_kws)


# ======================================================================
def fmtm(
        text,
        source=None,
        safe=True):
    """
    Perform string formatting from a mapping source.

    Args:
        text (str): Text to format.
        source (Mapping|None): The mapping to use as source.
            If None, uses caller's `vars()`.
        safe (bool): Apply mapping safely.
            Uses `flyingcircus.partial_format()`.

    Returns:
        result (str): The formatted text.

    Examples:
        >>> a, b, c = 1, 2, 3
        >>> dd = dict(a=10, b=20, c=30)
        >>> fmtm('{a} + {a} = {b}')
        '1 + 1 = 2'
        >>> fmtm('{a} + {a} = {b}', dd)
        '10 + 10 = 20'
        >>> fmtm('{} + {} = {}', 2, 2, 4)  # doctest:+ELLIPSIS
        Traceback (most recent call last):
            ...
        TypeError: fmtm() takes from 1 to 3 positional arguments ...
        >>> fmtm('{b} + {b} = {}', 4)  # doctest:+ELLIPSIS
        Traceback (most recent call last):
            ...
        TypeError: ...

    See Also:
        - flyingcircus.fmt()
        - flyingcircus.partial_format()
    """
    if source is None:
        frame = inspect.currentframe()
        source = frame.f_back.f_locals
    if safe:
        return safe_format_map(text, source)
    else:
        return text.format_map(source)


# ======================================================================
def msg(
        text,
        verb_lvl=D_VERB_LVL,
        verb_threshold=D_VERB_LVL,
        fmtt=True,
        *_args,
        **_kws):
    """
    Display a feedback message to the standard output.

    Args:
        text (str|Any): Message to display or object with `__str__`.
        verb_lvl (int): Current level of verbosity.
        verb_threshold (int): Threshold level of verbosity.
        fmtt (str|bool|None): Format of the message (if `blessed` supported).
            If True, a standard formatting is used.
            If False, empty string or None, no formatting is applied.
            If str, the specified formatting is used.
            This must be in the form of `{t.NAME}` where `NAME` refer to
            a formatting supported by `Terminal()` from `blessed`/`blessings`.
        *_args: Positional arguments for `print()`.
        **_kws: Keyword arguments for `print()`.

    Returns:
        None.

    Examples:
        >>> s = 'Hello World!'
        >>> msg(s)
        Hello World!
        >>> msg(s, VERB_LVL['medium'], VERB_LVL['low'])
        Hello World!
        >>> msg(s, VERB_LVL['low'], VERB_LVL['medium'])  # no output
        >>> msg(s, fmtt='{t.green}')  # if ANSI Terminal, green text
        Hello World!
        >>> msg('   :  a b c', fmtt='{t.red}{}')  # if ANSI Terminal, red text
           :  a b c
        >>> msg(' : a b c', fmtt='cyan')  # if ANSI Terminal, cyan text
         : a b c
    """
    if verb_lvl >= verb_threshold and text is not None:
        # if blessed/blessings is not present, no coloring
        try:
            from blessed import Terminal
        except ImportError:
            try:
                from blessings import Terminal
            except ImportError:
                Terminal = False

        t = Terminal() if callable(Terminal) else None
        if t is not None and fmtt:
            text = str(text)
            if fmtt is True:
                if VERB_LVL['low'] < verb_threshold <= VERB_LVL['medium']:
                    e = t.cyan
                elif VERB_LVL['medium'] < verb_threshold < VERB_LVL['debug']:
                    e = t.magenta
                elif verb_threshold >= VERB_LVL['debug']:
                    e = t.blue
                elif text.startswith('I:'):
                    e = t.green
                elif text.startswith('W:'):
                    e = t.yellow
                elif text.startswith('E:'):
                    e = t.red
                else:
                    e = t.white
                # first non-whitespace word
                txt1 = text.split(None, 1)[0] if len(text.strip()) > 0 else ''
                # initial whitespaces
                n = text.find(txt1)
                txt0 = text[:n]
                # rest
                txt2 = text[n + len(txt1):]
                txt_kws = dict(
                    e1=e + (t.bold if e == t.white else ''),
                    e2=e + (t.bold if e != t.white else ''),
                    t0=txt0, t1=txt1, t2=txt2, n=t.normal)
                text = '{t0}{e1}{t1}{n}{e2}{t2}{n}'.format_map(txt_kws)
            else:
                if 't.' not in fmtt:
                    fmtt = '{{t.{}}}'.format(fmtt)
                if '{}' not in fmtt:
                    fmtt += '{}'
                text = fmtt.format(text, t=t) + t.normal
        print(text, *_args, **_kws)


# ======================================================================
def dbg(obj, fmtt=None):
    """
    Print content of a variable for debug purposes.

    Args:
        obj: The name to be inspected.
        fmt (str): Format of the message (if `blessed` supported).
            If None, a standard formatting is used.

    Returns:
        None.

    Examples:
        >>> my_dict = {'a': 1, 'b': 1}
        >>> dbg(my_dict)
        dbg(my_dict): (('a', 1), ('b', 1))
        >>> dbg(my_dict['a'])
        dbg(my_dict['a']): 1
    """
    outer_frame = inspect.getouterframes(inspect.currentframe())[1]
    name_str = outer_frame[4][0][:-1]
    msg(name_str, fmtt=fmtt, end=': ')
    if isinstance(obj, dict):
        obj = tuple(sorted(obj.items()))
    text = repr(obj)
    msg(text, fmtt='normal')


# ======================================================================
def elapsed(
        name=None,
        time_point=None,
        events=_EVENTS):
    """
    Append a named event point to the events list.

    Args:
        name (str): The name of the event point.
        time_point (float): The time in seconds since the epoch.
        events (list[(str,datetime.datetime)]): A list of named time points.
            Each event is a 2-tuple: (label, datetime.datetime).

    Returns:
        None.
    """
    if name is None:
        # outer_frame = inspect.getouterframes(inspect.currentframe())[1]
        filename = __file__
        name = os.path.basename(filename)
    if not time_point:
        time_point = datetime.datetime.now()
    events.append((name, time_point))


# ======================================================================
def report(
        events=_EVENTS,
        title='Elapsed Time(s)',
        labels=('Label', 'Duration / s', 'Cum. Duration / s'),
        max_col_widths=(36, 20, 20),
        title_sep='=',
        label_sep='-',
        only_last=False):
    """
    Print quick-and-dirty elapsed times between named event points.

    Args:
        events (list[(str,datetime.datetime)]): A list of named time points.
            Each event is a 2-tuple: (label, time).
        title (str): heading of the elapsed time table.
        labels (Iterable[str]): Labels for the report.
            Three elements are expected.
        max_col_widths (Iterable[int]): Maximum width of columns in the report.
            Three elements are expected.
        title_sep (str): The separator used to underline the title.
        label_sep (str): The separator used to underline the labels.
        only_last (bool): print only the last event (useful inside a loop).

    Returns:
        None.
    """
    text = '\n'
    if events:
        if not only_last and len(events) > 2:
            fmtt = '{{!s:{}s}}  {{!s:>{}s}}  {{!s:>{}s}}\n'.format(
                *max_col_widths)
            title_sep = ((title_sep * len(title))[:len(title)] + '\n') \
                if title_sep else ''
            text += title + (
                '\n' + title_sep + '\n' if len(events) > 2 else ': ')

            if labels and len(events) > 2:
                text += (fmtt.format(*labels))

            if label_sep:
                text += (fmtt.format(
                    *[(label_sep * max_col_width)[:max_col_width]
                      for max_col_width in max_col_widths]))

            first_elapsed = events[0][1]
            for i in range(1, len(events)):
                name, curr_elapsed = events[i]
                prev_elapsed = events[i - 1][1]
                diff_first = curr_elapsed - first_elapsed
                diff_last = curr_elapsed - prev_elapsed
                if diff_first == diff_last:
                    diff_first = '-'
                text += (fmtt.format(
                    name[:max_col_widths[0]], diff_last, diff_first))
        elif len(events) > 1:
            fmtt = '{{!s:{}s}}  {{!s:>{}s}}'.format(*max_col_widths)
            name, curr_elapsed = events[-1]
            prev_elapsed = events[0][1]
            text += (fmtt.format(name, curr_elapsed - prev_elapsed))
        else:
            events = None

    if not events:
        text += 'No ' + title.lower() + ' to report!'
    return text


# ======================================================================
def pkg_paths(
        current_filepath=__file__,
        name=INFO['name'],
        author=INFO['author'],
        version=INFO['version']):
    """
    Generate application directories.

    Args:
        current_filepath (str): The current filepath.
        name (str): Application name.
        author (str): Application author.
        version (str): Application version.

    Returns:
        dirs (dict): The package directories.
            - 'config': directory for configuration files.
            - 'cache': directory for caching files.
            - 'data': directory for data files.
            - 'log': directory for log files.
            - 'base': base directory of the module.
            - 'resources': directory for the data resources.

    Examples:
        >>> sorted(pkg_paths().keys())
        ['base', 'cache', 'config', 'data', 'log', 'resources']
    """
    dirpaths = dict((
        # todo: fix for pyinstaller
        ('config', appdirs.user_config_dir(name, author, version)),
        ('cache', appdirs.user_cache_dir(name, author, version)),
        ('data', appdirs.user_data_dir(name, author, version)),
        ('log', appdirs.user_data_dir(name, author, version)),
    ))
    for name, dirpath in dirpaths.items():
        if not os.path.isdir(dirpath):
            os.makedirs(dirpath)
    dirpaths['base'] = os.path.dirname(current_filepath)
    dirpaths['resources'] = os.path.join(dirpaths['base'], 'resources')
    return dirpaths


# ======================================================================
def run_doctests(module_docstring):
    msg(module_docstring.strip())
    msg('Running `doctest.testmod()`... ', fmtt='bold')
    results = doctest.testmod()  # RUN TESTS HERE!
    results_ok = results.attempted - results.failed
    results_fmt = '{t.bold}{t.red}' \
        if results.failed > 0 else '{t.bold}{t.green}'
    msg(fmtm('Tests = {results.attempted}; '), fmtt='{t.bold}{t.cyan}', end='')
    msg(fmtm('OK = {results_ok}; '), fmtt='{t.bold}{t.green}', end='')
    msg(fmtm('Fail = {results.failed}'), fmtt=results_fmt)
    msg(report())


# ======================================================================
PATH = pkg_paths(__file__, INFO['name'], INFO['author'], INFO['version'])

# ======================================================================
elapsed(os.path.basename(__file__))

# ======================================================================
# : populate flyingcircus namespace with submodules
from flyingcircus.base import *
import flyingcircus.base

# warnings.warn(
#     '`flyingcircus.base` will not be available or will not provide'
#     '\nthe same names (functions, classes, etc.) in the future.',
#     PendingDeprecationWarning)

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
