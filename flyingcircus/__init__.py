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
import pkg_resources  # Manage package resource (from setuptools module)
import doctest  # Test interactive Python examples

# ======================================================================
# :: External Imports
# import flyingcircus as fc  # Everything you always wanted to have in Python.*
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
def fmt(
        text,
        *_args,
        **_kws):
    """
    Perform string formatting using `text.format()`.

    Args:
        text (str|Any): Text to format.
        *_args: Positional arguments for `str.format()`.
        **_kws: Keyword arguments for `str.format()`.

    Returns:
        None.

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
    """
    return text.format(*_args, **_kws)


# ======================================================================
def fmtm(
        text,
        source=None):
    """
    Perform string formatting using `text.format_map()`.

    Args:
        text (str|Any): Text to format.
        source (Mapping|None): The mapping to use as source.
            If None, uses caller's `vars()`.

    Returns:
        None.

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
        TypeError: fmtm() takes from 1 to 2 positional arguments ...
        >>> fmtm('{b} + {b} = {}', 4)
        Traceback (most recent call last):
            ...
        TypeError: 'int' object is not subscriptable

    See Also:
        - flyingcircus.fmt()
    """
    if source is None:
        frame = inspect.currentframe()
        source = frame.f_back.f_locals
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
                txt1 = text.split(None, 1)[0] if len(text) > 0 else ''
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
import flyingcircus.base
import flyingcircus.extra

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
