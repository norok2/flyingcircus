#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flyingcircus.base: Base subpackage.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import io  # Core tools for working with streams
import sys  # System-specific parameters and functions
import math  # Mathematical functions
import random  # Generate pseudo-random numbers
import statistics  # Mathematical statistics functions
import time  # Time access and conversions
import itertools  # Functions creating iterators for efficient looping
import functools  # Higher-order functions and operations on callable objects
import operator  # Standard operators as functions
import collections  # Container datatypes
import subprocess  # Subprocess management
import multiprocessing  # Process-based parallelism
import datetime  # Basic date and time types
import inspect  # Inspect live objects
import stat  # Interpreting stat() results
import shlex  # Simple lexical analysis
import warnings  # Warning control
import importlib  # The implementation of import
import gzip  # Support for gzip files
import bz2  # Support for bzip2 compression
import copy  # Shallow and deep copy operations
# import lzma  # Compression using the LZMA algorithm
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
# import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
import struct  # Interpret strings as packed binary data
import re  # Regular expression operations
import fnmatch  # Unix filename pattern matching
import bisect  # Array bisection algorithm
# import heapq  # Heap queue algorithm
import hashlib  # Secure hashes and message digests
import base64  # Base16, Base32, Base64, Base85 Data Encodings
import pickle  # Python object serialization
import string  # Common string operations
import gc  # Garbage Collector interface

# :: External Imports

# :: External Imports Submodules

# :: Local Imports
from flyingcircus import INFO, PATH
from flyingcircus import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from flyingcircus import elapsed, report, run_doctests
from flyingcircus import msg, dbg, fmt, fmtm, safe_format_map
from flyingcircus import do_nothing_decorator
from flyingcircus import HAS_JIT, jit
from flyingcircus import valid_index, find_all, nested_delimiters

# ======================================================================
# :: Custom defined constants


# ======================================================================
# :: Default values usable in functions
CSV_DELIMITER = '\t'
CSV_COMMENT_TOKEN = '#'
EXT = {
    'gzip': 'gz',
    'bzip': 'bz2',
    'bzip2': 'bz2',
    'lzma': 'lzma',
    'xz': 'xz',
    'lzip': 'lz',
}
D_TAB_SIZE = 8

# ======================================================================
# : define SI prefix
SI_ORDER_STEP = 3
SI_PREFIX = {
    'base1000+': {
        k: (v + 1)
        for v, k in enumerate(('k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y'))},
    'base1000-': {
        k: -(v + 1)
        for v, k in enumerate(('m', 'μ', 'n', 'p', 'f', 'a', 'z', 'y'))},
    'base1000': {'': 0},
    'base10': {'': 0},
    'base10+': {'da': 1, 'h': 2},
    'base10-': {'d': -1, 'c': -2}}
SI_PREFIX['base1000'].update(SI_PREFIX['base1000+'])
SI_PREFIX['base1000'].update(SI_PREFIX['base1000-'])
SI_PREFIX['base10+'].update({
    k: v * SI_ORDER_STEP for k, v in SI_PREFIX['base1000+'].items()})
SI_PREFIX['base10-'].update({
    k: v * SI_ORDER_STEP for k, v in SI_PREFIX['base1000-'].items()})
SI_PREFIX['base10'].update(SI_PREFIX['base10+'])
SI_PREFIX['base10'].update(SI_PREFIX['base10-'])
SI_PREFIX_ASCII = copy.deepcopy(SI_PREFIX)
SI_PREFIX_EXTRA = copy.deepcopy(SI_PREFIX)
# handle multiple micro signs: u, µ, μ
for _mode in ('base1000-', 'base10-', 'base1000', 'base10'):
    SI_PREFIX_EXTRA[_mode]['u'] = SI_PREFIX[_mode]['μ']
    SI_PREFIX_EXTRA[_mode]['µ'] = SI_PREFIX[_mode]['μ']
    SI_PREFIX_ASCII[_mode]['u'] = SI_PREFIX[_mode]['μ']
    del SI_PREFIX_ASCII[_mode]['μ']

# ======================================================================
# superscript / subscript maps
SUPERSCRIPT_MAP = str.maketrans(
    '0123456789.e+-=()n',
    '⁰¹²³⁴⁵⁶⁷⁸⁹·ᵉ⁺⁻⁼⁽⁾ⁿ')
SUBSCRIPT_MAP = str.maketrans(
    '0123456789.e+-=()n',
    '₀₁₂₃₄₅₆₇₈₉.ₑ₊₋₌₍₎ₙ')

# ======================================================================
# :: define C types
# : short form (base types used by `struct`)
_STRUCT_TYPES = (
    'x',  # pad bytes
    'c',  # char 1B
    'b',  # signed char 1B
    'B',  # unsigned char 1B
    '?',  # bool 1B
    'h',  # short int 2B
    'H',  # unsigned short int 2B
    'i',  # int 4B
    'I',  # unsigned int 4B
    'l',  # long 4B
    'L',  # unsigned long 4B
    'q',  # long long 8B
    'Q',  # unsigned long long 8B
    'f',  # float 4B
    'd',  # double 8B
    's', 'p',  # char[]
    'P',  # void * (only support mode: '@')
)
# : data type format conversion for `struct`
# ... same as: dict(zip(_STRUCT_TYPES, _STRUCT_TYPES))
DTYPE_STR = {s: s for s in _STRUCT_TYPES}
# : define how to interpreted Python types
DTYPE_STR.update({
    bool: '?',
    int: 'i',
    float: 'f',
    str: 's',
})
# : define how to interpret human-friendly types
DTYPE_STR.update({
    'bool': '?',
    'char': 'b',
    'uchar': 'B',
    'short': 'h',
    'ushort': 'H',
    'int': 'i',
    'uint': 'I',
    'long': 'l',
    'ulong': 'L',
    'llong': 'q',
    'ullong': 'Q',
    'float': 'f',
    'double': 'd',
    'str': 's',
})

# ======================================================================
_ZLIB_CHECKSUMS = {"crc32": 4, "adler32": 4}
_VAR_LENGTH_HASHLIB_ALGORITHMS = {"shake_128": 128, "shake_256": 256}
_BASE64_ENCODINGS = {"b16", "b32", "b64", "b85", "urlsafe_b64"}

# ======================================================================
_str_join = functools.partial(str.join, '')


# ======================================================================
def _is_hidden(filepath):
    """
    Heuristic to determine hidden files.

    Args:
        filepath (str): the input filepath.

    Returns:
        is_hidden (bool): True if is hidden, False otherwise.

    Notes:
        Only works with UNIX-like files, relying on prepended '.'.
    """
    return os.path.basename(filepath).startswith('.')


# ======================================================================
def _is_special(stats_mode):
    """
    Heuristic to determine non-standard files.

    Args:
        stats_mode (mode): A mode as specified by the `stat` module.

    Returns:
        is_special (bool): True if is hidden, False otherwise.

    Notes:
        Its working relies on Python stat module implementation.
    """
    is_special = \
        not stat.S_ISREG(stats_mode) and \
        not stat.S_ISDIR(stats_mode) and \
        not stat.S_ISLNK(stats_mode)
    return is_special


# ======================================================================
def _guess_container(seq, container=None):
    """
    Guess container for combinatorial computations.

    Args:
        seq (Sequence): The input items.
        container (callable|None): The container for the result.
            If None, this is inferred from `seq` if possible, otherwise
            uses `tuple`.

    Returns:
        container (callable): The container function.
    """
    if container is None:
        container = type(seq)
    if not callable(container):
        container = tuple
    elif container is str:
        container = _str_join  # cannot use local declaration
    elif container in (bytes, bytearray):
        container = bytes
    try:
        container(seq[0:0])
    except TypeError:
        container = tuple
    return container


# ======================================================================
def parametric(decorator):
    """
    Create a decorator-factory from a decorator.

    This is for creating decorators with parameters.

    Note: contrarily to function parameters, decorator parameter's names
    should never be reassigned.

    Args:
        decorator (callable): The input decorator.

    Returns:
        decorator_factory (callable): The decorated decorator-factory.

    Examples:
        >>> def multiply(func):
        ...     @functools.wraps(func)
        ...     def wrapper(*args, **kws):
        ...         return 10 * func(*args, **kws)
        ...     return wrapper
        >>> @multiply
        ... def my_sum(values):
        ...     return sum(values)
        >>> my_sum(range(10))
        450

        >>> @parametric
        ... def multiply(func, factor=1):
        ...     @functools.wraps(func)
        ...     def wrapper(*args, **kws):
        ...         return factor * func(*args, **kws)
        ...     return wrapper
        >>> @multiply()
        ... def my_sum(values):
        ...     return sum(values)
        >>> my_sum(range(10))
        45
        >>> @multiply(10)
        ... def my_sum(values):
        ...     return sum(values)
        >>> my_sum(range(10))
        450
    """

    @functools.wraps(decorator)
    def _decorator(*_args, **_kws):
        def _wrapper(func):
            return decorator(func, *_args, **_kws)

        return _wrapper

    return _decorator


# ======================================================================
@parametric
def auto_star_magic(
        func,
        mode='type',
        allow_empty=False):
    """
    Decorate a function to accept starred arguments instead of an iterable.

    Args:
        func (callable): The input function.
        mode (str): The auto-star detection mode.
            This determines on what condition `func` is decorated.
            Allowed modes:
             - 'type': if the first parameter is iterable.
             - 'nargs': if `func` accepts a single parameter.
        allow_empty (bool): Allow `func` to be called without parameters.
            This is effective only if the first parameter of `func`
            does not have a default value.

    Returns:
        wrapper (callable): The decorated function.

    Raises:
        TypeError: If `allow_empty` is False, `func`'s first parameter
            does not have a default value and no value is specified when
            calling the decorated function.

    Examples:
        >>> @auto_star_magic('type', True)
        ... def my_prod(items, start=1):
        ...     for item in items:
        ...         start *= item
        ...     return start
        >>> print(my_prod(1, 2, 3))
        6
        >>> print(my_prod(range(1, 4)))
        6
        >>> print(my_prod())
        1

        >>> @auto_star_magic('type', False)
        ... def my_prod(items, start=1):
        ...     for item in items:
        ...         start *= item
        ...     return start
        >>> print(my_prod(1, 2, 3))
        6
        >>> print(my_prod(range(1, 4)))
        6
        >>> print(my_prod())
        Traceback (most recent call last):
            ...
        TypeError: my_prod() missing 1 required positional argument: 'items'

        >>> @auto_star_magic('nargs', True)
        ... def my_prod(items):
        ...     start = 1
        ...     for item in items:
        ...         start *= item
        ...     return start
        >>> print(my_prod(1, 2, 3))
        6
        >>> print(my_prod(range(1, 4)))
        6
        >>> print(my_prod())
        1

        >>> @auto_star_magic('nargs', False)
        ... def my_prod(items):
        ...     start = 1
        ...     for item in items:
        ...         start *= item
        ...     return start
        >>> print(my_prod(1, 2, 3))
        6
        >>> print(my_prod(range(1, 4)))
        6
        >>> print(my_prod())
        Traceback (most recent call last):
            ...
        TypeError: my_prod() missing 1 required positional argument: 'items'
    """

    @functools.wraps(func)
    def wrapper(*args, **kws):
        sig = inspect.signature(func)
        first_param_name = next(iter(sig.parameters))
        func_name = func.__name__ if hasattr(func, '__name__') else '<unnamed>'
        has_default = \
            sig.parameters[first_param_name].default != inspect.Parameter.empty
        if not allow_empty and not has_default and len(args) == 0:
            raise TypeError(
                fmtm(
                    "{func_name}() missing 1 required positional argument:"
                    " '{first_param_name}'"))
        else:
            if mode == 'type':
                try:
                    iter(args[0])
                except (TypeError, IndexError):
                    use_star = False
                else:
                    use_star = True
            elif mode == 'nargs':
                use_star = len(args) == 1
            else:
                use_star = False
            return func(*args, **kws) if use_star else func(args, **kws)

    return wrapper


# ======================================================================
@parametric
def iterize(
        func,
        container=None,
        fill_param='fill'):
    """
    Decorator for iterize a function in its positional arguments.

    Furthermore, it adds support for an additional `fill` parameter
    (the exact parameter name can be changed).
    This parameter controls how shorter iterables are extented.
    If `fill` is None, the iterable is extended using an infinite loop.
    Otherwise, uses the `fill` value specified.

    Args:
        func (callable): The input function.
        container (callable|None): The container for the result.
            If callable, must accept a `map` object, otherwise
            the `map` object itself will be used.
        fill_param (str): The name of the extra parameter used for filling.


    Returns:
        wrapper (callable): The iterized function.

    Examples:
        >>> @iterize(list)
        ... def isum(*x):
        ...     return sum(x)
        >>> isum(range(10), 100)
        [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
        >>> isum(range(10), 100, fill=0)
        [100, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> isum(range(10), range(5))
        [0, 2, 4, 6, 8, 5, 7, 9, 11, 13]
        >>> isum(range(10), range(3))
        [0, 2, 4, 3, 5, 7, 6, 8, 10, 9]
        >>> isum(range(10), range(3), fill=0)
        [0, 2, 4, 3, 4, 5, 6, 7, 8, 9]
        >>> isum(range(4), range(8), range(12), fill=0)
        [0, 3, 6, 9, 8, 10, 12, 14, 8, 9, 10, 11]
        >>> isum(range(4), range(8), range(12))
        [0, 3, 6, 9, 8, 11, 14, 17, 8, 11, 14, 17]
    """

    @functools.wraps(func)
    def wrapper(*args, **kws):
        fill = kws.pop(fill_param) if fill_param in kws else None
        if fill is not None:
            def extend(items):
                for item in items:
                    yield item
                while True:
                    yield fill
        else:
            extend = itertools.cycle
        # find maximum length and sanitize arguments
        max_len = 0
        iargs = []
        for arg in args:
            try:
                iter(arg)
            except TypeError:
                arg = [arg]
            else:
                try:
                    max_len = max(max_len, len(arg))
                except TypeError:
                    pass
            iargs.append(arg)
        # extend the short arguments
        iargs = (
            iarg if len(iarg) == max_len else extend(iarg) for iarg in iargs)
        result = map(func, *iargs, **kws)
        if callable(container):
            result = container(result)
        return result

    return wrapper


# ======================================================================
class Infix(object):
    """
    Emulate an infix operator using an arbitrary variable.

    This can also be used as a decorator.

    Examples:
        >>> to = Infix(range)
        >>> to(1, 10)
        range(1, 10)
        >>> 1 | to | 10
        range(1, 10)
        >>> [x for x in 1 | to | 15]
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        >>> @Infix
        ... def till(a, b):
        ...     return range(a, b)
        >>> (1 | to | 10) == (1 | till | 10)
        True

        >>> ((1 + to + 9) == (1 - to - 9) == (1 * to * 9) == (1 / to / 9)
        ...  == (1 // to // 9) == (1 % to % 9) == (1 ** to ** 9)
        ...  == (1 >> to >> 9) == (1 << to << 9)
        ...  == (1 | to | 9) == (1 & to & 9) == (1 ^ to ^ 9)
        ...  == (1 << to >> 9) == (1 + to ^ 9))  # etc. (all combos work)
        True
    """

    # ----------------------------------------------------------
    def __init__(self, func):
        """
        Args:
            func (callable): The function to emulate the binary operator.
                The function must support two positional arguments.
        """
        self.func = func

    # ----------------------------------------------------------
    class RBind:
        # ----------------------------------------------------------
        def __init__(self, func, binded):
            self.func = func
            self.binded = binded

        # ----------------------------------------------------------
        def __call__(self, other):
            return self.func(other, self.binded)

        # ----------------------------------------------------------
        __radd__ = __rsub__ = __rmul__ = __rtruediv__ = __rfloordiv__ = \
            __rmod__ = __rpow__ = __rmatmul__ = __rlshift__ = __rrshift__ = \
            __ror__ = __rand__ = __rxor__ = __call__

    # ----------------------------------------------------------
    class LBind:
        # ----------------------------------------------------------
        def __init__(self, func, binded):
            self.func = func
            self.binded = binded

        # ----------------------------------------------------------
        def __call__(self, other):
            return self.func(self.binded, other)

        # ----------------------------------------------------------
        __add__ = __sub__ = __mul__ = __truediv__ = __floordiv__ = \
            __mod__ = __pow__ = __matmul__ = __lshift__ = __rshift__ = \
            __or__ = __and__ = __xor__ = __call__

    # ----------------------------------------------------------
    def rbind(self, other):
        return self.RBind(self.func, other)

    # ----------------------------------------------------------
    def lbind(self, other):
        return self.LBind(self.func, other)

    # ----------------------------------------------------------
    def __call__(self, value1, value2):
        return self.func(value1, value2)

    # ----------------------------------------------------------
    __add__ = __sub__ = __mul__ = __truediv__ = __floordiv__ = \
        __mod__ = __pow__ = __matmul__ = __lshift__ = __rshift__ = \
        __or__ = __and__ = __xor__ = rbind
    __radd__ = __rsub__ = __rmul__ = __rtruediv__ = __rfloordiv__ = \
        __rmod__ = __rpow__ = __rmatmul__ = __rlshift__ = __rrshift__ = \
        __ror__ = __rand__ = __rxor__ = lbind


# ======================================================================
def set_func_kws(
        func,
        func_kws):
    """
    Set keyword parameters of a function to specific or default values.

    Args:
        func (callable): The function to be inspected.
        func_kws (Mappable|None): The (key, value) pairs to set.
            If a value is None, it will be replaced by the default value.
            To use the names defined locally, use: `locals()`.

    Results:
        result (dict): A dictionary of the keyword parameters to set.

    See Also:
        inspect, locals, globals.
    """
    try:
        get_argspec = inspect.getfullargspec
    except AttributeError:
        get_argspec = inspect.getargspec
    inspected = get_argspec(func)
    defaults = dict(
        zip(reversed(inspected.args), reversed(inspected.defaults)))
    result = {}
    for key in inspected.args:
        if key in func_kws:
            result[key] = func_kws[key]
        elif key in defaults:
            result[key] = defaults[key]
    return result


# ======================================================================
def split_func_kws(
        func,
        func_kws):
    """
    Split a set of keywords into accepted and not accepted by some function.

    Args:
        func (callable): The function to be inspected.
        func_kws (Mappable|None): The (key, value) pairs to split.

    Results:
        result (tuple): The tuple
            contains:
             - in_func (dict): The keywords NOT accepted by `func`.
             - not_in_func (Mappable|None): The keywords accepted by `func`.

    See Also:
        inspect, locals, globals.
    """
    try:
        get_argspec = inspect.getfullargspec
    except AttributeError:
        get_argspec = inspect.getargspec
    inspected = get_argspec(func)
    in_func = {}
    not_in_func = {}
    for k, v in func_kws.items():
        if k in inspected.args:
            in_func[k] = v
        else:
            not_in_func[k] = v
    return in_func, not_in_func


# ======================================================================
def join(
        operands,
        coerce=True):
    """
    Join together multiple containers.

    Args:
        operands (Iterable): The operands to join.
        coerce (bool): Cast all operands into the type of the first.

    Returns:
        result: The joined object.

    Examples:
        >>> join(([1], [2], [3]))
        [1, 2, 3]
        >>> join(((1, 2), (3, 4)))
        (1, 2, 3, 4)
        >>> join(({1: 2}, {2: 3}, {3: 4}))
        {1: 2, 2: 3, 3: 4}
        >>> join(({1}, {2, 3}, {3, 4}))
        {1, 2, 3, 4}
        >>> join(([1], [2], (3, 4)))
        [1, 2, 3, 4]
        >>> join(((1,), [2], (3, 4)))
        (1, 2, 3, 4)
        >>> join(((1, 2), (3, 4)), coerce=False)
        (1, 2, 3, 4)
        >>> join(((1,), [2], (3, 4)), coerce=False)
        Traceback (most recent call last):
            ...
        TypeError: can only concatenate tuple (not "list") to tuple

        # These are side effect of duck-typing:
        >>> join([1, 2.5, 3])
        6
        >>> join([1.0, 2.5, 3])
        6.5
        >>> join([[1], 2, 3.0])
        Traceback (most recent call last):
            ...
        TypeError: 'int' object is not iterable
        >>> join([1, [2], 3.0])
        Traceback (most recent call last):
            ...
        TypeError: int() argument must be a string, a bytes-like object or a\
 number, not 'list'

        >>> join(['aaa', 'bbb', 'ccc'])
        'aaabbbccc'
        >>> join([b'aaa', b'bbb', b'ccc'])
        b'aaabbbccc'
        >>> join(x for x in string.ascii_lowercase)
        'abcdefghijklmnopqrstuvwxyz'
        >>> join(x.encode() for x in string.ascii_lowercase)
        b'abcdefghijklmnopqrstuvwxyz'

        >>> join(['aaa', b'bbb', 'ccc'])
        'aaabbbccc'
        >>> join([b'aaa', 'bbb', b'ccc'])
        b'aaabbbccc'
        >>> join(['aaa', b'bbb', 'ccc'], False)
        Traceback (most recent call last):
            ...
        TypeError: sequence item 0: expected str instance, bytes found
        >>> join([b'aaa', 'bbb', b'ccc'], False)
        Traceback (most recent call last):
            ...
        TypeError: sequence item 0: expected a bytes-like object, str found

    See Also:
        - flyingcircus.join_()
    """
    iter_operands = iter(operands)
    result = next(iter_operands)
    type_result = type(result)
    if type_result == str:
        if coerce:
            for x in iter_operands:
                if isinstance(x, bytes):
                    x = x.decode()
                elif not isinstance(x, str):
                    x = str(x)
                result += x
        else:
            result += ''.join(iter_operands)
    elif type_result in (bytes, bytearray):
        if coerce:
            for x in iter_operands:
                if isinstance(x, str):
                    x = x.encode()
                elif not isinstance(x, (bytes, bytearray)):
                    x = bytes(x)
                result += x
        else:
            result += b''.join(iter_operands)
    elif hasattr(result, '__iadd__'):
        if coerce:
            for x in iter_operands:
                result += x if isinstance(x, type_result) else type_result(x)
        else:
            for x in iter_operands:
                result += x
    elif hasattr(result, '__add__'):
        if coerce:
            for x in iter_operands:
                result = result + (
                    x if isinstance(x, type_result) else type_result(x))
        else:
            for x in iter_operands:
                result = result + x
    elif hasattr(result, 'update'):
        if coerce:
            for x in iter_operands:
                result.update(
                    x if isinstance(x, type_result) else type_result(x))
        else:
            for x in iter_operands:
                result.update(x)
    else:
        raise ValueError(fmtm('Cannot apply `join()` on `{operands}`'))
    return result


# ======================================================================
def join_(*operands, **_kws):
    """
    Star magic version of `flyingcircus.join()`.

    Examples:
        >>> join_([1], [2], [3])
        [1, 2, 3]
        >>> join_((1, 2), (3, 4))
        (1, 2, 3, 4)
        >>> join_({1: 2}, {2: 3}, {3: 4})
        {1: 2, 2: 3, 3: 4}
        >>> join_({1}, {2, 3}, {3, 4})
        {1, 2, 3, 4}
        >>> join_([1], [2], (3, 4))
        [1, 2, 3, 4]
        >>> join_((1,), [2], (3, 4))
        (1, 2, 3, 4)
        >>> join_((1, 2), (3, 4), coerce=False)
        (1, 2, 3, 4)
        >>> join_((1,), [2], (3, 4), coerce=False)
        Traceback (most recent call last):
            ...
        TypeError: can only concatenate tuple (not "list") to tuple

    See Also:
        - flyingcircus.join()
    """
    return join(operands, **_kws)


# ======================================================================
def get_nested_attr(
        obj,
        *names):
    """
    Get the nested attributes of an object.

    Args:
        obj (Any): The input object.
        *names (tuple[str]): The attributes to get.

    Returns:
        Any: The specified attribute.

    Examples:
        >>> print(get_nested_attr(1, 'real'))
        1
        >>> print(get_nested_attr(1, 'real', 'imag'))
        0
        >>> print(get_nested_attr(1, '__class__'))
        <class 'int'>
        >>> print(get_nested_attr(1, '__class__', '__name__'))
        int
        >>> print(get_nested_attr(1, '__class__', '__name__', '__class__'))
        <class 'str'>
    """
    for name in names:
        obj = getattr(obj, name)
    return obj


# ======================================================================
def checksum(
    obj,
    chksum_func="sha256",
    ser_func=pickle.dumps,
    to_str_func="hex",
    max_len=None,
):
    """Compute a checksum of an object.

    Args:
        obj (Any): The input object.
        chksum_func (callable|str): The checksum computing function.
            If str, must be in `hashlib.algorithm_available`
            or one of {"crc32", "adler32"}.
        ser_func (callable|str|None): The serialization function.
        to_str_func (callable|str|None): The checksum-to-string conversion function.
            If str, must be one of:
             - "hex": uses `binascii.b2a_hex()`
             - "b16": uses `base64.b16encode()`
             - "b32": uses `base64.b32encode()`
             - "b64": uses `base64.b64encode()`
             - "b85": uses `base64.b85encode()`
             - "urlsafe_b64": uses `base64.urlsafe_b64encode()`
        max_len: Max length of the checksum.
            If the checksum exceed this value, it is coerced to the specific length.
            If None, uses the default length.

    Returns:
        The computed checksum as string.

    Examples:
        >>> chksum(1)
        '018f5c4626b56e8489da7abb6c8b62331933c42c35d1342037a5242b8ed148f6'
        >>> chksum("This is some text")
        '388ef7c8f5cb46a241f12dbb7eefb5bde22b4cf1db539a03a71cb00b94b790e1'
        >>> chksum(None)
        '9c298d589a2158eb513cb52191144518a2acab2cb0c04f1df14fca0f712fa4a1'
        >>> chksum({1: {2: 3, 4: {5: [6, {7: 8}], 9: 0}}})
        'c6e3c985351ce6e394d77f931ae65fe8c78beb1c7f3521affa46c00cdd997557'
    """

    if callable(ser_func):
        obj = ser_func(obj)
    if isinstance(chksum_func, str):
        if chksum_func in hashlib.algorithms_available:
            if chksum_func in _VAR_LENGTH_HASHLIB_ALGORITHMS:
                length = _VAR_LENGTH_HASHLIB_ALGORITHMS[chksum_func]
                obj = getattr(hashlib, chksum_func)(obj).digest(length)
            else:
                obj = getattr(hashlib, chksum_func)(obj).digest()
        elif chksum_func in _ZLIB_CHECKSUMS:
            chksum_size = _ZLIB_CHECKSUMS[chksum_func]
            obj = getattr(zlib, chksum_func)(obj).to_bytes(chksum_size, "little")
            obj = binascii.b2a_hex(obj).decode("ascii")
        else:
            raise ValueError("Unsupported checksum method.")
    elif callable(chksum_func):
        obj = chksum_func(obj)
    if isinstance(to_str_func, str):
        if to_str_func == "hex":
            obj = binascii.b2a_hex(obj)
        elif to_str_func in _BASE64_ENCODINGS:
            obj = getattr(base64, f"{to_str_func}encode")(obj)
        else:
            raise ValueError("Unsupported conversion method.")
    elif callable(to_str_func):
        obj = to_str_func(obj)
    if isinstance(obj, bytes):
        try:
            obj = obj.decode("ascii")
        except UnicodeDecodeError:
            obj = binascii.b2a_hex(obj).decode("ascii")
    if max_len is not None:
        obj = obj[:max_len]
    return obj


# ======================================================================
def idir(
        obj,
        skip='_',
        methods=True,
        attributes=True,
        yield_attr=False):
    """
    Iteratively (and selectively) list the attributes of an object.

    Args:
        obj (Any): The input object.
        skip (str|callable): The skip criterion.
            If str, names starting with the specified string are skipped.
            If callable, skips when `skip(name)` evaluates to True.
        methods (bool): Include methods.
            If False, the callable attributes are excluded.
        attributes (bool): Include non-callable attributes.
            If False, the non-callable attributes are excluded.
        yield_attr (bool): Yield both the name and the attribute.
            If False, only the name is yielded (same behavior as `dir()`).

    Yields:
        str|tuple[str, Any]: The name or the name/attribute pair.
            The return type/mode depends on the value of `yield_attr`.

    Examples:
        >>> list(idir(1, '_', True, True))  # doctest:+ELLIPSIS
        [..., 'from_bytes', 'imag', 'numerator', 'real', 'to_bytes']
        >>> list(idir(1, '_', False, True))
        ['denominator', 'imag', 'numerator', 'real']
        >>> list(idir(1, '_', True, False))
        ['bit_length', 'conjugate', 'from_bytes', 'to_bytes']
        >>> list(idir(1, '_', False, False))
        []
        >>> list(idir(1, '_', False, True, True))
        [('denominator', 1), ('imag', 0), ('numerator', 1), ('real', 1)]
        >>> list(idir(1, '_', True, False, True))  # doctest:+ELLIPSIS
        [\
('bit_length', <built-in method bit_length of int object at ...>), \
('conjugate', <built-in method conjugate of int object at ...>), \
('from_bytes', <built-in method from_bytes of type object at ...>), \
('to_bytes', <built-in method to_bytes of int object at ...>)]
    """
    if not callable(skip):
        def skip(t, s=skip):
            return t.startswith(s)

    if not methods and not attributes:
        return
    for name in dir(obj):
        if not skip(name):
            attr = getattr(obj, name)
            is_callable = callable(attr)
            if (methods and is_callable) or (attributes and not is_callable):
                yield (name, attr) if yield_attr else name


# ======================================================================
def reverse_mapping(
        mapping,
        check=True):
    """
    Reverse the (key, value) relationship of a mapping.

    Given a (key, value) sequence, returns a (value, key) sequence.
    The values in the input mapping must be allowed to be used as `dict` keys.

    Args:
        mapping (Mapping): The input mapping
        check (bool): Perform a validity check.
            The check succeeds if all values in the original mapping are
            unique.

    Returns:
        result (dict): The reversed mapping.

    Raises:
        KeyError: if `check == True` and a duplicate key is detected.

    Examples:
        >>> mapping = {i: str(i) for i in range(5)}
        >>> print(sorted(mapping.items()))
        [(0, '0'), (1, '1'), (2, '2'), (3, '3'), (4, '4')]
        >>> print(sorted(reverse_mapping(mapping).items()))
        [('0', 0), ('1', 1), ('2', 2), ('3', 3), ('4', 4)]
        >>> reverse_mapping({1: 2, 2: 2})
        Traceback (most recent call last):
            ...
        KeyError: 'Reversing a mapping detected duplicate key in the result!'
        >>> reverse_mapping({1: 2, 2: 2}, False)
        {2: 2}

    See Also:
        - flyingcircus.reverse_mapping_iter()
    """
    result = {v: k for k, v in mapping.items()}
    if check and len(mapping) != len(result):
        raise KeyError(
            'Reversing a mapping detected duplicate key in the result!')
    else:
        return result


# ======================================================================
def as_dict(obj):
    """Transparently convert any object to dict.

    If the object cannot be converted to `dict`, returns an empty `dict`.

    Args:
        obj: The input object.

    Returns:
        The object converted to a dict.

    Examples:
        >>> as_dict({1: 2, 3: 4})
        {1: 2, 3: 4}
        >>> as_dict(((1, 2), (3, 4)))
        {1: 2, 3: 4}
        >>> as_dict([[1, 2], [3, 4]])
        {1: 2, 3: 4}
        >>> as_dict(())
        {}
        >>> as_dict(None)
        {}
    """
    try:
        return dict(obj)
    except TypeError:
        return {}


# ======================================================================
def reverse_mapping_iter(mapping):
    """
    Reverse the (key, values) relationship of a mapping.

    Given a (key, values) sequence, returns a reversed (value, keys) sequence.
    The input values need to be iterable and their items need to be hashable.
    Each of the values either generates a new key or gets appended to the new
    values in the result.

    Args:
        mapping (Mapping): The input mapping with iterable values.

    Returns:
        result (dict): The reversed mapping.

    Examples:
        >>> mapping = {i: [str(i)] for i in range(5)}
        >>> print(sorted(mapping.items()))
        [(0, ['0']), (1, ['1']), (2, ['2']), (3, ['3']), (4, ['4'])]
        >>> print(sorted(reverse_mapping_iter(mapping).items()))
        [('0', [0]), ('1', [1]), ('2', [2]), ('3', [3]), ('4', [4])]
        >>> reverse_mapping_iter({1: [2], 2: [2]})
        {2: [1, 2]}
        >>> reverse_mapping_iter({1: [1, 2], 2: [2]})
        {1: [1], 2: [1, 2]}
        >>> reverse_mapping_iter({1: [1, 2, 1], 2: [2, 1]})
        {1: [1, 1, 2], 2: [1, 2]}

    See Also:
        - flyingcircus.reverse_mapping()
    """
    result = {}
    for key, values in mapping.items():
        for value in values:
            if value in result:
                result[value].append(key)
            else:
                result[value] = [key]
    return result


# ======================================================================
def multi_at(
        seq,
        indexes,
        container=None):
    """
    Extract selected items according to the specified indexes.

    Note that this is mostly equivalent to (but faster than)
    `flyingcircus.iter_at()`.

    Args:
        seq (Sequence): The input items.
        indexes (Iterable[int|slice]): The items to select.
        container (callable|None): The container for the result.
            If None, this is inferred from `seq` if possible, otherwise
            uses `tuple`.

    Returns:
        result: The selected items.

    Examples:
        >>> items = [x ** 2 for x in range(20)]
        >>> print(multi_at(items, (0, 6, 7, 0, 1, 3)))
        [0, 36, 49, 0, 1, 9]
        >>> indexes = (0, 6, 7, slice(1, 3))
        >>> print(multi_at(items, (12, 4, 11, slice(1, 3))))
        [144, 16, 121, [1, 4]]
        >>> print(multi_at(items, 19))
        361
        >>> print(multi_at(items, slice(3, 9, 2)))
        [9, 25, 49]

        >>> items = string.ascii_letters
        >>> print(multi_at(items, (0, 6, 7, 0, 1, 3)))
        ('a', 'g', 'h', 'a', 'b', 'd')
        >>> indexes = (0, 6, 7, slice(1, 3))
        >>> print(multi_at(items, (12, 4, 11, slice(1, 3))))
        ('m', 'e', 'l', 'bc')
        >>> print(multi_at(items, 19))
        t
        >>> print(multi_at(items, slice(3, 9, 2)))
        dfh

    See Also:
        - flyingcircus.iter_at()
    """
    try:
        iter(indexes)
    except TypeError:
        return seq[indexes]
    else:
        if container is None:
            container = type(seq)
        if not callable(container):
            container = tuple
        result = operator.itemgetter(*indexes)(seq)
        return result if container == tuple else container(result)


# ======================================================================
def iter_at(
        seq,
        indexes):
    """
    Iterate over selected items according to the specified indexes.

    Note that this is mostly equivalent to `flyingcircus.multi_at()`
    except that this yields a generator.

    Args:
        seq (Sequence): The input items.
        indexes (Iterable[int|slice]): The items to select.

    Yields:
        item: The selected item.

    Examples:
        >>> items = [x ** 2 for x in range(20)]
        >>> print(list(iter_at(items, (0, 6, 7, 0, 1, 3))))
        [0, 36, 49, 0, 1, 9]
        >>> indexes = (0, 6, 7, slice(1, 3))
        >>> print(list(iter_at(items, (12, 4, 11, slice(1, 3)))))
        [144, 16, 121, [1, 4]]
        >>> print(list(iter_at(items, 19)))
        [361]
        >>> print(list(iter_at(items, slice(3, 9, 2))))
        [9, 25, 49]

    See Also:
        - flyingcircus.multi_at()
    """
    try:
        iter(indexes)
    except TypeError:
        try:
            iter(seq[indexes])
        except TypeError:
            yield seq[indexes]
        else:
            for item in seq[indexes]:
                yield item
    else:
        for index in indexes:
            yield seq[index]


# ======================================================================
def index_all(
        seq,
        item,
        first=0,
        last=-1):
    """
    Find all occurrences of an item in an sequence.

    For dense inputs (item is present in more than ~20% of the sequence),
    a looping comprehension may be faster.

    For string, bytes or bytearray inputs, `flyingcircus.find_all()`
    is typically faster.

    Args:
        seq (Sequence): The input sequence.
        item (Any): The item to find.
        first (int): The first index.
            The index is forced within boundaries.
        last (int): The last index (included).
            The index is forced within boundaries.

    Yields:
        position (int): The position of the next finding.

    Examples:
        >>> list(index_all([1, 2, 3, 5, 3, 4, 5, 7, 8], 3))
        [2, 4]
        >>> list(index_all((1, 2, 3, 5, 3, 4, 5, 7, 8), 3))
        [2, 4]
        >>> list(index_all((1, 2, 3, 5, 3, 4, 5, 7, 8), 0))
        []
        >>> list(index_all((1, 2, 3, 5, 3, 4, 5, 7, 8), [3, 5]))
        []
        >>> list(index_all((1, 2, 3, 5, 3, 4, 5, 7, None), None))
        [8]
        >>> list(index_all((1, 2, 3, 5, 3, 4, 5, 7, 8), ()))
        []
        >>> list(index_all('0010120123012340123450123456', '0'))
        [0, 1, 3, 6, 10, 15, 21]
        >>> list(index_all('  1 12 123 1234 12345 123456', '0'))
        []
        >>> list(index_all('  1 12 123 1234 12345 123456', '12'))
        [4, 7, 11, 16, 22]
        >>> list(index_all(b'  1 12 123 1234 12345 123456', b'12'))
        [4, 7, 11, 16, 22]
        >>> list(index_all('0123456789', ''))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> list(index_all('', ''))
        []
        >>> list(index_all('', '0123456789'))
        []
        >>> list(index_all(b'0123456789', b''))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> list(index_all(b'', b''))
        []
        >>> list(index_all(b'', b'0123456789'))
        []

    See Also:
        - flyingcircus.find_all()
    """
    n = len(seq)
    if n > 0:
        first = valid_index(first, n)
        last = valid_index(last, n)
        i = first
        try:
            while True:
                i = seq.index(item, i)
                if i <= last:
                    yield i
                    i += 1
                else:
                    return
        except ValueError:
            return
    else:
        return


# ======================================================================
def nested_pairs(
        seq,
        l_item,
        r_item,
        including=True):
    """
    Find matching delimiters in a sequence.

    The delimiters are matched according to nesting level.
    For string, bytes or bytearray inputs,
    `flyingcircus.find_nested_delims()` is a better option, because it
    supports multi-char delimiters and it is typically faster.

    Args:
        seq (Sequence): The input sequence.
        l_item (Any): The left delimiter item.
        r_item (Any): The right delimiter item.
        including (bool): Include delimiters.

    yields:
        result (tuple[int]): The matching delimiters.

    Examples:
        >>> s = '{a} {b:{c}}'
        >>> list(nested_pairs(s, '{', '}'))
        [(0, 3, 0), (7, 10, 1), (4, 11, 0)]
        >>> [s[i:j] for i, j, depth in nested_pairs(s, '{', '}')]
        ['{a}', '{c}', '{b:{c}}']
        >>> [s[i:j] for i, j, d in nested_pairs(s, '{', '}') if d == 0]
        ['{a}', '{b:{c}}']
        >>> list(nested_pairs('{a} {b:{c}', '{', '}'))
        Traceback (most recent call last):
            ...
        ValueError: Found `1` unmatched left token(s) `{` (position: 4).
        >>> list(nested_pairs('{a}} {b:{c}}', '{', '}'))
        Traceback (most recent call last):
            ...
        ValueError: Found `1` unmatched right token(s) `}` (position: 3).

    See Also:
        - flyingcircus.nested_delimiters()
    """
    l_offset = r_offset = int(including)
    stack = []

    l_items = set(index_all(seq, l_item))
    r_items = set(index_all(seq, r_item))
    positions = l_items.union(r_items)
    for pos in sorted(positions):
        if pos in l_items:
            stack.append(pos + 1)
        elif pos in r_items:
            if len(stack) > 0:
                prev = stack.pop()
                yield (prev - l_offset, pos + r_offset, len(stack))
            else:
                raise ValueError(fmt(
                    'Found `{}` unmatched right token(s) `{}`'
                    ' (position: {}).',
                    len(r_items) - len(l_items), r_item, pos))
    if len(stack) > 0:
        raise ValueError(fmt(
            'Found `{}` unmatched left token(s) `{}`'
            ' (position: {}).',
            len(l_items) - len(r_items), l_item, stack.pop() - 1))


# ======================================================================
@Infix
def span(
        first,
        second=None,
        step=None):
    """
    Span consecutive numbers in a range.

    This is useful to produce 1-based ranges, which first from 1 (if `start`
    is not specified) and include the `stop` element (if the `step` parameter
    allows).

    Args:
        first (int): The first value of the span.
            If `second` is None, the `start` value is 1,
            and this is the `stop` value.
            It is included if `step` is a multiple of the sequence length.
            Otherwise, this is the `start` value and it is always included.
        second (int|None): The second value of the span.
            If None, the start value is 1 and this parameter is ignored.
            Otherwise, this is the `stop` value of the range.
            It is included if `step` is a multiple of the sequence length.
            If `first < second` the sequence is yielded backwards.
        step (int): The step of the rows range.
            If `start > stop`, the step parameter should be negative in order
            to obtain a non-empty range.
            If None, this is computed automatically based on `first` and
            `second`, such that a non-empty sequence is avoided, if possible,
            i.e. `step == 1` if `start <= stop` else `step == -1`.

    Returns:
        result (range): The spanned range.

    Examples:
        >>> print(list(span(10)))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> print(list(span(-10)))
        [1, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
        >>> print(list(span(1)))
        [1]
        >>> print(list(span(1, 10)))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> print(list(span(-1, 10)))
        [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> print(list(span(-1, 9, 2)))
        [-1, 1, 3, 5, 7, 9]
        >>> print(list(span(-1, 10, 2)))
        [-1, 1, 3, 5, 7, 9]
        >>> print(list(span(10, 1)))
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        >>> print(list(span(10, 1, -2)))
        [10, 8, 6, 4, 2]
        >>> print(list(span(-1, -10, -2)))
        [-1, -3, -5, -7, -9]
        >>> print(list(span(-1, -11, -2)))
        [-1, -3, -5, -7, -9, -11]

        >>> 1 - span - 10 == span(1, 10)
    """
    if second is None:
        start, stop = 1, first
    else:
        start, stop = first, second
    if not step:
        step = 1 if start <= stop else -1
    stop = stop + (step if not ((start - stop) % step) else 0)
    return range(start, stop, step)


# ======================================================================
def any_range(
        start,
        stop,
        step=1,
        zero=0,
        adapt_step_sign=True):
    """
    Generate a range of objects.

    It is a generalization of `range()` built-in to non-integer arguments.
    The objects must be ordered and support addition and subtraction.

    Args:
        start (Any): The initial value (included).
        stop (Any): The final value (excluded).
        step (Any): The step value.
        zero (Any): The "zero" value (used to determine the sign of `step`).
        adapt_step_sign (bool): Swap the sign of `step`.
            This is used to ensure a non-empty range.

    Yields:
        The value(s) between `start` and `stop` (excluded) in `step` steps.

    Raises:
        ValueError: If the step is zero.
        
    Examples:
        >>> list(any_range(0, 4))
        [0, 1, 2, 3]
        >>> list(any_range(4, 0))
        [4, 3, 2, 1]
        >>> list(any_range(0, 4, 2))
        [0, 2]
        >>> list(any_range(4, 0, 2))
        [4, 2]
        >>> list(any_range(4, 0, adapt_step_sign=False))
        []
        >>> list(any_range(4, 0, -2))
        [4, 2]
        >>> list(any_range(4, 0, -2, adapt_step_sign=False))
        [4, 2]
    """
    if adapt_step_sign and (
            ((start < stop) and (step < zero))
            or ((start > stop) and (step > zero))):
        step = -step
    if step > zero:
        comparer = operator.lt
    elif step < zero:
        comparer = operator.gt
    else:
        raise ValueError

    curr = start
    while comparer(curr, stop):
        yield curr
        curr += step


# ======================================================================
def is_deep(
        obj,
        skip=(str, bytes, bytearray)):
    """
    Determine if an object is deep, i.e. it can be iterated through.

    Args:
        obj (Any): The object to test.
        skip (tuple|None): Types to skip descending into.

    Returns:
        result (bool): If the object is deep or not.

    Examples:
        >>> is_deep(1)
        False
        >>> is_deep(())
        True
        >>> is_deep([1, 2, 3])
        True
        >>> is_deep(range(4))
        True
        >>> is_deep((i for i in range(4)))
        True
        >>> is_deep('ciao')
        False
        >>> is_deep('c')
        False
        >>> is_deep('')
        False

        >>> is_deep(1, skip=None)
        False
        >>> is_deep('c', skip=None)
        True
        >>> is_deep('', skip=None)
        True
    """
    return hasattr(obj, '__iter__') and not (skip and isinstance(obj, skip))


# ======================================================================
def freeze(
        items,
        max_depth=-1,
        skip=(str, bytes, bytearray)):
    """
    Recursively convert mutable containers to immutable counterparts.

    The following conversions are performed:
     - Iterable -> tuple
     - set -> frozenset
     - dict (any container with `items()` method) -> tuple

    This is useful to sanitize default parameters in functions.

    Args:
        items: The input items.
        max_depth (int): Maximum depth to reach. Negative for unlimited.
        skip (tuple|None): Types to skip descending into.

    Returns:
        result (tuple): The output items.

    Examples:
        >>> freeze([1, 2, 'ciao', [2, 3]])
        (1, 2, 'ciao', (2, 3))
        >>> freeze([1, 2, 'ciao', {2, 3}])
        (1, 2, 'ciao', frozenset({2, 3}))
        >>> freeze([1, 2, 'ciao', {2: 3}])
        (1, 2, 'ciao', ((2, 3),))
        >>> freeze([1, 2, 'ciao', {2: 3}, {4: 5}])
        (1, 2, 'ciao', ((2, 3),), ((4, 5),))
    """
    if max_depth == 0:
        return items
    else:
        if hasattr(items, 'items'):
            return tuple(
                freeze(item, max_depth - 1, skip)
                if is_deep(item, skip) and item != items else item
                for item in items.items())
        else:
            container = frozenset if isinstance(items, set) else tuple
            return container(
                freeze(item, max_depth - 1, skip)
                if is_deep(item, skip) and item != items else item
                for item in items)


# ======================================================================
def unfreeze(
        items,
        max_depth=-1,
        skip=(str, bytes, bytearray)):
    """
    Recursively convert immutable containers to mutable counterparts.

    The following conversions are performed:
     - Iterable -> dict (if possible) or list
     - frozenset -> set

    Args:
        items: The input items.
        max_depth (int): Maximum depth to reach. Negative for unlimited.
        skip (tuple|None): Types to skip descending into.

    Returns:
        result (tuple): The output items.

    Examples:
        >>> unfreeze((1, 2, 'ciao', (2, 3)))
        [1, 2, 'ciao', [2, 3]]
        >>> unfreeze((1, 2, 'ciao', frozenset({2, 3})))
        [1, 2, 'ciao', {2, 3}]
        >>> unfreeze((1, 2, 'ciao', ((2, 3),)))
        [1, 2, 'ciao', {2: 3}]
        >>> unfreeze((1, 2, 'ciao', ((2, 3),), ((4, 5),)))
        [1, 2, 'ciao', {2: 3}, {4: 5}]
    """
    if max_depth == 0:
        return items
    else:
        try:
            return dict(
                unfreeze(item, max_depth - 1, skip)
                if is_deep(item, skip) and item != items else item
                for item in items)
        except TypeError:
            container = set if isinstance(items, frozenset) else list
            return container(
                unfreeze(item, max_depth - 1, skip)
                if is_deep(item, skip) and item != items else item
                for item in items)


# ======================================================================
def is_mutable(
        obj,
        max_depth=-1):
    """
    Recursively determine if an object is mutable.

    This is useful to inspect whether it is safe to use an object as default
    parameter value.

    Args:
        obj (Any): The object to inspect.
        max_depth (int): Maximum depth to reach. Negative for unlimited.

    Returns:
        result (bool): The result of the mutability check.

    Examples:
        >>> is_mutable(1)
        False
        >>> is_mutable([1])
        True
        >>> is_mutable((1, 2))
        False
        >>> is_mutable((1, 2, []))
        True
        >>> is_mutable((1, 2, []), max_depth=0)
        False
        >>> is_mutable(zip(([], []), ((), ())))
        True
        >>> is_mutable(zip(((), ()), ((), ())))
        False
        >>> is_mutable(map(lambda x: [], (1, 2)))
        True
        >>> is_mutable(map(lambda x: 'ciao', (1, 2)))
        False
        >>> is_mutable(filter(lambda x: len(x) >= 0, ((), (), [])))
        True
        >>> is_mutable(filter(lambda x: len(x) >= 0, ((), ())))
        False
    """
    shallow_immutables = (
        bool, int, float, str, bytes, slice, range, frozenset)
    if isinstance(obj, shallow_immutables):
        return False
    elif isinstance(obj, (tuple, zip, map, filter)):
        if max_depth != 0:
            for item in obj:
                if is_mutable(item, max_depth - 1):
                    return True
            else:
                return False
        else:
            return False
    else:
        return True


# ======================================================================
def multi_compare(
        grouper,
        items,
        comparison=operator.eq,
        symmetric=True):
    """
    Compute multiple comparisons.

    Args:
        grouper (callable): Determine how to group multiple comparisons.
            Must accept the following signature:
            multi_comparison(*Iterable[bool]): bool
            Can be either `all` or `any`, or any callable with the supported
            signature.
        items (Iterable): The input items.
        comparison (callable): Compute pair-wise comparison.
            Must accept the following signature:
            comparison(Any, Any): bool
        symmetric (bool): Assume that the comparison is symmetric.
            A comparison is symmetric if:
            comparison(a, b) == comparison(b, a).

    Returns:
        result (bool): The result of the multiple comparisons.

    Examples:
        >>> multi_compare(all, [1, 1, 1, 1, 1, 1, 1, 1, 1])
        True
        >>> multi_compare(any, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
        True
        >>> multi_compare(all, 'xxxxx')
        True
        >>> multi_compare(all, 'yxxxy')
        False
        >>> multi_compare(any, 'xxxxxyxxxy')
        True
        >>> all_equal({0, 1})
        False
    """
    pairwise = itertools.combinations(items, 2) \
        if symmetric else itertools.permutations(items, 2)
    return grouper(comparison(a, b) for a, b in pairwise)


# ======================================================================
def in_(
        obj,
        items,
        eq=operator.eq):
    """
    Check if an object is equal to any of the items.

    This is useful if any among the object or the items does not implement
    `__eq__` to return a `bool` for any given input.
    Note that this operation is always of O(N) complexity
    (N being the `items` size).

    Args:
        obj (Any): The object to check.
        items (Iterable): The container to check.
        eq (callable): The function to check for equality.
            Must have the following signature: eq(Any, Any): bool

    Returns:
        bool: The result of the check.

    Examples:
        >>> in_(1, [1, 2, 3])
        True
        >>> in_(0, [1, 2, 3])
        False
        >>> in_(1, {1, 2, 3})
        True
        >>> in_(0, {1, 2, 3})
        False
    """
    for item in items:
        if eq(obj, item):
            return True
    return False


# ======================================================================
def all_equal(items):
    """
    Check if all items are equal.

    Args:
        items (Iterable): The input items.

    Returns:
        result (bool): The result of the equality test.

    Examples:
        >>> all_equal([1, 1, 1, 1, 1, 1, 1, 1, 1])
        True
        >>> all_equal([1, 1, 1, 1, 0, 1, 1, 1, 1])
        False
        >>> all_equal('xxxxx')
        True
        >>> all_equal('xxxxy')
        False
        >>> all_equal({0})
        True
        >>> all_equal({0, 1})
        False
        >>> all(all_equal(x) for x in ((), [], {}, set()))
        True
    """
    if isinstance(items, collections.abc.Sequence):
        return items[1:] == items[:-1]
    else:
        iter_items = iter(items)
        try:
            first = next(iter_items)
        except StopIteration:
            return True
        # : slower alternative
        # return all(first == item for item in iter_items)
        for item in iter_items:
            if first != item:
                return False
        return True


# ======================================================================
def nesting_level(
        obj,
        deep=True,
        max_depth=-1,
        combine=max,
        skip=(str, bytes, bytearray)):
    """
    Compute the nesting level of nested iterables.

    Args:
        obj (Any): The object to test.
        deep (bool): Evaluate all item.
            If True, all elements within `obj` are evaluated.
            If False, only the first element of each deep object is evaluated.
            An object is considered deep using `is_deep()`.
        max_depth (int): Maximum depth to reach. Negative for unlimited.
        combine (callable): Combine multiple depth at the same level.
            If `deep` is False, this parameter is ignored.
        skip (tuple|None): Types to skip descending into.

    Returns:
        result (int): The nesting level.

    Examples:
        >>> nesting_level([])
        1
        >>> nesting_level([], True)
        1
        >>> nesting_level([[]], False)
        2
        >>> nesting_level(
        ...     [[(1, 2, 3), (4, 5)], [(1, 2), (3,)], ['1,2', [6, 7]]], True)
        3
        >>> nesting_level(
        ...     [[1, (2, 3), (4, 5)], [(1, 2), (3,)], ['1,2', [6, 7]]], False)
        2
        >>> nesting_level(
        ...     [1, [[[[[[[[[[[[[[[[[[[[[5]]]]]]]]]]]]]]]]]]]]]], True)
        22
        >>> nesting_level(
        ...     [1, [[[[[[[[[[[[[[[[[[[[[5]]]]]]]]]]]]]]]]]]]]]], False)
        1
        >>> nesting_level(((1, 2), 1, (1, (2, 3))), True)
        3
        >>> nesting_level(((1, 2), 1, (1, (2, 3))), True, combine=min)
        1
    """
    if not is_deep(obj, skip):
        return 0
    elif len(obj) == 0:
        return 1
    elif max_depth == 0:
        return 1
    else:
        if deep:
            next_level = combine(
                nesting_level(x, deep, max_depth - 1, combine, skip)
                for x in obj)
        else:
            next_level = nesting_level(
                obj[0], deep, max_depth - 1, combine, skip)
        return 1 + next_level


# ======================================================================
def nested_len(
        obj,
        deep=True,
        max_depth=-1,
        combine=max,
        check_same=True,
        skip=(str, bytes, bytearray)):
    """
    Compute the length of nested iterables.

    Args:
        obj (Any): The object to test.
        deep (bool): Evaluate all item.
            If True, all elements within `obj` are evaluated.
            If False, only the first element of each deep object is evaluated.
            An object is considered deep using `is_deep()`.
        max_depth (int): Maximum depth to reach. Negative for unlimited.
        combine (callable|None): Combine multiple depth at the same level.
            If None, the lengths do not get combined (using `combine=tuple`
            has the same effect).
            If `deep` is False, this parameter is ignored.
        check_same (bool): Check that same-level items have the same length.
        skip (tuple|None): Types to skip descending into.

    Returns:
        result (tuple[int]): The length of the nested iterables.

    Raises:
        ValueError: if `check_same` is True and items at the same nesting level
            have different length.

    Examples:
        >>> nested_len(((1, 1), (1, 1), (1, 1)))
        (3, 2)
        >>> nested_len(((1, 2), (1,), (1, (2, 3))), check_same=True)
        Traceback (most recent call last):
            ...
        ValueError: Same nesting level items with different length.
        >>> nested_len(((1, 2), (1,), (1, (2, 3))), check_same=False)
        (3, 2, 2)
        >>> nested_len(((1, 2), (1, 2), (1, 2)), check_same=False)
        (3, 2)
        >>> nested_len(((1,), (1,), (1,)))
        (3, 1)
        >>> nested_len((1, 2, 3))
        (3,)
        >>> nested_len(
        ...     ((1, 2), (1,), (1, (2, 3))), combine=None, check_same=False)
        (3, (2,), (1,), (2, (2,)))
        >>> nested_len(
        ...     ((1, 2), (1,), ((1,), (2, 3))), combine=None, check_same=False)
        (3, (2,), (1,), (2, (1,), (2,)))
        >>> nested_len(
        ...     ((1, 2), (1,), ((1, (2, 3)),)), combine=None, check_same=False)
        (3, (2,), (1,), (1, (2, (2,))))
    """
    if not is_deep(obj, skip):
        return ()
    else:
        if deep:
            next_level = tuple(
                nested_len(
                    x, deep, max_depth - 1, combine, check_same, skip)
                for x in obj)
            if check_same and not all_equal(next_level):
                raise ValueError(
                    'Same nesting level items with different length.')
            if not callable(combine):
                combine = tuple
            next_level = combine(next_level)
        else:
            next_level = nested_len(
                next(iter(obj)), deep, max_depth - 1, combine,
                check_same, skip)
        return (len(obj),) + tuple(x for x in next_level if x)


# ======================================================================
def auto_repeat(
        obj,
        n,
        force=False,
        check=False):
    """
    Automatically repeat the specified object n times.

    If the object is not Iterable, a tuple with the specified size is returned.
    If the object is Iterable, the object is repeated only if `n` is Iterable,
    in which case it is repeated accordingly.
    The resulting length depends on the value of `force`.

    Args:
        obj: The object to operate with.
        n (int|Sequence[int]): The length(s) of the output object.
            If Sequence, multiple nested tuples will be generated.
        force (bool): Force the repetition, even if the object is Iterable.
            If True, the nested length of the result is equal to `n` plus the
            length of the input, otherwise it is equal to `n` (but the last
            value) plus the length of the input.
        check (bool): Ensure that the object has length n.
            More precisely the `n` and the initial part of
            `nested_len(check_same=True)` must be identical.

    Returns:
        result (tuple): Returns obj repeated n times.

    Raises:
        AssertionError: If force is True and the object does not have length n.

    Examples:
        >>> auto_repeat(1, 3)
        (1, 1, 1)
        >>> auto_repeat([1], 3)
        [1]
        >>> auto_repeat([1, 3], 2)
        [1, 3]
        >>> auto_repeat([1, 3], 2, True)
        ([1, 3], [1, 3])
        >>> auto_repeat([1, 2, 3], 2, True, True)
        ([1, 2, 3], [1, 2, 3])
        >>> auto_repeat([1, 2, 3], 2, False, True)
        Traceback (most recent call last):
            ...
        ValueError: Incompatible input value length.
        >>> auto_repeat(1, (3,))
        (1, 1, 1)
        >>> auto_repeat(1, (3, 2))
        ((1, 1), (1, 1), (1, 1))
        >>> auto_repeat(1, (2, 3))
        ((1, 1, 1), (1, 1, 1))
        >>> auto_repeat([1], (3, 1), False, True)
        ([1], [1], [1])
        >>> auto_repeat([1], (3, 1), True, True)
        (([1],), ([1],), ([1],))
        >>> auto_repeat([1], (3, 1), False, False)
        ([1], [1], [1])
        >>> auto_repeat([1], (3, 1), True, False)
        (([1],), ([1],), ([1],))
        >>> auto_repeat([1], (3, 3), False, False)
        ([1], [1], [1])
        >>> auto_repeat([1], (3, 3), True, False)
        (([1], [1], [1]), ([1], [1], [1]), ([1], [1], [1]))
        >>> auto_repeat([1], (3, 3), True, True)
        (([1], [1], [1]), ([1], [1], [1]), ([1], [1], [1]))
        >>> auto_repeat([1], (3, 3), False, True)
        Traceback (most recent call last):
            ...
        ValueError: Incompatible input value length.
        >>> auto_repeat(((1, 1), (1, 1), (1, 1)), (3, 2), False, True)
        ((1, 1), (1, 1), (1, 1))
        >>> auto_repeat((1, 1), (3, 2), False, True)
        ((1, 1), (1, 1), (1, 1))
        >>> auto_repeat((1, 1, 1), (3, 2), False, True)
        Traceback (most recent call last):
            ...
        ValueError: Incompatible input value length.
        >>> auto_repeat((1, 1, 1), (3, 2), False, False)
        ((1, 1, 1), (1, 1, 1), (1, 1, 1))
        >>> auto_repeat((1, 1), (3, 2), True, True)
        (((1, 1), (1, 1)), ((1, 1), (1, 1)), ((1, 1), (1, 1)))

    See Also:
        - flyingcircus.stretch()
    """
    force = force or not hasattr(obj, '__iter__')
    result = obj
    if isinstance(n, int):
        if force:
            result = (obj,) * n
        if check and len(result) != n:
            raise ValueError('Incompatible input value length.')
    else:
        if nested_len(obj, check_same=True) != n or force:
            result = auto_repeat(obj, n[-1], force, check)
            for i in n[-2::-1]:
                result = auto_repeat(result, i, True, check)
        if check and \
                nested_len(result, check_same=True) \
                != n + (nested_len(obj, check_same=True) if force else ()):
            raise ValueError('Incompatible input value length.')
    return result


# ======================================================================
def stretch(
        items,
        shape,
        skip=None):
    """
    Automatically stretch the values to the target shape.

    This is similar to `flyingcircus.auto_repeat()`, except that it
    can flexibly repeat values only when needed.
    This is similar to shape broadcasting of multi-dimensional arrays.

    Args:
        items (Any|Iterable): The input items.
        shape (Sequence[int]): The target shape (nested lengths).
        skip (tuple|None): Types to skip descending into.

    Returns:
        result (tuple): The values stretched to match the target shape.

    Raises:
        ValueError: If `items` and `shape` are incompatible.

    Examples:
        >>> stretch(1, 10)
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1)
        >>> stretch(1, (2, 3))
        ((1, 1, 1), (1, 1, 1))
        >>> stretch(1, (2, 3, 2))
        (((1, 1), (1, 1), (1, 1)), ((1, 1), (1, 1), (1, 1)))
        >>> stretch((1, (2, 3)), (2, 2))
        ((1, 1), (2, 3))
        >>> stretch((1, (2, 3, 4)), (2, 3))
        ((1, 1, 1), (2, 3, 4))
        >>> stretch((1, (2, 3,)), (2, 3))
        Traceback (most recent call last):
            ...
        ValueError: Cannot stretch `(2, 3)` to `(3,)`.
        >>> stretch((1, (2, 3, 4, 5)), (2, 3))
        Traceback (most recent call last):
            ...
        ValueError: Cannot stretch `(2, 3, 4, 5)` to `(3,)`.
        >>> stretch(((1, 2),), (4, 2))
        ((1, 2), (1, 2), (1, 2), (1, 2))
        >>> stretch((1, 2, 3, 4), (4, 2))
        ((1, 1), (2, 2), (3, 3), (4, 4))
        >>> items = [[[[1], [2], [3]]], [[[4], [5], [6]]]]
        >>> print(nested_len(items))
        (2, 1, 3, 1)
        >>> stretch(items, (2, 4, 3, 4))
        ((((1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3)),\
 ((1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3)),\
 ((1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3)),\
 ((1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3))),\
 (((4, 4, 4, 4), (5, 5, 5, 5), (6, 6, 6, 6)),\
 ((4, 4, 4, 4), (5, 5, 5, 5), (6, 6, 6, 6)),\
 ((4, 4, 4, 4), (5, 5, 5, 5), (6, 6, 6, 6)),\
 ((4, 4, 4, 4), (5, 5, 5, 5), (6, 6, 6, 6))))

    See Also:
        - flyingcircus.auto_repeat()
    """
    if not is_deep(items, skip):
        result = auto_repeat(items, shape)
    else:
        old_shape = nested_len(items, check_same=False)
        if len(old_shape) == 1 and len(shape) > 1 and shape[0] == old_shape[0]:
            result = tuple(
                auto_repeat(item, shape[1:], True, True)
                for item in items)
        elif old_shape[0] == 1:
            result = tuple(
                stretch(item, shape[1:]) if shape[1:] else item
                for item in items) * shape[0]
        elif old_shape == shape:
            try:
                nested_len(items, check_same=True)
            except ValueError:
                result = tuple(
                    stretch(item, shape[1:])
                    if is_deep(item, skip)
                    else auto_repeat(item, shape[1:])
                    for item in items)
            else:
                result = items
        elif old_shape[0] == shape[0]:
            result = tuple(
                stretch(item, shape[1:])
                if is_deep(item, skip)
                else auto_repeat(item, shape[1:])
                for item in items)
        else:
            raise ValueError(
                'Cannot stretch `{}` to `{}`.'.format(items, shape))
    return result


# ======================================================================
def flatten(
        items,
        max_depth=-1,
        shallow=(str, bytes, bytearray)):
    """
    Recursively flattens nested Iterables.

    The maximum depth is limited by Python's recursion limit.

    Args:
        items (Iterable): The input items.
        max_depth (int): Maximum depth to reach. Negative for unlimited.
        shallow (tuple|None): Data types to always consider shallow.
            Note that recursive and self-slicing objects are handled
            separately.

    Yields:
        item (any): The next non-Iterable item of the flattened items.

    Examples:
        >>> ll = [[1, 2, 3], [4, 5, 6], [7], [8, 9]]
        >>> list(flatten(ll))
        [1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> list(flatten(ll)) == list(itertools.chain.from_iterable(ll))
        True
        >>> ll = [ll, ll]
        >>> list(flatten(ll))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> list(itertools.chain.from_iterable(ll))
        [[1, 2, 3], [4, 5, 6], [7], [8, 9], [1, 2, 3], [4, 5, 6], [7], [8, 9]]
        >>> list(flatten([1, 2, 3]))
        [1, 2, 3]
        >>> list(flatten(['best', ['function', 'ever']]))
        ['best', 'function', 'ever']
        >>> ll2 = [[(1, 2, 3), (4, 5)], [(1, 2), (3, 4, 5)], ['1, 2', [6, 7]]]
        >>> list(flatten(ll2))
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, '1, 2', 6, 7]
        >>> list(flatten(ll2, shallow=(tuple, str)))
        [(1, 2, 3), (4, 5), (1, 2), (3, 4, 5), '1, 2', 6, 7]
        >>> list(flatten(ll2, max_depth=1))
        [(1, 2, 3), (4, 5), (1, 2), (3, 4, 5), '1, 2', [6, 7]]
        >>> list(flatten(ll2, shallow=None))
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, '1', ',', ' ', '2', 6, 7]
        >>> list(
        ...     flatten([['best', 'func'], 'ever'], 1,
        ...     shallow=None))
        ['best', 'func', 'e', 'v', 'e', 'r']
        >>> list(
        ...     flatten([['best', 'func'], 'ever'],
        ...     shallow=None))
        ['b', 'e', 's', 't', 'f', 'u', 'n', 'c', 'e', 'v', 'e', 'r']
        >>> list(flatten(list(range(10))))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> list(flatten(range(10)))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> list(flatten([[0, 1], range(2, 5), (i for i in range(5, 10))]))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    for item in items:
        if shallow and isinstance(item, shallow) or \
                item == items or max_depth == 0:
            yield item
        else:
            try:
                for subitem in flatten(item, max_depth - 1, shallow):
                    yield subitem
            except TypeError:
                yield item


# ======================================================================
def transpose(
        items,
        longest=False,
        fill=None):
    """
    Transpose an iterable of iterables.

    Args:
        items (Iterable[Iterable]): The input items.
        longest (bool): Align to longest inner iterable.
            If True, aligns to the longest inner iterable from the input.
            If False, cuts the inner iterables to the shortest.
        fill (Any): Fill value to use for aligning to longest.
            If `longest` is False, this parameter has no effect.

    Returns:
        result (zip|itertools.zip_longest): The transposed items.

    Examples:
        >>> tuple(transpose(((1, 2), (3, 4), (5, 6))))
        ((1, 3, 5), (2, 4, 6))
        >>> tuple(transpose(((1, 2), (3, 4), (5, 6, 7))))
        ((1, 3, 5), (2, 4, 6))
        >>> tuple(transpose(((1, 2), (3, 4), (5, 6, 7)), True))
        ((1, 3, 5), (2, 4, 6), (None, None, 7))
    """
    if longest:
        return itertools.zip_longest(*items, fillvalue=fill)
    else:
        return zip(*items)


# ======================================================================
def flip_slice(
        obj,
        force_step=False):
    """
    Flip a slice or range.

    This is achieved by swapping its `start` and `stop` attributes.
    It works for any object implementing `start` and `stop` attributes.

    If `step` is specified and not None, `-step` is used as new `step`.

    Args:
        obj (slice|range): The input slice/range.
        force_step (bool): Force producing a slice with an explicit step.
            If True, `1` is used as `step` in place of `None`.

    Returns:
        obj (slice|range): The output slice/range.

    Examples:
        >>> flip_slice(slice(10, 20))
        slice(20, 10, None)
        >>> flip_slice(slice(10, 20, 2))
        slice(20, 10, -2)
        >>> flip_slice(slice(20, 10, -2))
        slice(10, 20, 2)
        >>> flip_slice(slice(10, 20), True)
        slice(20, 10, -1)
        >>> flip_slice(slice(20, 10))
        slice(10, 20, None)
        >>> flip_slice(slice(20, 10), True)
        slice(10, 20, 1)
        >>> flip_slice(slice(None, 10))
        slice(10, None, None)
        >>> flip_slice(slice(None, 10), True)
        slice(10, None, 1)
        >>> flip_slice(slice(20, None))
        slice(None, 20, None)
        >>> flip_slice(slice(20, None), True)
        slice(None, 20, 1)
        >>> flip_slice(slice(None, None))
        slice(None, None, None)
        >>> flip_slice(slice(None, None), True)
        slice(None, None, 1)

        >>> flip_slice(range(10, 20))
        range(20, 10, -1)
        >>> flip_slice(range(10, 20, 2))
        range(20, 10, -2)
        >>> flip_slice(range(20, 10, -2))
        range(10, 20, 2)
        >>> flip_slice(range(10, 20), True)
        range(20, 10, -1)
        >>> flip_slice(range(20, 10))
        range(10, 20, -1)
        >>> flip_slice(range(20, 10), True)
        range(10, 20, -1)
    """
    if hasattr(obj, 'step'):
        if obj.step is None and force_step:
            if all(x is not None for x in (obj.start, obj.stop)) \
                    and obj.start > obj.stop:
                step = -1
            else:
                step = 1
        else:
            step = obj.step
        if all(x is not None for x in (obj.start, obj.stop, step)):
            step = -step
        return type(obj)(obj.stop, obj.start, step)
    else:
        return type(obj)(obj.stop, obj.start)


# ======================================================================
def complement(
        seq,
        slice_):
    """
    Extract the elements not matching a given slice.

    Args:
        seq (Sequence): The input items.
        slice_ (slice): The slice to be complemented.

    Yields:
        item (Any): The next item not matching the slice pattern.

    Examples:
        >>> items = tuple(range(10))

        >>> s = slice(None)
        >>> print(items[s], tuple(complement(items, s)))
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) ()

        >>> s = slice(None, None, 2)
        >>> print(items[s], tuple(complement(items, s)))
        (0, 2, 4, 6, 8) (1, 3, 5, 7, 9)

        >>> s = slice(None, None, 3)
        >>> print(items[s], tuple(complement(items, s)))
        (0, 3, 6, 9) (1, 2, 4, 5, 7, 8)

        >>> s = slice(2, None, None)
        >>> print(items[s], tuple(complement(items, s)))
        (2, 3, 4, 5, 6, 7, 8, 9) (0, 1)

        >>> s = slice(None, 7, None)
        >>> print(items[s], tuple(complement(items, s)))
        (0, 1, 2, 3, 4, 5, 6) (7, 8, 9)

        >>> s = slice(2, None, 3)
        >>> print(items[s], tuple(complement(items, s)))
        (2, 5, 8) (0, 1, 3, 4, 6, 7, 9)

        >>> s = slice(None, 7, 3)
        >>> print(items[s], tuple(complement(items, s)))
        (0, 3, 6) (1, 2, 4, 5, 7, 8, 9)

        >>> s = slice(2, 7, 3)
        >>> print(items[s], tuple(complement(items, s)))
        (2, 5) (0, 1, 3, 4, 6, 7, 8, 9)

        >>> s = slice(None, None, -3)
        >>> print(items[s], tuple(complement(items, s)))
        (9, 6, 3, 0) (8, 7, 5, 4, 2, 1)

        >>> s = slice(2, None, -3)
        >>> print(items[s], tuple(complement(items, s)))
        (2,) (9, 8, 7, 6, 5, 4, 3, 1, 0)

        >>> s = slice(None, 7, -3)
        >>> print(items[s], tuple(complement(items, s)))
        (9,) (8, 7, 6, 5, 4, 3, 2, 1, 0)

        >>> s = slice(2, 7, -3)
        >>> print(items[s], tuple(complement(items, s)))
        () (9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

        >>> s = slice(7, 2, -3)
        >>> print(items[s], tuple(complement(items, s)))
        (7, 4) (9, 8, 6, 5, 3, 2, 1, 0)

        >>> items = tuple(1 for i in range(10))
        >>> s = slice(None, None, 3)
        >>> print(items[s], tuple(complement(items, s)))
        (1, 1, 1, 1) (1, 1, 1, 1, 1, 1)

        >>> items = tuple(i % 2 for i in range(10))
        >>> s = slice(None, None, 3)
        >>> print(items[s], tuple(complement(items, s)))
        (0, 1, 0, 1) (1, 0, 0, 1, 1, 0)

        >>> ll = list(range(1000))
        >>> vals = (3, 5, 7, 17, 101)
        >>> vals += tuple(-x for x in vals) + (None,)
        >>> print(vals)
        (3, 5, 7, 17, 101, -3, -5, -7, -17, -101, None)
        >>> sls = [slice(*x) for x in itertools.product(vals, vals, vals)]
        >>> all(
        ...     set(complement(ll, sl)).intersection(ll[sl]) == set()
        ...     for sl in sls)
        True
    """
    to_exclude = set(range(len(seq))[slice_])
    step = slice_.step if slice_.step else 1
    if step > 0:
        for i, item in enumerate(seq):
            if i not in to_exclude:
                yield item
    else:
        num_items = len(seq)
        for i, item in enumerate(reversed(seq)):
            if (num_items - i - 1) not in to_exclude:
                yield item


# ======================================================================
def conditional_apply(
        func,
        condition=None):
    """
    Modify a function so that it is applied only if a condition is satisfied.

    Args:
        func (callable): A function to apply to an object.
            Must have the following signature: func(Any): Any
        condition (callable|None): The condition function.
            If not None, the function `func` is applied to an object only
            if the condition on the object evaluates to True.
            Must have the following signature: condition(Any): bool

    Returns:
        result (callable): The conditional function.

    Examples:
        >>> conditional_apply(str)(1)
        '1'
        >>> list(map(conditional_apply(str), range(10)))
        ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        >>> [conditional_apply(str, lambda x: x > 5)(i) for i in range(10)]
        [0, 1, 2, 3, 4, 5, '6', '7', '8', '9']
        >>> list(map(conditional_apply(str, lambda x: x > 5), range(10)))
        [0, 1, 2, 3, 4, 5, '6', '7', '8', '9']
        >>> print(conditional_apply(str) == str)
        True
    """
    if callable(condition):
        return lambda x: func(x) if condition(x) else x
    else:
        return func


# ======================================================================
def deep_map(
        func,
        items,
        container=None,
        max_depth=-1,
        skip=(str, bytes, bytearray)):
    """
    Compute a function on each element of a nested structure of iterables.

    The result preserves the nested structure of the input.

    Args:
        func (callable): The function to apply to the individual item.
            Must have the following signature: func(Any): Any.
        items (Iterable): The input items.
        container (callable|None): The container for the result.
            If None, this is inferred from `seq` if possible, otherwise
            uses `tuple`.
        max_depth (int): Maximum depth to reach. Negative for unlimited.
        skip (tuple|None): Types to skip descending into.

    Returns:
        new_items (Iterable): The mapped items.

    Examples:
        >>> items = [1, 2, [3, 4, 5], 2, (3, [4, 5])]
        >>> print(deep_map(str, items))
        ['1', '2', ['3', '4', '5'], '2', ('3', ['4', '5'])]
        >>> print(deep_map(float, items))
        [1.0, 2.0, [3.0, 4.0, 5.0], 2.0, (3.0, [4.0, 5.0])]
        >>> print(deep_map(lambda x: x, items))
        [1, 2, [3, 4, 5], 2, (3, [4, 5])]
        >>> print(deep_map(lambda x: x, items, tuple))
        (1, 2, (3, 4, 5), 2, (3, (4, 5)))
    """
    final_container = type(items) if container is None else container
    if not callable(final_container):
        final_container = tuple

    # note: this cannot be rewritten as generator because of recursion!

    # : alternate implementation (slower)
    # if is_deep(items, skip):
    #     return final_container(
    #         deep_map(func, item, container, avoid, max_depth - 1)
    #         for item in items)
    # else:
    #     return final_container(func(items))

    # : alternate implementation (slower)
    # return final_container(
    #     deep_map(func, item, container, avoid, max_depth - 1)
    #     if is_deep(item, skip) else func(item)
    #     for item in items)

    new_items = []
    for item in items:
        if is_deep(item, skip):
            new_items.append(
                deep_map(func, item, container, max_depth - 1, skip))
        else:
            new_items.append(func(item))
    return final_container(new_items)


# ======================================================================
def deep_filter(
        func,
        items,
        container=None,
        max_depth=-1,
        is_deep_kws=None):
    """
    Filter the elements from a nested structure of iterables.

    The result preserves the nested structure of the input.

    Args:
        func (callable): The condition function to include the individual item.
            Must have the following signature: func(Any): bool
        items (Iterable): The input items.
        container (callable|None): The container for the result.
            If None, this is inferred from `seq` if possible, otherwise
            uses `tuple`.
        max_depth (int): Maximum depth to reach. Negative for unlimited.
        is_deep_kws (Mapping|None): Keyword parameters for `is_deep()`.
            These are passed to `flyingcircus.is_deep()`.

    Returns:
        new_items (Iterable): The filtered items.

    Examples:
        >>> items = [1, 2, [3, 4, 5], 2, (3, [4, 5])]
        >>> print(deep_filter(lambda x: x > 1, items))
        [2, [3, 4, 5], 2, (3, [4, 5])]
        >>> print(deep_filter(lambda x: x > 2, items))
        [[3, 4, 5], (3, [4, 5])]
        >>> print(deep_filter(lambda _: True, items))
        [1, 2, [3, 4, 5], 2, (3, [4, 5])]
    """
    is_deep_kws = {} if is_deep_kws is None else dict(is_deep_kws)
    final_container = type(items) if container is None else container
    if not callable(final_container):
        final_container = tuple

    # note: this cannot be rewritten as generator because of recursion!

    # : alternate implementation (slower)
    # if is_deep(items, **is_deep_kws):
    #     return final_container(
    #         deep_filter(func, item, container, avoid, max_depth - 1)
    #         for item in items
    #         if is_deep(item, **is_deep_kws) or func(item))
    # else:
    #     return final_container(items)

    # : alternate implementation (slower)
    # return final_container(
    #     deep_filter(func, item, container, avoid, max_depth - 1)
    #     if is_deep(item, **is_deep_kws) else item
    #     for item in items if is_deep(item, **is_deep_kws) or func(item))

    new_items = []
    for item in items:
        if is_deep(item, **is_deep_kws):
            new_items.append(
                deep_filter(func, item, container, max_depth - 1, is_deep_kws))
        else:
            if func(item):
                new_items.append(item)
    return final_container(new_items)


# ======================================================================
def deep_convert(
        container,
        items,
        max_depth=-1,
        skip=(str, bytes, bytearray)):
    """
    Convert the containers from a nested structure of iterables.

    Args:
        container (callable|None): The container to apply.
            Must have the following signature:
            Must have the following signature:
            container(Iterable): container.
            If None, no conversion is performed.
        items (Iterable): The input items.
        max_depth (int): Maximum depth to reach. Negative for unlimited.
        skip (tuple|None): Types to skip descending into.

    Returns:
        new_items (container): The converted nested structure of iterables.

    Examples:
        >>> items = [1, 2, [3, 4, 5], 2, (3, [4, 5], 'ciao')]
        >>> deep_convert(list, items)
        [1, 2, [3, 4, 5], 2, [3, [4, 5], 'ciao']]
        >>> deep_convert(tuple, items)
        (1, 2, (3, 4, 5), 2, (3, (4, 5), 'ciao'))
        >>> deep_convert(tuple, items, skip=None)
        (1, 2, (3, 4, 5), 2, (3, (4, 5), (('c',), ('i',), ('a',), ('o',))))
        >>> deep_convert(None, items)
        [1, 2, [3, 4, 5], 2, (3, [4, 5], 'ciao')]
        >>> print(
        ...     deep_map(lambda x: x, items, tuple)
        ...     == deep_convert(tuple, items))
        True
        >>> print(
        ...     deep_filter(lambda x: True, items, tuple)
        ...     == deep_convert(tuple, items))
        True
    """
    # note: this cannot be rewritten as generator because of recursion!
    if container is not None:
        new_items = []
        for item in items:
            if max_depth == 0 or not is_deep(item, skip) or item == items:
                new_items.append(item)
            else:
                new_items.append(
                    deep_convert(container, item, max_depth - 1, skip))
        return container(new_items)
    else:
        return items


# ======================================================================
def deep_filter_map(
        items,
        func=None,
        map_condition=None,
        filter_condition=None,
        container=None,
        max_depth=-1,
        skip=(str, bytes, bytearray)):
    """
    Apply conditional mapping, filtering and conversion on nested structures.

    The behavior of this function can be obtained by combining the following:
     - flyingcircus.conditional_apply()
     - flyingcircus.deep_map()
     - flyingcircus.deep_filter()
     - flyingcircus.deep_convert()

    In particular:

    deep_filter_map(
        items, func, map_condition, filter_condition, container,
        avoid, max_depth)

    is equivalent to:

    deep_convert(
        container,
        deep_map(
            conditional_apply(func, map_condition),
            deep_filter(filter_condition, items, avoid, max_depth),
            avoid, max_depth),
        avoid, max_depth)

    If some of the parameters of `deep_filter_map()` can be set to None, the
    equivalent expression can be simplified.
    However, if the call to `deep_filter_map()` would require both
    `deep_map()` and `deep_filter()`, then `deep_filter_map()` is generally
    (and typically also substantially) more performing.

    Args:
        items (Iterable): The input items.
        func (callable): The function to apply to the individual item.
            Must have the following signature: func(Any): Any.
        map_condition (callable|None): The map condition function.
            Only items matching the condition are mapped.
            Must have the following signature: map_condition(Any): bool.
        filter_condition (callable|None): The filter condition function.
            Only items matching the condition are included.
            Must have the following signature: filter_condition(Any): bool.
                container (callable|None): The container to apply.
            Must have the following signature:
            container(Iterable): container.
            If None, the original container is retained.
        container (callable|None): The container for the result.
            If None, this is inferred from `seq` if possible, otherwise
            uses `tuple`.
        max_depth (int): Maximum depth to reach. Negative for unlimited.
        skip (tuple|None): Types to skip descending into.

    Returns:
        new_items (Iterable): The mapped and filtered items.

    Examples:
        >>> items = [1, 2, [3, 4, 5], 2, (3, [4, 5])]
        >>> print(
        ...     deep_filter_map(items, str, lambda x: x > 2, lambda x: x > 1))
        [2, ['3', '4', '5'], 2, ('3', ['4', '5'])]
        >>> print(deep_map(
        ...     conditional_apply(str, lambda x: x > 2),
        ...     deep_filter(lambda x: x > 1, items)))
        [2, ['3', '4', '5'], 2, ('3', ['4', '5'])]
        >>> print(deep_filter_map(items, str, None, lambda x: x > 1))
        ['2', ['3', '4', '5'], '2', ('3', ['4', '5'])]
        >>> print(deep_filter_map(items, str, lambda x: x > 2, None))
        [1, 2, ['3', '4', '5'], 2, ('3', ['4', '5'])]
        >>> print(deep_filter_map(items, None, None, lambda x: x > 1))
        [2, [3, 4, 5], 2, (3, [4, 5])]
        >>> deep_filter_map(items, str, None, None)
        ['1', '2', ['3', '4', '5'], '2', ('3', ['4', '5'])]
        >>> deep_filter_map(items, str, None, None, tuple)
        ('1', '2', ('3', '4', '5'), '2', ('3', ('4', '5')))
        >>> def c1(x): return x > 1
        >>> def c2(x): return x > 2
        >>> print(
        ...     deep_filter_map(items, str, c2, c1, tuple)
        ...     == deep_convert(tuple, deep_map(conditional_apply(str, c2),
        ...         deep_filter(c1, items))))
        True
        >>> print(
        ...     deep_filter_map(items, str, c2, c1)
        ...     == deep_map(conditional_apply(str, c2),
        ...         deep_filter(c1, items)))
        True
        >>> print(
        ...    deep_filter_map(items, str, None, c1)
        ...    == deep_map(str, deep_filter(c1, items)))
        True
        >>> print(
        ...    deep_filter_map(items, str, c2, None)
        ...    == deep_map(conditional_apply(str, c2), items))
        True
        >>> print(
        ...    deep_filter_map(items, None, None, c1)
        ...    == deep_filter(c1, items))
        True
        >>> print(
        ...    deep_filter_map(items, str, None, None)
        ...    == deep_map(str, items))
        True
        >>> print(
        ...    deep_filter_map(items, None, None, None, tuple)
        ...    == deep_convert(tuple, items))
        True
    """
    # note: this cannot be rewritten as generator because of recursion!
    final_container = type(items) if container is None else container
    if not callable(final_container):
        final_container = tuple

    if func is None:
        def func(x): return x
    if map_condition is None:
        def map_condition(_): return True
    if filter_condition is None:
        def filter_condition(_): return True

    new_items = []
    for item in items:
        if max_depth == 0 or not is_deep(item, skip) or item == items:
            if filter_condition(item):
                new_items.append(func(item) if map_condition(item) else item)
        else:
            new_items.append(
                deep_filter_map(
                    item, func, map_condition, filter_condition, container,
                    max_depth - 1, skip))
    return final_container(new_items)


# ======================================================================
def random_int(
        first,
        second=None,
        rand_bits=random.getrandbits):
    """
    Pick a random integer in the specified range.

    Note that this is roughly equivalent to (but faster than)
    `random.randrange()` or `random.randint()`.

    Args:
        first (int): The first value of the range.
            If `second` is None, the `start` value is 0 and this is the `stop`
            value (not included in the range).
            Otherwise, this is the `start` value and it is always included.
        second (int|None): The second value of the range.
            If None, the start value is 1 and the stop value is `first`.
            Otherwise, this is the stop value and it is not included.
        rand_bits (callable): Function to generate random bits as int.
            Must accept the following signature: rand_bits(n: int): int
            The `n` parameter is the number of `bits`.

    Returns:
        result (int): The random value within the specified range.

    Raises:
        ValueError: if `first` and `second` would produce an empty range.
            This happens if `second` is None and `first` is not positive, or
            if `second <= first`.

    Examples:
        >>> random.seed(0); n = 16; [random_int(n) for _ in range(n)]
        [11, 12, 8, 12, 13, 1, 8, 14, 0, 15, 12, 13, 9, 10, 9, 14]
        >>> random.seed(0); n = 16; [random_int(10, 20) for _ in range(n)]
        [13, 16, 12, 14, 16, 10, 14, 15, 18, 17, 16, 14, 12, 13, 14, 15]
        >>> random.seed(0); n = 16; [random_int(-10, 10) for _ in range(n)]
        [-3, 2, -6, -2, 3, -9, -2, 0, 6, 5, 2, -1, -5, -4, -1, 0]
        >>> random.seed(0); n = 16; [random_int(-10, -5) for _ in range(n)]
        [-9, -7, -9, -8, -7, -10, -8, -8, -6, -7, -7, -8, -9, -9, -8, -8]
        >>> random_int(0)
        0
        >>> random_int(10, 10)
        10
        >>> random_int(10, 5)
        Traceback (most recent call last):
            ...
        ValueError: The range size must be greater than 0
    """
    if second is None:
        size, offset = first, 0
    else:
        size, offset = second - first, first
    if size > 0:
        if offset:
            return rand_bits(size.bit_length()) % size + offset
        else:
            return rand_bits(size.bit_length()) % size
    elif size == 0:
        return offset
    else:
        raise ValueError('The range size must be greater than 0')


# ======================================================================
def flip(
        seq,
        first=0,
        last=-1):
    """
    Reverse or flip in-place a sequence.

    Warning! This function modifies its `seq` parameter.

    This supports also partial reversing / flipping.

    Note that this is roughly equivalent to:

    seq[first:last + 1] = reversed(seq[first:last + 1])

    but it does not require additional memory.

    Args:
        seq (MutableSequence): The input sequence.
        first (int): The first index.
            The index is forced within boundaries.
        last (int): The last index (included).
            The index is forced within boundaries.

    Returns:
        result (int): The reversed sequence.

    Examples:
        >>> seq = list(range(10))
        >>> print(seq)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> print(flip(seq))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        >>> print(seq)
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

        >>> print(flip(seq, 3, 7))
        [9, 8, 7, 2, 3, 4, 5, 6, 1, 0]
        >>> print(flip(seq, 3, 7))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

        >>> print(flip(seq, 3, 6))
        [9, 8, 7, 3, 4, 5, 6, 2, 1, 0]
        >>> print(flip(seq, 3, 6))
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    """
    n = len(seq)
    first = valid_index(first, n)
    last = valid_index(last, n)
    if first == 0 and last == n - 1 and hasattr(seq, 'reverse'):
        seq.reverse()
    else:
        # : faster, but not purely in-place, alternative
        # subseq = seq[first:last + 1]
        # subseq.reverse()
        # seq[first:last + 1] = subseq
        # del subseq

        # : faster, but not purely in-place, alternative
        # seq[first:last + 1] = seq[last:first - 1:-1]

        m = last + first
        for i in range(first, (m + 1) // 2):
            j = m - i
            seq[i], seq[j] = seq[j], seq[i]
    return seq


# ======================================================================
def shuffle(
        seq,
        first=0,
        last=-1,
        k=1,
        rand_int=random_int):
    """
    Shuffle in-place a sequence.

    Warning! This function modifies its `seq` parameter.

    This use Fisher-Yates shuffle (Durstenfeld method).

    Args:
        seq (MutableSequence): The input sequence.
        first (int): The first index.
            The index is forced within boundaries.
        last (int): The last index (included).
            The index is forced within boundaries.
        k (int): Swap reduction factor.
            Larger values result in more speed and less randomness.
            Must be 1 or more, otherwise it is ignored
        rand_int (callable): The random integer generator.
            Must accept the signature: rand_int(a: int, b: int|None): int
            If `b` is None, must produce a random value in the [0, a) range
            (0 is included, `a` is excluded), otherwise must produce a random
            value in the [a, b) range (`a` is included, `b` is excluded)

    Returns:
        result (int): The shuffled sequence.

    Examples:
        >>> seq = list(range(16))
        >>> print(seq)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        >>> seq = list(range(16)); random.seed(0)
        >>> print(shuffle(seq))
        [15, 8, 1, 10, 5, 11, 3, 9, 7, 4, 0, 14, 2, 12, 6, 13]

        >>> seq = list(range(16)); random.seed(0)
        >>> print(shuffle(seq, k=2))
        [7, 9, 2, 11, 4, 0, 15, 3, 8, 5, 10, 1, 12, 6, 14, 13]

        >>> seq = list(range(16)); random.seed(0)
        >>> print(shuffle(seq, 6))
        [0, 1, 2, 3, 4, 5, 9, 13, 12, 6, 7, 11, 14, 10, 8, 15]

        >>> seq = list(range(16)); random.seed(0)
        >>> print(shuffle(seq, 3, 13))
        [0, 1, 2, 5, 10, 8, 12, 4, 3, 11, 13, 7, 9, 6, 14, 15]

        >>> seq = list(range(16)); random.seed(0)
        >>> print(shuffle(seq, 0, 13))
        [7, 8, 5, 9, 3, 10, 11, 2, 13, 12, 4, 1, 6, 0, 14, 15]

    See Also:
        - random.shuffle()
        - random.randrange()
        - flyingcircus.random_int()
        - https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
    """
    n = len(seq)
    first = valid_index(first, n)
    last = valid_index(last, n)
    if k < 1:
        k = 1
    # use Fisher-Yates shuffle (Durstenfeld method)
    if first > 0:
        for i in range(first, last, k):
            j = rand_int(i, last + 1)
            seq[i], seq[j] = seq[j], seq[i]
    else:
        for i in range(last, first, -k):
            j = rand_int(i)
            seq[i], seq[j] = seq[j], seq[i]
    return seq


# ======================================================================
def partition(
        seq,
        condition,
        first=0,
        last=-1):
    """
    Partition in-place a sequence according to a specified condition.

    Warning! This function modifies its `seq` parameter.

    Args:
        seq (MutableSequence): The input sequence.
        condition (callable): The partitioning condition.
            Must have the following signature: condition(Any): bool.
            If the condition is met,
        first (int): The first index.
            The index is forced within boundaries.
            If first < last, the partitioning is performed forward,
            otherwise the partitioning is performed backward.
        last (int): The last index (included).
            The index is forced within boundaries.
            If first < last, the partitioning is performed forward,
            otherwise the partitioning is performed backward.

    Returns:
        result (int): The index that delimits the partitioning.

    Examples:
        >>> seq = list(range(10))
        >>> k = partition(seq, lambda x: x % 2 == 0)
        >>> print(seq[:k], seq[k:])
        [0, 2, 4, 6, 8] [5, 3, 7, 1, 9]

        >>> k = partition(seq, lambda x: x >= 5)
        >>> print(seq[:k], seq[k:])
        [6, 8, 5, 7, 9] [4, 3, 0, 1, 2]

        >>> k = partition(seq, lambda x: x < 5)
        >>> print(seq[:k], seq[k:])
        [4, 3, 0, 1, 2] [6, 8, 5, 7, 9]

        >>> seq = list(range(10))
        >>> k = partition(seq, lambda x: x % 5, 2, 8)
        >>> print(seq[2:k], seq[k:9])
        [2, 3, 4, 6, 7, 8] [5]

        >>> seq = list(range(10))
        >>> k = partition(seq, lambda x: x % 2 == 0, -1, 0)
        >>> print(seq[:k], seq[k:])
        [5, 1, 9, 3, 7] [0, 2, 4, 6, 8]

    See Also:
        - flyingcircus.selection()
        - flyingcircus.quick_sort()
    """
    n = len(seq)
    first = valid_index(first, n)
    last = valid_index(last, n)
    step = 1 if first < last else -1
    for i in range(first, last + step, step):
        if condition(seq[i]):
            seq[first], seq[i] = seq[i], seq[first]
            first += step
    if step > 0:
        return first
    else:
        return first + 1


# ======================================================================
def selection(
        seq,
        k,
        first=0,
        last=-1,
        pivot=random_int,
        randomize=False):
    """
    Rearrange in-place a sequence so that the k-th element is at position k.

    Warning! This function modifies its `seq` parameter.

    This implements quick-select using an iterative approach with
    Hoare partitioning scheme.

    Essentially, this ensures that `seq[k]` is the k-th largest element.
    The order of the elements below or above `k` is ignored.
    The problem is also known as selection or k-th statistics.

    Args:
        seq (MutableSequence): The input sequence.
        k (int): The input index.
            This is 0-based and supports negative indexing.
            The index is forced within boundaries.
            If k is not in the (first, last) interval, the result is undefined.
        first (int): The first index.
            The index is forced within boundaries.
        last (int): The last index (included).
            The index is forced within boundaries.
        pivot (callable): The function for choosing the pivot index.
            Must accept the following signature:
            pivot(first: int, last: int): int
            The arguments are the first and last indices of the subsequence
            to be partitioned.
        randomize (bool|int): Pre-shuffle the input.
            If int, this is passed as `k` parameter
            (shuffling reduction factor) to `flyingcircus.shuffle()`.
            This render the worst case **very** unlikely.
            This is useful for deterministic pivoting.

    Returns:
        seq (MutableSequence): The partially sorted sequence.

    Examples:
        >>> seq = [2 * x for x in range(10)]
        >>> print(seq)
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

        >>> random.seed(0); seq.sort(); print(shuffle(seq))
        [4, 18, 10, 14, 0, 6, 2, 16, 12, 8]
        >>> k = 2
        >>> print(selection(seq, k)[k])
        4

        >>> random.seed(0); seq.sort(); print(shuffle(seq))
        [4, 18, 10, 14, 0, 6, 2, 16, 12, 8]
        >>> k = 5
        >>> print(selection(seq, k)[k])
        10

        >>> random.seed(0); seq.sort(); print(shuffle(seq))
        [4, 18, 10, 14, 0, 6, 2, 16, 12, 8]
        >>> k = -1
        >>> print(selection(seq, k)[k])
        18

        >>> random.seed(0); seq.sort(); print(shuffle(seq))
        [4, 18, 10, 14, 0, 6, 2, 16, 12, 8]
        >>> k = 7
        >>> print(selection(seq, k, 1, 7)[k])
        18

    See Also:
        - flyingcircus.partition()
        - flyingcircus.quick_sort()
    """
    n = len(seq)
    k = valid_index(k, n)
    first = valid_index(first, n)
    last = valid_index(last, n)
    if randomize:
        shuffle(seq, first, last, k=randomize)
    while first < last:
        # : compute a partition (Lomuto) and shrink extrema, alternate method
        # p = pivot(first, last + 1)
        # seq[p], seq[last] = seq[last], seq[p]
        # x = seq[last]
        # p = first
        # for i in range(first, last):
        #     if seq[i] < x:
        #         seq[p], seq[i] = seq[i], seq[p]
        #         p += 1
        # seq[p], seq[last] = seq[last], seq[p]
        # if p == k:
        #     break
        # elif p > k:
        #     last = p - 1
        # else:
        #     first = p + 1

        # : compute a partition (Hoare) and shrink extrema
        p = pivot(first, last + 1)
        x = seq[p]
        i = first
        j = last
        while True:
            while seq[i] < x:
                i += 1
            while seq[j] > x:
                j -= 1
            if i >= j:
                break
            else:
                seq[i], seq[j] = seq[j], seq[i]
                i += 1
                j -= 1
        if k <= j:
            last = j
        else:
            first = j + 1
    return seq


# ======================================================================
def insertion_sort(
        seq,
        first=0,
        last=-1):
    """
    Sort in-place a sequence using insertion sort.

    Warning! This function modifies its `seq` parameter.

    This is slower than `sorted()` or `list.sort()`, but uses less memory.

    The algorithm is:

     - best-case: O(n)
     - average-case: O(n²)
     - worst-case: O(n²)
     - memory: O(1)
     - stable

    Args:
        seq (MutableSequence): The input sequence.
        first (int): The first index.
            The index is forced within boundaries.
        last (int): The last index (included).
            The index is forced within boundaries.

    Returns:
        seq (MutableSequence): The sorted sequence.

    Examples:
        >>> seq = [1, 0, 3, 5, 7, 9, 2, 4, 6, 8]
        >>> insertion_sort(seq)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        >>> seq = [1, 0, 3, 5, 7, 9, 2, 4, 6, 8]
        >>> insertion_sort(seq, 3, 7)
        [1, 0, 3, 2, 4, 5, 7, 9, 6, 8]

        >>> seq = [9, 0, 2, 6, 3, 5, 1, 7, 8, 4, 1, 0, 3, 2, 4, 5, 7, 9, 6, 8]
        >>> insertion_sort(seq)
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]

        >>> seq = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        >>> insertion_sort(seq)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        >>> seq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        >>> insertion_sort(seq)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    See Also:
        - flyingcircus.selection_sort()
        - flyingcircus.step_sort()
        - flyingcircus.quick_sort()
        - flyingcircus.merge_sort()
        - flyingcirucs.nat_merge_sort()
    """
    n = len(seq)
    first = valid_index(first, n)
    last = valid_index(last, n)
    for i in range(first + 1, last + 1):
        j = i
        while j > first and seq[j - 1] > seq[j]:
            seq[j], seq[j - 1] = seq[j - 1], seq[j]
            j -= 1
    return seq


# ======================================================================
def selection_sort(
        seq,
        first=0,
        last=-1):
    """
    Sort in-place using (double-edged) selection sort.
    Warning! This function modifies its `seq` parameter.

    This is slower than `sorted()` or `list.sort()`, but uses less memory.

    The algorithm is:

     - best-case: O(n²)
     - average-case: O(n²)
     - worst-case: O(n²)
     - memory: O(1)
     - stable

    Args:
        seq (MutableSequence): The input sequence.
        first (int): The first index.
            The index is forced within boundaries.
        last (int): The last index (included).
            The index is forced within boundaries.

    Returns:
        seq (MutableSequence): The sorted sequence.

    Examples:
        >>> seq = [1, 0, 3, 5, 7, 9, 2, 4, 6, 8]
        >>> selection_sort(seq)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        >>> seq = [1, 0, 3, 5, 7, 9, 2, 4, 6, 8]
        >>> selection_sort(seq, 3, 7)
        [1, 0, 3, 2, 4, 5, 7, 9, 6, 8]

        >>> seq = [9, 0, 2, 6, 3, 5, 1, 7, 8, 4, 1, 0, 3, 2, 4, 5, 7, 9, 6, 8]
        >>> selection_sort(seq)
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]

        >>> seq = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        >>> selection_sort(seq)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        >>> seq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        >>> selection_sort(seq)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    See Also:
        - flyingcircus.insertion_sort()
        - flyingcircus.step_sort()
        - flyingcircus.quick_sort()
        - flyingcircus.merge_sort()
        - flyingcirucs.nat_merge_sort()
    """
    n = len(seq)
    first = valid_index(first, n)
    last = valid_index(last, n)
    while first < last:
        # : alternate (faster?) method for finding argmin / argmax
        # subseq = seq[first:last + 1]
        # min_val = min(subseq)
        # i = subseq.index(min_val) + first
        # max_val = max(subseq)
        # j = subseq.index(max_val) + first

        min_val = seq[first]
        i = first
        max_val = seq[last]
        j = last
        for k in range(first, last + 1):
            x = seq[k]
            if x < min_val:
                min_val = x
                i = k
            elif x > max_val:
                max_val = x
                j = k
        if first != i:
            seq[first], seq[i] = seq[i], seq[first]
        if first == j:
            j = i
        if last != j:
            seq[last], seq[j] = seq[j], seq[last]
        first += 1
        last -= 1
    return seq


# ======================================================================
def step_sort(
        seq,
        first=0,
        last=-1,
        steps=None):
    """
    Sort in-place a sequence using step insertion (Shell) sort.

    Warning! This function modifies its `seq` parameter.

    This is slower than `sorted()` or `list.sort()`, but uses less memory.

    The algorithm is:

     - best-case: O(n log n)
     - average-case: O(n log n) to O(n √n) -- depending on the steps
     - worst-case: O(n log² n) to O(n²) -- depending on the steps
     - memory: O(1)
     - unstable

    Args:
        seq (MutableSequence): The input sequence.
        first (int): The first index.
            The index is forced within boundaries.
        last (int): The last index (included).
            The index is forced within boundaries.
        steps (Iterable[int]|None): The steps to use.
            If None, uses pseudo-optimal values following the sequence:
            x(k) = 9 * x(k-1) // 4; with x(0) = 1, x(1) = 4
            in decreasing order not exceeding the sequence length.
            This is empirically fast, but theoretical performance is unknown.

    Returns:
        seq (MutableSequence): The sorted sequence.

    Examples:
        >>> seq = [1, 0, 3, 5, 7, 9, 2, 4, 6, 8]
        >>> step_sort(seq)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        >>> seq = [1, 0, 3, 5, 7, 9, 2, 4, 6, 8]
        >>> step_sort(seq, 3, 7)
        [1, 0, 3, 2, 4, 5, 7, 9, 6, 8]

        >>> seq = [9, 0, 2, 6, 3, 5, 1, 7, 8, 4, 1, 0, 3, 2, 4, 5, 7, 9, 6, 8]
        >>> step_sort(seq)
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]

        >>> seq = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        >>> step_sort(seq)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        >>> seq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        >>> step_sort(seq)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    See Also:
        - flyingcircus.insertion_sort()
        - flyingcircus.selection_sort()
        - flyingcircus.quick_sort()
        - flyingcircus.merge_sort()
        - flyingcirucs.nat_merge_sort()
    """
    n = len(seq)
    first = valid_index(first, n)
    last = valid_index(last, n)
    if steps is None:
        steps = [1]
        x, m, d = 4, 9, 4
        while x < n:
            steps.append(x)
            x = x * m // d
        steps.reverse()
    elif callable(steps):
        steps = sorted(steps(n), reverse=True)
    stop = last + 1
    for step in steps:
        for i in range(first + step, stop):
            temp = seq[i]
            j = i
            while j >= first + step and seq[j - step] > temp:
                seq[j] = seq[j - step]
                j -= step
            seq[j] = temp
    return seq


# ======================================================================
def quick_sort(
        seq,
        first=0,
        last=-1,
        pivot=random_int,
        randomize=False):
    """
    Sort in-place a sequence using quick (partition-exchange) sort.

    Warning! This function modifies its `seq` parameter.

    This is slower than `sorted()` or `list.sort()`, but uses less memory.

    Uses an iterative approach with Hoare partitioning scheme.

    The algorithm is:
     - best-case: O(n log n)
     - average-case: O(n log n)
     - worst-case: O(n²)
     - memory: O(log n) to O(n)
     - unstable

    Args:
        seq (MutableSequence): The input sequence.
        first (int): The first index.
            The index is forced within boundaries.
        last (int): The last index (included).
            The index is forced within boundaries.
        pivot (callable): The function for choosing the pivot index.
            Must accept the following signature:
            pivot(first: int, last: int): int
            The arguments are the first and last indices of the subsequence
            to be partitioned.
        randomize (bool): Pre-shuffle the input.
            If int, this is passed as `k` parameter
            (shuffling reduction factor) to `flyingcircus.shuffle()`.
            This render the worst case **very** unlikely.
            This is useful for deterministic pivoting.

    Returns:
        seq (MutableSequence): The sorted sequence.

    Examples:
        >>> seq = [1, 0, 3, 5, 7, 9, 2, 4, 6, 8]
        >>> quick_sort(seq)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        >>> seq = [1, 0, 3, 5, 7, 9, 2, 4, 6, 8]
        >>> quick_sort(seq, 3, 7)
        [1, 0, 3, 2, 4, 5, 7, 9, 6, 8]

        >>> seq = [9, 0, 2, 6, 3, 5, 1, 7, 8, 4, 1, 0, 3, 2, 4, 5, 7, 9, 6, 8]
        >>> quick_sort(seq)
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]

        >>> seq = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        >>> quick_sort(seq)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        >>> seq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        >>> quick_sort(seq)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    See Also:
        - flyingcircus.partition()
        - flyingcircus.selection()
        - flyingcircus.insertion_sort()
        - flyingcircus.selection_sort()
        - flyingcircus.step_sort()
        - flyingcircus.merge_sort()
        - flyingcirucs.nat_merge_sort()
    """
    n = len(seq)
    first = valid_index(first, n)
    last = valid_index(last, n)
    if randomize:
        shuffle(seq, first, last, randomize)
    indices = [(first, last)]
    # : Lomuto partitioning with mid pivoting
    # while indices:
    #     # compute a partition
    #     first, last = indices.pop()
    #     p = pivot(first, last + 1)
    #     seq[p], seq[last] = seq[last], seq[p]
    #     x = seq[last]
    #     p = first
    #     for i in range(first, last):
    #         if seq[i] <= x:
    #             seq[p], seq[i] = seq[i], seq[p]
    #             p += 1
    #     seq[p], seq[last] = seq[last], seq[p]
    #     # update indices
    #     if p - 1 > first:
    #         indices.append((first, p - 1))
    #     if last > p + 1:
    #         indices.append((p + 1, last))
    # return seq
    # : Hoare partitioning with alternating first third and mid pivoting
    while indices:
        first, last = indices.pop()
        # compute a partition
        p = pivot(first, last + 1)
        x = seq[p]
        i = first
        j = last
        while True:
            while seq[i] < x:
                i += 1
            while seq[j] > x:
                j -= 1
            if i >= j:
                break
            else:
                seq[i], seq[j] = seq[j], seq[i]
                i += 1
                j -= 1
        # j contains the pivotal index
        if j > first and j - first > 0:
            indices.append([first, j])
        if last > j + 1 and last - j > 1:
            indices.append([j + 1, last])
    return seq


# ======================================================================
def merge_sort(
        seq,
        first=0,
        last=-1):
    """
    Sort in-place a sequence using merge sort.

    Warning! This function modifies its `seq` parameter.

    This is slower than `sorted()` or `list.sort()`.

    Uses a bottom-up approach.
    Note that the merge step is not in-place in the sense that it still
    requires a memory buffer the size of the sequence.

    The algorithm is:
     - best-case: O(n log n)
     - average-case: O(n log n)
     - worst-case: O(n log n)
     - memory: O(n)
     - stable

    Args:
        seq (MutableSequence): The input sequence.
        first (int): The first index.
            The index is forced within boundaries.
        last (int): The last index (included).
            The index is forced within boundaries.

    Returns:
        seq (MutableSequence): The sorted sequence.

    Examples:
        >>> seq = [1, 0, 3, 5, 7, 9, 2, 4, 6, 8]
        >>> merge_sort(seq)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        >>> seq = [1, 0, 3, 5, 7, 9, 2, 4, 6, 8]
        >>> merge_sort(seq, 3, 7)
        [1, 0, 3, 2, 4, 5, 7, 9, 6, 8]

        >>> seq = [9, 0, 2, 6, 3, 5, 1, 7, 8, 4, 1, 0, 3, 2, 4, 5, 7, 9, 6, 8]
        >>> merge_sort(seq)
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]

        >>> seq = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        >>> merge_sort(seq)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        >>> seq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        >>> merge_sort(seq)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    See Also:
        - flyingcircus.insertion_sort()
        - flyingcircus.selection_sort()
        - flyingcircus.step_sort()
        - flyingcircus.quick_sort()
        - flyingcirucs.nat_merge_sort()
    """
    n = len(seq)
    first = valid_index(first, n)
    last = valid_index(last, n)
    size = last - first + 1
    stop = last + 1
    temp = [None] * size
    width = 1
    while width <= n:
        for l in range(first, stop, 2 * width):
            m = l + width
            if m > stop:
                m = stop
            h = l + 2 * width
            if h > stop:
                h = stop
            # merge
            i, j = l, m
            for k in range(l - first, h - first):
                if i < m and (j >= h or seq[i] <= seq[j]):
                    temp[k] = seq[i]
                    i += 1
                else:
                    temp[k] = seq[j]
                    j += 1
        seq[first:stop] = temp
        width *= 2
    return seq


# ======================================================================
def nat_merge_sort(
        seq,
        first=0,
        last=-1):
    """
    Sort in-place a sequence using natural merge sort.

    Warning! This function modifies its `seq` parameter.

    This is slower than `sorted()` or `list.sort()`.

    Uses a natural approach (exploiting partially sorted sub-sequences).
    The best-case is for sorted or reversed inputs.
    Note that the merge step is not in-place in the sense that it still
    requires a memory buffer the size of the sequence.

    The algorithm is:
     - best-case: O(n)
     - average-case: O(n log n)
     - worst-case: O(n log n)
     - memory: O(n)
     - stable

    Args:
        seq (MutableSequence): The input sequence.
        first (int): The first index.
            The index is forced within boundaries.
        last (int): The last index (included).
            The index is forced within boundaries.

    Returns:
        seq (MutableSequence): The sorted sequence.

    Examples:
        >>> seq = [1, 0, 3, 5, 7, 9, 2, 4, 6, 8]
        >>> nat_merge_sort(seq)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        >>> seq = [1, 0, 3, 5, 7, 9, 2, 4, 6, 8]
        >>> nat_merge_sort(seq, 3, 7)
        [1, 0, 3, 2, 4, 5, 7, 9, 6, 8]

        >>> seq = [9, 0, 2, 6, 3, 5, 1, 7, 8, 4, 1, 0, 3, 2, 4, 5, 7, 9, 6, 8]
        >>> nat_merge_sort(seq)
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9]

        >>> seq = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        >>> nat_merge_sort(seq)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        >>> seq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        >>> nat_merge_sort(seq)
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    See Also:
        - flyingcircus.insertion_sort()
        - flyingcircus.selection_sort()
        - flyingcircus.step_sort()
        - flyingcircus.quick_sort()
        - flyingcirucs.merge_sort()
    """
    n = len(seq)
    first = valid_index(first, n)
    last = valid_index(last, n)
    size = last - first + 1
    temp = [None] * size
    # reverse decreasing sequences
    l = first
    while True:
        h = l
        while h + 1 <= last and seq[h + 1] >= seq[h]:
            h += 1
        l = h
        if h < last:
            h += 1
            while h + 1 <= last and seq[h + 1] < seq[h]:
                h += 1
            if l < h:
                flip(seq, l, h)
        if h < last:
            l = h + 1
        else:
            break
    # sorting
    l = first
    while True:
        # find first increasing sequence
        m = l
        while m + 1 <= last and seq[m + 1] >= seq[m]:
            m += 1
        m += 1
        # find second increasing sequence
        h = m
        while h + 1 <= last and seq[h + 1] >= seq[h]:
            h += 1
        if l == first and h > last:
            break
        if m > last:
            m = last
        if h > last:
            h = last
        i, j = l, m
        for k in range(l - first, h - first + 1):
            if i < m and (j > h or seq[i] <= seq[j]):
                temp[k] = seq[i]
                i += 1
            else:
                temp[k] = seq[j]
                j += 1
        # reset lower index
        if h < last:
            l = h + 1
        else:
            seq[first:last + 1] = temp
            l = first
    return seq


# ======================================================================
def argsort(seq):
    """
    Sort the indexes of a sequence.

    This is useful to sort a sequence using the ordering from another sequence.

    Args:
        seq (Sequence): The input sequence.

    Returns:
        indexes (list): The sorted indexes.

    Examples:
        >>> x = 'flyingcircus'
        >>> i = argsort(x)
        >>> print(list(iter_at(x, i)))
        ['c', 'c', 'f', 'g', 'i', 'i', 'l', 'n', 'r', 's', 'u', 'y']
        >>> j = argsort(i)
        >>> print(list(iter_at(list(iter_at(x, i)), j)))
        ['f', 'l', 'y', 'i', 'n', 'g', 'c', 'i', 'r', 'c', 'u', 's']

        >>> y = 'abracadabras'
        >>> print(list(iter_at(y, i)))
        ['d', 'r', 'a', 'a', 'a', 'a', 'b', 'c', 'b', 's', 'a', 'r']
        >>> print(list(iter_at(list(iter_at(y, i)), j)))
        ['a', 'b', 'r', 'a', 'c', 'a', 'd', 'a', 'b', 'r', 'a', 's']

        >>> print(sorted(x) == list(iter_at(x, i)))
        True
        >>> print(list(x) == list(iter_at(list(iter_at(x, i)), j)))
        True
        >>> print(list(y) == list(iter_at(list(iter_at(y, i)), j)))
        True
    """
    return sorted(range(len(seq)), key=seq.__getitem__)


# ======================================================================
def select_ordinal(
        seq,
        k,
        first=0,
        last=-1):
    """
    Find the smallest k-th element in a sequence.

    This is roughly equivalent to, but asymptotically more efficient than:
    `sorted(seq)[k]`.
    The problem is also known as selection or k-th statistics.
    This is the not-in-place version of `flyingcircus.selection()`.
    This is similar to `flyingcircus.medoid()` and `flyingcircus.quantiloid()`,
    except that the arguments are different and they are potentially faster.

    Args:
        seq (Sequence): The input sequence.
        k (int): The input index.
            This is 0-based and supports negative indexing.
            If k > len(seq) or k < -len(seq), None is returned.
            If k is not in the (first, last) interval, the result is undefined.
        first (int): The first index.
            The index is forced within boundaries.
        last (int): The last index (included).
            The index is forced within boundaries.

    Returns:
        result: The smallest k-th element.

    Examples:
        >>> seq = [2 * x for x in range(10)]
        >>> print(seq)
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        >>> random.seed(0); print(shuffle(seq))
        [4, 18, 10, 14, 0, 6, 2, 16, 12, 8]
        >>> seq = tuple(seq)  # seq is now immutable
        >>> print(select_ordinal(seq, 0))
        0
        >>> print(select_ordinal(seq, 9))
        18
        >>> print(select_ordinal(seq, 4))
        8
        >>> print(select_ordinal(seq, -10))
        0
        >>> print(select_ordinal(seq, -1))
        18
        >>> print(select_ordinal(seq, -6))
        8
        >>> print(select_ordinal(seq, 10))
        None
        >>> print(select_ordinal(seq, -11))
        None
        >>> print(select_ordinal(seq, 7, 1, 7))
        18

    See Also:
        - flyingcircus.partition()
        - flyingcircus.medoid()
        - flyingcircus.quantiloid()
    """
    n = len(seq)
    if -n <= k < n:
        return selection(list(seq), k, first, last)[k]
    else:
        return None


# ======================================================================
def uniques(items):
    """
    Get unique items (keeping order of appearance).

    If the order of appearance is not important, use `set()`.

    Args:
        items (Iterable): The input items.

    Yields:
        item: Unique items.

    Examples:
        >>> items = (5, 4, 3, 2, 1, 1, 2, 3, 4, 5, 3, 2, 4, 2, 4, 1, 1)
        >>> tuple(uniques(items))
        (5, 4, 3, 2, 1)
        >>> tuple(set(items))
        (1, 2, 3, 4, 5)
        >>> sorted(set(items)) == sorted(uniques(items))
        True
        >>> tuple(uniques('abcde'))
        ('a', 'b', 'c', 'd', 'e')
    """
    seen = set()
    for item in items:
        if item not in seen and not seen.add(item):
            yield item


# ======================================================================
def combine_iter_len(
        items,
        combine=max,
        non_seq_len=1):
    """
    Combine the length of each item within items.

    For each item within items, determine if the item has a length and then
    use a given combination function to combine the multiple extracted length.
    If an item is not a sequence, its length is assumed to be 0.

    A useful application is to determine the longest item.

    Args:
        items (Iterable): The collection of items to inspect.
        combine (callable): The combination method.
            Must have the following signature: combine(int, int): int.
            The lengths are combined incrementally.
        non_seq_len (int): The length of non-sequence items.
            Typical choices are `0` or `1` depending on the application.

    Returns:
        num (int): The combined length of the collection.

    Examples:
        >>> a = list(range(10))
        >>> b = tuple(range(5))
        >>> c = set(range(20))
        >>> combine_iter_len((a, b, c))
        20
        >>> combine_iter_len((a, b, c), min)
        5
        >>> combine_iter_len((1, a))
        10
    """
    num = None
    for val in items:
        new_num = non_seq_len
        try:
            new_num = len(val)
        except TypeError:
            pass
        finally:
            if num is None:
                num = new_num
            else:
                num = combine(new_num, num)
    return num


# ======================================================================
def pairwise_map(
        func,
        items,
        reverse=False):
    """
    Apply a binary function to consecutive elements in an iterable.

    The same can be obtained combining `map()` and `flyingcircus.slide()`,
    but this is faster.

    Args:
        func (callable): The pairwise operator to apply.
        items (Iterable): The input items.
        reverse (bool): Reverse the order of the arguments in `func()`.

    Yields:
        value: The result of func applied to the next pair.

    Examples:
        >>> list(pairwise_map(lambda x, y: (x, y), range(8)))
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
        >>> list(pairwise_map(lambda x, y: (x, y), range(8), True))
        [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6)]
        >>> list(pairwise_map(lambda x, y: x < y, range(10)))
        [True, True, True, True, True, True, True, True, True]
        >>> list(pairwise_map(lambda x, y: x < y, []))
        []

        >>> def sub_args(x): return x[0] - x[1]
        >>> items = range(1000)
        >>> all(x == y for x, y in zip(
        ...     pairwise_map(operator.sub, items),
        ...     map(sub_args, slide(items, 2))))
        True

    See Also:
        - flyingcircus.slide()
    """
    iter_items = iter(items)
    try:
        last_item = next(iter_items)
    except StopIteration:
        pass
    else:
        # : condition is before the loop for performance
        if reverse:
            for item in iter_items:
                yield func(item, last_item)
                last_item = item
        else:
            for item in iter_items:
                yield func(last_item, item)
                last_item = item


# ======================================================================
def slide(
        items,
        size=2,
        step=1,
        truncate=True,
        fill=None,
        reverse=False):
    """
    Generate a sliding grouping / window across the items.

    The number of elements for each yield is fixed.

    This can be used to compute sliding/running/moving/rolling statistics
    for the general case, but rolling computations that can be expressed
    in terms of the previous iteration may be computed more efficiently
    with `flyingcircus.rolling()`.

    Args:
        items (Iterable): The input items.
        size (int): The windowing size.
        step (int|None): The windowing step.
            If int, must be larger than 0.
        truncate (bool): Determine how to handle uneven splits.
            If True, last groups are skipped if smaller than `size`.
        fill (Any): Value to use for filling the last group.
            This is only used when `truncate` is False.
        reverse (bool): Reverse the order within the window.

    Returns:
        result (zip|itertools.zip_longest): Iterable of items within window.

    Examples:
        >>> tuple(slide(range(8), 2))
        ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7))
        >>> tuple(slide(range(8), 3))
        ((0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6), (5, 6, 7))

        >>> tuple(slide(range(8), 3, 2))
        ((0, 1, 2), (2, 3, 4), (4, 5, 6))
        >>> tuple(slide(range(8), 3, 2, False))
        ((0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, None))
        >>> tuple(slide(range(8), 3, 2, False, -1))
        ((0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, -1))

        >>> tuple(slide(range(5), 3, 1))
        ((0, 1, 2), (1, 2, 3), (2, 3, 4))
        >>> tuple(slide(range(5), 3, 1, False))
        ((0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, None), (4, None, None))

        >>> tuple(slide(range(8), 2, 1))
        ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7))
        >>> tuple(slide(range(8), 1, 1))
        ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,))
        >>> tuple(slide(range(8), 1, 2))
        ((0,), (2,), (4,), (6,))
        >>> tuple(slide(range(8), 2, 2))
        ((0, 1), (2, 3), (4, 5), (6, 7))

        >>> tuple(slide(range(8), 2, reverse=True))
        ((1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 5), (7, 6))

        >>> def irange(*args):
        ...    for x in range(*args):
        ...        yield x
        >>> tuple(slide(irange(8), 2))
        ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7))

    See Also:
        - flyingcircus.sliding()
        - flyingcircus.separate()
        - flyingcircus.split()
        - flyingcircus.chunks()
        - flyingcircus.rolling()
    """
    iters = itertools.tee(iter(items), size)
    if step > 1:
        iters = [
            itertools.islice(itering, i, None, step)
            for i, itering in enumerate(iters)]
    else:
        # : alternate (slightly faster, but less flexible) implementation
        def consumed(iterator, n):
            next(itertools.islice(iterator, n, n), None)
            return iterator

        iters = [
            consumed(itering, i) for i, itering in enumerate(iters)]
    if reverse:
        iters = reversed(iters)
    if truncate:
        return zip(*iters)
    else:
        return itertools.zip_longest(*iters, fillvalue=fill)


# ======================================================================
def sliding(
        func,
        items,
        size,
        step=1,
        truncate=True,
        fill=None,
        star=False):
    """
    Apply a function to a sliding grouping / window of the items.

    Args:
        func (callable): The function to apply.
        items (Iterable): The input items.
        size (int): The windowing size.
        step (int|None): The windowing step.
            If int, must be larger than 0.
        truncate (bool): Determine how to handle uneven splits.
            If True, last groups are skipped if smaller than `size`.
        fill (Any): Value to use for filling the last group.
            This is only used when `truncate` is False.
        star(bool): Pass arguments to func using star magic.

    Yields:
        result: The function applied to the slided items.

    Examples:
        >>> items = list(range(10))
        >>> print(items)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> print(list(sliding(sum, items, 2)))
        [1, 3, 5, 7, 9, 11, 13, 15, 17]
        >>> print(list(sliding(sum, items, 3)))
        [3, 6, 9, 12, 15, 18, 21, 24]
        >>> print(list(sliding(sum, items, 3, 2)))
        [3, 9, 15, 21]

        >>> print(list(sliding(sum, items, 3, 2, False, 0)))
        [3, 9, 15, 21, 17]
        >>> print(list(sliding(sum, items, 3, 2, False)))
        Traceback (most recent call last):
            ...
        TypeError: unsupported operand type(s) for +: 'int' and 'NoneType'

        >>> print(list(sliding(operator.add, items, 2, star=True)))
        [1, 3, 5, 7, 9, 11, 13, 15, 17]
        >>> print(list(sliding(operator.add, items, 2)))
        Traceback (most recent call last):
            ...
        TypeError: add expected 2 arguments, got 1

    See Also:
        - flyingcircus.slide()
        - flyingcircus.separate()
        - flyingcircus.split()
        - flyingcircus.chunks()
        - flyingcircus.rolling()
    """
    if star:
        for batch in slide(items, size, step, truncate, fill):
            yield func(*batch)
    else:
        for batch in slide(items, size, step, truncate, fill):
            yield func(batch)


# ======================================================================
def separate(
        items,
        size,
        truncate=False,
        fill=None):
    """
    Separate items into groups with fixed size.

    The number of elements for each yield is fixed.

    For different handling of the last group for uneven splits, and for
    splitting into groups of varying size, see `flyingcircus.split()`.

    Args:
        items (Iterable): The input items.
        size (int): Number of elements to group together.
        truncate (bool): Determine how to handle uneven splits.
            If True, last group is skipped if smaller than `size`.
        fill (Any): Value to use for filling the last group.
            This is only used when `truncate` is False.

    Returns:
        result (zip|itertools.zip_longest): Iterable of grouped items.
            Each group is a tuple regardless of the original container.

    Examples:
        >>> l = list(range(10))
        >>> tuple(separate(l, 4))
        ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, None, None))
        >>> tuple(separate(tuple(l), 2))
        ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
        >>> tuple(separate(l, 4, True))
        ((0, 1, 2, 3), (4, 5, 6, 7))
        >>> tuple(separate(l, 4, False, 0))
        ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 0, 0))

        >>> tuple(separate(l, 2)) == tuple(slide(l, 2, 2))
        True

    See Also:
        - flyingcircus.slide()
        - flyingcircus.sliding()
        - flyingcircus.split()
        - flyingcircus.chunks()
    """
    # : alternate (slower) implementations
    # iterators = tuple(items[i::n] for i in range(n))
    # iterators = tuple(itertools.islice(items, i, None, n) for i in range(n))
    iterators = [iter(items)] * size
    if truncate:
        return zip(*iterators)
    else:
        return itertools.zip_longest(*iterators, fillvalue=fill)


# ======================================================================
def split(
        seq,
        sizes,
        slices=False):
    """
    Split items into groups according to size(s).

    The number of elements for each group can vary.
    Note that if the values in `sizes` are the same, `separate()` can be a
    faster alternative.
    All and only the elements in `items` are ever yielded.

    Args:
        seq (Sequence): The input items.
        sizes (int|Sequence[int]): The size(s) of each group.
            If Sequence, each group has the number of elements specified.
            If int, all groups have the same number of elements.
            The last group will have the remaining items (if any).
        slices (bool): Yield the slices.
            If True, yield the slices that would split the input.
            Otherwise, yield the splitted input.

    Yields:
        Sequence: The items/slices from the grouping.
            Its container matches the one of `items`.

    Examples:
        >>> l = list(range(10))
        >>> tuple(split(l, 4))
        ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9])
        >>> tuple(split(l, (2, 3)))
        ([0, 1], [2, 3, 4], [5, 6, 7, 8, 9])
        >>> tuple(split(l, (2, 4, 1)))
        ([0, 1], [2, 3, 4, 5], [6], [7, 8, 9])
        >>> tuple(split(l, (2, 4, 1, 20)))
        ([0, 1], [2, 3, 4, 5], [6], [7, 8, 9])
        >>> tuple(split(tuple(l), 4))
        ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9))
        >>> tuple(split(tuple(l), 4, True))
        (slice(0, 4, None), slice(4, 8, None), slice(8, 10, None))
        >>> tuple(split(tuple(l), 2)) == tuple(separate(l, 2))
        True

    See Also:
        - flyingcircus.slide()
        - flyingcircus.sliding()
        - flyingcircus.separate()
        - flyingcircus.chunks()
        - flyingcircus.split_by()
        - flyingcircus.group_by()
        - flyingcircus.regroup_by()
    """
    if isinstance(sizes, int):
        sizes = auto_repeat(sizes, len(seq) // sizes)

    num_items = len(seq)
    if sum(sizes) >= num_items:
        sizes = sizes[:-1]
    index = (0,) + tuple(itertools.accumulate(sizes)) + (num_items,)
    num = len(index) - 1
    if slices:
        for i in range(num):
            yield slice(index[i], index[i + 1])
    else:
        for i in range(num):
            yield seq[index[i]:index[i + 1]]


# ======================================================================
def chunks(
        seq,
        n,
        mode='+',
        balanced=True,
        slices=False):
    """
    Split items into groups according to the number desired.

    If the number of items does not allow groups (chunks) of the same size,
    the chunks are determined depending on the values of `balanced`.
    All and only the elements in `items` are ever yielded.

    Args:
        seq (Sequence): The input items.
        n (int): Approximate number of chunks.
            The exact number depends on the value of `mode`.
        mode (str|int): Determine which approximation to use.
            If str, valid inputs are:
             - 'upper', '+': at most `n` chunks are generated.
             - 'lower', '-': at least `n` chunks are generated.
             - 'closest', '~': the number of chunks is `n` or `n + 1`
               depending on which gives the most evenly distributed chunks
               sizes.
            If int, valid inputs are `+1`, `0` and `-1`, mapping to 'upper',
            'closest' and 'lower' respectively.
        balanced (bool): Produce balanced chunks.
            If True, the size of any two chunks is not larger than one.
            Otherwise, the first chunks except the last have the same size.
            This has no effect if the number of items is a multiple of `n`.
        slices (bool): Yield the slices.
            If True, yield the slices that would split the input.
            Otherwise, yield the splitted input.

    Yields:
        Sequence: The items/slices from the grouping.
            Its container matches the one of `items`.

    Examples:
        >>> l = list(range(10))
        >>> tuple(chunks(l, 5))
        ([0, 1], [2, 3], [4, 5], [6, 7], [8, 9])
        >>> tuple(chunks(l, 2))
        ([0, 1, 2, 3, 4], [5, 6, 7, 8, 9])
        >>> tuple(chunks(l, 3))
        ([0, 1, 2, 3], [4, 5, 6], [7, 8, 9])
        >>> tuple(chunks(l, 4))
        ([0, 1, 2], [3, 4, 5], [6, 7], [8, 9])
        >>> tuple(chunks(l, -1))
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],)
        >>> tuple(chunks(l, 3, balanced=False))
        ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9])
        >>> tuple(chunks(l, 3, '-'))
        ([0, 1, 2], [3, 4, 5], [6, 7], [8, 9])
        >>> tuple(chunks(l, 3, '-', False))
        ([0, 1, 2], [3, 4, 5], [6, 7, 8], [9])
        >>> tuple(chunks(list(range(10)), 3, '~'))
        ([0, 1, 2], [3, 4, 5], [6, 7], [8, 9])
        >>> tuple(chunks(list(range(10)), 3, '~', False))
        ([0, 1, 2], [3, 4, 5], [6, 7, 8], [9])

    See Also:
        - flyingcircus.slide()
        - flyingcircus.sliding()
        - flyingcircus.separate()
        - flyingcircus.split()
        - flyingcircus.split_by()
        - flyingcircus.group_by()
        - flyingcircus.regroup_by()
    """
    reversed_modes = {
        math.ceil: ['upper', '+', 1],
        math.floor: ['lower', '-', -1],
        round: ['closest', '~', 0]}
    modes = reverse_mapping_iter(reversed_modes)
    if mode in modes:
        approx = modes[mode][0]
    else:
        raise ValueError('Invalid mode `{mode}`'.format(mode=mode))
    n = max(1, n)
    size = int(approx(len(seq) / n))
    if balanced and 0 < len(seq) % size <= size // 2:
        k = len(seq) // size + 1
        q = -len(seq) % size
        size = (size,) * (k - q) + (size - 1,) * q
    for group in split(seq, size, slices):
        yield group


# ======================================================================
def split_by(
        seq,
        key=None,
        slices=False):
    """
    Split consecutive items into groups by some criterion.

    Args:
        seq (Sequence): The input items.
        key (callable|None): Splitting criterion.
            Must have the following signature: key(Any): Any
            Each element of the sequence is given as input.
            If None, the element itself is used.
        slices (bool): Yield the slices.
            If True, yield the slices that would split the input.
            Otherwise, yield the splitted input.

    Yields:
        Sequence: The items/slices from the grouping.
            Its container matches the one of `items`.

    Examples:
        >>> items = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        >>> print(list(split_by(items)))
        [[1, 1, 1], [2, 2, 2], [3, 3, 3]]

        >>> items = [0, 0, 2, 2, 2, 1, 1, 3, 3, 3]
        >>> print(list(split_by(items)))
        [[0, 0], [2, 2, 2], [1, 1], [3, 3, 3]]

        >>> items = [0, 0, 2, 2, 2, 1, 1, 3, 3, 3]
        >>> print(list(split_by(items, lambda x: x % 2)))
        [[0, 0, 2, 2, 2], [1, 1, 3, 3, 3]]

    See Also:
        - flyingcircus.split()
        - flyingcircus.chunks()
        - flyingcircus.group_by()
        - flyingcircus.regroup_by()
    """
    items = iter(seq)
    try:
        item = next(items)
    except StopIteration:
        return
    else:
        callable_key = callable(key)
        last = key(item) if callable_key else item
        i = j = 0
        for i, item in enumerate(items, 1):
            current = key(item) if callable_key else item
            if last != current:
                yield slice(j, i) if slices else seq[j:i]
                last = current
                j = i
        if i >= j:
            yield slice(j, i + 1) if slices else seq[j:i + 1]

    # : comparable alternative using `itertools.groupby`
    # container = _guess_container(seq)
    # for _, group in itertools.groupby(seq, key):
    #     yield container(group)


# ======================================================================
def group_by(
        items,
        key=None):
    """
    Split consecutive items into groups by some criterion.

    This similar to `flyingcircus.split_by()` except that it also
    yields the value that determines the splitting.

    Args:
        items (Iterable): The input items.
        key (callable|None): Splitting criterion.
            Must have the following signature: key(Any): Any
            Each element of the sequence is given as input.
            If None, the element itself is used.

    Yields:
        tuple[Any, list]: The grouping item and the groups.

    Examples:
        >>> items = [1, 1, 1, 2, 2, 2, 3, 3, 3]
        >>> print(list(group_by(items)))
        [(1, [1, 1, 1]), (2, [2, 2, 2]), (3, [3, 3, 3])]

        >>> items = [0, 0, 2, 2, 2, 1, 1, 3, 3, 3]
        >>> print(list(group_by(items)))
        [(0, [0, 0]), (2, [2, 2, 2]), (1, [1, 1]), (3, [3, 3, 3])]

        >>> items = [0, 0, 2, 2, 2, 1, 1, 3, 3, 3]
        >>> print(list(group_by(items, lambda x: x % 2)))
        [(0, [0, 0, 2, 2, 2]), (1, [1, 1, 3, 3, 3])]

        >>> items = (0, 0, 2, 2, 2, 1, 1, 3, 3, 3)
        >>> print(list(group_by(items, lambda x: x % 2)))
        [(0, [0, 0, 2, 2, 2]), (1, [1, 1, 3, 3, 3])]

    See Also:
        - flyingcircus.split()
        - flyingcircus.chunks()
        - flyingcircus.split_by()
        - flyingcircus.regroup_by()
    """
    items = iter(items)
    try:
        item = next(items)
    except StopIteration:
        return
    else:
        callable_key = callable(key)
        last = key(item) if callable_key else item
        group = [item]
        for item in items:
            current = key(item) if callable_key else item
            if last == current:
                group.append(item)
            else:
                yield last, group
                last = current
                group = [item]
        if group:
            yield last, group


# ======================================================================
def regroup_by(
        items,
        key=None):
    """
    Split items into groups by some criterion.

    The items are not required to be consecutive.
    For splitting consecutive items use `flyingcircus.split_by()`
    or `flyingcircus.group_by()`.

    Args:
        items (Iterable): The input items.
        key (callable|None): Splitting criterion.
            Must have the following signature: key(Any): Any
            Each element of the sequence is given as input.
            If None, the element itself is used.

    Returns:
        dict: Contains the criterion and a list of the grouped items.

    Examples:
        >>> items = [1, 2, 3, 2, 1, 3, 2, 1, 3, 2, 1, 3]
        >>> print(regroup_by(items))
        {1: [1, 1, 1, 1], 2: [2, 2, 2, 2], 3: [3, 3, 3, 3]}

        >>> items = [1, 2, 3, 0, 3, 1, 2, 0, 0, 2, 1, 3]
        >>> print(regroup_by(items, lambda x: x % 2))
        {1: [1, 3, 3, 1, 1, 3], 0: [2, 0, 2, 0, 0, 2]}

        >>> items = range(16)
        >>> print(regroup_by(items, lambda x: x % 2))
        {0: [0, 2, 4, 6, 8, 10, 12, 14], 1: [1, 3, 5, 7, 9, 11, 13, 15]}

    See Also:
        - flyingcircus.split()
        - flyingcircus.chunks()
        - flyingcircus.split_by()
        - flyingcircus.group_by()
    """
    # : defaultdict-based solution
    # result = collections.defaultdict(list)
    # callable_key = callable(key)
    # for item in items:
    #     result[key(item) if callable_key else item].append(item)
    # return dict(result)

    # : setdefault-based solution
    # result = {}
    # callable_key = callable(key)
    # for item in items:
    #     key_value = key(item) if callable_key else item
    #     result.setdefault(key_value, []).append(item)
    # return result

    result = {}
    callable_key = callable(key)
    for item in items:
        key_value = key(item) if callable_key else item
        if key_value not in result:
            result[key_value] = []
        result[key_value].append(item)
    return result


# ======================================================================
def rolling(
        seq,
        size,
        func,
        update=None,
        fill=None):
    """
    Compute a rolling function on a sequence.

    The function is expressed in terms of an initialization and an update.

    The following pairs can be used for notable rolling computations:
     - rolling mean
       - func: mean
       - update: lambda x, a, b, n: x + a / n - b / n

    Args:
        seq (Sequence[Any]): The input sequence.
        size (int): The rolling window.
        func (callable): The function to compute.
            Must have the following signature: func(Sequence): Any
            If `update` is a callable provided, this serves as initialization.
        update (callable|None): The updating function.
            Must have the following signature:
            func(last_value, new_item, old_item, size): Any
        fill (Any|None): The filling value.
            If None, the rolling starts and stops at the edges of the sequence.
            Otherwise, the fill value is used to compute partial windows
            at the edges.

    Yields:
        value: The next rolling value.

    Examples:
        >>> print(list(rolling(range(16), 2, sum)))
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        >>> print(list(rolling(range(16), 3, sum)))
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42]
        >>> print(list(rolling(range(16), 4, sum)))
        [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50, 54]

        >>> print(list(rolling([1] * 16, 4, sum)))
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        >>> print(list(rolling([1] * 16, 4, sum, fill=0)))
        [0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 2, 1, 0]
        >>> print(list(rolling([1] * 16, 4, sum, fill=1)))
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
        >>> print(list(rolling([1] * 8, 8, sum, fill=0)))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0]
        >>> print(list(rolling([1] * 8, 8, sum, fill=1)))
        [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]

        >>> print(list(rolling([1] * 8, 8, sum)))
        [8]
        >>> print(list(rolling([1] * 8, 9, sum)))
        []
        >>> print(list(rolling([1] * 8, 9, sum, fill=0)))
        []

        >>> def sum_update(x, a, b, n): return x + a - b
        >>> print(list(rolling(range(16), 2, sum)))
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]
        >>> print(list(rolling(range(16), 2, sum, sum_update)))
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29]

        >>> def mean_update(x, a, b, n): return x + a / n - b / n
        >>> print(list(rolling(range(12), 2, mean)))
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
        >>> print(list(rolling(range(12), 2, mean, mean_update)))
        [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]

        >>> def prod_update(x, a, b, n): return x * a / b
        >>> print(list(rolling(range(1, 13), 3, prod)))
        [6, 24, 60, 120, 210, 336, 504, 720, 990, 1320]
        >>> print(list(rolling(range(1, 13), 3, prod, prod_update)))
        [6, 24.0, 60.0, 120.0, 210.0, 336.0, 504.0, 720.0, 990.0, 1320.0]

        >>> def hash_func(seq): return sum(hash(x) for x in seq)
        >>> def hash_update(x, a, b, n): return x + hash(a) - hash(b)
        >>> print(list(rolling(range(12), 3, hash_func)))
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
        >>> print(list(rolling(range(12), 3, hash_func, hash_update)))
        [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

    See Also:
        - flyingcircus.slide()
        - flyingcircus.sliding()
    """
    n = len(seq)
    if n < size:
        return
    if update is not None:
        if fill is not None:
            value = func([fill] * size)
            yield value
            for i in range(size - 1):
                value = update(value, seq[i], fill)
                yield value
        value = func(seq[:size])
        yield value
        for i in range(1, n - size + 1):
            value = update(value, seq[i + size - 1], seq[i - 1], size)
            yield value
        if fill is not None:
            for i in range(n - size, n):
                value = update(value, fill, seq[i - 1])
                yield value
    else:
        if fill is not None:
            buffer = type(seq)([fill] * size)
            for i in range(size):
                yield func(buffer[i:] + seq[:i])
        for i in range(n - size + 1):
            yield func(seq[i:i + size])
        if fill is not None:
            for i in range(1, size + 1):
                yield func(seq[n - size + i:] + buffer[:i])


# ======================================================================
def combinations(
        seq,
        k,
        container=None):
    """
    Generate all possible k-combinations of given items.

    This is similar to `itertools.combinations()`.
    The `itertools` version should be preferred to this function.

    Args:
        seq (Sequence): The input items.
        k (int): The number of items to select.
        container (callable|None): The container for the result.
            If None, this is inferred from `seq` if possible, otherwise
            uses `tuple`.

    Yields:
        result (Sequence): The next output elements.

    Examples:
        >>> list(combinations(range(3), 2))
        [(0, 1), (0, 2), (1, 2)]

        >>> list(combinations(range(4), 0))
        []
        >>> list(combinations(range(4), 1))
        [(0,), (1,), (2,), (3,)]
        >>> list(combinations(range(4), 2))
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        >>> list(combinations(range(4), 3))
        [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
        >>> list(combinations(range(4), 4))
        [(0, 1, 2, 3)]
        >>> list(combinations(range(4), 5))
        []

        >>> list(combinations([], 0))
        []
        >>> list(combinations([], 1))
        []

        >>> list(combinations(range(2), 2))
        [(0, 1)]

        >>> list(combinations(tuple(range(3)), 2))
        [(0, 1), (0, 2), (1, 2)]
        >>> list(combinations(list(range(3)), 2))
        [[0, 1], [0, 2], [1, 2]]
        >>> list(combinations(range(3), 2))
        [(0, 1), (0, 2), (1, 2)]
        >>> list(combinations('abc', 2))
        ['ab', 'ac', 'bc']
        >>> list(combinations(b'abc', 2))
        [b'ab', b'ac', b'bc']

        >>> p1 = sorted(combinations(range(10), 3))
        >>> p2 = sorted(itertools.combinations(range(10), 3))
        >>> p1 == p2
        True

    See Also:
        - flyingcircus.multi_combinations()
        - flyingcircus.permutations()
        - flyingcircus.multi_permutations()
        - flyingcircus.cyclic_permutations()
        - flyingcircus.unique_permutations()
        - flyingcircus.cartesian_product()
    """
    container = _guess_container(seq, container)
    num = len(seq)
    if not num or k > num or k < 1:
        return
    indices = list(range(k))
    yield container(seq[i] for i in indices)
    while True:
        for i in range(k - 1, -1, -1):
            if indices[i] != i + num - k:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, k):
            indices[j] = indices[j - 1] + 1
        yield container(seq[i] for i in indices)


# ======================================================================
def multi_combinations(
        seq,
        k,
        container=None):
    """
    Generate all possible k-multi-combinations of given items.

    These are also called combinations with repetitions.

    This is similar to `itertools.combinations_with_replacement()`.
    The `itertools` version should be preferred to this function.

    Args:
        seq (Sequence): The input items.
        k (int): The number of items to select.
        container (callable|None): The container for the result.
            If None, this is inferred from `seq` if possible, otherwise
            uses `tuple`.

    Yields:
        result (Sequence): The next output elements.

    Examples:
        >>> list(multi_combinations(range(2), 0))
        []
        >>> list(multi_combinations(range(2), 1))
        [(0,), (1,)]
        >>> list(multi_combinations(range(2), 2))
        [(0, 0), (0, 1), (1, 1)]
        >>> list(multi_combinations(range(2), 3))
        [(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1)]
        >>> list(multi_combinations(range(2), 4))
        [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 1), (0, 1, 1, 1), (1, 1, 1, 1)]

        >>> list(multi_combinations([], 0))
        []
        >>> list(multi_combinations([], 1))
        []

        >>> list(multi_combinations(tuple(range(3)), 2))
        [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        >>> list(multi_combinations(list(range(3)), 2))
        [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]]
        >>> list(multi_combinations(range(3), 2))
        [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        >>> list(multi_combinations('abc', 2))
        ['aa', 'ab', 'ac', 'bb', 'bc', 'cc']
        >>> list(multi_combinations(b'abc', 2))
        [b'aa', b'ab', b'ac', b'bb', b'bc', b'cc']

        >>> p1 = sorted(multi_combinations(range(10), 3))
        >>> p2 = sorted(itertools.combinations_with_replacement(range(10), 3))
        >>> p1 == p2
        True

    See Also:
        - flyingcircus.combinations()
        - flyingcircus.permutations()
        - flyingcircus.multi_permutations()
        - flyingcircus.cyclic_permutations()
        - flyingcircus.unique_permutations()
        - flyingcircus.cartesian_product()
    """
    container = _guess_container(seq, container)
    num = len(seq)
    if not num or not k:
        return
    indices = [0] * k
    yield container(seq[i] for i in indices)
    while True:
        for i in range(k - 1, -1, -1):
            if indices[i] != num - 1:
                break
        else:
            return
        indices[i:] = [indices[i] + 1] * (k - i)
        yield container(seq[i] for i in indices)


# ======================================================================
def permutations(
        seq,
        k=None,
        container=None):
    """
    Generate all possible k-permutations of given items.

    If k is smaller than the number of input items, generates partial
    permutations.

    This is similar to `itertools.permutations()`.
    The `itertools` version should be preferred to this function.

    Args:
        seq (Sequence): The input items.
        k (int|None): The number of items to select.
            If k is None, all permutations are generated.
        container (callable|None): The container for the result.
            If None, this is inferred from `seq` if possible, otherwise
            uses `tuple`.

    Yields:
        result (Sequence): The next output elements.

    Examples:
        >>> list(permutations(range(3), 0))
        []
        >>> list(permutations(range(3), 1))
        [(0,), (1,), (2,)]
        >>> list(permutations(range(3), 2))
        [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        >>> list(permutations(range(3), 3))
        [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        >>> list(permutations(range(3), 4))
        []

        >>> list(permutations([], 0))
        []
        >>> list(permutations([], 1))
        []

        >>> list(permutations(tuple(range(3)), 2))
        [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        >>> list(permutations(list(range(3)), 2))
        [[0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1]]
        >>> list(permutations(range(3), 2))
        [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        >>> list(permutations('abc', 2))
        ['ab', 'ac', 'ba', 'bc', 'ca', 'cb']
        >>> list(permutations(b'abc', 2))
        [b'ab', b'ac', b'ba', b'bc', b'ca', b'cb']

        >>> p1 = sorted(permutations(range(10), 3))
        >>> p2 = sorted(itertools.permutations(range(10), 3))
        >>> p1 == p2
        True

    See Also:
        - flyingcircus.combinations()
        - flyingcircus.multi_combinations()
        - flyingcircus.multi_permutations()
        - flyingcircus.cyclic_permutations()
        - flyingcircus.unique_permutations()
        - flyingcircus.cartesian_product()
    """
    container = _guess_container(seq, container)
    num = len(seq)
    k = num if k is None else k
    if k > num or k < 1 or num < 1:
        return
    indices = list(range(num))
    cycles = list(range(num, num - k, -1))
    yield container(seq[i] for i in indices[:k])
    while num:
        for i in range(k - 1, -1, -1):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i + 1:] + indices[i:i + 1]
                cycles[i] = num - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield container(seq[i] for i in indices[:k])
                break
        else:
            return


# ======================================================================
def multi_permutations(
        seq,
        k=None,
        container=None):
    """
    Generate all possible k-multi-permutations of given items.

    These are also called permutations with repetitions or k-cartesian product.

    This is similar to `itertools.product()`.
    The `itertools` version should be preferred to this function.

    Args:
        seq (Sequence): The input items.
        k (int|None): The number of items to select.
            If k is None, all permutations are generated.
        container (callable|None): The container for the result.
            If None, this is inferred from `seq` if possible, otherwise
            uses `tuple`.

    Yields:
        result (Sequence): The next output elements.

    Examples:
        >>> list(multi_permutations(range(2), 0))
        []
        >>> list(multi_permutations(range(2), 1))
        [(0,), (1,)]
        >>> list(multi_permutations(range(2), 2))
        [(0, 0), (0, 1), (1, 0), (1, 1)]
        >>> list(multi_permutations(range(2), 3))
        [(0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1),\
 (1, 1, 0), (1, 1, 1)]

        >>> list(multi_permutations([], 0))
        []
        >>> list(multi_permutations([], 1))
        []

        >>> list(multi_permutations(tuple(range(3)), 2))
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1),\
 (2, 2)]
        >>> list(multi_permutations(list(range(3)), 2))
        [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1],\
 [2, 2]]
        >>> list(multi_permutations(range(4), 1))
        [(0,), (1,), (2,), (3,)]
        >>> list(multi_permutations('abc', 2))
        ['aa', 'ab', 'ac', 'ba', 'bb', 'bc', 'ca', 'cb', 'cc']
        >>> list(multi_permutations(b'abc', 2))
        [b'aa', b'ab', b'ac', b'ba', b'bb', b'bc', b'ca', b'cb', b'cc']

        >>> p1 = sorted(multi_permutations(range(10), 3))
        >>> p2 = sorted(itertools.product(range(10), repeat=3))
        >>> p1 == p2
        True

    See Also:
        - flyingcircus.combinations()
        - flyingcircus.multi_combinations()
        - flyingcircus.permutations()
        - flyingcircus.cyclic_permutations()
        - flyingcircus.unique_permutations()
        - flyingcircus.cartesian_product()
    """
    container = _guess_container(seq, container)
    num = len(seq)
    if not num or k < 1:
        return
    indices = [0] * k
    yield container(seq[i] for i in indices)
    while True:
        for i in range(k - 1, -1, -1):
            if indices[i] != num - 1:
                break
            else:
                for j in range(i, k):
                    indices[j] = 0
        else:
            return
        indices[i] += 1
        yield container(seq[i] for i in indices)


# ======================================================================
def cyclic_permutations(
        seq,
        forward=True):
    """
    Generate cyclic permutations of given items.

    Args:
        seq (Sequence): The input items.
        forward (bool): Determine how to advance through permutations.

    Yields:
        result (Sequence): The next output elements.

    See Also:
        - flyingcircus.combinations()
        - flyingcircus.multi_combinations()
        - flyingcircus.permutations()
        - flyingcircus.multi_permutations()
        - flyingcircus.unique_permutations()
        - flyingcircus.cartesian_product()

    Examples:
        >>> list(cyclic_permutations(tuple(range(4))))
        [(0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)]
        >>> list(cyclic_permutations(list(range(4))))
        [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]

        >>> list(cyclic_permutations(list(range(3)), True))
        [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
        >>> list(cyclic_permutations(list(range(3)), False))
        [[0, 1, 2], [2, 0, 1], [1, 2, 0]]

        >>> list(cyclic_permutations('abcdef', True))
        ['abcdef', 'bcdefa', 'cdefab', 'defabc', 'efabcd', 'fabcde']
        >>> list(cyclic_permutations('abcdef', False))
        ['abcdef', 'fabcde', 'efabcd', 'defabc', 'cdefab', 'bcdefa']
        >>> list(cyclic_permutations(b'abcdef', True))
        [b'abcdef', b'bcdefa', b'cdefab', b'defabc', b'efabcd', b'fabcde']
        >>> list(cyclic_permutations(b'abcdef', False))
        [b'abcdef', b'fabcde', b'efabcd', b'defabc', b'cdefab', b'bcdefa']
    """
    sign = 1 if forward else -1
    for i in range(len(seq)):
        yield seq[sign * i:] + seq[:sign * i]


# ======================================================================
def unique_permutations(
        seq,
        container=None):
    """
    Generate unique permutations of items in an efficient way.

    If items does not contain repeating elements, this is equivalent to
    `flyingcircus.permutations()`.

    Args:
        seq (Sequence): The input items.
        container (callable|None): The container for the result.
            If None, this is inferred from `seq` if possible, otherwise
            uses `tuple`.

    Yields:
        result (Sequence): The next output elements.

    Examples:
        >>> list(unique_permutations([0, 0, 0]))
        [[0, 0, 0]]
        >>> list(itertools.permutations([0, 0, 0]))
        [(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]

        >>> list(unique_permutations([0, 0, 2]))
        [[0, 0, 2], [0, 2, 0], [2, 0, 0]]
        >>> list(itertools.permutations([0, 0, 2]))
        [(0, 0, 2), (0, 2, 0), (0, 0, 2), (0, 2, 0), (2, 0, 0), (2, 0, 0)]

        >>> list(unique_permutations([0, 1, 2]))
        [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        >>> list(permutations([0, 1, 2]))
        [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        >>> list(itertools.permutations([0, 1, 2]))
        [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]

        >>> list(unique_permutations([]))
        [[]]

        >>> list(unique_permutations('aabb'))
        ['aabb', 'abab', 'abba', 'baab', 'baba', 'bbaa']
        >>> list(unique_permutations(b'aabb'))
        [b'aabb', b'abab', b'abba', b'baab', b'baba', b'bbaa']

        >>> p1 = sorted(unique_permutations(tuple(range(8))))
        >>> p2 = sorted(itertools.permutations(tuple(range(8))))
        >>> p1 == p2
        True

    References:
        - Donald Knuth, The Art of Computer Programming, Volume 4, Fascicle
          2: Generating All Permutations.

    See Also:
        - flyingcircus.combinations()
        - flyingcircus.multi_combinations()
        - flyingcircus.permutations()
        - flyingcircus.multi_permutations()
        - flyingcircus.cyclic_permutations()
        - flyingcircus.cartesian_product()
    """
    container = _guess_container(seq, container)
    indexes = range(len(seq) - 1, -1, -1)
    seq = sorted(seq)
    while True:
        yield container(seq)
        for k in indexes[1:]:
            if seq[k] < seq[k + 1]:
                break
        else:
            return
        k_val = seq[k]
        for i in indexes:
            if k_val < seq[i]:
                break
        else:
            i = 0
        seq[k], seq[i] = seq[i], seq[k]
        seq[k + 1:] = seq[-1:k:-1]


# ======================================================================
def cartesian_product(
        *seqs,
        k=1,
        container=None):
    """
    Generate the cartesian product of the input sequences.

    This is similar to `itertools.product()`.
    The `itertools` version should be preferred to this function.

    Args:
        *seqs (Sequence[Sequence]): The input sequences.
        k (int): Repetition factor for the input sequences.
            The input sequences are repeated `k` times.
        container (callable|None): The container for the result.
            If None, this is inferred from `seq` if possible, otherwise
            uses `tuple`.

    Yields:
        result (Sequence): The next output elements.

    Examples:
        >>> sorted(cartesian_product([1, 2], [3, 4, 5]))
        [[1, 3], [1, 4], [1, 5], [2, 3], [2, 4], [2, 5]]
        >>> sorted(cartesian_product((1, 2), (3, 4, 5)))
        [(1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5)]
        >>> sorted(cartesian_product(range(2), range(3)))
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        >>> sorted(cartesian_product('abc', 'ABC'))
        ['aA', 'aB', 'aC', 'bA', 'bB', 'bC', 'cA', 'cB', 'cC']
        >>> sorted(cartesian_product([1, 2], k=2))
        [[1, 1], [1, 2], [2, 1], [2, 2]]
        >>> sorted(cartesian_product('ab', k=3))
        ['aaa', 'aab', 'aba', 'abb', 'baa', 'bab', 'bba', 'bbb']
        >>> sorted(cartesian_product('ab', 'c', k=2))
        ['acac', 'acbc', 'bcac', 'bcbc']
        >>> sorted(cartesian_product('ab', 'cd', k=2))
        ['acac', 'acad', 'acbc', 'acbd', 'adac', 'adad', 'adbc', 'adbd',\
 'bcac', 'bcad', 'bcbc', 'bcbd', 'bdac', 'bdad', 'bdbc', 'bdbd']

    See Also:
        - flyingcircus.combinations()
        - flyingcircus.multi_combinations()
        - flyingcircus.permutations()
        - flyingcircus.multi_permutations()
        - flyingcircus.cyclic_permutations()
        - flyingcircus.unique_permutations()
    """
    containers = [_guess_container(seq, container) for seq in seqs]
    container = tuple if not all_equal(containers) else containers[0]
    if k > 1:
        seqs = seqs * k

    # : iterative method (fast but very memory-consuming method)
    # results = ((),)
    # for pool in seqs:
    #     results = tuple(x + (y,) for x in results for y in pool)
    # for result in results:
    #     yield result

    # : recursive version (fast but very memory-consuming method)
    # if not seqs:
    #     yield ()
    # else:
    #     for item in seqs[0]:
    #         for items in cartesian_product(*seqs[1:]):
    #             yield (item,) + items

    # # : iterative method with explicit indexing
    # len_seqs = list(map(len, seqs))
    # indexes = [0] * len(seqs)
    # while True:
    #     yield container(seq[i] for seq, i in zip(seqs, indexes))
    #     j = n - 1
    #     while True:
    #         indexes[j] += 1
    #         if indexes[j] < len_seqs[j]:
    #             break
    #         indexes[j] = 0
    #         j -= 1
    #         if j < 0:
    #             return

    # : iterative method with modulo arithmetics (faster but backward loops)
    # i = 0
    # while True:
    #     result = []
    #     k = i
    #     for seq in seqs:
    #         m = len(seq)
    #         result.append(seq[k % m])
    #         k //= m
    #     if k > 0:
    #         return
    #     else:
    #         yield container(result)
    #         i += 1

    # : iterative method with modulo arithmetics
    n = len(seqs)
    r_seqs = list(reversed(seqs))
    seqs = list(zip(r_seqs, list(map(len, r_seqs))))
    i = 0
    result = [None] * n
    while True:
        k = i
        for j, (seq, m) in enumerate(seqs):
            result[n - j - 1] = seq[k % m]
            k //= m
        if k > 0:
            return
        else:
            yield container(result)
            i += 1


# ======================================================================
def partitions(
        seq,
        k,
        container=None):
    """
    Generate all k-partitions for the items.

    Args:
        seq (Sequence): The input items.
        k (int): The number of splitting partitions.
            Each group has exactly `k` elements.
        container (callable|None): The container for the result.
            If None, this is inferred from `seq` if possible, otherwise
            uses `tuple`.

    Yields:
        partition (Sequence[Sequence]]): The grouped items.
            Each partition contains `k` grouped items from the source.

    Examples:
        >>> items = tuple(range(3))
        >>> tuple(partitions(items, 2))
        (((0,), (1, 2)), ((0, 1), (2,)))
        >>> tuple(partitions(items, 3))
        (((0,), (1,), (2,)),)

        >>> items = 'abc'
        >>> tuple(partitions(items, 2))
        (('a', 'bc'), ('ab', 'c'))
        >>> tuple(partitions(items, 3))
        (('a', 'b', 'c'),)

        >>> items = range(3)
        >>> tuple(partitions(items, 2))
        ((range(0, 1), range(1, 3)), (range(0, 2), range(2, 3)))
        >>> tuple(partitions(items, 3))
        ((range(0, 1), range(1, 2), range(2, 3)),)

        >>> tuple(partitions(tuple(range(4)), 3))
        (((0,), (1,), (2, 3)), ((0,), (1, 2), (3,)), ((0, 1), (2,), (3,)))
        >>> tuple(partitions(list(range(4)), 3))
        ([[0], [1], [2, 3]], [[0], [1, 2], [3]], [[0, 1], [2], [3]])
    """
    if container is None:
        container = type(seq)
    if not callable(container) or container in (str, bytes, bytearray):
        container = tuple
    try:
        container(seq)
    except TypeError:
        container = tuple
    num = len(seq)
    indexes = tuple(
        (0,) + tuple(index) + (num,)
        for index in itertools.combinations(range(1, num), k - 1))
    for index in indexes:
        yield container(seq[index[i]:index[i + 1]] for i in range(k))


# ======================================================================
def random_unique_combinations_k(
        seq,
        k,
        container=None,
        pseudo=False):
    """
    Obtain a number of random unique combinations of a sequence of sequences.

    Args:
        seq (Sequence[Sequence]): The input sequence of sequences.
        k (int): The number of random unique combinations to obtain.
        container (callable|None): The container for the result.
            If None, this is inferred from `seq` if possible, otherwise
            uses `tuple`.
        pseudo (bool): Generate random combinations somewhat less randomly.
            If True, the memory requirements for intermediate steps will
            be significantly lower (but still all `k` items are required to
            fit in memory).

    Yields:
        combination (Sequence): The next random unique combination.

    Examples:
        >>> import string
        >>> max_lens = list(range(2, 10))
        >>> items = tuple(
        ...     string.ascii_lowercase[:max_len] for max_len in max_lens)
        >>> random.seed(0)
        >>> num = 10
        >>> for i in random_unique_combinations_k(items, num):
        ...     print(i)
        ('b', 'a', 'd', 'a', 'd', 'a', 'a', 'f')
        ('a', 'a', 'c', 'c', 'b', 'f', 'd', 'f')
        ('b', 'b', 'b', 'e', 'c', 'b', 'e', 'a')
        ('a', 'b', 'a', 'b', 'd', 'g', 'c', 'd')
        ('b', 'c', 'd', 'd', 'b', 'b', 'f', 'g')
        ('a', 'a', 'b', 'a', 'f', 'd', 'c', 'g')
        ('a', 'c', 'd', 'a', 'f', 'a', 'c', 'f')
        ('b', 'c', 'd', 'a', 'f', 'd', 'h', 'd')
        ('a', 'c', 'b', 'b', 'a', 'e', 'b', 'g')
        ('a', 'c', 'c', 'b', 'e', 'b', 'f', 'e')
        >>> max_lens = list(range(2, 4))
        >>> items = tuple(
        ...     string.ascii_uppercase[:max_len] for max_len in max_lens)
        >>> random.seed(0)
        >>> num = 10
        >>> for i in random_unique_combinations_k(items, num):
        ...     print(i)
        ('B', 'B')
        ('B', 'C')
        ('A', 'A')
        ('B', 'A')
        ('A', 'B')
        ('A', 'C')
    """
    if container is None:
        container = type(seq)
    if not callable(container):
        container = tuple
    if pseudo:
        # randomize generators
        comb_gens = list(seq)
        for num, comb_gen in enumerate(comb_gens):
            shuffle(list(comb_gens[num]))
        # get the first `k` combinations
        combs = list(itertools.islice(itertools.product(*comb_gens), k))
        shuffle(combs)
        for combination in itertools.islice(combs, k):
            yield container(combination)
    else:
        max_lens = [len(list(item)) for item in seq]
        max_k = prod(max_lens)
        try:
            for num in random.sample(range(max_k), min(k, max_k)):
                indexes = []
                for max_len in max_lens:
                    indexes.append(num % max_len)
                    num = num // max_len
                yield container(item[i] for i, item in zip(indexes, seq))
        except OverflowError:
            # use `set` to ensure uniqueness
            index_combs = set()
            # make sure that with the chosen number the next loop can exit
            # WARNING: if `k` is too close to the total number of combinations,
            # it may take a while until the next valid combination is found
            while len(index_combs) < min(k, max_k):
                index_combs.add(
                    tuple(random_int(max_len) for max_len in max_lens))
            # make sure their order is shuffled
            # (`set` seems to sort its content)
            index_combs = list(index_combs)
            shuffle(index_combs)
            for index_comb in itertools.islice(index_combs, k):
                yield container(item[i] for i, item in zip(index_comb, seq))


# ======================================================================
def unique_partitions(
        seq,
        k,
        container=None):
    """
    Generate all k-partitions for all unique permutations of the items.

    Args:
        seq (Sequence): The input items.
        k (int): The number of splitting partitions.
            Each group has exactly `k` elements.
        container (callable|None): The container for the result.
            If None, this is inferred from `seq` if possible, otherwise
            uses `tuple`.

    Yields:
        partitions (Sequence[Sequence[Sequence]]]): The items partitions.
            More precisely, all partitions of size `num` for each unique
            permutations of `items`.

    Examples:
        >>> list(unique_partitions([0, 1], 2))
        [[[[0], [1]]], [[[1], [0]]]]
        >>> tuple(unique_partitions((0, 1), 2))
        ((((0,), (1,)),), (((1,), (0,)),))
    """
    if container is None:
        container = type(seq)
    if not callable(container):
        container = tuple
    for perms in unique_permutations(seq):
        yield container(partitions(container(perms), k))


# ======================================================================
def latin_square(
        seq,
        randomize=True,
        cyclic=True,
        forward=True):
    """
    Generate a latin square.

    Args:
        seq (Sequence): The input items.
        randomize (bool): Shuffle the output "rows".
        cyclic (bool): Generate cyclic permutations only.
        forward (bool): Determine how to advance through permutations.
            Note that for cyclic permutations (`cyclic == True`), this means
            that the cycling is moving forward (or not),
            whereas for non-cyclic permutations (`cyclic == False`), this is
            equivalent to reversing the items order prior to generating the
            permutations, i.e.
            `permutations(reversed(items)) == reversed(permutations(items))`.

    Returns:
        result (list[Sequence]): The latin square.

    Examples:
        >>> random.seed(0)
        >>> latin_square(list(range(4)))
        [[2, 3, 0, 1], [3, 0, 1, 2], [1, 2, 3, 0], [0, 1, 2, 3]]

        >>> latin_square(list(range(4)), False, False, False)
        [[3, 2, 1, 0], [2, 3, 0, 1], [1, 0, 3, 2], [0, 1, 2, 3]]
        >>> latin_square(list(range(4)), False, True, False)
        [[0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1], [1, 2, 3, 0]]
        >>> latin_square(list(range(4)), False, False, True)
        [[0, 1, 2, 3], [1, 0, 3, 2], [2, 3, 0, 1], [3, 2, 1, 0]]
        >>> latin_square(list(range(4)), False, True, True)
        [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]]

        >>> random.seed(0)
        >>> latin_square(list(range(4)), True, False, False)
        [[1, 0, 3, 2], [0, 1, 2, 3], [2, 3, 0, 1], [3, 2, 1, 0]]
        >>> random.seed(0)
        >>> latin_square(list(range(4)), True, True, False)
        [[2, 3, 0, 1], [1, 2, 3, 0], [3, 0, 1, 2], [0, 1, 2, 3]]
        >>> random.seed(0)
        >>> latin_square(list(range(4)), True, False, True)
        [[2, 3, 0, 1], [3, 2, 1, 0], [1, 0, 3, 2], [0, 1, 2, 3]]
        >>> random.seed(0)
        >>> latin_square(list(range(4)), True, True, True)
        [[2, 3, 0, 1], [3, 0, 1, 2], [1, 2, 3, 0], [0, 1, 2, 3]]

        >>> random.seed(0)
        >>> latin_square(tuple(range(4)))
        [(2, 3, 0, 1), (3, 0, 1, 2), (1, 2, 3, 0), (0, 1, 2, 3)]

        >>> random.seed(0)
        >>> latin_square('abcde')
        ['eabcd', 'abcde', 'deabc', 'bcdea', 'cdeab']
        >>> print('\\n'.join(latin_square('0123456789')))
        7890123456
        8901234567
        5678901234
        9012345678
        3456789012
        4567890123
        1234567890
        2345678901
        0123456789
        6789012345
    """
    if cyclic:
        result = list(cyclic_permutations(seq, forward))
    else:
        result = []
        # note: reversed(permutations(items)) == permutations(reversed(items))
        if not forward:
            seq = list(reversed(seq))
        for elems in permutations(seq):
            valid = True
            for i, elem in enumerate(elems):
                orthogonals = [x[i] for x in result] + [elem]
                if len(set(orthogonals)) < len(orthogonals):
                    valid = False
                    break
            if valid:
                result.append(elems)
    if randomize:
        shuffle(result)
    return result


# ======================================================================
def is_sorted(
        items,
        compare=lambda x, y: x <= y,
        both=True):
    """
    Determine if an iterable is sorted or not.

    Args:
        items (Iterable): The input items.
        compare (callable): The comparison function.
            Must have the signature: compare(Any, Any): bool.
        both (bool): Compare the items both forward and backward.

    Returns:
        result (bool): The result of the comparison for all consecutive pairs.

    Examples:
        >>> items = list(range(10))
        >>> print(items)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> is_sorted(items)
        True
        >>> is_sorted(items[::-1])
        True
        >>> is_sorted(items[::-1], both=False)
        False
        >>> is_sorted(items[::-1], lambda x, y: x > y, both=False)
        True

        >>> items = [1] * 10
        >>> print(items)
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        >>> is_sorted(items)
        True
        >>> is_sorted(items, lambda x, y: x < y)
        False

        >>> is_sorted([0, 1, 2, 4, 4, 5, 6, 7, 8, 9])
        True
        >>> is_sorted([0, 1, 2, 5, 4, 5, 6, 7, 8, 9])
        False
        >>> is_sorted([])
        True
        >>> is_sorted(iter([0, 1, 2, 4, 4, 5, 6, 7, 8, 9]))
        True
        >>> is_sorted(iter([0, 1, 2, 5, 4, 5, 6, 7, 8, 9]))
        False

    See Also:
        - flyingcircus.pairwise_map()
    """
    result = all(pairwise_map(compare, items, False))
    if both:
        result = result or all(pairwise_map(compare, items, True))
    return result


# ======================================================================
def search_sorted(
        seq,
        item):
    """
    Search for an item in a sorted sequence.

    If the sequence is not sorted the results are unpredictable.
    Internally, it uses a binary search, which is the theoretical optimal
    algorithm for searching sorted sequences.

    Note that if the item is present in the sequence, for built-in sequences,
    the `.index()` method may be faster.

    Args:
        seq (Sequence): The input sequence.
        item (Any): The item to search for.

    Returns:
        result (int): The matching index.
            If `item` is in `seq`, this is the index at which `item` is found,
            i.e. `item == seq[result]`
            Otherwise, this is the index at which inserting `item` in `seq`
            would result in a sorted sequence.

    Examples:
        >>> search_sorted(list(range(1000)), 500)
        500
        >>> search_sorted([x ** 2 for x in range(100)], 500)
        23

        >>> items = [x ** 2 for x in range(10)]
        >>> print(items)
        [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

        >>> value = 9
        >>> i = search_sorted(items, value)
        >>> print(items[i] == value)
        True
        >>> new_items = items[:i] + [value] + items[i:]
        >>> print(new_items)
        [0, 1, 4, 9, 9, 16, 25, 36, 49, 64, 81]
        >>> print(is_sorted(new_items))
        True

        >>> value = 5
        >>> i = search_sorted(items, value)
        >>> print(items[i] == value)
        False
        >>> new_items = items[:i] + [value] + items[i:]
        >>> print(new_items)
        [0, 1, 4, 5, 9, 16, 25, 36, 49, 64, 81]
        >>> print(is_sorted(new_items))
        True

        >>> all(search_sorted(items, x) == items.index(x) for x in items)
        True

        >>> search_sorted([], 'ciao')
        0
        >>> search_sorted(string.ascii_lowercase, 'x')
        23

        These fails because the input is not sorted!
        >>> items = [1, 4, 6, 8, 2, 3, 5]
        >>> value = 5
        >>> search_sorted(items, value) == items.index(value)
        False
        >>> search_sorted(string.ascii_letters, 'X')
        0

    See Also:
        - flyingcircus.is_sorted()
    """
    first = 0
    last = len(seq) - 1
    found = False
    while first <= last and not found:
        midpoint = (first + last) // 2
        if seq[midpoint] == item:
            first = midpoint
            found = True
        else:
            if item < seq[midpoint]:
                last = midpoint - 1
            else:
                first = midpoint + 1
    return first


# ======================================================================
def find_subseq(
        seq,
        subseq,
        first=0,
        last=-1):
    """
    Find occurrences of a sub-sequence in a sequence.

    Args:
        seq (Sequence): The input sequence.
        subseq (Sequence): The input sub-sequence.
        first (int): The first index.
            The index is forced within boundaries.
        last (int): The last index (included).
            The index is forced within boundaries.

    Yields:
        result (int): The index of the next match.

    Examples:
        >>> list(find_subseq(list(range(10)), list(range(5, 7))))
        [5]
        >>> list(find_subseq(list(range(10)), []))
        []
        >>> list(find_subseq([], list(range(5, 7))))
        []
        >>> list(find_subseq([], []))
        []
        >>> list(find_subseq(list(range(10)), list(range(5, 12))))
        []
        >>> list(find_subseq(list(range(10)), list(range(-2, 5))))
        []
        >>> list(find_subseq(list(range(10)), list(range(-5, -1))))
        []
        >>> list(find_subseq(list(range(10)), list(range(11, 12))))
        []
        >>> list(find_subseq(list(range(10)), list(range(3, 8))))
        [3]
        >>> list(find_subseq(list(range(10)), list(range(5))))
        [0]
        >>> list(find_subseq(list(range(10)), list(range(5, 10))))
        [5]
        >>> list(find_subseq(list(range(10)) * 10, list(range(3, 8))))
        [3, 13, 23, 33, 43, 53, 63, 73, 83, 93]
        >>> list(find_subseq(list(range(10)) * 10, list(range(3, 8)), 10, 90))
        [13, 23, 33, 43, 53, 63, 73, 83]

        >>> seq = '12431243123431212'
        >>> list(find_subseq(seq, '1234'))
        [8]
        >>> list(find_subseq(seq, '12'))
        [0, 4, 8, 13, 15]
        >>> list(find_subseq(seq, '12')) == list(find_all(seq, '12'))
        True

    See Also:
        - flyingcircus.find_all()
    """
    n = len(seq)
    m = len(subseq)
    if n > 0 and m > 0:
        first = valid_index(first, n)
        last = valid_index(last, n)

        # # : naive with fast looping, slicing and short-circuit
        # for i in index_all(seq, subseq[0], first, last - m + 1):
        #     if seq[i + m - 1] == subseq[m - 1] and seq[i:i + m] == subseq:
        #         yield i

        # # : naive with looping
        # for i in range(first, last - m + 1):
        #     if all(seq[i + j] == subseq[j] for j in range(m)):
        #         yield i

        # # : naive with slicing
        # for i in range(first, last - m + 1):
        #     if seq[i:i + m] == subseq:
        #         yield i

        # # : naive with slicing and short-circuit
        # for i in range(first, last - m + 1):
        #     if seq[i] == subseq[0] and seq[i:i + m] == subseq:
        #         yield i

        # : Knuth–Morris–Pratt (KMP) algorithm
        offsets = [0] * m
        j = 1
        k = 0
        while j < m:
            if subseq[j] == subseq[k]:
                k += 1
                offsets[j] = k
                j += 1
            else:
                if k != 0:
                    k = offsets[k - 1]
                else:
                    offsets[j] = 0
                    j += 1
        i = first
        j = 0
        while i <= last:
            if seq[i] == subseq[j]:
                i += 1
                j += 1
            if j == m:
                yield i - j
                j = offsets[j - 1]
            elif i <= last and seq[i] != subseq[j]:
                if j != 0:
                    j = offsets[j - 1]
                else:
                    i += 1

        # # : Rabin–Karp (RK) algorithm
        # if seq[first:first + m] == subseq:
        #     yield 0
        # hash_subseq = sum(hash(x) for x in subseq)
        # curr_hash = sum(hash(x) for x in seq[first:first + m])
        # for i in range(first + 1, last - m + 2):
        #     curr_hash += hash(seq[i + m - 1]) - hash(seq[i - 1])
        #     if hash_subseq == curr_hash and seq[i:i + m] == subseq:
        #         yield i
    else:
        return


# ======================================================================
def seqmap2mapseq(
        data,
        labels=None,
        d_val=None,
        mapping_container=None,
        sequence_container=None):
    """
    Convert tabular data from a Sequence of Mappings to a Mapping of Sequences.

    Args:
        data (Sequence[Mapping]): The input tabular data.
        labels (Sequence|None): The labels of the tabular data.
            If Sequence, all elements should be present as keys of the dicts.
            If the dicts contain keys not specified in `labels` they will be
            ignored.
            If None, the `labels` are guessed from the data.
        d_val (Any): The default value to use for incomplete dicts.
            This will be inserted in the lists to keep track of missing data.
        mapping_container (callable|None): The mapping container.
            If None, this is inferred from `data` if possible, otherwise
            uses `dict`.
        sequence_container (callable|None): The sequence container.
            If None, this is inferred from `data` if possible, otherwise
            uses `list`.

    Returns:
        result (Mapping[Sequence]): The output tabular data.
            All list will have the same size.

    Examples:
        >>> data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6}]
        >>> tuple(seqmap2mapseq(data).items())
        (('a', [1, 3, 5]), ('b', [2, 4, 6]))
        >>> data == mapseq2seqmap((seqmap2mapseq(data)))
        True

        >>> data = [{'a': 1}, {'b': 4}, {'b': 6}]
        >>> tuple(seqmap2mapseq(data).items())
        (('a', [1, None, None]), ('b', [None, 4, 6]))
        >>> data == mapseq2seqmap((seqmap2mapseq(data)))
        True

    See Also:
        - flyingcircus.mapseq2seqmap()
    """
    if mapping_container is None:
        mapping_container = type(next(iter(data)))
    if not callable(mapping_container):
        mapping_container = dict
    if sequence_container is None:
        sequence_container = type(data)
    if not callable(sequence_container):
        sequence_container = list

    if not labels:
        labels = sorted(functools.reduce(
            lambda x, y: x.union(y), [set(x.keys()) for x in data]))
    return mapping_container(
        (label,
         sequence_container(
             item[label] if label in item else d_val for item in data))
        for label in labels)


# ======================================================================
def mapseq2seqmap(
        data,
        labels=None,
        d_val=None,
        mapping_container=None,
        sequence_container=None):
    """
    Convert tabular data from a Mapping of Sequences to a Sequence of Mappings.

    Args:
        data (Mapping[Sequence]): The input tabular data.
            All lists must have the same length.
        labels (Iterable|None): The labels of the tabular data.
            If None, the `labels` are guessed from the data.
        d_val (Any): The default value to be used for reducing dicts.
            The values matching `d_val` are not included in the dicts.
        mapping_container (callable|None): The mapping container.
            If None, this is inferred from `data` if possible, otherwise
            uses `dict`.
        sequence_container (callable|None): The sequence container.
            If None, this is inferred from `data` if possible, otherwise
            uses `list`.

    Returns:
        result (dict[Any:list]): The tabular data as a dict of lists.
            All list will have the same size.

    Examples:
        >>> data = {'a': [1, 3, 5], 'b': [2, 4, 6]}
        >>> [sorted(d.items()) for d in mapseq2seqmap(data)]
        [[('a', 1), ('b', 2)], [('a', 3), ('b', 4)], [('a', 5), ('b', 6)]]
        >>> data == seqmap2mapseq((mapseq2seqmap(data)))
        True

        >>> data = {'a': [1, None, None], 'b': [None, 4, 6]}
        >>> [sorted(d.items()) for d in mapseq2seqmap(data)]
        [[('a', 1)], [('b', 4)], [('b', 6)]]
        >>> data == seqmap2mapseq((mapseq2seqmap(data)))
        True

    See Also:
        - flyingcircus.seqmap2mapseq()
    """
    if mapping_container is None:
        mapping_container = type(data)
    if not callable(mapping_container):
        mapping_container = dict
    if sequence_container is None:
        sequence_container = type(data[next(iter(data))])
    if not callable(sequence_container):
        sequence_container = list

    if not labels:
        labels = tuple(data.keys())
    num_elems = len(data[next(iter(labels))])
    return sequence_container(
        mapping_container(
            (label, data[label][i]) for label in labels
            if data[label][i] is not d_val)
        for i in range(num_elems))


# ======================================================================
def bits(value):
    """
    Get the bit values for a number (lower to higher significance).

    Args:
        value (int): The input value.

    Yields:
        result (int): The bits.

    Examples:
        >>> list(bits(100))
        [0, 0, 1, 0, 0, 1, 1]
    """
    # : this alternative may be faster for larger values
    # return map(int, bin(value)[:1:-1])
    while value:
        yield value & 1
        value >>= 1


# ======================================================================
def bits_r(value):
    """
    Get the reversed bit values for a number (higher to lower significance).

    Args:
        value (int): The input value.

    Yields:
        result (int): The bits.

    Examples:
        >>> list(bits(100))
        [0, 0, 1, 0, 0, 1, 1]
    """
    # : this alternative may be faster for larger values
    # return map(int, bin(value)[2:])
    b = value.bit_length()
    for i in range(b - 1, -1, -1):
        yield (value >> i) & 1


# ======================================================================
def get_bit(value, i):
    """
    Get the bit value for a number at a given position.

    Args:
        value (int): The input value.
        i (int): The bit position.

    Returns:
        result (int): The bit value.

    Examples:
        >>> get_bit(100, 2)
        1
        >>> [get_bit(100, i) for i in range(10)]
        [0, 0, 1, 0, 0, 1, 1, 0, 0, 0]
    """
    # : altenate form
    # return ((1 << i) & value) >> i
    return (value >> i) & 1


# ======================================================================
def put_bit(value, i, x):
    """
    Put a bit value on a number at a given position.

    Args:
        value (int): The input value.
        i (int): The bit position.
        x (int|bool): The bit value.

    Returns:
        result (int): The output value.

    Examples:
        >>> put_bit(100, 2, 1)
        100
        >>> put_bit(100, 2, 0)
        96
        >>> put_bit(100, 3, 1)
        108
        >>> put_bit(100, 3, 0)
        100
    """
    # : slower alternative
    # return (value & ~(1 << i)) | (x << i)
    if x:
        return value | (1 << i)
    else:
        return value & ~(1 << i)


# ======================================================================
def set_bit(value, i):
    """
    Set the bit value for a number at a given position.

    Args:
        value (int): The input value.
        i (int): The bit position.

    Returns:
        result (int): The output value.

    Examples:
        >>> set_bit(100, 2)
        100
        >>> set_bit(96, 2)
        100
    """
    return value | (1 << i)


# ======================================================================
def unset_bit(value, i):
    """
    Unset the bit value for a number at a given position.

    Args:
        value (int): The input value.
        i (int): The bit position.

    Returns:
        result (int): The output value.

    Examples:
        >>> unset_bit(100, 2)
        96
        >>> unset_bit(96, 2)
        96
    """
    return value & ~(1 << i)


# ======================================================================
def flip_bit(value, i):
    """
    Flip (toggle) the bit value for a number at a given position.

    Args:
        value (int): The input value.
        i (int): The bit position.

    Returns:
        result (int): The output value.

    Examples:
        >>> flip_bit(100, 2)
        96
        >>> flip_bit(96, 2)
        100

        >>> all(
        ...      x == flip_bit(flip_bit(x, j), j)
        ...      for x in range(1024) for j in range(10))
        True
    """
    return value ^ (1 << i)


# ======================================================================
def decode_bits(value, tokens):
    """
    Decode the bits for a number according to the specified tokens.

    Args:
        value (int): The input value.
        tokens (Sequence): The tokens for decoding.

    Yields:
        token (Any): The codified bit as token.

    Examples:
        >>> ''.join(decode_bits(5, 'abc'))
        'ac'
        >>> ''.join(decode_bits(6, 'rwx'))
        'wx'
    """
    for token, bit in zip(tokens, bits(value)):
        if bit:
            yield token


# ======================================================================
def encode_bits(items, tokens):
    """
    Encode the items into a number according to the specified tokens.

    Args:
        items (Iterable): The input items.
        tokens (Sequence): The tokens for encoding.

    Returns:
        result (int): The encoded number.

    Examples:
        >>> encode_bits('abc', 'abc')
        7
        >>> encode_bits('ac', 'abc')
        5
        >>> encode_bits('rw', 'rwx')
        3
        >>> encode_bits('rx', 'rwx')
        5
    """
    value = 0
    for item in items:
        value = set_bit(value, tokens.index(item))
    return value


# ======================================================================
def unravel_bits(value, tokens):
    """
    Unravel the bits for a number according to the specified tokens.

    This is similar to `decode_bits()` except that this yields additional info.

    Args:
        value (int): The input value.
        tokens (Sequence): The tokens for decoding.

    Yields:
        token (tuple): The position, token and bit value.
    """
    for i, token in enumerate(tokens):
        yield i, token, get_bit(value, i)


# ======================================================================
def alternate_sign(
        items,
        start=1):
    """
    Alternate the sign of arbitrary items.

    A similar result could be achieved with `itertools.cycle()` and `zip()`.

    Args:
        items (Iterable[Number]): The input items.
        start (Number): The initial value.
            In strict terms, this should be 1 or -1 but this is not enforced.

    Yields:
        item (Number): The item with alternating sign.

    Examples:
        >>> print(list(alternate_sign(range(16))))
        [0, -1, 2, -3, 4, -5, 6, -7, 8, -9, 10, -11, 12, -13, 14, -15]
        >>> print(list(alternate_sign(range(16), -1)))
        [0, 1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12, 13, -14, 15]
        >>> print(list(alternate_sign(range(16), 2)))
        [0, -2, 4, -6, 8, -10, 12, -14, 16, -18, 20, -22, 24, -26, 28, -30]
        >>> print(list(alternate_sign(itertools.repeat(1, 16))))
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
    """
    for item in items:
        yield start * item
        start = -start


# ======================================================================
def prod(
        items,
        start=1):
    """
    Compute the product of arbitrary items.

    This is similar to `sum`, but uses product instead of addition.

    Args:
        items (Iterable[Number]): The input items.
        start (Number): The initial value.

    Returns:
        result (Number): The cumulative product of `items`.

    Examples:
        >>> prod([2] * 10)
        1024
        >>> prod(range(1, 11))
        3628800

        >>> all(prod(range(1, n + 1)) == math.factorial(n) for n in range(10))
        True
    """
    for item in items:
        start *= item
    return start


# ======================================================================
def diff(
        items,
        reverse=False):
    """
    Compute the pairwise difference of arbitrary items.

    This is similar to `flyingcircus.div()`, but uses subtraction instead
    of division.

    Args:
        items (Iterable[Number]): The input items.
        reverse (bool): Reverse the order of the operands.

    Yields:
        value (Number): The next pairwise difference.


    Examples:
        >>> list(diff(range(10)))
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
        >>> list(diff(range(10), True))
        [-1, -1, -1, -1, -1, -1, -1, -1, -1]
    """
    # : equivalent (slower) implementation
    # return map(lambda x: x[1] - x[0], slide(items, 2, reverse=reverse))
    return pairwise_map(operator.sub, items, reverse=not reverse)


# ======================================================================
def div(
        items,
        reverse=False):
    """
    Compute the pairwise division of arbitrary items.

    This is similar to `flyingcircus.diff()`, but uses division instead
    of subtraction.

    Args:
        items (Iterable[Number]): The input items.
        reverse (bool): Reverse the order of the operands.

    Yields:
        value (Number): The next pairwise division.

    Examples:
        >>> items = [2 ** x for x in range(10)]
        >>> items
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        >>> list(div(items))
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        >>> list(div(items, True))
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    """
    # : equivalent (slower) implementation
    # return map(lambda x: x[1] / x[0], slide(items, 2, reverse=reverse))
    return pairwise_map(operator.truediv, items, reverse=not reverse)


# ======================================================================
def round_up(x):
    """
    Round to the largest close integer.

    To round down, just use `int()`

    Args:
        x (Number): The input number.

    Returns:
        x (int): The rounded-up integer.

    Examples:
        >>> round_up(10.4)
        11
        >>> round_up(10.9)
        11
        >>> round_up(11.0)
        11
        >>> round_up(-10.4)
        -11
        >>> round_up(-10.9)
        -11
        >>> round_up(-11.0)
        -11
    """
    int_x = int(x)
    frac_x = x % 1
    return int_x + ((1 if int_x > 0 else -1) if frac_x > 0 else 0)


# ======================================================================
def div_ceil(a, b):
    """
    Compute integer ceil division.

    This is such that:

    q = div_ceil(a, b)
    r = mod_ceil(a, b)
    q * b + r == a

    Args:
        a (int): The dividend.
        b (int): The divisor.

    Returns:
        q (int): The quotient.

    Examples:
        >>> div_ceil(6, 3)
        2
        >>> div_ceil(6, -3)
        -2
        >>> div_ceil(-6, 3)
        -2
        >>> div_ceil(-6, -3)
        2
        >>> div_ceil(7, 3)
        3
        >>> div_ceil(7, -3)
        -2
        >>> div_ceil(-7, 3)
        -2
        >>> div_ceil(-7, -3)
        3
        >>> div_ceil(3, 7)
        1
        >>> div_ceil(3, -7)
        0
        >>> div_ceil(-3, 7)
        0
        >>> div_ceil(-3, -7)
        1
        >>> div_ceil(1, 1)
        1
        >>> div_ceil(1, -1)
        -1
        >>> div_ceil(-1, 1)
        -1
        >>> div_ceil(-1, -1)
        1

    See Also:
        - operator.div()
        - operator.mod()
        - divmod()
        - flyingcircus.mod_ceil()
        - flyingcircus.divmod_ceil()
        - flyingcircus.div_round()
        - flyingcircus.mod_round()
        - flyingcircus.divmod_round()
    """
    q, r = divmod(a, b)
    if r:
        q += 1
    return q


# ======================================================================
def mod_ceil(a, b):
    """
    Compute integer ceil modulus.

    This is such that:

    q = div_ceil(a, b)
    r = mod_ceil(a, b)
    q * b + r == a

    Args:
        a (int): The dividend.
        b (int): The divisor.

    Returns:
        r (int): The remainder.

    Examples:
        >>> mod_ceil(6, 3)
        0
        >>> mod_ceil(6, -3)
        0
        >>> mod_ceil(-6, 3)
        0
        >>> mod_ceil(-6, -3)
        0
        >>> mod_ceil(7, 3)
        -2
        >>> mod_ceil(7, -3)
        1
        >>> mod_ceil(-7, 3)
        -1
        >>> mod_ceil(-7, -3)
        2
        >>> mod_ceil(3, 7)
        -4
        >>> mod_ceil(3, -7)
        3
        >>> mod_ceil(-3, 7)
        -3
        >>> mod_ceil(-3, -7)
        4
        >>> mod_ceil(1, 1)
        0
        >>> mod_ceil(1, -1)
        0
        >>> mod_ceil(-1, 1)
        0
        >>> mod_ceil(-1, -1)
        0

    See Also:
        - operator.div()
        - operator.mod()
        - divmod()
        - flyingcircus.div_ceil()
        - flyingcircus.divmod_ceil()
        - flyingcircus.div_round()
        - flyingcircus.mod_round()
        - flyingcircus.divmod_round()
    """
    q, r = divmod(a, b)
    if r:
        q += 1
    return a - q * b


# ======================================================================
def divmod_ceil(a, b):
    """
    Compute integer ceil division and modulus.

    This is such that:

    q, r = divmod_ceil(a, b)
    q * b + r == a

    Args:
        a (int): The dividend.
        b (int): The divisor.

    Returns:
        r (int): The remainder.

    Examples:
        >>> n = 100
        >>> l = list(range(-n, n + 1))
        >>> ll = [(a, b) for a, b in itertools.product(l, repeat=2) if b]
        >>> result = True
        >>> for a, b in ll:
        ...     q, r = divmod_ceil(a, b)
        ...     result = result and (q == div_ceil(a, b))
        ...     result = result and (r == mod_ceil(a, b))
        ...     result = result and (q * b + r == a)
        >>> print(result)
        True

    See Also:
        - operator.div()
        - operator.mod()
        - divmod()
        - flyingcircus.div_ceil()
        - flyingcircus.mod_ceil()
        - flyingcircus.div_round()
        - flyingcircus.mod_round()
        - flyingcircus.divmod_round()
    """
    q, r = divmod(a, b)
    if r:
        q += 1
    return q, a - q * b


# ======================================================================
def div_round(a, b):
    """
    Compute integer round division.

    This behaves the same as C `/` operator on integers.

    This is such that:

    q = div_round(a, b)
    r = mod_round(a, b)
    q * b + r == a

    Args:
        a (int): The dividend.
        b (int): The divisor.

    Returns:
        q (int): The quotient.

    Examples:
        >>> div_round(6, 3)
        2
        >>> div_round(6, -3)
        -2
        >>> div_round(-6, 3)
        -2
        >>> div_round(-6, -3)
        2
        >>> div_round(7, 3)
        2
        >>> div_round(7, -3)
        -2
        >>> div_round(-7, 3)
        -2
        >>> div_round(-7, -3)
        2
        >>> div_round(3, 7)
        0
        >>> div_round(3, -7)
        0
        >>> div_round(-3, 7)
        0
        >>> div_round(-3, -7)
        0
        >>> div_round(1, 1)
        1
        >>> div_round(1, -1)
        -1
        >>> div_round(-1, 1)
        -1
        >>> div_round(-1, -1)
        1

    See Also:
        - operator.div()
        - operator.mod()
        - divmod()
        - flyingcircus.div_ceil()
        - flyingcircus.mod_ceil()
        - flyingcircus.divmod_ceil()
        - flyingcircus.mod_round()
        - flyingcircus.divmod_round()
    """
    if (a >= 0) != (b >= 0) and a % b:
        return a // b + 1
    else:
        return a // b


# ======================================================================
def mod_round(a, b):
    """
    Compute integer round modulus.

    This behaves the same as C `%` operator on integers.

    This is such that:

    q = div_round(a, b)
    r = mod_round(a, b)
    q * b + r == a

    Args:
        a (int): The dividend.
        b (int): The divisor.

    Returns:
        r (int): The remainder.

    Examples:
        >>> mod_round(6, 3)
        0
        >>> mod_round(6, -3)
        0
        >>> mod_round(-6, 3)
        0
        >>> mod_round(-6, -3)
        0
        >>> mod_round(7, 3)
        1
        >>> mod_round(7, -3)
        1
        >>> mod_round(-7, 3)
        -1
        >>> mod_round(-7, -3)
        -1
        >>> mod_round(3, 7)
        3
        >>> mod_round(3, -7)
        3
        >>> mod_round(-3, 7)
        -3
        >>> mod_round(-3, -7)
        -3
        >>> mod_round(1, 1)
        0
        >>> mod_round(1, -1)
        0
        >>> mod_round(-1, 1)
        0
        >>> mod_round(-1, -1)
        0

    See Also:
        - operator.div()
        - operator.mod()
        - divmod()
        - flyingcircus.div_ceil()
        - flyingcircus.mod_ceil()
        - flyingcircus.divmod_ceil()
        - flyingcircus.div_round()
        - flyingcircus.divmod_round()
    """
    if a >= 0:
        if b >= 0:
            return a % b
        else:
            return a % -b
    else:
        if b >= 0:
            return -(-a % b)
        else:
            return a % b


# ======================================================================
def divmod_round(a, b):
    """
    Compute integer round division and modulus.

    This behaves the same as C `/` and `%` operators on integers.

    This is such that:

    q, r = divmod_round(a, b)
    q * b + r == a

    Args:
        a (int): The dividend.
        b (int): The divisor.

    Returns:
        r (int): The remainder.

    Examples:
        >>> n = 100
        >>> l = list(range(-n, n + 1))
        >>> ll = [(a, b) for a, b in itertools.product(l, repeat=2) if b]
        >>> result = True
        >>> for a, b in ll:
        ...     q, r = divmod_round(a, b)
        ...     result = result and (q == div_round(a, b))
        ...     result = result and (r == mod_round(a, b))
        ...     result = result and (q * b + r == a)
        >>> print(result)
        True

    See Also:
        - operator.div()
        - operator.mod()
        - divmod()
        - flyingcircus.div_ceil()
        - flyingcircus.mod_ceil()
        - flyingcircus.divmod_ceil()
        - flyingcircus.div_round()
        - flyingcircus.mod_round()
    """
    if a >= 0:
        if b >= 0:
            r = a % b
        else:
            r = a % -b
    else:
        if b >= 0:
            r = -(-a % b)
        else:
            r = a % b
    return (a - r) // b, r


# ======================================================================
def ilog2(num):
    """
    Compute the integer base-2 logarithm of a number.

    This is defined as the largest integer whose power-2 value is smaller than
    the number, i.e. floor(log2(n))

    Args:
        num (int): The input number.

    Returns:
        result (int): The integer base-2 logarithm of the abs(num).
            If the input is zero, returns -1.

    Examples:
        >>> ilog2(1024)
        10
        >>> ilog2(1023)
        9
        >>> ilog2(1025)
        10
        >>> ilog2(2 ** 400)
        400
        >>> ilog2(-1024)
        10
        >>> ilog2(-1023)
        9
        >>> ilog2(-1025)
        10
        >>> ilog2(0)
        -1
        >>> all(ilog2(2 ** i) == i for i in range(1000))
        True
        >>> all(ilog2(-(2 ** i)) == i for i in range(1000))
        True
    """
    # : alternate (slower) method
    # result = -1
    # if n < 0:
    #     n = -n
    # while n > 0:
    #     n >>= 1
    #     result += 1
    # return result
    return num.bit_length() - 1


# ======================================================================
def isqrt(num):
    """
    Compute the integer square root of a number.

    This is defined as the largest integer whose square is smaller than the
    number, i.e. floor(sqrt(num))

    Args:
        num (int): The input number.

    Returns:
        result (int): The integer square root of num.

    Examples:
        >>> isqrt(1024)
        32
        >>> isqrt(1023)
        31
        >>> isqrt(1025)
        32
        >>> isqrt(2 ** 400)
        1606938044258990275541962092341162602522202993782792835301376
        >>> isqrt(2 ** 400) == 2 ** 200
        True
        >>> all(isqrt(2 ** (2 * i)) == 2 ** i for i in range(1000))
        True
    """
    if num < 0:
        num = -num
    guess = (num >> num.bit_length() // 2) + 1
    result = (guess + num // guess) // 2
    while abs(result - guess) > 1:
        guess = result
        result = (guess + num // guess) // 2
    while result * result > num:
        result -= 1
    return result


# ======================================================================
def iroot(
        num,
        k=2):
    """
    Compute the integer k-th root of a number.

    This is defined as the largest integer whose n-th power is smaller than
    the number, i.e. floor(num ** (1 / base))

    Args:
        num (int): The input number.
        k (int): The order of the root.
            Must be k >= 1. When k == 1, the result is `num`.

    Returns:
        result (int): The integer k-th root of num.

    Examples:
        >>> iroot(64, 3)
        4
        >>> iroot(63, 3)
        3
        >>> iroot(65, 3)
        4
        >>> iroot(2 ** 300, 3)
        1267650600228229401496703205376
        >>> iroot(2 ** 300, 3) == 2 ** 100
        True
        >>> all(
        ...     iroot(2 ** (k * i), k) == 2 ** i
        ...     for i in range(100) for k in range(2, 10))
        True
    """
    if num < 0:
        num = -num
    if k > 1:
        result = 1 << num.bit_length() // 2
        update = result
        guess = result ** k
        limit = 1 << k  # same as: 2 ** k but faster
        while abs(guess - num) > limit and update > 1:
            update //= 2
            if guess < num:
                result += update
            else:
                result -= update
            guess = result ** k
        while result ** k < num:
            result += 1
        while result ** k > num:
            result -= 1
    else:
        return num
    return result


# ======================================================================
def perm(n, k):
    """
    Compute the number of ways to permuting n items into groups of size k.

    This is equivalent to the number of ways to choose k items from n items
    without repetition and with order.

    This is often indicated as `n_P_k` or `P(n, k)`.

    n! / (n - k)!

    Args:
        n (int): The number of items.
            Must be non-negative.
        k (int): The group size.
            Must be non-negative and smaller than or equal to `n`.

    Returns:
        result (int): The number of k-permutation of n items.

    Raises:
        ValueError: if the inputs are invalid.

    Examples:
        >>> perm(10, 5)
        30240
        >>> [perm(8, i) for i in range(8 + 1)]
        [1, 8, 56, 336, 1680, 6720, 20160, 40320, 40320]
        >>> [perm(9, i) for i in range(9 + 1)]
        [1, 9, 72, 504, 3024, 15120, 60480, 181440, 362880, 362880]
        >>> perm(0, 0)
        1
        >>> perm(1, 0)
        1
        >>> perm(0, 1)
        Traceback (most recent call last):
            ...
        ValueError: Values must be non-negative and n >= k in perm(n, k)
        >>> all(perm(n, n) == math.factorial(n) for n in range(20))
        True

    References:
        - https://en.wikipedia.org/wiki/Permutation
    """
    if not 0 <= k <= n:
        raise ValueError(
            'Values must be non-negative and n >= k in perm(n, k)')
    else:
        return prod(range(1, n + 1))



# ======================================================================
def factorial(n):
    """
    Compute the factorial of a number `n!`.

    n! = n * (n - 1) * ... * 1

    Args:
        n (int): The input number.
            Must be non-negative.

    Returns:
        result (int): The number of k-permutation of n items.

    Raises:
        ValueError: if the inputs are invalid.

    Examples:
        >>> factorial(5)
        120
        >>> [factorial(i) for i in range(12)]
        [1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800, 39916800]
        >>> factorial(0)
        1
        >>> factorial(1)
        1
        >>> factorial(-1)
        Traceback (most recent call last):
            ...
        ValueError: Value must be non-negative
        >>> all(factorial(n) == math.factorial(n) for n in range(20))
        True

    References:
        - https://en.wikipedia.org/wiki/Factorial
    """
    if n < 0:
        raise ValueError('Value must be non-negative')
    else:
        return prod(range(1, n + 1))


# ======================================================================
def comb(n, k):
    """
    Compute the number of ways to combining n items into groups of size k.

    This is essentially binomial coefficient of order n and position k.
    This is equivalent to the number of ways to choose k items from n items
    without repetition and without order.

    This is often indicated as `(n k)`, `n_C_k` or `C(n, k)`.

    If more than one binomial coefficient of the same order are needed, then
    `flyingcircus.get_binomial_coeffs()` may be more efficient.

    Args:
        n (int): The number of items.
            Must be non-negative.
        k (int): The group size.
            Must be non-negative and smaller than or equal to `n`.

    Returns:
        result (int): The number of k-combinations of n items.

    Raises:
        ValueError: if the inputs are invalid.

    Examples:
        >>> comb(10, 5)
        252
        >>> [comb(10, i) for i in range(10 + 1)]
        [1, 10, 45, 120, 210, 252, 210, 120, 45, 10, 1]
        >>> [comb(11, i) for i in range(11 + 1)]
        [1, 11, 55, 165, 330, 462, 462, 330, 165, 55, 11, 1]

        >>> N = 10
        >>> for n in range(N):
        ...     print([comb(n, k) for k in range(n + 1)])
        [1]
        [1, 1]
        [1, 2, 1]
        [1, 3, 3, 1]
        [1, 4, 6, 4, 1]
        [1, 5, 10, 10, 5, 1]
        [1, 6, 15, 20, 15, 6, 1]
        [1, 7, 21, 35, 35, 21, 7, 1]
        [1, 8, 28, 56, 70, 56, 28, 8, 1]
        [1, 9, 36, 84, 126, 126, 84, 36, 9, 1]
        >>> comb(0, 0)
        1
        >>> comb(1, 1)
        1

        >>> comb(1, 0)
        1
        >>> comb(0, 1)
        Traceback (most recent call last):
            ...
        ValueError: Values must be non-negative and n=0 >= k=1
        >>> comb(1, 2)
        Traceback (most recent call last):
            ...
        ValueError: Values must be non-negative and n=1 >= k=2
        >>> comb(2, 1)
        2
        >>> all(comb(n, k) * math.factorial(k) == perm(n, k)
        ...     for n in range(20) for k in range(n))
        True

    See Also:
        - flyingcircus.get_binomial_coeffs()
        - flyingcircus.binomial_triangle_range()

    References:
        - https://en.wikipedia.org/wiki/Combination
        - https://en.wikipedia.org/wiki/Binomial_coefficient
        - https://en.wikipedia.org/wiki/Binomial_triangle
    """
    if not 0 <= k <= n:
        raise ValueError(
            fmtm('Values must be non-negative and n={n} >= k={k}'))
    # elif k == 0:
    #     return 1
    # elif k == 1 or n == k:
    #     return n
    else:
        k = k if k < n - k else n - k
        # : purely factorial (slower) implementation
        # return \
        #     math.factorial(n) // math.factorial(n - k) // math.factorial(k)

        # : purely iterative (slower) implementation
        # result = 1
        # for i in range(1, k + 1):
        #     result = result * (n - i + 1) // i
        # return result

        # equivalent to: `perm(n, k) // math.factorial(k)`
        return prod(range(n - k + 1, n + 1)) // math.factorial(k)


# ======================================================================
def get_binomial_coeffs(
        num,
        full=True,
        cached=False):
    """
    Generate the numbers of a given row of the binomial triangle.

    This is also known as Pascal's triangle.

    These are the numbers in the `num`-th row (order) of the binomial triangle.
    If only a specific binomial coefficient is required, use
    `flyingcircus.comb()`.

    Args:
        num (int): The row index of the triangle.
            Indexing starts from 0.
        full (bool): Compute all numbers of the row.
            If True, all numbers are yielded.
            Otherwise, it only yields non-repeated numbers (exploiting the
            fact that they are palindromes).
        cached (bool): Cache the non-repeated numbers.
            If True and `full == True`, the non-repeated numbers are cached
            (thus allocating some additional temporary memory).
            Otherwise, if False or `full == False` this has no effect,
            i.e. the numbers are computed directly.

    Yields:
        value (int): The next binomial coefficient of a given order / row.

    Examples:
        >>> list(get_binomial_coeffs(12))
        [1, 12, 66, 220, 495, 792, 924, 792, 495, 220, 66, 12, 1]
        >>> list(get_binomial_coeffs(13))
        [1, 13, 78, 286, 715, 1287, 1716, 1716, 1287, 715, 286, 78, 13, 1]
        >>> list(get_binomial_coeffs(12, cached=True))
        [1, 12, 66, 220, 495, 792, 924, 792, 495, 220, 66, 12, 1]
        >>> list(get_binomial_coeffs(13, cached=True))
        [1, 13, 78, 286, 715, 1287, 1716, 1716, 1287, 715, 286, 78, 13, 1]
        >>> list(get_binomial_coeffs(12, full=False))
        [1, 12, 66, 220, 495, 792, 924]
        >>> list(get_binomial_coeffs(13, full=False))
        [1, 13, 78, 286, 715, 1287, 1716]
        >>> num = 10
        >>> for n in range(num):
        ...     print(list(get_binomial_coeffs(n)))
        [1]
        [1, 1]
        [1, 2, 1]
        [1, 3, 3, 1]
        [1, 4, 6, 4, 1]
        [1, 5, 10, 10, 5, 1]
        [1, 6, 15, 20, 15, 6, 1]
        [1, 7, 21, 35, 35, 21, 7, 1]
        [1, 8, 28, 56, 70, 56, 28, 8, 1]
        [1, 9, 36, 84, 126, 126, 84, 36, 9, 1]
        >>> all(list(get_binomial_coeffs(n))
        ...     == list(get_binomial_coeffs(n, cached=True))
        ...     for n in range(10))
        True
        >>> all(list(get_binomial_coeffs(n))
        ...     == [comb(n, k) for k in range(n + 1)]
        ...     for n in range(10))
        True

    See Also:
        - flyingcircus.comb()
        - flyingcircus.binomial_triangle_range()

    References:
        - https://en.wikipedia.org/wiki/Binomial_coefficient
        - https://en.wikipedia.org/wiki/Binomial_triangle
    """
    value = 1
    stop = (num + 1) if full and not cached else (num // 2 + 1)
    if full and cached:
        cache = list(get_binomial_coeffs(num, full=False, cached=False))
        for value in cache:
            yield value
        for value in cache[(-1 if num % 2 else -2)::-1]:
            yield value
    else:
        for i in range(stop):
            yield value
            value = value * (num - i) // (i + 1)


# ======================================================================
def get_fibonacci(
        max_count=-1,
        first=0,
        second=1):
    """
    Generate the first Fibonacci-like numbers.

    The next number is computed by adding the last two.

    This is useful for generating a sequence of Fibonacci numbers.
    To generate a specific Fibonacci number,
    use `flyingcircus.fibonacci()`.

    Args:
        max_count (int): The maximum number of values to yield.
            If `max_count == -1`, the generation proceeds indefinitely.
        first (int): The first number of the sequence.
        second (int): The second number of the sequence.

    Yields:
        value (int): The next Fibonacci number.

    Examples:
        >>> [x for x in get_fibonacci(16)]
        [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
        >>> [x for x in get_fibonacci(16, 0, 2)]
        [0, 2, 2, 4, 6, 10, 16, 26, 42, 68, 110, 178, 288, 466, 754, 1220]
        >>> [x for x in get_fibonacci(16, 3, 1)]
        [3, 1, 4, 5, 9, 14, 23, 37, 60, 97, 157, 254, 411, 665, 1076, 1741]

    See Also:
        - flyingcircus.get_gen_fibonacci()
        - flyingcircus.fibonacci()
        - https://en.wikipedia.org/wiki/Fibonacci_number
    """
    i = 0
    while i != max_count:
        yield first
        first, second = second, first + second
        i += 1


# ======================================================================
def get_gen_fibonacci(
        max_count=-1,
        values=(0, 1),
        weights=1):
    """
    Generate the first generalized Fibonacci-like numbers.

    The next number is computed as a linear combination of the last values
    multiplied by their respective weights.

    These can be used to compute n-step Fibonacci, Lucas numbers,
    Pell numbers, Perrin numbers, etc.

    Args:
        max_count (int): The maximum number of values to yield.
            If `max_count == -1`, the generation proceeds indefinitely.
        values (int|Sequence[int]): The initial numbers of the sequence.
            If int, the value is repeated for the number of `weights`, and
            `weights` must be a sequence.
        weights (int|Sequence[int]): The weights for the linear combination.
            If int, the value is repeated for the number of `values`, and
            `values` must be a sequence.

    Yields:
        value (int): The next number of the sequence.

    Examples:
        >>> [x for x in get_gen_fibonacci(16)]
        [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
        >>> [x for x in get_gen_fibonacci(16, (1, 1, 1))]
        [1, 1, 1, 3, 5, 9, 17, 31, 57, 105, 193, 355, 653, 1201, 2209, 4063]
        >>> [x for x in get_gen_fibonacci(16, (1, 0, 1))]
        [1, 0, 1, 2, 3, 6, 11, 20, 37, 68, 125, 230, 423, 778, 1431, 2632]
        >>> [x for x in get_gen_fibonacci(13, (0, 1), 2)]
        [0, 1, 2, 6, 16, 44, 120, 328, 896, 2448, 6688, 18272, 49920]
        >>> [x for x in get_gen_fibonacci(16, (1, 0, 1), (1, 2, 1))]
        [1, 0, 1, 2, 4, 9, 19, 41, 88, 189, 406, 872, 1873, 4023, 8641, 18560]

    See Also:
        - flyingcircus.get_fibonacci()
        - flyingcircus.fibonacci()
        - https://en.wikipedia.org/wiki/Fibonacci_number
    """
    num = combine_iter_len((values, weights))
    values = auto_repeat(values, num, check=True)
    weights = auto_repeat(weights, num, check=True)
    i = 0
    while i != max_count:
        yield values[0]
        values = values[1:] + (sum(w * x for w, x in zip(weights, values)),)
        i += 1


# ======================================================================
def fibonacci(
        num,
        first=0,
        second=1):
    """
    Generate the n-th Fibonacci number.

    This is useful for generating a specific Fibonacci number.
    For generating a sequence of Fibonacci numbers, use
    `flyingcircus.get_fibonacci()`.

    Args:
        num (int): The ordinal to generate.
            Starts counting from 0.
            If `num < 0`, the first number is returned.
        first (int): The first number of the sequence.
        second (int): The second number of the sequence.

    Returns:
        value (int): The Fibonacci number.

    Examples:
        >>> fibonacci(10)
        55
        >>> fibonacci(100)
        354224848179261915075
        >>> fibonacci(200)
        280571172992510140037611932413038677189525
        >>> fibonacci(300)
        222232244629420445529739893461909967206666939096499764990979600
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(15, 0, 2)
        1220
        >>> fibonacci(15, 3, 1)
        1741

    See Also:
        - flyingcircus.get_fibonacci()
        - flyingcircus.get_gen_fibonacci()
        - https://en.wikipedia.org/wiki/Fibonacci_number
    """
    for _ in range(num):
        first, second = second, first + second
    return first


# ======================================================================
def binomial_triangle(
        first,
        second=None,
        step=None,
        container=tuple):
    """
    Generate the binomial triangle rows in a given range.
    
    This is also known as Pascal's triangle.

    See `flyingcircus.get_binomial_coeffs()` for generating any given
    row of the triangle.

    Args:
        first (int): The first value of the range.
            Must be non-negative.
            If `second == None` this is the `stop` value, and is not included.
            Otherwise, this is the `start` value and is included.
            If `first < second` the sequence is yielded backwards.
        second (int|None): The second value of the range.
            If None, the start value is 0.
            Otherwise, this is the `stop` value and is not included.
            Must be non-negative.
            If `first < second` the sequence is yielded backwards.
        step (int): The step of the rows range.
            If the sequence is yielded backward, the step should be negative,
            otherwise an empty sequence is yielded.
            If None, this is computed automatically based on `first` and
            `second`, such that a non-empty sequence is avoided, if possible.
        container (callable): The row container.
            This should be a Sequence constructor.

    Yields:
        row (Sequence[int]): The rows of the binomial triangle.

    Examples:
        >>> tuple(binomial_triangle(5))
        ((1,), (1, 1), (1, 2, 1), (1, 3, 3, 1), (1, 4, 6, 4, 1))
        >>> tuple(binomial_triangle(5, 7))
        ((1, 5, 10, 10, 5, 1), (1, 6, 15, 20, 15, 6, 1))
        >>> tuple(binomial_triangle(7, 9))
        ((1, 7, 21, 35, 35, 21, 7, 1), (1, 8, 28, 56, 70, 56, 28, 8, 1))
        >>> tuple(binomial_triangle(5, 2))
        ((1, 5, 10, 10, 5, 1), (1, 4, 6, 4, 1), (1, 3, 3, 1))
        >>> tuple(binomial_triangle(0, 7, 2))
        ((1,), (1, 2, 1), (1, 4, 6, 4, 1), (1, 6, 15, 20, 15, 6, 1))
        >>> tuple(binomial_triangle(0, 6, 2))
        ((1,), (1, 2, 1), (1, 4, 6, 4, 1))
        >>> tuple(binomial_triangle(7, 1, -2))
        ((1, 7, 21, 35, 35, 21, 7, 1), (1, 5, 10, 10, 5, 1), (1, 3, 3, 1))
        >>> tuple(binomial_triangle(7, 1, 2))  # empty range!
        ()
        >>> list(binomial_triangle(3))
        [(1,), (1, 1), (1, 2, 1)]
        >>> list(binomial_triangle(5, container=list))
        [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]
        >>> for row in binomial_triangle(10):
        ...    print(row)
        (1,)
        (1, 1)
        (1, 2, 1)
        (1, 3, 3, 1)
        (1, 4, 6, 4, 1)
        (1, 5, 10, 10, 5, 1)
        (1, 6, 15, 20, 15, 6, 1)
        (1, 7, 21, 35, 35, 21, 7, 1)
        (1, 8, 28, 56, 70, 56, 28, 8, 1)
        (1, 9, 36, 84, 126, 126, 84, 36, 9, 1)

    See Also:
        - flyingcircus.comb()
        - flyingcircus.get_binomial_coeffs()

    References
        - https://en.wikipedia.org/wiki/Binomial_coefficient
        - https://en.wikipedia.org/wiki/Binomial_triangle
    """
    if second is None:
        start, stop = 0, first
    else:
        start, stop = first, second
    if not step:
        step = 1 if start < stop else -1
    for i in range(start, stop, step):
        yield container(get_binomial_coeffs(i))


# ======================================================================
def _is_prime(num):
    """
    Determine if a number is prime.

    A prime number is only divisible by 1 and itself.
    0 and 1 are considered special cases; in this implementations they are
    considered primes.

    It is implemented by testing for possible factors using wheel increment.

    Args:
        num (int): The number to check for primality.
            Must be greater than 1.

    Returns:
        result (bool): The result of the primality.

    Examples:
        >>> _is_prime(100)
        False
        >>> _is_prime(101)
        True
        >>> _is_prime(0)
        True
        >>> all(is_prime(n) for n in (-2, -1, 0, 1, 2))
        True

    See Also:
        - flyingcircus.is_prime()
        - flyingcircus.primes_range()

    References:
        - https://en.wikipedia.org/wiki/Prime_number
        - https://en.wikipedia.org/wiki/Trial_division
        - https://en.wikipedia.org/wiki/Wheel_factorization
    """
    # hard-coded (2, 3) wheel
    if (not (num % 2) and num > 2) or (not (num % 3) and num > 3):
        return False
    i = 5
    while i * i <= num:
        if not (num % i and num % (i + 2)):
            return False
        else:
            i += 6
    return True


# ======================================================================
SMALL_PRIMES = (2,) + tuple(
    n for n in range(3, 2 ** 10 + 1, 2) if _is_prime(n))


# ======================================================================
def is_prime(
        num,
        small_primes=SMALL_PRIMES,
        hashed_small_primes=frozenset(SMALL_PRIMES)):
    """
    Determine if a number is prime.

    A prime number is only divisible by 1 and itself.
    0 and 1 are considered special cases; in this implementations they are
    considered primes.

    The implementation is using a certain number of precomputed primes,
    later switching to trial division with hard-coded (2, 3) wheel
    (up to 2 ** 20).
    After, it uses `flyingcircus.is_pseudo_prime()` with bases consisting
    of consecutive primes, and this is guaranteed to be deterministic up to
    3317044064679887385961981 > 2 ** 81.

    Args:
        num (int): The number to test for primality.
            Only works for numbers larger than 1.
        small_primes (Sequence|None): The first prime numbers (sorted).
            Must contain prime starting from 2 (included).
            Must be in increasing order.
            If None, uses a hard-coded sequence and skip fast hashed checks.
        hashed_small_primes (Container|None): The first prime numbers.
            Must contain prime starting from 2 (included).
            Should use a high performance container like `set` or `frozenset`.
            Must contain the same numbers included in `small_primes`.
            If None, skip fast hashed checks.

    Returns:
        result (bool): The result of the primality test.

    Examples:
        >>> is_prime(100)
        False
        >>> is_prime(101)
        True
        >>> is_prime(-100)
        False
        >>> is_prime(-101)
        True
        >>> is_prime(2 ** 17)
        False
        >>> is_prime(17 * 19)
        False
        >>> is_prime(2 ** 17 - 1)
        True
        >>> is_prime(2 ** 31 - 1)
        True
        >>> is_prime(2 ** 17 - 1, (2,))
        True
        >>> is_prime(2 ** 17 - 1, (2, 3))
        True
        >>> is_prime(2 ** 17 - 1, (2, 3, 5))
        True
        >>> all(is_prime(n) for n in (-2, -1, 0, 1, 2))
        True

    See Also:
        - flyingcircus.is_prime()
        - flyingcircus.is_pseudo_prime()
        - flyingcircus.primes_range()

    References:
        - https://en.wikipedia.org/wiki/Prime_number
        - https://en.wikipedia.org/wiki/Trial_division
        - https://en.wikipedia.org/wiki/Wheel_factorization
        - https://en.wikipedia.org/wiki/Miller–Rabin_primality_test
    """
    if num < 0:
        num = -num
    if not small_primes:
        small_primes = 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 43
        hashed_small_primes = None
    if num < small_primes[0]:
        return True
    elif num <= small_primes[-1] and hashed_small_primes:
        return num in hashed_small_primes
    elif num <= 1048576:  # 2 ** 20
        for prime in small_primes:
            if not num % prime:
                return False
            elif prime * prime > num:
                return True
        i = 5 + (small_primes[-1] - 1) // 6 * 6
        while i * i <= num:
            if not num % i or not num % (i + 2):
                return False
            else:
                i += 6
        return True
    else:
        limits = (
            25326001, 3215031751, 2152302898747, 3474749660383,
            341550071728321, None, 3825123056546413051, None, None,
            318665857834031151167461, 3317044064679887385961981)
        num_bases = num.bit_length() // 2
        for i, limit in enumerate(limits):
            if limit and num < limit:
                num_bases = i + 3
                break
        bases = tuple(get_primes(num_bases))
        return is_pseudo_prime(num, bases)


# ======================================================================
def is_pseudo_prime(
        num,
        bases=None):
    """
    Determine if a number is pseudo-prime (using the Miller-Rabin test).

    Args:
        num (int): The number to test for primality.
        bases (Iterable[int]|None): The bases for the test.
            If None, uses the theoretical values for which the test is proven
            to be deterministic, if the extended Riemann hypothesis is True:
            bases = range(2, min(n-2, log2(n) ** 2)

    Returns:
        result (bool): The result of the pseudo-primality test.

    Examples:
        >>> is_pseudo_prime(100, (2,))
        False
        >>> is_pseudo_prime(101, (2,))
        True
        >>> is_pseudo_prime(-100, (2,))
        False
        >>> is_pseudo_prime(-101, (2,))
        True
        >>> is_pseudo_prime(2 ** 17, (2,))
        False
        >>> is_pseudo_prime(17 * 19, (2,))
        False
        >>> is_pseudo_prime(2 ** 17 - 1, (2,))
        True
        >>> is_pseudo_prime(2 ** 31 - 1, (2,))
        True
        >>> is_pseudo_prime(2 ** 17 - 1, (2,))
        True
        >>> is_pseudo_prime(2 ** 17 - 1, (2, 3))
        True
        >>> is_pseudo_prime(2 ** 17 - 1, (2, 3, 5))
        True
        >>> all(is_pseudo_prime(n) for n in (-2, -1, 0, 1, 2))
        True
        >>> is_pseudo_prime(2047, (2,))
        True
        >>> is_prime(2047)
        False

    See Also:
        - flyingcircus.is_prime()
        - flyingcircus.is_pseudo_prime()
        - flyingcircus.primes_range()

    References:
        - https://en.wikipedia.org/wiki/Prime_number
        - https://en.wikipedia.org/wiki/Trial_division
        - https://en.wikipedia.org/wiki/Wheel_factorization
        - https://en.wikipedia.org/wiki/Miller–Rabin_primality_test
    """
    if num < 0:
        num = -num
    if num < 3:
        return True
    elif not num % 2:
        return False
    elif not bases:
        bases = range(2, min(num - 2, num.bit_length() ** 2))
    # num = 2 ** r * d + 1
    r = ((num - 1) & -(num - 1)).bit_length() - 1
    d = num >> r
    for base in bases:
        if base >= num:
            base %= num
        if base >= 2:
            # x = base ** d mod num
            x = pow(base, d, num)
            if x != 1 and x != num - 1:
                skip = False
                for _ in range(1, r):
                    # x = x ** 2 mod num
                    x = pow(x, 2, num)
                    if x == num - 1:
                        skip = True
                        break
                    if x == 1:
                        return False
                if not skip:
                    return False
    return True


# ======================================================================
def get_primes(
        max_count=-1,
        start=2,
        small_primes=SMALL_PRIMES,
        wheel=2):
    """
    Generate the next prime numbers.

    New possible primes are obtained using a hard-coded (2, 3) wheel.

    Args:
        max_count (int): The maximum number of values to yield.
            If `max_count == -1`, the generation proceeds indefinitely.
        start (int): The initial value.
            This must be positive.
        small_primes (Sequence|None): The first prime numbers (sorted).
            Must contains prime starting from 2 (included).
            Must be in increasing order.
            If None, uses a hard-coded sequence.
        wheel (int): The number of primes to use as wheel.
            Must be > 1.
            The wheel is generated using `flyingcircus.get_primes()`.

    Yields:
        num (int): The next prime number.

    Examples:
        >>> n = 15
        >>> primes = get_primes()
        >>> [next(primes) for i in range(n)]
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        >>> list(get_primes(n))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        >>> list(get_primes(10, 101))
        [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]
        >>> list(get_primes(10, 1000))
        [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061]

        >>> list(get_primes(n, 2, None))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        >>> list(get_primes(10, 101, None))
        [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]
        >>> list(get_primes(10, 1000, None))
        [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061]

        >>> list(get_primes(n, 2, None, 3))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        >>> list(get_primes(10, 101, None, 3))
        [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]
        >>> list(get_primes(10, 1000, None, 3))
        [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061]

    See Also:
        - flyingcircus.is_prime()
        - flyingcircus.primes_range()

    References:
        - https://en.wikipedia.org/wiki/Prime_number
        - https://en.wikipedia.org/wiki/Wheel_factorization
    """
    i = 0
    if start < 0:
        raise ValueError(
            fmtm('`get_primes()` must start at a positive value, not `{num}`'))
    if not small_primes:
        small_primes = 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    if start < small_primes[-1]:
        for prime in small_primes:
            if prime >= start:
                yield prime
                i += 1
                if i == max_count:
                    return
                else:
                    start = prime
    if wheel <= 2:
        num = 5 + (start - 1) // 6 * 6
        while True:
            if is_prime(num):
                yield num
                i += 1
                if i == max_count:
                    return
            if is_prime(num + 2):
                yield num + 2
                i += 1
                if i == max_count:
                    return
            num += 6
    else:
        if wheel > len(small_primes):
            wheel = tuple(get_primes(wheel, 2))
        else:
            wheel = small_primes[:wheel]
        prod_wheel = prod(wheel)
        coprimes = tuple(
            n for n in range(2, prod_wheel + 2)
            if all(math.gcd(n, k) == 1 for k in wheel))
        deltas = tuple(diff(coprimes + (coprimes[0] + prod_wheel,)))
        len_deltas = len(deltas)
        num = coprimes[0] + (start - 1) // prod_wheel * prod_wheel
        j = 0
        while i != max_count:
            if num >= start and is_prime(num):
                yield num
                i += 1
            num += deltas[j]
            j += 1
            j %= len_deltas


# ======================================================================
def get_primes_r(
        max_count=-1,
        start=2,
        small_primes=SMALL_PRIMES,
        wheel=2):
    """
    Generate the previous prime numbers.

    New possible primes are obtained using a hard-coded (2, 3) wheel.

    Args:
        max_count (int): The maximum number of values to yield.
            If equal to -1 or larger than the number of primes smaller than or
            equal to `start`, the generation proceeds until the smallest
            proper prime number (2) is yielded.
        start (int): The initial value.
            This must be positive.
        small_primes (Sequence|None): The first prime numbers (sorted).
            Must contains prime starting from 2 (included).
            Must be in increasing order.
            If None, uses a hard-coded sequence.
        wheel (int): The number of primes to use as wheel.
            Must be > 1.
            The wheel is generated using `flyingcircus.get_primes()`.

    Yields:
        num (int): The next prime number.

    Examples:
        >>> n = 15
        >>> primes = get_primes_r(-1, 47)
        >>> [next(primes) for i in range(n)]
        [47, 43, 41, 37, 31, 29, 23, 19, 17, 13, 11, 7, 5, 3, 2]
        >>> list(get_primes_r(n, 47))
        [47, 43, 41, 37, 31, 29, 23, 19, 17, 13, 11, 7, 5, 3, 2]
        >>> list(get_primes_r(10, 150))
        [149, 139, 137, 131, 127, 113, 109, 107, 103, 101]
        >>> list(get_primes_r(10, 1062))
        [1061, 1051, 1049, 1039, 1033, 1031, 1021, 1019, 1013, 1009]
        >>> list(get_primes_r(10, 10))
        [7, 5, 3, 2]

        >>> list(get_primes_r(n, 47, None))
        [47, 43, 41, 37, 31, 29, 23, 19, 17, 13, 11, 7, 5, 3, 2]
        >>> list(get_primes_r(10, 150, None))
        [149, 139, 137, 131, 127, 113, 109, 107, 103, 101]
        >>> list(get_primes_r(10, 1062, None))
        [1061, 1051, 1049, 1039, 1033, 1031, 1021, 1019, 1013, 1009]
        >>> list(get_primes_r(10, 10, None))
        [7, 5, 3, 2]

        >>> list(get_primes_r(n, 47, None, 3))
        [47, 43, 41, 37, 31, 29, 23, 19, 17, 13, 11, 7, 5, 3, 2]
        >>> list(get_primes_r(10, 150, None, 3))
        [149, 139, 137, 131, 127, 113, 109, 107, 103, 101]
        >>> list(get_primes_r(10, 1062, None, 3))
        [1061, 1051, 1049, 1039, 1033, 1031, 1021, 1019, 1013, 1009]
        >>> list(get_primes_r(10, 10, None, 3))
        [7, 5, 3, 2]

    See Also:
        - flyingcircus.is_prime()
        - flyingcircus.primes_range()

    References:
        - https://en.wikipedia.org/wiki/Prime_number
        - https://en.wikipedia.org/wiki/Wheel_factorization
    """
    i = 0
    if start < 0:
        raise ValueError(
            fmtm('`get_primes()` must start at a positive value, not `{num}`'))
    if not small_primes:
        small_primes = 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    if wheel <= 2:
        num = 5 + (start - 1) // 6 * 6
        while num > small_primes[-1]:
            if is_prime(num):
                yield num
                i += 1
                if i == max_count:
                    return
            if is_prime(num - 4):
                yield num - 4
                i += 1
                if i == max_count:
                    return
            num -= 6
    else:
        if wheel > len(small_primes):
            wheel = tuple(get_primes(wheel, 2))
        else:
            wheel = small_primes[:wheel]
        prod_wheel = prod(wheel)
        coprimes = tuple(
            n for n in range(prod_wheel + 1, 1, -1)
            if all(math.gcd(n, k) == 1 for k in wheel))
        deltas = tuple(diff(coprimes + (coprimes[0] - prod_wheel,)))
        len_deltas = len(deltas)
        num = coprimes[0] + (start - 1) // prod_wheel * prod_wheel
        j = 0
        while num > start:
            num += deltas[j]
            j += 1
            j %= len_deltas
        while num > small_primes[-1] and i != max_count:
            if is_prime(num):
                yield num
                i += 1
            num += deltas[j]
            j += 1
            j %= len_deltas
    if i != max_count:
        for prime in reversed(small_primes):
            if prime <= start and prime <= num:
                yield prime
                i += 1
                if i == max_count:
                    return
                else:
                    start = prime


# ======================================================================
def primes_range(
        first,
        second=None):
    """
    Compute the prime numbers in the range.

    Args:
        first (int): The first value of the range.
            If `second == None` this is the `stop` value, and is not included.
            Otherwise, this is the start value and can be included
            (if it is a prime number).
            If `first < second` the sequence is yielded backwards.
        second (int|None): The second value of the range.
            If None, the start value is 2.
            If `first < second` the sequence is yielded backwards.

    Yields:
        num (int): The next prime number.

    Examples:
        >>> list(primes_range(50))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        >>> list(primes_range(51, 100))
        [53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        >>> list(primes_range(101, 150))
        [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]
        >>> list(primes_range(151, 200))
        [151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]
        >>> list(primes_range(1000, 1050))
        [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049]
        >>> list(primes_range(1050, 1000))
        [1049, 1039, 1033, 1031, 1021, 1019, 1013, 1009]
        >>> list(primes_range(1, 50))
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        >>> list(primes_range(50, 1))
        [47, 43, 41, 37, 31, 29, 23, 19, 17, 13, 11, 7, 5, 3, 2]

    See Also:
        - flyingcircus.is_prime()
        - flyingcircus.get_primes()
        - https://en.wikipedia.org/wiki/Prime_number
        - https://en.wikipedia.org/wiki/Wheel_factorization
    """
    if second is None:
        start, stop = 2, first
    else:
        start, stop = first, second
    if start <= stop:
        for prime in get_primes(-1, start):
            if prime < stop:
                yield prime
            else:
                return
    else:  # start >= stop
        for prime in get_primes_r(-1, start):
            if prime > stop:
                yield prime
            else:
                return


# ======================================================================
def get_factors(
        num,
        small_primes=SMALL_PRIMES,
        wheel=3):
    """
    Find all factors of a number.

    It is implemented by testing for possible factors using wheel increment.

    This is not suitable for large numbers.

    Args:
        num (int|float): The number to factorize.
            If float, its nearest integer is factorized.
        small_primes (Sequence|None): The first prime numbers (sorted).
            Must contains prime starting from 2 (included).
            Must be in increasing order.
            If None, uses a hard-coded sequence.
        wheel (int): The number of primes to use as wheel.
            Must be > 1.
            The wheel is generated using `flyingcircus.get_primes()`.

    Yields:
        factor (int): The next factor of the number.
            Factors are yielded in increasing order.

    Examples:
        >>> list(get_factors(100))
        [2, 2, 5, 5]
        >>> list(get_factors(1234567890))
        [2, 3, 3, 5, 3607, 3803]
        >>> list(get_factors(-65536))
        [-1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        >>> list(get_factors(0))
        [0]
        >>> list(get_factors(1))
        [1]
        >>> list(get_factors(-1))
        [-1]
        >>> list(get_factors(987654321.0))
        [3, 3, 17, 17, 379721]
        >>> all(n == prod(get_factors(n)) for n in range(1000))
        True

    See Also:
        - flyingcircus.is_prime()
        - flyingcircus.primes_range()
        - flyingcircus.get_primes()

    References:
        - https://en.wikipedia.org/wiki/Trial_division
        - https://en.wikipedia.org/wiki/Wheel_factorization
    """
    # : deal with special numbers: 0, 1, and negative
    if not isinstance(num, int):
        num = int(round(num))
    if num == 0:
        yield 0
        return
    if num < 0:
        yield -1
        num = -num
    elif num == 1:
        yield num
    # : use small primes
    if not small_primes:
        small_primes = 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47
    for prime in small_primes:
        while not (num % prime):
            yield prime
            num //= prime
    # : wheel factorization
    if wheel > len(small_primes):
        wheel = tuple(get_primes(wheel, 2))
    else:
        wheel = small_primes[:wheel]
    for prime in wheel:
        while not (num % prime):
            yield prime
            num //= prime
    prod_wheel = prod(wheel)
    coprimes = tuple(
        n for n in range(2, prod_wheel + 2)
        if all(math.gcd(n, k) == 1 for k in wheel))
    deltas = tuple(diff(coprimes + (coprimes[0] + prod_wheel,)))
    len_deltas = len(deltas)
    j = 0
    # prime is the largest prime in wheel
    k = coprimes[0] + (prime - 1) // prod_wheel * prod_wheel
    while k * k <= num:
        while not (num % k):
            yield k
            num //= k
        k += deltas[j]
        j += 1
        j %= len_deltas
    if num > 1:
        yield num


# ======================================================================
def get_factors_as_dict(num):
    """
    Find all factors of a number and collect them in an ordered dict.

    Args:
        num (int): The number to factorize.

    Returns:
        factors (collections.Counter): The factors of the number.

    Examples:
        >>> sorted(get_factors_as_dict(100).items())
        [(2, 2), (5, 2)]
        >>> sorted(get_factors_as_dict(1234567890).items())
        [(2, 1), (3, 2), (5, 1), (3607, 1), (3803, 1)]
        >>> sorted(get_factors_as_dict(65536).items())
        [(2, 16)]
    """
    return collections.Counter(get_factors(num))


# ======================================================================
def get_factors_as_str(
        num,
        mode='ascii'):
    """
    Find all factors of a number and output a human-readable text.

    Args:
        num (int): The number to factorize.
        mode (str|Iterable[str|None]): The formatting mode.
            If str, available options:
             - `ascii`: uses `^` for power, and ` * ` for multiplication.
             - `py`: uses ` ** ` for power, and ` * ` for multiplication.
             - `sup`: uses superscript for power, and ` ` for multiplication.
             - `utf8`: uses superscript for power, and `×` for multiplication.
            If Iterable of str or None, must have length 2:
             - the first is used as exponent symbol if str, otherwise if None,
               uses superscript notation.
             - the second is used as multiplication symbol (requires str).

    Returns:
        text (str): The factors of the number.

    Examples:
        >>> get_factors_as_str(100, 'ascii')
        '2^2 * 5^2'
        >>> get_factors_as_str(1234567890, 'ascii')
        '2 * 3^2 * 5 * 3607 * 3803'
        >>> get_factors_as_str(65536, 'ascii')
        '2^16'

        >>> get_factors_as_str(100, 'py')
        '2 ** 2 * 5 ** 2'
        >>> get_factors_as_str(1234567890, 'py')
        '2 * 3 ** 2 * 5 * 3607 * 3803'
        >>> get_factors_as_str(65536, 'py')
        '2 ** 16'

        >>> get_factors_as_str(100, 'utf8')
        '2²×5²'
        >>> get_factors_as_str(1234567890, 'utf8')
        '2×3²×5×3607×3803'
        >>> get_factors_as_str(65536, 'utf8')
        '2¹⁶'

        >>> get_factors_as_str(100, 'sup')
        '2² 5²'
        >>> get_factors_as_str(1234567890, 'sup')
        '2 3² 5 3607 3803'
        >>> get_factors_as_str(65536, 'sup')
        '2¹⁶'

        >>> get_factors_as_str(100, tuple('^*'))
        '2^2*5^2'
        >>> get_factors_as_str(1234567890, tuple('^*'))
        '2*3^2*5*3607*3803'
        >>> get_factors_as_str(65536, tuple('^*'))
        '2^16'
    """
    if isinstance(mode, str):
        mode = mode.lower()
        if mode == 'ascii':
            exp_sep, fact_sep = '^', ' * '
        elif mode == 'py':
            exp_sep, fact_sep = ' ** ', ' * '
        elif mode == 'sup':
            exp_sep, fact_sep = None, ' '
        elif mode == 'utf8':
            exp_sep, fact_sep = None, '×'
    else:
        exp_sep, fact_sep = mode
    text = ''
    last_factor = 1
    exp = 0
    for factor in get_factors(num):
        if factor == last_factor:
            exp += 1
        else:
            if exp > 1:
                if exp_sep:
                    text += exp_sep + str(exp)
                else:
                    text += str(exp).translate(SUPERSCRIPT_MAP)
            if last_factor > 1:
                text += fact_sep
            text += str(factor)
            last_factor = factor
            exp = 1
    if exp > 1:
        if exp_sep:
            text += exp_sep + str(exp)
        else:
            text += str(exp).translate(SUPERSCRIPT_MAP)
    return text


# ======================================================================
def get_divisors(num):
    """
    Find all divisors of a number.

    It is implemented by computing the possible unique combinations of the
    factors.

    This is not suitable for large numbers.

    Args:
        num (int|float): The number to factorize.
            If float, its nearest integer is factorized.

    Yields:
        factor (int): The next factor of the number.
            Factors are yielded in increasing order.

    Examples:
        >>> list(get_divisors(100))
        [1, 2, 4, 5, 10, 20, 25, 50, 100]
        >>> list(get_divisors(-100))
        [-1, 1, 2, 4, 5, 10, 20, 25, 50, 100]
        >>> list(get_divisors(2 ** 8))
        [1, 2, 4, 8, 16, 32, 64, 128, 256]
        >>> list(get_divisors(12345))
        [1, 3, 5, 15, 823, 2469, 4115, 12345]
        >>> list(get_divisors(0))
        [0]
        >>> list(get_divisors(1))
        [1]
        >>> list(get_divisors(2))
        [1, 2]
        >>> list(get_divisors(101))
        [1, 101]

    See Also:
        - flyingcircus.is_prime()
        - flyingcircus.primes_range()
        - flyingcircus.get_primes()
        - flyingcircus.get_factors()

    References:
        - https://en.wikipedia.org/wiki/Trial_division
        - https://en.wikipedia.org/wiki/Wheel_factorization
    """
    if num in (0, 1):
        yield num
        return
    if num < 0:
        yield -1
        num = -num
    unique_factors, unique_powers = zip(*sorted(
        collections.Counter(get_factors(num)).items(), reverse=True))
    for powers in itertools.product(*(range(n + 1) for n in unique_powers)):
        yield prod(
            factor ** power for factor, power in zip(unique_factors, powers))


# =====================================================================
def get_k_factors_all(
        num,
        k=2,
        sort=None,
        reverse=False):
    """
    Find all possible factorizations with k factors.

    Ones are not present, unless because there are not enough factors.

    Args:
        num (int): The number of elements to arrange.
        k (int): The number of factors.
        sort (callable): The sorting function.
            This is passed to the `key` arguments of `sorted()`.
        reverse (bool): The sorting direction.
            This is passed to the `reverse` arguments of `sorted()`.
            If False, sorting is ascending.
            Otherwise, sorting is descending.

    Returns:
        factorizations (tuple[tuple[int]]): The possible factorizations.
            Each factorization has exactly `k` items.
            Eventually, `1`s are used to ensure the number of items.

    Examples:
        >>> nums = (32, 41, 46, 60)
        >>> for i in nums:
        ...     get_k_factors_all(i, 2)
        ((2, 16), (4, 8), (8, 4), (16, 2))
        ((1, 41), (41, 1))
        ((2, 23), (23, 2))
        ((2, 30), (3, 20), (4, 15), (5, 12), (6, 10), (10, 6), (12, 5),\
 (15, 4), (20, 3), (30, 2))
        >>> for i in nums:
        ...     get_k_factors_all(i, 3)
        ((2, 2, 8), (2, 4, 4), (2, 8, 2), (4, 2, 4), (4, 4, 2), (8, 2, 2))
        ((1, 1, 41), (1, 41, 1), (41, 1, 1))
        ((1, 2, 23), (1, 23, 2), (2, 1, 23), (2, 23, 1), (23, 1, 2),\
 (23, 2, 1))
        ((2, 2, 15), (2, 3, 10), (2, 5, 6), (2, 6, 5), (2, 10, 3), (2, 15, 2),\
 (3, 2, 10), (3, 4, 5), (3, 5, 4), (3, 10, 2), (4, 3, 5), (4, 5, 3),\
 (5, 2, 6), (5, 3, 4), (5, 4, 3), (5, 6, 2), (6, 2, 5), (6, 5, 2),\
 (10, 2, 3), (10, 3, 2), (15, 2, 2))
    """
    factors = tuple(get_factors(num))
    factors += (1,) * (k - len(factors))
    factorizations = [
        item
        for subitems in unique_partitions(factors, k)
        for item in subitems]
    factorizations = list(set(factorizations))
    for i in range(len(factorizations)):
        factorizations[i] = tuple(
            functools.reduce(lambda x, y: x * y, j) for j in factorizations[i])
    return tuple(sorted(set(factorizations), key=sort, reverse=reverse))


# =====================================================================
def get_k_factors(
        num,
        k=2,
        mode='=',
        balanced=True):
    """
    Generate a factorization of a number with k factors.

    Each factor contains (approximately) the same number of prime factors.

    Args:
        num (int): The number of elements to arrange.
        k (int): The number of factors.
        mode (str): The generation mode.
            This determines the factors order before splitting.
            The splitting itself is obtained with `chunks()`.
            Accepted values are:
             - 'increasing', 'ascending', '+': factors are sorted increasingly
               before splitting;
             - 'decreasing', 'descending', '-': factors are sorted decreasingly
               before splitting;
             - 'random': factors are shuffled before splitting;
             - 'seedX' where 'X' is an int, str or bytes: same as random, but
               'X' is used to initialize the random seed;
             - 'altX' where 'X' is an int: starting from 'X', factors are
               alternated before splitting;
             - 'alt1': factors are alternated before splitting;
             - 'optimal', 'similar', '!', '=': factors have the similar sizes.
        balanced (bool): Balance the number of primes in each factor.
            See `flyingcircus.chunks()` for more info.

    Returns:
        tuple (int): A listing of `k` factors of `num`.

    Examples:
        >>> [get_k_factors(402653184, k) for k in range(3, 6)]
        [(1024, 768, 512), (192, 128, 128, 128), (64, 64, 64, 48, 32)]
        >>> [get_k_factors(402653184, k) for k in (2, 12)]
        [(24576, 16384), (8, 8, 8, 8, 6, 4, 4, 4, 4, 4, 4, 4)]
        >>> get_k_factors(6, 4)
        (3, 2, 1, 1)
        >>> get_k_factors(-12, 4)
        (3, 2, 2, -1)
        >>> get_k_factors(0, 4)
        (1, 1, 1, 0)
        >>> get_k_factors(720, 4)
        (6, 6, 5, 4)
        >>> get_k_factors(720, 4, '+')
        (4, 4, 9, 5)
        >>> get_k_factors(720, 3)
        (12, 10, 6)
        >>> get_k_factors(720, 3, '+')
        (8, 6, 15)
        >>> get_k_factors(720, 3, mode='-')
        (45, 4, 4)
        >>> get_k_factors(720, 3, mode='seed0')
        (18, 10, 4)
        >>> get_k_factors(720, 3, 'alt')
        (30, 4, 6)
        >>> get_k_factors(720, 3, 'alt1')
        (12, 6, 10)
        >>> get_k_factors(720, 3, '=')
        (12, 10, 6)
    """
    if k > 1:
        factors = list(get_factors(num))
        if len(factors) < k:
            factors.extend([1] * (k - len(factors)))
        groups = None
        if mode in ('increasing', 'ascending', '+'):
            factors = sorted(factors)
        elif mode in ('decreasing', 'descending', '-'):
            factors = sorted(factors, reverse=True)
        elif mode == 'random':
            shuffle(factors)
        elif mode.startswith('seed'):
            seed = auto_convert(mode[len('seed'):])
            random.seed(seed)
            shuffle(factors)
        elif mode.startswith('alt'):
            try:
                i = int(mode[len('alt'):]) % (len(factors) - 1)
            except ValueError:
                i = 0
            factors[i::2] = factors[i::2][::-1]
        elif mode in ('optimal', 'similar', '!', '='):
            groups = [[] for _ in itertools.repeat(None, k)]
            # could this algorithm could be improved?
            for factor in sorted(factors, reverse=True):
                groups = sorted(
                    groups, key=lambda x: prod(x) if len(x) > 0 else 0)
                groups[0].append(factor)
            groups = sorted(groups, key=prod, reverse=True)
        if not groups:
            groups = chunks(factors, k, mode='+', balanced=balanced)
        factorization = tuple(
            functools.reduce(lambda x, y: x * y, j) for j in groups)
    else:
        factorization = (num,)

    return factorization


# =====================================================================
def optimal_shape(
        num,
        dims=2,
        sort=lambda x: (sum(x), x[::-1]),
        reverse=False):
    """
    Find the optimal shape for arranging n elements into a rank-k tensor.

    Args:
        num (int): The number of elements to arrange.
        dims (int): The rank of the tensor.
        sort (callable): The function defining optimality.
            The factorization that minimizes (or maximizes)
            This is passed to the `key` arguments of `sorted()`.
        reverse (bool): The sorting direction.
            This is passed to the `reverse` arguments of `sorted()`.
            If False, sorting is ascending and the minimum of the optimization
            function is picked.
            Otherwise, sorting is descending and the maximum of the
            optimization
            function is picked.

    Returns:
        ratios (tuple[int]): The optimal ratio for tensor dims.

    Examples:
        >>> n1, n2 = 40, 46
        >>> [optimal_shape(i) for i in range(n1, n2)]
        [(8, 5), (41, 1), (7, 6), (43, 1), (11, 4), (9, 5)]
        >>> [optimal_shape(i, sort=max) for i in range(n1, n2)]
        [(5, 8), (1, 41), (6, 7), (1, 43), (4, 11), (5, 9)]
        >>> [optimal_shape(i, sort=min) for i in range(n1, n2)]
        [(2, 20), (1, 41), (2, 21), (1, 43), (2, 22), (3, 15)]
        >>> [optimal_shape(i, 3) for i in range(n1, n2)]
        [(5, 4, 2), (41, 1, 1), (7, 3, 2), (43, 1, 1), (11, 2, 2), (5, 3, 3)]
    """
    factorizations = get_k_factors_all(num, dims)
    return sorted(factorizations, key=sort, reverse=reverse)[0]


# ======================================================================
def _gcd(a, b):
    """
    Compute the greatest common divisor (GCD) of a and b.

    Unless b==0, the result will have the same sign as b (so that when
    b is divided by it, the result comes out positive).

    Examples:
        >>> _gcd(123, 45)
        3
        >>> _gcd(45, 123)
        3
        >>> _gcd(0, 1)
        1
        >>> _gcd(-3, 1)
        1
        >>> _gcd(211815584, 211815584)
        211815584

    Note:
        This should never be used as `math.gcd` offers identical functionality,
        but it is faster.
    """
    while b:
        a, b = b, a % b
    return a


# =====================================================================
def gcd(values):
    """
    Find the greatest common divisor (GCD) of a list of numbers.

    Args:
        values (Iterable[int]): The input numbers.

    Returns:
        result (int): The value of the greatest common divisor (GCD).

    Examples:
        >>> gcd((12, 24, 18))
        6
        >>> gcd((12, 24, 18, 42, 600, 66, 666, 768))
        6
        >>> gcd((12, 24, 18, 42, 600, 66, 666, 768, 101))
        1
        >>> gcd((12, 24, 18, 3))
        3
    """
    iter_values = iter(values)
    result = next(iter_values)
    for val in iter_values:
        result = math.gcd(result, val)
        if result == 1:
            break
    return result


# ======================================================================
def gcd_(*values):
    """
    Star magic version of `flyingcircus.gcd()`.

    Examples:
        >>> gcd_(12, 24, 18)
        6
        >>> gcd_(12, 24, 18, 42, 600, 66, 666, 768)
        6
        >>> gcd_(12, 24, 18, 42, 600, 66, 666, 768, 101)
        1
        >>> gcd_(12, 24, 18, 3)
        3
    """
    return gcd(values)


# ======================================================================
def lcm(values):
    """
    Find the least common multiple (LCM) of a list of numbers.

    Args:
        values (Iterable[int]): The input numbers.

    Returns:
        result (int): The value of the least common multiple (LCM).

    Examples:
        >>> lcm((2, 3, 4))
        12
        >>> lcm((9, 8))
        72
        >>> lcm((12, 23, 34, 45, 56))
        985320
    """
    iter_values = iter(values)
    result = next(iter_values)
    for value in iter_values:
        result *= value // math.gcd(result, value)
    return result


# ======================================================================
def lcm_(*items):
    """
    Star magic version of `flyingcircus.lcm()`.

    Examples:
        >>> lcm_(2, 3, 4)
        12
        >>> lcm_(9, 8)
        72
        >>> lcm_(12, 23, 34, 45, 56)
        985320
    """
    return lcm(items)


# ======================================================================
def simplify_frac(
        a,
        b,
        sign=True):
    """
    Simply a fraction.

    Args:
        a (int): The numerator.
        b (int): The denominator.
        sign (bool): Keep the sign in both the numerator and the denominator.
            If False, the sign is kept only in the numerator,
            unless the numerator is 0.

    Returns:
        result (tuple[int]): The tuple contains:
            - `a_n`: The simplified numerator.
            - `b_n`: The simplified denominator.

    Examples:
        >>> simplify_frac(12 * 7, 13 * 7)
        (12, 13)
        >>> simplify_frac(123, 321, True)
        (41, 107)
        >>> simplify_frac(123, -321, True)
        (41, -107)
        >>> simplify_frac(-123, 321, True)
        (-41, 107)
        >>> simplify_frac(-123, -321, True)
        (-41, -107)
        >>> simplify_frac(123, 321, False)
        (41, 107)
        >>> simplify_frac(123, -321, False)
        (-41, 107)
        >>> simplify_frac(-123, 321, False)
        (-41, 107)
        >>> simplify_frac(-123, -321, False)
        (41, 107)
        >>> simplify_frac(1234, 0)
        (1, 0)
        >>> simplify_frac(0, 1234)
        (0, 1)
        >>> simplify_frac(-1234, 0)
        (-1, 0)
        >>> simplify_frac(0, -1234)
        (0, -1)
        >>> simplify_frac(0, 0)
        (0, 0)
        >>> simplify_frac(-0, 0)
        (0, 0)
    """
    gcd = math.gcd(a, b)
    if gcd:
        if not sign and (b < 0 and a):
            return -a // gcd, -b // gcd
        else:
            return a // gcd, b // gcd
    else:
        return 0, 0


# ======================================================================
def cont_frac(b, a=1, limit=None):
    """
    Compute a fractional approximation of a continued fraction.

    The continued fraction is expressed in terms of the denominators and the
    numerators sequences:

    .. math::

        \\frac{a}{b} = b_0 + \\frac{a_1}{b_1 +} \\frac{a_2}{b_2 +} \\ldots

    Args:
        b (int|Iterable[int]): Generalized continued fraction denominators.
            This should include also the first integer.
            The first `limit` values are used.
            All but the first value must be positive.
        a (int|Iterable[int]): Generalized continued fraction numerators.
            The first value does not contribute to `a_n / b_n` and can
            safely be set to 1.
            All values must be positive.
        limit (int|None): Maximum number of iterations to produce.

    Returns:
        result (tuple[int]): The tuple contains:
            - `a_n`: The numerator at the specified iteration level.
            - `b_n`: The denominator at the specified iteration level.

    Examples:
        >>> num, den = cont_frac([1, 2])
        >>> print(num, den, round(num / den, 8))
        3 2 1.5

        >>> num, den = cont_frac([0, 1, 5, 2, 2])
        >>> print(num, den, round(num / den, 8))
        27 32 0.84375

        >>> num, den = cont_frac([2, 2, 2], [1, 2, 2])
        >>> print(num, den, round(num / den, 8))
        8 3 2.66666667

        >>> num, den = cont_frac([2, 2, 2, 2], [1, 2, 2, 2])
        >>> print(num, den, round(num / den, 8))
        11 4 2.75

        >>> num, den = cont_frac([1, 2, 3, 4], [1, 2, 3, 4])
        >>> print(num, den, round(num / den, 8))
        19 11 1.72727273

        >>> num, den = cont_frac(range(1, 10), range(1, 20), 4)
        >>> print(num, den, round(num / den, 8))
        19 11 1.72727273

        >>> num, den = cont_frac(2, 2, 3)
        >>> print(num, den, round(num / den, 8))
        8 3 2.66666667

        >>> num, den = cont_frac([1, 2, 3, 4], 2)
        >>> print(num, den, round(num / den, 8))
        16 9 1.77777778

        >>> num, den = cont_frac(2, [1, 2, 3, 4])
        >>> print(num, den, round(num / den, 8))
        30 11 2.72727273

        >>> num, den = cont_frac(2, [12, 2, 3, 4])
        >>> print(num, den, round(num / den, 8))
        30 11 2.72727273

        Computing √2

        >>> n = 12
        >>> b = [1] + [2] * (n - 1)
        >>> num, den = cont_frac(b)
        >>> print(num, den, round(num / den, 8))
        19601 13860 1.41421356

        Computing Archimede's constant π

        >>> num, den = cont_frac([3, 7, 15, 1, 292, 1, 1, 1, 2, 1, 3, 1])
        >>> print(num, den, round(num / den, 8))
        5419351 1725033 3.14159265
        >>> n = 16
        >>> b = [0] + [(2 * i + 1) for i in range(n - 1)]
        >>> a = [1, 4] + [(i + 1) ** 2 for i in range(n - 2)]
        >>> num, den = cont_frac(b, a)
        >>> print(num, den, round(num / den, 8))
        10392576224 3308059755 3.14159265
        >>> n = 16
        >>> b = [3] + [6] * (n - 1)
        >>> a = [1] + [(2 * i + 1) ** 2 for i in range(n - 1)]
        >>> num, den = cont_frac(b, a)
        >>> print(num, den, round(num / den, 8))
        226832956041173 72201776446800 3.14165339
        >>> n = 16
        >>> b = [0, 1] + [2] * (n - 2)
        >>> a = [1, 4] + [(2 * i + 1) ** 2 for i in range(n - 2)]
        >>> num, den = cont_frac(b, a)
        >>> print(num, den, round(num / den, 8))
        467009482388 145568097675 3.20818565

        Computing Euler's Number e

        >>> n = 16
        >>> b = [2] + [2 * i // 3 if not i % 3 else 1 for i in range(2, n + 1)]
        >>> num, den = cont_frac(b)
        >>> print(num, den, round(num / den, 8))
        566827 208524 2.71828183

        Computing Golden ratio φ

        >>> n = 24
        >>> num, den = cont_frac(1, 1, n)
        >>> print(num, den, round(num / den, 8))
        75025 46368 1.61803399
    """
    if limit is None:
        try:
            len_b = len(b)
        except TypeError:
            len_b = -1
        try:
            len_a = len(a)
        except TypeError:
            len_a = -1
        if len_b < 0 and len_a < 0:
            raise ValueError('Limit must be specified for given `a` and `b`.')
        elif len_b > 0 and len_a > 0:
            limit = min(len_b, len_a)
        else:
            limit = max(len_b, len_a)
    try:
        b = b[:limit]
    except TypeError:
        pass
    try:
        a = a[:limit]
    except TypeError:
        pass
    try:
        r_iter_b = reversed(b)
    except TypeError:
        try:
            r_iter_b = reversed([x for x, _ in zip(b, range(limit))])
        except TypeError:
            r_iter_b = itertools.cycle([b])
    try:
        r_iter_a = reversed(a)
    except TypeError:
        try:
            r_iter_a = reversed([x for x, _ in zip(a, range(limit))])
        except TypeError:
            r_iter_a = itertools.cycle([a])
    a_i = 1
    a_n, b_n = 1, 0
    for a_i, b_i, _ in zip(r_iter_a, r_iter_b, range(limit)):
        a_n, b_n = b_n + a_n * b_i, a_n * a_i
    b_n //= a_i
    return simplify_frac(a_n, b_n)


# ======================================================================
def polynomial(
        p,
        x,
        method='horner'):
    """
    Yield the result of the polynomial evaluation at given points.

    y_i = sum(p_i * x_i ** i for i, p_i in enumerate(p[::-1]))

    Args:
        p (Sequence[Number]): The coefficients of the polynomial.
            Must be given in decreasing order.
        x (Number|Iterable[Number]): The evaluation point(s).
        method (str): Polynomial computation method.
            Accepted values are:
             - 'direct': use the direct computation (slower)
             - 'horner': use Horner's formula (faster, but less accurate)

    Yields:
        y_i (Number): The evaluated polynomial.

    Examples:
        >>> list(polynomial([1, 2, 3], range(16)))
        [3, 6, 11, 18, 27, 38, 51, 66, 83, 102, 123, 146, 171, 198, 227, 258]
        >>> list(polynomial([4, 0, 2, 1], range(10)))
        [1, 7, 37, 115, 265, 511, 877, 1387, 2065, 2935]
        >>> list(polynomial([1, -2], range(-8, 8)))
        [-10, -9, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        >>> list(polynomial([1], range(-8, 8)))
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        >>> list(polynomial([1, 0], range(-8, 8)))
        [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
        >>> list(polynomial([4, 0, 2, 1], 1))
        [7]
        >>> list(polynomial([1, 2, 3], range(16), 'direct'))
        [3, 6, 11, 18, 27, 38, 51, 66, 83, 102, 123, 146, 171, 198, 227, 258]

    References:
        - Horner, W. G., and Davies Gilbert. “XXI. A New Method of Solving
          Numerical Equations of All Orders, by Continuous Approximation.”
          Philosophical Transactions of the Royal Society of London 109 (
          January 1, 1819): 308–35. https://doi.org/10.1098/rstl.1819.0023.
        - https://en.wikipedia.org/wiki/Horner%27s_method
    """
    method = method.lower()
    if not is_deep(x):
        x = (x,)
    if method == 'direct':
        d = len(p) - 1
        for x_i in x:
            yield sum(p_i * x_i ** (d - i) for i, p_i in enumerate(p))
    elif method == 'horner':
        for x_i in x:
            y_i = 0
            for p_i in p:
                y_i = x_i * y_i + p_i
            yield y_i


# ======================================================================
def mean(seq):
    """
    Compute the arithmetic mean of a numeric sequence.

    For iterative computation see:
     - `flyingcircus.next_amean()`
     - `flyingcircus.i_amean()`
     - `flyingcircus.i_mean()`

    This is substantially faster than `statistics.mean()`.

    Args:
        seq (Sequence[Number]): The input items.

    Returns:
        result (Number): The arithmetic mean.

    Examples:
        >>> items = range(0, 52, 2)
        >>> mean(items)
        25.0
        >>> statistics.mean(items) == mean(items)
        True

        >>> [mean(range(n + 1)) for n in range(10)]
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

    See Also:
        - flyingcircus.gmean()
        - flyingcircus.hmean()
        - flyingcircus.mean_and_soad()
        - flyingcircus.mean_and_sosd()
        - flyingcircus.mean_and_var()
        - flyingcircus.mean_and_stdev()
        - flyingcircus.mean_and_mean_abs_dev()
        - flyingcircus.median()
        - flyingcircus.next_amean()
        - flyingcircus.i_amean()
        - flyingcircus.i_mean()
    """
    return sum(seq) / len(seq)


# ======================================================================
def gmean(
        seq,
        valid=True):
    """
    Compute the geometric mean of a numeric sequence.

    Args:
        seq (Sequence[Number]): The input items.
        valid (bool): Include only valid (non-zero) items.

    Returns:
        result (Number): The geometric mean.

    Examples:
        >>> items = range(0, 52, 2)
        >>> round(gmean(items), 3)
        20.354
        >>> gmean(items, False)
        0.0

        >>> [round(gmean(range(n + 1)), 3) for n in range(10)]
        [1.0, 1.0, 1.414, 1.817, 2.213, 2.605, 2.994, 3.38, 3.764, 4.147]

    See Also:
        - flyingcircus.mean()
        - flyingcircus.hmean()
        - flyingcircus.median()
        - flyingcircus.next_gmean()
        - flyingcircus.i_gmean()
    """
    if valid and seq:
        seq = tuple(i for i in seq if i)
        if not seq:
            return 1.0
    return prod(seq) ** (1 / len(seq))


# ======================================================================
def hmean(
        seq,
        valid=True):
    """
    Compute the harmonic mean of a numeric sequence.

    Args:
        seq (Sequence[Number]): The input items.
            The values within the sequence should be numeric.
        valid (bool): Include only valid (non-zero) items.

    Returns:
        result (Number): The harmonic mean.

    Examples:
        >>> items = range(0, 52, 2)
        >>> round(hmean(items), 3)
        13.103
        >>> hmean(items, False)
        Traceback (most recent call last):
            ...
        ZeroDivisionError: division by zero

        >>> [round(hmean(range(n + 1)), 3) for n in range(10)]
        [0.0, 1.0, 1.333, 1.636, 1.92, 2.19, 2.449, 2.7, 2.943, 3.181]

    See Also:
        - flyingcircus.mean()
        - flyingcircus.gmean()
        - flyingcircus.median()
        - flyingcircus.next_hmean()
        - flyingcircus.i_hmean()
    """
    if valid:
        seq = tuple((1 / i) for i in seq if i)
        if not seq:
            return 0.0
    else:
        seq = tuple((1 / i) for i in seq)
    return 1 / mean(seq)


# ======================================================================
def absolute_deviations(
        value,
        seq):
    """
    Yield the absolute deviations of the items from a value.

    Args:
        value (Number): The input value.
        seq (Sequence[Number]): The input items.

    Yields:
        result (Number): The next absolute deviation.

    Examples:
        >>> list(absolute_deviations(10, range(16)))
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 1, 2, 3, 4, 5]

    See Also:
        - flyingcircus.squared_deviations()
        - flyingcircus.soad()
        - flyingcircus.mean_and_soad()
        - flyingcircus.mean_abs_dev()
    """
    for item in seq:
        yield abs(value - item)


# ======================================================================
def squared_deviations(
        value,
        seq):
    """
    Yield the squared deviations of the items from a value.

    Args:
        value (Number): The input value.
        seq (Sequence[Number]): The input items.

    Yields:
        result (Number): The next squared deviation.

    Examples:
        >>> list(squared_deviations(10, range(16)))
        [100, 81, 64, 49, 36, 25, 16, 9, 4, 1, 0, 1, 4, 9, 16, 25]

    See Also:
        - flyingcircus.absolute_deviations()
        - flyingcircus.sosd()
        - flyingcircus.mean_and_sosd()
        - flyingcircus.var()
    """
    for item in seq:
        yield (value - item) * (value - item)


# ======================================================================
def mean_and_soad(seq):
    """
    Compute the mean and the sum-of-absolute-deviations of a numeric sequence.

    The sum-of-absolute-deviations (SoAD) is useful for computing the
    mean absolute deviation (MeanAD).

    Args:
        seq (Sequence[Number]): The input items.

    Returns:
        result (tuple): The tuple
            contains:
             - `mean_val` (Number): The mean of the items.
             - `soad_val` (Number): The SoAD of the items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> mean_and_soad(items)
        (25.0, 338.0)
        >>> mean_and_soad(items) == (mean(items), soad(items))
        True

    See Also:
        - flyingcircus.absolute_deviations()
        - flyingcircus.mean()
        - flyingcircus.soad()
        - flyingcircus.mean_and_sosd()
        - flyingcircus.mean_and_mean_abs_dev()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.i_mean_and_sosd()
    """
    mean_val = mean(seq)
    soad_val = sum(absolute_deviations(mean_val, seq))
    return mean_val, soad_val


# ======================================================================
def soad(seq):
    """
    Compute the sum-of-absolute-deviations of a numeric sequence.

    The sum-of-absolute-deviations (SoAD) is useful for computing the
    mean absolute deviation (MeanAD).

    Args:
        seq (Sequence[Number]): The input items.

    Returns:
        result (Number): The sum-of-absolute-deviations (SoAD) of the items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> soad(items)
        338.0

    See Also:
        - flyingcircus.absolute_deviations()
        - flyingcircus.mean_and_soad()
        - flyingcircus.sosd()
        - flyingcircus.mean_and_sosd()
    """
    return sum(absolute_deviations(mean(seq), seq))


# ======================================================================
def mean_and_sosd(seq):
    """
    Compute the mean and the sum-of-squared-deviations of a numeric sequence.

    For iterative computation see:
     - `flyingcircus.next_mean_and_sosd()`
     - `flyingcircus.i_mean_and_sosd()`

    The sum-of-squared-deviations (SoSD) is useful for numerically stable
    computation of the variance and the standard deviation.

    Args:
        seq (Sequence[Number]): The input items.

    Returns:
        result (tuple): The tuple
            contains:
             - `mean_val` (Number): The mean of the items.
             - `sosd_val` (Number): The SoSD of the items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> mean_and_sosd(items)
        (25.0, 5850.0)
        >>> mean_and_sosd(items) == (mean(items), sosd(items))
        True

    See Also:
        - flyingcircus.squared_deviations()
        - flyingcircus.mean()
        - flyingcircus.sosd()
        - flyingcircus.mean_and_soad()
        - flyingcircus.mean_and_var()
        - flyingcircus.mean_and_stdev()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.i_mean_and_sosd()
    """
    mean_val = mean(seq)
    sosd_val = sum(squared_deviations(mean_val, seq))
    return mean_val, sosd_val


# ======================================================================
def sosd(seq):
    """
    Compute the mean and the sum-of-squared-deviations of a numeric sequence.

    For iterative computation see:
     - `flyingcircus.next_mean_and_sosd()`
     - `flyingcircus.i_mean_and_sosd()`

    The sum-of-squared-deviations (SoSD) is useful for numerically stable
    computation of the variance and the standard deviation.

    Args:
        seq (Sequence[Number]): The input items.

    Returns:
        result (Number): The sum-of-squared-deviations (SoSD) of the items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> sosd(items)
        5850.0

    See Also:
        - flyingcircus.squared_deviations()
        - flyingcircus.mean_and_sosd()
        - flyingcircus.soad()
        - flyingcircus.sosd2var()
        - flyingcircus.var2sosd()
        - flyingcircus.sosd2stdev()
        - flyingcircus.stdev2sosd()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.i_mean_and_sosd()
    """
    return sum(squared_deviations(mean(seq), seq))


# ======================================================================
def sosd2var(
        sosd_,
        num,
        ddof=0):
    """
    Compute the variance from the sum-of-squared-deviations.

    Args:
        sosd_ (Number): The sum-of-squared-deviations value.
        num (int): The number of items.
        ddof (int): The number of degrees of freedom.

    Returns:
        result (Number): The variance value.

    Examples:
        >>> sosd2var(8, 3, 1)
        4.0

        >>> all([var2sosd(sosd2var(x, n, m), n, m) == x
        ...     for x in range(9) for n in range(3, 9) for m in range(3)])
        True

    See Also:
        - flyingcircus.sosd()
        - flyingcircus.var()
        - flyingcircus.var2sosd()
        - flyingcircus.sosd2stdev()
        - flyingcircus.stdev2sosd()
    """
    return sosd_ / (num - ddof)


# ======================================================================
def var2sosd(
        var_,
        num,
        ddof=0):
    """
    Compute the sum-of-squared-deviations from the variance.

    Args:
        var_ (Number): The variance value.
        num (int): The number of items.
        ddof (int): The number of degrees of freedom.

    Returns:
        result (Number): The sum-of-squared-deviations value.

    Examples:
        >>> var2sosd(4, 3, 1)
        8

        >>> all([sosd2var(var2sosd(x, n, m), n, m) == x
        ...     for x in range(9) for n in range(3, 9) for m in range(3)])
        True

    See Also:
        - flyingcircus.sosd()
        - flyingcircus.var()
        - flyingcircus.sosd2var()
        - flyingcircus.sosd2stdev()
        - flyingcircus.stdev2sosd()
    """
    return var_ * (num - ddof)


# ======================================================================
def sosd2stdev(
        sosd_,
        num,
        ddof=0):
    """
    Compute the standard deviation from the sum-of-squared-deviations.

    Args:
        sosd_ (Number): The sum-of-squared-deviations value.
        num (int): The number of items.
        ddof (int): The number of degrees of freedom.

    Returns:
        result (Number): The standard deviation value.

    Examples:
        >>> sosd2stdev(8, 3, 1)
        2.0

        >>> all([sosd2stdev(stdev2sosd(x, n, m), n, m) == x
        ...     for x in range(9) for n in range(3, 9) for m in range(3)])
        True

    See Also:
        - flyingcircus.sosd()
        - flyingcircus.stdev()
        - flyingcircus.sosd2var()
        - flyingcircus.var2sosd()
        - flyingcircus.stdev2sosd()
    """
    return (sosd_ / (num - ddof)) ** 0.5


# ======================================================================
def stdev2sosd(
        stdev_,
        num,
        ddof=0):
    """
    Compute the sum-of-squared-deviations from the standard deviation.

    Args:
        stdev_ (Number): The variance value.
        num (int): The number of items.
        ddof (int): The number of degrees of freedom.

    Returns:
        sosd (Number): The sum-of-squared-deviations value.

    Examples:
        >>> stdev2sosd(2, 3, 1)
        8

        >>> all([abs(stdev2sosd(sosd2stdev(x, n, m), n, m) - x) < 1e-9
        ...     for x in range(9) for n in range(3, 9) for m in range(3)])
        True

    See Also:
        - flyingcircus.sosd()
        - flyingcircus.stdev()
        - flyingcircus.sosd2var()
        - flyingcircus.var2sosd()
        - flyingcircus.sosd2stdev()
    """
    return (stdev_ * stdev_) * (num - ddof)


# ======================================================================
def var(
        seq,
        ddof=0):
    """
    Compute the variance of a numeric sequence.

    For iterative computation see:
     - `flyingcircus.next_mean_and_var()`
     - `flyingcircus.next_mean_and_sosd()` and `.sosd2var()`.
     - `flyingcircus.i_var()`
     - `flyingcircus.i_mean_and_var()`

    This is substantially faster than `statistics.variance()`.

    Note that the variance is the mean of squared deviations.

    Args:
        seq (Sequence[Number]): The input items.
        ddof (int): The number of degrees of freedom.

    Returns:
        result (Number): The variance of the items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> var(items)
        225.0
        >>> statistics.variance(items) == var(items, 1)
        True

    See Also:
        - flyingcircus.i_var()
        - flyingcircus.i_mean_and_var()
        - flyingcircus.next_mean_and_var()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.mean_and_var()
        - flyingcircus.mean_and_sosd()
        - flyingcircus.stdev()
        - flyingcircus.sosd2var()
        - flyingcircus.var2sosd()
    """
    mean_val, sosd_val = mean_and_sosd(seq)
    return sosd2var(sosd_val, len(seq), ddof)


# ======================================================================
def stdev(
        seq,
        ddof=0):
    """
    Compute the standard deviation of a numeric sequence.

    For iterative computation see:
     - `flyingcircus.next_mean_and_stdev()`
     - `flyingcircus.next_mean_and_sosd()` and `.sosd2stdev()`.
     - `flyingcircus.i_stdev()`
     - `flyingcircus.i_mean_and_stdev()`

    This is substantially faster than `statistics.stdev()`.

    Note that the standard deviation is the square root of the variance,
    and hence it is also the square root of the mean of squared deviations.

    Args:
        seq (Sequence[Number]): The input items.
            The values within the sequence should be numeric.
        ddof (int): The number of degrees of freedom.

    Returns:
        result (Number): The standard deviation of the items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> stdev(items)
        15.0
        >>> statistics.stdev(items) == stdev(items, 1)
        True

    See Also:
        - flyingcircus.i_stdev()
        - flyingcircus.i_mean_and_stdev()
        - flyingcircus.next_mean_and_stdev()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.mean_and_stdev()
        - flyingcircus.mean_and_sosd()
        - flyingcircus.var()
        - flyingcircus.sosd2stdev()
        - flyingcircus.stdev2sosd()
    """
    mean_val, sosd_val = mean_and_sosd(seq)
    return sosd2stdev(sosd_val, len(seq), ddof)


# ======================================================================
def mean_and_var(
        seq,
        ddof=0):
    """
    Compute the mean and variance of a numeric sequence.

    For iterative computation see:
     - `flyingcircus.next_amean()`
     - `flyingcircus.next_mean_and_var()`
     - `flyingcircus.next_mean_and_sosd()` and `.sosd2var()`.
     - `flyingcircus.i_mean()`
     - `flyingcircus.i_var()`
     - `flyingcircus.i_mean_and_var()`
     - `flyingcircus.i_mean_and_sosd()` and `.sosd2var()`

    This is faster than computing the two values separately.

    Args:
        seq (Sequence[Number]): The input items.
            The values within the sequence should be numeric.
        ddof (int): The number of degrees of freedom.

    Returns:
        result (tuple): The tuple
            contains:
             - `mean_` (Number): The mean of the items.
             - `var_` (Number): The variance of the items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> mean_and_var(items)
        (25.0, 225.0)
        >>> mean_and_var(items) == (mean(items), var(items))
        True

    See Also:
        - flyingcircus.mean()
        - flyingcircus.var()
        - flyingcircus.i_mean_and_var()
        - flyingcircus.next_mean_and_var()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.mean_and_sosd()
        - flyingcircus.sosd2var()
        - flyingcircus.var2sosd()
    """
    mean_, sosd_ = mean_and_sosd(seq)
    return mean_, sosd2var(sosd_, len(seq), ddof)


# ======================================================================
def mean_and_stdev(
        seq,
        ddof=0):
    """
    Compute the mean and the standard deviation of a numeric sequence.

    For iterative computation see:
     - `flyingcircus.next_amean()`
     - `flyingcircus.next_mean_and_stdev()`
     - `flyingcircus.next_mean_and_sosd()` and `.sosd2stdev()`.
     - `flyingcircus.i_mean()`
     - `flyingcircus.i_stdev()`
     - `flyingcircus.i_mean_and_stdev()`
     - `flyingcircus.i_mean_and_sosd()` and `.sosd2stdev()`

    This is faster than computing the two values separately.

    Args:
        seq (Sequence[Number]): The input items.
            The values within the sequence should be numeric.
        ddof (int): The number of degrees of freedom.

    Returns:
        result (tuple): The tuple
            contains:
             - `mean_` (Number): The mean of the items.
             - `stdev_` (Number): The standard deviation of the items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> mean_and_stdev(items)
        (25.0, 15.0)
        >>> mean_and_stdev(items) == (mean(items), stdev(items))
        True

    See Also:
        - flyingcircus.mean()
        - flyingcircus.stdev()
        - flyingcircus.i_mean_and_stdev()
        - flyingcircus.next_mean_and_stdev()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.mean_and_sosd()
        - flyingcircus.sosd2stdev()
        - flyingcircus.stdev2sosd()
    """
    mean_, sosd_ = mean_and_sosd(seq)
    return mean_, sosd2stdev(sosd_, len(seq), ddof)


# ======================================================================
def mean_and_mean_abs_dev(seq):
    """
    Compute the mean and the mean absolute deviation of a numeric sequence.

    This is faster than computing the two values separately.

    Args:
        seq (Sequence[Number]): The input items.

    Returns:
        result (tuple): The tuple
            contains:
             - `mean_` (Number): The mean of the items.
             - `mean_abs_dev_` (Number): The mean abs. dev. of the items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> mean_and_mean_abs_dev(items)
        (25.0, 13.0)
        >>> mean_and_mean_abs_dev(items) == (mean(items), mean_abs_dev(items))
        True

    See Also:
        - flyingcircus.mean()
        - flyingcircus.mean_abs_dev()
        - flyingcircus.mean_and_soad()
        - flyingcircus.soad()
        - flyingcircus.absolute_deviations()
    """
    mean_, soad_ = mean_and_soad(seq)
    mean_abs_dev_ = soad_ / len(seq)
    return mean_, mean_abs_dev_


# ======================================================================
def mean_abs_dev(seq):
    """
    Compute the mean absolute deviation of a numeric sequence.

    Args:
        seq (Sequence[Number]): The input items.

    Returns:
        result (Number): The mean absolute deviation of the items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> mean_abs_dev(items)
        13.0

    See Also:
        - flyingcircus.mean()
        - flyingcircus.mean_abs_dev()
        - flyingcircus.mean_and_soad()
        - flyingcircus.soad()
        - flyingcircus.absolute_deviations()
    """
    return soad(seq) / len(seq)


# ======================================================================
def median(
        seq,
        force_sort=True):
    """
    Compute the median of a numeric sequence.

    For iterative computation see:
     - `flyingcircus.next_median()`
     - `flyingcircus.next_medoid_and_median()`
     - `flyingcircus.i_median()`
     - `flyingcircus.i_medoid_and_median()`
     - `flyingcircus.i_median_and_median_abs_dev()`

    This is roughly comparable to `statistics.median()`.

    If more than one among median, medoid, quantile, quantiloid is needed,
    it is more efficient to sort the items prior to calling and then
    perform the multiple required calle with `force_sort` set to False.

    Args:
        seq (Sequence[Number]): The input items.
        force_sort (bool): Force sorting of the input items.
            If the items are already sorted, this can be safely set to False.
            Otherwise, it should be set to True.

    Returns:
        result (Number): The median of the items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> median(items)
        25.0
        >>> statistics.median(items) == median(items)
        True

    See Also:
        - flyingcircus.median_and_median_abs_dev()
        - flyingcircus.next_median()
        - flyingcircus.next_medoid_and_median()
        - flyingcircus.i_median()
        - flyingcircus.i_medoid_and_median()
        - flyingcircus.i_median_and_median_abs_dev()
        - flyingcircus.medoid()
        - flyingcircus.quantile()
        - flyingcircus.interquantilic_range()
        - flyingcircus.sym_interquantilic_range()
        - flyingcircus.median_abs_dev()
    """
    n = len(seq)
    i = n // 2
    sorted_items = sorted(seq) if force_sort else seq
    if not (n % 2) and sorted_items[i - 1] != sorted_items[i]:
        median_ = (sorted_items[i - 1] + sorted_items[i]) / 2
    else:
        median_ = sorted_items[i]
    return median_


# ======================================================================
def median_and_median_abs_dev(
        seq,
        force_sort=True):
    """
    Compute the median and the median absolute deviation of a numeric sequence.

    Args:
        seq (Sequence[Number]): The input items.
        force_sort (bool): Force sorting of the input items.
            If the items are already sorted, this can be safely set to False.
            Otherwise, it must be set to True.

    Returns:
        result (Number): The median deviation of the items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> median_and_median_abs_dev(items)
        (25.0, 13.0)
        >>> (median_and_median_abs_dev(items) ==
        ...     (median(items), median_abs_dev(items)))
        True

    See Also:
        - flyingcircus.median()
        - flyingcircus.median_abs_dev()
        - flyingcircus.i_median()
        - flyingcircus.i_median_abs_dev()
        - flyingcircus.i_median_and_median_abs_dev()
        - flyingcircus.absolute_deviations()

    """
    median_val = median(seq, force_sort=force_sort)
    delta_items = tuple(absolute_deviations(median_val, seq))
    return median_val, median(delta_items, force_sort=force_sort)


# ======================================================================
def median_abs_dev(
        seq,
        force_sort=True):
    """
    Compute the median absolute deviation of a numeric sequence.

    Args:
        seq (Sequence[Number]): The input items.
        force_sort (bool): Force sorting of the input items.
            If the items are already sorted, this can be safely set to False.
            Otherwise, it must be set to True.

    Returns:
        result (Number): The median absolute deviation of the items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> median_abs_dev(items)
        13.0

    See Also:
        - flyingcircus.median()
        - flyingcircus.median_and_median_abs_dev()
        - flyingcircus.i_median()
        - flyingcircus.i_median_abs_dev()
        - flyingcircus.i_median_and_median_abs_dev()
        - flyingcircus.absolute_deviations()
    """
    return median(
        tuple(
            absolute_deviations(median(seq, force_sort=force_sort), seq)),
        force_sort=force_sort)


# ======================================================================
def quantile(
        seq,
        factor,
        int_base=100,
        interp='linear',
        tol=1e-7,
        force_sort=True):
    """
    Compute the quantile of a numeric sequence.

    There is no efficient iterative version for any arbitrary factor.

    If more than one among median, medoid, quantile, quantiloid is needed,
    it is more efficient to sort the items prior to calling and then
    perform the multiple required calle with `force_sort` set to False.

    Args:
        seq (Sequence[Number]): The input items.
        factor (int|float|Iterable[int|float]): The quantile index.
            If float, must be a number in the [0, 1] range.
            If int, it is divided by `int_base` and its value must be in the
            [0, int_base] range.
        int_base (int|float): The denominator used to scale integer factors.
        interp (str): Interpolation for inexact quantiles.
            This determines
        tol (float): Tolerance for detecting exact indices.
        force_sort (bool): Force sorting of the input items.
            If the items are already sorted, this can be safely set to False.
            Otherwise, it must be set to True.

    Returns:
        result (Number|tuple[Number]): The quantiles of the item(s).
            Returns the tuple if `factor` is a sequence (of any length),
            otherwise returns a Number.

    Examples:
        >>> items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> quantile(items, 0.25)
        2.5
        >>> quantile(items, range(0, 101, 10))
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        >>> ((min(items), median(items), max(items))
        ...     == quantile(items, (0.0, 0.5, 1.0)))
        True

        >>> interps = 'linear', 'lower', 'upper', 'midpoint', 'nearest'
        >>> [quantile([0, 1], 1, 2, interp) for interp in interps]
        [0.5, 0, 1, 0.5, 0]

        >>> [quantile([0, 1], 0.6, interp=interp) for interp in interps]
        [0.6, 0, 1, 0.5, 1]

        >>> [quantile([0, 1, 2], 1, 2, interp=interp) for interp in interps]
        [1, 1, 1, 1, 1]

        >>> [quantile([0, 1, 2], 0.5, interp=interp) for interp in interps]
        [1, 1, 1, 1, 1]

        >>> quantile([10, 11, 12], [0, 1, 2], 2)
        (10, 11, 12)

        >>> quantile([10, 11, 12], -1)
        Traceback (most recent call last):
            ...
        ValueError: Invalid factor `-1` (negative)
        >>> quantile([10, 11, 12], -0.1)
        Traceback (most recent call last):
            ...
        ValueError: Invalid factor `-0.1` (negative)
        >>> quantile([10, 11, 12], 101)
        Traceback (most recent call last):
            ...
        ValueError: Invalid integer factor `101` (larger than `int_base=100`)
        >>> quantile([10, 11, 12], 1.1)
        Traceback (most recent call last):
            ...
        ValueError: Invalid factor `1.1` (larger than 1.0)

    See Also:
        - flyingcircus.median()
        - flyingcircus.quantiloid()
        - flyingcircus.interquantilic_range()
        - flyingcircus.sym_interquantilic_range()
    """
    # interps = 'linear', 'lower', 'upper', 'midpoint', 'nearest'
    use_tuple = is_deep(factor)
    kk = auto_repeat(factor, 1, False, False)
    n = len(seq)
    interp = interp.lower()
    sorted_items = sorted(seq) if force_sort else seq
    is_exact = not ((n - 1) % int_base)
    exact_factor = (n - 1) // int_base
    result = []
    for k in kk:
        if k < 0:
            raise ValueError(
                fmtm('Invalid factor `{k}` (negative)'))
        elif isinstance(k, int) and k > int_base:
            raise ValueError(
                fmtm('Invalid integer factor `{k}`'
                     ' (larger than `int_base={int_base}`)'))
        if isinstance(k, int) and is_exact:
            i = k * exact_factor
            x = sorted_items[i]
        else:
            if isinstance(k, int):
                k = k / int_base
            if k > 1.0:
                raise ValueError(
                    fmtm('Invalid factor `{k}` (larger than 1.0)'))
            i = k * (n - 1)
            int_i = int(i)
            frac_i = i % 1
            if abs(frac_i) < tol:
                x = sorted_items[int_i]
            else:  # if 0 <= int_i < (n - 1):
                if interp == 'nearest':
                    x = sorted_items[int(round(i))]
                elif interp == 'lower':
                    x = sorted_items[int_i]
                elif interp == 'upper':
                    x = sorted_items[int_i + 1]
                elif interp in ('midpoint', 'linear'):
                    a = sorted_items[int_i]
                    b = sorted_items[int_i + 1]
                    if a == b:
                        x = sorted_items[int_i]
                    else:
                        if interp == 'linear':
                            x = a + (b - a) * frac_i
                        else:  # if interp == 'midpoint':
                            x = (b + a) / 2
                else:
                    raise ValueError(fmtm('Invalid interpolation `{interp}`.'))
        result.append(x)
    return tuple(result) if use_tuple else result[0]


# ======================================================================
def medoid(
        seq,
        force_sort=True,
        lower=True):
    """
    Compute the medoid of an arbitrary sequence.

    For some inputs `select_ordinal()` may be faster.

    If more than one among median, medoid, quantile, quantiloid is needed,
    it is more efficient to sort the items prior to calling and then
    perform the multiple required calls with `force_sort` set to False.

    For iterative computation see:
     - `flyingcircus.next_medoid()`
     - `flyingcircus.i_medoid()`

    If the number of items is not odd, returns the medoid from the lower or
    upper half depending on the value of `lower` being True or False.

    Args:
        seq (Sequence[Number|Any]): The input items.
            The values within the sequence do not need to be numeric.
        force_sort (bool): Force sorting of the input items.
            If the items are already sorted, this can be safely set to False.
            Otherwise, it should be set to True.
        lower (bool): Pick the lower medoid for even-sized items.

    Returns:
        result (Any): The medoid of the items.

    Examples:
        >>> items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> medoid(items)
        5
        >>> medoid(items, lower=False)
        5

        >>> items = range(0, 20, 2)
        >>> medoid(items)
        8
        >>> medoid(items, lower=False)
        10

        >>> items = string.ascii_lowercase
        >>> medoid(items)
        'm'
        >>> medoid(items, lower=False)
        'n'

    See Also:
        - flyingcirucs.select_ordinal()
        - flyingcircus.next_medoid()
        - flyingcircus.next_medoid_and_median()
        - flyingcircus.i_medoid()
        - flyingcircus.i_medoid_and_median()
        - flyingcircus.median()
        - flyingcircus.quantiloid()
        - flyingcircus.interquantilic_range()
        - flyingcircus.sym_interquantilic_range()
    """
    n = len(seq)
    i = int(round((n - 1) / 2)) if lower else n // 2
    sorted_items = sorted(seq) if force_sort else seq
    medoid_ = sorted_items[i]
    return medoid_


# ======================================================================
def quantiloid(
        seq,
        factor,
        int_base=100,
        rounding=round,
        force_sort=True):
    """
    Compute the quantiloid of an arbitrary sequence.

    There is no efficient iterative version for any arbitrary factor.
    For some inputs `select_ordinal()` may be faster.

    If more than one among median, medoid, quantile, quantiloid is needed,
    it is more efficient to sort the items prior to calling and then
    perform the multiple required calle with `force_sort` set to False.

    Args:
        seq (Sequence[Number|Any]): The input items.
            The values within the sequence do not need to be numeric.
        factor (int|float|Iterable[int|float]): The quantile index.
            If float, must be a number in the [0, 1] range.
            If int, it is divided by `int_base` to get to the [0, 1] range.
        int_base (int|None): The denominator for scaling integer factors.
            If None (or zero), uses `len(items)`.
        rounding (callable): The rounding to use for determining the index.
            Use `round` to pick the closest index.
            Use `math.floor` to pick the lower index.
            Use `math.ceil` to pick the upper index.
        force_sort (bool): Force sorting of the input items.
            If the items are in the intended order, it should be set to False.
            Otherwise, it should be set to True.

    Returns:
        result (Any|tuple[Any]): The quantiloid of the item(s).
            Returns the tuple if `factor` is a sequence (of any length),
            otherwise returns an element of items.

    Examples:
        >>> items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> quantiloid(items, 0.25)
        2
        >>> quantiloid(items, range(0, 101, 10))
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        >>> ((min(items), medoid(items), max(items))
        ...     == quantiloid(items, (0.0, 0.5, 1.0)))
        True

        >>> items = string.ascii_lowercase
        >>> quantiloid(items, 0.25)
        'g'
        >>> quantiloid(items, range(0, 101, 10))
        ('a', 'c', 'f', 'i', 'k', 'm', 'p', 's', 'u', 'w', 'z')
        >>> ((min(items), medoid(items), max(items))
        ...     == quantiloid(items, (0.0, 0.5, 1.0)))
        True

    See Also:
        - flyingcircus.medoid()
        - flyingcircus.quantile()
        - flyingcircus.interquantilic_range()
        - flyingcircus.sym_interquantilic_range()
    """
    n = len(seq)
    if not int_base:
        int_base = n
    kk = auto_repeat(factor, 1, False, False)
    kk = tuple(
        int(rounding((k if isinstance(k, float) else k / int_base) * (n - 1)))
        for k in kk)
    sorted_items = sorted(seq) if force_sort else seq
    result = operator.itemgetter(*kk)(sorted_items)
    return result


# ======================================================================
def interquantilic_range(
        seq,
        lower_factor=0.25,
        upper_factor=0.75,
        int_base=100,
        force_sort=True,
        quantilic_func=quantile,
        quantilic_kws=None):
    """
    Compute the interquantilic range of a numeric sequence.

    If more than one among median, medoid, quantile, quantiloid is needed,
    it is more efficient to sort the items prior to calling and then
    perform the multiple required calle with `force_sort` set to False.

    Args:
        seq (Sequence[Number]): The input items.
            The values within the sequence may not need to be numeric,
            depending on the `quantilic_func` used.
        lower_factor (int|float): The lower quantilic index.
            If float, must be a number in the [0, 1] range.
            If int, it is divided by `int_base` to get to the [0, 1] range.
        upper_factor (int|float): The upper quantilic index.
            If float, must be a number in the [0, 1] range.
            If int, it is divided by `int_base` to get to the [0, 1] range.
        int_base (int|float): The denominator used to scale integer factors.
        force_sort (bool): Force sorting of the input items.
            If the items are already sorted, this can be safely set to False.
            Otherwise, it must be set to True.
        quantilic_func (callable): The quantilic function.
            Must be either `flyingcircus.quantile()` or
            `flyingcircus.quantiloid()` (or any other callable
            implementing the same signature).
        quantilic_kws (Mapping|None): Keyword parameters for `quantilic_func`.

    Returns:
        result (Number): The inter-quantilic range.

    Examples:
        >>> items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> interquantilic_range(items)
        5.0

    See Also:
        - flyingcircus.quantile()
        - flyingcircus.quantiloid()
        - flyingcircus.sym_interquantilic_range()
        - flyingcircus.median_abs_dev()
    """
    quantilic_kws = dict(quantilic_kws) if quantilic_kws is not None else {}
    q_a, q_b = quantilic_func(
        seq, (lower_factor, upper_factor), int_base, force_sort=force_sort,
        **quantilic_kws)
    return q_b - q_a


# ======================================================================
def sym_interquantilic_range(
        seq,
        sym_factor=0.25,
        force_sort=True,
        quantilic_func=quantile,
        quantilic_kws=None):
    """
    Compute the symmetric interquantilic range of a numeric sequence.

    The quantilic range is defined as: [0.5 - sym_factor, 0.5 + sym_factor].

    If more than one among median, medoid, quantile, quantiloid is needed,
    it is more efficient to sort the items prior to calling and then
    perform the multiple required calle with `force_sort` set to False.

    Args:
        seq (Sequence[Number]): The input items.
        sym_factor (float): The symmetric quantilic factor.
            Must be a number in the [0, 0.5] range.
        force_sort (bool): Force sorting of the input items.
            If the items are already sorted, this can be safely set to False.
            Otherwise, it must be set to True.
        quantilic_func (callable): The quantilic function.
            Must be either `flyingcircus.quantile()` or
            `flyingcircus.quantiloid()` (or any other callable
            implementing the same signature).
        quantilic_kws (Mapping|None): Keyword parameters for `quantilic_func`.

    Returns:
        result (Number): The inter-quantilic range.

    Examples:
        >>> items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> sym_interquantilic_range(items)
        5.0

    See Also:
        - flyingcircus.quantile()
        - flyingcircus.quantiloid()
        - flyingcircus.interquantilic_range()
        - flyingcircus.median_abs_dev()
    """
    return interquantilic_range(
        seq, 0.5 - sym_factor, 0.5 + sym_factor, force_sort=force_sort,
        quantilic_func=quantilic_func, quantilic_kws=quantilic_kws)


# ======================================================================
def next_amean(
        value,
        mean_,
        num):
    """
    Compute the arithmetic mean for (num + 1) items.

    This is useful for low memory footprint computation.

    Args:
        value (Number): The next value to consider.
        mean_ (Number): The aggregate mean of the previous n items.
        num (int): The number of items in the aggregate.

    Returns:
        result (tuple): The tuple
            contains:
             - mean_ (Number): The updated arithmetic mean.
             - num_ (int): The number of items included.

    Examples:
        >>> items = range(0, 52, 2)
        >>> mean_ = 0.0
        >>> for i, val in enumerate(items):
        ...     mean_, num_ = next_amean(val, mean_, i)
        >>> print((mean_, num_))
        (25.0, 26)
        >>> mean_ == mean(items)
        True
        
    See Also:
        - flyingcircus.mean()
        - flyingcircus.i_amean()
        - flyingcircus.i_mean()
        - flyingcircus.i_mean_and_sosd()
        - flyingcircus.next_gmean()
        - flyingcircus.next_hmean()
        - flyingcircus.next_mean_and_var()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.next_median()
        - flyingcircus.next_medoid()
        - flyingcircus.next_medoid_and_median()
    """
    return (num * mean_ + value) / (num + 1), num + 1


# ======================================================================
def next_gmean(
        value,
        gmean_,
        num,
        valid=True):
    """
    Compute the geometric mean for (num + 1) items.

    This is useful for low memory footprint computation.

    Args:
        value (Number): The next value to consider.
        gmean_ (Number): The aggregate geometric mean of the previous n items.
        num (int): The number of items in the aggregate.
        valid (bool): Include only valid items.
            Valid items must be non-zero.

    Returns:
        result (tuple): The tuple
            contains:
             - gmean_ (Number): The updated geometric mean.
             - num_ (int): The number of items included.

    Examples:
        >>> items = range(0, 52, 2)
        >>> gmean_, num_ = 1.0, 0
        >>> for i in items:
        ...     gmean_, num_ = next_gmean(i, gmean_, num_)
        >>> print((round(gmean_, 3), num_))
        (20.354, 25)
        >>> gmean_ == gmean(items)
        True

    See Also:
        - flyingcircus.mean()
        - flyingcircus.gmean()
        - flyingcircus.i_gmean()
        - flyingcircus.next_amean()
        - flyingcircus.next_hmean()
        - flyingcircus.next_mean_and_var()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.next_median()
        - flyingcircus.next_medoid()
        - flyingcircus.next_medoid_and_median()
    """
    if value or not valid:
        # : alternate (slower) implementation
        # return gmean_ * value * max(1, num) / (num + 1), num + 1
        if not num:
            return value, num + 1
        else:
            gmean_ = gmean_ ** (num / (num + 1)) * value ** (1 / (num + 1))
            return gmean_, num + 1
    else:
        return gmean_, num


# ======================================================================
def next_hmean(
        value,
        hmean_,
        num,
        valid=True):
    """
    Compute the harmonic mean for (num + 1) items.

    This is useful for low memory footprint computation.

    Args:
        value (Number): The next value to consider.
        hmean_ (Number): The aggregate harmonic mean of the previous n items.
        num (int): The number of items in the aggregate.
        valid (bool): Include only valid (non-zero) items.

    Returns:
        result (tuple): The tuple
            contains:
             - hmean_ (Number): The updated geometric mean.
             - num_ (int): The number of items included.

    Examples:
        >>> items = range(0, 52, 2)
        >>> hmean_, num_ = 0.0, 0
        >>> for i in items:
        ...     hmean_, num_ = next_hmean(i, hmean_, num_)
        >>> print((round(hmean_, 3), num_))
        (13.103, 25)
        >>> abs(hmean_ - hmean(items)) < 1e-14
        True

    See Also:
        - flyingcircus.mean()
        - flyingcircus.hmean()
        - flyingcircus.i_hmean()
        - flyingcircus.next_amean()
        - flyingcircus.next_gmean()
        - flyingcircus.next_mean()
        - flyingcircus.next_mean_and_var()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.next_median()
        - flyingcircus.next_medoid()
        - flyingcircus.next_medoid_and_median()
    """
    if value or not valid:
        if not num:
            return value, num + 1
        else:
            hmean_ = (num + 1) / (num / hmean_ + 1 / value)
            return hmean_, num + 1
    else:
        return hmean_, num


# ======================================================================
def next_mean(
        value,
        mean_,
        num):
    """
    Compute the arithmetic mean for (num + 1) items.

    This is useful for low memory footprint computation.

    The difference between `flyingcircus.next_amean()` and
    `flyingcircus.next_mean()` is that the latter does return the number
    of items processed.

    Args:
        value (Number): The next value to consider.
        mean_ (Number): The aggregate mean of the previous n items.
        num (int): The number of items in the aggregate.

    Returns:
        result (tuple): The tuple
            contains:
             - mean_ (Number): The updated arithmetic mean.
             - num_ (int): The number of items included.

    Examples:
        >>> items = range(0, 52, 2)
        >>> mean_ = 0.0
        >>> for i, val in enumerate(items):
        ...     mean_, num_ = next_amean(val, mean_, i)
        >>> print((mean_, num_))
        (25.0, 26)
        >>> mean_ == mean(items)
        True

    See Also:
        - flyingcircus.mean()
        - flyingcircus.i_amean()
        - flyingcircus.i_mean()
        - flyingcircus.i_mean_and_sosd()
        - flyingcircus.next_amean()
        - flyingcircus.next_gmean()
        - flyingcircus.next_hmean()
        - flyingcircus.next_mean_and_var()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.next_median()
        - flyingcircus.next_medoid()
        - flyingcircus.next_medoid_and_median()
    """
    return (num * mean_ + value) / (num + 1)


# ======================================================================
def next_mean_and_var(
        value,
        mean_,
        var_,
        num):
    """
    Compute the mean and variance for (num + 1) numeric items.

    This is useful for low memory footprint computation.

    Note that both mean and variance MUST be updated at each iteration,
    therefore a stand-alone `next_var()` is sub-optimal.

    Warning! This algorithm is not numerically stable.
    For better numerical stability use `next_mean_and_sosd()`.

    Args:
        value (Number): The next value to consider.
        mean_ (Number): The aggregate mean of the previous n items.
        var_ (Number): The aggregate variance of the previous n items.
        num (int): The number of items in the aggregate.

    Returns:
        result (tuple): The tuple
            contains:
             - mean (Number): The updated mean.
             - var (Number): The updated variance.
             - num_ (int): The number of items included.

    Examples:
        >>> items = range(0, 52, 2)
        >>> mean_, var_, num_ = 0.0, 0.0, 0
        >>> for i in items:
        ...     mean_, var_, num_ = next_mean_and_var(i, mean_, var_, num_)
        >>> print((mean_, var_))
        (25.0, 225.0)
        >>> (mean_, var_) == mean_and_var(items)
        True

    See Also:
        - flyingcircus.mean()
        - flyingcircus.var()
        - flyingcircus.i_amean()
        - flyingcircus.i_var()
        - flyingcircus.i_mean_and_var()
        - flyingcircus.next_amean()
        - flyingcircus.next_gmean()
        - flyingcircus.next_hmean()
        - flyingcircus.next_mean()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.next_median()
        - flyingcircus.next_medoid()
        - flyingcircus.next_medoid_and_median()
    """
    last = mean_
    mean_, num_ = next_amean(value, mean_, num)
    var_ = ((var_ * num) + (value - last) * (value - mean_)) / num_
    return mean_, var_, num_


# ======================================================================
def next_mean_and_sosd(
        value,
        mean_,
        sosd_,
        num=0):
    """
    Compute the mean and sum-of-squared-deviations for (num + 1) numeric items.

    This is useful for low memory footprint computation.
    This algorithm is numerically stable.

    The sum-of-squared-deviations (SoSD) can be computed as below and
    it is equivalent to the variance multiplied by the number of items:

    sosd = sum((x_i - mu) ** 2)
    sosd = var * n

    Note that both mean and variance MUST be updated at each iteration,
    therefore a stand-alone `next_sosd()` is sub-optimal.

    Args:
        value (Number): The next value to consider.
        mean_ (Number): The aggregate mean of the previous n items.
        sosd_ (Number): The aggregate SoSD of the previous n items.
        num (int): The number of items in the aggregate value.

    Returns:
        result (tuple): The tuple
            contains:
             - mean_ (Number): The updated mean.
             - sosd_ (Number): The updated sum-of-squared-deviations.
             - num_ (int): The number of items included.

    Examples:
        >>> items = range(0, 52, 2)
        >>> mean_, sosd_, num_ = 0.0, 0.0, 0
        >>> for i in items:
        ...     mean_, sosd_, num_ = next_mean_and_sosd(i, mean_, sosd_, num_)
        >>> print((mean_, sosd_, num_))
        (25.0, 5850.0, 26)
        >>> (mean_, sosd_) == mean_and_sosd(items)
        True

    References:
         - Welford, B.P. (1962). "Note on a method for calculating corrected
           sums of squares and products". Technometrics 4(3):419–420.
           doi:10.2307/1266577

    See Also:
        - flyingcircus.squared_deviations()
        - flyingcircus.mean()
        - flyingcircus.sosd()
        - flyingcircus.mean_and_sosd()
        - flyingcircus.mean_and_var()
        - flyingcircus.mean_and_stdev()
        - flyingcircus.i_mean_and_sosd()
        - flyingcircus.next_amean()
        - flyingcircus.next_gmean()
        - flyingcircus.next_hmean()
        - flyingcircus.next_mean()
        - flyingcircus.next_mean_and_var()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.next_median()
        - flyingcircus.next_medoid()
        - flyingcircus.next_medoid_and_median()
    """
    last = mean_
    mean_, num = next_amean(value, mean_, num)
    sosd_ += (value - last) * (value - mean_)
    return mean_, sosd_, num


# ======================================================================
def next_median(
        value,
        buffer,
        max_buffer=2 ** 10):
    """
    Compute the (approximate) median for (num + 1) items.

    This computes an approximate value.
    Relies on the robustness of the median.
    Requires the values to be shuffled (within at least half the buffer size),
    or that the buffer size is larger than the number of values.

    This is useful for low memory footprint approximate computation.

    Args:
        value (Number): The next value to consider.
        buffer (list): The buffer from previous iterations.
        max_buffer (int): The maximum size of the buffer.
            If `max_buffer >= len(items)` the result is exact.

    Returns:
        median_ (Number): The updated median.

    Examples:
        >>> items = range(0, 52, 2)
        >>> buffer = []
        >>> for i in items:
        ...     median_ = next_median(i, buffer)
        >>> print(median_)
        25.0

        >>> buffer = []
        >>> for i in items:
        ...     median_ = next_median(i, buffer, max_buffer=4)
        >>> print(median_)
        46

        >>> items = list(items)
        >>> random.seed(0)
        >>> items = shuffle(items)
        >>> buffer = []
        >>> for i in items:
        ...     median_ = next_median(i, buffer, max_buffer=4)
        >>> print(median_)
        12

    See Also:
        - flyingcircus.median()
        - flyingcircus.i_median()
        - flyingcircus.next_amean()
        - flyingcircus.next_gmean()
        - flyingcircus.next_hmean()
        - flyingcircus.next_mean()
        - flyingcircus.next_mean_and_var()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.next_medoid()
        - flyingcircus.next_medoid_and_median()
    """
    if not buffer:
        median_ = value
        buffer.append(value)
    else:
        # equivalent to: `buffer.append(value); buffer.sort()` but faster
        bisect.insort(buffer, value)
        median_ = median(buffer, False)
        if len(buffer) > max_buffer:
            if value > median_:
                del buffer[0]
            else:
                del buffer[-1]
    return median_


# ======================================================================
def next_medoid(
        value,
        buffer,
        lower=True,
        max_buffer=2 ** 10 + 1):
    """
    Compute the (approximate) medoid for (num + 1) items.

    This computes an approximate value.
    Relies on the robustness of the medoid.
    Requires the values to be shuffled (within at least half the buffer size),
    or that the buffer size is larger than the number of values.

    This is useful for low memory footprint approximate computation.
    If the number of items is not odd, returns the medoid from the lower or
    upper half depending on the value of `lower` being True or False.

    Args:
        value (Any): The next value to consider.
        buffer (list): The buffer from previous iterations.
        lower (bool): Pick the lower medoid for even-sized buffer.
        max_buffer (int): The maximum size of the buffer.
            If `max_buffer >= len(items)` the result is exact.

    Returns:
        result (Any): The updated approximate medoid value.

    Examples:
        >>> items = range(0, 52, 2)
        >>> buffer = []
        >>> for i in items:
        ...     medoid_ = next_medoid(i, buffer)
        >>> print(medoid_)
        24

        >>> buffer = []
        >>> for i in items:
        ...     medoid_ = next_medoid(i, buffer, max_buffer=4)
        >>> print(medoid_)
        46

        >>> items = list(items)
        >>> random.seed(0)
        >>> items = shuffle(items)
        >>> buffer = []
        >>> for i in items:
        ...     medoid_ = next_medoid(i, buffer, max_buffer=4)
        >>> print(medoid_)
        12

    See Also:
        - flyingcircus.medoid()
        - flyingcircus.i_medoid()
        - flyingcircus.i_medoid_and_median()
        - flyingcircus.next_amean()
        - flyingcircus.next_gmean()
        - flyingcircus.next_hmean()
        - flyingcircus.next_mean()
        - flyingcircus.next_mean_and_var()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.next_median()
        - flyingcircus.next_medoid()
        - flyingcircus.next_medoid_and_median()
    """
    if not buffer:
        medoid_ = value
        buffer.append(value)
    else:
        # equivalent to: `buffer.append(value); buffer.sort()` but faster
        bisect.insort(buffer, value)
        medoid_ = medoid(buffer, False, lower)
        if len(buffer) > max_buffer:
            if value > medoid_:
                del buffer[0]
            else:
                del buffer[-1]
    return medoid_


# ======================================================================
def next_medoid_and_median(
        value,
        buffer,
        lower=True,
        max_buffer=2 ** 10 + 1):
    """
    Compute the (approximate) medoid and median for (num + 1) items.

    This computes an approximate value.
    Relies on the robustness of the medoid and the median.
    Requires the values to be shuffled (within at least half the buffer size),
    or that the buffer size is larger than the number of values.

    This is used when both the medoid and the median are needed. In this case,
    a single buffer could be used. Note that calling both `next_medoid()` and
    `next_median()` consecutively over the same buffer MUST be avoided,
    because this introduces a bias, since the elements are inserted twice in
    buffer.

    This is useful for low memory footprint approximate computation.
    If the number of items is not odd, returns the medoid from the lower or
    upper half depending on the value of `lower` being True or False.

    Args:
        value (Any): The next value to consider.
        buffer (list): The buffer from previous iterations.
        lower (bool): Pick the lower medoid for even-sized buffer.
        max_buffer (int): The maximum size of the buffer.
            If `max_buffer >= len(items)` the result is exact.

    Returns:
        result (tuple): The tuple
            contains:
             - `medoid_` (Any): The updated approximate median value.
             - `median_` (Number): The updated medoid value.

    Examples:
        >>> items = range(0, 52, 2)
        >>> buffer = []
        >>> for i in items:
        ...     medoid_, median_ = next_medoid_and_median(i, buffer)
        >>> print((medoid_, median_))
        (24, 25.0)

        >>> buffer = []
        >>> for i in items:
        ...     medoid_, median_ = next_medoid_and_median(
        ...         i, buffer, max_buffer=4)
        >>> print((medoid_, median_))
        (46, 46)

        >>> items = list(items)
        >>> random.seed(0)
        >>> items = shuffle(items)
        >>> buffer = []
        >>> for i in items:
        ...     medoid_, median_ = next_medoid_and_median(
        ...         i, buffer, max_buffer=4)
        >>> print((medoid_, median_))
        (12, 12)
        >>> buffer = []
        >>> for i in items:
        ...     medoid_, median_ = next_medoid_and_median(
        ...         i, buffer, max_buffer=12)
        >>> print((medoid_, median_))
        (24, 24)

    See Also:
        - flyingcircus.median()
        - flyingcircus.i_median()
        - flyingcircus.medoid()
        - flyingcircus.i_medoid()
        - flyingcircus.i_medoid_and_median()
        - flyingcircus.next_amean()
        - flyingcircus.next_gmean()
        - flyingcircus.next_hmean()
        - flyingcircus.next_mean()
        - flyingcircus.next_mean_and_var()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.next_median()
        - flyingcircus.next_medoid()
    """
    if not buffer:
        medoid_ = value
        median_ = value
        buffer.append(value)
    else:
        # equivalent to: `buffer.append(value); buffer.sort()` but faster
        bisect.insort(buffer, value)
        medoid_ = medoid(buffer, False, lower)
        median_ = median(buffer, False)
        if len(buffer) > max_buffer:
            if value > medoid_:
                del buffer[0]
            else:
                del buffer[-1]
    return medoid_, median_


# ======================================================================
def i_amean(
        items,
        mean_=0.0,
        num=0):
    """
    Compute the arithmetic mean a numeric iterable.

    Internally uses `flyingcircus.next_amean()`.

    This is substantially faster than `statistics.mean()`.

    The difference between `flyingcircus.i_amean()` and
    `flyingcircus.i_mean()` is that the latter does return the number
    of items processed.

    Args:
        items (Iterable[Number]): The input items.
        mean_ (Number): The start mean value.
        num (int): The number of items included in the start value.

    Returns:
        result (tuple): The tuple
            contains:
             - `mean_` (: The mean of the items.
             - `num`: The number of items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> i_amean(items)
        (25.0, 26)
        >>> (mean(items), len(items)) == i_amean(items)
        True
        
    See Also:
        - flyingcircus.mean()
        - flyingcircus.next_amean()
        - flyingcircus.i_amean()
        - flyingcircus.i_gmean()
        - flyingcircus.i_hmean()
        - flyingcircus.i_mean_and_sosd()
        - flyingcircus.i_var()
        - flyingcircus.i_stdev()
        - flyingcircus.i_mean_and_var()
        - flyingcircus.i_mean_and_stdev()
    """
    for item in items:
        mean_, num = next_amean(item, mean_, num)
    return mean_, num


# ======================================================================
def i_gmean(
        items,
        gmean_=1.0,
        num=0,
        valid=True):
    """
    Compute the mean a numeric iterable.

    Internally uses `flyingcircus.next_gmean()`.

    This is substantially faster than `statistics.mean()`.

    Args:
        items (Iterable[Number]): The input items.
        gmean_ (Number): The start mean value.
        num (int): The number of items included in the start value.
        valid (bool): Include only valid (non-zero) items.

    Returns:
        result (tuple): The tuple
            contains:
             - `gmean_` (Number): The mean of the items.
             - `num` (int): The number of items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> gmean_, num = i_gmean(items)
        >>> print((round(gmean_, 3), num))
        (20.354, 25)
        >>> (gmean(items), len(items) - 1) == i_gmean(items)
        True
    """
    for item in items:
        gmean_, num = next_gmean(item, gmean_, num, valid)
    return gmean_, num


# ======================================================================
def i_hmean(
        items,
        hmean_=0.0,
        num=0,
        valid=True):
    """
    Compute the mean a numeric iterable.

    Internally uses `flyingcircus.next_hmean()`.

    This is substantially faster than `statistics.mean()`.

    Args:
        items (Iterable[Number]): The input items.
        hmean_ (Number): The start harmonic mean value.
        num (int): The number of items included in the start value.
        valid (bool): Include only valid (non-zero) items.

    Returns:
        result (tuple): The tuple
            contains:
             - `mean_` (Number): The mean of the items.
             - `num` (int): The number of items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> hmean_, num = i_hmean(items)
        >>> print((round(hmean_, 3), num))
        (13.103, 25)
        >>> abs(hmean_ - hmean(items)) < 1e-14
        True
    """
    for i, item in enumerate(items):
        hmean_, num = next_hmean(item, hmean_, num, valid)
    return hmean_, num


# ======================================================================
def i_mean(
        items,
        mean_=0.0,
        num=0):
    """
    Compute the arithmetic mean a numeric iterable.

    Internally uses `flyingcircus.i_amean()`.

    This is substantially faster than `statistics.mean()`.

    The difference between `flyingcircus.i_amean()` and
    `flyingcircus.i_mean()` is that the latter does return the number
    of items processed.

    Args:
        items (Iterable[Number]): The input items.
        mean_ (Number): The start mean value.
        num (int): The number of items included in the start value.

    Returns:
        result (tuple): The tuple
            contains:
             - `mean_` (: The mean of the items.
             - `num`: The number of items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> i_mean(items)
        25.0
        >>> mean(items) == i_mean(items)
        True

    See Also:
        - flyingcircus.mean()
        - flyingcircus.next_amean()
        - flyingcircus.i_amean()
        - flyingcircus.i_gmean()
        - flyingcircus.i_hmean()
        - flyingcircus.i_mean_and_sosd()
        - flyingcircus.i_var()
        - flyingcircus.i_stdev()
        - flyingcircus.i_mean_and_var()
        - flyingcircus.i_mean_and_stdev()
    """
    mean_, num = i_amean(items, mean_, num)
    return mean_


# ======================================================================
def i_mean_and_sosd(
        items,
        mean_=0.0,
        sosd_=0.0,
        num=0):
    """
    Compute the mean and the variance of a numeric iterable.

    Internally uses `flyingcircus.next_mean_and_sosd()`.

    This is useful for low memory footprint computation.

    This is substantially faster than computing the two values separately.

    Args:
        items (Iterable[Number]): The input items.
        mean_ (Number): The start mean value.
        sosd_ (Number): The start sum-of-squared-deviation value.
        num (int): The number of items included in the start mean value.

    Returns:
        result (tuple): The tuple
            contains:
             - `mean_` (Number): The mean of the items.
             - `var_` (Number): The variance of the items.
             - `num` (int): The number of items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> i_mean_and_sosd(items)
        (25.0, 5850.0, 26)
        >>> mean_and_sosd(items) + (len(items),) == i_mean_and_sosd(items)
        True

    See Also:
        - flyingcircus.mean_and_sosd()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.i_amean()
        - flyingcircus.i_gmean()
        - flyingcircus.i_hmean()
        - flyingcircus.i_mean()
        - flyingcircus.i_var()
        - flyingcircus.i_stdev()
        - flyingcircus.i_mean_and_var()
        - flyingcircus.i_mean_and_stdev()
    """
    for i, item in enumerate(items):
        mean_, sosd_, num = next_mean_and_sosd(item, mean_, sosd_, num)
    return mean_, sosd_, num


# ======================================================================
def i_var(
        items,
        mean_=0.0,
        var_=0.0,
        num=0,
        ddof=0):
    """
    Compute the variance of a numeric iterable.

    Internally uses `flyingcircus.next_mean_and_sosd()` and
    `flyingcircus.sosd2var()`.

    This is useful for low memory footprint computation.

    Note that both mean and variance MUST be updated at each iteration,
    therefore if both the mean and the variance are required use
    `flyingcircus.i_mean_var()`.

    Args:
        items (Iterable[Number]): The input items.
        mean_ (Number): The start mean value.
        var_ (Number): The start variance value.
        num (int): The number of items included in the start values.
        ddof (int): The number of degrees of freedom.

    Returns:
        result (Number): The variance of the items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> i_var(items)
        225.0
        >>> var(items) == i_var(items)
        True

    See Also:
        - flyingcircus.var()
        - flyingcircus.next_mean_and_var()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.i_amean()
        - flyingcircus.i_gmean()
        - flyingcircus.i_hmean()
        - flyingcircus.i_mean()
        - flyingcircus.i_mean_and_sosd()
        - flyingcircus.i_stdev()
        - flyingcircus.i_mean_and_var()
        - flyingcircus.i_mean_and_stdev()
        - flyingcircus.sosd2var()
        - flyingcircus.var2sosd()
    """
    sosd_ = var2sosd(var_, num, ddof)
    mean_, sosd_, num_ = i_mean_and_sosd(items, mean_, sosd_, num)
    return sosd2var(sosd_, num_, ddof)


# ======================================================================
def i_stdev(
        items,
        mean_=0.0,
        stdev_=0.0,
        num=0,
        ddof=0):
    """
    Compute the standard deviation of a numeric iterable.

    Internally uses `flyingcircus.next_mean_and_sosd()` and
    `flyingcircus.sosd2stdev()`.

    This is useful for low memory footprint computation.

    Note that both mean and std. deviation MUST be updated at each iteration,
    therefore if both the mean and the standard deviation are required use
    `flyingcircus.i_mean_stdev()`.

    Args:
        items (Iterable[Number]): The input items.
        mean_ (Number): The start mean value.
        stdev_ (Number): The start standard deviation value.
        num (int): The number of items included in the start values.
        ddof (int): The number of degrees of freedom.

    Returns:
        result (Number): The standard deviation of the items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> i_stdev(items)
        15.0
        >>> stdev(items) == i_stdev(items)
        True

    See Also:
        - flyingcircus.stdev()
        - flyingcircus.next_mean_and_var()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.i_amean()
        - flyingcircus.i_gmean()
        - flyingcircus.i_hmean()
        - flyingcircus.i_mean()
        - flyingcircus.i_mean_and_sosd()
        - flyingcircus.i_var()
        - flyingcircus.i_mean_and_var()
        - flyingcircus.i_mean_and_stdev()
        - flyingcircus.sosd2stdev()
        - flyingcircus.stdev2sosd()
    """
    sosd_ = stdev2sosd(stdev_, num, ddof)
    mean_, sosd_, num = i_mean_and_sosd(items, mean_, sosd_, num)
    return sosd2stdev(sosd_, num, ddof)


# ======================================================================
def i_mean_and_var(
        items,
        mean_=0.0,
        var_=0.0,
        num=0,
        ddof=0):
    """
    Compute the mean and the variance of a numeric iterable.

    Internally uses `flyingcircus.next_mean_and_sosd()` and
    `flyingcircus.sosd2var()`.

    This is useful for low memory footprint computation.

    This is substantially faster than:
     - computing the two values separately.
     - both `statistics.mean()` and `statistics.variance()`.

    Args:
        items (Iterable[Number]): The input items.
        mean_ (Number): The start mean value.
        var_ (Number): The start variance value.
        num (int): The number of items included in the start values.
        ddof (int): The number of degrees of freedom.

    Returns:
        result (tuple): The tuple
            contains:
             - `mean_` (Number): The mean of the items.
             - `var_` (Number): The variance of the items.
             - `num` (int): The number of items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> i_mean_and_var(items)
        (25.0, 225.0, 26)
        >>> mean_and_var(items) + (len(items),) == i_mean_and_var(items)
        True

    See Also:
        - flyingcircus.mean()
        - flyingcircus.var()
        - flyingcircus.next_mean_and_var()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.i_amean()
        - flyingcircus.i_gmean()
        - flyingcircus.i_hmean()
        - flyingcircus.i_mean()
        - flyingcircus.i_mean_and_sosd()
        - flyingcircus.i_var()
        - flyingcircus.i_stdev()
        - flyingcircus.i_mean_and_stdev()
        - flyingcircus.sosd2var()
        - flyingcircus.var2sosd()
    """
    sosd_ = var2sosd(var_, num, ddof)
    mean_, sosd_, num = i_mean_and_sosd(items, mean_, sosd_, num)
    return mean_, sosd2var(sosd_, num, ddof), num


# ======================================================================
def i_mean_and_stdev(
        items,
        mean_=0.0,
        stdev_=0.0,
        num=0,
        ddof=0):
    """
    Compute the mean and standard deviation of a numeric iterable.

    Internally uses `flyingcircus.next_mean_and_sosd()` and
    `flyingcircus.sosd2stdev()`.

    This is useful for low memory footprint computation.

    This is substantially faster than:
     - computing the two values separately.
     - both `statistics.mean()` and `statistics.stdev()`.

    Args:
        items (Iterable[Number]): The input items.
        mean_ (Number): The start mean value.
        stdev_ (Number): The start standard deviation value.
        num (int): The number of items included in the start values.
        ddof (int): The number of degrees of freedom.

    Returns:
        result (tuple): The tuple
            contains:
             - `mean_` (Number): The mean of the items.
             - `stdev_` (Number): The std. deviation of the items.
             - `num` (int): The number of items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> i_mean_and_stdev(items)
        (25.0, 15.0, 26)
        >>> mean_and_stdev(items) + (len(items),) == i_mean_and_stdev(items)
        True

    See Also:
        - flyingcircus.mean()
        - flyingcircus.var()
        - flyingcircus.next_mean_and_var()
        - flyingcircus.next_mean_and_sosd()
        - flyingcircus.i_amean()
        - flyingcircus.i_gmean()
        - flyingcircus.i_hmean()
        - flyingcircus.i_mean()
        - flyingcircus.i_mean_and_sosd()
        - flyingcircus.i_var()
        - flyingcircus.i_stdev()
        - flyingcircus.i_mean_and_var()
        - flyingcircus.sosd2stdev()
        - flyingcircus.stdev2sosd()
    """
    sosd_ = stdev2sosd(stdev_, num, ddof)
    mean_, sosd_, num = i_mean_and_sosd(items, mean_, sosd_, num)
    return mean_, sosd2stdev(sosd_, num, ddof), num


# ======================================================================
def i_median(
        items,
        max_buffer=2 ** 10):
    """
    Compute the (approximate) median of a numeric iterable.

    This computes an approximate value.
    Relies on the robustness of the median.

    This is useful for low memory footprint approximate computation.

    Args:
        items (Iterable[Number]): The input items.
        max_buffer (int): The maximum size of the buffer.
            If `max_buffer >= len(items)` the result is exact.

    Returns:
        result (Number): The median value.

    Examples:
        >>> items = range(0, 52, 2)
        >>> i_median(items)
        25.0
        >>> i_median(items) == median(items)
        True

        >>> items = [1, 2, 3, 4, 100]
        >>> i_median(items)
        3

        >>> items = [81, 73, 99, 86, 94, 40, 75, 46, 8, 50, 51, 53, 35, 87]
        >>> sorted(items)
        [8, 35, 40, 46, 50, 51, 53, 73, 75, 81, 86, 87, 94, 99]
        >>> i_median(items)
        63.0
        >>> [i_median(items, i) for i in range(3, len(items))]
        [50.5, 51, 52.0, 51, 52.0, 51, 52.0, 51, 52.0, 53, 63.0]

        >>> items = [81, 73, 99, 86, 94, 40, 75, 46, 8, 50, 51, 53, 35]
        >>> sorted(items)
        [8, 35, 40, 46, 50, 51, 53, 73, 75, 81, 86, 94, 99]
        >>> i_median(items)
        53
        >>> [i_median(items, i) for i in range(3, len(items))]
        [50.5, 51, 52.0, 51, 52.0, 51, 52.0, 51, 52.0, 53]

        >>> all([i_median(range(i)) == median(range(i)) for i in range(1, 40)])
        True
        >>> all([i_median(range(0, i)) == i_medoid(range(0, i))
        ...     for i in range(1, 40, 2)])
        True

    See Also:
        - flyingcircus.median()
        - flyingcircus.median_and_median_abs_dev()
        - flyingcircus.next_median()
        - flyingcircus.next_medoid_and_median()
        - flyingcircus.i_median()
        - flyingcircus.i_medoid()
        - flyingcircus.i_medoid_and_median()
        - flyingcircus.i_median_and_median_abs_dev()
    """
    iter_items = iter(items)
    median_ = next(iter_items)
    buffer = [median_]
    for item in iter_items:
        median_ = next_median(item, buffer=buffer, max_buffer=max_buffer)
    return median_


# ======================================================================
def i_medoid(
        items,
        lower=True,
        max_buffer=2 ** 10 + 1):
    """
    Compute the (approximate) medoid of an arbitrary iterable.

    This computes an approximate value.
    Relies on the robustness of the medoid.

    This is useful for low memory footprint approximate computation.

    Args:
        items (Iterable[Any]): The input items.
        lower (bool): Pick the lower medoid for even-sized items.
        max_buffer (int): The maximum size of the buffer.
            If `max_buffer >= len(items)` the result is exact.
            If `max_buffer` is 0, use all items.

    Returns:
        result (Any): The medoid value.

    Examples:
        >>> items = range(0, 52, 2)
        >>> i_medoid(items)
        24
        >>> i_medoid(items) == medoid(items)
        True

        >>> items = [1, 2, 2, 2, 100]
        >>> i_medoid(items)
        2

        >>> items = [81, 73, 99, 86, 94, 40, 75, 46, 8, 50, 51, 53, 35, 87]
        >>> sorted(items)
        [8, 35, 40, 46, 50, 51, 53, 73, 75, 81, 86, 87, 94, 99]
        >>> i_medoid(items)
        53
        >>> [i_medoid(items, max_buffer=i) for i in range(3, len(items))]
        [51, 51, 51, 51, 51, 51, 51, 51, 53, 53, 53]

        >>> items = [81, 73, 99, 86, 94, 40, 75, 46, 8, 50, 53, 35, 87]
        >>> sorted(items)
        [8, 35, 40, 46, 50, 53, 73, 75, 81, 86, 87, 94, 99]
        >>> i_medoid(items)
        73
        >>> [i_medoid(items, max_buffer=i) for i in range(3, len(items))]
        [50, 50, 50, 50, 50, 50, 50, 53, 73, 73]

        >>> all([i_medoid(range(i)) == medoid(range(i)) for i in range(1, 40)])
        True
        >>> all([i_medoid(range(0, i)) == i_median(range(0, i))
        ...     for i in range(1, 40, 2)])
        True

    See Also:
        - flyingcircus.medoid()
        - flyingcircus.next_medoid()
        - flyingcircus.next_medoid_and_median()
        - flyingcircus.i_median()
        - flyingcircus.i_medoid()
        - flyingcircus.i_medoid_and_median()
    """
    if max_buffer:
        iter_items = iter(items)
        medoid_ = next(iter_items)
        buffer = [medoid_]
        for item in iter_items:
            medoid_ = next_medoid(item, buffer, lower, max_buffer)
    else:
        medoid_ = medoid(tuple(items), True, False)
    return medoid_


# ======================================================================
def i_medoid_and_median(
        items,
        lower=True,
        max_buffer=2 ** 10 + 1):
    """
    Compute the (approximate) medoid and median for a numeric iterable.

    This computes an approximate value.
    Relies on the robustness of the medoid.

    This is useful for low memory footprint approximate computation.

    This is faster and more memory efficient than computing both
    `flyingcircus.i_medoid()` and `flyingcircus.i_median()`
    separately.

    Args:
        items (Iterable[Number]): The input items.
        lower (bool): Pick the lower medoid for even-sized items.
        max_buffer (int): The maximum size of the buffer.
            If `max_buffer >= len(items)` the result is exact.
            If `max_buffer` is 0, use all items.

    Returns:
        result (tuple): The tuple
            contains:
             - `medoid_`: The medoid of the items.
             - `median_`: The median of the items.
             - `num`: The number of items.

    Examples:
        >>> items = range(0, 52, 2)
        >>> i_medoid_and_median(items)
        (24, 25.0, 26)
        >>> (i_medoid_and_median(items)
        ...     == (medoid(items), median(items), len(items)))
        True

        >>> items = [1, 2, 2, 2, 100]
        >>> i_medoid_and_median(items)
        (2, 2, 5)

        >>> items = [81, 73, 99, 86, 94, 40, 75, 46, 8, 50, 51, 53, 35, 87]
        >>> sorted(items)
        [8, 35, 40, 46, 50, 51, 53, 73, 75, 81, 86, 87, 94, 99]
        >>> i_medoid_and_median(items)
        (53, 63.0, 14)
        >>> [i_medoid_and_median(items, max_buffer=i) for i in range(10, 14)]
        [(51, 51, 14), (53, 52.0, 14), (53, 53, 14), (53, 63.0, 14)]

        >>> items = [81, 73, 99, 86, 94, 40, 75, 46, 8, 50, 53, 35, 87]
        >>> sorted(items)
        [8, 35, 40, 46, 50, 53, 73, 75, 81, 86, 87, 94, 99]
        >>> i_medoid_and_median(items)
        (73, 73, 13)
        >>> [i_medoid_and_median(items, max_buffer=i) for i in range(10, 14)]
        [(53, 53, 13), (73, 63.0, 13), (73, 73, 13), (73, 73, 13)]

    See Also:
        - flyingcircus.median()
        - flyingcircus.medoid()
        - flyingcircus.next_median()
        - flyingcircus.next_medoid()
        - flyingcircus.next_medoid_and_median()
        - flyingcircus.i_median()
        - flyingcircus.i_medoid()
        - flyingcircus.i_medoid_and_median()
    """
    if max_buffer:
        i = 0
        iter_items = iter(items)
        medoid_ = next(iter_items)
        median_ = medoid_
        buffer = [medoid_]
        for i, item in enumerate(iter_items):
            medoid_, median_ = next_medoid_and_median(
                item, buffer, lower, max_buffer)
        num = i + 2  # 1 from from `for`, 1 from `next(iter_items)`
    else:
        items = sorted(items)
        medoid_ = medoid(items, force_sort=False, lower=lower)
        median_ = median(items, force_sort=False)
        num = len(items)
    return medoid_, median_, num


# ======================================================================
def align(
        num,
        base,
        mode='closest'):
    """
    Align (round) a number to a multiple of the specified base.

    The resulting number is computed using the formula:

    num = num + func(num % base / base) * base - num % base

    where `func` is a rounding function, as determined by `mode`.

    Args:
        num (int|float): The input number.
        base (int|float|str|None): The number to align to.
            If int, then calculate a multiple of `align` close to `num`.
            If str, possible options are:
             - 'powX' (where X >= 2 must be an int): calculate a power of X
               that is close to `num`.
            The exact number being calculated depends on the value of `mode`.
        mode (int|str|bool): Determine the rounding mode.
            If str, valid inputs are:
             - 'upper', '+': round to smallest multiple larger than `num`.
             - 'lower', '-': round to the largest multiple smaller than `num`.
             - 'closest', '~': round to the multiple closest to `num`.
            If int, valid inputs are `+1`, `0` and `-1`, mapping to 'upper',
            'closest' and 'lower' respectively.

    Returns:
        num (int): The aligned number.

    Examples:
        >>> align(9, 2)
        8
        >>> align(447, 32, mode=1)
        448
        >>> align(447, 32, mode=-1)
        416
        >>> align(447, 32, mode=0)
        448
        >>> align(432, 'pow2', mode=1)
        512
        >>> align(432, 'pow2', mode=-1)
        256
        >>> align(432, 'pow2', mode=0)
        512
        >>> align(45, 90, mode=0)
        0
        >>> align(6, 'pow2', mode=0)
        8
        >>> align(6543, 'pow10', mode=0)
        10000
        >>> align(1543, 'pow10', mode=0)
        1000
        >>> align(128, 128, mode=1)
        128
        >>> align(123.37, 0.5, mode=1)
        123.5
        >>> align(123.37, 0.5, mode=0)
        123.5
        >>> align(123.37, 0.5, mode=-1)
        123.0
        >>> align(123.37, None)
        123.37
    """
    reversed_modes = {
        math.ceil: ['upper', 1],
        math.floor: ['lower', -1],
        round: ['closest', 0]}
    modes = reverse_mapping_iter(reversed_modes)
    if mode in modes:
        approx = modes[mode][0]
    else:
        raise ValueError('Invalid mode `{mode}`'.format(mode=mode))

    if base:
        if isinstance(base, str):
            if base.startswith('pow'):
                base = int(base[len('pow'):])
                exp = math.log(num, base)
                num = int(base ** int(approx(exp)))
            else:
                raise ValueError('Invalid align `{align}`'.format(align=base))

        elif isinstance(base, (int, float)):
            modulus = num % base
            num += approx(modulus / base) * base - modulus

        else:
            warnings.warn('Will not align `{num}` to `{align}`.'.format(
                num=num, align=base))

    return num


# =====================================================================
def p_ratio(x, y):
    """
    Compute the pseudo-ratio of x, y: 1 / ((x / y) + (y / x))

    .. math::
        \\frac{1}{\\frac{x}{y}+\\frac{y}{x}} = \\frac{xy}{x^2+y^2}

    Args:
        x (int|float): First input value.
        y (int|float): Second input value.

    Returns:
        result (float): 1 / ((x / y) + (y / x))

    Examples:
        >>> p_ratio(2, 2)
        0.5
        >>> p_ratio(200, 200)
        0.5
        >>> p_ratio(1, 2)
        0.4
        >>> p_ratio(100, 200)
        0.4
        >>> items = 100, 200
        >>> (p_ratio(*items) == p_ratio(*items[::-1]))
        True
    """
    return (x * y) / (x ** 2 + y ** 2)


# =====================================================================
def gen_p_ratio(values):
    """
    Compute the generalized pseudo-ratio of x_i: 1 / sum_ij [ x_i / x_j ]

    .. math::
        \\frac{1}{\\sum_{ij} \\frac{x_i}{x_j}}

    Args:
        values (Iterable[int|float]): Input values.

    Returns:
        result (float): 1 / sum_ij [ x_i / x_j ]

    Examples:
        >>> gen_p_ratio((2, 2, 2, 2, 2))
        0.05
        >>> gen_p_ratio((200, 200, 200, 200, 200))
        0.05
        >>> gen_p_ratio((1, 2))
        0.4
        >>> gen_p_ratio((100, 200))
        0.4
        >>> values1 = [x * 10 for x in range(2, 10)]
        >>> values2 = [x * 1000 for x in range(2, 10)]
        >>> gen_p_ratio(values1) - gen_p_ratio(values2) < 1e-10
        True
        >>> items = list(range(2, 10))
        >>> gen_p_ratio(values2) - gen_p_ratio(items[::-1]) < 1e-10
        True
    """
    return 1 / sum(x / y for x, y in itertools.permutations(values, 2))


# ======================================================================
def multi_split(
        text,
        seps,
        sep=None,
        filter_empty=True):
    """
    Split a string using multiple separators.

    This is achieved by replacing all (secondary) separators with a base
    separator, and finally split using the base separator.

    Args:
        text (str|bytes|bytearray): The input string.
        seps (Iterable[str|bytes|bytearray]): The additional separators.
        sep (str|bytes||bytearrayNone): The separator for the final splitting.
            If None, uses `.split(None)` default behavior,
            i.e. splits blank character (' ', '\t', '\n') simultaneously.
        filter_empty (bool): Remove empty string from the results.

    Returns:
        result (list[str|bytes|bytearray]): The split strings.

    Examples:
        >>> text = 'no-go to go, where? anti-matter'
        >>> print(multi_split(text, ('-', ',')))
        ['no', 'go', 'to', 'go', 'where?', 'anti', 'matter']
        >>> print(multi_split(text, ('-', ','), '?'))
        ['no', 'go to go', ' where', ' anti', 'matter']
        >>> print(multi_split(text, ('-', ',', ' '), '?'))
        ['no', 'go', 'to', 'go', 'where', 'anti', 'matter']
        >>> print(multi_split(text, ('-', ',', ' '), '?', False))
        ['no', 'go', 'to', 'go', '', 'where', '', 'anti', 'matter']

        >>> text = b'no-go to go, where? anti-matter'
        >>> print(multi_split(text, (b'-', b',')))
        [b'no', b'go', b'to', b'go', b'where?', b'anti', b'matter']
        >>> print(multi_split(text, (b'-', b','), b'?'))
        [b'no', b'go to go', b' where', b' anti', b'matter']
        >>> print(multi_split(text, (b'-', b',', b' '), b'?'))
        [b'no', b'go', b'to', b'go', b'where', b'anti', b'matter']
        >>> print(multi_split(text, (b'-', b',', b' '), b'?', False))
        [b'no', b'go', b'to', b'go', b'', b'where', b'', b'anti', b'matter']
    """
    _sep = (' ' if isinstance(text, str) else b' ') if sep is None else sep
    # text = functools.reduce(lambda t, s: t.replace(s, _sep), seps, text)
    for sep_ in seps:
        text = text.replace(sep_, _sep)
    if sep is None:
        return text.split()
    elif filter_empty:
        return list(filter(bool, text.split(sep)))
    else:
        return text.split(sep)


# ======================================================================
def multi_replace(
        text,
        replaces):
    """
    Perform multiple replacements in a string.

    The replaces are concatenated together, therefore the order may matter.

    Args:
        text (str|bytes|bytearray): The input string.
        replaces (tuple[tuple[str|bytes|bytearray]]): The replacements.
            Format: ((<old>, <new>), ...).

    Returns:
        text (str|bytes): The string after the performed replacements.

    Examples:
        >>> multi_replace('X Y', (('X', 'flying'), ('Y', 'circus')))
        'flying circus'
        >>> multi_replace('x-x-x-x', (('x', 'est'), ('est', 'test')))
        'test-test-test-test'
        >>> multi_replace('x-x-', (('-x-', '.test'),))
        'x.test'
        >>> multi_replace('x-x-', (('x', 'test'), ('te', 'be')))
        'best-best-'
        >>> multi_replace('x-x-', (('te', 'be'), ('x-', 'test-')))
        'test-test-'

        >>> multi_replace(b'X Y', ((b'X', b'flying'), (b'Y', b'circus')))
        b'flying circus'
        >>> multi_replace(b'x-x-x-x', ((b'x', b'est'), (b'est', b'test')))
        b'test-test-test-test'
        >>> multi_replace(b'x-x-', ((b'-x-', b'.test'),))
        b'x.test'
        >>> multi_replace(b'x-x-', ((b'x', b'test'), (b'te', b'be')))
        b'best-best-'
        >>> multi_replace(b'x-x-', ((b'te', b'be'), (b'x-', b'test-')))
        b'test-test-'
    """
    # return functools.reduce(lambda t, r: t.replace(*r), replaces, text)
    for replace in replaces:
        text = text.replace(*replace)
    return text


# ======================================================================
def multi_replace_char(
        text,
        replaces):
    """
    Perform multiple replacements of single characters in a string.

    The replaces are concatenated together, therefore the order may matter.

    Args:
        text (str): The input string.
        replaces (tuple[tuple[str]]|dict): The listing of the replacements.
            Format: ((<old>, <new>), ...) where <old> must be a single char.

    Returns:
        text (str): The string after the performed replacements.

    Examples:
        >>> multi_replace_char('X Y', (('X', 'flying'), ('Y', 'circus')))
        'flying circus'
        >>> multi_replace_char('X Y', (('Y', 'circus'), ('X', 'flying')))
        'flying circus'
        >>> multi_replace_char('x-y-x-y', (('x', 'flying'), ('y', 'circus')))
        'flying-circus-flying-circus'
        >>> multi_replace_char('x-y-x-y', (('x', 'flying'), ('g', 'circus')))
        'flying-y-flying-y'
        >>> multi_replace_char('x-g-x-g', (('x', 'flying'), ('g', 'circus')))
        'flying-circus-flying-circus'
        >>> multi_replace_char('x-x-', (('x-', 'bye'),))
        'x-x-'
    """
    replaces = dict(replaces)
    return ''.join(replaces.get(k, k) for k in text)


# ======================================================================
def common_subseq_2(
        seq1,
        seq2,
        sorting=None):
    """
    Find the longest common consecutive subsequence(s).
    This version works for two Sequences.

    This is known as the `longest common substring` problem, or LCS for short.

    Args:
        seq1 (Sequence): The first input sequence.
            Must be of the same type as seq2.
        seq2 (Sequence): The second input sequence.
            Must be of the same type as seq1.
        sorting (callable): Sorting function passed to 'sorted' via `key` arg.

    Returns:
        commons (list[Sequence]): The longest common subsequence(s).

    Examples:
        >>> common_subseq_2('academy', 'abracadabra')
        ['acad']
        >>> common_subseq_2('los angeles', 'lossless')
        ['los', 'les']
        >>> common_subseq_2('los angeles', 'lossless', lambda x: x)
        ['les', 'los']
        >>> common_subseq_2((1, 2, 3, 4, 5), (0, 1, 2))
        [(1, 2)]
    """
    # note: [[0] * (len(seq2) + 1)] * (len(seq1) + 1) will not work!
    counter = [[0 for _ in range(len(seq2) + 1)] for _ in range(len(seq1) + 1)]
    longest = 0
    commons = []
    for i1, item1 in enumerate(seq1):
        for i2, item2 in enumerate(seq2):
            if item1 == item2:
                tmp = counter[i1][i2] + 1
                counter[i1 + 1][i2 + 1] = tmp
                if tmp > longest:
                    commons = []
                    longest = tmp
                    commons.append(seq1[i1 - tmp + 1:i1 + 1])
                elif tmp == longest:
                    commons.append(seq1[i1 - tmp + 1:i1 + 1])
    if sorting is None:
        return commons
    else:
        return sorted(commons, key=sorting)


# ======================================================================
def common_subseq(
        seqs,
        sorting=None):
    """
    Find the longest common consecutive subsequence(s).
    This version works for an Iterable of Sequences.

    This is known as the `longest common substring` problem, or LCS for short.

    Args:
        seqs (Iterable[Sequence]): The input sequences.
            All the items must be of the same type.
        sorting (callable): Sorting function passed to 'sorted' via `key` arg.

    Returns:
        commons (list[Sequence]): The longest common subsequence(s).

    Examples:
        >>> common_subseq(['academy', 'abracadabra', 'cadet'])
        ['cad']
        >>> common_subseq(['los angeles', 'lossless', 'les alos'])
        ['los', 'les']
        >>> common_subseq(['los angeles', 'lossless', 'les alos', 'losles'])
        ['los', 'les']
        >>> common_subseq(['los angeles', 'lossless', 'dolos'])
        ['los']
        >>> common_subseq([(1, 2, 3, 4, 5), (1, 2, 3), (0, 1, 2)])
        [(1, 2)]
    """
    seqs = iter(seqs)
    commons = [next(seqs)]
    for text in seqs:
        tmps = []
        for common in commons:
            tmp = common_subseq_2(common, text, sorting)
            if len(tmps) == 0 or len(tmp[0]) == len(tmps[0]):
                tmps.extend(common_subseq_2(common, text, sorting))
        commons = tmps
    return commons


# ======================================================================
def is_bin_file(file_obj):
    """
    Check if a file object is in binary mode.

    Args:
        file_obj (File): The input file.

    Returns:
        result (bool): The binary mode status.

    Examples:
        >>> with open(__file__, 'rb') as file_obj:
        ...     is_bin_file(file_obj)
        True
        >>> with open(__file__, 'r') as file_obj:
        ...     is_bin_file(file_obj)
        False
        >>> is_bin_file(io.BytesIO(b'ciao'))
        True
        >>> is_bin_file(io.StringIO('ciao'))
        False
    """
    return isinstance(file_obj, io.IOBase) \
           and not isinstance(file_obj, io.TextIOBase)


# ======================================================================
def is_reading_bytes(file_obj):
    """
    Check if reading from a file object will return bytes.

    Args:
        file_obj (File): The input file.

    Returns:
        result (bool): The result of the check.

    Examples:
        >>> is_reading_bytes(io.BytesIO(b'ciao'))
        True
        >>> is_reading_bytes(io.StringIO('ciao'))
        False
    """
    return isinstance(file_obj.read(0), bytes)


# ======================================================================
def is_writing_bytes(file_obj):
    """
    Check if writing from a file object will require bytes.

    Args:
        file_obj (File): The input file.

    Returns:
        result (bool): The result of the check.

    Examples:
        >>> is_writing_bytes(io.BytesIO(b'ciao'))
        True
        >>> is_writing_bytes(io.StringIO('ciao'))
        False
    """
    try:
        file_obj.write(b'')
    except TypeError:
        return False
    else:
        return True


# ======================================================================
def same_file(
        grouper,
        *files,
        on_error=False):
    """
    Determine if two or more file objects refer to the same file.

    Args:
        files (Iterable[file]): The input file objects.
        grouper (callable): Determine how to group multiple comparisons.
            Must accept the following signature:
            grouper(*Iterable[bool]): bool
            Can be either `all` or `any`, or any callable with the supported
            signature.
        on_error (bool): Determine what to do if `fileno` is unsupported.

    Returns:
        result (bool): If the the two file objects refer to the same file.

    Examples:
        >>> same_file(all, open(__file__, 'r'), open(__file__, 'r'))
        True
        >>> same_file(all, open(__file__, 'r'), io.StringIO('FlyingCircus'))
        False
        >>> same_file(all, io.StringIO('fc'), io.StringIO('fc'))
        False
    """
    try:
        stats = tuple(os.fstat(file_.fileno()) for file_ in files)
    except io.UnsupportedOperation:
        return on_error
    else:
        inodes_devices = tuple((stat.st_ino, stat.st_dev) for stat in stats)
        return multi_compare(grouper, inodes_devices)


# ======================================================================
def blocks(
        file_obj,
        size=64 * 1024,
        reset_offset=True):
    """
    Yields the data within a file in blocks of given (max) size.

    For non-binary file objects, the size of the block may be reduced as a
    result of multi-byte support in the encoding.
    The last block may have a smaller size regardless of the opening mode.

    Args:
        file_obj (file): The input file.
        size (int|None): The block size.
            If int, the file is yielded in blocks of the specified size.
            If None, the file is yielded at once.
        reset_offset (bool): Reset the file offset.
            If True, starts reading from the beginning of the file.
            Otherwise, starts reading from where the file current position is.

    Yields:
        block (bytes|str): The data within the blocks.

    Examples:
        >>> src = 'flyingcircus-' * 4
        >>> tgt = ''.join(blocks(io.StringIO(src), 3))
        >>> print(tgt)
        flyingcircus-flyingcircus-flyingcircus-flyingcircus-
        >>> print(src == tgt)
        True

        >>> src = b'flyingcircus-' * 4
        >>> tgt = b''.join(blocks(io.BytesIO(src), 3))
        >>> print(tgt)
        b'flyingcircus-flyingcircus-flyingcircus-flyingcircus-'
        >>> print(src == tgt)
        True

        >>> src = 'φλυινγκιρκυσ-' * 4
        >>> tgt = ''.join(blocks(io.StringIO(src), 3))
        >>> print(tgt)
        φλυινγκιρκυσ-φλυινγκιρκυσ-φλυινγκιρκυσ-φλυινγκιρκυσ-
        >>> print(src == tgt)
        True

        >>> with open(__file__, 'r') as file_obj:
        ...     src = file_obj.read()
        ...     tgt = ''.join([b for b in blocks(file_obj, 100)])
        ...     src == tgt
        True
        >>> with open(__file__, 'rb') as file_obj:
        ...     src = file_obj.read()
        ...     tgt = b''.join([b for b in blocks(file_obj, 100)])
        ...     src == tgt
        True
    """
    if reset_offset:
        file_obj.seek(0)
    while True:
        block = file_obj.read(size)
        if not block:
            break
        else:
            yield block


# ======================================================================
def blocks_r(
        file_obj,
        size=64 * 1024,
        reset_offset=True):
    """
    Yields the data within a file in reverse order blocks of given (max) size.

    Note that:
     - the content of the block is NOT reversed.
     - files opened in text mode are not supported!

    Args:
        file_obj (file): The input file.
        size (int|None): The block size.
            If int, the file is yielded in blocks of the specified size.
            If None, the file is yielded at once.
        reset_offset (bool): Reset the file offset.
            If True, starts reading from the end of the file.
            Otherwise, starts reading from where the file current position is.

    Yields:
        block (bytes|str): The data within the blocks.

    Examples:
        >>> src = 'flyingcircus' * 4
        >>> my_blocks = list(blocks_r(io.StringIO(src), 12))
        >>> tgt = ''.join(my_blocks[::-1])
        >>> print(my_blocks)
        ['flyingcircus', 'flyingcircus', 'flyingcircus', 'flyingcircus']
        >>> print(src)
        flyingcircusflyingcircusflyingcircusflyingcircus
        >>> print(tgt)
        flyingcircusflyingcircusflyingcircusflyingcircus
        >>> print(src == tgt)
        True

        >>> src = b'flyingcircus' * 4
        >>> my_blocks = list(blocks_r(io.BytesIO(src), 12))
        >>> tgt = b''.join(my_blocks[::-1])
        >>> print(my_blocks)
        [b'flyingcircus', b'flyingcircus', b'flyingcircus', b'flyingcircus']
        >>> print(src)
        b'flyingcircusflyingcircusflyingcircusflyingcircus'
        >>> print(tgt)
        b'flyingcircusflyingcircusflyingcircusflyingcircus'
        >>> print(src == tgt)
        True

        >>> src = 'φλυινγκιρκυσ' * 4
        >>> my_blocks = list(blocks_r(io.StringIO(src), 12))
        >>> tgt = ''.join(my_blocks[::-1])
        >>> print(my_blocks)
        ['φλυινγκιρκυσ', 'φλυινγκιρκυσ', 'φλυινγκιρκυσ', 'φλυινγκιρκυσ']
        >>> print(src)
        φλυινγκιρκυσφλυινγκιρκυσφλυινγκιρκυσφλυινγκιρκυσ
        >>> print(tgt)
        φλυινγκιρκυσφλυινγκιρκυσφλυινγκιρκυσφλυινγκιρκυσ
        >>> print(src == tgt)
        True

        # THIS IS NOT SUPPORTED!
        # >>> with open(__file__, 'r') as file_obj:
        # ...     src = file_obj.read()
        # ...     tgt = ''.join([b for b in blocks_r(file_obj, 100)][::-1])
        # ...     src == tgt
        # True
        >>> with open(__file__, 'rb') as file_obj:
        ...     src = file_obj.read()
        ...     tgt = b''.join([b for b in blocks_r(file_obj, 100)][::-1])
        ...     src == tgt
        True
    """
    # : does not work well for files opened in text mode
    offset = file_obj.seek(0, os.SEEK_END) if reset_offset else file_obj.tell()
    while offset > 0:
        block_size = min(offset, size)
        offset -= block_size
        file_obj.seek(offset)
        block = file_obj.read(block_size)
        yield block


# ======================================================================
def auto_open(
        file_,
        *_args,
        **_kws):
    """
    Automatically open a file if a path is provided.

    Args:
        file_ (str|bytes|file): The file path or file object.
        *_args: Positional arguments for `open()`.
        **_kws: Keyword arguments for `open()`.

    Returns:
        file_ (file): The opened file object.
    """
    return open(file_, *_args, **_kws) \
        if isinstance(file_, (str, bytes)) else file_


# ======================================================================
def read_stream(
        in_file,
        dtype,
        mode='@',
        num_blocks=1,
        offset=None,
        whence=io.SEEK_SET):
    """
    Read data from stream.

    Args:
        in_file (str|bytes|file): The input file.
            Can be either a valid file path or a readable binary file object.
            See `flyingcircus.auto_open()` for more details.
        offset (int|None): The offset where to start reading.
        dtype (str): The data type to read.
            Accepted values are:
             - 'bool': boolean type (same as: '?', 1B)
             - 'char': signed char type (same as: 'b', 1B)
             - 'uchar': unsigned char type (same as: 'B', 1B)
             - 'short': signed short int type (same as: 'h', 2B)
             - 'ushort': unsigned short int  type (same as: 'H', 2B)
             - 'int': signed int type (same as: 'i', 4B)
             - 'uint': unsigned int type (same as: 'I', 4B)
             - 'long': signed long type (same as: 'l', 4B)
             - 'ulong': unsigned long type (same as: 'L', 4B)
             - 'llong': signed long long type (same as: 'q', 8B)
             - 'ullong': unsigned long long type (same as: 'Q', 8B)
             - 'float': float type (same as: 'f', 4B)
             - 'double': double type (same as: 'd', 8B)
             - 'str': c-str type (same as: 's', 'p')
            See Python's `struct` module for more information.
        num_blocks (int): The number of blocks to read.
        mode (str): Determine the byte order, size and alignment.
            Accepted values are:
             - '@': endianness: native,  size: native,   align: native.
             - '=': endianness:	native,  size: standard, align: none.
             - '<': endianness:	little,  size: standard, align: none.
             - '>': endianness:	big,     size: standard, align: none.
             - '!': endianness:	network, size: standard, align: none.
        whence (int): Where to reference the offset.
            Accepted values are:
             - '0': absolute file positioning.
             - '1': seek relative to the current position.
             - '2': seek relative to the file's end.

    Returns:
        data (tuple): The data read.
    """
    in_file_obj = auto_open(in_file, 'rb')
    if offset is not None:
        in_file_obj.seek(offset, whence)
    struct_format = mode + str(num_blocks) + DTYPE_STR[dtype]
    read_size = struct.calcsize(struct_format)
    data = struct.unpack_from(struct_format, in_file_obj.read(read_size))
    if in_file is not in_file_obj:
        in_file_obj.close()
    return data


# ======================================================================
def read_cstr(
        in_file,
        offset=None,
        whence=io.SEEK_SET):
    """
    Read a C-type string from file.

    Args:
        in_file (str|bytes|file): The input file.
            Can be either a valid file path or a readable binary file object.
            See `flyingcircus.auto_open()` for more details.
        offset (int|None): The offset where to start reading.
        whence (int): Where to reference the offset.
            Accepted values are:
             - '0': absolute file positioning.
             - '1': seek relative to the current position.
             - '2': seek relative to the file's end.

    Returns:
        text (str): The string read.
    """
    in_file_obj = auto_open(in_file, 'rb')
    if offset is not None:
        in_file_obj.seek(offset, whence)
    buffer = []
    while True:
        c = in_file_obj.read(1).decode('ascii')
        if c is None or c == '\0':
            break
        else:
            buffer.append(c)
    text = ''.join(buffer)
    if in_file is not in_file_obj:
        in_file_obj.close()
    return text


# ======================================================================
def process_stream(
        in_file,
        out_file,
        func,
        args=None,
        kws=None,
        block_size=100,
        as_binary=False):
    """
    Process the content of the input file and write it to the output file.

    Note: `in_file` and `out_file` should be different!

    Args:
        in_file (str|bytes|file): The input file.
            Can be either a valid file path or readable file object.
            Should not be the same as `out_file`.
            See `flyingcircus.auto_open()` for more details.
        out_file (str|bytes|file): The output file.
            Can be either a valid file path or writable file object.
            Should not be the same as `in_file`.
            See `flyingcircus.auto_open()` for more details.
        func (callable): The conversion function.
            Must accept a string or bytes (depending on `as_binary`) as first
            argument: func(str|bytes, *args, **kws): str|bytes
        args (Iterable|None): Positional arguments for `func()`.
        kws (Mappable|None): Keyword arguments for `func()`.
        block_size (int): The block size to use.
            If positive, apply `func` iteratively on input blocks of the
            specified size.
            If negative, apply `func` iteratively on each line.
            If zero, apply `func` on the entire input at once.
        as_binary (bool): Force binary I/O operations.
            If True, the first argument of `func` must be of type `bytes`.
            Otherwise, must be of type `str`.

    Returns:
        None.

    Examples:
        >>> text = 'flyingcircus-' * 4

        >>> i_file = io.StringIO(text)
        >>> o_file = io.StringIO('')
        >>> process_stream(i_file, o_file, str.upper)
        >>> pos = o_file.seek(0)
        >>> print(o_file.read())
        FLYINGCIRCUS-FLYINGCIRCUS-FLYINGCIRCUS-FLYINGCIRCUS-

        >>> i_file = io.BytesIO(text.encode())
        >>> o_file = io.BytesIO(b'')
        >>> process_stream(i_file, o_file, bytes.upper)
        >>> pos = o_file.seek(0)
        >>> print(o_file.read())
        b'FLYINGCIRCUS-FLYINGCIRCUS-FLYINGCIRCUS-FLYINGCIRCUS-'

        >>> i_file = io.StringIO(text)
        >>> o_file = io.StringIO('')
        >>> process_stream(
        ...     i_file, o_file, lambda x: '*' if x in 'irc' else x,
        ...     block_size=1)
        >>> pos = o_file.seek(0)
        >>> print(o_file.read())
        fly*ng****us-fly*ng****us-fly*ng****us-fly*ng****us-
    """
    args = tuple(args) if args is not None else ()
    kws = dict(kws) if kws is not None else {}
    bin_mode = 'b' if as_binary else ''
    # : open files
    in_file_obj = auto_open(in_file, 'r' + bin_mode)
    out_file_obj = auto_open(out_file, 'w' + bin_mode)
    input_equals_output = same_file(all, in_file_obj, out_file_obj)
    # : process content
    if input_equals_output:
        content = in_file_obj.read()
        content = func(content, *args, **kws)
        out_file_obj.write()
    else:
        if block_size < 0:
            for line in in_file_obj:
                line = func(line, *args, **kws)
                out_file_obj.writelines((line,))
        elif block_size > 0:
            for block in blocks(in_file_obj, block_size):
                block = func(block, *args, **kws)
                out_file_obj.write(block)
        else:
            out_file_obj.write(func(in_file_obj.read(), *args, **kws))
    # : close files (only if opened here)
    if in_file is not in_file_obj:
        in_file_obj.close()
    if out_file is not out_file_obj:
        out_file_obj.close()


# ======================================================================
def hash_file(
        in_file,
        hash_algorithm=hashlib.md5,
        filtering=base64.urlsafe_b64encode,
        coding='ascii',
        block_size=64 * 1024):
    """
    Compute the hash of a file.

    Args:
        in_file (str|bytes|file): The input file.
            Can be either a valid file path or a readable binary file object.
            See `flyingcircus.auto_open()` for more details.
        hash_algorithm (callable): The hashing algorithm.
            This must support the methods provided by `hashlib` module, like
            `md5`, `sha1`, `sha256`, `sha512`.
        filtering (callable|None): The filtering function.
            If callable, must have the following signature:
            filtering(bytes): bytes|str.
            If None, no additional filering is performed.
        coding (str): The coding for converting the returning object to str.
            If str, must be a valid coding.
            If None, the object is kept as bytes.
        block_size (int|None): The block size.
            See `size` argument of `flyingcircus.blocks` for exact
            behavior.

    Returns:
        hash_key (str|bytes): The result of the hashing.
    """
    hash_obj = hash_algorithm()
    with auto_open(in_file, 'rb') as file_obj:
        file_obj.seek(0)
        for block in blocks(file_obj, block_size):
            hash_obj.update(block)
    hash_key = hash_obj.digest()
    if filtering:
        hash_key = filtering(hash_key)
    if coding and isinstance(hash_key, bytes):
        hash_key = hash_key.decode(coding)
    return hash_key


# ======================================================================
def hash_object(
        obj,
        serializer=lambda x: repr(x).encode(),
        hash_algorithm=hashlib.md5,
        filtering=base64.urlsafe_b64encode,
        coding='ascii'):
    """
    Compute the hash of an object.

    Args:
        obj: The input object.
        serializer (callable): The function used to convert the object.
            Must have the following signature:
            serializer(Any): bytes
        hash_algorithm (callable): The hashing algorithm.
            This must support the methods provided by `hashlib` module, like
            `md5`, `sha1`, `sha256`, `sha512`.
        filtering (callable|None): The filtering function.
            If callable, must have the following signature:
            filtering(bytes): bytes.
            If None, no additional filering is performed.
        coding (str): The coding for converting the returning object to str.
            If str, must be a valid coding.
            If None, the object is kept as bytes.

    Returns:
        hash_key (str|bytes): The result of the hashing.
    """
    obj_bytes = serializer(obj)
    hash_key = hash_algorithm(obj_bytes).digest()
    if filtering:
        hash_key = filtering(hash_key)
    if coding and isinstance(hash_key, bytes):
        hash_key = hash_key.decode(coding)
    return hash_key


# ======================================================================
def from_cached(
        func,
        kws=None,
        dirpath=None,
        filename='{hash_key}.p',
        save_func=pickle.dump,
        load_func=pickle.load,
        force=False):
    """
    Compute or load from cache the result of a computation.

    Args:
        func (callable): The computation to perform.
        kws (dict|None): Keyword arguments for `func`.
        dirpath (str): The path of the caching directory.
        filename (str): The filename of the caching file.
            This is processed by `format` with `locals()`.
        save_func (callable): The function used to save caching file.
            Must have the following signature:
            save_func(file_obj, Any): None
            The value returned from `save_func` is not used.
        load_func (callable): The function used to load caching file.
            Must have the following signature:
            load_func(file_obj): Any
        force (bool): Force the calculation, regardless of caching state.

    Returns:
        result (Any): The result of the cached computation.
    """
    kws = dict(kws) if kws else {}
    hash_key = hash_object((func, kws))  # can be used in filename
    filepath = os.path.join(dirpath, fmtm(filename))
    if os.path.isfile(filepath) and not force:
        with open(filepath, 'rb') as file_obj:
            result = load_func(file_obj)
    else:
        result = func(**kws)
        with open(filepath, 'wb') as file_obj:
            save_func(file_obj, result)
    return result


# ======================================================================
def readline(
        file_obj,
        reverse=False,
        skip_empty=True,
        append_newline=True,
        block_size=64 * 1024,
        reset_offset=True):
    """
    Flexible function for reading lines incrementally.

    Args:
        file_obj (file): The input file.
        reverse (bool): Read the file in reverse mode.
            If True, the lines will be read in reverse order.
            This requires a binary file object.
            The content of each line will NOT be reversed.
        skip_empty (bool): Skip empty lines.
        append_newline (bool):
        block_size (int|None): The block size.
            If int, the file is processed in blocks of the specified size.
            If None, the file is processed at once.
        reset_offset (bool): Reset the file offset.
            If True, starts reading from the beginning of the file.
            Otherwise, starts reading from where the file current position is.
            This is passed to `blocks()` or `blocks_r()` (depending on the
            value of reverse).

    Yields:
        line (str|bytes): The next line.

    Examples:
        >>> with open(__file__, 'rb') as file_obj:
        ...     lines = [l for l in readline(file_obj, False)]
        ...     lines_r = [l for l in readline(file_obj, True)][::-1]
        ...     lines == lines_r
        True
    """
    is_bytes = is_reading_bytes(file_obj)
    newline = b'\n' if is_bytes else '\n'
    empty = b'' if is_bytes else ''
    remainder = empty
    block_generator_kws = dict(size=block_size, reset_offset=reset_offset)
    if not reverse:
        block_generator = blocks
    else:
        block_generator = blocks_r
    for block in block_generator(file_obj, **block_generator_kws):
        lines = block.split(newline)
        if remainder:
            if not reverse:
                lines[0] = remainder + lines[0]
            else:
                lines[-1] = lines[-1] + remainder
        remainder = lines[-1 if not reverse else 0]
        mask = slice(0, -1, 1) if not reverse else slice(-1, 0, -1)
        for line in lines[mask]:
            if line or not skip_empty:
                yield line + (newline if append_newline else empty)
    if remainder or not skip_empty:
        yield remainder + (newline if append_newline else empty)


# ======================================================================
def iwalk2(
        base,
        follow_links=False,
        follow_mounts=False,
        allow_special=False,
        allow_hidden=True,
        max_depth=-1,
        on_error=None):
    """
    Recursively walk through sub-paths of a base directory.

    This produces a generator for the next sub-path item.

    Args:
        base (str): Directory where to operate.
        follow_links (bool): Follow links during recursion.
        follow_mounts (bool): Follow mount points during recursion.
        allow_special (bool): Include special files.
        allow_hidden (bool): Include hidden files.
        max_depth (int): Maximum depth to reach. Negative for unlimited.
        on_error (callable): Function to call on error.

    Yields:
        result (tuple): The tuple
            contains:
             - path (str): Path to the next object.
             - stats (stat_result): File stats information.

    Returns:
        None.
    """

    # def _or_not_and(flag, check):
    #     return flag or not flag and check

    def _or_not_and_not(flag, check):
        return flag or not flag and not check

    try:
        for name in os.listdir(base):
            path = os.path.join(base, name)
            stats = os.stat(path)
            mode = stats.st_mode
            # for some reasons, stat.S_ISLINK and os.path.islink results differ
            allow = \
                _or_not_and_not(follow_links, os.path.islink(path)) and \
                _or_not_and_not(follow_mounts, os.path.ismount(path)) and \
                _or_not_and_not(allow_special, _is_special(mode)) and \
                _or_not_and_not(allow_hidden, _is_hidden(path))
            if allow:
                yield path, stats
                if os.path.isdir(path):
                    if max_depth != 0:
                        next_level = iwalk2(
                            path, follow_links, follow_mounts,
                            allow_special, allow_hidden, max_depth - 1,
                            on_error)
                        for next_path, next_stats in next_level:
                            yield next_path, next_stats

    except OSError as error:
        if on_error is not None:
            on_error(error)
        return


# ======================================================================
def walk2(
        base,
        follow_links=False,
        follow_mounts=False,
        allow_special=False,
        allow_hidden=True,
        max_depth=-1,
        on_error=None):
    """
    Recursively walk through sub paths of a base directory.

    This differs from `iwalk2()` in that it returns a list.

    Args:
        base (str): Directory where to operate.
        follow_links (bool): Follow links during recursion.
        follow_mounts (bool): Follow mount points during recursion.
        allow_special (bool): Include special files.
        allow_hidden (bool): Include hidden files.
        max_depth (int): Maximum depth to reach. Negative for unlimited.
        on_error (callable): Function to call on error.

    Returns:
        items (list[tuple]): The list of items.
            Each item contains the tuple with:
             - path (str): Path to the next object.
             - stats (stat_result): File stats information.
    """
    return [item for item in iwalk2(
        base,
        follow_links=follow_links, follow_mounts=follow_mounts,
        allow_special=allow_special, allow_hidden=allow_hidden,
        max_depth=max_depth, on_error=on_error)]


# ======================================================================
def which(args):
    """
    Determine the full path of an executable, if possible.

    It mimics the behavior of the POSIX command `which`.

    This has a similar behavior as `shutil.which()` except that it works on
    a list of arguments as returned by `shlex.split()` where the first item
    is the command to test.

    Args:
        args (str|list[str]): Command to execute as a list of tokens.
            If str this is filtered (tokenized) by `shlex.split()`.
            Otherwise, assumes an input like the output of `shlex.split()`.

    Returns:
        args (str|list[str]): Command to execute as a list of tokens.
            The first item of the list is the full path of the executable.
            If the executable is not found in path, returns the first token of
            the input.
            Other items are identical to input, if the input was a str list.
            Otherwise it will be the tokenized version of the passed string,
            except for the first token.
        is_valid (bool): True if path of executable is found, False otherwise.
    """

    def is_executable(file_path):
        return os.path.isfile(file_path) and os.access(file_path, os.X_OK)

    # ensure args in the correct format
    try:
        args = shlex.split(args)
    except AttributeError:
        pass

    cmd = os.path.expanduser(args[0])
    dirpath, filename = os.path.split(cmd)
    if dirpath:
        is_valid = is_executable(cmd)
    else:
        is_valid = False
        for dirpath in os.environ['PATH'].split(os.pathsep):
            dirpath = dirpath.strip('"')
            tmp = os.path.join(dirpath, cmd)
            is_valid = is_executable(tmp)
            if is_valid:
                cmd = tmp
                break
    return [cmd] + args[1:], is_valid


# ======================================================================
def execute(
        cmd,
        in_pipe=None,
        mode='call',
        timeout=None,
        encoding='utf-8',
        log=None,
        dry=False,
        verbose=D_VERB_LVL):
    """
    Execute command and retrieve/print output at the end of execution.

    Better handles `stdin`, `stdout` and `stderr`, as well as timeout,
    dry-run and verbosity compared to the many alternatives
    (as provided by the `subprocess` module, `os.system()`, `os.spawn*()`).

    For some applications the high-level API provided by `subprocess.run()`
    may be more appropriate.

    Args:
        cmd (str|Iterable[str]): The command to execute.
            If str, must be a shell-compatible invocation.
            If Iterable, each token must be a separate argument.
        in_pipe (str|None): Input data to be used as stdin of the process.
        mode (str): Set the execution mode (affects the return values).
            Allowed modes:
             - 'spawn': Spawn a new process. stdout and stderr will be lost.
             - 'call': Call new process and wait for execution.
                Once completed, obtain the return code, stdout, and stderr.
             - 'flush': Call new process and get stdout+stderr immediately.
                Once completed, obtain the return code.
        timeout (float): Timeout of the process in seconds.
        encoding (str): The encoding to use.
        log (str): The template filename to be used for logs.
            If None, no logs are produced.
        dry (bool): Print rather than execute the command (dry run).
        verbose (int): Set level of verbosity.

    Returns:
        ret_code (int|None): if mode not `spawn`, return code of the process.
        p_stdout (str|None): if mode not `spawn`, the stdout of the process.
        p_stderr (str|None): if mode is `call`, the stderr of the process.
    """
    ret_code, p_stdout, p_stderr = None, None, None

    cmd, is_valid = which(cmd)
    if is_valid:
        msg('{} {}'.format('$$' if dry else '>>', ' '.join(cmd)),
            verbose, D_VERB_LVL if dry else VERB_LVL['medium'])
    else:
        msg('W: `{}` is not in available in $PATH.'.format(next(iter(cmd))))

    if not dry and is_valid:
        if in_pipe is not None:
            msg('< {}'.format(in_pipe),
                verbose, VERB_LVL['highest'])

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if in_pipe and not mode == 'flush' else None,
            stdout=subprocess.PIPE if mode != 'spawn' else None,
            stderr=subprocess.PIPE if mode == 'call' else subprocess.STDOUT,
            shell=False)

        # handle stdout and stderr
        if mode == 'flush' and not in_pipe:
            p_stdout = ''
            while proc.poll() is None:
                out_buff = proc.stdout.readline().decode(encoding)
                p_stdout += out_buff
                msg(out_buff, fmtt='', end='')
                sys.stdout.flush()
            ret_code = proc.wait()
        elif mode == 'call':
            # try:
            p_stdout, p_stderr = proc.communicate(
                in_pipe.encode(encoding) if in_pipe else None)
            # except subprocess.TimeoutExpired:
            #     proc.kill()
            #     p_stdout, p_stderr = proc.communicate()
            p_stdout = p_stdout.decode(encoding)
            p_stderr = p_stderr.decode(encoding)
            if p_stdout:
                msg(p_stdout, verbose, VERB_LVL['high'], fmtt='')
            if p_stderr:
                msg(p_stderr, verbose, VERB_LVL['high'], fmtt='')
            ret_code = proc.wait()
        else:
            proc.kill()
            msg('E: mode `{}` and `in_pipe` not supported.'.format(mode))

        if log:
            name = os.path.basename(cmd[0])
            pid = proc.pid
            for stream, source in ((p_stdout, 'out'), (p_stderr, 'err')):
                if stream:
                    log_filepath = fmtm(log)
                    with open(log_filepath, 'wb') as fileobj:
                        fileobj.write(stream.encode(encoding))
    return ret_code, p_stdout, p_stderr


# ======================================================================
def parallel_execute(
        cmds,
        pool_size=None,
        poll_interval=60,
        callback=None,
        callback_args=None,
        callback_kws=None,
        verbose=D_VERB_LVL):
    """
    Spawn parallel processes and wait until all processes are completed.

    Args:
        cmds (Sequence[str|Iterable[str]): The commands to execute.
            Each item must be a separate command.
            If the item is a str, it must be a shell-compatible invocation.
            If the item is an Iterable, each token must be a separate argument.
        pool_size (int|None): The size of the parallel pool.
            This corresponds to the maximum number of concurrent processes
            spawned by the function.
        poll_interval (int): The poll interval in s.
            This is the time between status updates in seconds.
        callback (callable|None): A callback function.
            This is called after each poll interval.
        callback_args (Iterable|None): Positional arguments for `callback()`.
        callback_kws (Mappable|None): Keyword arguments for `callback()`.
        verbose (int): Set level of verbosity.

    Returns:
        None.
    """
    callback_args = tuple(callback_args) if callback_args is not None else ()
    callback_kws = dict(callback_args) if callback_kws is not None else {}
    if not pool_size:
        pool_size = multiprocessing.cpu_count() + 1
    num_total = len(cmds)
    num_processed = 0
    begin_dt = datetime.datetime.now()
    procs = [
        subprocess.Popen(cmd, shell=True) for cmd in cmds[:pool_size]]
    for proc in procs:
        msg('{} {}'.format('>>', ' '.join(proc.args)),
            verbose, VERB_LVL['medium'])
    num_submitted = len(procs)
    done = False
    while not done:
        num_batch = len(procs)
        procs = [proc for proc in procs if proc.poll() is None]
        num_running = len(procs)
        num_done = num_batch - num_running
        if num_done > 0:
            num_processed += num_done
            new_procs = [
                subprocess.Popen(cmd, shell=True)
                for cmd in cmds[num_submitted:num_submitted + num_done]]
            for proc in new_procs:
                msg('{} {}'.format('>>', ' '.join(proc.args)),
                    verbose, VERB_LVL['medium'])
            num_submitted += len(new_procs)
            procs += new_procs
            num_running = len(procs)
        elapsed_dt = datetime.datetime.now() - begin_dt
        text = fmtm(
            'I: {num_processed} / {num_total} processed'
            ' ({num_running} running) - Elapsed: {elapsed_dt}')
        msg(text, verbose, D_VERB_LVL)
        if callable(callback):
            callback(*callback_args, **callback_kws)
        if num_processed == num_total:
            done = True
        else:
            time.sleep(poll_interval)


# ======================================================================
def realpath(
        path,
        create=True):
    """
    Get the expanded absolute path from its short or relative counterpart.

    Args:
        path (str): The path to expand.
        create (bool): Automatically create the path if not exists.

    Returns:
        new_path (str): the expanded path.

    Raises:
        OSError: if the expanded path does not exists.
    """
    new_path = os.path.abspath(os.path.realpath(os.path.expanduser(path)))
    if not os.path.exists(new_path):
        if create:
            os.makedirs(new_path)
        else:
            raise OSError
    return new_path


# ======================================================================
def listdir(
        path,
        file_ext='',
        full_path=True,
        sorting=True,
        verbose=D_VERB_LVL):
    """
    Retrieve a sorted list of files matching specified extension and pattern.

    Args:
        path (str): Path to search.
        file_ext (str|None): File extension. Empty string for all files.
            None for directories.
        full_path (bool): Include the full path.
        sorting (bool): Sort results alphabetically.
        verbose (int): Set level of verbosity.

    Returns:
        list[str]: List of file names/paths
    """
    if file_ext is None:
        msg('Scanning for dirs on:\n{}'.format(path),
            verbose, VERB_LVL['debug'])
        filepaths = [
            os.path.join(path, filename) if full_path else filename
            for filename in os.listdir(path)
            if os.path.isdir(os.path.join(path, filename))]
    else:
        msg('Scanning for {} on:\n{}'.format(
            ('`' + file_ext + '`') if file_ext else 'files', path),
            verbose, VERB_LVL['debug'])
        # extracts only those ending with specific file_ext
        filepaths = [
            os.path.join(path, filename) if full_path else filename
            for filename in os.listdir(path)
            if filename.lower().endswith(file_ext.lower())]
    if sorting:
        filepaths = sorted(filepaths)
    return filepaths


# ======================================================================
def iflistdir(
        patterns='*',
        dirpath='.',
        unix_style=True,
        re_kws=None,
        walk_kws=None):
    """
    Recursively list the content of a directory matching the pattern(s).

    Args:
        dirpath (str): The base directory.
        patterns (str|Iterable[str]): The pattern(s) to match.
            These must be either a Unix-style pattern or a regular expression,
            depending on the value of `unix_style`.
        unix_style (bool): Interpret the patterns as Unix-style.
            This is achieved by using `fnmatch`.
        re_kws (Mappable|None): Keyword arguments for `re.compile()`.
        walk_kws (Mappable|None): Keyword arguments for `os.walk()`.

    Yields:
        filepath (str): The next matched filepath.
    """
    if isinstance(patterns, str):
        patterns = (patterns,)
    if re_kws is None:
        re_kws = dict()
    if walk_kws is None:
        walk_kws = dict()
    for pattern in patterns:
        if unix_style:
            pattern = fnmatch.translate(pattern)
        re_obj = re.compile(pattern, **re_kws)
        for root, dirs, files in os.walk(dirpath, **walk_kws):
            for base in (dirs + files):
                filepath = os.path.join(root, base)
                if re_obj.match(filepath):
                    yield filepath


# ======================================================================
def flistdir(
        patterns='*',
        dirpath='.',
        unix_style=True,
        re_kws=None,
        walk_kws=None):
    """
    Recursively list the content of a directory matching the pattern(s).

    Args:
        dirpath (str): The base directory.
        patterns (str|Iterable[str]): The pattern(s) to match.
            These must be either a Unix-style pattern or a regular expression,
            depending on the value of `unix_style`.
        unix_style (bool): Interpret the patterns as Unix-style.
            This is achieved by using `fnmatch`.
        re_kws (Mappable|None): Keyword arguments for `re.compile()`.
        walk_kws (Mappable|None): Keyword arguments for `os.walk()`.

    Returns:
        filepaths (list[str]): The matched filepaths.
    """
    return [
        item for item in iflistdir(
            patterns=patterns, dirpath=dirpath, unix_style=unix_style,
            re_kws=re_kws, walk_kws=walk_kws)]


# ======================================================================
def add_extsep(
        ext,
        extsep=os.path.extsep):
    """
    Add a extsep char to a filename extension, if it does not have one.

    Args:
        ext (str|None): Filename extension to which the dot has to be added.
        extsep (str): The string to use a filename extension separator.

    Returns:
        ext (str): Filename extension with a prepending dot.

    Examples:
        >>> add_extsep('txt')
        '.txt'
        >>> add_extsep('.txt')
        '.txt'
        >>> add_extsep('')
        '.'
        >>> add_extsep(None)
        '.'
    """
    if ext is None:
        ext = ''
    return ('' if ext.startswith(extsep) else extsep) + ext


# ======================================================================
def split_ext(
        filepath,
        ext=None,
        case_sensitive=False,
        auto_multi_ext=True):
    """
    Split the filepath into a pair (root, ext), so that: root + ext == path.
    root is everything that precedes the first extension separator.
    ext is the extension (including the separator).

    It can automatically detect multiple extensions.
    Since `os.path.extsep` is often '.', a `os.path.extsep` between digits is
    not considered to be generating and extension.

    Args:
        filepath (str): The input filepath.
        ext (str|None): The expected extension (with or without the dot).
            If None, it will be obtained automatically.
            If empty, no split is performed.
        case_sensitive (bool): Case-sensitive match of old extension.
            If `ext` is None or empty, it has no effect.
        auto_multi_ext (bool): Automatically detect multiple extensions.
            If True, include multiple extensions.
            If False, only the last extension is detected.
            If `ext` is not None or empty, it has no effect.

    Returns:
        result (tuple): The tuple
            contains:
             - root (str): The filepath without the extension.
             - ext (str): The extension including the separator.

    Examples:
        >>> split_ext('test.txt', '.txt')
        ('test', '.txt')
        >>> split_ext('test.txt')
        ('test', '.txt')
        >>> split_ext('test.txt.gz')
        ('test', '.txt.gz')
        >>> split_ext('test_1.0.txt')
        ('test_1.0', '.txt')
        >>> split_ext('test.0.txt')
        ('test', '.0.txt')
        >>> split_ext('test.txt', '')
        ('test.txt', '')
    """
    root = filepath
    if ext is not None:
        ext = add_extsep(ext)
        has_ext = filepath.lower().endswith(ext.lower()) \
            if not case_sensitive else filepath.endswith(ext)
        if has_ext:
            root = filepath[:-len(ext)]
        else:
            ext = ''
    else:
        if auto_multi_ext:
            ext = ''
            is_valid = True
            while is_valid:
                tmp_filepath_noext, tmp_ext = os.path.splitext(root)
                if tmp_filepath_noext and tmp_ext:
                    is_valid = not (tmp_ext[1].isdigit() and
                                    tmp_filepath_noext[-1].isdigit())
                    if is_valid:
                        root = tmp_filepath_noext
                        ext = tmp_ext + ext
                else:
                    is_valid = False
        else:
            root, ext = os.path.splitext(filepath)
    return root, ext


# ======================================================================
def split_path(
        filepath,
        auto_multi_ext=True):
    """
    Split the filepath into (root, base, ext).

    Note that: root + os.path.sep + base + ext == path.
    (and therfore: root + base + ext != path).

    root is everything that preceeds the last path separator.
    base is everything between the last path separator and the first
    extension separator.
    ext is the extension (including the separator).

    Note that this separation is performed only on the string and it is not
    aware of the filepath actually existing, being a file, a directory,
    or similar aspects.

    Args:
        filepath (str): The input filepath.
        auto_multi_ext (bool): Automatically detect multiple extensions.
            Refer to `split_ext()` for more details.

    Returns:
        result (tuple): The tuple
            contains:
             - root (str): The filepath without the last item.
             - base (str): The file name without the extension.
             - ext (str): The extension including the extension separator.

    Examples:
        >>> split_path('/path/to/file.txt')
        ('/path/to', 'file', '.txt')
        >>> split_path('/path/to/file.tar.gz')
        ('/path/to', 'file', '.tar.gz')
        >>> split_path('file.tar.gz')
        ('', 'file', '.tar.gz')
        >>> split_path('/path/to/file')
        ('/path/to', 'file', '')

        >>> root, base, ext = split_path('/path/to/file.ext')
        >>> root + os.path.sep + base + ext
        '/path/to/file.ext'

    See Also:
        - flyingcircus.join_path()
        - flyingcircus.multi_split_path()
    """
    root, base_ext = os.path.split(filepath)
    base, ext = split_ext(base_ext, auto_multi_ext=auto_multi_ext)
    return root, base, ext


# ======================================================================
def multi_split_path(
        filepath,
        auto_multi_ext=True):
    """
    Split the filepath into (subdir, subdir, ..., base, ext).

    Note that: splits[0] + os.path.sep.join(splits[1:-1]) + ext == path
    (and therfore e.g.: ''.join(splits) != path).

    `root` is everything that preceeds the last path separator.
    `base` is everything between the last path separator and the first
    extension separator.
    `ext` is the extension (including the separator).

    Note that this separation is performed only on the string and it is not
    aware of the filepath actually existing, being a file, a directory,
    or similar aspects.

    Args:
        filepath (str): The input filepath.
        auto_multi_ext (bool): Automatically detect multiple extensions.
            Refer to `split_ext()` for more details.

    Returns:
        result (tuple[str]): The parts of the file.
            If the first char of the filepath is `os.path.sep`, then
            the first item is set to `os.path.sep`.
            The first (n - 2) items are subdirectories, the penultimate item
            is the file name without the extension, the last item is the
            extension including the extension separator.

    Examples:
        >>> multi_split_path('/path/to/file.txt')
        ('/', 'path', 'to', 'file', '.txt')
        >>> multi_split_path('/path/to/file.tar.gz')
        ('/', 'path', 'to', 'file', '.tar.gz')
        >>> multi_split_path('file.tar.gz')
        ('file', '.tar.gz')
        >>> multi_split_path('/path/to/file')
        ('/', 'path', 'to', 'file', '')

        >>> splits = multi_split_path('/path/to/file.ext')
        >>> splits[0] + os.path.sep.join(splits[1:-1]) + splits[-1]
        '/path/to/file.ext'

    See Also:
        - flyingcircus.join_path()
        - flyingcircus.split_path()
    """
    root, base_ext = os.path.split(filepath)
    base, ext = split_ext(base_ext, auto_multi_ext=auto_multi_ext)
    if root:
        dirs = root.split(os.path.sep)
        if dirs[0] == '':
            dirs[0] = os.path.sep
    else:
        dirs = ()
    return tuple(dirs) + (base, ext)


# ======================================================================
def join_path(texts):
    """
    Join a list of items into a filepath.

    The last item is treated as the file extension.
    Path and extension separators do not need to be manually included.

    Note that this is the inverse of `split_path()`.

    Args:
        texts (Iterable[str]): The path elements to be concatenated.
            The last item is treated as the file extension.

    Returns:
        filepath (str): The output filepath.

    Examples:
        >>> join_path(('/path/to', 'file', '.txt'))
        '/path/to/file.txt'
        >>> join_path(('/path/to', 'file', '.tar.gz'))
        '/path/to/file.tar.gz'
        >>> join_path(('', 'file', '.tar.gz'))
        'file.tar.gz'
        >>> join_path(('path/to', 'file', ''))
        'path/to/file'
        >>> paths = [
        ...     '/path/to/file.txt', '/path/to/file.tar.gz', 'file.tar.gz']
        >>> all(path == join_path(split_path(path)) for path in paths)
        True
        >>> paths = [
        ...     '/path/to/file.txt', '/path/to/file.tar.gz', 'file.tar.gz']
        >>> all(path == join_path(multi_split_path(path)) for path in paths)
        True

    See Also:
        - flyingcircus.split_path()
        - flyingcircus.multi_split_path()
    """
    return ((os.path.join(*texts[:-1]) if texts[:-1] else '') +
            (add_extsep(texts[-1]) if texts[-1] else ''))


# ======================================================================
def basename(
        filepath,
        ext=None,
        case_sensitive=False,
        auto_multi_ext=True):
    """
    Remove path AND the extension from a filepath.

    Args:
        filepath (str): The input filepath.
        ext (str|None): The expected extension (with or without the dot).
            Refer to `split_ext()` for more details.
        case_sensitive (bool): Case-sensitive match of expected extension.
            Refer to `split_ext()` for more details.
        auto_multi_ext (bool): Automatically detect multiple extensions.
            Refer to `split_ext()` for more details.

    Returns:
         root (str): The file name without path and extension.

    Examples:
        >>> basename('/path/to/file/test.txt', '.txt')
        'test'
        >>> basename('/path/to/file/test.txt.gz')
        'test'
    """
    filepath = os.path.basename(filepath)
    root, ext = split_ext(
        filepath, ext, case_sensitive, auto_multi_ext)
    return root


# ======================================================================
def change_ext(
        root,
        new_ext,
        ext=None,
        case_sensitive=False,
        auto_multi_ext=True):
    """
    Substitute the old extension with a new one in a filepath.

    Args:
        root (str): The input filepath.
        new_ext (str): The new extension (with or without the dot).
        ext (str|None): The expected extension (with or without the dot).
            Refer to `split_ext()` for more details.
        case_sensitive (bool): Case-sensitive match of expected extension.
            Refer to `split_ext()` for more details.
        auto_multi_ext (bool): Automatically detect multiple extensions.
            Refer to `split_ext()` for more details.

    Returns:
        filepath (str): Output filepath

    Examples:
        >>> change_ext('test.txt', 'dat', 'txt')
        'test.dat'
        >>> change_ext('test.txt', '.dat', 'txt')
        'test.dat'
        >>> change_ext('test.txt', '.dat', '.txt')
        'test.dat'
        >>> change_ext('test.txt', 'dat', '.txt')
        'test.dat'
        >>> change_ext('test.txt', 'dat', 'TXT', False)
        'test.dat'
        >>> change_ext('test.txt', 'dat', 'TXT', True)
        'test.txt.dat'
        >>> change_ext('test.tar.gz', 'tgz')
        'test.tgz'
        >>> change_ext('test.tar.gz', 'tgz', 'tar.gz')
        'test.tgz'
        >>> change_ext('test.tar.gz', 'tgz', auto_multi_ext=False)
        'test.tar.tgz'
        >>> change_ext('test.tar', 'gz', '')
        'test.tar.gz'
        >>> change_ext('test.tar', 'gz', None)
        'test.gz'
        >>> change_ext('test.tar', '')
        'test'
    """
    root, ext = split_ext(
        root, ext, case_sensitive, auto_multi_ext)
    filepath = root + (add_extsep(new_ext) if new_ext else '')
    return filepath


# ======================================================================
def next_filepath(
        filepath,
        out_template='{basepath}__{counter}{ext}',
        verbose=D_VERB_LVL):
    """
    Generate a non-existing filepath if current exists.

    Args:
        filepath (str): The input filepath.
        out_template (str): Template for the output filepath.
            The following variables are available for interpolation:
             - `dirpath`: The directory of the input.
             - `base`: The input base file name without extension.
             - `ext`: The input file extension (with leading separator).
             - `basepath`: The input filepath without extension.
        verbose (int): Set level of verbosity.

    Returns:
        filepath (str)
    """
    if os.path.exists(filepath):
        msg('OLD: {}'.format(filepath), verbose, VERB_LVL['high'])
        dirpath, base, ext = split_path(filepath)
        basepath = os.path.join(dirpath, base)
        counter = 0
        while os.path.exists(filepath):
            counter += 1
            filepath = fmtm(out_template)
        msg('NEW: {}'.format(filepath), verbose, VERB_LVL['medium'])
    return filepath


# ======================================================================
def safe_filename(
        text,
        allowed='a-zA-Z0-9._-',
        replacing='_',
        group_consecutive=True):
    """
    Return a string containing a safe filename.

    Args:
        text (str): The input string.
        allowed (str):  The valid characters.
            Must comply to Python's regular expression syntax.
        replacing (str): The replacing text.
        group_consecutive (bool): Group consecutive non-allowed.
            If True, consecutive non-allowed characters are replaced by a
            single instance of `replacing`.
            Otherwise, each character is replaced individually.

    Returns:
        text (str): The filtered text.

    Examples:
        >>> safe_filename('flyingcircus.txt')
        'flyingcircus.txt'
        >>> safe_filename('flyingcircus+12.txt')
        'flyingcircus_12.txt'
        >>> safe_filename('flyingcircus+12.txt')
        'flyingcircus_12.txt'
        >>> safe_filename('flyingcircus+++12.txt')
        'flyingcircus_12.txt'
        >>> safe_filename('flyingcircus+++12.txt', group_consecutive=False)
        'flyingcircus___12.txt'
        >>> safe_filename('flyingcircus+12.txt', allowed='a-zA-Z0-9._+-')
        'flyingcircus+12.txt'
        >>> safe_filename('flyingcircus+12.txt', replacing='-')
        'flyingcircus-12.txt'
    """
    return re.sub(
        r'[^{allowed}]{greedy}'.format(
            allowed=allowed, greedy='+' if group_consecutive else ''),
        replacing, text)


# ======================================================================
def magic_open(
        filepath,
        *_args,
        **_kws):
    """
    Auto-magically open a compressed file.

    Supports `gzip` and `bzip2`.

    Note: all compressed files should be opened as binary.
    Opening in text mode is not supported.

    Args:
        filepath (str|bytes): The file path.
        *_args: Positional arguments for `open()`.
        **_kws: Keyword arguments for `open()`.

    Returns:
        file_obj: A file object.

    Raises:
        IOError: on failure.

    See Also:
        open(), gzip.open(), bz2.open()

    Examples:
        >>> file_obj = magic_open(__file__, 'rb')
    """
    zip_module_names = 'gzip', 'bz2'
    file_obj = None
    for zip_module_name in zip_module_names:
        try:
            zip_module = importlib.import_module(zip_module_name)
            file_obj = zip_module.open(filepath, *_args, **_kws)
            file_obj.read(1)
        except (OSError, IOError, AttributeError, ImportError):
            file_obj = None
        else:
            file_obj.seek(0)
            break
    if not file_obj:
        file_obj = open(filepath, *_args, **_kws)
    return file_obj


# ======================================================================
def zopen(
        filepath,
        mode='rb',
        *_args,
        **_kws):
    """
    Auto-magically open a compressed file.

    Supports the following file formats:
     - `gzip`: GNU Zip -- DEFLATE (Lempel-Ziv 77 + Huffman coding)
     - `bzip2`: Burrows-Wheeler algorithm
     - `lzma`/`xz`: Lempel-Ziv-Markov chain algorithm (LZMA) (Python 3 only)

    This is achieved through the Python standard library modules.

    Notes:
    - All compressed files should be opened as binary.
      Opening in text mode is not supported.

    Args:
        filepath (str): The file path.
            This cannot be a file object.
        mode (str): The mode for file opening.
            See `open()` for more info.
            If the `t` mode is not specified, `b` mode is assumed.
            If `t` mode is specified, the file cannot be compressed.
        *_args: Positional arguments for `open()`.
        **_kws: Keyword arguments for `open()`.

    Returns:
        file_obj: A file object.

    Raises:
        IOError: on failure.

    See Also:
        open(), gzip.open(), bz2.open()

    Examples:
        >>> file_obj = zopen(__file__, 'rb')
    """
    if 't' not in mode and 'b' not in mode:
        mode += 'b'

    valid_mode = 'b' in mode and 't' not in mode

    # try open file as normal
    file_obj = open(filepath, mode=mode, *_args, **_kws)

    if valid_mode:
        # test if file is compressed using its header
        # for gzip: magic
        # for bzip2: magic, version, hundred_k_blocksize, compressed_magic
        # for xz/lzma: non-standard magic
        try:
            import lzma  # Compression using the LZMA algorithm
        except ImportError:
            lzma = None
        try:
            head = file_obj.read(16)
            gz_by_header = head[:2] == b'\x1f\x8b'
            bz2_by_header = (
                    head[:2] == b'BZ' and head[2:3] == b'h'
                    and head[3:4].isdigit()
                    and head[4:10] == b'\x31\x41\x59\x26\x53\x59')
            xz_by_header = head[:1] == b'\xfd7zXz\x00'
        except io.UnsupportedOperation:
            gz_by_header = False
            bz2_by_header = False
            xz_by_header = False
        finally:
            file_obj.seek(0)

        gz_by_ext = split_ext(filepath)[1].endswith(add_extsep(EXT['gzip']))
        bz2_by_ext = split_ext(filepath)[1].endswith(add_extsep(EXT['bzip2']))
        xz_by_ext = (
                split_ext(filepath)[1].endswith(add_extsep(EXT['xz'])) or
                split_ext(filepath)[1].endswith(add_extsep(EXT['lzma'])))

        if gz_by_header or gz_by_ext:
            file_obj = gzip.GzipFile(fileobj=file_obj, mode=mode)
        elif bz2_by_header or bz2_by_ext:
            file_obj = bz2.BZ2File(filename=file_obj, mode=mode)
        elif lzma and xz_by_header or xz_by_ext:
            file_obj = lzma.LZMAFile(filename=file_obj, mode=mode)

    return file_obj


# ======================================================================
def compact_num_str(
        val,
        max_lim=D_TAB_SIZE - 1):
    """
    Convert a number into the most informative string within specified limit.

    Args:
        val (int|float): The number to be converted to string.
        max_lim (int): The maximum number of characters allowed for the string.

    Returns:
        val_str (str): The string with the formatted number.

    Examples:
        >>> compact_num_str(100.0, 3)
        '100'
        >>> compact_num_str(100.042, 6)
        '100.04'
        >>> compact_num_str(100.042, 9)
        '100.04200'
    """

    try:
        # this is to simplify formatting (and accepting even strings)
        val = float(val)
        # helpers
        extra_char_in_exp = 5
        extra_char_in_dec = 2
        extra_char_in_sign = 1
        # 'order' of zero is 1 for our purposes, because needs 1 char
        order = math.log10(abs(val)) if abs(val) > 0.0 else 1
        # adjust limit for sign
        limit = max_lim - extra_char_in_sign if val < 0.0 else max_lim
        # perform the conversion
        if order > float(limit) or order < -float(extra_char_in_exp - 1):
            limit -= extra_char_in_exp + 1
            val_str = '{:.{size}e}'.format(val, size=limit)
        elif -float(extra_char_in_exp - 1) <= order < 0.0:
            limit -= extra_char_in_dec
            val_str = '{:.{size}f}'.format(val, size=limit)
        elif val % 1.0 == 0.0:
            # currently, no distinction between int and float is made
            limit = 0
            val_str = '{:.{size}f}'.format(val, size=limit)
        else:
            limit -= (extra_char_in_dec + int(order))
            if limit < 0:
                limit = 0
            val_str = '{:.{size}f}'.format(val, size=limit)
    except (TypeError, ValueError):
        warnings.warn('Could not convert value `{}` to float'.format(val))
        val_str = 'NaN'
    return val_str


# ======================================================================
def obj2str(
        obj,
        names=None,
        skip='_',
        blacklist=(),
        base=('__class__', '__name__'),
        base_sep=': ',
        attr_sep=', ',
        kv_sep='=',
        pre_delim='<',
        post_delim='>'):
    """
    Generate a meaningful string representation of an object.

    Args:
        obj (Any): The input object.
        names (Iterable[str]|None): The attribute names to consider.
            If None, uses `idir(obj, skip, methods=False, attributes=True)`
        skip (str|callable): The skip criterion.
            If str, names starting with it are skipped.
            If callable, skips when `skip(name)` evaluates to True.
        base (str|Iterable[str]|callable): The base object name.
            If str, uses `getattr(obj, base)`.
            If Iterable[str], uses `get_nested_attr(obj, *base)`.
            If callable, uses `base(obj)`.
            If None, the base part is skipped.
        base_sep (str): The name-to-attributes separator.
        attr_sep (str): The attributes separator.
        kv_sep (str): The key-value separator.
        pre_delim (str): The prefix delimiter.
            This is prepended to the final string.
        post_delim (str): The postfix delimiter.
            This is appended to the final string.

    Returns:
        str: A text representation of the object.

    Examples:
        >>> print(obj2str(1))
        <int: denominator=1, imag=0, numerator=1, real=1>
        >>> print(obj2str(1, base=None))
        <denominator=1, imag=0, numerator=1, real=1>
        >>> print(obj2str(1, ('real', 'imag')))
        <int: real=1, imag=0>
        >>> print(obj2str(1, pre_delim='{', post_delim='}'))
        {int: denominator=1, imag=0, numerator=1, real=1}
        >>> print(obj2str(1, pre_delim='<!', post_delim='>'))
        <!int: denominator=1, imag=0, numerator=1, real=1>
        >>> print(obj2str(1, base_sep='::', attr_sep=',', kv_sep=':'))
        <int::denominator:1,imag:0,numerator:1,real:1>
        >>> print(obj2str(1, blacklist=('denominator', 'imag')))
        <int: numerator=1, real=1>
    """
    if callable(base):
        base_name = base(obj)
    elif isinstance(base, str):
        base_name = getattr(obj, base)
    elif base is not None:
        base_name = get_nested_attr(obj, *base)
    else:
        base_name = base_sep = ''
    if names is None:
        if blacklist:
            def skip(name, s=skip):
                return name.startswith(s) or name in blacklist
        text = attr_sep.join(
            '{}{}{}'.format(name, kv_sep, attr)
            for name, attr in idir(obj, skip, False, True, True))
    else:
        text = attr_sep.join(
            '{}{}{}'.format(name, kv_sep, getattr(obj, name))
            for name in names if hasattr(obj, name))
    return '{}{}{}{}{}'.format(
        pre_delim, base_name, base_sep, text, post_delim)


# ======================================================================
def has_delim(
        text,
        pre_delim='"',
        post_delim='"'):
    """
    Determine if a string is delimited by some characters (decorators).

    Args:
        text (str): The text input string.
        pre_delim (str): initial string decorator.
        post_delim (str): final string decorator.

    Returns:
        has_delim (bool): True if text is delimited by the specified chars.

    Examples:
        >>> has_delim('"test"')
        True
        >>> has_delim('"test')
        False
        >>> has_delim('<test>', '<', '>')
        True
    """
    return text.startswith(pre_delim) and text.endswith(post_delim)


# ======================================================================
def strip_delim(
        text,
        pre_delim='"',
        post_delim='"'):
    """
    Strip initial and final character sequences (decorators) from a string.

    Args:
        text (str): The text input string.
        pre_delim (str): initial string decorator.
        post_delim (str): final string decorator.

    Returns:
        text (str): the text without the specified decorators.

    Examples:
        >>> strip_delim('"test"')
        'test'
        >>> strip_delim('"test')
        'test'
        >>> strip_delim('<test>', '<', '>')
        'test'
    """
    begin = len(pre_delim) if text.startswith(pre_delim) else None
    end = -len(post_delim) if text.endswith(post_delim) else None
    return text[begin:end]


# ======================================================================
def to_bool(
        value,
        mappings=(('false', 'true'), ('0', '1'), ('off', 'on')),
        case_sensitive=False,
        strip=True):
    """
    Conversion to boolean value.

    This is especially useful to interpret strings are booleans, because
    the built-in `bool()` method evaluates to False for empty strings and
    True for non-empty strings.

    Args:
        value (str|Any): The input value.
            If not string, attempt the built-in `bool()` casting.
        mappings (Sequence[Sequence]): The string values to map as boolean.
            Each item consists of an Sequence. Within the inner Sequence,
            the first element is mapped to False and all other elements map
            to True.
        case_sensitive (bool): Perform case-sensitive comparison.
        strip (bool): Strip whitespaces from input string.
            If input is not `str`, it has no effect.

    Returns:
        result (bool): The value converted to boolean.

    Raises:
        ValueError: if the conversion to boolean fails.

    Examples:
        >>> to_bool('false')
        False
        >>> to_bool('true')
        True
        >>> to_bool('0')
        False
        >>> to_bool('off')
        False
        >>> to_bool('0 ')
        False
        >>> to_bool(1)
        True
        >>> to_bool(0)
        False
        >>> to_bool(0j)
        False
        >>> to_bool('False')
        False
        >>> to_bool('False', case_sensitive=True)
        Traceback (most recent call last):
            ....
        ValueError: Cannot convert to bool
        >>> to_bool('False ', strip=False)
        Traceback (most recent call last):
            ....
        ValueError: Cannot convert to bool
    """
    if isinstance(value, str):
        if strip:
            value = value.strip()
        if not case_sensitive:
            value = value.lower()
            mappings = tuple(
                tuple(match.lower() for match in mapping)
                for mapping in mappings)
        for mapping in mappings:
            for i, match in enumerate(mapping):
                if value == match:
                    # : equivalento to, but faster than:
                    # return bool(i)
                    return i > 0
        else:
            raise ValueError('Cannot convert to bool')
    else:
        return bool(value)


# ======================================================================
def auto_convert(
        text,
        pre_delim=None,
        post_delim=None,
        casts=(int, float, complex, to_bool)):
    """
    Convert value to numeric if possible, or strip delimiters from string.

    Args:
        text (str|Number): The text input string.
        pre_delim (str): initial string decorator.
        post_delim (str): final string decorator.
        casts (Iterable[callable]): The cast conversion methods.
            Each callable must be able to perform the desired conversion,
            or raise either a ValueError or a TypeError on failure.

    Returns:
        val (Number): The numeric value of the string.

    Examples:
        >>> auto_convert('<100>', '<', '>')
        100
        >>> auto_convert('<100.0>', '<', '>')
        100.0
        >>> auto_convert('100.0+50j')
        (100+50j)
        >>> auto_convert('1e3')
        1000.0
        >>> auto_convert(1000)
        1000
        >>> auto_convert(1000.0)
        1000.0
        >>> auto_convert('False')
        False
    """
    if isinstance(text, str):
        if pre_delim and post_delim and \
                has_delim(text, pre_delim, post_delim):
            text = strip_delim(text, pre_delim, post_delim)
        val = None
        for cast in casts:
            try:
                val = cast(text)
            except (TypeError, ValueError):
                pass
            else:
                break
        if val is None:
            val = text
    else:
        val = text
    return val


# ======================================================================
def is_number(val):
    """
    Determine if a variable contains a number.

    Args:
        val (str): The variable to test.

    Returns:
        result (bool): True if the values can be converted, False otherwise.

    Examples:
        >>> is_number('<100.0>')
        False
        >>> is_number('100.0+50j')
        True
        >>> is_number('1e3')
        True
    """
    try:
        complex(val)
    except (TypeError, ValueError):
        result = False
    else:
        result = True
    return result


# ======================================================================
def order_of_magnitude(
        val,
        base=10,
        exp=1):
    """
    Determine the order of magnitude of a number.

    Args:
        val (Number): The input number.
        base (Number): The base for defining the magnitude order.
        exp (Number): The exp for defining the magnitude order.

    Returns:
        result (int): The order of magnitude according to `(base ** exp)`.

    Examples:
        >>> order_of_magnitude(10.0)
        1
        >>> order_of_magnitude(1.0)
        0
        >>> order_of_magnitude(0.1)
        -1
        >>> order_of_magnitude(0.0)
        0
        >>> order_of_magnitude(-0.0)
        0
        >>> order_of_magnitude(634.432)
        2
        >>> order_of_magnitude(1024, 2)
        10
        >>> order_of_magnitude(1234, 10, 3)
        1
        >>> order_of_magnitude(1234, 2, 10)
        1
        >>> all(order_of_magnitude(i) == 0 for i in range(10))
        True
    """
    return int(math.log(abs(val), base) // exp) if val != 0.0 else 0


# ======================================================================
def prefix_to_order(
        prefix,
        prefixes=freeze(SI_PREFIX_EXTRA['base10'])):
    """
    Get the order corresponding to a given prefix.

    This works chiefly with SI prefixes.

    Args:
        prefix (str): The prefix to inspect.
        prefixes (Mappable): The conversion table for the prefixes.
            This must have the form: (prefix, order).
            Multiple prefixes can point to the same order.

    Returns:
        order (int|float): The order associate to the prefix.

    Raises:
        IndexError: If no prefix is found in the given prefixes.

    Examples:
        >>> [prefix_to_order(x)
        ...     for x in ('', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')]
        [0, 3, 6, 9, 12, 15, 18, 21, 24]
        >>> [prefix_to_order(x)
        ...     for x in ('', 'm', 'µ', 'n', 'p', 'f', 'a', 'z', 'y')]
        [0, -3, -6, -9, -12, -15, -18, -21, -24]
        >>> [prefix_to_order(x) for x in ('u', 'µ', 'μ')]
        [-6, -6, -6]
        >>> all(prefix_to_order(order_to_prefix(i)) == i
        ...     for i in range(-24, 25, SI_ORDER_STEP))
        True

    See Also:
        - flyingcircus.order_to_prefix()
        - flyingcircus.prefix_to_factor()
        - flyingcircus.factor_to_prefix()
        - flyingcircus.order_to_factor()
        - flyingcircus.factor_to_order()
    """
    prefixes = dict(prefixes)
    if prefix in prefixes:
        return prefixes[prefix]
    else:
        raise IndexError(fmtm('Prefix `{prefix}` not found!'))


# ======================================================================
def order_to_prefix(
        order,
        prefixes=freeze(SI_PREFIX['base10'])):
    """
    Get the prefix corresponding to the order of magnitude.

    This works chiefly with SI prefixes.

    Args:
        order (int): The order of magnitude.
        prefixes (Mappable): The conversion table for the prefixes.
            This must have the form: (prefix, order). Must be 1-to-1.

    Returns:
        prefix (str): The prefix for the corresponding order.

    Raises:
        ValueError: If no prefix exists for the input order.

    Examples:
        >>> [order_to_prefix(i * SI_ORDER_STEP) for i in range(9)]
        ['', 'k', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
        >>> [order_to_prefix(-i * SI_ORDER_STEP) for i in range(9)]
        ['', 'm', 'μ', 'n', 'p', 'f', 'a', 'z', 'y']
        >>> order_to_prefix(10)
        Traceback (most recent call last):
            ...
        ValueError: Invalid order `10` for given prefixes.

    See Also:
        - flyingcircus.prefix_to_order()
        - flyingcircus.prefix_to_factor()
        - flyingcircus.factor_to_prefix()
        - flyingcircus.order_to_factor()
        - flyingcircus.factor_to_order()
    """
    reverted_prefixes = reverse_mapping(dict(prefixes))
    if order in reverted_prefixes:
        return reverted_prefixes[order]
    else:
        raise ValueError(fmtm('Invalid order `{order}` for given prefixes.'))


# ======================================================================
def order_to_factor(
        order,
        base=10):
    """
    Compute the factor from the order (for a given base).

    Args:
        order (int|float): The order.
        base (int|float): The base to use for the order-to-factor conversion.
            Must be strictly positive.

    Returns:
        factor (int|float): The factor.

    Examples:
        >>> [order_to_factor(i) for i in range(-5, 6)]
        [1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]
        >>> [order_to_factor(float(i)) for i in range(-5, 5)]
        [1e-05, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]

    See Also:
        - flyingcircus.prefix_to_order()
        - flyingcircus.order_to_prefix()
        - flyingcircus.prefix_to_factor()
        - flyingcircus.factor_to_prefix()
        - flyingcircus.factor_to_order()
    """
    return base ** order


# ======================================================================
def factor_to_order(
        factor,
        base=10):
    """
    Compute the order from the factor (for a given base).

    Args:
        factor (int|float): The factor.
        base (int|float): The base to use for the order-to-factor conversion.
            Must be strictly positive.

    Returns:
        order (int|float): The order.

    Examples:
        >>> [factor_to_order(x) for x in (1, 10, 1000, 100000)]
        [0, 1, 3, 5]
        >>> [factor_to_order(x) for x in (1.0, 100.0, 10000.0, 123.45)]
        [0, 2, 4, 2]

    See Also:
        - flyingcircus.prefix_to_order()
        - flyingcircus.order_to_prefix()
        - flyingcircus.prefix_to_factor()
        - flyingcircus.factor_to_prefix()
        - flyingcircus.order_to_factor()
    """
    return int(round(math.log(factor, base)))


# ======================================================================
def prefix_to_factor(
        prefix,
        prefixes=freeze(SI_PREFIX_EXTRA['base10']),
        base=10):
    """
    Get the order corresponding to a given prefix.

    This works chiefly with SI prefixes.

    Args:
        prefix (str): The prefix to inspect.
        prefixes (Mappable): The conversion table for the prefixes.
            This must have the form: (prefix, order).
            Multiple prefixes can point to the same order.
        base (int|float): The base to use for the factor.

    Returns:
        factor (int|float): The factor associate to the prefix.

    Examples:
        >>> [prefix_to_factor(x) for x in ('', 'k', 'M', 'G', 'T', 'P')]
        [1, 1000, 1000000, 1000000000, 1000000000000, 1000000000000000]
        >>> [prefix_to_factor(x)
        ...     for x in ('', 'm', 'µ', 'n', 'p', 'f', 'a', 'z', 'y')]
        [1, 0.001, 1e-06, 1e-09, 1e-12, 1e-15, 1e-18, 1e-21, 1e-24]
        >>> [prefix_to_factor(x) for x in ('u', 'µ', 'μ')]
        [1e-06, 1e-06, 1e-06]
        >>> all(prefix_to_order(order_to_prefix(i)) == i
        ...     for i in range(-24, 25, SI_ORDER_STEP))
        True

    See Also:
        - flyingcircus.prefix_to_order()
        - flyingcircus.order_to_prefix()
        - flyingcircus.factor_to_prefix()
        - flyingcircus.order_to_factor()
        - flyingcircus.factor_to_order()
    """
    return order_to_factor(prefix_to_order(prefix, prefixes), base)


# ======================================================================
def factor_to_prefix(
        factor,
        prefixes=freeze(SI_PREFIX['base10']),
        base=10):
    """
    Get the order corresponding to a given prefix.

    This works chiefly with SI prefixes.

    Args:
        factor (int|float): The factor to inspect.
        prefixes (Mappable): The conversion table for the prefixes.
            This must have the form: (prefix, order). Must be 1-to-1.
        base (int|float): The base to use for the order-to-factor conversion.
            Must be strictly positive.

    Returns:
        prefix (str): The prefix for the corresponding factor.

    Examples:
        >>> factor_to_prefix(1e12)
        'T'
        >>> [factor_to_prefix(10 ** i) for i in range(-3, 4)]
        ['m', 'c', 'd', '', 'da', 'h', 'k']

    See Also:
        - flyingcircus.prefix_to_order()
        - flyingcircus.order_to_prefix()
        - flyingcircus.prefix_to_factor()
        - flyingcircus.order_to_factor()
        - flyingcircus.factor_to_order()
    """
    return order_to_prefix(factor_to_order(factor, base), prefixes)


# ======================================================================
def scale_to_order(
        val,
        base=10,
        exp=1,
        order=None):
    """
    Scale a number according to its order of magnitude.

    Args:
        val (Number): The input number.
        base (Number): The base for defining the magnitude order.
        exp (Number): The exp for defining the magnitude order.
        order (int|None): The order of magnitude.
            If None, this is computed from `base` and `exp`.

    Returns:
        result (Number): The scaled value.

    Examples:
        >>> round(scale_to_order(1234.123), 6)
        1.234123
        >>> round(scale_to_order(1234.123, 10, 1, 2), 6)
        12.34123
        >>> round(scale_to_order(1234.123, 10, 3, 2), 6)
        0.001234
        >>> round(scale_to_order(0.001234, 10, 3, -2), 6)
        1234.0

        >>> n = 12345.67890
        >>> all(
        ...     round(scale_to_order(scale_to_order(n, order=i), order=-i), 5)
        ...     == n
        ...     for i in range(10))
        True
    """
    if order is None:
        order = order_of_magnitude(val, base, exp)
    return val / (base ** exp) ** order


# ======================================================================
def guess_decimals(
        val,
        n_max=16,
        base=10,
        fp=16):
    """
    Guess the number of decimals in a given float number.

    Args:
        val (float): The input value.
        n_max (int): Maximum number of guessed decimals.
        base (int): The base used for the number representation.
        fp (int): The floating point maximum precision.
            A number with precision is approximated by the underlying platform.
            The default value corresponds to the limit of the IEEE-754 floating
            point arithmetic, i.e. 53 bits of precision: log10(2 ** 53) = 16
            approximately. This value should not be changed unless the
            underlying platform follows a different floating point arithmetic.

    Returns:
        prec (int): the guessed number of decimals.

    Examples:
        >>> guess_decimals(10)
        0
        >>> guess_decimals(1)
        0
        >>> guess_decimals(0.1)
        1
        >>> guess_decimals(0.01)
        2
        >>> guess_decimals(10.01)
        2
        >>> guess_decimals(0.000001)
        6
        >>> guess_decimals(-0.72)
        2
        >>> guess_decimals(0.9567)
        4
        >>> guess_decimals(0.12345678)
        8
        >>> guess_decimals(0.9999999999999)
        13
        >>> guess_decimals(0.1234567890123456)
        16
        >>> guess_decimals(0.9999999999999999)
        16
        >>> guess_decimals(0.1234567890123456, 6)
        6
        >>> guess_decimals(0.54235, 10)
        5
        >>> guess_decimals(0x654321 / 0x10000, 16, 16)
        4
    """
    offset = 2
    prec = 0
    fp -= math.ceil(math.log10(abs(val)))
    tol = 10 ** -fp
    x = (val - int(val)) * base
    while base - abs(x) > tol and abs(x % tol) < tol < abs(x) and prec < n_max:
        x = (x - int(x)) * base
        tol = 10 ** -(fp - prec - offset)
        prec += 1
    return prec


# ======================================================================
def significant_figures(
        val,
        num,
        keep_zeros=4):
    """
    Format a number with the correct number of significant figures.

    Args:
        val (str|float|int): The numeric value to be correctly formatted.
        num (str|int): The number of significant figures to be displayed.
        keep_zeros (int): The number of zeros to keep after the figures.
            This is useful for preventing the use of the scientific notation.

    Returns:
        val (str): String containing the properly formatted number.

    Examples:
        >>> significant_figures(1.2345, 1)
        '1'
        >>> significant_figures(1.2345, 4)
        '1.234'
        >>> significant_figures(1.234e3, 2)
        '1200'
        >>> significant_figures(-1.234e3, 3)
        '-1230'
        >>> significant_figures(12345678, 4)
        '12350000'
        >>> significant_figures(1234567890, 4)
        '1.235e+9'
        >>> significant_figures(-0.1234, 1)
        '-0.1'
        >>> significant_figures(0.0001, 2)
        '1.0e-4'

    See Also:
        The 'decimal' Python standard module.
    """
    val = float(val)
    num = int(num)
    order = order_of_magnitude(val, 10, 1)
    prec = num - order - 1
    ofm = ''
    val = round(val, prec)
    if abs(prec) > keep_zeros:
        val = val * 10 ** (-order)
        prec = num - 1
        ofm = 'e{:+d}'.format(order)
    elif prec < 0:
        prec = 0
    # print('val={}, num={}, ord={}, prec={}, ofm={}'.format(
    #     val, num, order, prec, ofm))  # DEBUG
    val = '{val:.{prec}f}{ofm}'.format(val=val, prec=prec, ofm=ofm)
    return val


# ======================================================================
def format_value_error(
        val,
        err,
        num=2,
        keep_zeros=4):
    """
    Outputs correct value/error pairs formatting.

    Args:
        val (str|float|int): The numeric value to be correctly formatted.
        err (str|float|int): The numeric error to be correctly formatted.
        num (str|int): The precision to be used for the error (usually 1 or 2).
        keep_zeros (int): The number of zeros to keep after the figures.
            This is useful for preventing the use of the scientific notation.

    Returns:
        val_str (str): The string with the correctly formatted numeric value.
        err_str (str): The string with the correctly formatted numeric error.

    Examples:
        >>> format_value_error(1234.5, 6.7)
        ('1234.5', '6.7')
        >>> format_value_error(123.45, 6.7, 1)
        ('123', '7')
        >>> format_value_error(12345.6, 7.89, 2)
        ('12345.6', '7.9')
        >>> format_value_error(12345.6, 78.9, 2)
        ('12346', '79')
        >>> format_value_error(12345.6, 0)
        ('12345.6', '0.0')
        >>> format_value_error(12345.6, 0, 0)
        ('12346', '0')
        >>> format_value_error(12345.6, 67)
        ('12346', '67')
        >>> format_value_error(12345.6, 670)
        ('12350', '670')
        >>> format_value_error(1234567890.0, 123456.0)
        ('1234570000', '120000')
        >>> format_value_error(1234567890.0, 1234567.0)
        ('1.2346e+9', '1.2e+6')
        >>> format_value_error(-0.470, 1.722)
        ('-0.5', '1.7')
        >>> format_value_error(0.0025, 0.0001)
        ('2.50e-3', '1.0e-4')
    """
    val = float(val)
    err = float(err)
    num = int(num) if num != 0 else 1
    val_order = order_of_magnitude(val, 10, 1)
    err_order = order_of_magnitude(err, 10, 1)
    try:
        # print('val_order={}, err_order={}, num={}'.format(
        #     val_order, err_order, num))  # DEBUG
        val_str = significant_figures(
            val, val_order - err_order + num, keep_zeros)
        err_str = significant_figures(
            err, num, keep_zeros)
    except ValueError:
        val_str = str(val)
        err_str = str(err)
    return val_str, err_str


# ======================================================================
def parse_units_prefix(
        text,
        force_si=True,
        check_si=True):
    """
    Extract factor and base units from units.

    Supports all base SI units, a number of non-base SI units and some
    non-SI units.

    Args:
        text (str):
        force_si:
        check_si:

    Returns:
        result (tuple):
    """
    raise NotImplementedError


# ======================================================================
def i_progression(
        name,
        start,
        step,
        num=None):
    """
    Compute the n-th term of a notable progression.

    Args:
        name (str): The name of the progression.
            Accepted values are:
             - `a`, `arithmetic`: Use arithmetic progression.
             - `g`, `geometric`: Use geometric progression.
             - `h`, `harmonic`: Use harmonic progression.
        start (Number): The initial value.
        step (Number): The step value.
        num (int|None): The number of values to yield.
            If None, yields values indefinitely (use with care!).

    Yields:
        result (Number): The n-th term of the progression.

    Raises:
        ValueError: If `name` is not supported.

    Examples:
        >>> list(i_progression('a', 10, 2, 12))
        [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
        >>> list(i_progression('g', 10, 2, 12))
        [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480]
        >>> [round(x, 3) for x in i_progression('h', 1, 1, 10)]
        [1.0, 0.5, 0.333, 0.25, 0.2, 0.167, 0.143, 0.125, 0.111, 0.1]
        >>> list(i_progression('ga', 10, 2, 12))
        Traceback (most recent call last):
            ...
        ValueError: Unknown progression `ga`

    References:
        - https://en.wikipedia.org/wiki/Arithmetic_progression
        - https://en.wikipedia.org/wiki/Geometric_progression
        - https://en.wikipedia.org/wiki/Harmonic_progression_(mathematics)
    """
    name = name.lower()
    if name in ('a', 'arithmetic'):
        i = 0
        while i < num or num is None:
            yield start
            start += step
            i += 1
    elif name in ('g', 'geometric'):
        i = 0
        while i < num or num is None:
            yield start
            start *= step
            i += 1
    elif name in ('h', 'harmonic'):
        i = 0
        while i < num or num is None:
            yield 1 / start
            start += step
            i += 1
    else:
        raise ValueError(fmtm('Unknown progression `{name}`'))


# ======================================================================
def progression(
        name,
        start,
        step,
        num):
    """
    Compute the n-th term of a notable progression.

    Args:
        name (str): The name of the progression.
            Accepted values are:
             - `a`, `arithmetic`: Use arithmetic progression.
             - `g`, `geometric`: Use geometric progression.
             - `h`, `harmonic`: Use harmonic progression.
        start (Number): The initial value.
        step (Number): The step value.
        num (int): The ordinal of the progression.
            The initial value is the 0th.

    Returns:
        result (Number): The n-th term of the progression.

    Raises:
        ValueError: If `name` is not supported.

    Examples:
        >>> progression('a', 0, 6, 7)
        42
        >>> progression('g', 1, 2, 10)
        1024
        >>> progression('h', 1, 1, 99)
        0.01
        >>> progression('ga', 10, 2, 12)
        Traceback (most recent call last):
            ...
        ValueError: Unknown progression `ga`

    References:
        - https://en.wikipedia.org/wiki/Arithmetic_progression
        - https://en.wikipedia.org/wiki/Geometric_progression
        - https://en.wikipedia.org/wiki/Harmonic_progression_(mathematics)
    """
    name = name.lower()
    if name in ('a', 'arithmetic'):
        return start + step * num
    elif name in ('g', 'geometric'):
        return start * step ** num
    elif name in ('h', 'harmonic'):
        return 1 / (start + step * num)
    else:
        raise ValueError(fmtm('Unknown progression `{name}`'))


# ======================================================================
def sum_progression(name, start, step, num=None):
    """
    Compute the sum of the first n terms of a notable progression.

    The sum of the elements of a progression is also called a series.

    Args:
        name (str): The name of the progression.
            Accepted values are:
             - `a`, `arithmetic`: Use arithmetic progression.
             - `g`, `geometric`: Use geometric progression.
             - `h`, `harmonic`: Use harmonic progression.
        start (Number): The initial value.
        step (Number): The step value.
        num (int|None): The number of values to sum.
            If None, sums to infinity.

    Returns:
        result (Number|None): The sum value.
            If the series diverges, returns None.

    Examples:
        >>> sum_progression('a', 10, 2, 12)
        252
        >>> sum_progression('g', 10, 2, 12)
        40950.0
        >>> round(sum_progression('h', 1, 1, 10), 3)
        2.929

        >>> print(sum_progression('a', 10, 2, None))
        None
        >>> print(sum_progression('g', 10, 0.5, None))
        20.0
        >>> print(sum_progression('h', 10, 0.5, None))
        None

        >>> sum_progression('ga', 10, 2, 12)
        Traceback (most recent call last):
            ...
        ValueError: Unknown progression `ga`
        >>> print(sum_progression('ga', 10, 2, None))
        Traceback (most recent call last):
            ...
        ValueError: Unknown progression `ga`

        >>> a, d, n = 10, 2, 20
        >>> all(
        ...     sum(i_progression(s, a, d, n)) == sum_progression(s, a, d, n)
        ...     for s in ('a', 'g', 'h'))
        True

    References:
        - https://en.wikipedia.org/wiki/Series_(mathematics)
        - https://en.wikipedia.org/wiki/Arithmetic_progression
        - https://en.wikipedia.org/wiki/Geometric_progression
        - https://en.wikipedia.org/wiki/Harmonic_progression_(mathematics)
    """
    name = name.lower()
    if name in ('a', 'arithmetic'):
        if num is not None:
            return num * start + (num * (num - 1) // 2) * step
        else:
            return num
    elif name in ('g', 'geometric'):
        if num is not None:
            if step == 1:
                return start * num
            else:
                return start * (1 - step ** num) / (1 - step)
        elif abs(step) < 1:
            return start / (1 - step)
        else:
            return num
    else:
        if num is not None:
            return sum(i_progression(name, start, step, num))
        else:
            progression(name, start, step, 1)  # test if progression is valid
            return num


# ======================================================================
def guess_numerical_sequence(
        seq,
        rounding=3):
    """
    Guess a compact expression for a numerical sequence.

    Args:
        seq (Sequence[Number]): The input items.
        rounding (int|None): The maximum number of decimals to show.

    Returns:
        result (str): The compact expression.
            Supported numerical sequences are:
             - constant sequences: '[val] * len(items)'
             - linear sequences: 'range(start, stop, step)'
               Note that both float and complex number will be detected
               (contrarily to Python's `range()`).
             - geometric sequences: 'base ** range(start, stop, step)'

    Examples:
        >>> items = [1.0] * 10
        >>> print(items)
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        >>> print(guess_numerical_sequence(items))
        [1.0] * 10

        >>> items = list(range(10))
        >>> print(items)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> print(guess_numerical_sequence(items))
        range(0, 10, 1)

        >>> items = list(range(5, 25, 2))
        >>> print(items)
        [5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        >>> print(guess_numerical_sequence(items))
        range(5, 25, 2)

        >>> items = [x / 1000 for x in range(5, 25, 2)]
        >>> print(items)
        [0.005, 0.007, 0.009, 0.011, 0.013, 0.015, 0.017, 0.019, 0.021, 0.023]
        >>> print(guess_numerical_sequence(items))
        range(0.005, 0.025, 0.002)

        >>> items = [2 ** x for x in range(5, 25, 2)]
        >>> print(items)
        [32, 128, 512, 2048, 8192, 32768, 131072, 524288, 2097152, 8388608]
        >>> print(guess_numerical_sequence(items))
        2.0 ** range(5.0, 24.0, 2)

        >>> items = [10 ** x for x in range(8)]
        >>> print(items)
        [1, 10, 100, 1000, 10000, 100000, 1000000, 10000000]
        >>> print(guess_numerical_sequence(items))
        10.0 ** range(0.0, 8.0, 1)

        >>> items = [3.5 ** x for x in range(5, 13, 2)]
        >>> print(items)
        [525.21875, 6433.9296875, 78815.638671875, 965491.5737304688]
        >>> print(guess_numerical_sequence(items))
        3.5 ** range(5.0, 12.0, 2)

        >>> items = [4.1 ** x for x in range(4, 11, 3)]
        >>> print(items)
        [282.5760999999999, 19475.42738809999, 1342265.9310152389]
        >>> print(guess_numerical_sequence(items))
        4.1 ** range(4.0, 11.0, 3)

        >>> items = [4.1 ** (x / 10) for x in range(4, 11, 3)]
        >>> # 4.1 ** range(0.4, 1.1, 0.3)
        >>> print(items)
        [1.7583832685816119, 2.6850272626520213, 4.1]
        >>> print(guess_numerical_sequence(items))
        1.527 ** range(1.333, 4.333, 1)
    """
    tol = 10 ** -min(guess_decimals(item) for item in seq if item)
    result = None
    diffs = tuple(diff(seq))
    base = diffs[0]
    if all(x - base < tol for x in diffs):
        if base < tol:
            # : constant sequence
            result = '[{}] * {}'.format(round(seq[0], rounding), len(seq))
        else:
            # : linear sequence
            result = 'range({}, {}, {})'.format(
                round(seq[0], rounding), round(seq[-1] + base, rounding),
                round(base, rounding))
    else:
        divs = tuple(div(seq))
        base = divs[0]
        if all(x - base < tol for x in divs):
            # find optimal base (least number of decimals)
            bases, firsts, lasts, steps = [], [], [], []
            i = 0
            step = 1
            new_base = 2
            min_sum_decimals = 4 * 16
            min_i = 0
            while new_base >= 2.0:
                new_base = base ** (1 / step)
                bases.append(new_base)
                firsts.append(math.log2(seq[0]) / math.log2(new_base))
                lasts.append(
                    math.log2(seq[-1] * new_base) / math.log2(new_base))
                steps.append(step)
                sum_decimals = sum(
                    [guess_decimals(x) if x else 0
                     for x in (bases[i], firsts[i], lasts[i], steps[i])])
                if sum_decimals < min_sum_decimals:
                    min_sum_decimals = sum_decimals
                    min_i = i
                step += 1
                i += 1
            # geometrical sequence
            new_base, first_item, last_item, step = \
                list(zip(bases, firsts, lasts, steps))[min_i]
            result = '{} ** range({}, {}, {})'.format(
                round(new_base, rounding), round(first_item, rounding),
                round(last_item, rounding), round(step, rounding))
    return result


# ======================================================================
def elide(
        seq,
        length=79,
        ending='..',
        sep=None,
        start=0):
    """
    Cut and append an ellipsis sequence at the end if length is excessive.

    This is especially useful for strings.

    Args:
        seq (Sequence): The input sequence.
        length (int): The maximum allowed length
        ending (Sequence|None): The ending to append.
            If None, it is ignored.
        sep (Any|None): The separator to use.
            If None, it is ignored, otherwise cuts only at the separator.
        start (int): The start index for the output.

    Returns:
        result (Sequence): The sequence with the elision.

    Examples:
        >>> elide('Flying Circus FlyingCircus', 24)
        'Flying Circus FlyingCi..'
        >>> elide('Flying Circus FlyingCircus', 24, '...')
        'Flying Circus FlyingC...'
        >>> elide('Flying Circus FlyingCircus', 24, '...', ' ')
        'Flying Circus...'
        >>> elide('Flying Circus FlyingCircus', 24, None)
        'Flying Circus FlyingCirc'
        >>> elide('Flying Circus FlyingCircus', 24, '..', None, 1)
        'lying Circus FlyingCir..'
        >>> elide('Flying Circus FlyingCircus', 1)
        '.'
        >>> elide('Flying Circus FlyingCircus', 2)
        '..'
        >>> elide('Flying Circus FlyingCircus', 3)
        'F..'
        >>> elide('Flying Circus FlyingCircus', 5, '..', ' ')
        '..'
        >>> elide('a bc def ghij klmno pqrstu vwxyzab', 16, '..', ' ')
        'a bc def ghij..'
        >>> elide('a bc def ghij klmno pqrstu vwxyzab', 5, '..', ' ')
        'a..'
        >>> elide('a bc def ghij klmno pqrstu vwxyzab', 6, '..', ' ')
        'a bc..'

        >>> elide(list(range(100)), 16, [-1])
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -1]
        >>> elide(list(range(100)), 16, [-1, -1])
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, -1, -1]
        >>> elide([x % 3 for x in range(100)], 16, [-1, -1], 0)
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, -1, -1]
        >>> elide(list(range(100)), 16, None)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        >>> elide(list(range(100)), 16, [-1, -1], None, 1)
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, -1, -1]
    """
    if len(seq) > length:
        len_ending = len(ending) if ending is not None else 0
        if sep is not None:
            try:
                limit = length - len_ending \
                        - seq[length - len_ending::-1].index(sep)
            except ValueError:
                limit = 0
        else:
            limit = max(length - len_ending, 0)
        if ending is not None:
            return seq[start:start + limit] + ending[:length]
        else:
            return seq[start:start + limit]
    else:
        return seq


# ======================================================================
def str2dict(
        in_str,
        entry_sep=',',
        key_val_sep='=',
        pre_delim='{',
        post_delim='}',
        strip_key_str=None,
        strip_val_str=None,
        convert=True):
    """
    Convert a string to a dictionary.

    Escaping and quotes are not supported.
    Dictionary name is always a string.

    Args:
        in_str (str): The input string.
        entry_sep (str): The entry separator.
        key_val_sep (str): The key-value separator.
        pre_delim (str): initial decorator (to be removed before parsing).
        post_delim (str): final decorator (to be removed before parsing).
        strip_key_str (str): Chars to be stripped from both ends of the key.
            If None, whitespaces are stripped. Empty string for no stripping.
        strip_val_str (str): Chars to be stripped from both ends of the value.
            If None, whitespaces are stripped. Empty string for no stripping.
        convert (bool): Enable automatic conversion of string to numeric.

    Returns:
        out_dict (dict): The output dictionary generated from the string.

    Examples:
        >>> d = str2dict('{a=10,b=20,c=test}')
        >>> for k in sorted(d.keys()): print(k, ':', d[k])  # display dict
        a : 10
        b : 20
        c : test

    See Also:
        dict2str
    """
    if has_delim(in_str, pre_delim, post_delim):
        in_str = strip_delim(in_str, pre_delim, post_delim)
    entries = in_str.split(entry_sep)
    out_dict = {}
    for entry in entries:
        # fetch entry
        key_val = entry.split(key_val_sep)
        # parse entry
        if len(key_val) == 1:
            key, val = key_val[0], None
        elif len(key_val) == 2:
            key, val = key_val
            val = val.strip(strip_val_str)
        elif len(key_val) > 2:
            key, val = key_val[0], key_val[1:]
            val = [tmp_val.strip(strip_val_str) for tmp_val in val]
        else:
            key = val = None
        # strip dict key
        key = key.strip(strip_key_str)
        # add to dictionary
        if key is not None:
            if convert:
                val = auto_convert(val)
            out_dict[key] = val
    return out_dict


# ======================================================================
def dict2str(
        in_dict,
        entry_sep=',',
        key_val_sep='=',
        pre_delim='{',
        post_delim='}',
        strip_key_str=None,
        strip_val_str=None,
        sorting=None):
    """
    Convert a dictionary to a string.

    Args:
        in_dict (dict): The input dictionary.
        entry_sep (str): The entry separator.
        key_val_sep (str): The key-value separator.
        pre_delim (str): initial decorator (to be appended to the output).
        post_delim (str): final decorator (to be appended to the output).
        strip_key_str (str): Chars to be stripped from both ends of the key.
            If None, whitespaces are stripped. Empty string for no stripping.
        strip_val_str (str): Chars to be stripped from both ends of the value.
            If None, whitespaces are stripped. Empty string for no stripping.
        sorting (callable): Sorting function passed to 'sorted' via `key` arg.
            Used for sorting the dictionary keys.

    Returns:
        out_str (str): The output string generated from the dictionary.

    Examples:
        >>> dict2str({'a': 10, 'b': 20, 'c': 'test'})
        '{a=10,b=20,c=test}'

    See Also:
        str2dict
    """
    keys = sorted(in_dict.keys(), key=sorting)
    out_list = []
    for key in keys:
        key = key.strip(strip_key_str)
        val = str(in_dict[key]).strip(strip_val_str)
        out_list.append(key_val_sep.join([key, val]))
    out_str = pre_delim + entry_sep.join(out_list) + post_delim
    return out_str


# ======================================================================
def string_between(
        text,
        begin_str,
        end_str,
        incl_begin=False,
        incl_end=False,
        greedy=True):
    """
    Isolate the string contained between two tokens

    Args:
        text (str): String to parse
        begin_str (str): Token at the beginning
        end_str (str): Token at the ending
        incl_begin (bool): Include 'begin_string' in the result
        incl_end (bool): Include 'end_str' in the result.
        greedy (bool): Output the largest possible string.

    Returns:
        text (str): The string contained between the specified tokens (if any)

    Examples:
        >>> string_between('roses are red violets are blue', 'ses', 'lets')
        ' are red vio'
        >>> string_between('roses are red, or not?', 'a', 'd')
        're re'
        >>> string_between('roses are red, or not?', ' ', ' ')
        'are red, or'
        >>> string_between('roses are red, or not?', ' ', ' ', greedy=False)
        'are'
        >>> string_between('roses are red, or not?', 'r', 'r')
        'oses are red, o'
        >>> string_between('roses are red, or not?', 'r', 'r', greedy=False)
        'oses a'
        >>> string_between('roses are red, or not?', 'r', 's', True, False)
        'rose'
        >>> string_between('roses are red violets are blue', 'x', 'y')
        ''
    """
    incl_begin = len(begin_str) if not incl_begin else 0
    incl_end = len(end_str) if incl_end else 0
    if begin_str in text and end_str in text:
        if greedy:
            begin = text.find(begin_str) + incl_begin
            end = text.rfind(end_str) + incl_end
        else:
            begin = text.find(begin_str) + incl_begin
            end = text[begin:].find(end_str) + incl_end + begin
        text = text[begin:end]
    else:
        text = ''
    return text


# ======================================================================
def remove_ansi_escapes(text):
    """
    Remove ANSI escape sequences from text.

    Args:
        text (str): The input text.

    Returns:
        result (str): The output text.

    Examples:
        >>> s = '\u001b[0;35mfoo\u001b[0m \u001b[0;36mbar\u001b[0m'
        >>> print(repr(remove_ansi_escapes(s)))
        'foo bar'
        >>> remove_ansi_escapes(s) == 'foo bar'
        True
    """
    ansi_escape = re.compile(r'(\x9B|\x1B\[)[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)


# ======================================================================
def check_redo(
        in_filepaths,
        out_filepaths,
        force=False,
        verbose=D_VERB_LVL,
        makedirs=False,
        no_empty_input=False):
    """
    Check if input files are newer than output files, to force calculation.

    Args:
        in_filepaths (str|Iterable[str]|None): Input filepaths for computation.
        out_filepaths (str|Iterable[str]): Output filepaths for computation.
        force (bool): Force computation to be re-done.
        verbose (int): Set level of verbosity.
        makedirs (bool): Create output dirpaths if not existing.
        no_empty_input (bool): Check if the input filepath list is empty.

    Returns:
        force (bool): True if the computation is to be re-done.

    Raises:
        IndexError: If the input filepath list is empty.
            Only if `no_empty_input` is True.
        IOError: If any of the input files do not exist.
    """
    # generate singleton list from str argument
    if isinstance(in_filepaths, str):
        in_filepaths = [in_filepaths]
    if isinstance(out_filepaths, str):
        out_filepaths = [out_filepaths]

    # check if output exists
    if not force:
        for out_filepath in out_filepaths:
            if out_filepath and not os.path.exists(out_filepath):
                force = True
                break

    # create output directories
    if force and makedirs:
        for out_filepath in out_filepaths:
            out_dirpath = os.path.dirname(out_filepath)
            if not os.path.isdir(out_dirpath):
                msg('mkdir: {}'.format(out_dirpath),
                    verbose, VERB_LVL['highest'])
                os.makedirs(out_dirpath)

    # check if input is older than output
    if not force:
        # check if input is not empty
        if in_filepaths:
            # check if input exists
            for in_filepath in in_filepaths:
                if not os.path.exists(in_filepath):
                    raise IOError('Input file does not exists.')

            for in_filepath, out_filepath in \
                    itertools.product(in_filepaths, out_filepaths):
                if os.path.getmtime(in_filepath) > os.path.getmtime(
                        out_filepath):
                    force = True
                    break
        elif no_empty_input:
            raise IOError('Input file list is empty.')

    if force:
        msg('Calc: {}'.format(out_filepaths), verbose, VERB_LVL['higher'])
        msg('From: {}'.format(in_filepaths), verbose, VERB_LVL['highest'])
    else:
        msg('Skip: {}'.format(out_filepaths), verbose, VERB_LVL['higher'])
        msg('From: {}'.format(in_filepaths), verbose, VERB_LVL['highest'])
    return force


# =====================================================================
def is_increasing(
        items,
        strict=True):
    """
    Check if items are increasing.

    Args:
        items (Iterable): The items to check.
        strict (bool): Check for strict monotonicity.
            If True, consecutive items cannot be equal.
            Otherwise they can be also equal.

    Returns:
        result (bool): True if items are increasing, False otherwise.

    Examples:
        >>> is_increasing([-20, -2, 1, 3, 5, 7, 8, 9])
        True
        >>> is_increasing([1, 3, 5, 7, 8, 9, 9, 10, 400])
        False
        >>> is_increasing([1, 3, 5, 7, 8, 9, 9, 10, 400], False)
        True
        >>> is_increasing([-20, -2, 1, 3, 5, 7, 8, 9])
        True
        >>> is_increasing([-2, -2, 1, 30, 5, 7, 8, 9])
        False
    """
    others = iter(items)
    next(others)
    if strict:
        result = all(x < y for x, y in zip(items, others))
    else:
        result = all(x <= y for x, y in zip(items, others))
    return result


# =====================================================================
def is_decreasing(
        items,
        strict=True):
    """
    Check if items are decreasing.

    Args:
        items (Iterable): The items to check.
        strict (bool): Check for strict monotonicity.
            If True, consecutive items cannot be equal.
            Otherwise they can be also equal.

    Returns:
        result (bool): True if items are decreasing, False otherwise.

    Examples:
        >>> is_decreasing([312, 54, 53, 7, 3, -5, -100])
        True
        >>> is_decreasing([312, 53, 53, 7, 3, -5, -100])
        False
        >>> is_decreasing([312, 53, 53, 7, 3, -5, -100], False)
        True
        >>> is_decreasing([312, 54, 53, 7, 3, -5, -100])
        True
        >>> is_decreasing([312, 5, 53, 7, 3, -5, -100])
        False
    """
    others = iter(items)
    next(others)
    if strict:
        result = all(x > y for x, y in zip(items, others))
    else:
        result = all(x >= y for x, y in zip(items, others))
    return result


# ======================================================================
def is_same_sign(items):
    """
    Determine if all items in an Iterable have the same sign.

    Args:
        items (Iterable): The items to check.
            The comparison operators '>=' and '<' with `0` must be defined
            for all items.

    Returns:
        same_sign (bool): The result of the comparison.
            True if the items are all positive or all negative.
            False otherwise, i.e. they have mixed signs.

    Examples:
        >>> is_same_sign((0, 1, 2 ,4))
        True
        >>> is_same_sign((-1, -2 , -4))
        True
        >>> is_same_sign((-1, 1))
        False
        >>> is_same_sign([])
        True
    """
    # : slower one-line alternative using `all_equal`
    # return all_equal(item >= 0 for item in items)
    iter_items = iter(items)
    try:
        first = next(iter_items) >= 0
    except StopIteration:
        return True
    # : slower alternative for the loop below
    # return all(first == (item >= 0) for item in items)
    for item in iter_items:
        if first != (item >= 0):
            return False
    return True


# ======================================================================
def is_percent(text):
    """
    Determine if the input string contains a percent value.

    Args:
        text (str): The input string.

    Returns:
        result (bool): The result of the check.
            True if the input contains a valid percentage.
            False otherwise.

    Examples:
        >>> print(is_percent('100%'))
        True
        >>> print(is_percent('0.5%'))
        True
        >>> print(is_percent('421.43%'))
        True
        >>> print(is_percent('-433%'))
        True
        >>> print(is_percent('421.%'))
        True
        >>> print(is_percent('.5%'))
        True
        >>> print(is_percent('1.e2%'))
        True
        >>> print(is_percent('421.43'))
        False
        >>> print(is_percent('ciao421.43%'))
        False
    """
    return \
        (isinstance(text, str)
         and re.match(r'[+-]?\d*(?:\.\d*)?(?:[eE][+-]?\d+)?%', text.strip())
         is not None)


# ======================================================================
def to_percent(text):
    """
    Convert the input string to a float value as percentage.

    Args:
        text (str): The input string.

    Returns:
        result (float|None): The percent value.
            If the input is invalid, returns None.

    Examples:
        >>> print(to_percent('100%'))
        1.0
        >>> print(to_percent('0.5%'))
        0.005
        >>> print(to_percent('421.43%'))
        4.2143
        >>> print(to_percent('-433%'))
        -4.33
        >>> print(to_percent('421.%'))
        4.21
        >>> print(to_percent('.1%'))
        0.001
        >>> print(to_percent('421.43'))
        None
        >>> print(to_percent('ciao421.43%'))
        None
    """
    match = re.match(r'[+-]?\d*(?:\.\d*)?(?:[eE][+-]?\d+)?%', text.strip())
    if match:
        return float(match.string[:-1]) / 100
    else:
        return None


# ======================================================================
def scale_to_int(
        val,
        scale,
        rounding=round):
    """
    Scale a float value by the specified size.

    Args:
        val (int|float): The value to scale.
            If int, the number is left untouched.
        scale (int|float): The scale size.
        rounding (callable): Rounding function.
            Sensible choices are: `math.floor()`, `math.ceil()`, `round()`.

    Returns:
        result (Any): The scaled value.

    Examples:
        >>> scale_to_int(0.1, 10)
        1
        >>> scale_to_int(0.1, 10.0)
        1
        >>> scale_to_int(1.0, 10.0)
        10
        >>> scale_to_int(0.5, 11.0)
        6
        >>> scale_to_int(0.5, 11.0, math.floor)
        5
        >>> scale_to_int(1, 10.0)
        1
        >>> scale_to_int(1, 10)
        1
    """
    return int(rounding(val * scale)) if not isinstance(val, int) else val


# ======================================================================
def multi_scale_to_int(
        vals,
        scales,
        shape=(None, 2),
        combine=None):
    """
    Ensure values scaling of multiple values.

    Args:
        vals (int|float|Sequence[int|float|Sequence]): The input value(s)
            If Sequence, a value for each scale must be specified.
            If not Sequence, all pairs will have the same value.
            If any value is int, it is not scaled further.
            If any value is float, it is scaled to the corresponding scale,
            if `combine` is None, otherwise it is scaled to a combined scale
            according to the result of `combine(scales)`.
        scales (Sequence[int]): The scale sizes for the pairs.
        shape (Sequence[int|None]): The shape of the output.
            It must be a 2-tuple of int or None.
            None entries are replaced by `len(scales)`.
        combine (callable|None): The function for combining pad width scales.
            Must accept: combine(Sequence[int]): int|float
            This is used to compute a reference scaling value for the
            float to int conversion, using `combine(scales)`.
            For the int values of `width`, this parameter has no effect.
            If None, uses the corresponding scale from the scales.

    Returns:
        vals (int|tuple[tuple[int]]): The scaled value(s).

    See Also:
        - flyingcircus.stretch()
        - flyingcircus.scale_to_int()

    Examples:
        >>> scales = (10, 20, 30)

        >>> multi_scale_to_int(0.1, scales)
        ((1, 1), (2, 2), (3, 3))
        >>> multi_scale_to_int(0.1, scales, combine=max)
        ((3, 3), (3, 3), (3, 3))
        >>> multi_scale_to_int(2, scales)
        ((2, 2), (2, 2), (2, 2))
        >>> multi_scale_to_int((1, 1, 2), scales)
        ((1, 1), (1, 1), (2, 2))
        >>> multi_scale_to_int((0.1, 1, 2), scales)
        ((1, 1), (1, 1), (2, 2))
        >>> multi_scale_to_int((0.1, 1, 2), scales, combine=max)
        ((3, 3), (1, 1), (2, 2))
        >>> multi_scale_to_int(((0.1, 0.5),), scales)
        ((1, 5), (2, 10), (3, 15))
        >>> multi_scale_to_int(((2, 3),), scales)
        ((2, 3), (2, 3), (2, 3))
        >>> multi_scale_to_int(((2, 3), (1, 2)), scales)
        Traceback (most recent call last):
            ...
        ValueError: Cannot stretch `((2, 3), (1, 2))` to `(3, 2)`.
        >>> multi_scale_to_int(((0.1, 0.2),), scales, combine=min)
        ((1, 2), (1, 2), (1, 2))
        >>> multi_scale_to_int(((0.1, 0.2),), scales, combine=max)
        ((3, 6), (3, 6), (3, 6))
        >>> multi_scale_to_int(((1, 2), (3, 4)), (2, 3))
        ((1, 2), (3, 4))
        >>> multi_scale_to_int(((1, 2), 3), (2, 3))
        ((1, 2), (3, 3))
        >>> multi_scale_to_int(((1, 2), 0.7), (2, 3))
        ((1, 2), (2, 2))
        >>> multi_scale_to_int(((1, 1),), (3,), (None, 2))
        ((1, 1),)
    """
    shape = tuple(x if x else len(scales) for x in shape)
    vals = stretch(vals, shape)
    if callable(combine):
        combined = combine(scales)
        result = tuple(
            tuple(scale_to_int(x, combined) for x in val)
            for val in vals)
    else:
        result = tuple(
            tuple(scale_to_int(x, scale) for x in val)
            for val, scale in zip(vals, scales))
    return result


# ======================================================================
def _format_summary(
        summary,
        template='',
        kws_limit=8,
        line_limit=79,
        more=False):
    """
    Format summary result from `flyingcircus.time_profile()`.

    Args:
        summary (dict): The input summary.
        template (str): The template to use
        kws_limit (int): The maximum number of chars to use for input preview.
            This is only effective if `more == True`.
        line_limit (int): Limit for single line.
        more (bool): Display more information.

    Returns:
        result (str): The formatted summary.

    See Also:
        - flyingcircus.time_profile()
        - flyingcircus.multi_benchmark()
    """
    val_order = order_of_magnitude(summary['val'], 10, SI_ORDER_STEP)
    err_order = order_of_magnitude(summary['err'], 10, SI_ORDER_STEP)
    loop_order = order_of_magnitude(summary['num'], 10, SI_ORDER_STEP)
    batch_order = order_of_magnitude(summary['batch'], 10, SI_ORDER_STEP)
    val, err = format_value_error(
        scale_to_order(summary['val'], 10, SI_ORDER_STEP, val_order),
        scale_to_order(summary['err'], 10, SI_ORDER_STEP, val_order),
        SI_ORDER_STEP, 6)
    t_units = order_to_prefix(val_order, SI_PREFIX['base1000'])
    loop_num = int(round(
        scale_to_order(summary['num'], 10, SI_ORDER_STEP, loop_order)))
    batch_num = int(round(
        scale_to_order(summary['batch'], 10, SI_ORDER_STEP, batch_order)))
    l_units = order_to_prefix(loop_order, SI_PREFIX['base1000'])
    b_units = order_to_prefix(batch_order, SI_PREFIX['base1000'])
    if more:
        kws_list = [
            elide(k + '=' + str(v), kws_limit)
            for k, v in summary['kws'].items()]
        args_list = [
            elide(str(arg), line_limit // 2) for arg in summary['args']]
        args = '(' + ', '.join(args_list + kws_list) + ')'
        if len(args) > line_limit // 2:
            args = '(' + '\n\t' + ',\n\t'.join(args_list + kws_list) + ')'
    else:
        args = '(..)'
    name = summary['func_name']
    name_s = fmtm('{name}{args}')
    time_s = fmtm('({val} ± {err}) {t_units}s')
    time_ss = fmtm('t = {time_s}')
    time_ls = fmtm('time = {time_s}')
    loop_s = fmtm('{loop_num}{l_units}')
    loop_ss = fmtm('l = {loop_s}')
    loop_ls = fmtm('loops = {loop_s}')
    batch_s = fmtm('{batch_num}{b_units}')
    batch_ss = fmtm('b = {batch_s}')
    batch_ls = fmtm('batch = {batch_s}')
    if template:
        result = fmtm(template)
    else:
        result = ': ' + '; '.join([name_s, time_ls, loop_ls, batch_ls])
    if line_limit > 0:
        result = '\n'.join(
            [elide(line, line_limit) for line in result.splitlines()])
    if not template and more:
        result += \
            '\n  ' + 'mean_time = ' + str(summary['mean']) + ';  ' \
            + 'stdev_time = ' + str(summary['stdev']) + ';\n  ' \
            + 'min_time = ' + str(summary['min']) + ';  ' \
            + 'max_time = ' + str(summary['max']) + ';\n  ' \
            + 'minmax_range = ' + str(summary['minmax_range'])
    return result


# ======================================================================
def estimate_timer_error(
        timer,
        num=2 ** 16):
    """
    Estimate the error associated to a specific timer.

    Args:
        timer (callable):
        num (int): The number of repetitions.

    Returns:
        result (float): The estimated timer error.

    Examples:
        >>> timers = time.perf_counter, time.process_time, time.time
        >>> timer_errors = [estimate_timer_error(timer) for timer in timers]
        >>> print([math.log10(t) < 0 for t in timer_errors])
        [True, True, True]
    """
    result = 0.0
    for i in range(num):
        begin_time = timer()
        end_time = timer()
        time_delta = abs(end_time - begin_time)
        result = next_mean(time_delta, result, i)
    return result


# ======================================================================
@parametric
def time_profile(
        func,
        timeout=16.0,
        batch_timeout=0.5,
        max_iter=2 ** 24,  # 16 M
        min_iter=8,
        max_batch=2 ** 20,  # 1 M
        min_batch=1,
        val_func=median,
        err_func=median_abs_dev,
        timer=time.perf_counter,
        use_gc=True,
        quick=True,
        text=': {name_s};  {time_ss};  {loop_ss};  {batch_ss}',
        fmtt=True,
        verbose=D_VERB_LVL):
    """
    Estimate the execution time of a function using multiple repetitions.

    The function is repeated in batches.
    Each batch repeats the function to ensure a reliable time measurement.
    The run time is computed as the the batch time divided by the batch size.

    This is similar to the functionality provided by the `timeit` module,
    but also works as a decorator.

    Note that the default values are choosen under the assumption that the
    timer error is much larger than the execution time of the function.

    Args:
        func (callable): The input function.
        timeout (int|float): Maximum time for testing in s.
            There will be at least `min_iter` iterations.
        batch_timeout (int|float): Maximum time for each batch in s.
            If quick is False, there will be at least `min_batch` repetitions.
        max_iter (int): Maximum number of iterations.
            Must be at least 2.
        min_iter (int): Minimum number of iterations.
            If the min number of iterations requires longer than `max_time`,
            the `max_time` limit is ignored until at least `min_iter`
            iterations are performed.
        max_batch (int): Maximum size of the batch.
            Must be at least 1.
        min_batch (int): Minimum size of the batch.
            Must be at least 1.
        val_func (callable|None): Compute timing value from batch times.
            If callable, must have the signature:
            func(Sequence[int|float]): int|float
            If None, uses the mean of the runtimes.
        err_func (callable|None): Compute timing error from batch times.
            If callable, must have the signature:
            func(Sequence[int|float]): int|float
            If None, uses the standard deviation of the mean of the runtimes.
        timer (callable): The function used to measure the timings.
        use_gc (bool): Use the garbage collection during the timing.
        quick (bool): Force exiting the repetition loops.
            If this is True, the `max_time` is forced within the execution
            time of a single instance.
            The minimum number of iterations is always performed.
        text (str): Text to use for printing output.
        fmtt (str|bool|None): Format of the printed output.
            This is passed to `flyingcircus.msg()`.
        verbose (int): Set level of verbosity.

    Returns:
        decorator_profile_time (callable): The decorator.

    Examples:
        >>> @time_profile(timeout=0.1, fmtt=False)
        ... def my_func(a, b):
        ...     return [0 for _ in range(a) for _ in range(b)]
        >>> x, summary = my_func(100, 100)  # doctest:+ELLIPSIS
        : my_func(..);  t = (... ± ...) ...s;  l = ...;  b = ...
        >>> x, summary = my_func(1000, 1000)  # doctest:+ELLIPSIS
        : my_func(..);  t = (... ± ...) ...s;  l = ...;  b = ...

        >>> def my_func(a, b):
        ...     return [0 for _ in range(a) for _ in range(b)]
        >>> my_func = time_profile(timeout=0.1)(my_func)
        >>> x, summary = my_func(100, 100)  # doctest:+ELLIPSIS
        : my_func(..);  t = (... ± ...) ...s;  l = ...;  b = ...
        >>> x, summary = my_func(1000, 1000)  # doctest:+ELLIPSIS
        : my_func(..);  t = (... ± ...) ...s;  l = ...;  b = ...

        >>> print(list(summary.keys()))
        ['result', 'func_name', 'args', 'kws', 'num', 'val', 'err', 'mean',\
 'sosd', 'var', 'stdev', 'min', 'max', 'median', 'medoid', 'batch']
 
    See Also:
        - flyingcircus.multi_benchmark()
        - flyingcircus._format_summary()
    """

    # ----------------------------------------------------------
    @functools.wraps(func)
    def wrapper(*_args, **_kws):
        gc_was_enabled = gc.isenabled()
        gc.enable() if use_gc else gc.disable()
        mean_time = sosd_time = min_time = max_time = 0.0
        init_time = timer()
        total_time = 0.0
        result = None
        collect_runtimes = callable(val_func) or callable(err_func)
        timer_err = estimate_timer_error(timer)
        b_timeout = batch_timeout if batch_timeout else timer_err * max_batch
        batch_err = timer_err
        if collect_runtimes:
            run_times = []
        else:
            sorted_run_times = []
        mean_batch = 0.0
        min_time = timeout
        max_time = 0.0
        for i in range(max_iter):
            j = 0
            batch_time = 0.0
            begin_time = timer()
            for j in range(max_batch):
                result = func(*_args, **_kws)
                end_time = timer()
                total_time = end_time - init_time
                batch_time = end_time - begin_time
                if total_time > timeout and (i >= min_iter or quick):
                    break
                if batch_time > b_timeout and (j >= min_batch or quick):
                    break
            mean_batch = next_mean(j + 1, mean_batch, i)
            run_time = (batch_time - batch_err) / (j + 1)
            if collect_runtimes:
                run_times.append(run_time)
            mean_time, sosd_time, _ = next_mean_and_sosd(
                run_time, mean_time, sosd_time, i)
            if not collect_runtimes:
                medoid_time, median_time, _ = next_medoid_and_median(
                    run_time, sorted_run_times)
            if run_time < min_time:
                min_time = run_time
            if run_time > max_time:
                max_time = run_time
            if total_time > timeout and i >= min_iter:
                break
        var_time, stdev_time = sosd2var(sosd_time, i), sosd2stdev(sosd_time, i)
        if collect_runtimes:
            median_time = median(run_times)
            medoid_time = medoid(run_times)
        val_time = mean_time
        err_time = stdev_time / i ** 0.5
        if collect_runtimes:
            if callable(val_func):
                val_time = val_func(run_times)
            if callable(err_func):
                err_time = err_func(run_times)
        summary = dict(
            result=result, func_name=func.__name__, args=_args, kws=_kws,
            num=i + 1, val=val_time, err=err_time,
            mean=mean_time, sosd=sosd_time,
            var=var_time, stdev=stdev_time, min=min_time, max=max_time,
            median=median_time, medoid=medoid_time,
            batch=int(mean_batch))
        gc.enable() if gc_was_enabled else gc.disable()
        if text:
            msg(_format_summary(summary, text), verbose, D_VERB_LVL, fmtt)
        return result, summary

    return wrapper


# ======================================================================
def multi_benchmark(
        funcs,
        argss=None,
        kwss=None,
        input_sizes=tuple(int(2 ** (2 + (3 * i) / 4)) for i in range(10)),
        gen_input=lambda n: [random.random() for _ in range(n)],
        equal_output=lambda a, b: a == b,
        time_prof_kws=None,
        store_all=False,
        text_funcs=':{lbl_s:<{len_lbls}s}  N={input_size!s:<{len_n}s} '
                   '{is_equal_s:>3s}  {time_s:<26s}  {loop_s:>5} * {batch_s}',
        text_inputs=' ',
        fmtt=True,
        verbose=D_VERB_LVL):
    """
    Benchmark multiple functions for varying input sizes.

    Sensible choices for input sizes are:

    - for linear (n) problems

      - (5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000)
      - tuple(10 * (i + 1) for i in range(24))
        # lin-spaced
      - tuple(2 ** (i + 1) for i in range(16))
        # geom-spaced
      - tuple(int(2 ** (2 + (3 * i) / 4)) for i in range(20))
        # frac-pow-spaced

    - for quadratic (n²) problems

      - (5, 10, 50, 100, 500, 1000, 5000)
      - tuple(10 * (i + 1) for i in range(24))
        # lin-space
      - tuple(2 ** (i + 1) for i in range(12))
        # geom-space
      - tuple(int(2 ** (2 + (3 * i) / 4)) for i in range(15))
        # frac-pow-spaced

    Args:
        funcs (Iterable[callable]): The functions to test.
            Must have the signature: func(input_, *args, **kws).
            `args` and `kws` are from `argss` and `kwss` respectively.
        argss (Iterable|None): The positional arguments for `funcs`.
            Each item correspond to an element of `funcs`.
        kwss (Iterable|None): The keyword arguments for `funcs`.
            Each item correspond to an element of `funcs`.
        input_sizes (Iterable[int]): The input sizes.
        gen_input (callable): The function used to generate the input.
            Must have the signature: gen_input(int): Any.
        equal_output (callable): The function used to compare the output.
            Must have the signature: gen_input(Any, Any): bool.
        time_prof_kws (Mappable|None): Keyword parameters for `time_profile`.
            These are passed to `flyingcircus.time_profile()`.
        store_all (bool): Store all results.
            If True, all results are stores.
        text_funcs (str): Text to use for printing output for functions.
        text_inputs (str): Text to use for printing output for inputs.
        fmtt (str|bool|None): Format of the printed output.
            This is passed to `flyingcircus.msg()`.
        verbose (int): Set level of verbosity.

    Returns:
        result (tuple): The tuple
            contains:
             - summaries (list[list[dict]]): The summary for each benchmark.
             - labels (list[str]): The function names.
             - results (list): The full results.
                This is non-empty only if `store_all` is True.

    Examples:
        >>> def f1(items):
        ...     return [item * item for item in items]
        >>> def f2(items):
        ...     return [item ** 2 for item in items]
        >>> def func_with_longer_name(items):
        ...     return [item ** 2 for item in items]
        >>> funcs = f1, f2, func_with_longer_name
        >>> summaries, labels, results = multi_benchmark(
        ...     funcs, input_sizes=tuple(10 ** (i + 1) for i in range(3)),
        ...     time_prof_kws=dict(timeout=0.1, batch_timeout=0.01),
        ...     fmtt=False)  # doctest:+ELLIPSIS
        N = (10, 100, 1000)
        <BLANKLINE>
        :f1()                     N=10    OK  (... ± ...) ...s  ... * ...
        :f2()                     N=10    OK  (... ± ...) ...s  ... * ...
        :func_with_longer_name()  N=10    OK  (... ± ...) ...s  ... * ...
        <BLANKLINE>
        :f1()                     N=100   OK  (... ± ...) ...s  ... * ...
        :f2()                     N=100   OK  (... ± ...) ...s  ... * ...
        :func_with_longer_name()  N=100   OK  (... ± ...) ...s  ... * ...
        <BLANKLINE>
        :f1()                     N=1000  OK  (... ± ...) ...s  ... * ...
        :f2()                     N=1000  OK  (... ± ...) ...s  ... * ...
        :func_with_longer_name()  N=1000  OK  (... ± ...) ...s  ... * ...
        >>> print(labels, results)
        ['f1', 'f2', 'func_with_longer_name'] []
        >>> summary_headers = list(summaries[0][0].keys())
        >>> print(summary_headers)
        ['result', 'func_name', 'args', 'kws', 'num', 'val', 'err', 'mean',\
 'sosd', 'var', 'stdev', 'min', 'max', 'median', 'medoid', 'batch',\
 'is_equal']

    See Also:
        - flyingcircus.time_profile()
        - flyingcircus._format_summary()
    """
    labels = [func.__name__ for func in funcs]
    if argss is None:
        argss = ()
    if kwss is None:
        kwss = ()
    time_prof_kws = dict(time_prof_kws) if time_prof_kws is not None else {}
    if 'verbose' not in time_prof_kws:
        time_prof_kws['verbose'] = VERB_LVL['none']
    if 'quick' not in time_prof_kws:
        time_prof_kws['quick'] = True
    len_n = max(map(lambda x: int(math.ceil(math.log10(x))), input_sizes)) + 1
    len_lbls = max(map(len, labels)) + len('()')
    msg(fmtm('N = {input_sizes}'), verbose, D_VERB_LVL, fmtt)
    summaries = []
    results = []
    for i, input_size in enumerate(input_sizes):
        inner_summaries = []
        input_data = gen_input(input_size)
        truth = None
        if text_inputs:
            msg(fmtm(text_inputs), verbose, D_VERB_LVL, fmtt)
        for j, (func, args, kws) in \
                enumerate(itertools.zip_longest(funcs, argss, kwss)):
            args = tuple(args) if args is not None else ()
            kws = dict(kws) if kws is not None else {}
            func = time_profile(**time_prof_kws)(func)
            result, summary = func(input_data, *args, **kws)
            if j == 0:
                truth = result
            is_equal = equal_output(truth, result)
            is_equal_s = 'OK' if is_equal else 'ERR'
            lbl_s = func.__name__ if hasattr(func, '__name__') else '<UNNAMED>'
            lbl_s += '()'
            if text_funcs:
                msg(_format_summary(summary, fmtm(text_funcs)),
                    verbose, D_VERB_LVL, fmtt)
            summary['is_equal'] = is_equal
            inner_summaries.append(summary)
            if store_all:
                results.append(result)
        summaries.append(inner_summaries)
    return summaries, labels, results


# ======================================================================
elapsed(os.path.basename(__file__))

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
