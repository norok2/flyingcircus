#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flyingcircus.util: generic basic utilities.
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
import time  # Time access and conversions
import itertools  # Functions creating iterators for efficient looping
import functools  # Higher-order functions and operations on callable objects
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
# import lzma  # Compression using the LZMA algorithm
# import json  # JSON encoder and decoder [JSON: JavaScript Object Notation]
import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
import struct  # Interpret strings as packed binary data
import re  # Regular expression operations
import fnmatch  # Unix filename pattern matching
import random  # Generate pseudo-random numbers
import hashlib  # Secure hashes and message digests
import base64  # Base16, Base32, Base64, Base85 Data Encodings
import pickle  # Python object serialization

# :: External Imports

# :: External Imports Submodules

# :: Local Imports
from flyingcircus import INFO, PATH
from flyingcircus import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from flyingcircus import elapsed, report
from flyingcircus import msg, dbg
from flyingcircus import do_nothing_decorator
from flyingcircus import HAS_JIT, jit

# ======================================================================
# :: Custom defined constants


# ======================================================================
# :: Default values usable in functions.
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
def span(
        first,
        second=None,
        step=None):
    """
    Span consecutive numbers in a range.

    This is useful to produce 1-based ranges, which start from 1 (if `start`
    is not specified) and include the `stop` element (if the `step` parameter
    allows).

    Args:
        first (int): The first value of the range.
            Must be non-negative.
            If `second == None` this is the `stop` value is included
            if `step` is a multiple of the length of the sequence.
            Otherwise, this is the start value and is included.
            If `first < second` the sequence is yielded backwards.
        second (int|None): The second value of the range.
            If None, the start value is 1.
            Otherwise, this is the stop value and is included
            if `step` is a multiple of the length of the sequence.
            If `first < second` the sequence is yielded backwards.
        step (int): The step of the rows range.
            If start > stop, the step parameter should be negative in order
            to obtain a non-empty range.
            If None, this is computed automatically based on `first` and
            `second`, such that a non-empty sequence is avoided, if possible.

    Returns:
        result (range): The spanned range.

    Examples:
        >>> print(list(span(10)))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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
    """
    if second is None:
        start, stop = 1, first
    else:
        start, stop = first, second
    if not step:
        step = 1 if start < stop else -1
    stop = stop + (step if (start - stop) % step == 0 else 0)
    return range(start, stop, step)


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
        in_file (str|file): The input file.
            If str, the file is open for reading (as binary).
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
    if isinstance(in_file, str):
        file_obj = open(in_file, 'rb')
    else:
        file_obj = in_file
    if offset is not None:
        file_obj.seek(offset, whence)
    fmt = mode + str(num_blocks) + DTYPE_STR[dtype]
    read_size = struct.calcsize(fmt)
    data = struct.unpack_from(fmt, file_obj.read(read_size))
    if isinstance(in_file, str):
        file_obj.close()
    return data


# ======================================================================
def read_cstr(
        in_file,
        offset=None,
        whence=io.SEEK_SET):
    """
    Read a C-type string from file.

    Args:
        in_file (str|file): The input file.
            If str, the file is open for reading (as binary).
        offset (int|None): The offset where to start reading.
        whence (int): Where to reference the offset.
            Accepted values are:
             - '0': absolute file positioning.
             - '1': seek relative to the current position.
             - '2': seek relative to the file's end.

    Returns:
        text (str): The string read.
    """
    if isinstance(in_file, str):
        file_obj = open(in_file, 'rb')
    else:
        file_obj = in_file
    if offset is not None:
        file_obj.seek(offset, whence)
    buffer = []
    while True:
        c = file_obj.read(1).decode('ascii')
        if c is None or c == '\0':
            break
        else:
            buffer.append(c)
    text = ''.join(buffer)
    if isinstance(in_file, str):
        file_obj.close()
    return text


# ======================================================================
def is_deep(
        obj,
        avoid=(str, bytes),
        max_depth=-1):
    """
    Determine if an object is deep, i.e. it can be iterated through.

    Args:
        obj (Any): The object to test.
        avoid (tuple|None): Data types to skip.
        max_depth (int): Maximum depth to reach. Negative for unlimited.

    Returns:
        result (bool): If the object is deep or not.

    Examples:
        >>> is_deep(1)
        False
        >>> is_deep(())
        True
        >>> is_deep([1, 2, 3])
        True
        >>> is_deep('ciao', avoid=None)
        True
        >>> is_deep('ciao')
        False
    """
    try:
        no_expand = avoid and isinstance(obj, avoid)
        if no_expand or max_depth == 0 or obj == next(iter(obj)):
            raise TypeError
    except TypeError:
        return False
    except StopIteration:
        return True
    else:
        return True


# ======================================================================
def nesting_level(
        obj,
        deep=True,
        avoid=(str, bytes),
        max_depth=-1,
        combine=max):
    """
    Compute the nesting level of nested iterables.

    Args:
        obj (Any): The object to test.
        deep (bool): Evaluate all item.
            If True, all elements within `obj` are evaluated.
            If False, only the first element of each deep object is evaluated.
            An object is considered deep using `is_deep()`.
        avoid (tuple|None): Data types to skip.
        max_depth (int): Maximum depth to reach. Negative for unlimited.
        combine (callable): Combine multiple depth at the same level.
            If `deep` is False, this parameter is ignored.

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
        >>> nesting_level(((1, 2), (1), (1, (2, 3))), True)
        3
        >>> nesting_level(((1, 2), (1), (1, (2, 3))), True, combine=min)
        1
    """
    if not is_deep(obj, avoid, max_depth):
        return 0
    elif len(obj) == 0:
        return 1
    elif max_depth == 0:
        return 1
    else:
        if deep:
            next_level = combine(
                nesting_level(x, deep, avoid, max_depth - 1, combine)
                for x in obj)
        else:
            next_level = nesting_level(
                obj[0], deep, avoid, max_depth - 1, combine)
        return 1 + next_level


# ======================================================================
def nested_len(
        obj,
        deep=True,
        avoid=(str, bytes),
        max_depth=-1,
        combine=max,
        check_same=True):
    """
    Compute the length of nested iterables.

    Args:
        obj (Any): The object to test.
        deep (bool): Evaluate all item.
            If True, all elements within `obj` are evaluated.
            If False, only the first element of each deep object is evaluated.
            An object is considered deep using `is_deep()`.
        avoid (tuple|None): Data types to skip.
        max_depth (int): Maximum depth to reach. Negative for unlimited.
        combine (callable|None): Combine multiple depth at the same level.
            If None, the lengths do not get combined (using `combine=tuple`
            has the same effect).
            If `deep` is False, this parameter is ignored.
        check_same (bool): Check that same-level items have the same length.

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
    if not is_deep(obj, avoid, max_depth):
        return ()
    else:
        if deep:
            next_level = tuple(
                nested_len(
                    x, deep, avoid, max_depth - 1, combine, check_same)
                for x in obj)
            if check_same and any(x != next_level[0] for x in next_level):
                raise ValueError(
                    'Same nesting level items with different length.')
            if not callable(combine):
                combine = tuple
            next_level = combine(next_level)
        else:
            next_level = nested_len(
                obj[0], deep, avoid, max_depth - 1, combine, check_same)
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
        n (int|Iterable[int]): The length(s) of the output object.
            If Iterable, multiple nested tuples will be generated.
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
    """
    try:
        iter(obj)
    except TypeError:
        force = True
    finally:
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
        shape):
    """
    Automatically stretch the values to the target shape.

    This is similar to `flyingcircus.util.auto_repeat()`, except that it
    can flexibly repeat values only when needed.
    This is similar to shape broadcasting of multi-dimensional arrays.

    Args:
        items (Any|Iterable): The input items.
        shape (Iterable[int]): The target shape (nested lengths).

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
    """
    if not is_deep(items):
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
                    if is_deep(item) else auto_repeat(item, shape[1:])
                    for item in items)
            else:
                result = items
        elif old_shape[0] == shape[0]:
            result = tuple(
                stretch(item, shape[1:])
                if is_deep(item) else auto_repeat(item, shape[1:])
                for item in items)
        else:
            raise ValueError(
                'Cannot stretch `{}` to `{}`.'.format(items, shape))
    return result


# ======================================================================
def flatten(
        items,
        avoid=(str, bytes),
        max_depth=-1):
    """
    Recursively flattens nested Iterables.

    The maximum depth is limited by Python's recursion limit.

    Args:
        items (Iterable): The input items.
        avoid (tuple|None): Data types to skip.
        max_depth (int): Maximum depth to reach. Negative for unlimited.

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
        >>> list(flatten(ll2, avoid=(tuple, str)))
        [(1, 2, 3), (4, 5), (1, 2), (3, 4, 5), '1, 2', 6, 7]
        >>> list(flatten(ll2, max_depth=1))
        [(1, 2, 3), (4, 5), (1, 2), (3, 4, 5), '1, 2', [6, 7]]
        >>> list(flatten(ll2, None))
        [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, '1', ',', ' ', '2', 6, 7]
        >>> list(flatten([['best', 'func'], 'ever'], None, 1))
        ['best', 'func', 'e', 'v', 'e', 'r']
        >>> list(flatten([['best', 'func'], 'ever'], None))
        ['b', 'e', 's', 't', 'f', 'u', 'n', 'c', 'e', 'v', 'e', 'r']
        >>> list(flatten(list(range(10))))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        >>> list(flatten(range(10)))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    """
    for item in items:
        if is_deep(item, avoid, max_depth):
            # yield from flatten(item, avoid, max_depth - 1)
            for subitem in flatten(item, avoid, max_depth - 1):
                yield subitem
        else:
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
def complement_slice(
        items,
        slice_,
        container=None):
    """
    Extract the elements not matching a given slice.

    Args:
        items (Sequence): The input items.
        slice_ (slice): The slice to be complemented.
        container (callable): The container for the result.

    Returns:
        result (Sequence): The items not matching the slice pattern.

    Examples:
        >>> items = tuple(range(10))

        >>> s = slice(None)
        >>> print(items[s], tuple(complement_slice(items, s)))
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9) ()

        >>> s = slice(None, None, 2)
        >>> print(items[s], tuple(complement_slice(items, s)))
        (0, 2, 4, 6, 8) (1, 3, 5, 7, 9)

        >>> s = slice(None, None, 3)
        >>> print(items[s], tuple(complement_slice(items, s)))
        (0, 3, 6, 9) (1, 2, 4, 5, 7, 8)

        >>> s = slice(2, None, None)
        >>> print(items[s], tuple(complement_slice(items, s)))
        (2, 3, 4, 5, 6, 7, 8, 9) (0, 1)

        >>> s = slice(None, 7, None)
        >>> print(items[s], tuple(complement_slice(items, s)))
        (0, 1, 2, 3, 4, 5, 6) (7, 8, 9)

        >>> s = slice(2, None, 3)
        >>> print(items[s], tuple(complement_slice(items, s)))
        (2, 5, 8) (0, 1, 3, 4, 6, 7, 9)

        >>> s = slice(None, 7, 3)
        >>> print(items[s], tuple(complement_slice(items, s)))
        (0, 3, 6) (1, 2, 4, 5, 7, 8, 9)

        >>> s = slice(2, 7, 3)
        >>> print(items[s], tuple(complement_slice(items, s)))
        (2, 5) (0, 1, 3, 4, 6, 7, 8, 9)

        >>> s = slice(None, None, -3)
        >>> print(items[s], tuple(complement_slice(items, s)))
        (9, 6, 3, 0) (8, 7, 5, 4, 2, 1)

        >>> s = slice(2, None, -3)
        >>> print(items[s], tuple(complement_slice(items, s)))
        (2,) (9, 8, 7, 6, 5, 4, 3, 1, 0)

        >>> s = slice(None, 7, -3)
        >>> print(items[s], tuple(complement_slice(items, s)))
        (9,) (8, 7, 6, 5, 4, 3, 2, 1, 0)

        >>> s = slice(2, 7, -3)
        >>> print(items[s], tuple(complement_slice(items, s)))
        () (9, 8, 7, 6, 5, 4, 3, 2, 1, 0)

        >>> s = slice(7, 2, -3)
        >>> print(items[s], tuple(complement_slice(items, s)))
        (7, 4) (9, 8, 6, 5, 3, 2, 1, 0)

        >>> items = tuple(1 for i in range(10))
        >>> s = slice(None, None, 3)
        >>> print(items[s], tuple(complement_slice(items, s)))
        (1, 1, 1, 1) (1, 1, 1, 1, 1, 1)

        >>> items = tuple(i % 2 for i in range(10))
        >>> s = slice(None, None, 3)
        >>> print(items[s], tuple(complement_slice(items, s)))
        (0, 1, 0, 1) (1, 0, 0, 1, 1, 0)

        >>> ll = list(range(1000))
        >>> vals = (3, 5, 7, 17, 101)
        >>> vals += tuple(-x for x in vals) + (None,)
        >>> print(vals)
        (3, 5, 7, 17, 101, -3, -5, -7, -17, -101, None)
        >>> sls = [slice(*x) for x in itertools.product(vals, vals, vals)]
        >>> all([
        ...     set(complement_slice(ll, sl)).intersection(ll[sl]) == set()
        ...     for sl in sls])
        True
    """
    if container is None:
        container = type(items)
    to_exclude = set(range(len(items))[slice_])
    step = slice_.step if slice_.step else 1
    result = container(
        item for i, item in enumerate(items) if i not in to_exclude)
    return result if step > 0 else result[::-1]


# ======================================================================
def conditional_apply(
        func,
        condition=None):
    """
    Modify a function so that it is applied only if a condition is satisfied.

    Args:
        func (callable): A function to apply to an object.
            Must have the following signature: func(Any) -> Any
        condition (callable|None): The condition function.
            If not None, the function `func` is applied to an object only
            if the condition on the object evaluates to True.
            Must have the following signature: condition(Any) -> bool

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
        avoid=(str, bytes),
        max_depth=-1):
    """
    Compute a function on each element of a nested structure of iterables.

    The result preserves the nested structure of the input.

    Args:
        func (callable): The function to apply to the individual item.
            Must have the following signature: func(Any) -> Any.
        items (Iterable): The input items.
        avoid (tuple|None): Data types to skip.
        max_depth (int): Maximum depth to reach. Negative for unlimited.

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
    """
    # : alternate implementation
    # if is_deep(items, avoid, max_depth):
    #     return type(items)(
    #         deep_map(func, item, avoid, max_depth - 1) for item in items)
    # else:
    #     return func(items)

    # : alternate implementation
    # return type(items)(
    #     deep_map(func, item, avoid, max_depth - 1)
    #     if is_deep(item, avoid, max_depth) else func(item)
    #     for item in items)

    new_items = []
    for item in items:
        if is_deep(item, avoid, max_depth):
            new_items.append(
                deep_map(func, item, avoid, max_depth - 1))
        else:
            new_items.append(func(item))
    return type(items)(new_items)


# ======================================================================
def deep_filter(
        func,
        items,
        avoid=(str, bytes),
        max_depth=-1):
    """
    Filter the elements from a nested structure of iterables.

    The result preserves the nested structure of the input.

    Args:
        func (callable): The condition function to include the individual item.
            Must have the following signature: func(Any) -> bool
        items (Iterable): The input items.
        avoid (tuple|None): Data types to skip.
        max_depth (int): Maximum depth to reach. Negative for unlimited.

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
    # : alternate implementation
    # if is_deep(items, avoid, max_depth):
    #     return type(items)(
    #         deep_filter(func, item, avoid, max_depth - 1)
    #         for item in items
    #         if is_deep(item, avoid, max_depth) or func(item))
    # else:
    #     return items

    # : alternate implementation
    # return type(items)(
    #     deep_filter(func, item, avoid, max_depth - 1)
    #     if is_deep(item, avoid, max_depth) else item
    #     for item in items if is_deep(item, avoid, max_depth) or func(item))

    new_items = []
    for item in items:
        if is_deep(item, avoid, max_depth):
            new_items.append(
                deep_filter(func, item, avoid, max_depth - 1))
        else:
            if func(item):
                new_items.append(item)
    return type(items)(new_items)


# ======================================================================
def deep_convert(
        container,
        items,
        avoid=(str, bytes),
        max_depth=-1):
    """
    Convert the containers from a nested structure of iterables.

    Args:
        container (callable|None): The container to apply.
            Must have the following signature:
            container(Iterable) -> container.
            If None, no conversion is performed.
        items (Iterable): The input items.
        avoid (tuple|None): Data types to skip.
        max_depth (int): Maximum depth to reach. Negative for unlimited.

    Returns:
        new_items (container): The converted nested structure of iterables.

    Examples:
        >>> items = [1, 2, [3, 4, 5], 2, (3, [4, 5], 'ciao')]
        >>> deep_convert(list, items)
        [1, 2, [3, 4, 5], 2, [3, [4, 5], 'ciao']]
        >>> deep_convert(tuple, items)
        (1, 2, (3, 4, 5), 2, (3, (4, 5), 'ciao'))
        >>> deep_convert(tuple, items, avoid=None)
        (1, 2, (3, 4, 5), 2, (3, (4, 5), ('c', 'i', 'a', 'o')))
        >>> deep_convert(None, items)
        [1, 2, [3, 4, 5], 2, (3, [4, 5], 'ciao')]
    """
    if container is not None:
        new_items = []
        for item in items:
            if is_deep(item, avoid, max_depth):
                new_items.append(
                    deep_convert(container, item, avoid, max_depth - 1))
            else:
                new_items.append(item)
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
        avoid=(str, bytes),
        max_depth=-1):
    """
    Apply conditional mapping, filtering and conversion on nested structures.

    The behavior of this function can be obtained by combining the following:
     - flyingcircus.util.conditional_apply()
     - flyingcircus.util.deep_map()
     - flyingcircus.util.deep_filter()
     - flyingcircus.util.deep_convert()

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
    However, if the call to `deep_filter_map()` would require at least two of
    `deep_map()`, `deep_filter()`, `deep_convert()`, then
    `deep_filter_map()` is generally much more performing.

    Args:
        items (Iterable): The input items.
        func (callable): The function to apply to the individual item.
            Must have the following signature: func(Any) -> Any.
        map_condition (callable|None): The map condition function.
            Only items matching the condition are mapped.
            Must have the following signature: map_condition(Any) -> bool.
        filter_condition (callable|None): The filter condition function.
            Only items matching the condition are included.
            Must have the following signature: filter_condition(Any) -> bool.
                container (callable|None): The container to apply.
            Must have the following signature:
            container(Iterable) -> container.
            If None, the original container is retained.
        avoid (tuple|None): Data types to skip.
        max_depth (int): Maximum depth to reach. Negative for unlimited.

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
    if func is None:
        def func(x): return x
    if map_condition is None:
        def map_condition(_): return True
    if filter_condition is None:
        def filter_condition(_): return True
    new_items = []
    for item in items:
        try:
            no_expand = avoid and isinstance(item, avoid)
            if no_expand or max_depth == 0 or item == next(iter(item)):
                raise TypeError
        except TypeError:
            if filter_condition(item):
                new_items.append(func(item) if map_condition(item) else item)
        else:
            new_items.append(
                deep_filter_map(
                    item, func, map_condition, filter_condition, container,
                    avoid, max_depth - 1))
    if container is None:
        container = type(items)
    return container(new_items)


# ======================================================================
def prod(items):
    """
    Calculate the cumulative product of arbitrary items.

    This is similar to `sum`, but uses product instead of addition.

    Args:
        items (Iterable): The input items.

    Returns:
        result: The cumulative product of `items`.

    Examples:
        >>> prod([2] * 10)
        1024
    """
    return functools.reduce(lambda x, y: x * y, items)


# ======================================================================
def diff(items):
    """
    Calculate the pairwise difference of arbitrary items.

    This is similar to `div`, but uses subtraction instead of division.

    Args:
        items (Iterable): The input items.

    Yields:
        value: The next pairwise difference.

    Examples:
        >>> list(diff(range(10)))
        [1, 1, 1, 1, 1, 1, 1, 1, 1]
    """
    items = iter(items)
    last_item = next(items)
    for i, item in enumerate(items):
        yield item - last_item
        last_item = item


# ======================================================================
def div(items):
    """
    Calculate the pairwise division of arbitrary items.

    This is similar to `diff`, but uses division instead of subtraction.

    Args:
        items (Iterable): The input items.

    Yields:
        value: The next pairwise difference.

    Examples:
        >>> items = [2 ** x for x in range(10)]
        >>> items
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        >>> list(div(items))
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    """
    items = iter(items)
    last_item = next(items)
    for i, item in enumerate(items):
        yield item / last_item
        last_item = item


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
    """
    seen = set()
    for item in items:
        if item not in seen and not seen.add(item):
            yield item


# ======================================================================
def replace_iter(
        items,
        condition,
        replace=None,
        cycle=True):
    """
    Replace items matching a specific condition.

    Args:
        items (Iterable):
        condition (callable):
        replace (any|Iterable|callable): The replacement.
            If Iterable, its elements are used for replacement.
            If callable, it is applied to the elements matching `condition`.
            Otherwise,
        cycle (bool): Cycle through the replace.
            If True and `replace` is Iterable, its elements are cycled through.
            Otherwise `items` beyond last replacement are lost.

    Yields:
        item: The next item from items or its replacement.

    Examples:
        >>> ll = list(range(10))
        >>> list(replace_iter(ll, lambda x: x % 2 == 0))
        [None, 1, None, 3, None, 5, None, 7, None, 9]
        >>> list(replace_iter(ll, lambda x: x % 2 == 0, lambda x: x ** 2))
        [0, 1, 4, 3, 16, 5, 36, 7, 64, 9]
        >>> list(replace_iter(ll, lambda x: x % 2 == 0, 100))
        [100, 1, 100, 3, 100, 5, 100, 7, 100, 9]
        >>> list(replace_iter(ll, lambda x: x % 2 == 0, range(10, 0, -1)))
        [10, 1, 9, 3, 8, 5, 7, 7, 6, 9]
        >>> list(replace_iter(ll, lambda x: x % 2 == 0, range(10, 8, -1)))
        [10, 1, 9, 3, 10, 5, 9, 7, 10, 9]
        >>> list(replace_iter(
        ...     ll, lambda x: x % 2 == 0, range(10, 8, -1), False))
        [10, 1, 9, 3]
    """
    if not callable(replace):
        try:
            replace = iter(replace)
        except TypeError:
            replace = (replace,)
            cycle = True
        if cycle:
            replace = itertools.cycle(replace)
    for item in items:
        if not condition(item):
            yield item
        else:
            yield replace(item) if callable(replace) else next(replace)


# ======================================================================
def combine_iter_len(
        items,
        combine=max,
        scalar_len=1):
    """
    Combine the length of each item within items.

    For each item within items, determine if the item is iterable and then
    use a given combination function to combine the multiple extracted length.
    If an item is not iterable, its length is assumed to be 0.

    A useful application is to determine the longest item.

    Args:
        items (Iterable): The collection of items to inspect.
        combine (callable): The combination method.
            Must have the following signature: combine(int, int) -> int.
            The lengths are combined incrementally.
        scalar_len (int): The length of non-iterable items.
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
        try:
            new_num = len(val)
        except TypeError:
            new_num = scalar_len
        finally:
            if num is None:
                num = new_num
            else:
                num = combine(new_num, num)
    return num


# ======================================================================
def window(
        items,
        size=2,
        step=None,
        fill=None):
    """
    Generate a sliding window across the items.

    This can be used, for example, to compute running/moving/rolling statics.

    Args:
        items (Iterable): The input items.
        size (int): The windowing size.
        step (int|None): The windowing step.
            If int, must be larger than 0.
            If None, uses a step equal to 1 and will not go beyond last item.
        fill: The value to use to fill in window past the end of items.
            This is used only if step is not None.

    Returns:
        result (zip|itertools.zip_longest): Iterable of items within window.

    Examples:
        >>> tuple(window(range(8), 2))
        ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7))
        >>> tuple(window(range(8), 3))
        ((0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6), (5, 6, 7))
        >>> tuple(window(range(8), 3, 2))
        ((0, 1, 2), (2, 3, 4), (4, 5, 6), (6, 7, None))
        >>> tuple(
        ...     x for x in window(range(8), 3, 2)
        ...     if not any([y is None for y in x]))
        ((0, 1, 2), (2, 3, 4), (4, 5, 6))
        >>> tuple(window(range(8), 2, 1))
        ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, None))
        >>> tuple(window(range(8), 1))
        ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,))
        >>> tuple(window(range(8), 1, 1))
        ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,))
    """
    # : alternate (slightly faster, but less flexible: no step) implementation
    # def consumed(iterator, n):
    #     next(itertools.islice(iterator, n, n), None)
    #     return iterator
    # iterators = [consumed(iter(items), i) for i in range(size)]
    iterators = [
        itertools.islice(iter(items), i, None, step) for i in range(size)]
    if step:
        return itertools.zip_longest(*iterators, fillvalue=fill)
    else:
        return zip(*iterators)


# ======================================================================
def group_by(
        items,
        n,
        truncate=False,
        fill=None):
    """
    Generate grouped items (with constant group size).

    For different handling of the last group for uneven splits, see
    `flyingcircus.util.grouping()`.

    Args:
        items (Iterable): The input items.
        n (int): Number of elements to group together.
        truncate (bool): Determine how to handle uneven splits.
            If True, last group is skipped if its length is smaller than `n`.
        fill (Any): Value to use for fill group.
            This is only used when `truncate` is False.

    Returns:
        groups (zip|itertools.zip_longest): Iterable of grouped items.
            Each group is a tuple regardless of the original container.

    Examples:
        >>> l = list(range(10))
        >>> tuple(group_by(l, 4))
        ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, None, None))
        >>> tuple(group_by(tuple(l), 2))
        ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))
        >>> tuple(group_by(l, 4, True))
        ((0, 1, 2, 3), (4, 5, 6, 7))
        >>> tuple(group_by(l, 4, False, 0))
        ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 0, 0))

    See Also:
        flyingcircus.util.grouping()
    """
    # : alternate (slower) implementations
    # iterators = tuple(items[i::n] for i in range(n))
    # iterators = tuple(itertools.islice(items, i, None, n) for i in range(n))
    iterators = [iter(items)] * n
    if truncate:
        return zip(*iterators)
    else:
        return itertools.zip_longest(*iterators, fillvalue=fill)


# ======================================================================
def grouping(
        items,
        splits):
    """
    Generate grouped items (with varying grouping size)

    Note that for integer splits, `group_by()` is a faster alternative.

    Args:
        items (Iterable): The input items.
        splits (int|Iterable[int]): Grouping information.
            If Iterable, each group has the number of elements specified.
            If int, all groups have the same number of elements.
            The last group will have the remaing items (if any).

    Yields:
        group (Iterable): The items from the grouping.
            Its container matches the one of `items`.

    Examples:
        >>> l = list(range(10))
        >>> tuple(grouping(l, 4))
        ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9])
        >>> tuple(grouping(l, (2, 3)))
        ([0, 1], [2, 3, 4], [5, 6, 7, 8, 9])
        >>> tuple(grouping(l, (2, 4, 1)))
        ([0, 1], [2, 3, 4, 5], [6], [7, 8, 9])
        >>> tuple(grouping(l, (2, 4, 1, 20)))
        ([0, 1], [2, 3, 4, 5], [6], [7, 8, 9])
        >>> tuple(grouping(tuple(l), 4))
        ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9))
        >>> tuple(grouping(tuple(l), 2)) == tuple(group_by(l, 2))
        True

    See Also:
        flyingcircus.util.group_by()
    """
    if isinstance(splits, int):
        splits = auto_repeat(splits, len(items) // splits)

    num_items = len(items)
    if sum(splits) >= num_items:
        splits = splits[:-1]
    index = (0,) + tuple(itertools.accumulate(splits)) + (num_items,)
    num = len(index) - 1
    for i in range(num):
        yield items[index[i]:index[i + 1]]


# ======================================================================
def chunks(
        items,
        n,
        mode='+',
        balanced=True):
    """
    Yield items into approximately N equally sized chunks.

    If the number of items does not allow chunks of the same size, the chunks
    are determined depending on the values of `balanced`

    Args:
        items (Iterable): The input items.
        n (int): Approximate number of chunks.
            The exact number depends on the value of `mode`.
        mode (str): Determine which approximation to use.
            If str, valid inputs are:
             - 'upper', '+': at most `n` chunks are generated.
             - 'lower', '-': at least `n` chunks are genereated.
             - 'closest', '~': the number of chunks is `n` or `n + 1`
               depending on which gives the most evenly distributed chunks
               sizes.
        balanced (bool): Produce balanced chunks.
            If True, the size of any two chunks is not larger than one.
            Otherwise, the first chunks except the last have the same size.
            This has no effect if the number of items is a multiple of `n`.

    Returns:
        groups (tuple[Iterable]): Grouped items from the source.

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
    """
    if mode in ('upper', '+'):
        approx = math.ceil
    elif mode in ('lower', '-'):
        approx = math.floor
    elif mode in ('closest', '~'):
        approx = round
    else:
        raise ValueError('Invalid mode `{mode}`'.format(mode=mode))
    n = max(1, n)
    split = int(approx(len(items) / n))
    if balanced and 0 < len(items) % split <= split // 2:
        k = len(items) // split + 1
        q = -len(items) % split
        split = (split,) * (k - q) + (split - 1,) * q
    return grouping(items, split)


# ======================================================================
def partitions(
        items,
        k,
        container=tuple):
    """
    Generate all k-partitions for the items.

    Args:
        items (Iterable): The input items.
        k (int): The number of splitting partitions.
            Each group has exactly `k` elements.
        container (callable): The group container.

    Yields:
        partition (tuple[Iterable]]): The grouped items.
            Each partition contains `k` grouped items from the source.

    Examples:
        >>> tuple(partitions(tuple(range(3)), 2))
        (((0,), (1, 2)), ((0, 1), (2,)))
        >>> tuple(partitions(tuple(range(3)), 3))
        (((0,), (1,), (2,)),)
        >>> tuple(partitions(tuple(range(4)), 3))
        (((0,), (1,), (2, 3)), ((0,), (1, 2), (3,)), ((0, 1), (2,), (3,)))
    """
    num = len(items)
    indexes = tuple(
        (0,) + tuple(index) + (num,)
        for index in itertools.combinations(range(1, num), k - 1))
    for index in indexes:
        yield tuple(
            container(
                items[index[i]:index[i + 1]] for i in range(k)))


# ======================================================================
def random_unique_combinations_k(items, k, pseudo=False):
    """
    Obtain a number of random unique combinations of a sequence of sequences.

    Args:
        items (Sequence[Sequence]): The input sequence of sequences.
        k (int): The number of random unique combinations to obtain.
        pseudo (bool): Generate random combinations somewhat less randomly.
            If True, the memory requirements for intermediate steps will
            be significantly lower (but still all `k` items are required to
            fit in memory).

    Yields:
        combination (Sequence): The next random unique combination.

    Examples:
        >>> import string
        >>> max_lens = list(range(2, 10))
        >>> items = [string.ascii_lowercase[:max_len] for max_len in max_lens]
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
        >>> items = [string.ascii_uppercase[:max_len] for max_len in max_lens]
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
    if pseudo:
        # randomize generators
        comb_gens = list(items)
        for num, comb_gen in enumerate(comb_gens):
            random.shuffle(list(comb_gens[num]))
        # get the first `k` combinations
        combinations = list(itertools.islice(itertools.product(*comb_gens), k))
        random.shuffle(combinations)
        for combination in itertools.islice(combinations, k):
            yield tuple(combination)
    else:
        max_lens = [len(list(item)) for item in items]
        max_k = prod(max_lens)
        try:
            for num in random.sample(range(max_k), min(k, max_k)):
                indexes = []
                for max_len in max_lens:
                    indexes.append(num % max_len)
                    num = num // max_len
                yield tuple(item[i] for i, item in zip(indexes, items))
        except OverflowError:
            # use `set` to ensure uniqueness
            index_combs = set()
            # make sure that with the chosen number the next loop can exit
            # WARNING: if `k` is too close to the total number of combinations,
            # it may take a while until the next valid combination is found
            while len(index_combs) < min(k, max_k):
                index_combs.add(tuple(
                    random.randint(0, max_len - 1) for max_len in max_lens))
            # make sure their order is shuffled
            # (`set` seems to sort its content)
            index_combs = list(index_combs)
            random.shuffle(index_combs)
            for index_comb in itertools.islice(index_combs, k):
                yield tuple(item[i] for i, item in zip(index_comb, items))


# ======================================================================
def unique_permutations(
        items,
        container=tuple):
    """
    Yield unique permutations of items in an efficient way.

    Args:
        items (Iterable): The input items.
        container (callable): The group container.

    Yields:
        items (Iterable): The next unique permutation of the items.

    Examples:
        >>> list(unique_permutations([0, 0, 0]))
        [(0, 0, 0)]
        >>> list(unique_permutations([0, 0, 2]))
        [(0, 0, 2), (0, 2, 0), (2, 0, 0)]
        >>> list(unique_permutations([0, 1, 2]))
        [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
        >>> p1 = sorted(unique_permutations((0, 1, 2, 3, 4)))
        >>> p2 = sorted(itertools.permutations((0, 1, 2, 3, 4)))
        >>> p1 == p2
        True

    References:
        - Donald Knuth, The Art of Computer Programming, Volume 4, Fascicle
          2: Generating All Permutations.
    """
    indexes = range(len(items) - 1, -1, -1)
    items = sorted(items)
    while True:
        if callable(container):
            yield container(items)
        else:
            yield items.copy()

        for k in indexes[1:]:
            if items[k] < items[k + 1]:
                break
        else:
            return

        k_val = items[k]
        for i in indexes:
            if k_val < items[i]:
                break

        items[k], items[i] = items[i], items[k]
        items[k + 1:] = items[-1:k:-1]


# ======================================================================
def unique_partitions(
        items,
        k):
    """
    Generate all k-partitions for all unique permutations of the items.

    Args:
        items (Iterable): The input items.
        k (int): The number of splitting partitions.
            Each group has exactly `k` elements.

    Yields:
        partitions (Iterable[Iterable[Iterable]]]): The items partitions.
            More precisely, all partitions of size `num` for each unique
            permutations of `items`.

    Examples:
        >>> list(unique_partitions([0, 1], 2))
        [(((0,), (1,)),), (((1,), (0,)),)]

    """
    for permutations in unique_permutations(items):
        yield tuple(partitions(tuple(permutations), k))


# ======================================================================
def listdict2dictlist(
        listdict,
        labels=None,
        d_val=None):
    """
    Convert tabular data from a list of dicts to a dict of lists.

    Args:
        listdict (Iterable[dict]): The tabular data as a list of dicts.
        labels (Iterable|None): The labels of the tabular data.
            If Iterable, all elements should be present as keys of the dicts.
            If the dicts contain keys not specified in `labels` they will be
            ignored.
            If None, the `labels` are guessed from the data.
        d_val (Any): The default value to use for incomplete dicts.
            This will be inserted in the lists to keep track of missing data.

    Returns:
        dictlist (dict[Any:list]): The tabular data as a dict of lists.
            All list will have the same size.

    Examples:
        >>> ld = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6}]
        >>> tuple(listdict2dictlist(ld).items())
        (('a', [1, 3, 5]), ('b', [2, 4, 6]))
        >>> ld = [{'a': 1}, {'b': 4}, {'b': 6}]
        >>> tuple(listdict2dictlist(ld).items())
        (('a', [1, None, None]), ('b', [None, 4, 6]))
        >>> ld == dictlist2listdict((listdict2dictlist(ld)))
        True
    """
    if not labels:
        labels = sorted(functools.reduce(
            lambda x, y: x.union(y), [set(x.keys()) for x in listdict]))
    dictlist = {
        label: [item[label] if label in item else d_val for item in listdict]
        for label in labels}
    return dictlist


# ======================================================================
def dictlist2listdict(
        dictlist,
        labels=None,
        d_val=None):
    """
    Convert tabular data from a dict of lists to a list of dicts.

    Args:
        dictlist (Iterable[dict): The tabular data as a dict of lists.
            All lists must have the same length.
        labels (Iterable|None): The labels of the tabular data.
            If None, the `labels` are guessed from the data.
        d_val (Any): The default value to be used for reducing dicts.
            The values matching `d_val` are not included in the dicts.

    Returns:
        dictlist (dict[Any:list]): The tabular data as a dict of lists.
            All list will have the same size.

    Examples:
        >>> dl = {'a': [1, 3, 5], 'b': [2, 4, 6]}
        >>> [sorted(d.items()) for d in dictlist2listdict(dl)]
        [[('a', 1), ('b', 2)], [('a', 3), ('b', 4)], [('a', 5), ('b', 6)]]
        >>> dl = {'a': [1, None, None], 'b': [None, 4, 6]}
        >>> [sorted(d.items()) for d in dictlist2listdict(dl)]
        [[('a', 1)], [('b', 4)], [('b', 6)]]
        >>> dl == listdict2dictlist((dictlist2listdict(dl)))
        True
    """
    if not labels:
        labels = tuple(dictlist.keys())
    num_elems = len(dictlist[labels[0]])
    listdict = [
        {label: dictlist[label][i] for label in labels
         if dictlist[label][i] is not d_val}
        for i in range(num_elems)]
    return listdict


# ======================================================================
def round_up(x):
    """
    Round to the largest close integer.

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

    Args:
        a (int): The dividend.
        b (int): The divisor.

    Returns:
        qc (int): The ceiled quotient.
            This is the quotient if `a` is divisible by `b`, otherwise
            it gived the quotient plus one.

    Examples:
        >>> div_ceil(6, 3)
        2
        >>> div_ceil(7, 3)
        3
        >>> div_ceil(6, 3) == 6 // 3
        True
        >>> div_ceil(7, 3) == (7 // 3) + 1
        True
    """
    return a // b + (1 if a % b else 0)


# ======================================================================
def isqrt(num):
    """
    Calculate the integer square root of a number.

    This is defined as the largest integer whose square is smaller then the
    number, i.e. floor(sqrt(n))

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
    """
    num = abs(num)
    guess = (num >> num.bit_length() // 2) + 1
    result = (guess + num // guess) // 2
    while abs(result - guess) > 1:
        guess = result
        result = (guess + num // guess) // 2
    while result * result > num:
        result -= 1
    return result


# ======================================================================
def get_pascal_numbers(
        num,
        full=True,
        cached=False):
    """
    Generate the numbers of a given row of Pascal's triangle.

    These are the numbers in the `num`-th row (order) of the Pascal's triangle.
    If only a specific binomial coefficient is required, use
    `scipy.special.comb(exact=True)`.

    Args:
        num (int): The row index of the Pascal's triangle.
            Indexing start from 0.
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
        value (int): The next Pascal number of a given row.

    Examples:
        >>> list(get_pascal_numbers(12))
        [1, 12, 66, 220, 495, 792, 924, 792, 495, 220, 66, 12, 1]
        >>> list(get_pascal_numbers(13))
        [1, 13, 78, 286, 715, 1287, 1716, 1716, 1287, 715, 286, 78, 13, 1]
        >>> list(get_pascal_numbers(12, False))
        [1, 12, 66, 220, 495, 792, 924]
        >>> list(get_pascal_numbers(13, False))
        [1, 13, 78, 286, 715, 1287, 1716]
        >>> num = 10
        >>> for n in range(num):
        ...     print(list(get_pascal_numbers(n)))
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

    See Also:
        - flyingcircus.util.pascal_triangle_range()
        - sp.special.comb()
        - sp.special.binom()
        - https://en.wikipedia.org/wiki/Binomial_coefficient
        - https://en.wikipedia.org/wiki/Pascal%27s_triangle
    """
    value = 1
    stop = (num + 1) if full and not cached else (num // 2 + 1)
    if full and cached:
        cache = [value * (num - i) // (i + 1) for i in range(stop)]
        # equivalent to:
        # cache = [value for value in get_pascal_numbers(num, False, False)]
        for value in cache:
            yield value
        for value in cache[(-1 if num % 2 else -2)::-1]:
            yield value
    else:
        for i in range(stop):
            yield value
            value = value * (num - i) // (i + 1)


# ======================================================================
def pascal_triangle_range(
        first,
        second=None,
        step=None,
        container=tuple):
    """
    Generate the Pascal's triangle rows in a given range.

    See `get_pascal_numbers()` for generating any given row of the triangle.

    Args:
        first (int): The first value of the range.
            Must be non-negative.
            If `second == None` this is the `stop` value, and is not included.
            Otherwise, this is the start value and is included.
            If `first < second` the sequence is yielded backwards.
        second (int|None): The second value of the range.
            If None, the start value is 0.
            Otherwise, this is the stop value and is not included.
            Must be non-negative.
            If `first < second` the sequence is yielded backwards.
        step (int): The step of the rows range.
            If the sequence is yielded backward, the step should be negative,
            otherwise an empty sequence is yielded.
            If None, this is computed automatically based on `first` and
            `second`, such that a non-empty sequence is avoided, if possible.
        container (callable): The row container.

    Yields:
        row (Iterable[int]): The rows of the Pascal's triangle.

    Examples:
        >>> tuple(pascal_triangle_range(5))
        ((1,), (1, 1), (1, 2, 1), (1, 3, 3, 1), (1, 4, 6, 4, 1))
        >>> tuple(pascal_triangle_range(5, 7))
        ((1, 5, 10, 10, 5, 1), (1, 6, 15, 20, 15, 6, 1))
        >>> tuple(pascal_triangle_range(7, 9))
        ((1, 7, 21, 35, 35, 21, 7, 1), (1, 8, 28, 56, 70, 56, 28, 8, 1))
        >>> tuple(pascal_triangle_range(5, 2))
        ((1, 5, 10, 10, 5, 1), (1, 4, 6, 4, 1), (1, 3, 3, 1))
        >>> tuple(pascal_triangle_range(0, 7, 2))
        ((1,), (1, 2, 1), (1, 4, 6, 4, 1), (1, 6, 15, 20, 15, 6, 1))
        >>> tuple(pascal_triangle_range(0, 6, 2))
        ((1,), (1, 2, 1), (1, 4, 6, 4, 1))
        >>> tuple(pascal_triangle_range(7, 1, -2))
        ((1, 7, 21, 35, 35, 21, 7, 1), (1, 5, 10, 10, 5, 1), (1, 3, 3, 1))
        >>> tuple(pascal_triangle_range(7, 1, 2))  # empty range!
        ()
        >>> list(pascal_triangle_range(3))
        [(1,), (1, 1), (1, 2, 1)]
        >>> list(pascal_triangle_range(5, container=list))
        [[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]]
        >>> for row in pascal_triangle_range(10):
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
        - flyingcircus.util.get_pascal_numbers()
        - sp.special.comb()
        - sp.special.binom()
        - https://en.wikipedia.org/wiki/Binomial_coefficient
        - https://en.wikipedia.org/wiki/Pascal%27s_triangle
    """
    if second is None:
        start, stop = 0, first
    else:
        start, stop = first, second
    if not step:
        step = 1 if start < stop else -1
    for i in range(start, stop, step):
        yield container(get_pascal_numbers(i))


# ======================================================================
def is_prime(num):
    """
    Determine if number is prime.

    A prime number is only divisible by 1 and itself.
    0 and 1 are considered special cases; in this implementations they are
    considered primes.

    It is implemented by directly testing for possible factors
    (i.e. the trial division algorithm, excluding multiples of 2 and 3).

    Args:
        num (int): The number to check for primality.
            Only works for numbers larger than 1.

    Returns:
        is_divisible (bool): The result of the primality.

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
        >>> is_prime(0)
        True
        >>> is_prime(1)
        True

    See Also:
        - flyingcircus.util.is_prime()
        - flyingcircus.util.primes_range()
        - https://en.wikipedia.org/wiki/Prime_number
        - https://en.wikipedia.org/wiki/Trial_division
    """
    # : fastest implementation (skip both 2 and 3 multiples!)
    num = abs(num)
    if (num % 2 == 0 and num > 2) or (num % 3 == 0 and num > 3):
        return False
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        else:
            i += 6
    return True


# ======================================================================
def primes_range(
        first,
        second=None):
    """
    Calculate the prime numbers in the range.

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
        - flyingcircus.util.is_prime()
        - flyingcircus.util.get_primes()
        - https://en.wikipedia.org/wiki/Prime_number
    """
    if second is None:
        start, stop = 2, first
    else:
        start, stop = first, second
    if start < 2 and start < stop:
        start = 2
    step = 2 if start < stop else -2
    if start % 2 == 0:
        if start == 2:
            yield start
        start += step // 2
    for num in range(start, stop, step):
        if is_prime(num):
            yield num
    if start > stop and stop < 2:
        yield 2


# ======================================================================
def get_primes(
        max_count=-1,
        num=2):
    """
    Calculate prime numbers.

    Args:
        max_count (int): The maximum number of values to yield.
            If `max_count == -1`, the generation proceeds indefinitely.
        num (int): The initial value.

    Yields:
        num (int): The next prime number.

    Examples:
        >>> n = 15
        >>> primes = get_primes()
        >>> [next(primes) for i in range(n)]
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        >>> [x for x in get_primes(n)]
        [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        >>> [x for x in get_primes(10, 101)]
        [101, 103, 107, 109, 113, 127, 131, 137, 139, 149]
        >>> [x for x in get_primes(10, 1000)]
        [1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061]

    See Also:
        - flyingcircus.util.is_prime()
        - flyingcircus.util.primes_range()
        - https://en.wikipedia.org/wiki/Prime_number
    """
    i = 0
    while num <= 2:
        if is_prime(num):
            yield num
            i += 1
        num += 1
    if num % 2 == 0:
        num += 1
    while i != max_count:
        if is_prime(num):
            yield num
            i += 1
        num += 2


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
    use `flyingcircus.util.fibonacci()`.

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
        - flyingcircus.util.get_gen_fibonacci()
        - flyingcircus.util.fibonacci()
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
        values (int|Iterable[int]): The initial numbers of the sequence.
            If int, the value is repeated for the number of `weights`, and
            `weights` must be an iterable.
        weights (int|Iterable[int]): The weights for the linear combination.
            If int, the value is repeated for the number of `values`, and
            `values` must be an iterable.

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
        - flyingcircus.util.get_fibonacci()
        - flyingcircus.util.fibonacci()
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
    `flyingcircus.util.get_fibonacci()`.

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
        - flyingcircus.util.get_fibonacci()
        - flyingcircus.util.get_gen_fibonacci()
        - https://en.wikipedia.org/wiki/Fibonacci_number
    """
    for _ in range(num):
        first, second = second, first + second
    return first


# ======================================================================
def factorize(num):
    """
    Find all factors of a number.

    Args:
        num (int): The number to factorize.

    Yields:
        factor (int): The next factor of the number.
            Factors are yielded in increasing order.

    Examples:
        >>> list(factorize(100))
        [2, 2, 5, 5]
        >>> list(factorize(1234567890))
        [2, 3, 3, 5, 3607, 3803]
        >>> list(factorize(-65536))
        [-1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        >>> list(factorize(0))
        [0]
        >>> list(factorize(1))
        [1]
        >>> list(factorize(-1))
        [-1]
        >>> all([n == prod(factorize(n)) for n in range(1000)])
        True
    """
    # deal with special numbers: 0, 1, and negative
    if num == 0:
        text = 'Factorization of `0` is undefined.'
        warnings.warn(text)

    if num < 0:
        yield -1
    elif num <= 1:
        yield num

    num = abs(num)

    primes = get_primes()
    prime = next(primes)
    while prime * prime <= num:
        while num % prime == 0:
            num //= prime
            yield prime
        prime = next(primes)
    if num > 1:
        yield num


# ======================================================================
def factorize_as_dict(num):
    """
    Find all factors of a number and collect them in an ordered dict.

    Args:
        num (int): The number to factorize.

    Returns:
        factors (collections.OrderedDict): The factors of the number.

    Examples:
        >>> factorize_as_dict(100)
        OrderedDict([(2, 2), (5, 2)])
        >>> factorize_as_dict(1234567890)
        OrderedDict([(2, 1), (3, 2), (5, 1), (3607, 1), (3803, 1)])
        >>> factorize_as_dict(65536)
        OrderedDict([(2, 16)])
    """
    factors = list(factorize(num))
    return collections.OrderedDict(collections.Counter(factors))


# ======================================================================
def factorize_as_str(
        num,
        exp_sep='^',
        fact_sep=' * '):
    """
    Find all factors of a number and output a human-readable text.

    Args:
        num (int): The number to factorize.
        exp_sep (str): The exponent separator.
        fact_sep (str): The factors separator.

    Returns:
        text (str): The factors of the number.

    Examples:
        >>> factorize_as_str(100)
        '2^2 * 5^2'
        >>> factorize_as_str(1234567890)
        '2 * 3^2 * 5 * 3607 * 3803'
        >>> factorize_as_str(65536)
        '2^16'
    """
    text = ''
    last_factor = 1
    exp = 0
    for factor in factorize(num):
        if factor == last_factor:
            exp += 1
        else:
            if exp > 1:
                text += exp_sep + str(exp)
            if last_factor > 1:
                text += fact_sep
            text += str(factor)
            last_factor = factor
            exp = 1
    if exp > 1:
        text += exp_sep + str(exp)
    return text


# =====================================================================
def factorize_k_all(
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
        ...     factorize_k_all(i, 2)
        ((2, 16), (4, 8), (8, 4), (16, 2))
        ((1, 41), (41, 1))
        ((2, 23), (23, 2))
        ((2, 30), (3, 20), (4, 15), (5, 12), (6, 10), (10, 6), (12, 5),\
 (15, 4), (20, 3), (30, 2))
        >>> for i in nums:
        ...     factorize_k_all(i, 3)
        ((2, 2, 8), (2, 4, 4), (2, 8, 2), (4, 2, 4), (4, 4, 2), (8, 2, 2))
        ((1, 1, 41), (1, 41, 1), (41, 1, 1))
        ((1, 2, 23), (1, 23, 2), (2, 1, 23), (2, 23, 1), (23, 1, 2),\
 (23, 2, 1))
        ((2, 2, 15), (2, 3, 10), (2, 5, 6), (2, 6, 5), (2, 10, 3), (2, 15, 2),\
 (3, 2, 10), (3, 4, 5), (3, 5, 4), (3, 10, 2), (4, 3, 5), (4, 5, 3),\
 (5, 2, 6), (5, 3, 4), (5, 4, 3), (5, 6, 2), (6, 2, 5), (6, 5, 2),\
 (10, 2, 3), (10, 3, 2), (15, 2, 2))
    """
    factors = tuple(factorize(num))
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
def factorize_k(
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
            See `flyingcircus.util.chunks()` for more info.

    Returns:
        tuple (int): A listing of `k` factors of `num`.

    Examples:
        >>> [factorize_k(402653184, k) for k in range(3, 6)]
        [(1024, 768, 512), (192, 128, 128, 128), (64, 64, 64, 48, 32)]
        >>> [factorize_k(402653184, k) for k in (2, 12)]
        [(24576, 16384), (8, 8, 8, 8, 6, 4, 4, 4, 4, 4, 4, 4)]
        >>> factorize_k(6, 4)
        (3, 2, 1, 1)
        >>> factorize_k(-12, 4)
        (3, 2, 2, -1)
        >>> factorize_k(0, 4)
        (1, 1, 1, 0)
        >>> factorize_k(720, 4)
        (6, 6, 5, 4)
        >>> factorize_k(720, 4, '+')
        (4, 4, 9, 5)
        >>> factorize_k(720, 3)
        (12, 10, 6)
        >>> factorize_k(720, 3, '+')
        (8, 6, 15)
        >>> factorize_k(720, 3, mode='-')
        (45, 4, 4)
        >>> factorize_k(720, 3, mode='seed0')
        (12, 6, 10)
        >>> factorize_k(720, 3, 'alt')
        (30, 4, 6)
        >>> factorize_k(720, 3, 'alt1')
        (12, 6, 10)
        >>> factorize_k(720, 3, '=')
        (12, 10, 6)
    """
    if k > 1:
        factors = list(factorize(num))
        if len(factors) < k:
            factors.extend([1] * (k - len(factors)))
        groups = None
        if mode in ('increasing', 'ascending', '+'):
            factors = sorted(factors)
        elif mode in ('decreasing', 'descending', '-'):
            factors = sorted(factors, reverse=True)
        elif mode == 'random':
            random.shuffle(factors)
        elif mode.startswith('seed'):
            seed = auto_convert(mode[len('seed'):])
            random.seed(seed)
            random.shuffle(factors)
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
            groups = chunks(factors, k, mode='+', balanced=True)
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
    factorizations = factorize_k_all(num, dims)
    return sorted(factorizations, key=sort, reverse=reverse)[0]


# ======================================================================
def _gcd(a, b):
    """
    Calculate the greatest common divisor (GCD) of a and b.

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
def gcd(*nums):
    """
    Find the greatest common divisor (GCD) of a list of numbers.

    Args:
        *nums (*Iterable[int]): The input numbers.

    Returns:
        gcd_val (int): The value of the greatest common divisor (GCD).

    Examples:
        >>> gcd(12, 24, 18)
        6
        >>> gcd(12, 24, 18, 42, 600, 66, 666, 768)
        6
        >>> gcd(12, 24, 18, 42, 600, 66, 666, 768, 101)
        1
        >>> gcd(12, 24, 18, 3)
        3
    """
    gcd_val = nums[0]
    for num in nums[1:]:
        gcd_val = math.gcd(gcd_val, num)
    return gcd_val


# ======================================================================
def lcm(*nums):
    """
    Find the least common multiple (LCM) of a list of numbers.

    Args:
        *nums (*Iterable[int]): The input numbers.

    Returns:
        gcd_val (int): The value of the least common multiple (LCM).

    Examples:
        >>> lcm(2, 3, 4)
        12
        >>> lcm(9, 8)
        72
        >>> lcm(12, 23, 34, 45, 56)
        985320
    """
    lcm_val = nums[0]
    for num in nums[1:]:
        lcm_val = lcm_val * num // math.gcd(lcm_val, num)
    return lcm_val


# ======================================================================
def mean(items):
    """
    Calculate the mean of arbitrary items.

    For iterative computation see:
     - `flyingcircus.util.next_mean()`
     - `flyingcircus.util.imean()`

    This is substantially faster than `statistics.mean()`.

    Args:
        items (Sequence): The input items.
            The values within the sequence should be numeric.

    Returns:
        result (Any): The mean of the items.

    Examples:
        >>> mean(range(0, 20, 2))
        9.0
    """
    return sum(items) / len(items)


# ======================================================================
def var(items):
    """
    Calculate the variance of arbitrary items.

    For iterative computation see:
     - `flyingcircus.util.next_mean_var()`
     - `flyingcircus.util.next_mean_sosd()` and `.sosd2var()`.
     - `flyingcircus.util.ivar()`

    This is substantially faster than `statistics.variance()`.

    Args:
        items (Sequence): The input items.
            The values within the sequence should be numeric.

    Returns:
        result (Any): The variance of the items.

    Examples:
        >>> var(range(0, 20, 2))
        33.0
    """
    mean_ = mean(items)
    return sum((i - mean_) ** 2 for i in items) / len(items)


# ======================================================================
def mean_var(items):
    """
    Calculate the variance of arbitrary items.

    For iterative computation see:
     - `flyingcircus.util.next_mean_var()`
     - `flyingcircus.util.next_mean_sosd()` and `.sosd2var()`.
     - `flyingcircus.util.ivar()`

    This is substantially faster than `statistics.variance()`.

    Args:
        items (Sequence): The input items.
            The values within the sequence should be numeric.

    Returns:
        result (tuple): The tuple
            contains:
             - `mean_`: The mean of the items.
             - `var_`: The variance of the items.

    Examples:
        >>> mean_var(range(0, 20, 2))
        (9.0, 33.0)
    """
    mean_ = mean(items)
    return mean_, sum((i - mean_) ** 2 for i in items) / len(items)


# ======================================================================
def stdev(
        items,
        ddof=0):
    """
    Calculate the standard deviation of arbitrary items.

    For iterative computation see:
     - `flyingcircus.util.next_mean_var()`
     - `flyingcircus.util.next_mean_sosd()` and `.sosd2stdev()`.
     - `flyingcircus.util.istdev()`

    This is substantially faster than `statistics.stdev()`.

    Args:
        items (Sequence): The input items.
            The values within the sequence should be numeric.
        ddof (int): The number of degrees of freedom.

    Returns:
        result (Any): The standard deviation of the items.

    Examples:
        >>> round(stdev(range(0, 20, 2)), 3)
        5.745
    """
    mean_ = mean(items)
    return (sum((i - mean_) ** 2 for i in items) / (len(items) - ddof)) ** 0.5


# ======================================================================
def next_mean(
        value,
        mean_,
        num):
    """
    Compute the mean for (num + 1) items.

    This is useful for low memory footprint computation of the mean.

    Args:
        value (Any): The next to consider.
        mean_ (Any): The aggregate mean of the previous n items.
        num (int): The number of items in the aggregate.

    Returns:
        mean (Any): The updated mean.

    Examples:
        >>> mean = 0.0
        >>> for i, val in enumerate(range(0, 20, 2)):
        ...     mean = next_mean(val, mean, i)
        >>> print(mean)
        9.0
    """
    return (num * mean_ + value) / (num + 1)


# ======================================================================
def next_mean_var(
        value,
        mean_,
        var_,
        num):
    """
    Compute the mean and variance for (num + 1) items.

    This is useful for low memory footprint computation of the variance.

    Note that both mean and variance MUST be updated at each iteration,
    therefore a stand-alone `next_var()` is sub-optimal.

    Args:
        value (Any): The next value to consider.
        mean_ (Any): The aggregate mean of the previous n items.
        var_ (Any): The aggregate variance of the previous n items.
        num (int): The number of items in the aggregate.

    Returns:
        mean (Any): The tuple
            contains:
             - mean (Any): The updated mean.
             - var (Any): The updated variance.

    Examples:
        >>> mean, var = 0.0, 0.0
        >>> for i, val in enumerate(range(0, 20, 2)):
        ...     mean, var = next_mean_var(val, mean, var, i)
        >>> print(mean, var)
        9.0 33.0
    """
    last = mean_
    mean_ = next_mean(value, mean_, num)
    var_ = ((var_ * num) + (value - last) * (value - mean_)) / (num + 1)
    return mean_, var_


# ======================================================================
def next_mean_sosd(
        value,
        mean_,
        sosd,
        num=0):
    """
    Compute the mean and sum-of-squared-deviations for (num + 1) items.

    The sum is the variance multiplied by the number of items:

    sosd = sum((x_i - mu) ** 2)
    sosd = var * n

    This is useful for low memory footprint computation of the variance
    with a numerically stable algorithm.

    Note that both mean and variance MUST be updated at each iteration,
    therefore a stand-alone `next_var()` is sub-optimal.

    Args:
        value (Any): The next value to consider.
        mean_ (Any): The aggregate mean of the previous n items.
        sosd (Any): The aggregate modified variance of the previous n items.
        num (int): The number of items in the aggregate value.

    Returns:
        mean (Any): The tuple
            contains:
             - mean (Any): The updated mean.
             - sosd (Any): The updated sum-of-squared-deviations.

    Examples:
        >>> mean_, sosd = 0.0, 0.0
        >>> items = range(0, 20, 2)
        >>> for i, val in enumerate(items):
        ...     mean_, sosd = next_mean_sosd(val, mean_, sosd, i)
        >>> var_ = sosd2var(sosd, i + 1)
        >>> stdev_ = sosd2stdev(sosd, i + 1)
        >>> print(mean_, var_, round(stdev_, 3))
        9.0 33.0 5.745
        >>> print(mean(items), var(items), round(stdev(items), 3))
        9.0 33.0 5.745

    References:
         - Welford, B.P. (1962). "Note on a method for calculating corrected
           sums of squares and products". Technometrics 4(3):419–420.
           doi:10.2307/1266577
    """
    last = mean_
    mean_ = next_mean(value, mean_, num)
    sosd = (sosd + (value - last) * (value - mean_))
    return mean_, sosd


# ======================================================================
def sosd2var(
        sosd,
        num):
    """
    Compute the variance from the sum-of-squared-deviations.

    Args:
        sosd (Any): The sum-of-squared-deviations value.
        num (int): The number of items.

    Returns:
        result (Any): The variance value.
    """
    return sosd / num


# ======================================================================
def sosd2stdev(
        sosd,
        num,
        ddof=0):
    """
    Compute the standard deviation from the sum-of-squared-deviations.

    Args:
        sosd (Any): The sum-of-squared-deviations value.
        num (int): The number of items.
        ddof (int): The number of degrees of freedom.

    Returns:
        result: The standard deviation value.
    """
    return (sosd / (num - ddof)) ** 0.5


# ======================================================================
def i_mean(
        items,
        mean_=0.0,
        num=0):
    """
    Calculate the mean of arbitrary items.

    For iterative computation see:
     - `flyingcircus.util.next_mean()`
     - `flyingcircus.util.imean()`

    This is substantially faster than `statistics.mean()`.

    Args:
        items (Iterable): The input items.
            The iterable is not required to support `len()`.
        mean_ (Any): The start mean value.
        num (int): The number of items included in the start mean value.

    Returns:
        result: The mean of the items.

    Examples:
        >>> i_mean(range(0, 20, 2))
        9.0
    """
    for i, item in enumerate(items):
        mean_ = next_mean(item, mean_, i + num)
    return mean_


# ======================================================================
def i_var(
        items,
        mean_=0.0,
        var_=0.0,
        num=0):
    """
    Calculate the variance of arbitrary items.

    The length of `items`

    Internally uses `flyingcircus.util.next_mean_sosd()` and
    `flyingcircus.util.sosd2var()`.

    This is substantially faster than `statistics.variance()`.

    Args:
        items (Iterable): The input items.
            The iterable is not required to support `len()`.
        mean_ (Any): The start mean value.
        var_ (Any): The start variance value.
        num (int): The number of items included in the start mean value.

    Returns:
        result: The variance of the items.

    Examples:
        >>> i_var(range(0, 20, 2))
        33.0
    """
    i = 0
    sosd = var_ * num
    for i, item in enumerate(items):
        mean_, sosd = next_mean_sosd(item, mean_, sosd, i + num)
    return sosd2var(sosd, i + num + 1)


# ======================================================================
def i_mean_var(
        items,
        mean_=0.0,
        var_=0.0,
        num=0):
    """
    Calculate the variance of arbitrary items.

    For iterative computation see:
     - `flyingcircus.util.next_mean_var()`
     - `flyingcircus.util.next_mean_sosd()` and `.sosd2var()`.
     - `flyingcircus.util.ivar()`

    This is substantially faster than `statistics.variance()`.

    Args:
        items (Iterable): The input items.
            The iterable is not required to support `len()`.
        mean_ (Any): The start mean value.
        var_ (Any): The start variance value.
        num (int): The number of items included in the start mean value.

    Returns:
        result (tuple): The tuple
            contains:
             - `mean_`: The mean of the items.
             - `var_`: The variance of the items.
             - `num`: The number of items.

    Examples:
        >>> i_mean_var(range(0, 20, 2))
        (9.0, 33.0, 10)
    """
    i = 0
    sosd = var_ * num
    for i, item in enumerate(items):
        mean_, sosd = next_mean_sosd(item, mean_, sosd, i + num)
    return mean_, sosd2var(sosd, i + num + 1), i + num + 1


# ======================================================================
def i_stdev(
        items,
        mean_=0.0,
        var_=0.0,
        num=0,
        ddof=0):
    """
    Calculate the standard deviation of arbitrary items.

    For iterative computation see:
     - `flyingcircus.util.next_mean_var()`
     - `flyingcircus.util.next_mean_sosd()` and `.sosd2stdev()`.
     - `flyingcircus.util.istdev()`

    This is substantially faster than `statistics.stdev()`.

    Args:
        items (Iterable): The input items.
            The iterable is not required to support `len()`.
        mean_ (Any): The start mean value.
        var_ (Any): The start variance value.
        num (int): The number of items included in the start mean value.
        ddof (int): The number of degrees of freedom.

    Returns:
        result: The standard deviation of the items.

    Examples:
        >>> round(i_stdev(range(0, 20, 2)), 3)
        5.745
    """
    i = 0
    sosd = var_ * num
    for i, item in enumerate(items):
        mean_, sosd = next_mean_sosd(item, mean_, sosd, i + num)
    return sosd2stdev(sosd, i + num + 1, ddof)


# ======================================================================
def num_align(
        num,
        align='pow2',
        mode=1):
    """
    Align a number to a specified value, so as to make it multiple of it.

    The resulting number is computed using the formula:

    num = num + func(num % align / align) * align - num % align

    where `func` is a rounding function, as determined by `mode`.

    Args:
        num (int|float): The input number.
        align (int|float|str|None): The number to align to.
            If int, then calculate a multiple of `align` close to `num`.
            If str, possible options are:
             - 'powX' (where X >= 2 must be an int): calculate a power of X
               that is close to `num`.
            The exact number being calculated depends on the value of `mode`.
        mode (int|str): Determine which multiple to convert the number to.
            If str, valid inputs are:
             - 'upper': converts to the smallest multiple larger than `num`.
             - 'lower': converts to the largest multiple smaller than `num`.
             - 'closest': converts to the multiple closest to `num`.
            If int, valid inputs are:
             - '+1' has the same behavior as 'upper'.
             - '-1' has the same behavior as  'lower'.
             - '0' has the same behavior as  'closest'.

    Returns:
        num (int): The aligned number.

    Examples:
        >>> num_align(432)
        512
        >>> num_align(432, mode=-1)
        256
        >>> num_align(432, mode=0)
        512
        >>> num_align(447, 32, mode=1)
        448
        >>> num_align(447, 32, mode=-1)
        416
        >>> num_align(447, 32, mode=0)
        448
        >>> num_align(45, 90, mode=0)
        0
        >>> num_align(6, 'pow2', mode=0)
        8
        >>> num_align(128, 128, mode=1)
        128
        >>> num_align(123.37, 0.5, mode=1)
        123.5
        >>> num_align(123.37, 0.5, mode=0)
        123.5
        >>> num_align(123.37, 0.5, mode=-1)
        123.0
        >>> num_align(123.37, None)
        123.37
    """
    if mode == 'upper' or mode == +1:
        func = math.ceil
    elif mode == 'lower' or mode == -1:
        func = math.floor
    elif mode == 'closest' or mode == 0:
        func = round
    else:
        raise ValueError('Invalid mode `{mode}`'.format(mode=mode))

    if align:
        if isinstance(align, str):
            if align.startswith('pow'):
                base = int(align[len('pow'):])
                exp = math.log(num, base)
                num = int(base ** func(exp))
            else:
                raise ValueError('Invalid align `{align}`'.format(align=align))

        elif isinstance(align, (int, float)):
            modulus = num % align
            num += func(modulus / align) * align - modulus

        else:
            warnings.warn('Will not align `{num}` to `{align}`.'.format(
                num=num, align=align))

    return num


# ======================================================================
def merge_dicts(*dicts):
    """
    Merge dictionaries into a new dict (new keys overwrite the old ones).

    Args:
        dicts (*Iterable[dict]): Dictionaries to be merged together.

    Returns:
        merged (dict): The merged dict (new keys overwrite the old ones).

    Examples:
        >>> d1 = {1: 2, 3: 4, 5: 6}
        >>> d2 = {2: 1, 4: 3, 6: 5}
        >>> d3 = {1: 1, 3: 3, 6: 5}
        >>> dd = merge_dicts(d1, d2)
        >>> print(tuple(sorted(dd.items())))
        ((1, 2), (2, 1), (3, 4), (4, 3), (5, 6), (6, 5))
        >>> dd = merge_dicts(d1, d3)
        >>> print(tuple(sorted(dd.items())))
        ((1, 1), (3, 3), (5, 6), (6, 5))
    """
    merged = {}
    for item in dicts:
        merged.update(item)
    return merged


# =====================================================================
def p_ratio(x, y):
    """
    Calculate the pseudo-ratio of x, y: 1 / ((x / y) + (y / x))

    .. math::
        \\frac{1}{\\frac{x}{y}+\\frac{y}{x}} = \\frac{xy}{x^2+y^2}

    Args:
        x (int|float): First input value.
        y (int|float): Second input value.

    Returns:
        result: 1 / ((x / y) + (y / x))

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
def gen_p_ratio(*items):
    """
    Calculate the generalized pseudo-ratio of x_i: 1 / sum_ij [ x_i / x_j ]

    .. math::
        \\frac{1}{\\sum_{ij} \\frac{x_i}{x_j}}

    Args:
        *items (*Iterable[int|float|]): Input values.

    Returns:
        result: 1 / sum_ij [ x_i / x_j ]

    Examples:
        >>> gen_p_ratio(2, 2, 2, 2, 2)
        0.05
        >>> gen_p_ratio(200, 200, 200, 200, 200)
        0.05
        >>> gen_p_ratio(1, 2)
        0.4
        >>> gen_p_ratio(100, 200)
        0.4
        >>> items1 = [x * 10 for x in range(2, 10)]
        >>> items2 = [x * 1000 for x in range(2, 10)]
        >>> gen_p_ratio(*items1) - gen_p_ratio(*items2) < 1e-10
        True
        >>> items = list(range(2, 10))
        >>> gen_p_ratio(*items) - gen_p_ratio(*items[::-1]) < 1e-10
        True
    """
    return 1 / sum(x / y for x, y in itertools.permutations(items, 2))


# ======================================================================
def multi_replace(
        text,
        replaces):
    """
    Perform multiple replacements in a string.

    The replaces are concatenated together, therefore the order may matter.

    Args:
        text (str): The input string.
        replaces (tuple[tuple[str]]): The listing of the replacements.
            Format: ((<old>, <new>), ...).

    Returns:
        text (str): The string after the performed replacements.

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
    """
    return functools.reduce(lambda s, r: s.replace(*r), replaces, text)


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
    This version works for two Iterables.

    This is known as the `longest common substring` problem, or LCS for short.

    Args:
        seq1 (Iterable): The first input sequence.
            Must be of the same type as seq2.
        seq2 (Iterable): The second input sequence.
            Must be of the same type as seq1.
        sorting (callable): Sorting function passed to 'sorted' via `key` arg.

    Returns:
        commons (list[Iterable]): The longest common subsequence(s).

    Examples:
        >>> common_subseq_2('academy','abracadabra')
        ['acad']
        >>> common_subseq_2('los angeles','lossless')
        ['los', 'les']
        >>> common_subseq_2('los angeles','lossless',lambda x: x)
        ['les', 'los']
        >>> common_subseq_2((1, 2, 3, 4, 5),(0, 1, 2))
        [(1, 2)]
    """
    # note: [[0] * (len(seq2) + 1)] * (len(seq1) + 1) will not work!
    counter = [[0 for j in range(len(seq2) + 1)] for i in range(len(seq1) + 1)]
    longest = 0
    commons = []
    for i, item in enumerate(seq1):
        for j, jtem in enumerate(seq2):
            if item == jtem:
                tmp = counter[i][j] + 1
                counter[i + 1][j + 1] = tmp
                if tmp > longest:
                    commons = []
                    longest = tmp
                    commons.append(seq1[i - tmp + 1:i + 1])
                elif tmp == longest:
                    commons.append(seq1[i - tmp + 1:i + 1])
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
    This version works for an Iterable of Iterables.

    This is known as the `longest common substring` problem, or LCS for short.

    Args:
        seqs (Iterable[Iterable]): The input sequences.
            All the items must be of the same type.
        sorting (callable): Sorting function passed to 'sorted' via `key` arg.

    Returns:
        commons (list[Iterable]): The longest common subsequence(s).

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
    commons = [seqs[0]]
    for text in seqs[1:]:
        tmps = []
        for common in commons:
            tmp = common_subseq_2(common, text, sorting)
            if len(tmps) == 0 or len(tmp[0]) == len(tmps[0]):
                tmps.extend(common_subseq_2(common, text, sorting))
        commons = tmps
    return commons


# ======================================================================
def set_func_kws(
        func,
        func_kws):
    """
    Set keyword parameters of a function to specific or default values.

    Args:
        func (callable): The function to be inspected.
        func_kws (dict): The (key, value) pairs to set.
            If a value is None, it will be replaced by the default value.
            To use the names defined locally, use: `locals()`.

    Results:
        kws (dict): A dictionary of the keyword parameters to set.

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
    kws = {}
    for key in inspected.args:
        if key in func_kws:
            kws[key] = func_kws[key]
        elif key in defaults:
            kws[key] = defaults[key]
    return kws


# ======================================================================
def split_func_kws(
        func,
        func_kws):
    """
    Split a set of keywords into accepted and not accepted by some function.

    Args:
        func (callable): The function to be inspected.
        func_kws (dict): The (key, value) pairs to split.

    Results:
        result (tuple): The tuple
            contains:
             - kws (dict): The keywords NOT accepted by `func`.
             - func_kws (dict): The keywords accepted by `func`.

    See Also:
        inspect, locals, globals.
    """
    try:
        get_argspec = inspect.getfullargspec
    except AttributeError:
        get_argspec = inspect.getargspec
    inspected = get_argspec(func)
    kws = {k: v for k, v in func_kws.items() if k not in inspected.args}
    func_kws = {k: v for k, v in func_kws.items() if k in inspected.args}
    return func_kws, kws


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
        >>> with open(__file__, 'r') as file_obj:
        ...     content = file_obj.read()
        ...     content2 = ''.join([b for b in blocks(file_obj, 100)])
        ...     content == content2
        True
        >>> with open(__file__, 'rb') as file_obj:
        ...     content = file_obj.read()
        ...     content2 = b''.join([b for b in blocks(file_obj, 100)])
        ...     content == content2
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
        reset_offset=True,
        encoding=None):
    """
    Yields the data within a file in reverse order blocks of given (max) size.

    Note that:
     - the content of the block is NOT reversed.
     - if the file is open in text mode, the actual size of the block may be
       shorter for multi-byte encodings (such as `utf8`, which is the default).

    Args:
        file_obj (file): The input file.
        size (int|None): The block size.
            If int, the file is yielded in blocks of the specified size.
            If None, the file is yielded at once.
        reset_offset (bool): Reset the file offset.
            If True, starts reading from the end of the file.
            Otherwise, starts reading from where the file current position is.
        encoding (str|None): The encoding for correct block size computation.
            If `str`, must be a valid string encoding.
            If None, the default encoding is used.

    Yields:
        block (bytes|str): The data within the blocks.

    Examples:
        >>> with open(__file__, 'r') as file_obj:
        ...     content = file_obj.read()
        ...     content2 = ''.join([b for b in blocks_r(file_obj, 100)][::-1])
        ...     content == content2
        True
        >>> with open(__file__, 'rb') as file_obj:
        ...     content = file_obj.read()
        ...     content2 = b''.join([b for b in blocks_r(file_obj, 100)][::-1])
        ...     content == content2
        True
    """
    offset = 0
    if reset_offset:
        file_size = remaining_size = file_obj.seek(0, os.SEEK_END)
    else:
        file_size = remaining_size = file_obj.tell()
    rounding = 0
    while remaining_size > 0:
        offset = min(file_size, offset + size)
        file_obj.seek(file_size - offset)
        block = file_obj.read(min(remaining_size, size))
        if not isinstance(block, bytes):
            real_size = len(
                block.encode(encoding) if encoding else block.encode())
            rounding = len(block) - real_size
        remaining_size -= size
        yield block[:len(block) + rounding] if rounding else block


# ======================================================================
def xopen(
        the_file,
        *args,
        **kwargs):
    """
    Ensure that `the_file` is a file object, if a file path is provided.

    Args:
        the_file (str|bytes|file): The input file.
        *args: Positional arguments passed to `open()`.
        **kwargs: Keyword arguments passed to `open()`.

    Returns:
        the_file (file):
    """
    return open(the_file, *args, **kwargs) \
        if isinstance(the_file, (str, bytes)) else the_file


# ======================================================================
def hash_file(
        the_file,
        hash_algorithm=hashlib.md5,
        filtering=base64.urlsafe_b64encode,
        coding='ascii',
        block_size=64 * 1024):
    """
    Compute the hash of a file.

    Args:
        the_file (str|bytes|file): The input file.
            Can be either a valid file path or a file object.
            See `xopen()` for more details.
        hash_algorithm (callable): The hashing algorithm.
            This must support the methods provided by `hashlib` module, like
            `md5`, `sha1`, `sha256`, `sha512`.
        filtering (callable|None): The filtering function.
            If callable, must have the following signature:
            filtering(bytes) -> bytes|str.
            If None, no additional filering is performed.
        coding (str): The coding for converting the returning object to str.
            If str, must be a valid coding.
            If None, the object is kept as bytes.
        block_size (int|None): The block size.
            See `size` argument of `flyingcircus.util.blocks` for exact
            behavior.

    Returns:
        hash_key (str|bytes): The result of the hashing.
    """
    hash_obj = hash_algorithm()
    with xopen(the_file, 'rb') as file_obj:
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
            serializer(Any) -> bytes
        hash_algorithm (callable): The hashing algorithm.
            This must support the methods provided by `hashlib` module, like
            `md5`, `sha1`, `sha256`, `sha512`.
        filtering (callable|None): The filtering function.
            If callable, must have the following signature:
            filtering(bytes) -> bytes.
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
        func_kws=None,
        dirpath=None,
        filename='{hash_key}.p',
        save_func=pickle.dump,
        load_func=pickle.load,
        force=False):
    """
    Compute or load from cache the result of a computation.

    Args:
        func (callable): The computation to perform.
        func_kws (dict|None): Keyword arguments passed to `func`.
        dirpath (str): The path of the caching directory.
        filename (str): The filename of the caching file.
            This is processed by `format` with `locals()`.
        save_func (callable): The function used to save caching file.
            Must have the following signature:
            save_func(file_obj, Any) -> None
            The value returned from `save_func` is not used.
        load_func (callable): The function used to load caching file.
            Must have the following signature:
            load_func(file_obj) -> Any
        force (bool): Force the calculation, regardless of caching state.

    Returns:
        result (Any): The result of the cached computation.
    """
    func_kws = dict(func_kws) if func_kws else {}
    hash_key = hash_object((func, func_kws))
    filepath = os.path.join(dirpath, filename.format(**locals()))
    if os.path.isfile(filepath) and not force:
        result = load_func(open(filepath, 'rb'))
    else:
        result = func(**func_kws)
        save_func(open(filepath, 'wb'), result)
    return result


# ======================================================================
def readline(
        file_obj,
        reverse=False,
        skip_empty=True,
        append_newline=True,
        block_size=64 * 1024,
        reset_offset=True,
        encoding=None):
    """
    Flexible function for reading lines incrementally.

    Args:
        file_obj (file): The input file.
        reverse (bool): Read the file in reverse mode.
            If True, the lines will be read in reverse order.
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
        encoding (str|None): The encoding for correct block size computation.
            If `str`, must be a valid string encoding.
            If None, the default encoding is used.
            This is passed to `blocks_r()`. Only used when `reverse` is True.

    Yields:
        line (str|bytes): The next line.

    Examples:
        >>> with open(__file__, 'rb') as file_obj:
        ...     lines = [l for l in readline(file_obj, False)]
        ...     lines_r = [l for l in readline(file_obj, True)][::-1]
        ...     lines == lines_r
        True
    """
    is_bytes = isinstance(file_obj.read(0), bytes)
    newline = b'\n' if is_bytes else '\n'
    empty = b'' if is_bytes else ''
    remainder = empty
    block_generator_kws = dict(size=block_size, reset_offset=reset_offset)
    if not reverse:
        block_generator = blocks
    else:
        block_generator = blocks_r
        block_generator_kws.update(dict(encoding=encoding))
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
        command,
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
        command (str|Iterable[str]): The command to execute.
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

    command, is_valid = which(command)
    if is_valid:
        msg('{} {}'.format('$$' if dry else '>>', ' '.join(command)),
            verbose, D_VERB_LVL if dry else VERB_LVL['medium'])
    else:
        msg('W: `{}` is not in available in $PATH.'.format(command[0]))

    if not dry and is_valid:
        if in_pipe is not None:
            msg('< {}'.format(in_pipe),
                verbose, VERB_LVL['highest'])

        proc = subprocess.Popen(
            command,
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
                msg(out_buff, fmt='', end='')
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
                msg(p_stdout, verbose, VERB_LVL['high'], fmt='')
            if p_stderr:
                msg(p_stderr, verbose, VERB_LVL['high'], fmt='')
            ret_code = proc.wait()
        else:
            proc.kill()
            msg('E: mode `{}` and `in_pipe` not supported.'.format(mode))

        if log:
            name = os.path.basename(command[0])
            pid = proc.pid
            for stream, source in ((p_stdout, 'out'), (p_stderr, 'err')):
                if stream:
                    log_filepath = log.format(**locals())
                    with open(log_filepath, 'wb') as fileobj:
                        fileobj.write(stream.encode(encoding))
    return ret_code, p_stdout, p_stderr


# ======================================================================
def parallel_execute(
        commands,
        pool_size=None,
        poll_interval=60,
        callback=None,
        callback_args=None,
        callback_kws=None,
        verbose=D_VERB_LVL):
    """
    Spawn parallel processes and wait until all processes are completed.

    Args:
        commands (Iterable[str|Iterable[str]): The commands to execute.
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
        callback_kws (dict|tuple|None): Keyword arguments for `callback()`.
        verbose (int): Set level of verbosity.

    Returns:
        None.
    """
    callback_args = tuple(callback_args) if callback_args is not None else ()
    callback_kws = dict(callback_args) if callback_kws is not None else {}
    if not pool_size:
        pool_size = multiprocessing.cpu_count() + 1
    num_total = len(commands)
    num_processed = 0
    begin_dt = datetime.datetime.now()
    procs = [
        subprocess.Popen(cmd, shell=True) for cmd in commands[:pool_size]]
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
                for cmd in commands[num_submitted:num_submitted + num_done]]
            for proc in new_procs:
                msg('{} {}'.format('>>', ' '.join(proc.args)),
                    verbose, VERB_LVL['medium'])
            num_submitted += len(new_procs)
            procs += new_procs
            num_running = len(procs)
        elapsed_dt = datetime.datetime.now() - begin_dt
        text = ('I: {num_processed} / {num_total} processed'
                ' ({num_running} running) - Elapsed: {elapsed_dt}'
                ).format(**locals())
        msg(text, verbose, D_VERB_LVL)
        if callable(callback):
            callback(*callback_args, **callback_kws)
        if num_processed == num_total:
            done = True
        else:
            time.sleep(poll_interval)


# ======================================================================
def realpath(path):
    """
    Get the expanded absolute path from its short or relative counterpart.

    Args:
        path (str): The path to expand.

    Returns:
        new_path (str): the expanded path.

    Raises:
        OSError: if the expanded path does not exists.
    """
    new_path = os.path.abspath(os.path.realpath(os.path.expanduser(path)))
    if not os.path.exists(new_path):
        raise OSError
    return new_path


# ======================================================================
def listdir(
        path,
        file_ext='',
        full_path=True,
        is_sorted=True,
        verbose=D_VERB_LVL):
    """
    Retrieve a sorted list of files matching specified extension and pattern.

    Args:
        path (str): Path to search.
        file_ext (str|None): File extension. Empty string for all files.
            None for directories.
        full_path (bool): Include the full path.
        is_sorted (bool): Sort results alphabetically.
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
    if is_sorted:
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
        re_kws (dict|None): Keyword arguments passed to `re.compile()`.
        walk_kws (dict|None): Keyword arguments passed to `os.walk()`.

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
        re_kws (dict|None): Keyword arguments passed to `re.compile()`.
        walk_kws (dict|None): Keyword arguments passed to `os.walk()`.

    Returns:
        filepaths (list[str]): The matched filepaths.
    """
    return [
        item for item in iflistdir(
            patterns=patterns, dirpath=dirpath, unix_style=unix_style,
            re_kws=re_kws, walk_kws=walk_kws)]


# ======================================================================
def add_extsep(ext):
    """
    Add a extsep char to a filename extension, if it does not have one.

    Args:
        ext (str|None): Filename extension to which the dot has to be added.

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
    ext = (
            ('' if ext and ext.startswith(
                os.path.extsep) else os.path.extsep) +
            (ext if ext else ''))
    return ext


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

    See Also:
        util.join_path(), util.multi_split_path()
    """
    root, base_ext = os.path.split(filepath)
    base, ext = split_ext(base_ext, auto_multi_ext=auto_multi_ext)
    return root, base, ext


# ======================================================================
def multi_split_path(
        filepath,
        auto_multi_ext=True):
    """
    Split the filepath into (root, base, ext).

    Note that: os.path.sep.join(*dirs, base) + ext == path.
    (and therfore: ''.join(dirs) + base + ext != path).

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

    See Also:
        util.join_path(), util.split_path()
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
def join_path(*args):
    """
    Join a list of items into a filepath.

    The last item is treated as the file extension.
    Path and extension separators do not need to be manually included.

    Note that this is the inverse of `split_path()`.

    Args:
        *args (*Iterable[str]): The path elements to be concatenated.
            The last item is treated as the file extension.

    Returns:
        filepath (str): The output filepath.

    Examples:
        >>> join_path('/path/to', 'file', '.txt')
        '/path/to/file.txt'
        >>> join_path('/path/to', 'file', '.tar.gz')
        '/path/to/file.tar.gz'
        >>> join_path('', 'file', '.tar.gz')
        'file.tar.gz'
        >>> join_path('path/to', 'file', '')
        'path/to/file'
        >>> paths = [
        ...     '/path/to/file.txt', '/path/to/file.tar.gz', 'file.tar.gz']
        >>> all([path == join_path(*split_path(path)) for path in paths])
        True
        >>> paths = [
        ...     '/path/to/file.txt', '/path/to/file.tar.gz', 'file.tar.gz']
        >>> all([path == join_path(*multi_split_path(path)) for path in paths])
        True

    See Also:
        util.split_path(), util.multi_split_path()
    """
    return ((os.path.join(*args[:-1]) if args[:-1] else '') +
            (add_extsep(args[-1]) if args[-1] else ''))


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
        filepath (str): The input filepath.
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
            filepath = out_template.format(**locals())
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
        group_consecutive (str): Group consecutive non-allowed.
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
def auto_open(filepath, *args, **kwargs):
    """
    Auto-magically open a compressed file.

    Supports `gzip` and `bzip2`.

    Note: all compressed files should be opened as binary.
    Opening in text mode is not supported.

    Args:
        filepath (str): The file path.
        *args (Iterable): Positional arguments passed to `open()`.
        **kwargs (dict): Keyword arguments passed to `open()`.

    Returns:
        file_obj: A file object.

    Raises:
        IOError: on failure.

    See Also:
        open(), gzip.open(), bz2.open()

    Examples:
        >>> file_obj = auto_open(__file__, 'rb')
    """
    zip_module_names = 'gzip', 'bz2'
    file_obj = None
    for zip_module_name in zip_module_names:
        try:
            zip_module = importlib.import_module(zip_module_name)
            file_obj = zip_module.open(filepath, *args, **kwargs)
            file_obj.read(1)
        except (OSError, IOError, AttributeError, ImportError):
            file_obj = None
        else:
            file_obj.seek(0)
            break
    if not file_obj:
        file_obj = open(filepath, *args, **kwargs)
    return file_obj


# ======================================================================
def zopen(filepath, mode='rb', *args, **kwargs):
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
        *args (Iterable): Positional arguments passed to `open()`.
        **kwargs (dict): Keyword arguments passed to `open()`.

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
    file_obj = open(filepath, mode=mode, *args, **kwargs)

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
                    head[:2] == b'BZ' and head[2:3] == b'h' and head[
                                                                3:4].isdigit()
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
def has_decorator(
        text,
        pre_decor='"',
        post_decor='"'):
    """
    Determine if a string is delimited by some characters (decorators).

    Args:
        text (str): The text input string.
        pre_decor (str): initial string decorator.
        post_decor (str): final string decorator.

    Returns:
        has_decorator (bool): True if text is delimited by the specified chars.

    Examples:
        >>> has_decorator('"test"')
        True
        >>> has_decorator('"test')
        False
        >>> has_decorator('<test>', '<', '>')
        True
    """
    return text.startswith(pre_decor) and text.endswith(post_decor)


# ======================================================================
def strip_decorator(
        text,
        pre_decor='"',
        post_decor='"'):
    """
    Strip initial and final character sequences (decorators) from a string.

    Args:
        text (str): The text input string.
        pre_decor (str): initial string decorator.
        post_decor (str): final string decorator.

    Returns:
        text (str): the text without the specified decorators.

    Examples:
        >>> strip_decorator('"test"')
        'test'
        >>> strip_decorator('"test')
        'test'
        >>> strip_decorator('<test>', '<', '>')
        'test'
    """
    begin = len(pre_decor) if text.startswith(pre_decor) else None
    end = -len(post_decor) if text.endswith(post_decor) else None
    return text[begin:end]


# ======================================================================
def to_bool(
        text,
        mappings=(('false', 'true'), ('0', '1'), ('off', 'on')),
        case_sensitive=False,
        strip=True):
    """
    Conversion to boolean value.

    This is especially useful to interpret strings are booleans, because
    the built-in `bool()` method evaluates to False for empty strings and
    True for non-empty strings.

    Args:
        text (str|Any): The input value.
            If not string, attempt the built-in `bool()` casting.
        mappings (Iterable[Iterable]): The string values to map as boolean.
            Each item consists of an iterable. Within the inner iterable,
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
    if isinstance(text, str):
        if strip:
            text = text.strip()
        if not case_sensitive:
            text = text.lower()
            mappings = tuple(
                tuple(match.lower() for match in mapping)
                for mapping in mappings)
        for mapping in mappings:
            for i, match in enumerate(mapping):
                if text == match:
                    return bool(i)
        else:
            raise ValueError('Cannot convert to bool')
    else:
        return bool(text)


# ======================================================================
def auto_convert(
        text,
        pre_decor=None,
        post_decor=None,
        casts=(int, float, complex, to_bool)):
    """
    Convert value to numeric if possible, or strip delimiters from string.

    Args:
        text (str|Number): The text input string.
        pre_decor (str): initial string decorator.
        post_decor (str): final string decorator.
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
        if pre_decor and post_decor and \
                has_decorator(text, pre_decor, post_decor):
            text = strip_decorator(text, pre_decor, post_decor)
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
def is_number(var):
    """
    Determine if a variable contains a number.

    Args:
        var (str): The variable to test.

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
        complex(var)
    except (TypeError, ValueError):
        result = False
    else:
        result = True
    return result


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
    order = int(math.floor(math.log10(abs(val)))) if abs(val) != 0.0 else 0
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
    val_order = int(math.floor(math.log10(abs(val)))) if val != 0 else 0
    err_order = int(math.floor(math.log10(abs(err)))) if err != 0 else 0
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
def guess_numerical_sequence(
        items,
        rounding=3):
    """
    Guess a compact expression for a numerical sequence.

    Args:
        items (Iterable[Number]): The input items.
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
    tol = 10 ** -min([guess_decimals(item) for item in items if item])
    result = None
    diffs = tuple(diff(items))
    base = diffs[0]
    if all(x - base < tol for x in diffs):
        if base < tol:
            # : constant sequence
            result = '[{}] * {}'.format(round(items[0], rounding), len(items))
        else:
            # : linear sequence
            result = 'range({}, {}, {})'.format(
                round(items[0], rounding), round(items[-1] + base, rounding),
                round(base, rounding))
    else:
        divs = tuple(div(items))
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
                firsts.append(math.log2(items[0]) / math.log2(new_base))
                lasts.append(
                    math.log2(items[-1] * new_base) / math.log2(new_base))
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
def str2dict(
        in_str,
        entry_sep=',',
        key_val_sep='=',
        pre_decor='{',
        post_decor='}',
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
        pre_decor (str): initial decorator (to be removed before parsing).
        post_decor (str): final decorator (to be removed before parsing).
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
    if has_decorator(in_str, pre_decor, post_decor):
        in_str = strip_decorator(in_str, pre_decor, post_decor)
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
            key = None
        # strip dict key
        key = key.strip(strip_key_str)
        # add to dictionary
        if key:
            if convert:
                val = auto_convert(val)
            out_dict[key] = val
    return out_dict


# ======================================================================
def dict2str(
        in_dict,
        entry_sep=',',
        key_val_sep='=',
        pre_decor='{',
        post_decor='}',
        strip_key_str=None,
        strip_val_str=None,
        sorting=None):
    """
    Convert a dictionary to a string.

    Args:
        in_dict (dict): The input dictionary.
        entry_sep (str): The entry separator.
        key_val_sep (str): The key-value separator.
        pre_decor (str): initial decorator (to be appended to the output).
        post_decor (str): final decorator (to be appended to the output).
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
    out_str = pre_decor + entry_sep.join(out_list) + post_decor
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
    if strict:
        result = all(x < y for x, y in zip(items, items[1:]))
    else:
        result = all(x <= y for x, y in zip(items, items[1:]))
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
    if strict:
        result = all(x > y for x, y in zip(items, items[1:]))
    else:
        result = all(x >= y for x, y in zip(items, items[1:]))
    return result


# ======================================================================
def is_same_sign(items):
    """
    Determine if all items in an Iterable have the same sign.

    Args:
        items (Iterable): The items to check.
            The comparison operators '>=' and '<' must be defined.

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
    """
    return all(item >= 0 for item in items) or all(item < 0 for item in items)


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
        rounding=True):
    """
    Scale a value by the specified size.

    Args:
        val (Any): The value to scale.
        scale (Any): The scale size.
        rounding (bool): Perform rounding.
            If True, call `round()` before int conversion.

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
        >>> scale_to_int(0.5, 11.0, False)
        5
        >>> scale_to_int(1, 10.0)
        1
    """
    return int(round(val * scale) if rounding else (val * scale)) \
        if not isinstance(val, int) else val


# ======================================================================
def multi_scale_to_int(
        vals,
        scales,
        shape=(None, 2),
        combine=None):
    """
    Ensure values scaling of multiple values.

    Args:
        vals (int|float|Iterable[int|float|Iterable]): The input value(s)
            If Iterable, a value for each scale must be specified.
            If not Iterable, all pairs will have the same value.
            If any value is int, it is not scaled further.
            If any value is float, it is scaled to the corresponding scale,
            if `combine` is None, otherwise it is scaled to a combined scale
            according to the result of `combine(scales)`.
        scales (Iterable[int]): The scale sizes for the pairs.
        shape (Iterable[int|None]): The shape of the output.
            It must be a 2-tuple of int or None.
            None entries are replaced by `len(scales)`.
        combine (callable|None): The function for combining pad width scales.
            Must accept: combine(Iterable[int]) -> int|float
            This is used to compute a reference scaling value for the
            float to int conversion, using `combine(scales)`.
            For the int values of `width`, this parameter has no effect.
            If None, uses the corresponding scale from the scales.

    Returns:
        pad_width (int|tuple[tuple[int]]): The absolute `pad_width`.
            If input `pad_width` is not Iterable, result is not Iterable.

    See Also:
        - np.pad()
        - flyingcircus.num.padding()

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
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    import doctest  # Test interactive Python examples

    msg(__doc__.strip())
    doctest.testmod()
    msg(report())
