#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyMRT: code that is now deprecated but can still be useful for legacy scripts.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import sys  # System-specific parameters and functions
import math  # Mathematical functions
import functools  # Higher-order functions and operations on callable objects
import doctest  # Test interactive Python examples
import string  # Common string operations
import itertools  # Functions creating iterators for efficient looping

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
import flyingcircus as fc  # Everything you always wanted to have in Python*

# :: External Imports Submodules
import scipy.optimize  # SciPy: Optimization Algorithms
import scipy.signal  # SciPy: Signal Processing

from flyingcircus import INFO, PATH
from flyingcircus import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from flyingcircus import elapsed, report
from flyingcircus import msg, dbg, fmt, fmtm
from flyingcircus import HAS_JIT, jit


# ======================================================================
def tty_colorify(
        text,
        color=None):
    """
    Add color TTY-compatible color code to a string, for pretty-printing.

    DEPRECATED! (use `blessed` module)

    Args:
        text (str): The text to color.
        color (str|int|None): Identifier for the color coding.
            Lowercase letters modify the forground color.
            Uppercase letters modify the background color.
            Available colors:

             - r/R: red
             - g/G: green
             - b/B: blue
             - c/C: cyan
             - m/M: magenta
             - y/Y: yellow (brown)
             - k/K: black (gray)
             - w/W: white (gray)

    Returns:
        text (str): The colored text.

    See also:
        tty_colors
    """
    tty_colors = {
        'r': 31, 'g': 32, 'b': 34, 'c': 36, 'm': 35, 'y': 33, 'w': 37, 'k': 30,
        'R': 41, 'G': 42, 'B': 44, 'C': 46, 'M': 45, 'Y': 43, 'W': 47, 'K': 40,
    }

    if color in tty_colors:
        tty_color = tty_colors[color]
    elif color in tty_colors.values():
        tty_color = color
    else:
        tty_color = None
    if tty_color and sys.stdout.isatty():
        return '\x1b[1;{color}m{}\x1b[1;m'.format(text, color=tty_color)
    else:
        return text


# ======================================================================
def is_prime_verbose(val):
    """
    Determine if num is a prime number.

    A prime number is only divisible by 1 and itself.
    0 and 1 are considered special cases; in this implementations they are
    considered primes.

    It is implemented by directly testing for possible factors.

    Args:
        val (int): The number to check for primality.
            Only works for numbers larger than 1.

    Returns:
        is_divisible (bool): The result of the primality.

    Examples:
        >>> is_prime_verbose(100)
        False
        >>> is_prime_verbose(101)
        True
        >>> is_prime_verbose(-100)
        False
        >>> is_prime_verbose(-101)
        True
        >>> is_prime_verbose(2 ** 17)
        False
        >>> is_prime_verbose(17 * 19)
        False
        >>> is_prime_verbose(2 ** 17 - 1)
        True
        >>> is_prime_verbose(0)
        False
        >>> is_prime_verbose(1)
        False
    """
    # : verbose implementation (skip 2 multiples!)
    is_divisible = val == 1 or (val != 2 and not (val % 2))
    i = 3
    while not is_divisible and i * i < val:
        is_divisible = not (val % i)
        # only odd factors needs to be tested
        i += 2
    return not is_divisible


# ======================================================================
def is_prime_optimized(val):
    """
    Determine if num is a prime number.

    A prime number is only divisible by 1 and itself.
    0 and 1 are considered special cases; in this implementations they are
    considered primes.

    It is implemented by directly testing for possible factors.

    Args:
        val (int): The number to check for primality.
            Only works for numbers larger than 1.

    Returns:
        is_divisible (bool): The result of the primality.

    Examples:
        >>> is_prime_optimized(100)
        False
        >>> is_prime_optimized(101)
        True
        >>> is_prime_optimized(-100)
        False
        >>> is_prime_optimized(-101)
        True
        >>> is_prime_optimized(2 ** 17)
        False
        >>> is_prime_optimized(17 * 19)
        False
        >>> is_prime_optimized(2 ** 17 - 1)
        True
        >>> is_prime_optimized(0)
        True
        >>> is_prime_optimized(1)
        True
    """
    # : optimized implementation (skip 2 multiples!)
    if val < 0:
        val = -val
    if not (val % 2) and val > 2:
        return False
    for i in range(3, int(val ** 0.5) + 1, 2):
        if not (val % i):
            return False
    return True


# ======================================================================
def interval_size(interval):
    """
    Calculate the (signed) size of an interval given as a 2-tuple (A,B)

    DEPRECATED! (by `numpy.ptp()`)

    Args:
        interval (float,float): Interval for computation

    Returns:
        val (float): The converted value

    Examples:
        >>> interval_size((0, 1))
        1
    """
    return interval[1] - interval[0]


# ======================================================================
def replace_iter(
        items,
        condition,
        replace=None,
        cycle=True):
    """
    Replace items matching a specific condition.

    This is fairly useless:

    If `replace` is callable:
        replace_iter(items, condition, replace)
    becomes:
        [replace(x) if condition(x) else x for x in items]

    If `replace` is not Iterable:
        list(replace_iter(items, condition, replace))
    becomes:
        [replace if condition(x) else x for x in items]

    If `replace` is Iterable and cycle == False:
        list(replace_iter(items, condition, replace))
    becomes:
        iter_replace = iter(replace)
        [next(iter_replace) if condition(x) else x for x in items]

    If `replace` is Iterable and cycle == True:
        list(replace_iter(items, condition, replace))
    becomes:
        iter_replace = itertools.cycle(replace)
        [next(iter_replace) if condition(x) else x for x in items]


    Args:
        items (Iterable): The input items.
        condition (callable): The condition for the replacement.
        replace (any|Iterable|callable): The replacement.
            If Iterable, its elements are used for replacement.
            If callable, it is applied to the elements matching `condition`.
            Otherwise, the object itself is used.
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

        >>> ll = list(range(10))
        >>> (list(replace_iter(ll, lambda x: x % 2 == 0, lambda x: x ** 2))
        ...     == [x ** 2 if x % 2 == 0 else x for x in ll])
        True
        >>> (list(replace_iter(ll, lambda x: x % 2 == 0, 'X'))
        ...     == ['X' if x % 2 == 0 else x for x in ll])
        True
        >>> iter_ascii_letters = iter(string.ascii_letters)
        >>> (list(replace_iter(ll, lambda x: x % 2 == 0, string.ascii_letters))
        ...     == [next(iter_ascii_letters) if x % 2 == 0 else x for x in ll])
        True
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
            try:
                yield replace(item) if callable(replace) else next(replace)
            except StopIteration:
                return


# ======================================================================
def binomial_coeff(
        n,
        k):
    """
    Compute the binomial coefficient.

    DEPRECATED! (by `scipy.special.comb(exact=True)`)

    This is similar to `scipy.special.binom()` and identical to
    `scipy.special.comb(exact=True)` except that the `scipy` version
    is faster.

    If all binomial coefficient for a given `n` are required, then
    `flyingcircus.base.get_pascal_numbers()` is computationally more efficient.

    Args:
        n (int): The major index of the binomial coefficient.
        k (int): The minor index of the binomial coefficient.

    Returns:
        value (int): The binomial coefficient.
            If `k > n` returns 0.

    Examples:
        >>> binomial_coeff(10, 5)
        252
        >>> binomial_coeff(50, 25)
        126410606437752
        >>> N = 10
        >>> for n in range(N):
        ...     print([binomial_coeff(n, k) for k in range(n + 1)])
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
        >>> binomial_coeff(0, 0)
        1
        >>> binomial_coeff(0, 1)
        Traceback (most recent call last):
            ...
        ValueError: Invalid values `n=0` `k=1` (0 <= k <= n)
        >>> binomial_coeff(1, 0)
        1
        >>> binomial_coeff(1, 1)
        1
        >>> binomial_coeff(1, 2)
        Traceback (most recent call last):
            ...
        ValueError: Invalid values `n=1` `k=2` (0 <= k <= n)
        >>> from scipy.special import binom
        >>> N = 15
        >>> all(binomial_coeff(n, k) == int(binom(n, k))
        ...     for n in range(N) for k in range(n + 1))
        True

    See Also:
        - flyingcircus.base.get_pascal_numbers()
        - flyingcircus.base.pascal_triangle()
        - https://en.wikipedia.org/wiki/Binomial_coefficient
        - https://en.wikipedia.org/wiki/Pascal%27s_triangle
    """
    if not 0 <= k <= n:
        text = 'Invalid values `n={}` `k={}` (0 <= k <= n)'.format(n, k)
        raise ValueError(text)
    value = 1
    for i in range(n + 1):
        if i == k or i == n - k:
            break
        value = value * (n - i) // (i + 1)
    return value


# ======================================================================
def is_prime_binomial(val):
    """
    Determine if number is prime.

    WARNING! DO NOT USE THIS ALGORITHM AS IT IS EXTREMELY INEFFICIENT!

    A prime number is only divisible by 1 and itself.
    0 and 1 are considered special cases; in this implementations they are
    considered primes.

    It is implemented by using the binomial triangle primality test.
    This is known to be extremely inefficient.

    Args:
        val (int): The number to check for primality.
            Only works for numbers larger than 1.

    Returns:
        is_divisible (bool): The result of the primality.

    Examples:
        >>> is_prime_binomial(100)
        False
        >>> is_prime_binomial(101)
        True
        >>> is_prime_binomial(-100)
        False
        >>> is_prime_binomial(-101)
        True
        >>> is_prime_binomial(2 ** 17)
        False
        >>> is_prime_binomial(17 * 19)
        False
        >>> is_prime_binomial(2 ** 13 - 1)
        True
        >>> is_prime_binomial(0)
        True
        >>> is_prime_binomial(1)
        True

    See Also:
        - flyingcircus.base.is_prime()
        - flyingcircus.base.primes_range()
        - https://en.wikipedia.org/wiki/Prime_number
    """
    if val < 0:
        val = -val
    if val in (0, 1):
        return True
    if not ((val % 2 and val > 2) and (val % 3 and val > 3)):
        return False
    elif val == 2 or val == 3:
        return True
    elif all(
            not (n % val) for n in fc.get_binomial_coeffs(val, full=False)
            if n > 1):
        return True
    else:
        return False


# ======================================================================
def is_prime_wheel(
        num,
        wheel=None):
    """
    Determine if a number is prime.

    This uses a wheel factorization implementation.

    A prime number is only divisible by 1 and itself.
    0 and 1 are considered special cases; in this implementations they are
    considered primes.

    It is implemented by testing for possible factors using wheel increment.

    Args:
        num (int): The number to check for primality.
            Only works for numbers larger than 1.
        wheel (int|Sequence[int]|None): The generators of the wheel.
            If int, this is the number of prime numbers to use (must be > 1),
            generated using `flyingcircus.base.gen_primes()`.
            If Sequence, it must consist of the first N prime numbers in
            increasing order.
            If None, uses a hard-coded (2, 3) wheel, which is faster
            for smaller inputs.

    Returns:
        is_divisible (bool): The result of the primality.

    Examples:
        >>> is_prime_wheel(100)
        False
        >>> is_prime_wheel(101)
        True
        >>> is_prime_wheel(-100)
        False
        >>> is_prime_wheel(-101)
        True
        >>> is_prime_wheel(2 ** 17)
        False
        >>> is_prime_wheel(17 * 19)
        False
        >>> is_prime_wheel(2 ** 17 - 1)
        True
        >>> is_prime_wheel(2 ** 31 - 1)
        True
        >>> is_prime_wheel(2 ** 17 - 1, 2)
        True
        >>> is_prime_wheel(2 ** 17 - 1, (2, 3))
        True
        >>> is_prime_wheel(2 ** 17 - 1, (2, 3, 5))
        True
        >>> is_prime_wheel(0)
        True
        >>> is_prime_wheel(1)
        True

    See Also:
        - flyingcircus.base.is_prime()
        - flyingcircus.base.primes_range()

    References:
        - https://en.wikipedia.org/wiki/Prime_number
        - https://en.wikipedia.org/wiki/Trial_division
        - https://en.wikipedia.org/wiki/Wheel_factorization
    """
    # : fastest implementation (skip both 2 and 3 multiples!)
    if num < 0:
        num = -num
    if wheel is None:
        wheel = (2, 3)
    elif isinstance(wheel, int):
        wheel = list(fc.get_primes(wheel, 2))
    else:
        wheel = tuple(sorted(wheel))
    for k in wheel:
        if not num % k:
            return num <= k
    prod_wheel = fc.prod(wheel)
    coprimes = tuple(
        n for n in range(2, prod_wheel + 2)
        if all(math.gcd(n, k) == 1 for k in wheel))
    deltas = tuple(fc.diff(coprimes + (coprimes[0] + prod_wheel,)))
    len_deltas = len(deltas)
    j = 0
    i = coprimes[0]
    while i * i <= num:
        if not (num % i):
            return False
        i += deltas[j]
        j += 1
        j %= len_deltas
    return True


# ======================================================================
def ndstack(arrs, axis=-1):
    """
    Stack a list of arrays of the same size along a specific axis.

    DEPRECATED! (by `numpy.stack()`)

    Args:
        arrs (list[ndarray]): A list of (N-1)-dim arrays of the same size.
        axis (int): Direction for the concatenation of the arrays.

    Returns:
        arr (ndarray): The concatenated N-dim array.
    """
    arr = arrs[0]
    n_dim = arr.ndim + 1
    if axis < 0:
        axis += n_dim
    if axis < 0:
        axis = 0
    if axis > n_dim:
        axis = n_dim
    # calculate new shape
    shape = arr.shape[:axis] + tuple([len(arrs)]) + arr.shape[axis:]
    # stack arrays together
    arr = np.zeros(shape, dtype=arr.dtype)
    for i, src in enumerate(arrs):
        index = [slice(None)] * n_dim
        index[axis] = i
        arr[tuple(index)] = src
    return arr


# ======================================================================
def ndsplit(arr, axis=-1):
    """
    Split an array along a specific axis into a list of arrays

    DEPRECATED! (by `numpy.split()`)

    Args:
        arr (ndarray): The N-dim array to split.
        axis (int): Direction for the splitting of the array.

    Returns:
        arrs (list[ndarray]): A list of (N-1)-dim arrays of the same size.
    """
    # split array apart
    arrs = []
    for i in range(arr.shape[axis]):
        # determine index for slicing
        index = [slice(None)] * arr.ndim
        index[axis] = i
        arrs.append(arr[index])
    return arrs


# ======================================================================
def slice_array(
        arr,
        axis=0,
        index=None):
    """
    Slice a (N-1)-dim sub-array from an N-dim array.

    DEPRECATED! (Use advanced `numpy` slicing instead!)

    Args:
        arr (np.ndarray): The input N-dim array
        axis (int): The slicing axis.
        index (int): The slicing index.
            If None, mid-value is taken.

    Returns:
        sliced (np.ndarray): The sliced (N-1)-dim sub-array

    Raises:
        ValueError: if index is out of bounds

    Examples:
        >>> arr = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        >>> slice_array(arr, 2, 1)
        array([[ 1,  5,  9],
               [13, 17, 21]])
        >>> slice_array(arr, 1, 2)
        array([[ 8,  9, 10, 11],
               [20, 21, 22, 23]])
        >>> slice_array(arr, 0, 0)
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])
        >>> slice_array(arr, 0, 1)
        array([[12, 13, 14, 15],
               [16, 17, 18, 19],
               [20, 21, 22, 23]])
    """
    # initialize slice index
    slab = [slice(None)] * arr.ndim
    # ensure index is meaningful
    if index is None:
        index = np.int(arr.shape[axis] / 2.0)
    # check index
    if (index >= arr.shape[axis]) or (index < 0):
        raise ValueError('Invalid array index in the specified direction')
    # determine slice index
    slab[axis] = index
    # slice the array
    return arr[tuple(slab)]


# ======================================================================
def sequence(
        start,
        stop,
        step=None,
        precision=None):
    """
    Generate a sequence that steps linearly from start to stop.

    Args:
        start (int|float): The starting value.
        stop (int|float): The final value.
            This value is present in the resulting sequence only if the step is
            a multiple of the interval size.
        step (int|float): The step value.
            If None, it is automatically set to unity (with appropriate sign).
        precision (int): The number of decimal places to use for rounding.
            If None, this is estimated from the `step` paramenter.

    Yields:
        item (int|float): the next element of the sequence.

    Examples:
        >>> list(sequence(0, 1, 0.1))
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        >>> list(sequence(0, 1, 0.3))
        [0.0, 0.3, 0.6, 0.9]
        >>> list(sequence(0, 10, 1))
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> list(sequence(0.4, 4.6, 0.72))
        [0.4, 1.12, 1.84, 2.56, 3.28, 4.0]
        >>> list(sequence(0.4, 4.72, 0.72, 2))
        [0.4, 1.12, 1.84, 2.56, 3.28, 4.0, 4.72]
        >>> list(sequence(0.4, 4.72, 0.72, 4))
        [0.4, 1.12, 1.84, 2.56, 3.28, 4.0, 4.72]
        >>> list(sequence(0.4, 4.72, 0.72, 1))
        [0.4, 1.1, 1.8, 2.6, 3.3, 4.0, 4.7]
        >>> list(sequence(0.73, 5.29))
        [0.73, 1.73, 2.73, 3.73, 4.73]
        >>> list(sequence(-3.5, 3.5))
        [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
        >>> list(sequence(3.5, -3.5))
        [3.5, 2.5, 1.5, 0.5, -0.5, -1.5, -2.5, -3.5]
        >>> list(sequence(10, 1, -1))
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
        >>> list(sequence(10, 1, 1))
        []
        >>> list(sequence(10, 20, 10))
        [10, 20]
        >>> list(sequence(10, 20, 15))
        [10]
    """
    if step is None:
        step = 1 if stop > start else -1
    if precision is None:
        precision = fc.guess_decimals(step)
    for i in range(int(round(stop - start, precision + 1) / step) + 1):
        item = start + i * step
        if precision:
            item = round(item, precision)
        yield item


# ======================================================================
def accumulate(
        items,
        func=lambda x, y: x + y):
    """
    Cumulatively apply the specified function to the elements of the list.

    Args:
        items (Iterable): The items to process.
        func (callable): func(x,y) -> z
            The function applied cumulatively to the first n items of the list.
            Defaults to cumulative sum.

    Returns:
        lst (list): The cumulative list.

    See Also:
        itertools.accumulate.
    Examples:
        >>> accumulate(list(range(5)))
        [0, 1, 3, 6, 10]
        >>> accumulate(list(range(5)), lambda x, y: (x + 1) * y)
        [0, 1, 4, 15, 64]
        >>> accumulate([1, 2, 3, 4, 5, 6, 7, 8], lambda x, y: x * y)
        [1, 2, 6, 24, 120, 720, 5040, 40320]
    """
    return [
        functools.reduce(func, list(items)[:i + 1])
        for i in range(len(items))]


# ======================================================================
def cyclic_padding_loops(
        arr,
        shape,
        offsets):
    """
    Generate a cyclical padding of an array to a given shape with offsets.

    Implemented using single element loops.

    Args:
        arr (np.ndarray): The input array.
        shape (int|Iterable[int]): The output shape.
            If int, a shape matching the input dimension is generated.
        offsets (int|float|Iterable[int|float]): The input offset.
            The input is shifted by the specified offset before padding.
            If int or float, the same offset is applied to all dimensions.
            If float, the offset is scaled to the difference between the
            input shape and the output shape.

    Returns:
        result (np.ndarray): The cyclic padded array of given shape.

    Examples:
        >>> arr = fc.extra.arange_nd((2, 3)) + 1
        >>> print(arr)
        [[1 2 3]
         [4 5 6]]
        >>> print(cyclic_padding_loops(arr, (4, 5), (1, 1)))
        [[5 6 4 5 6]
         [2 3 1 2 3]
         [5 6 4 5 6]
         [2 3 1 2 3]]
    """
    shape = fc.auto_repeat(shape, arr.ndim, check=True)
    offsets = fc.auto_repeat(offsets, arr.ndim, check=True)
    offsets = tuple(
        (int(round((new_dim - dim) * offset))
         if isinstance(offset, float) else offset) % dim
        for dim, new_dim, offset in zip(arr.shape, shape, offsets))
    assert (arr.ndim == len(shape) == len(offsets))
    result = np.zeros(shape, dtype=arr.dtype)
    for ij in itertools.product(*tuple(range(dim) for dim in result.shape)):
        slicing = tuple(
            (i + offset) % dim
            for i, offset, dim in zip(ij, offsets, arr.shape))
        result[ij] = arr[slicing]
    return result


# ======================================================================
def symmetric_padding_loops(
        arr,
        shape,
        offsets):
    """
    Generate a symmetrical padding of an array to a given shape with offsets.

    Implemented using single element loops.

    Args:
        arr (np.ndarray): The input array.
        shape (int|Iterable[int]): The output shape.
            If int, a shape matching the input dimension is generated.
        offsets (int|float|Iterable[int|float]): The input offset.
            The input is shifted by the specified offset before padding.
            If int or float, the same offset is applied to all dimensions.
            If float, the offset is scaled to the difference between the
            input shape and the output shape.

    Returns:
        result (np.ndarray): The symmetric padded array of given shape.

    Examples:
        >>> arr = fc.extra.arange_nd((2, 3)) + 1
        >>> print(arr)
        [[1 2 3]
         [4 5 6]]
        >>> print(symmetric_padding_loops(arr, (4, 5), (-1, -1)))
        [[6 6 5 4 4]
         [6 6 5 4 4]
         [3 3 2 1 1]
         [3 3 2 1 1]]
    """
    shape = fc.auto_repeat(shape, arr.ndim, check=True)
    offsets = fc.auto_repeat(offsets, arr.ndim, check=True)
    offsets = tuple(
        (int(round((new_dim - dim) * offset))
         if isinstance(offset, float) else offset) % dim
        for dim, new_dim, offset in zip(arr.shape, shape, offsets))
    assert (arr.ndim == len(shape) == len(offsets))
    result = np.zeros(shape, dtype=arr.dtype)
    for ij in itertools.product(*tuple(range(dim) for dim in result.shape)):
        slicing = tuple(
            (i + offset) % dim
            if not ((i + offset) // dim % 2) else
            (dim - 1 - i - offset) % dim
            for i, offset, dim in zip(ij, offsets, arr.shape))
        result[ij] = arr[slicing]
    return result


# ======================================================================
def cyclic_padding_tile(
        arr,
        shape,
        offsets):
    """
    Generate a cyclical padding of an array to a given shape with offsets.

    Implemented using single element loops.

    Args:
        arr (np.ndarray): The input array.
        shape (int|Iterable[int]): The output shape.
            If int, a shape matching the input dimension is generated.
        offsets (int|float|Iterable[int|float]): The input offset.
            The input is shifted by the specified offset before padding.
            If int or float, the same offset is applied to all dimensions.
            If float, the offset is scaled to the difference between the
            input shape and the output shape.

    Returns:
        result (np.ndarray): The cyclic padded array of given shape.

    Examples:
        >>> arr = fc.extra.arange_nd((2, 3)) + 1
        >>> print(arr)
        [[1 2 3]
         [4 5 6]]
        >>> print(cyclic_padding_tile(arr, (4, 5), (1, 1)))
        [[5 6 4 5 6]
         [2 3 1 2 3]
         [5 6 4 5 6]
         [2 3 1 2 3]]
    """
    shape = fc.auto_repeat(shape, arr.ndim, check=True)
    offsets = fc.auto_repeat(offsets, arr.ndim, check=True)
    offsets = tuple(
        (int(round((new_dim - dim) * offset))
         if isinstance(offset, float) else offset) % dim
        for dim, new_dim, offset in zip(arr.shape, shape, offsets))
    assert (arr.ndim == len(shape) == len(offsets))
    tiling = tuple(
        new_dim // dim + (1 if new_dim % dim else 0) + (1 if offset else 0)
        for offset, dim, new_dim in zip(offsets, arr.shape, shape))
    result = np.tile(arr, tiling)
    slicing = tuple(
        slice(offset, offset + new_dim)
        for offset, new_dim in zip(offsets, shape))
    return result[slicing]


# ======================================================================
def cyclic_padding_pad(
        arr,
        shape,
        offsets):
    """
    Generate a cyclical padding of an array to a given shape with offsets.

    Implemented using padding.

    Args:
        arr (np.ndarray): The input array.
        shape (int|Iterable[int]): The output shape.
            If int, a shape matching the input dimension is generated.
        offsets (int|float|Iterable[int|float]): The input offset.
            The input is shifted by the specified offset before padding.
            If int or float, the same offset is applied to all dimensions.
            If float, the offset is scaled to the difference between the
            input shape and the output shape.

    Returns:
        result (np.ndarray): The cyclic padded array of given shape.

    Examples:
        >>> arr = fc.extra.arange_nd((2, 3)) + 1
        >>> print(arr)
        [[1 2 3]
         [4 5 6]]
        >>> print(cyclic_padding_pad(arr, (4, 5), (1, 1)))
        [[5 6 4 5 6]
         [2 3 1 2 3]
         [5 6 4 5 6]
         [2 3 1 2 3]]
    """
    shape = fc.auto_repeat(shape, arr.ndim, check=True)
    offsets = fc.auto_repeat(offsets, arr.ndim, check=True)
    offsets = tuple(
        -(int(round((new_dim - dim) * offset))
          if isinstance(offset, float) else offset) % dim
        for dim, new_dim, offset in zip(arr.shape, shape, offsets))
    assert (arr.ndim == len(shape) == len(offsets))
    width = tuple(
        (0, new_dim - dim)
        for dim, new_dim, offset in zip(arr.shape, shape, offsets))
    result = np.pad(np.roll(arr, offsets, range(arr.ndim)), width, mode='wrap')
    return result


# ======================================================================
def cyclic_padding_slicing(
        arr,
        shape,
        offsets):
    """
    Generate a cyclical padding of an array to a given shape with offsets.

    Implemented using slicing.

    Args:
        arr (np.ndarray): The input array.
        shape (int|Iterable[int]): The output shape.
            If int, a shape matching the input dimension is generated.
        offsets (int|float|Iterable[int|float]): The input offset.
            The input is shifted by the specified offset before padding.
            If int or float, the same offset is applied to all dimensions.
            If float, the offset is scaled to the difference between the
            input shape and the output shape.

    Returns:
        result (np.ndarray): The cyclic padded array of given shape.

    Examples:
        >>> arr = fc.extra.arange_nd((2, 3)) + 1
        >>> print(arr)
        [[1 2 3]
         [4 5 6]]
        >>> print(cyclic_padding_slicing(arr, (4, 5), (1, 1)))
        [[5 6 4 5 6]
         [2 3 1 2 3]
         [5 6 4 5 6]
         [2 3 1 2 3]]
    """
    offsets = fc.auto_repeat(offsets, arr.ndim, check=True)
    offsets = tuple(
        (int(round((new_dim - dim) * offset))
         if isinstance(offset, float) else offset) % dim
        for dim, new_dim, offset in zip(arr.shape, shape, offsets))
    assert (arr.ndim == len(shape) == len(offsets))
    views = tuple(
        tuple(
            slice(max(0, dim * i - offset), dim * (i + 1) - offset)
            for i in range((new_dim + offset) // dim))
        + (slice(dim * ((new_dim + offset) // dim) - offset, new_dim),)
        for offset, dim, new_dim in zip(offsets, arr.shape, shape))
    views = tuple(
        tuple(slice_ for slice_ in view if slice_.start < slice_.stop)
        for view in views)
    result = np.zeros(shape, dtype=arr.dtype)
    for view in itertools.product(*views):
        slicing = tuple(
            slice(None)
            if slice_.stop - slice_.start == dim else (
                slice(offset, offset + (slice_.stop - slice_.start))
                if slice_.start == 0 else
                slice(0, (slice_.stop - slice_.start)))
            for slice_, offset, dim in zip(view, offsets, arr.shape))
        result[view] = arr[slicing]
    return result


# ======================================================================
def frame(
        arr,
        borders=0.05,
        background=0.0,
        use_longest=True):
    """
    Add a background frame to an array specifying the borders.

    Note that this is similar to `fc.extra.padding()` but it is significantly
    faster, although less flexible.

    Args:
        arr (np.ndarray): The input array.
        borders (int|float|Iterable[int|float]): The border size(s).
            If int, this is in units of pixels.
            If float, this is proportional to the initial array shape.
            If int or float, uses the same value for all dimensions.
            If Iterable, the size must match `arr` dimensions.
            If 'use_longest' is True, use the longest dimension for the
            calculations.
        background (int|float): The background value to be used for the frame.
        use_longest (bool): Use longest dimension to get the border size.

    Returns:
        result (np.ndarray): The result array with added borders.

    See Also:
        - flyingcircus.extra.reframe()
        - flyingcircus.extra.padding()
    """
    borders = base.auto_repeat(borders, arr.ndim)
    if any(borders) < 0:
        raise ValueError('relative border cannot be negative')
    if isinstance(borders[0], float):
        if use_longest:
            dim = max(arr.shape)
            borders = [round(border * dim) for border in borders]
        else:
            borders = [
                round(border * dim) for dim, border in zip(arr.shape, borders)]
    result = np.full(
        [dim + 2 * border for dim, border in zip(arr.shape, borders)],
        background, dtype=arr.dtype)
    inner = tuple(
        slice(border, border + dim, None)
        for dim, border in zip(arr.shape, borders))
    result[inner] = arr
    return result


# ======================================================================
def reframe(
        arr,
        new_shape,
        position=0.5,
        background=0.0):
    """
    Add a frame to an array by centering the input array into a new shape.

    Args:
        arr (np.ndarray): The input array.
        new_shape (int|Iterable[int]): The shape of the output array.
            If int, uses the same value for all dimensions.
            If Iterable, the size must match `arr` dimensions.
            Additionally, each value of `new_shape` must be greater than or
            equal to the corresponding dimensions of `arr`.
        position (int|float|Iterable[int|float]): Position within new shape.
            Determines the position of the array within the new shape.
            If int or float, it is considered the same in all dimensions,
            otherwise its length must match the number of dimensions of the
            array.
            If int or Iterable of int, the values are absolute and must be
            less than or equal to the difference between the shape of the array
            and the new shape.
            If float or Iterable of float, the values are relative and must be
            in the [0, 1] range.
        background (int|float): The background value to be used for the frame.

    Returns:
        result (np.ndarray): The result array with added borders.

    Raises:
        IndexError: input and output shape sizes must match.
        ValueError: output shape cannot be smaller than the input shape.

    See Also:
        - flyingcircus.extra.frame()
        - flyingcircus.extra.padding()

    Examples:
        >>> arr = np.ones((2, 3))
        >>> reframe(arr, (4, 5))
        array([[0., 0., 0., 0., 0.],
               [0., 1., 1., 1., 0.],
               [0., 1., 1., 1., 0.],
               [0., 0., 0., 0., 0.]])
        >>> reframe(arr, (4, 5), 0)
        array([[1., 1., 1., 0., 0.],
               [1., 1., 1., 0., 0.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.]])
        >>> reframe(arr, (4, 5), (2, 0))
        array([[0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.],
               [1., 1., 1., 0., 0.],
               [1., 1., 1., 0., 0.]])
        >>> reframe(arr, (4, 5), (0.0, 1.0))
        array([[0., 0., 1., 1., 1.],
               [0., 0., 1., 1., 1.],
               [0., 0., 0., 0., 0.],
               [0., 0., 0., 0., 0.]])
    """
    new_shape = fc.auto_repeat(new_shape, arr.ndim, check=True)
    position = fc.auto_repeat(position, arr.ndim, check=True)
    if any(old > new for old, new in zip(arr.shape, new_shape)):
        raise ValueError('new shape cannot be smaller than the old one.')
    position = tuple(
        int(round((new - old) * x_i)) if isinstance(x_i, float) else x_i
        for old, new, x_i in zip(arr.shape, new_shape, position))
    if any(old + x_i > new
           for old, new, x_i in zip(arr.shape, new_shape, position)):
        raise ValueError(
            'Incompatible `new_shape`, `array shape` and `position`.')
    result = np.full(new_shape, background)
    inner = tuple(
        slice(offset, offset + dim, None)
        for dim, offset in zip(arr.shape, position))
    result[inner] = arr
    return result


# ======================================================================
def transparent_compression(func):
    """WIP"""

    def _wrapped(fp):
        from importlib import import_module


        zip_module_names = "gzip", "bz2"
        fallback_module_name = "builtins"
        open_module_names = zip_module_names + (fallback_module_name,)
        for open_module_name in open_module_names:
            try:
                open_module = import_module(open_module_name)
                tmp_fp = open_module.open(fp, "rb")
                tmp_fp.read(1)
            except (OSError, IOError, AttributeError, ImportError) as e:
                if open_module_name is fallback_module_name:
                    raise e
            else:
                tmp_fp.seek(0)
                fp = tmp_fp
                break
        return func(fp=fp)

    return _wrapped


# ======================================================================
def ssim(
        arr1,
        arr2,
        arr_interval=None,
        aa=(1, 1, 1),
        kk=(0.010, 0.030, 0.015)):
    """
    Calculate the structure similarity index, SSIM.

    This is defined as: SSIM = (lum ** alpha) * (con ** beta) * (sti ** gamma)
     - lum is a measure of the luminosity, with exp. weight alpha
     - con is a measure of the contrast, with exp. weight beta
     - sti is a measure of the structural information, with exp. weight gamma

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.
        arr_interval (tuple[float]): Minimum and maximum allowed values.
            The values of both arr1 and arr2 should be within this interval.
        aa (tuple[float]): The exponentiation weight factors. Must be 3.
            Modulate the relative weight of the three SSIM components
            (luminosity, contrast and structural information).
            If they are all equal to 1, the computation can be simplified.
        kk (tuple[float]): The ratio regularization constant factors. Must
        be 3.
            Determine the regularization constants as a factors of the total
            interval size (squared) for the three SSIM components
            (luminosity, contrast and structural information).
            Must be numbers much smaller than 1.

    Returns:
        ssim (float): The structure similarity index.

    See Also:
        Wang, Zhou, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. “Image
        Quality Assessment: From Error Visibility to Structural Similarity.”
        IEEE Transactions on Image Processing 13, no. 4 (April 2004):
        600–612. doi:10.1109/TIP.2003.819861.
    """
    assert (arr1.shape == arr2.shape)
    if arr_interval is None:
        arr_interval = (
            min(np.min(arr1), np.min(arr2)), max(np.max(arr1), np.max(arr2)))
    range_size = np.ptp(arr_interval)
    cc = [(k * range_size) ** 2 for k in kk]
    mu1 = np.mean(arr1)
    mu2 = np.mean(arr2)
    sigma1 = np.std(arr1)
    sigma2 = np.std(arr2)
    sigma12 = np.sum((arr1 - mu1) * (arr2 - mu2)) / (arr1.size - 1)
    ff = [
        (2 * mu1 * mu2 + cc[0]) / (mu1 ** 2 + mu2 ** 2 + cc[0]),
        (2 * sigma1 * sigma2 + cc[1]) / (sigma1 ** 2 + sigma2 ** 2 + cc[1]),
        (sigma12 + cc[2]) / (sigma1 * sigma2 + cc[2])
    ]
    return np.prod(np.array([f ** a for (f, a) in zip(ff, aa)]), 0)


# ======================================================================
def ssim_map(
        arr1,
        arr2,
        filter_sizes=5,
        sigmas=1.5,
        arr_interval=None,
        aa=(1, 1, 1),
        kk=(0.010, 0.030, 0.015)):
    """
    Calculate the local structure similarity index map.

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.
        filter_sizes (tuple[int]|int): The size of the filter in px.
            If a single value is given, is is assumed to be equal in all dims.
        sigmas (tuple[float]|float): The sigma of the gaussian kernel in px.
            If a single value is given, it is assumed to be equal in all dims.
        arr_interval (tuple[float]): Minimum and maximum allowed values.
            The values of both arr1 and arr2 should be within this interval.
        aa (tuple[float]): The exponentiation weight factors.
            Must be of length 3.
            Modulate the relative weight of the three SSIM components
            (luminosity, contrast and structural information).
            If they are all equal to 1, the computation can be simplified.
        kk (tuple[float]): The ratio regularization constant factors.
            Must be of length 3.
            Determine the regularization constants as a factors of the total
            interval size (squared) for the three SSIM components
            (luminosity, contrast and structural information).
            Must be numbers much smaller than 1.

    Returns:
        ssim_arr (np.ndarray): The local structure similarity index map
        ssim (float): The global (mean) structure similarity index.

    See Also:
        Wang, Zhou, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli. “Image
        Quality Assessment: From Error Visibility to Structural Similarity.”
        IEEE Transactions on Image Processing 13, no. 4 (April 2004):
        600–612. doi:10.1109/TIP.2003.819861.
    """
    assert (arr1.shape == arr2.shape)
    if arr_interval is None:
        arr_interval = (
            min(np.min(arr1), np.min(arr2)), max(np.max(arr1), np.max(arr2)))
    range_size = np.ptp(arr_interval)
    ndim = arr1.ndim
    arr_filter = fc.extra.gaussian_nd(filter_sizes, sigmas, 0.5, ndim, True)
    convolve = sp.signal.fftconvolve
    mu1 = convolve(arr1, arr_filter, 'same')
    mu2 = convolve(arr2, arr_filter, 'same')
    mu1_mu1 = mu1 ** 2
    mu2_mu2 = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sg1_sg1 = convolve(arr1 ** 2, arr_filter, 'same') - mu1_mu1
    sg2_sg2 = convolve(arr2 ** 2, arr_filter, 'same') - mu2_mu2
    sg12 = convolve(arr1 * arr2, arr_filter, 'same') - mu1_mu2
    cc = [(k * range_size) ** 2 for k in kk]
    # determine whether to use the simplified expression
    if all(aa) == 1 and 2 * cc[2] == cc[1]:
        ssim_arr = ((2 * mu1_mu2 + cc[0]) * (2 * sg12 + cc[1])) / (
                (mu1_mu1 + mu2_mu2 + cc[0]) * (sg1_sg1 + sg2_sg2 + cc[1]))
    else:
        sg1 = np.sqrt(np.abs(sg1_sg1))
        sg2 = np.sqrt(np.abs(sg2_sg2))
        ff = [
            (2 * mu1_mu2 + cc[0]) / (mu1_mu1 + mu2_mu2 + cc[0]),
            (2 * sg1 * sg2 + cc[1]) / (sg1_sg1 + sg2_sg2 + cc[1]),
            (sg12 + cc[2]) / (sg1 * sg2 + cc[2])
        ]
        ssim_arr = np.prod(np.array([f ** a for (f, a) in zip(ff, aa)]), 0)
    ssim_val = np.mean(ssim_arr)
    return ssim_arr, ssim_val


# ======================================================================
if __name__ == '__main__':
    msg(__doc__.strip())
    doctest.testmod()
