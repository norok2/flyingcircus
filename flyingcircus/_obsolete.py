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

import flyingcircus as fc
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
        - flyingcircus.is_prime()
        - flyingcircus.primes_range()
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
            generated using `flyingcircus.gen_primes()`.
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
        - flyingcircus.is_prime()
        - flyingcircus.primes_range()

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
def steps_alt(
        n,
        a=2,
        b=3):
    """
    Generate steps for `step_sort()` (Shell sort).

    These are of the form a ** n * b ** m with n == m or n == m + 1

    Args:
        n (int): The length of the sequence.
        a (int): The first number.
        b (int): The second number.

    Yields:
        int: The next step.

    Examples:
        >>> list(steps_alt(100000))
        [1, 6, 18, 36, 108, 216, 648, 1296, 3888, 7776, 23328, 46656]
        >>> list(steps_alt(100000, 3, 2))
        [1, 6, 12, 36, 72, 216, 432, 1296, 2592, 7776, 15552, 46656, 93312]
    """
    yield 1
    x = a * b
    p = 0
    while x < n:
        yield x
        if p:
            x *= a
            p = 0
        else:
            x *= b
            p = 1


# ======================================================================
def steps_m_d(
        n,
        init=(1, 4, 10, 23, 57, 132, 301, 701),
        m=9,
        d=4):
    """
    Generate steps for `step_sort()` (Shell sort).

    These are of the form: x(k) = m * x(k - 1) // d
    with some initial predefined values.

    Must be that `m > d`.

    If `init=(1, 4, 10, 23, 57, 132, 301, 701), m=9, d=4` this has been
    empirically suggested as yielding optimum empirical performances in the
    average case [Ciura2001], but unknown worst-case.
    If `init=(1, 4,), m=9, d=4` this is very close to yielding near-optimal
    empirical performances in the average case [Tokuda1992], but unknown
    worst-case.

    Args:
        n (int): The length of the sequence.
        init (Iterable[int]): The initial sequence.
        m (int): The multiplier for subsequent values.
        d (int): The divider for subsequent values.

    Yields:
        int: The next step.

    Examples:
        >>> list(steps_m_d(50000))
        [1, 4, 10, 23, 57, 132, 301, 701, 1577, 3548, 7983, 17961, 40412]
        >>> list(steps_m_d(50000, (1, 4)))
        [1, 4, 9, 20, 45, 101, 227, 510, 1147, 2580, 5805, 13061, 29387]

    See Also:
        - Ciura, Marcin (2001). "Best Increments for the Average Case of
          Shellsort" (PDF). In Freiwalds, Rusins (ed.). Proceedings of the
          13th International Symposium on Fundamentals of Computation Theory.
          London: Springer-Verlag. pp. 106–117. ISBN 978-3-540-42487-1
        - Tokuda, Naoyuki (1992). "An Improved Shellsort". In van Leeuven,
          Jan (ed.). Proceedings of the IFIP 12th World Computer Congress on
          Algorithms, Software, Architecture. Amsterdam: North-Holland
          Publishing Co. pp. 449–457. ISBN 978-0-444-89747-3.
    """
    try:
        for x in init:
            yield x
    except ValueError:
        x = 1
        yield x
    x = x * m // d
    while x < n:
        yield x
        x = x * m // d


# ======================================================================
def steps_d_m(
        n,
        d=11,
        m=5,
        stopping=None):
    """
    Generate steps for `step_sort()` (Shell sort).

    These are of the form: x(k) = m * x(k - 1) // d
    with initial value equal `n`.

    Must be that `d > m`.

    If `d=11, m=4`, this is yielding near-optimal empirical performances
    in the average case [Gonnet1991], but unknown worst-case.

    Args:
        n (int): The length of the sequence.
        d (int): The divider for subsequent values.
        m (int): The multiplier for subsequent values.
        stopping (callable|None): The modifier for the stopping value.
            If None, defaults to `n`.
            If callable, must have the signature: stopping(int): int.
            The input is `n`.

    Yields:
        int: The next step.

    Examples:
        >>> list(steps_d_m(100000))
        [45454, 20660, 9390, 4268, 1940, 881, 400, 181, 82, 37, 16, 7, 3, 1]

    See Also:
        - Gonnet, Gaston H.; Baeza-Yates, Ricardo (1991). "Shellsort".
          Handbook of Algorithms and Data Structures: In Pascal and C (2nd
          ed.). Reading, Massachusetts: Addison-Wesley. pp. 161–163.
          ISBN 978-0-201-41607-7.
    """
    if callable(stopping):
        n = stopping(n)
        yield n
    while True:
        n = n * m // d
        if n > 1:
            yield n
        else:
            yield 1
            break


# ======================================================================
def steps_exp(n, x=1, k=2):
    """
    Generate steps for `step_sort()` (Shell sort).

    These follow an exponential law.

    Args:
        n (int): The length of the sequence.
        x (int): The initial value.
        k (int): The initial exponent.

    Yields:
        int: The next step.

    Examples:
        >>> list(steps_exp(100000))
        [1, 7, 20, 54, 148, 403, 1096, 2980, 8103, 22026, 59874]
    """
    while x < n:
        yield x
        x = int(math.exp(k))
        k += 1


# ======================================================================
def steps_pow_k(
        n,
        x=1,
        k=2,
        b=3,
        c=-1,
        d=2,
        stopping=lambda x: x // 3):
    """
    Generate steps for `step_sort()` (Shell sort).

    These are of the form: x(k) = (b ** k + c) // d
    with some initial value.

    With `x=1, k=2, b=2, c=-1, d=1, stopping=None` this has worst-case
    complexity O(n √n) [Hibbard1963].
    With `x=1, k=2, b=2, c=1, d=1, stopping=None` this has worst-case
    complexity O(n √n) [Papernov1965].
    With `x=1, k=2, b=3, c=-1, d=2, stopping=lambda x: x // 3` this has
    worst-case complexity O(n √n) [Knuth1973].

    Args:
        n (int): The length of the sequence.
        x (int): The initial value.
        k (int): The initial exponent.
        b (int): The base.
        c (int): The constant term.
        d (int): The deonominator.
        stopping (callable|None): The modifier for the stopping value.
            If None, defaults to `n`.
            If callable, must have the signature: stopping(int): int.
            The input is `n`.

    Yields:
        int: The next step.

    Examples:
        >>> list(steps_pow_k(500000))
        [1, 4, 13, 40, 121, 364, 1093, 3280, 9841, 29524, 88573]
        >>> list(steps_pow_k(20000, x=1, k=2, b=2, c=-1, d=1, stopping=None))
        [1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383]
        >>> list(steps_pow_k(20000, x=1, k=2, b=2, c=1, d=1, stopping=None))
        [1, 5, 9, 17, 33, 65, 129, 257, 513, 1025, 2049, 4097, 8193, 16385]

    See Also:
        - Knuth, Donald E. (1997). "Shell's method". The Art of Computer
          Programming. Volume 3: Sorting and Searching (2nd ed.). Reading,
          Massachusetts: Addison-Wesley. pp. 83–95. ISBN 978-0-201-89685-5.
    """
    if callable(stopping):
        n = stopping(n)
    while x < n:
        yield x
        x = (b ** k + c) // d
        k += 1


# ======================================================================
def steps_bi_factors(n, a=2, b=3):
    """
    Generate steps for `step_sort()` (Shell sort).

    These are of the form: x(k) = (a ** i) * (b ** j)
    with some initial value.

    With `a=2, b=3` this has worst-case complexity O(n log² n) [Pratt1971].
    However, for applications, other sequences are preferred as they perform
    generally faster, even if they have inferior worst-case complexity.

    Args:
        n (int): The length of the sequence.
        a (int): The first base.
        b (int): The second base.

    Yields:
        int: The next step.

    Examples:
        >>> sorted(steps_bi_factors(100))
         [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 27, 32, 36, 48, 54, 72, 81]

    See Also:
        - Pratt, Vaughan Ronald (1979). Shellsort and Sorting Networks
          (Outstanding Dissertations in the Computer Sciences). Garland.
          ISBN 978-0-8240-4406-0.
    """
    m = fc.ilog2(n)
    for k in range(m):
        for i in range(k + 1):
            j = k - i
            x = a ** i * b ** j
            if x < n:
                yield x


# ======================================================================
def steps_sum_pow_k(
        n,
        x=1,
        k=0,
        a=(1, 3),
        b=(4, 2),
        c=1):
    """
    Generate steps for `step_sort()` (Shell sort).

    These are of the form: x(k) = (b ** k + c) // d
    with some initial value.

    With `x=1, k=0, a=(1, 3), b=(4, 2), c=1` this has worst-case complexity
    O(n ^ (4/3)) [Sedgewick1985].

    Args:
        n (int): The length of the sequence.
        x (int): The initial value.
        k (int): The initial exponent.
        a (int): The amplitudes.
            Must match the number of elements.
        b (int): The bases.
        c (int): The constant term.

    Yields:
        int: The next step.

    Examples:
        >>> list(steps_sum_pow_k(2000000))
        [1, 8, 23, 77, 281, 1073, 4193, 16577, 65921, 262913, 1050113]

    See Also:
        - Sedgewick, Robert (1998). Algorithms in C. 1 (3rd ed.).
          Addison-Wesley. pp. 273–281. ISBN 978-0-201-31452-6.
    """
    offset = len(a) - 1
    while x < n:
        yield x
        x = c + sum(
            a_i * b_i ** (k + offset - i)
            for i, (a_i, b_i) in enumerate(zip(a, b)))
        k += 1


# ======================================================================
def steps_pow_frac(
        n,
        x=1,
        k=1,
        m=9,
        d=4,
        a=-4,
        b=5,
        c=1):
    """
    Generate steps for `step_sort()` (Shell sort).

    These are of the form: x(k) = (m ** (k + 1) // d ** k + a) // b + c
    with some initial value.

    With `x=1, k=1, m=9, d=4, a=-4, b=5, c=1` this has near optimal empirical
    average-case complexity [Tokuda1992].

    This is very similar, but not identical to:
     `steps_m_d(init=(1, 4), m=9, d=4)`.

    Args:
        n (int): The length of the sequence.
        x (int): The initial value.
        k (int): The initial exponent.
        a (int): The amplitudes.
            Must match the number of elements.
        b (int): The bases.
        c (int): The constant term.

    Yields:
        int: The next step.

    Examples:
        >>> list(steps_pow_frac(50000))
        [1, 4, 9, 20, 46, 103, 233, 525, 1182, 2660, 5985, 13467, 30301]

    See Also:
        - Tokuda, Naoyuki (1992). "An Improved Shellsort". In van Leeuven,
          Jan (ed.). Proceedings of the IFIP 12th World Computer Congress on
          Algorithms, Software, Architecture. Amsterdam: North-Holland
          Publishing Co. pp. 449–457. ISBN 978-0-444-89747-3.
    """
    while x < n:
        yield x
        x = (m ** (k + 1) // d ** k + a) // b + c
        k += 1


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
def merge_dicts(items):
    """
    Merge dictionaries into a new dict (new keys overwrite the old ones).

    This is obsoleted by `flyingcircus.join()`.

    Args:
        items (Iterable[dict]): Dictionaries to be merged together.

    Returns:
        merged (dict): The merged dict (new keys overwrite the old ones).

    Examples:
        >>> d1 = {1: 2, 3: 4, 5: 6}
        >>> d2 = {2: 1, 4: 3, 6: 5}
        >>> d3 = {1: 1, 3: 3, 6: 5}
        >>> dd = merge_dicts((d1, d2))
        >>> print(tuple(sorted(dd.items())))
        ((1, 2), (2, 1), (3, 4), (4, 3), (5, 6), (6, 5))
        >>> dd = merge_dicts((d1, d3))
        >>> print(tuple(sorted(dd.items())))
        ((1, 1), (3, 3), (5, 6), (6, 5))
    """
    merged = {}
    for item in items:
        merged.update(item)
    return merged


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
if __name__ == '__main__':
    msg(__doc__.strip())
    doctest.testmod()
