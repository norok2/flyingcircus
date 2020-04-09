#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flyingcircus.BitMask: BitMask class.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import inspect  # Inspect live objects
import itertools  # Functions creating iterators for efficient looping
import string  # Common string operations


# ======================================================================
def fmtm(
        text,
        source=None):
    """
    Perform string formatting from a mapping source.

    Args:
        text (str): Text to format.
        source (Mapping|None): The mapping to use as source.
            If None, uses caller's `vars()`.

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
        TypeError: ...
        >>> fmtm('{b} + {b} = {}', 4)  # doctest:+ELLIPSIS
        Traceback (most recent call last):
            ...
        TypeError: ...
    """
    if source is None:
        frame = inspect.currentframe()
        source = frame.f_back.f_locals
    return text.format_map(source)


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
        [False, False, True, False, False, True, True]
    """
    # : this alternative may be faster for larger values
    # return map(int, bin(value)[:1:-1])
    while value:
        yield (value & 1) == 1
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
        [False, False, True, False, False, True, True]
    """
    # : this alternative may be faster for larger values
    # return map(int, bin(value)[2:])
    b = value.bit_length()
    for i in range(b - 1, -1, -1):
        yield ((value >> i) & 1) == 1


# ======================================================================
class BitMaskMeta(type):
    def __new__(mcs, cls_name, cls_bases, cls_dict):
        keys = cls_dict['KEYS']
        empty = cls_dict['EMPTY']
        if sorted(keys) != sorted(set(keys)):
            raise ValueError(
                fmtm('{cls_name}.KEYS must be unique'))
        if empty in keys:
            raise ValueError(
                fmtm('{cls_name}.EMPTY must not be in {cls_name}.KEYS'))
        return type.__new__(mcs, cls_name, cls_bases, cls_dict)


# ======================================================================
class BitMask(object, metaclass=BitMaskMeta):
    """
    Generic bit-mask class.

    The recommended usage is to subclass for specialized representations.

    Typically, while subclassing, `KEYS` and `EMPTY` should probably be
    replaced.
    Additionally, `FULL_REPR` and `IGNORE_INVALID` control the bit-mask
    default behavior.
    In particular:
     - `KEYS` determines which flags to use for encoding/decoding the bit-mask.
       **IMPORTANT!**: Items in `KEYS` must be unique.
     - `EMPTY` is used during encoding/decoding for marking empty/unset flags.
       **IMPORTANT!**: `EMPTY` must not be present in `KEYS`.
     - `FULL_REPR` determines whether the default decoding, including its
        string representation, should include empty/unset flags.
     - `IGNORE_INVALID` determines whether invalid input values should be
       ignored during encoding or rather a ValueError should be raised.

    Examples:
        >>> print(BitMask(127))
        BitMask(abcdefg)
        >>> repr(BitMask(127))
        "BitMask(0b1111111){'a': True, 'b': True, 'c': True, 'd': True,\
 'e': True, 'f': True, 'g': True, 'h': False, 'i': False, 'j': False,\
 'k': False, 'l': False, 'm': False, 'n': False, 'o': False, 'p': False,\
 'q': False, 'r': False, 's': False, 't': False, 'u': False, 'v': False,\
 'w': False, 'x': False, 'y': False, 'z': False, 'A': False, 'B': False,\
 'C': False, 'D': False, 'E': False, 'F': False, 'G': False, 'H': False,\
 'I': False, 'J': False, 'K': False, 'L': False, 'M': False, 'N': False,\
 'O': False, 'P': False, 'Q': False, 'R': False, 'S': False, 'T': False,\
 'U': False, 'V': False, 'W': False, 'X': False, 'Y': False, 'Z': False}"

        >>> int(BitMask(12))
        12

        >>> print(BitMask(8) + BitMask(1))
        BitMask(ad)
        >>> print(BitMask('a') + BitMask('d'))
        BitMask(ad)

        >>> print(list(BitMask(11)))
        ['a', 'b', 'd']

        >>> print(list(reversed(BitMask(11))))
        ['d', 'b', 'a']

        >>> class UnixPermissions(BitMask):
        ...     KEYS = 'rwx'
        ...     EMPTY = '-'
        ...     FULL_REPR = True
        >>> acl = UnixPermissions()
        >>> print(acl)
        UnixPermissions(---)
        >>> repr(acl)
        "UnixPermissions(0b0){'r': False, 'w': False, 'x': False}"
        >>> acl.r = True
        >>> print(acl)
        UnixPermissions(r--)
        >>> acl['x'] = True
        >>> print(acl)
        UnixPermissions(r-x)
        >>> acl.bitmask = 7
        >>> print(acl)
        UnixPermissions(rwx)
        >>> del acl.x
        >>> print(acl)
        UnixPermissions(rw-)

        >>> class RGB(BitMask):
        ...     KEYS = ['red', 'green', 'blue']
        ...     EMPTY = None
        ...     FULL_REPR = True
        >>> rgb = RGB()
        >>> print(rgb)
        RGB([None, None, None])
        >>> repr(rgb)
        "RGB(0b0){'red': False, 'green': False, 'blue': False}"
        >>> rgb.red = True
        >>> print(rgb)
        RGB(['red', None, None])
        >>> rgb['blue'] = True
        >>> print(rgb)
        RGB(['red', None, 'blue'])
        >>> print(list(rgb))
        ['red', 'blue']
        >>> rgb.bitmask = 7
        >>> print(rgb)
        RGB(['red', 'green', 'blue'])
        >>> del rgb.blue
        >>> print(acl)
        UnixPermissions(rw-)

    """
    KEYS = string.ascii_letters
    EMPTY = ' '
    FULL_REPR = False
    IGNORE_INVALID = True

    # ----------------------------------------------------------
    def __init__(
            self,
            bitmask=0):
        """
        Instantiate a BitMask object.

        Args:
            bitmask (int|Sequence): The input bit-mask.
                If int, the flags of the bit-mask are according to the bit
                boolean values.
                If a sequence, uses `KEYS` to encode the input into the
                corresponding bit-mask, using `from_keys()` method.
                The value of `EMPTY` is used to mark empty / unset bits.
                The value of `IGNORE_INVALID` determines whether invalid
                items in the sequence are ignored or will raise a ValueError.
        """
        cls = type(self)
        super(BitMask, self).__setattr__('_bitmask', None)
        if isinstance(bitmask, int):
            self.bitmask = bitmask
        else:
            self.bitmask = self.encode(
                bitmask, cls.KEYS, cls.EMPTY, cls.IGNORE_INVALID)

    # ----------------------------------------------------------
    @property
    def bitmask(self):
        return super(BitMask, self).__getattribute__('_bitmask')

    # ----------------------------------------------------------
    @bitmask.setter
    def bitmask(self, bitmask_):
        max_bitmask = (1 << self.size) - 1
        if 0 <= bitmask_ <= max_bitmask:
            super(BitMask, self).__setattr__('_bitmask', bitmask_)
        else:
            raise ValueError(fmtm(
                '{self.__class__.__name__}() bit-mask must be between'
                ' 0 and {max_bitmask}, not `{bitmask_}`'))

    # ----------------------------------------------------------
    @bitmask.deleter
    def bitmask(self):
        del self._bitmask

    # ----------------------------------------------------------
    @classmethod
    def is_valid(cls):
        valid_keys = sorted(cls.KEYS) == sorted(set(cls.KEYS))
        valid_empty = cls.EMPTY not in cls.KEYS
        return valid_keys and valid_empty

    # ----------------------------------------------------------
    @property
    def size(self):
        """
        Compute the maximum lenght for the bit-mask (given the class keys).

        Returns:
            result (int): The maximum length for the bit-mask.

        Examples:
            >>> BitMask().size
            52
            >>> BitMask(123).size
            52
            >>> len(BitMask(123))
            52
        """
        return len(type(self).KEYS)

    # ----------------------------------------------------------
    __len__ = KEYS.__len__

    # ----------------------------------------------------------
    @property
    def active_len(self):
        """
        Compute the actual lenght for the bit-mask.

        Returns:
            result (int): The actual length of the bit-mask.

        Examples:
            >>> BitMask().active_len
            0
            >>> BitMask(123).active_len
            7
        """
        return self._bitmask.bit_length()

    # ----------------------------------------------------------
    def values(self):
        """
        Iterate forward over the bit-mask values.

        The iteration is from the least to the most significant bit.

        Yields:
            value (bool): The next bit value.

        Examples:
            >>> print(list(BitMask('ac').values()))
            [True, False, True, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False]
            >>> print(list(BitMask(11).values()))
            [True, True, False, True, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False]
        """
        for i in range(self.size):
            yield (self._bitmask & (1 << i)) > 0

    # ----------------------------------------------------------
    def values_r(self):
        """
        Iterate backward over the bit-mask values.

        The iteration is from the most to the least significant bit.

        Yields:
            value (bool): The previous bit value.

        Examples:
            >>> print(list(BitMask('ac').values_r()))
            [False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, True, False, True]
            >>> print(list(BitMask(11).values_r()))
            [False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, True, False, True, True]
        """
        for i in range(self.size - 1, -1, -1):
            yield ((self._bitmask >> i) & 1) > 0

    # ----------------------------------------------------------
    def active_values(self):
        """
        Iterate forward over the active bit-mask values.

        Yields:
            value (bool): The next active value.

        Examples:
            >>> flags = BitMask('ac')
            >>> print(list(flags.active_values()))
            [True, False, True]
            >>> print(list(BitMask(11).active_values()))
            [True, True, False, True]
        """
        yield from bits(self._bitmask)

# ----------------------------------------------------------
    def active_values_r(self):
        """
        Iterate backward over the active bit-mask values.

        Yields:
            value (bool): The previous active value.

        Examples:
            >>> flags = BitMask('ac')
            >>> print(list(flags.active_values()))
            [True, False, True]
            >>> print(list(BitMask(11).active_values()))
            [True, True, False, True]
        """
        yield from bits_r(self._bitmask)

    # ----------------------------------------------------------
    def keys(self):
        """
        Yields the keys of the bit-mask.

        Yields:
            key (str): The next key.

        Examples:
            >>> print(''.join((BitMask(7).keys())))
            abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
        """
        for key in type(self).KEYS:
            yield key

    # ----------------------------------------------------------
    def keys_r(self):
        """
        Yields the keys of the bit-mask in reversed order.

        Yields:
            key (str): The previous key.

        Examples:
            >>> print(''.join((BitMask(7).keys_r())))
            ZYXWVUTSRQPONMLKJIHGFEDCBAzyxwvutsrqponmlkjihgfedcba
        """
        for key in reversed(type(self).KEYS):
            yield key

    # ----------------------------------------------------------
    def active_keys(self):
        """
        Yields the active keys of the bit-mask.

        Yields:
            key (str): The next key.

        Examples:
            >>> print(list(BitMask(7).active_keys()))
            ['a', 'b', 'c']
            >>> print(list(BitMask(5).active_keys()))
            ['a', 'c']
        """
        for key, value in zip(
                type(self).KEYS, bits(self._bitmask)):
            if value:
                yield key

    # ----------------------------------------------------------
    def active_keys_r(self):
        """
        Yields the active keys of the bit-mask in reversed order.

        Yields:
            key (str): The previous key.

        Examples:
            >>> print(list(BitMask(7).active_keys_r()))
            ['c', 'b', 'a']
        """
        for key, value in zip(
                reversed(type(self).KEYS[:self.active_len]),
                bits_r(self._bitmask)):
            if value:
                yield key

    # ----------------------------------------------------------
    def items(self):
        """
        Yields the key-value pairs of the bit-mask.

        Yields:
            key (str): The next key.
            value (bool): The next value.

        Examples:
            >>> print(dict(BitMask(7).items()))
            {'a': True, 'b': True, 'c': True, 'd': False, 'e': False,\
 'f': False, 'g': False, 'h': False, 'i': False, 'j': False, 'k': False,\
 'l': False, 'm': False, 'n': False, 'o': False, 'p': False, 'q': False,\
 'r': False, 's': False, 't': False, 'u': False, 'v': False, 'w': False,\
 'x': False, 'y': False, 'z': False, 'A': False, 'B': False, 'C': False,\
 'D': False, 'E': False, 'F': False, 'G': False, 'H': False, 'I': False,\
 'J': False, 'K': False, 'L': False, 'M': False, 'N': False, 'O': False,\
 'P': False, 'Q': False, 'R': False, 'S': False, 'T': False, 'U': False,\
 'V': False, 'W': False, 'X': False, 'Y': False, 'Z': False}
        """
        for key, value in zip(type(self).KEYS, self.values()):
            yield key, value

    # ----------------------------------------------------------
    def items_r(self):
        """
        Yields the key-value pairs of the bit-mask.

        Yields:
            key (str): The next key.
            value (bool): The next value.

        Examples:
            >>> print(dict(BitMask(7).items_r()))
            {'Z': False, 'Y': False, 'X': False, 'W': False, 'V': False,\
 'U': False, 'T': False, 'S': False, 'R': False, 'Q': False, 'P': False,\
 'O': False, 'N': False, 'M': False, 'L': False, 'K': False, 'J': False,\
 'I': False, 'H': False, 'G': False, 'F': False, 'E': False, 'D': False,\
 'C': False, 'B': False, 'A': False, 'z': False, 'y': False, 'x': False,\
 'w': False, 'v': False, 'u': False, 't': False, 's': False, 'r': False,\
 'q': False, 'p': False, 'o': False, 'n': False, 'm': False, 'l': False,\
 'k': False, 'j': False, 'i': False, 'h': False, 'g': False, 'f': False,\
 'e': False, 'd': False, 'c': True, 'b': True, 'a': True}
        """
        for key, value in zip(
                reversed(type(self).KEYS), self.values_r()):
            yield key, value

    # ----------------------------------------------------------
    def active_items(self):
        """
        Yields the active key-value pairs of the bit-mask.

        Yields:
            key (str): The next key.
            value (bool): The next value.

        Examples:
            >>> print(dict(BitMask(7).active_items()))
            {'a': True, 'b': True, 'c': True}
            >>> print(dict(BitMask(5).active_items()))
            {'a': True, 'c': True}
        """
        for key, value in zip(
                type(self).KEYS, bits(self._bitmask)):
            if value:
                yield key, value

    # ----------------------------------------------------------
    def active_items_r(self):
        """
        Yields the active key-value pairs of the bit-mask.

        Yields:
            key (str): The next key.
            value (bool): The next value.

        Examples:
            >>> print(dict(BitMask(7).active_items_r()))
            {'c': True, 'b': True, 'a': True}
            >>> print(dict(BitMask(5).active_items_r()))
            {'c': True, 'a': True}
        """
        for key, value in zip(
                reversed(type(self).KEYS[:self.active_len]),
                bits_r(self._bitmask)):
            if value:
                yield key, value

    # ----------------------------------------------------------
    __iter__ = active_keys
    __reversed__ = active_keys_r

    # ----------------------------------------------------------
    def __getitem__(self, key):
        """
        Get the value at the specified key or position.

        Args:
            key (int|str): The index or key of the item.

        Returns:
            result (bool): The bit value at the given index in the bit-mask.

        Examples:
            >>> flags = BitMask('ac')
            >>> print([flags[i] for i in range(flags.active_len)])
            [True, False, True]
            >>> print([flags[c] for c in 'abc'])
            [True, False, True]
        """
        if isinstance(key, str):
            key = self.KEYS.index(key)
        return (self._bitmask & (1 << key)) > 0

    # ----------------------------------------------------------
    def __setitem__(self, key, value):
        """
        Set the item at the specified position.

        Args:
            key (int|str): The index or key of the item.
            value (bool): The bit value to set.

        Returns:
            None.

        Examples:
            >>> flags = BitMask('c')
            >>> flags[0] = True
            >>> print(flags)
            BitMask(ac)
            >>> flags[0] = False
            >>> print(flags)
            BitMask(c)
            >>> flags[1] = True
            >>> print(flags)
            BitMask(bc)
            >>> flags[2] = False
            >>> print(flags)
            BitMask(b)
            >>> flags['a'] = True
            >>> print(flags)
            BitMask(ab)
            >>> flags['b'] = False
            >>> print(flags)
            BitMask(a)
        """
        if isinstance(key, str):
            key = self.KEYS.index(key)
        if value:
            self._bitmask = self._bitmask | (1 << key)
        else:
            self._bitmask = self._bitmask & ~(1 << key)

    # ----------------------------------------------------------
    def __delitem__(self, key):
        """
        Unset the item at the specified position.

        Args:
            key (int|str): The index or key of the item.

        Returns:
            result (bool): The bit value before deletion.

        Examples:
            >>> flags = BitMask(7)
            >>> print(flags)
            BitMask(abc)
            >>> del flags[0]
            >>> print(flags)
            BitMask(bc)
            >>> del flags['b']
            >>> print(flags)
            BitMask(c)
        """
        bit_value = self[key]
        self[key] = False
        return bit_value

    # ----------------------------------------------------------
    def __contains__(self, key):
        """
        Check if the specified key is present in the bit-mask.

        Args:
            key (str): The key of the item.

        Returns:
            result (bool): The result of the check.

        Examples:
            >>> 'a' in BitMask()
            True
            >>> 'fc' in BitMask()
            False
            >>> 'a' in BitMask(0) and 'a' in BitMask(1)
            True
        """
        return key in self.KEYS

    # ----------------------------------------------------------
    def index(self, key, start=None, stop=None):
        """
        Return the index of a specific key.

        Args:
            key (str): The key of the item.
            start (int|None): The minimum index to look into.
            stop (int|None): The maximum index to look into.

        Returns:
            result (int): The index corresponding to a specific key.

        Examples:
            >>> print(BitMask().index('a'))
            0
            >>> print(BitMask().index('A'))
            26
        """
        if start is None and stop is None:
            return self.KEYS.index(key)
        if start is not None and stop is None:
            return self.KEYS.index(key, start=start)
        elif start is None and stop is not None:
            return self.KEYS.index(key, stop=stop)
        else:  # start is not None and stop is not None
            return self.KEYS.index(key, start, stop)

    # ----------------------------------------------------------
    def count(self, value=True):
        """
        Count the number of values (True or False).

        When counting the values of True, this is also known as the
        Hamming weight (or the Hamming distance to 0).

        Args:
            value (bool): The value to count.

        Returns:
            result (int): The number of matches found.

        Examples:
            >>> BitMask().count(True)
            0
            >>> BitMask().count(False)
            52

            >>> flags = BitMask(127)
            >>> print(flags.count())
            7
            >>> print(flags.count(True))
            7
            >>> print(flags.count(False))
            45
        """
        ones = bin(self._bitmask).count('1')
        if value in (1, True):
            return ones
        elif value in (0, False):
            return self.size - ones
        else:
            raise ValueError('BitMask only contains boolean values.')

    def clear(self):
        self._bitmask = 0

    # ----------------------------------------------------------
    def __ior__(self, other):
        if type(self) == type(other):
            self._bitmask |= other._bitmask
            return self
        else:
            raise NotImplemented

    # ----------------------------------------------------------
    def __or__(self, other):
        """
        Perform bitwise-or operation between two bit-masks.

        Args:
            other (BitMask): The other bit-mask.

        Returns:
            result (BitMask): The result of the operation.

        Examples:
            >>> a = BitMask(1)
            >>> b = BitMask(2)

            >>> print(a | b)
            BitMask(ab)
            >>> print(a)
            BitMask(a)
            >>> print(b)
            BitMask(b)

            >>> print(a + b)
            BitMask(ab)
            >>> print(a)
            BitMask(a)
            >>> print(b)
            BitMask(b)

        """
        result = type(self)(self._bitmask)
        result |= other
        return result

    __add__ = __or__
    __iadd__ = __ior__

    # ----------------------------------------------------------
    def __iand__(self, other):
        if type(self) == type(other):
            self._bitmask &= other._bitmask
            return self
        else:
            raise NotImplemented

    # ----------------------------------------------------------
    def __and__(self, other):
        """
        Perform bitwise-and operation between two bit-masks.

        Args:
            other (BitMask): The other bit-mask.

        Returns:
            result (BitMask): The result of the operation.

        Examples:
            >>> a = BitMask(3)
            >>> b = BitMask(6)

            >>> print(a & b)
            BitMask(b)
            >>> print(a)
            BitMask(ab)
            >>> print(b)
            BitMask(bc)

            >>> print(a * b)
            BitMask(b)
            >>> print(a)
            BitMask(ab)
            >>> print(b)
            BitMask(bc)

        """
        result = type(self)(self._bitmask)
        result &= other
        return result

    __mul__ = __and__
    __imul__ = __iand__

    # ----------------------------------------------------------
    def __ixor__(self, other):
        if isinstance(other, int):
            self.flip(other)
            return self
        else:
            raise NotImplemented

    # ----------------------------------------------------------
    def __xor__(self, item):
        """
        Flip the specified bit from the bit-masks.

        Args:
            item (int): The position of the bit to flip.

        Returns:
            result (BitMask): The result of the operation.

        Examples:
            >>> flags = BitMask('bd')
            >>> flags ^= 0
            >>> print(flags)
            BitMask(abd)
            >>> flags ^= 0
            >>> print(flags)
            BitMask(bd)
            >>> print(flags ^ 0)
            BitMask(abd)
        """
        result = type(self)(self._bitmask)
        result ^= item
        return result

    # ----------------------------------------------------------
    def __eq__(self, other):
        return type(self) == type(other) and self._bitmask == other._bitmask

    # ----------------------------------------------------------
    def __ne__(self, other):
        return type(self) != type(other) or self._bitmask != other._bitmask

    # ----------------------------------------------------------
    def __repr__(self):
        return self.__class__.__name__ + '(' + bin(self._bitmask) + ')' \
               + str(dict(self.items()))

    # ----------------------------------------------------------
    def __str__(self):
        keys = self.decode(self._bitmask)
        text = str(keys) if keys else self.EMPTY
        return self.__class__.__name__ + '(' + text + ')'

    # ----------------------------------------------------------
    def __mod__(self, ext_keys):
        """
        Decode bit-mask using the specified extended keys.

        Extended keys include the empty token as first element.

        Args:
            ext_keys (Sequence): The extended decoding keys.
                The first element is used for decoding the empty bits.
                All the other keys follow.

        Returns:
            result (Sequence): The decoded bit-mask.

        Examples:
            >>> flags = BitMask(3)
            >>> flags % '-rwx'
            'rw-'
        """
        return self.decode(
            self._bitmask, ext_keys[1:], ext_keys[0], True, True)

    # ----------------------------------------------------------
    def __truediv__(self, keys):
        """
        Decode bit-mask using the specified keys.

        Args:
            keys (Sequence): The decoding keys.
                Unset bits are ignored.

        Returns:
            result (Sequence): The decoded bit-mask.

        Examples:
            >>> flags = BitMask(3)
            >>> flags / '123'
            '12'
        """
        return self.decode(self._bitmask, keys, None, False, True)

    # ----------------------------------------------------------
    def __floordiv__(self, keys):
        """
        Decode bit-mask using the specified keys (and a list container).

        Args:
            keys (Sequence): The decoding keys.
                Unset bits are ignored.

        Returns:
            result (Sequence): The decoded bit-mask.

        Examples:
            >>> flags = BitMask(3)
            >>> flags // '123'
            ['1', '2']
        """
        return self.decode(self._bitmask, keys, None, False, list)

    # ----------------------------------------------------------
    def flip(self, item):
        """
        Flip (toggle) the item at the specified position.

        Args:
            item (int): The index of the item to access.

        Returns:
            result (bool): The value of the item after flipping.

        Examples:
            >>> flags = BitMask(7)
            >>> print(flags[0])
            True
            >>> print(flags.flip_slice(0))
            False
            >>> print(flags[0])
            False
            >>> print(flags.flip_slice(0))
            True
            >>> print(flags[0])
            True
        """
        self[item] = not self[item]
        return self[item]

    toggle = flip

    # ----------------------------------------------------------
    @classmethod
    def decode(
            cls,
            bitmask,
            keys=None,
            empty=None,
            full_repr=None,
            container=True):
        """
        Decode a bit-mask according to the specified keys.

        Args:
            bitmask (int): The input bitmask.
            keys (Iterable): The keys to use for decoding the bit-mask.
            empty (Any|None): The value to use for unset flags.
            full_repr (bool): Represent all available flags.
            container (bool|callable|None): Determine the container to use.
                If callable, must be a sequence constructor.
                If True, the container is inferred from the type of `KEYS`.
                Otherwise, the generator itself is returned.

        Returns:
            result (Sequence|generator): The keys associated to the bit-mask.

        Examples:
            >>> BitMask.decode(42)
            'bdf'
        """
        if keys is None:
            keys = cls.KEYS
        if empty is None:
            empty = cls.EMPTY
        if full_repr is None:
            full_repr = cls.FULL_REPR
        if full_repr:
            result = (
                key if value else empty
                for key, value in
                itertools.zip_longest(keys, bits(bitmask), fillvalue=False))
        else:
            result = (key for key, value in zip(keys, bits(bitmask)) if value)
        if callable(container):
            return container(result)
        elif container:
            if isinstance(keys, str):
                return ''.join(result)
            else:
                return type(keys)(result)
        else:
            return result

    # ----------------------------------------------------------
    @classmethod
    def encode(
            cls,
            items,
            keys=None,
            empty=None,
            ignore_invalid=None):
        """
        Encode a bit-mask according to the specified keys.

        Args:
            items (Iterable): The items from the bit-mask.
            keys (Iterable): The keys to use for decoding the bit-mask.
            empty (Any|None): The value to use for unset flags.
            ignore_invalid (bool): Ignore invalid items.
                If False, a ValueError is raised if invalid items are present.

        Returns:
            result (int): The value associated to the items in the bit-mask.

        Raises:
            ValueError: If ignore is False and invalid items are present in
                in the input.

        Examples:
            >>> BitMask.encode('acio')
            16645
        """
        if keys is None:
            keys = cls.KEYS
        if empty is None:
            empty = cls.EMPTY
        if ignore_invalid is None:
            ignore_invalid = cls.IGNORE_INVALID
        valid_keys = set(keys)
        valid_keys.add(empty)
        value = 0
        for i, item in enumerate(items):
            if item in valid_keys:
                if item != empty:
                    value |= 1 << keys.index(item)
            elif not ignore_invalid:
                raise ValueError(
                    fmtm('Invalid input `{item}` at index: {i}.'))
        return value

    # ----------------------------------------------------------
    def __int__(self):
        return self._bitmask

    # ----------------------------------------------------------
    def __getattr__(self, name):
        """
        Get the value of a given flag using attributes.

        Args:
            name (str): The attribute name.

        Returns:
            value (bool): The value of the flag.

        Examples:
            >>> print(BitMask(1).a)
            True
            >>> print(BitMask(0).a)
            False
            >>> print(BitMask(1).something)
            Traceback (most recent call last):
                ...
            AttributeError: Unknown key `something`.
        """
        if name in {'KEYS', 'EMPTY'}:
            return getattr(type(self), name)
        else:
            try:
                i = type(self).KEYS.index(name)
            except ValueError:
                i = None
            if i is None:
                raise AttributeError(fmtm('Unknown key `{name}`.'))
            else:
                return self[i]

    # ----------------------------------------------------------
    def __setattr__(self, name, value):
        """
        Set the value of a given flag using attributes.

        Args:
            name (str): The attribute name.

        Returns:
            None.

        Examples:
            >>> flags = BitMask(6)
            >>> print(flags)
            BitMask(bc)
            >>> flags.a = True
            >>> print(flags)
            BitMask(abc)
            >>> flags.a = False
            >>> print(flags)
            BitMask(bc)
            >>> flags.something = True
            Traceback (most recent call last):
                ...
            AttributeError: Unknown key `something`.
        """
        if name in {'bitmask', '_bitmask'}:
            super(BitMask, self).__setattr__(name, value)
        else:
            try:
                i = type(self).KEYS.index(name)
            except ValueError:
                i = None
            if i is None:
                raise AttributeError(fmtm('Unknown key `{name}`.'))
            else:
                self[i] = value

    # ----------------------------------------------------------
    def __delattr__(self, name):
        """
        Delete the value of a given flag using attributes.

        Args:
            name (str): The attribute name.

        Returns:
            None.

        Examples:
            >>> flags = BitMask(6)
            >>> print(flags)
            BitMask(bc)
            >>> del flags.a
            >>> print(flags)
            BitMask(bc)
            >>> del flags.b
            >>> print(flags)
            BitMask(c)
            >>> flags.something = True
            Traceback (most recent call last):
                ...
            AttributeError: Unknown key `something`.
        """
        try:
            i = type(self).KEYS.index(name)
        except ValueError:
            i = None
        if i is None:
            raise AttributeError(fmtm('Unknown key `{name}`.'))
        else:
            del self[i]
