#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flyingcircus.BitField: Bit Field class.
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
class BitFieldMeta(type):
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
class BitField(object, metaclass=BitFieldMeta):
    """
    Generic bit-field class.

    The recommended usage is to subclass for specialized representations.

    Typically, while subclassing, `KEYS` and `EMPTY` should probably be
    replaced.
    Additionally, `FULL_REPR` and `IGNORE_INVALID` control the bit-field
    default behavior.
    In particular:
     - `KEYS` determines which flags to use for encoding/decoding the bit-field.
       **IMPORTANT!**: Items in `KEYS` must be unique.
     - `EMPTY` is used during encoding/decoding for marking empty/unset flags.
       **IMPORTANT!**: `EMPTY` must not be present in `KEYS`.
     - `FULL_REPR` determines whether the default decoding, including its
        string representation, should include empty/unset flags.
     - `IGNORE_INVALID` determines whether invalid input values should be
       ignored during encoding or rather a ValueError should be raised.

    Examples:
        >>> print(BitField(127))
        BitField(abcdefg)
        >>> repr(BitField(127))
        "BitField(0b1111111){'a': True, 'b': True, 'c': True, 'd': True,\
 'e': True, 'f': True, 'g': True, 'h': False, 'i': False, 'j': False,\
 'k': False, 'l': False, 'm': False, 'n': False, 'o': False, 'p': False,\
 'q': False, 'r': False, 's': False, 't': False, 'u': False, 'v': False,\
 'w': False, 'x': False, 'y': False, 'z': False, 'A': False, 'B': False,\
 'C': False, 'D': False, 'E': False, 'F': False, 'G': False, 'H': False,\
 'I': False, 'J': False, 'K': False, 'L': False, 'M': False, 'N': False,\
 'O': False, 'P': False, 'Q': False, 'R': False, 'S': False, 'T': False,\
 'U': False, 'V': False, 'W': False, 'X': False, 'Y': False, 'Z': False}"

        >>> int(BitField(12))
        12

        >>> print(BitField(8) + BitField(1))
        BitField(ad)
        >>> print(BitField('a') + BitField('d'))
        BitField(ad)

        >>> print(list(BitField(11)))
        ['a', 'b', 'd']

        >>> print(list(reversed(BitField(11))))
        ['d', 'b', 'a']

        >>> class UnixPermissions(BitField):
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
        >>> acl.BitField = 7
        >>> print(acl)
        UnixPermissions(rwx)
        >>> del acl.x
        >>> print(acl)
        UnixPermissions(rw-)

        >>> class RGB(BitField):
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
        >>> rgb.BitField = 7
        >>> print(rgb)
        RGB(['red', 'green', 'blue'])
        >>> del rgb.blue
        >>> print(rgb)
        RGB(['red', 'green', None])
    """
    KEYS = string.ascii_letters
    EMPTY = ' '
    FULL_REPR = False
    IGNORE_INVALID = True

    # ----------------------------------------------------------
    def __init__(
            self,
            BitField=0):
        """
        Instantiate a BitField object.

        Args:
            BitField (int|Sequence): The input bit-field.
                If int, the flags of the bit-field are according to the bit
                boolean values.
                If a sequence, uses `KEYS` to encode the input into the
                corresponding bit-field, using `from_keys()` method.
                The value of `EMPTY` is used to mark empty / unset bits.
                The value of `IGNORE_INVALID` determines whether invalid
                items in the sequence are ignored or will raise a ValueError.
        """
        cls = type(self)
        super(BitField, self).__setattr__('_BitField', None)
        if isinstance(BitField, int):
            self.BitField = BitField
        else:
            self.BitField = self.encode(
                BitField, cls.KEYS, cls.EMPTY, cls.IGNORE_INVALID)

    # ----------------------------------------------------------
    @property
    def BitField(self):
        return super(BitField, self).__getattribute__('_BitField')

    # ----------------------------------------------------------
    @BitField.setter
    def BitField(self, BitField_):
        max_BitField = (1 << self.size) - 1
        if 0 <= BitField_ <= max_BitField:
            super(BitField, self).__setattr__('_BitField', BitField_)
        else:
            raise ValueError(fmtm(
                '{self.__class__.__name__}() bit-field must be between'
                ' 0 and {max_BitField}, not `{BitField_}`'))

    # ----------------------------------------------------------
    @BitField.deleter
    def BitField(self):
        del self._BitField

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
        Compute the maximum lenght for the bit-field (given the class keys).

        Returns:
            result (int): The maximum length for the bit-field.

        Examples:
            >>> BitField().size
            52
            >>> BitField(123).size
            52
            >>> len(BitField(123))
            52
        """
        return len(type(self).KEYS)

    # ----------------------------------------------------------
    __len__ = KEYS.__len__

    # ----------------------------------------------------------
    @property
    def active_len(self):
        """
        Compute the actual lenght for the bit-field.

        Returns:
            result (int): The actual length of the bit-field.

        Examples:
            >>> BitField().active_len
            0
            >>> BitField(123).active_len
            7
        """
        return self._BitField.bit_length()

    # ----------------------------------------------------------
    def values(self):
        """
        Iterate forward over the bit-field values.

        The iteration is from the least to the most significant bit.

        Yields:
            value (bool): The next bit value.

        Examples:
            >>> print(list(BitField('ac').values()))
            [True, False, True, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False]
            >>> print(list(BitField(11).values()))
            [True, True, False, True, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False]
        """
        for i in range(self.size):
            yield (self._BitField & (1 << i)) > 0

    # ----------------------------------------------------------
    def values_r(self):
        """
        Iterate backward over the bit-field values.

        The iteration is from the most to the least significant bit.

        Yields:
            value (bool): The previous bit value.

        Examples:
            >>> print(list(BitField('ac').values_r()))
            [False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, True, False, True]
            >>> print(list(BitField(11).values_r()))
            [False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, True, False, True, True]
        """
        for i in range(self.size - 1, -1, -1):
            yield ((self._BitField >> i) & 1) > 0

    # ----------------------------------------------------------
    def active_values(self):
        """
        Iterate forward over the active bit-field values.

        Yields:
            value (bool): The next active value.

        Examples:
            >>> flags = BitField('ac')
            >>> print(list(flags.active_values()))
            [True, False, True]
            >>> print(list(BitField(11).active_values()))
            [True, True, False, True]
        """
        yield from bits(self._BitField)

# ----------------------------------------------------------
    def active_values_r(self):
        """
        Iterate backward over the active bit-field values.

        Yields:
            value (bool): The previous active value.

        Examples:
            >>> flags = BitField('ac')
            >>> print(list(flags.active_values()))
            [True, False, True]
            >>> print(list(BitField(11).active_values()))
            [True, True, False, True]
        """
        yield from bits_r(self._BitField)

    # ----------------------------------------------------------
    def keys(self):
        """
        Yields the keys of the bit-field.

        Yields:
            key (str): The next key.

        Examples:
            >>> print(''.join((BitField(7).keys())))
            abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
        """
        for key in type(self).KEYS:
            yield key

    # ----------------------------------------------------------
    def keys_r(self):
        """
        Yields the keys of the bit-field in reversed order.

        Yields:
            key (str): The previous key.

        Examples:
            >>> print(''.join((BitField(7).keys_r())))
            ZYXWVUTSRQPONMLKJIHGFEDCBAzyxwvutsrqponmlkjihgfedcba
        """
        for key in reversed(type(self).KEYS):
            yield key

    # ----------------------------------------------------------
    def active_keys(self):
        """
        Yields the active keys of the bit-field.

        Yields:
            key (str): The next key.

        Examples:
            >>> print(list(BitField(7).active_keys()))
            ['a', 'b', 'c']
            >>> print(list(BitField(5).active_keys()))
            ['a', 'c']
        """
        for key, value in zip(
                type(self).KEYS, bits(self._BitField)):
            if value:
                yield key

    # ----------------------------------------------------------
    def active_keys_r(self):
        """
        Yields the active keys of the bit-field in reversed order.

        Yields:
            key (str): The previous key.

        Examples:
            >>> print(list(BitField(7).active_keys_r()))
            ['c', 'b', 'a']
        """
        for key, value in zip(
                reversed(type(self).KEYS[:self.active_len]),
                bits_r(self._BitField)):
            if value:
                yield key

    # ----------------------------------------------------------
    def items(self):
        """
        Yields the key-value pairs of the bit-field.

        Yields:
            key (str): The next key.
            value (bool): The next value.

        Examples:
            >>> print(dict(BitField(7).items()))
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
        Yields the key-value pairs of the bit-field.

        Yields:
            key (str): The next key.
            value (bool): The next value.

        Examples:
            >>> print(dict(BitField(7).items_r()))
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
        Yields the active key-value pairs of the bit-field.

        Yields:
            key (str): The next key.
            value (bool): The next value.

        Examples:
            >>> print(dict(BitField(7).active_items()))
            {'a': True, 'b': True, 'c': True}
            >>> print(dict(BitField(5).active_items()))
            {'a': True, 'c': True}
        """
        for key, value in zip(
                type(self).KEYS, bits(self._BitField)):
            if value:
                yield key, value

    # ----------------------------------------------------------
    def active_items_r(self):
        """
        Yields the active key-value pairs of the bit-field.

        Yields:
            key (str): The next key.
            value (bool): The next value.

        Examples:
            >>> print(dict(BitField(7).active_items_r()))
            {'c': True, 'b': True, 'a': True}
            >>> print(dict(BitField(5).active_items_r()))
            {'c': True, 'a': True}
        """
        for key, value in zip(
                reversed(type(self).KEYS[:self.active_len]),
                bits_r(self._BitField)):
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
            result (bool): The bit value at the given index in the bit-field.

        Examples:
            >>> flags = BitField('ac')
            >>> print([flags[i] for i in range(flags.active_len)])
            [True, False, True]
            >>> print([flags[c] for c in 'abc'])
            [True, False, True]
        """
        if isinstance(key, str):
            key = self.KEYS.index(key)
        return (self._BitField & (1 << key)) > 0

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
            >>> flags = BitField('c')
            >>> flags[0] = True
            >>> print(flags)
            BitField(ac)
            >>> flags[0] = False
            >>> print(flags)
            BitField(c)
            >>> flags[1] = True
            >>> print(flags)
            BitField(bc)
            >>> flags[2] = False
            >>> print(flags)
            BitField(b)
            >>> flags['a'] = True
            >>> print(flags)
            BitField(ab)
            >>> flags['b'] = False
            >>> print(flags)
            BitField(a)
        """
        if isinstance(key, str):
            key = self.KEYS.index(key)
        if value:
            self._BitField = self._BitField | (1 << key)
        else:
            self._BitField = self._BitField & ~(1 << key)

    # ----------------------------------------------------------
    def __delitem__(self, key):
        """
        Unset the item at the specified position.

        Args:
            key (int|str): The index or key of the item.

        Returns:
            result (bool): The bit value before deletion.

        Examples:
            >>> flags = BitField(7)
            >>> print(flags)
            BitField(abc)
            >>> del flags[0]
            >>> print(flags)
            BitField(bc)
            >>> del flags['b']
            >>> print(flags)
            BitField(c)
        """
        bit_value = self[key]
        self[key] = False
        return bit_value

    # ----------------------------------------------------------
    def __contains__(self, key):
        """
        Check if the specified key is present in the bit-field.

        Args:
            key (str): The key of the item.

        Returns:
            result (bool): The result of the check.

        Examples:
            >>> 'a' in BitField()
            True
            >>> 'fc' in BitField()
            False
            >>> 'a' in BitField(0) and 'a' in BitField(1)
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
            >>> print(BitField().index('a'))
            0
            >>> print(BitField().index('A'))
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
            >>> BitField().count(True)
            0
            >>> BitField().count(False)
            52

            >>> flags = BitField(127)
            >>> print(flags.count())
            7
            >>> print(flags.count(True))
            7
            >>> print(flags.count(False))
            45
        """
        ones = bin(self._BitField).count('1')
        if value in (1, True):
            return ones
        elif value in (0, False):
            return self.size - ones
        else:
            raise ValueError('BitField only contains boolean values.')

    def clear(self):
        self._BitField = 0

    # ----------------------------------------------------------
    def __ior__(self, other):
        if type(self) == type(other):
            self._BitField |= other._BitField
            return self
        else:
            raise NotImplemented

    # ----------------------------------------------------------
    def __or__(self, other):
        """
        Perform bitwise-or operation between two bit-fields.

        Args:
            other (BitField): The other bit-field.

        Returns:
            result (BitField): The result of the operation.

        Examples:
            >>> a = BitField(1)
            >>> b = BitField(2)

            >>> print(a | b)
            BitField(ab)
            >>> print(a)
            BitField(a)
            >>> print(b)
            BitField(b)

            >>> print(a + b)
            BitField(ab)
            >>> print(a)
            BitField(a)
            >>> print(b)
            BitField(b)

        """
        result = type(self)(self._BitField)
        result |= other
        return result

    __add__ = __or__
    __iadd__ = __ior__

    # ----------------------------------------------------------
    def __iand__(self, other):
        if type(self) == type(other):
            self._BitField &= other._BitField
            return self
        else:
            raise NotImplemented

    # ----------------------------------------------------------
    def __and__(self, other):
        """
        Perform bitwise-and operation between two bit-fields.

        Args:
            other (BitField): The other bit-field.

        Returns:
            result (BitField): The result of the operation.

        Examples:
            >>> a = BitField(3)
            >>> b = BitField(6)

            >>> print(a & b)
            BitField(b)
            >>> print(a)
            BitField(ab)
            >>> print(b)
            BitField(bc)

            >>> print(a * b)
            BitField(b)
            >>> print(a)
            BitField(ab)
            >>> print(b)
            BitField(bc)

        """
        result = type(self)(self._BitField)
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
        Flip the specified bit from the bit-fields.

        Args:
            item (int): The position of the bit to flip.

        Returns:
            result (BitField): The result of the operation.

        Examples:
            >>> flags = BitField('bd')
            >>> flags ^= 0
            >>> print(flags)
            BitField(abd)
            >>> flags ^= 0
            >>> print(flags)
            BitField(bd)
            >>> print(flags ^ 0)
            BitField(abd)
        """
        result = type(self)(self._BitField)
        result ^= item
        return result

    # ----------------------------------------------------------
    def __eq__(self, other):
        return type(self) == type(other) and self._BitField == other._BitField

    # ----------------------------------------------------------
    def __ne__(self, other):
        return type(self) != type(other) or self._BitField != other._BitField

    # ----------------------------------------------------------
    def __repr__(self):
        return self.__class__.__name__ + '(' + bin(self._BitField) + ')' \
               + str(dict(self.items()))

    # ----------------------------------------------------------
    def __str__(self):
        keys = self.decode(self._BitField)
        text = str(keys) if keys else self.EMPTY
        return self.__class__.__name__ + '(' + text + ')'

    # ----------------------------------------------------------
    def __mod__(self, ext_keys):
        """
        Decode bit-field using the specified extended keys.

        Extended keys include the empty token as first element.

        Args:
            ext_keys (Sequence): The extended decoding keys.
                The first element is used for decoding the empty bits.
                All the other keys follow.

        Returns:
            result (Sequence): The decoded bit-field.

        Examples:
            >>> flags = BitField(3)
            >>> flags % '-rwx'
            'rw-'
        """
        return self.decode(
            self._BitField, ext_keys[1:], ext_keys[0], True, True)

    # ----------------------------------------------------------
    def __truediv__(self, keys):
        """
        Decode bit-field using the specified keys.

        Args:
            keys (Sequence): The decoding keys.
                Unset bits are ignored.

        Returns:
            result (Sequence): The decoded bit-field.

        Examples:
            >>> flags = BitField(3)
            >>> flags / '123'
            '12'
        """
        return self.decode(self._BitField, keys, None, False, True)

    # ----------------------------------------------------------
    def __floordiv__(self, keys):
        """
        Decode bit-field using the specified keys (and a list container).

        Args:
            keys (Sequence): The decoding keys.
                Unset bits are ignored.

        Returns:
            result (Sequence): The decoded bit-field.

        Examples:
            >>> flags = BitField(3)
            >>> flags // '123'
            ['1', '2']
        """
        return self.decode(self._BitField, keys, None, False, list)

    # ----------------------------------------------------------
    def flip(self, item):
        """
        Flip (toggle) the item at the specified position.

        Args:
            item (int): The index of the item to access.

        Returns:
            result (bool): The value of the item after flipping.

        Examples:
            >>> flags = BitField(7)
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
            BitField,
            keys=None,
            empty=None,
            full_repr=None,
            container=True):
        """
        Decode a bit-field according to the specified keys.

        Args:
            BitField (int): The input BitField.
            keys (Iterable): The keys to use for decoding the bit-field.
            empty (Any|None): The value to use for unset flags.
            full_repr (bool): Represent all available flags.
            container (bool|callable|None): Determine the container to use.
                If callable, must be a sequence constructor.
                If True, the container is inferred from the type of `KEYS`.
                Otherwise, the generator itself is returned.

        Returns:
            result (Sequence|generator): The keys associated to the bit-field.

        Examples:
            >>> BitField.decode(42)
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
                itertools.zip_longest(keys, bits(BitField), fillvalue=False))
        else:
            result = (key for key, value in zip(keys, bits(BitField)) if value)
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
        Encode a bit-field according to the specified keys.

        Args:
            items (Iterable): The items from the bit-field.
            keys (Iterable): The keys to use for decoding the bit-field.
            empty (Any|None): The value to use for unset flags.
            ignore_invalid (bool): Ignore invalid items.
                If False, a ValueError is raised if invalid items are present.

        Returns:
            result (int): The value associated to the items in the bit-field.

        Raises:
            ValueError: If ignore is False and invalid items are present in
                in the input.

        Examples:
            >>> BitField.encode('acio')
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
        return self._BitField

    # ----------------------------------------------------------
    def __getattr__(self, name):
        """
        Get the value of a given flag using attributes.

        Args:
            name (str): The attribute name.

        Returns:
            value (bool): The value of the flag.

        Examples:
            >>> print(BitField(1).a)
            True
            >>> print(BitField(0).a)
            False
            >>> print(BitField(1).something)
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
            >>> flags = BitField(6)
            >>> print(flags)
            BitField(bc)
            >>> flags.a = True
            >>> print(flags)
            BitField(abc)
            >>> flags.a = False
            >>> print(flags)
            BitField(bc)
            >>> flags.something = True
            Traceback (most recent call last):
                ...
            AttributeError: Unknown key `something`.
        """
        if name in {'BitField', '_BitField'}:
            super(BitField, self).__setattr__(name, value)
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
            >>> flags = BitField(6)
            >>> print(flags)
            BitField(bc)
            >>> del flags.a
            >>> print(flags)
            BitField(bc)
            >>> del flags.b
            >>> print(flags)
            BitField(c)
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
