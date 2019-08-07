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
import itertools  # Functions creating iterators for efficient looping
import string  # Common string operations

# :: External Imports

# :: External Imports Submodules

# :: Local Imports
from flyingcircus import fmtm


# ======================================================================
def _bits(value):
    while value:
        # : equivalent to, but faster than:
        # yield bool(value & 1)
        yield (value & 1) == 1
        value >>= 1


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
        >>> print(BitMask(127))
        BitMask(abcdefg)
        >>> print(BitMask(8) + BitMask(1))
        BitMask(ad)
        >>> print(BitMask('a') + BitMask('d'))
        BitMask(ad)
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
        max_bitmask = (1 << self.len) - 1
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
    def len(self):
        """
        Compute the maximum lenght for the bit-mask (given the class keyS).

        Returns:
            result (int): The maximum length for the bit-mask.

        Examples:
            >>> BitMask().len
            52
            >>> BitMask(123).len
            52
        """
        return len(type(self).KEYS)

    # ----------------------------------------------------------
    @property
    def active_len(self):
        """
        Compute the maximum lenght for the bit-mask (given the class keyS).

        Returns:
            result (int): The maximum length for the bit-mask.

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
        Iterate over the flags of the bit-mask.

        Yields:
            value (bool): The value

        Examples:
            >>> print([flag for flag in BitMask('ac').values()])
            [True, False, True, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False]
            >>> print([flag for flag in BitMask(11).values()])
            [True, True, False, True, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False, False,\
 False, False, False, False, False, False, False, False, False, False]
        """
        for i in range(self.len):
            # : equivalent to, but faster than:
            # yield bool(self._bitmask & (1 << i))
            yield (self._bitmask & (1 << i)) > 0

    # ----------------------------------------------------------
    def active_values(self):
        """
        Iterate over the set flags of the bit-mask.

        Yields:
            value (bool): The value

        Examples:
            >>> flags = BitMask('ac')
            >>> print([flag for flag in flags.active_values()])
            [True, False, True]
            >>> print([flag for flag in BitMask(11).active_values()])
            [True, True, False, True]
        """
        return _bits(self._bitmask)

    __iter__ = values
    __len__ = len

    # ----------------------------------------------------------
    def items(self):
        for key, value in zip(type(self).KEYS, self.values()):
            yield key, value

    # ----------------------------------------------------------
    def active_items(self):
        for key, value in zip(
                type(self).KEYS, self.active_values(self.bitmask)):
            if value:
                yield key, value

    # ----------------------------------------------------------
    def keys(self):
        for key in type(self).KEYS:
            yield key

    # ----------------------------------------------------------
    def __getitem__(self, item):
        """
        Get the item at the specified position.

        Args:
            item (int): The index of the item to access.

        Returns:
            result (bool): The bit value at the given index in the bit-mask.

        Examples:
            >>> flags = BitMask('ac')
            >>> print([flags[i] for i in range(flags.active_len)])
            [True, False, True]
        """
        # : equivalent to, but faster than:
        # return bool(self._bitmask & (1 << item))
        return (self._bitmask & (1 << item)) > 0

    # ----------------------------------------------------------
    def __setitem__(self, item, value):
        """
        Set the item at the specified position.

        Args:
            item (int): The index of the item to modify.
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
        """
        if value:
            self._bitmask = self._bitmask | (1 << item)
        else:
            self._bitmask = self._bitmask & ~(1 << item)

    # ----------------------------------------------------------
    def __delitem__(self, item):
        """
        Unset the item at the specified position.

        Args:
            item (int): The index of the item to modify.

        Returns:
            result (bool): The bit value before deletion.

        Examples:
            >>> flags = BitMask(7)
            >>> print(flags)
            BitMask(abc)
            >>> del flags[0]
            >>> print(flags)
            BitMask(bc)
        """
        bit_value = self[item]
        self[item] = False
        return bit_value

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
               + str(self.as_dict())

    # ----------------------------------------------------------
    def __str__(self):
        keys = self.decode(self.bitmask)
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
        return self.decode(self.bitmask, ext_keys[1:], ext_keys[0], True, True)

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
        return self.decode(self.bitmask, keys, None, False, True)

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
        return self.decode(self.bitmask, keys, None, False, list)

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
            >>> print(flags.flip(0))
            False
            >>> print(flags[0])
            False
            >>> print(flags.flip(0))
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
            full=FULL_REPR,
            container=True):
        """
        Decode a bit-mask according to the specified keys.

        Args:
            keys (Iterable): The keys to use for decoding the bit-mask.
            empty (Any|None): The value to use for unset flags.
            full (bool): Represent all available flags.
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
        if full:
            result = (
                key if value else empty
                for key, value in
                itertools.zip_longest(keys, _bits(bitmask), fillvalue=False))
        else:
            result = (key for key, value in zip(keys, _bits(bitmask)) if value)
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
            ignore (bool): Ignore invalid items.
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
    def as_dict(self):
        return dict(self.items())

    # ----------------------------------------------------------
    def __getattr__(self, name):
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
        try:
            i = type(self).KEYS.index(name)
        except ValueError:
            i = None
        if i is None:
            raise AttributeError(fmtm('Unknown key `{name}`.'))
        else:
            del self[i]
