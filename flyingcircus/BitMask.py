import string
import itertools
from flyingcircus import fmtm


# ======================================================================
class BitMask(object):
    """
    Generic bit-mask class.

    The recommended usage is to subclass for specialized representations.

    Examples:
        >>> repr(BitMask(127))
        '0b1111111'
        >>> print(BitMask(127))
        abcdefg
        >>> print(BitMask(8) + BitMask(1))
        ad
        >>> print(BitMask('a') + BitMask('d'))
        ad
    """
    KEYS = string.ascii_letters + string.digits + string.punctuation
    EMPTY = ' '
    FULL_REPR = False
    IGNORE_INVALID = True

    # ----------------------------------------------------------
    def __init__(
            self,
            value=0):
        """
        Instantiate a BitMask.

        Args:
            value (int|Sequence): The input value.
                If int, the flags of the bitmask are according to the bit
                boolean values.
                If str, uses `string.ascii_letters` + `string.digits` to set
                single bits to True in the bitmask.
        """
        if isinstance(value, int):
            self.value = value
        else:
            self.value = self.from_keys(
                value, self.KEYS, self.EMPTY, self.IGNORE_INVALID)

    # ----------------------------------------------------------
    @property
    def len(self):
        """
        Compute the maximum lenght for the bitmask (given the class keyS).

        Returns:
            result (int): The maximum length for the bitmask.

        Examples:
            >>> BitMask().len
            94
        """
        return len(self.KEYS)

    # ----------------------------------------------------------
    @property
    def set_len(self):
        """
        Compute the maximum lenght for the bitmask (given the class keyS).

        Returns:
            result (int): The maximum length for the bitmask.

        Examples:
            >>> BitMask().set_len
            94
        """
        return self.value.bit_length()

    # ----------------------------------------------------------
    def values(self):
        """
        Iterate over the flags of the bitmask.

        Yields:
            value (bool): The value

        Examples:
            >>> flags = BitMask('ac')
            >>> print([flag for flag in flags])
            [True, False, True]
            >>> print([flag for flag in BitMask(11)])
            [True, True, False, True]
        """
        for i in range(self.len):
            yield bool(self.value & (1 << i))

    # ----------------------------------------------------------
    def set_values(self):
        """
        Iterate over the set flags of the bitmask.

        Yields:
            value (bool): The value

        Examples:
            >>> flags = BitMask('ac')
            >>> print([flag for flag in flags])
            [True, False, True]
            >>> print([flag for flag in BitMask(11)])
            [True, True, False, True]
        """
        value = self.value
        i = 0
        while value:
            yield bool(value & 1)
            i += 1
            value >>= 1

    __iter__ = set_values
    __len__ = set_len

    # ----------------------------------------------------------
    def __getitem__(self, i):
        """

        Args:
            i:

        Returns:

        Examples:
            >>> flags = BitMask('ac')
            >>> print([flags[i] for i in range(4)])
            [True, False, True, False]

        """
        return bool(self.value & (1 << i))

    # ----------------------------------------------------------
    def __setitem__(self, i, x):
        """

        Args:
            i:

        Returns:

        Examples:
            >>> flags = BitMask('c')
            >>> flags[0] = True
            >>> print(flags)
            ac
            >>> flags[0] = False
            >>> print(flags)
            c
            >>> flags[1] = True
            >>> print(flags)
            bc
            >>> flags[2] = False
            >>> print(flags)
            b

        """
        if x:
            self.value = self.value | (1 << i)
        else:
            self.value = self.value & ~(1 << i)

    # ----------------------------------------------------------
    def __delitem__(self, i):
        """

        Args:
            i:

        Returns:

        """
        bit_value = self[i]
        self[i] = False
        return bit_value

    # ----------------------------------------------------------
    def __ior__(self, other):
        self.value |= other.value
        return self

    # ----------------------------------------------------------
    def __or__(self, other):
        """

        Args:
            other:

        Returns:

        Examples:
            >>> a = BitMask(1)
            >>> b = BitMask(2)
            >>> print(a + b)
            ab
            >>> print(a)
            a
            >>> print(b)
            b

        """
        result = type(self)(self.value)
        result |= other
        return result

    __add__ = __or__
    __iadd__ = __ior__

    # ----------------------------------------------------------
    def __iand__(self, other):
        self.value &= other.value
        return self

    # ----------------------------------------------------------
    def __and__(self, other):
        """

        Args:
            other:

        Returns:

        Examples:
            >>> a = BitMask(3)
            >>> b = BitMask(6)
            >>> print(a * b)
            b
            >>> print(a)
            ab
            >>> print(b)
            bc

        """
        result = type(self)(self.value)
        result &= other
        return result

    __mul__ = __and__
    __imul__ = __iand__

    # ----------------------------------------------------------
    def __ixor__(self, i):
        if isinstance(i, int):
            self.toggle(i)
            return self
        else:
            raise NotImplemented

    # ----------------------------------------------------------
    def __xor__(self, i):
        """


        Args:
            i:

        Returns:

        Examples:
            >>> flags = BitMask('bd')
            >>> flags ^= 0
            >>> print(flags)
            abd
            >>> print(flags ^ 0)
            bd
        """
        result = type(self)(self.value)
        result ^= i
        return result

    # ----------------------------------------------------------
    def __eq__(self, other):
        return type(self) == type(other) and self.value == other.value

    # ----------------------------------------------------------
    def __ne__(self, other):
        return type(self) != type(other) or self.value != other.value

    # ----------------------------------------------------------
    def __repr__(self):
        return bin(self.value)

    # ----------------------------------------------------------
    def __str__(self):
        return str(self.to_keys(
            self.KEYS, self.EMPTY, self.FULL_REPR))

    # ----------------------------------------------------------
    def __mod__(self, ext_keys):
        """

        Args:
            keys:

        Returns:

        Examples:
            >>> flags = BitMask(3)
            >>> flags % '-rwx'
            'rw-'
        """
        return self.to_keys(ext_keys[1:], ext_keys[0], True, None)

    # ----------------------------------------------------------
    def __truediv__(self, keys):
        """

        Args:
            keys:

        Returns:

        Examples:
            >>> flags = BitMask(3)
            >>> flags / '123'
            ['1', '2']
        """
        return self.to_keys(keys, None, False, list)

    # ----------------------------------------------------------
    def toggle(self, i):
        self[i] = not self[i]
        return self[i]

    # ----------------------------------------------------------
    def to_keys(
            self,
            keys=KEYS,
            empty=EMPTY,
            full=FULL_REPR,
            container=None):
        """

        Args:
            keys:
            empty:
            full:
            container:

        Returns:

        Examples:
            >>> BitMask(42).to_keys()
            'bdf'
        """
        if full:
            result = (
                key if value else empty
                for key, value in
                itertools.zip_longest(keys, self, fillvalue=False))
        else:
            result = (key for key, value in zip(keys, self) if value)
        if container is None:
            if isinstance(self.KEYS, str):
                return ''.join(result)
            else:
                return type(self.KEYS)(result)
        elif callable(container):
            return container(result)
        else:
            return result

    # ----------------------------------------------------------
    @staticmethod
    def from_keys(
            seq,
            keys=KEYS,
            empty_key=EMPTY,
            ignore=IGNORE_INVALID):
        """

        Args:
            seq:
            keys:
            empty_key:
            ignore:

        Returns:

        Examples:
            >>> BitMask.from_keys('acio')
            16645
        """
        valid_keys = set(keys)
        valid_keys.add(empty_key)
        value = 0
        for i, item in enumerate(seq):
            if item in valid_keys:
                if item != empty_key:
                    value |= 1 << keys.index(item)
            elif not ignore:
                raise ValueError(
                    fmtm('Invalid input `{item}` at index: {i}.'))
        return value

    # ----------------------------------------------------------
    def items(self):
        for key, value in zip(self.KEYS, self.values()):
            yield key, value

    # ----------------------------------------------------------
    def set_items(self):
        for key, value in zip(self.KEYS, self.values()):
            if value:
                yield key, value

    # ----------------------------------------------------------
    def as_dict(self):
        return dict(self.items())

    # ----------------------------------------------------------
    @property
    def names(self):
        class _Names(object):
            def __getattr__(_self, key):
                try:
                    i = self.KEYS.index(key)
                except ValueError:
                    i = None
                if i is None:
                    raise AttributeError(fmtm('Unknown key `{key}`.'))
                else:
                    return self[i]

            def __setattr__(_self, key, value):
                try:
                    i = self.KEYS.index(key)
                except ValueError:
                    i = None
                if i is None:
                    raise AttributeError(fmtm('Unknown key `{key}`.'))
                else:
                    self[i] = value

            def __delattr__(_self, item):
                try:
                    i = self.KEYS.index(item)
                except ValueError:
                    i = None
                if i is None:
                    raise AttributeError(fmtm('Unknown key `{key}`.'))
                else:
                    del self[i]

            def __str__(_self):
                return 'Names: ' + str(list(self.KEYS))

            __repr__ = __str__

        return _Names()


class Color(BitMask):
    KEYS = ('RED', 'GREEN', 'BLUE')


c = BitMask('dumb')

print(c)
print(repr(c))
print(dict(c.items()))
c.value = 7
print(c)
print(c.names)
c.names.b = False
print(c.names.b)
print(c)
