import string
import itertools
from flyingcircus import fmtm


# ======================================================================
class BitMask(object):
    """
    Generic bit-mask class.

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
    STR_TOKENS = '_' + string.ascii_letters + string.digits
    STR_FULL = False

    # ----------------------------------------------------------
    def __init__(
            self,
            value=None,
            num=None,
            ignore=True):
        if isinstance(value, str):
            self.value = self.from_tokens(value, self.STR_TOKENS, ignore)
        else:
            self.value = value
        if num:
            self.STR_TOKENS = self.STR_TOKENS[:num + 1]

    # ----------------------------------------------------------
    def __iter__(self):
        """

        Yields:
            value (bool): The value

        Examples:
            >>> flags = BitMask('ac')
            >>> print([flag for flag in flags])
            [True, False, True]
        """
        value = self.value
        i = 0
        while value:
            yield bool(value & 1)
            i += 1
            value >>= 1

    # ----------------------------------------------------------
    def __len__(self):
        return self.value.bit_length()

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
        return ''.join(
            self.to_tokens(
                self.STR_TOKENS[1:], self.STR_TOKENS[0], self.STR_FULL))

    # ----------------------------------------------------------
    def __mod__(self, tokens):
        """

        Args:
            tokens:

        Returns:

        Examples:
            >>> flags = BitMask(3)
            >>> flags % '-rwx'
            'rw-'
        """
        return ''.join(self.to_tokens(tokens[1:], tokens[0], True))

    # ----------------------------------------------------------
    def __truediv__(self, tokens):
        """

        Args:
            tokens:

        Returns:

        Examples:
            >>> flags = BitMask(3)
            >>> flags / '123'
            ['1', '2']
        """
        return self.to_tokens(tokens, None, False)

    # ----------------------------------------------------------
    def toggle(self, i):
        self[i] = not self[i]
        return self[i]

    # ----------------------------------------------------------
    def to_tokens(
            self,
            tokens,
            empty,
            full):
        """

        Args:
            tokens:
            empty:
            full:

        Returns:

        """
        if full:
            return [
                token if value else empty
                for token, value in
                itertools.zip_longest(tokens, self, fillvalue=False)]
        else:
            return [
                token for token, value in zip(tokens, self) if value]

    # ----------------------------------------------------------
    def from_tokens(
            self,
            seq,
            tokens,
            ignore):
        """

        Args:
            seq:
            tokens:
            ignore:

        Returns:

        """
        if tokens is None:
            tokens = self.STR_TOKENS
        tokens = tokens[1:]
        valid_tokens = set(tokens)
        value = 0
        for i, item in enumerate(seq):
            if item in valid_tokens:
                value |= 1 << tokens.index(item)
            elif not ignore:
                raise ValueError(
                    fmtm('Invalid input `{item}` at index: {i}.'))
        return value
