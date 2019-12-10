#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flyingcircus.extra: generic extra utilities (numerical).
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import os  # Miscellaneous operating system interfaces
import warnings  # Warning control
import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
import itertools  # Functions creating iterators for efficient looping
import random  # Generate pseudo-random numbers
import string  # Common string operations
import math  # Mathematical functions

# :: External Imports
import numpy as np  # NumPy (multidimensional numerical arrays library)
import scipy as sp  # SciPy (signal and image processing library)
# import matplotlib as mpl  # Matplotlib (2D/3D plotting library)
# import sympy as sym  # SymPy (symbolic CAS library)
# import PIL  # Python Image Library (image manipulation toolkit)
# import SimpleITK as sitk  # Image ToolKit Wrapper
# import nibabel as nib  # NiBabel (NeuroImaging I/O Library)
# import nipy  # NiPy (NeuroImaging in Python)
# import nipype  # NiPype (NiPy Pipelines and Interfaces)

# :: External Imports Submodules
# import matplotlib.pyplot as plt  # Matplotlib's pyplot: MATLAB-like syntax
import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
import scipy.stats  # SciPy: Statistical functions
import scipy.signal  # SciPy: Signal processing
import scipy.ndimage  # SciPy: ND-image Manipulation
import scipy.special  # SciPy: Special functions

from numpy.fft import fftshift, ifftshift
from numpy.fft import fftn, ifftn

# :: Local Imports
import flyingcircus as fc  # Everything you always wanted to have in Python.*

from flyingcircus import INFO, PATH
from flyingcircus import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from flyingcircus import elapsed, report, run_doctests
from flyingcircus import msg, dbg, fmt, fmtm
from flyingcircus import HAS_JIT, jit


# ======================================================================
def arange_nd(shape):
    """
    Generate sequential numbers shaped in a n-dim array.

    This is useful for quick testing purposes.

    Args:
        shape (Iterable[int]): The final shape of the array.

    Returns:
        result (np.ndarray): The n-dim array with sequential values.

    Examples:
        >>> print(arange_nd((2, 2)))
        [[0 1]
         [2 3]]
        >>> print(arange_nd((2, 10)))
        [[ 0  1  2  3  4  5  6  7  8  9]
         [10 11 12 13 14 15 16 17 18 19]]
    """
    return np.arange(fc.base.prod(shape)).reshape(shape)


# ======================================================================
def apply_at(
        arr,
        func,
        mask=None,
        else_=None,
        in_place=False):
    """
    Apply a function to an array.

    Warning! Depending on the value of `in_place`, this function may alter
    the input array.

    Args:
        arr (np.ndarray): The input array.
        func (callable): The function to use.
            Must have the signature: func(np.ndarray) -> np.ndarray
        mask (np.ndarray[bool]): The mask where the function is applied.
            Must have the same shape as `arr`.
        else_ (callable|Any|None): The alternate behavior.
            If callable, this is a function applied to non-masked values.
            Must have the signature: func(np.ndarray) -> np.ndarray
            If Any, the value is assigned (through broadcasting) to the
            non-masked value.
            If None, the npn-masked value are left untouched.
        in_place (bool): Determine if the function is applied in-place.
            If True, the input gets modified.
            If False, the modification happen on a copy of the input.

    Returns:
        arr (np.ndarray): The output array.

    Examples:
        >>> arr = np.arange(10) - 5
        >>> print(arr)
        [-5 -4 -3 -2 -1  0  1  2  3  4]
        >>> print(apply_at(arr, np.abs, arr < 0))
        [5 4 3 2 1 0 1 2 3 4]
        >>> print(apply_at(arr, np.abs, arr < 2, 0))
        [5 4 3 2 1 0 1 0 0 0]
        >>> print(apply_at(arr, np.abs, arr < 0, lambda x: x ** 2, True))
        [ 5  4  3  2  1  0  1  4  9 16]
        >>> print(arr)
        [ 5  4  3  2  1  0  1  4  9 16]
    """
    if not in_place:
        arr = arr.copy()
    if mask is not None:
        arr[mask] = func(arr[mask])
        if else_ is not None:
            if callable(else_):
                arr[~mask] = else_(arr[~mask])
            else:
                arr[~mask] = else_
    else:
        arr[...] = func(arr)
    return arr


# ======================================================================
def ndim_slice(
        arr,
        axes=0,
        indexes=None):
    """
    Slice a M-dim sub-array from an N-dim array (with M < N).

    Args:
        arr (np.ndarray): The input N-dim array
        axes (Iterable[int]|int): The slicing axis
        indexes (Iterable[int|float|None]|None): The slicing index.
            If None, mid-value is taken.
            Otherwise, its length must match that of axes.
            If an element is None, again the mid-value is taken.
            If an element is a number between 0 and 1, it is interpreted
            as relative to the size of the array for corresponding axis.
            If an element is an integer, it is interpreted as absolute and must
            be smaller than size of the array for the corresponding axis.

    Returns:
        sliced (np.ndarray): The sliced M-dim sub-array

    Raises:
        ValueError: if index is out of bounds

    Examples:
        >>> arr = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        >>> ndim_slice(arr, 2, 1)
        array([[ 1,  5,  9],
               [13, 17, 21]])
        >>> ndim_slice(arr, 1, 2)
        array([[ 8,  9, 10, 11],
               [20, 21, 22, 23]])
        >>> ndim_slice(arr, 0, 0)
        array([[ 0,  1,  2,  3],
               [ 4,  5,  6,  7],
               [ 8,  9, 10, 11]])
        >>> ndim_slice(arr, 0, 1)
        array([[12, 13, 14, 15],
               [16, 17, 18, 19],
               [20, 21, 22, 23]])
        >>> ndim_slice(arr, (0, 1), None)
        array([16, 17, 18, 19])
    """
    # initialize slice index
    slab = [slice(None)] * arr.ndim
    # ensure index is meaningful
    axes = fc.base.auto_repeat(axes, 1)
    if indexes is None:
        indexes = fc.base.auto_repeat(None, len(axes))
    else:
        indexes = fc.base.auto_repeat(indexes, 1)
    indexes = list(indexes)
    for i, (index, axis) in enumerate(zip(indexes, axes)):
        if index is None:
            indexes[i] = index = 0.5
        if isinstance(index, float) and index < 1.0:
            indexes[i] = int(arr.shape[axis] * index)
    # check index
    if any((index >= arr.shape[axis]) or (index < 0)
           for index, axis in zip(indexes, axes)):
        raise ValueError('Invalid array index in the specified direction')
    # determine slice index
    for index, axis in zip(indexes, axes):
        slab[axis] = index
    # print(slab)  # debug
    # slice the array
    return arr[tuple(slab)]


# ======================================================================
def ix_broadcast(slicing):
    """
    Automatically broadcast multiple-indexes for N-dim multi-slicing.

    The recommended usage pattern is:

    .. code:: python

        ixb = ix_broadcast
        arr[ixb(a, b, c)]

    in place of:

    .. code:: python

        arr[a, b, c]

    Note that, due to the way NumPy deals with mixed simple and advanced
    indexing this is, in general, not equivalent to:

    .. code:: python

        arr[a, :, :][:, b, :][:, :, c]

    To obtain the above use `flyingcircus.extra.multi_slicing()`.

    Args:
        slicing (Iterable[slice|int|Iterable[int]]): A sequence of slices.
            The slicing is applied such that non-int and non-slice elements
            are automatically broadcasted together.

    Returns:
        result (tuple[slice|int|tuple[int]]: The broadcasted indexes.

    See Also:
        - flyingcircus.extra.multi_slicing()

    Examples:
        >>> arr = arange_nd((3, 4, 5))
        >>> ixb = ix_broadcast  # recommended usage
        >>> ixb_ = ix_broadcast_  # recommended usage
        >>> slicing = (slice(None), (0, 2, 3), (0, 2, 3, 4))
        >>> new_arr = arr[ixb(slicing)]
        >>> print(new_arr.shape)
        (3, 3, 4)

        >>> slicing = (slice(2), (0, 2, 3), (0, 2, 3, 4))
        >>> new_arr = arr[ixb(slicing)]
        >>> print(new_arr.shape)
        (2, 3, 4)

        >>> slicing = (slice(2), (0,), (0, 2, 3, 4))
        >>> new_arr = arr[ixb(slicing)]
        >>> print(new_arr.shape)
        (2, 1, 4)

        >>> slicing = (slice(2), 1, (0, 2, 3, 4))
        >>> new_arr = arr[ixb(slicing)]
        >>> print(new_arr.shape)
        (2, 4)

        >>> slicing = ((0, 1), slice(3), (0, 3, 4))
        >>> new_arr = arr[ixb(slicing)]
        >>> print(new_arr.shape)
        (2, 3, 3)

        >>> new_arr = arr[ixb_(0, 0, slice(3))]
        >>> print(new_arr.shape)
        (3,)

        >>> new_arr = arr[ixb_(0, 0, 0)]
        >>> print(new_arr.shape)
        ()

        >>> slicing = (slice(0, 1), slice(3), (0, 1))
        >>> new_slicing = ix_broadcast(slicing)
        >>> print(new_slicing)
        (slice(0, 1, None), slice(None, 3, None), array([0, 1]))
        >>> new_arr = arr[slicing]
        >>> print(new_arr.shape)
        (1, 3, 2)
        >>> new_arr = arr[new_slicing]
        >>> print(new_arr.shape)
        (1, 3, 2)

        >>> slicing = (0, slice(3), (0, 1))
        >>> new_slicing = ix_broadcast(slicing)
        >>> print(new_slicing)
        (0, slice(None, 3, None), array([0, 1]))
        >>> new_arr = arr[slicing]
        >>> print(new_arr.shape)
        (2, 3)
        >>> new_arr = arr[new_slicing]
        >>> print(new_arr.shape)
        (2, 3)

        >>> print(
        ...     np.all(arr[ixb((slice(None), (0, 1), (0, 1)))]
        ...     == arr[:, :2, :2]))
        True

        >>> print(  # False -> reordering in mixed simple/advanced indexing
        ...     np.all(arr[ixb(((0, 1), slice(2), (0, 1)))]
        ...     == arr[:2, :2, :2]))
        False
    """
    try:
        # : always keep dims
        # indexes, objs = tuple(zip(*(
        #     (i, obj if not isinstance(obj, int) else (obj,))
        #     for i, obj in enumerate(slicing) if not isinstance(obj, slice))))
        # : relies on default NumPy behavior for mixed indices
        indexes, objs = tuple(zip(*(
            (i, obj) for i, obj in enumerate(slicing)
            if not isinstance(obj, (int, slice)))))
        broadcasted = np.ix_(*objs)
    except ValueError:
        result = slicing
    else:
        result = list(slicing)
        for j, i in enumerate(indexes):
            result[i] = broadcasted[j]
    return tuple(result)


# ======================================================================
def ix_broadcast_(*slicing):
    """
    Star magic version of `flyingcircus.extra.ix_broadcast()`.
    """
    return ix_broadcast(slicing)


# ======================================================================
def multi_slicing(
        arr,
        slicing):
    """
    Slice an array object with automatically broadcasted multiple indexes.

    This is useful to ensure that Iterable elements of the slicing are
    automatically broadcasted together.

    Args:
        arr (np.ndarray): The input array.
        slicing (Iterable[slice|int|Iterable[int]]): A sequence of slices.
            The slicing is applied such that non-int and non-slice elements
            are automatically broadcasted together.

    Returns:
        result (np.ndarray): The multi-sliced array.

    Examples:
        >>> arr = arange_nd((3, 4, 5))
        >>> slicing = (slice(None), (0, 2, 3), (0, 2, 3, 4))
        >>> new_arr = multi_slicing(arr, slicing)
        >>> print(new_arr.shape)
        (3, 3, 4)
        >>> np.all(
        ...     multi_slicing(arr, slicing)
        ...     == arr[:, (0, 2, 3), :][:, :, (0, 2, 3, 4)])
        True
        >>> np.all(
        ...     multi_slicing(arr, slicing) == arr[ix_broadcast(slicing)])
        True

        >>> slicing = (slice(2), (0, 2, 3), (0, 2, 3, 4))
        >>> new_arr = multi_slicing(arr, slicing)
        >>> print(new_arr.shape)
        (2, 3, 4)
        >>> np.all(
        ...     multi_slicing(arr, slicing)
        ...     == arr[:2, (0, 2, 3), :][:2, :, (0, 2, 3, 4)])
        True
        >>> np.all(
        ...     multi_slicing(arr, slicing) == arr[ix_broadcast(slicing)])
        True

        >>> slicing = (slice(2), (0,), (0, 2, 3, 4))
        >>> new_arr = multi_slicing(arr, slicing)
        >>> print(new_arr.shape)
        (2, 1, 4)
        >>> np.all(
        ...     multi_slicing(arr, slicing)
        ...     == arr[:2, (0,), :][:2, :, (0, 2, 3, 4)])
        True
        >>> np.all(
        ...     multi_slicing(arr, slicing) == arr[ix_broadcast(slicing)])
        True

        >>> slicing = (slice(2), 1, (0, 2, 3, 4))
        >>> new_arr = multi_slicing(arr, slicing)
        >>> print(new_arr.shape)
        (2, 4)
        >>> np.all(
        ...     multi_slicing(arr, slicing)
        ...     == arr[:2, 1, :][:2, (0, 2, 3, 4)])
        True
        >>> np.all(
        ...     multi_slicing(arr, slicing) == arr[ix_broadcast(slicing)])
        True

        >>> slicing = ((0, 1), slice(3), (0, 1, 2, 3))
        >>> new_arr = multi_slicing(arr, slicing)
        >>> print(new_arr.shape)
        (2, 3, 4)
        >>> np.all(
        ...     multi_slicing(arr, slicing) == arr[:2, :3, :4])
        True
        >>> np.all(
        ...     multi_slicing(arr, slicing)
        ...     == arr[:2, :, :][:, :3, :][:, :, :4])
        True
        >>> np.all(  # False -> reordering in mixed simple/advanced indexing
        ...     multi_slicing(arr, slicing) == arr[ix_broadcast(slicing)])
        False
    """
    if sum(1 for obj in slicing if not isinstance(obj, (slice, int))) > 1:
        # # : alternate method with reshape at the end
        # result = arr
        # base_slicing = [slice(None) for obj_ in slicing]
        # for i, obj in enumerate(slicing):
        #     base_slicing[i] = obj \
        #         if not isinstance(obj, int) else slice(obj, obj + 1)
        #     result = result[tuple(base_slicing)]
        #     base_slicing[i] = slice(None)
        # true_shape = tuple(
        #     dim for dim, obj in zip(result.shape, slicing)
        #     if not isinstance(obj, int))
        # result = result.reshape(true_shape)

        result = arr
        base_slicing = [slice(None)] * len(slicing)
        i = 0
        for obj in slicing:
            base_slicing[i] = obj
            result = result[tuple(base_slicing)]
            if isinstance(obj, int):
                del base_slicing[i]
            else:
                base_slicing[i] = slice(None)
                i += 1
    else:
        result = arr[slicing]
    return result


# ======================================================================
def nbytes(arr):
    """
    Determine the actual memory consumption of a NumPy array.

    Derive the `nbytes` value from the `base` attribute recursively.
    Works both for `np.broadcast_to()` and `np.lib.stride_tricks.as_strided()`.

    Args:
        arr (np.array): The input array.

    Returns:
        result (int): The actual memory consumption of a NumPy array.
            This also works for views/broadcasted/strided arrays.

    Examples:
        >>> arr = np.array([1, 2, 3], dtype=np.int64)
        >>> print(nbytes(arr) == arr.nbytes)
        True

        >>> new_arr = np.broadcast_to(arr.reshape(3, 1), (3, 1000))
        >>> print(nbytes(new_arr) == new_arr.nbytes)
        False
        >>> print(nbytes(new_arr) == new_arr.base.nbytes)
        True
        >>> print(nbytes(arr), nbytes(new_arr), new_arr.nbytes)
        24 24 24000

        >>> new_arr = np.lib.stride_tricks.as_strided(
        ...     arr, shape=(3, 1000), strides=(8, 0), writeable=False)
        >>> print(nbytes(new_arr) == new_arr.nbytes)
        False
        >>> print(nbytes(new_arr) == new_arr.base.base.nbytes)
        True
        >>> print(nbytes(arr), nbytes(new_arr), new_arr.nbytes)
        24 24 24000
    """
    return arr.nbytes if arr.base is None else nbytes(arr.base)


# ======================================================================
def nd_windowing(
        arr,
        window,
        steps=1,
        window_steps=1,
        as_view=True,
        writeable=False,
        shape_mode='end'):
    """
    Generate a N-dimensional moving windowing view of an array.

    Args:
        arr (np.ndarray): The input array.
        window (int|Iterable[int]): The window sizes.
        steps (int|Iterable[int]): The step sizes.
            This determines the step used for moving to the next window.
        window_steps (int|Iterable[int]): The window step sizes.
            This determines the step used for moving within the window.
        as_view (bool): Determine if the result uses additional memory.
            If True, a view on the original input is given.
            If False, each entry of the result will have its own memory.
            If False and `writeable` is True, then evenutal changes to `result`
            will back-propagate to the input and vice-versa.
        writeable (bool): Determine if the result entries can be overwritten.
            If True and `as_view` is True, then eventual changes to `result`,
            will back-propagate to the input and vice-versa.
            If `as_view` is False, this has no effect.
        shape_mode (str): Determine the shape of the result.
            Accepted values are:
             - 'begin': Window shape dims are at the beginning of shape
             - 'end': Window shape dims are at the end of shape
             - 'mix': Window shape dims are mixed with window position dims.
             - 'mix_r': Window position dims are mixed with window shape dims.

    Returns:
        result (np.ndarray): The windowing array.
            This has shape equal to `arr.shape` + `sizes`.

    Examples:
        >>> print(nd_windowing(np.zeros((11, 13, 17)), (2, 3, 2)).shape)
        (10, 11, 16, 2, 3, 2)
        >>> print(nd_windowing(np.zeros((11, 13, 17)), 2).shape)
        (10, 12, 16, 2, 2, 2)
        >>> print(nd_windowing(np.zeros((11, 13, 17)), 3).shape)
        (9, 11, 15, 3, 3, 3)
        >>> print(nd_windowing(np.zeros((11, 13, 17)), 3, 2).shape)
        (5, 6, 8, 3, 3, 3)

        Block-wise view
        >>> arr = arange_nd((2 * 2, 2 * 3))
        >>> print(arr)
        [[ 0  1  2  3  4  5]
         [ 6  7  8  9 10 11]
         [12 13 14 15 16 17]
         [18 19 20 21 22 23]]
        >>> print(nd_windowing(arr, (2, 2), (2, 2)))
        [[[[ 0  1]
           [ 6  7]]
        <BLANKLINE>
          [[ 2  3]
           [ 8  9]]
        <BLANKLINE>
          [[ 4  5]
           [10 11]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[12 13]
           [18 19]]
        <BLANKLINE>
          [[14 15]
           [20 21]]
        <BLANKLINE>
          [[16 17]
           [22 23]]]]

        >>> arr = arange_nd((4, 5))
        >>> print(arr)
        [[ 0  1  2  3  4]
         [ 5  6  7  8  9]
         [10 11 12 13 14]
         [15 16 17 18 19]]

        >>> print(nd_windowing(arr, (2, 1), shape_mode='begin').shape)
        (2, 1, 3, 5)
        >>> print(nd_windowing(arr, (2, 1), shape_mode='end').shape)
        (3, 5, 2, 1)
        >>> print(nd_windowing(arr, (2, 1), shape_mode='mix').shape)
        (3, 2, 5, 1)
        >>> print(nd_windowing(arr, (2, 1), shape_mode='mix_r').shape)
        (2, 3, 1, 5)

        >>> print(nd_windowing(arr, (2, 3)))
        [[[[ 0  1  2]
           [ 5  6  7]]
        <BLANKLINE>
          [[ 1  2  3]
           [ 6  7  8]]
        <BLANKLINE>
          [[ 2  3  4]
           [ 7  8  9]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[ 5  6  7]
           [10 11 12]]
        <BLANKLINE>
          [[ 6  7  8]
           [11 12 13]]
        <BLANKLINE>
          [[ 7  8  9]
           [12 13 14]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[10 11 12]
           [15 16 17]]
        <BLANKLINE>
          [[11 12 13]
           [16 17 18]]
        <BLANKLINE>
          [[12 13 14]
           [17 18 19]]]]

        >>> print(nd_windowing(arr, (2, 3), (2, 2), shape_mode='end'))
        [[[[ 0  1  2]
           [ 5  6  7]]
        <BLANKLINE>
          [[ 2  3  4]
           [ 7  8  9]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[10 11 12]
           [15 16 17]]
        <BLANKLINE>
          [[12 13 14]
           [17 18 19]]]]
        >>> print(nd_windowing(arr, (2, 3), (2, 2), shape_mode='begin'))
        [[[[ 0  2]
           [10 12]]
        <BLANKLINE>
          [[ 1  3]
           [11 13]]
        <BLANKLINE>
          [[ 2  4]
           [12 14]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[ 5  7]
           [15 17]]
        <BLANKLINE>
          [[ 6  8]
           [16 18]]
        <BLANKLINE>
          [[ 7  9]
           [17 19]]]]
        >>> print(nd_windowing(arr, (2, 3), (2, 2), shape_mode='mix'))
        [[[[ 0  1  2]
           [ 2  3  4]]
        <BLANKLINE>
          [[ 5  6  7]
           [ 7  8  9]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[10 11 12]
           [12 13 14]]
        <BLANKLINE>
          [[15 16 17]
           [17 18 19]]]]
        >>> print(nd_windowing(arr, (2, 3), (2, 2), shape_mode='mix_r'))
        [[[[ 0  2]
           [ 1  3]
           [ 2  4]]
        <BLANKLINE>
          [[10 12]
           [11 13]
           [12 14]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[ 5  7]
           [ 6  8]
           [ 7  9]]
        <BLANKLINE>
          [[15 17]
           [16 18]
           [17 19]]]]

        >>> print(nd_windowing(arr, 2, 1, 2))
        [[[[ 0  2]
           [10 12]]
        <BLANKLINE>
          [[ 1  3]
           [11 13]]
        <BLANKLINE>
          [[ 2  4]
           [12 14]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[ 5  7]
           [15 17]]
        <BLANKLINE>
          [[ 6  8]
           [16 18]]
        <BLANKLINE>
          [[ 7  9]
           [17 19]]]]

        >>> print(nd_windowing(arr, (2, 3), 2, 0))
        [[[[ 0  0  0]
           [ 0  0  0]]
        <BLANKLINE>
          [[ 2  2  2]
           [ 2  2  2]]
        <BLANKLINE>
          [[ 4  4  4]
           [ 4  4  4]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[10 10 10]
           [10 10 10]]
        <BLANKLINE>
          [[12 12 12]
           [12 12 12]]
        <BLANKLINE>
          [[14 14 14]
           [14 14 14]]]]

        >>> print(nd_windowing(arr, (2, 3), 2, (0, 1)))
        [[[[ 0  1  2]
           [ 0  1  2]]
        <BLANKLINE>
          [[ 2  3  4]
           [ 2  3  4]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[10 11 12]
           [10 11 12]]
        <BLANKLINE>
          [[12 13 14]
           [12 13 14]]]]

        >>> new_arr = nd_windowing(arr, 2, 3, 1)
        >>> new_arr[0] = 100
        Traceback (most recent call last):
            ....
        ValueError: assignment destination is read-only

        >>> new_arr = nd_windowing(arr, 2, 3, (0, 1), True, True)
        >>> print(new_arr)
        [[[[ 0  1]
           [ 0  1]]
        <BLANKLINE>
          [[ 3  4]
           [ 3  4]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[15 16]
           [15 16]]
        <BLANKLINE>
          [[18 19]
           [18 19]]]]
        >>> new_arr[0, 0, 0, 0] = 100
        >>> print(new_arr)
        [[[[100   1]
           [100   1]]
        <BLANKLINE>
          [[  3   4]
           [  3   4]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[ 15  16]
           [ 15  16]]
        <BLANKLINE>
          [[ 18  19]
           [ 18  19]]]]
        >>> print(arr)
        [[100   1   2   3   4]
         [  5   6   7   8   9]
         [ 10  11  12  13  14]
         [ 15  16  17  18  19]]
        >>> arr[0, 0] = 0
        >>> print(new_arr)
        [[[[ 0  1]
           [ 0  1]]
        <BLANKLINE>
          [[ 3  4]
           [ 3  4]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[15 16]
           [15 16]]
        <BLANKLINE>
          [[18 19]
           [18 19]]]]
        >>> print(nbytes(new_arr), new_arr.nbytes)
        160 128

        >>> new_arr = nd_windowing(arr, 2, 3, (0, 1), False)
        >>> new_arr[0, 0, 0, 0] = 100
        >>> print(new_arr)
        [[[[100   1]
           [  0   1]]
        <BLANKLINE>
          [[  3   4]
           [  3   4]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[ 15  16]
           [ 15  16]]
        <BLANKLINE>
          [[ 18  19]
           [ 18  19]]]]
        >>> print(nbytes(new_arr), new_arr.nbytes)
        128 128

        >>> new_arr = nd_windowing(arange_nd((100, 100)), 5)
        >>> print(nbytes(new_arr), new_arr.nbytes)
        80000 1843200

        >>> new_arr = nd_windowing(arange_nd((8,)), 2, 2)
        >>> print(new_arr)
        [[0 1]
         [2 3]
         [4 5]
         [6 7]]

        >>> new_arr = nd_windowing(arange_nd((9,)), 2, 2)
        >>> print(new_arr)
        [[0 1]
         [2 3]
         [4 5]
         [6 7]]

        >>> new_arr = nd_windowing(arange_nd((5, 4)), (1, 2), (1, 2))
        >>> print(new_arr)
        [[[[ 0  1]]
        <BLANKLINE>
          [[ 2  3]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[ 4  5]]
        <BLANKLINE>
          [[ 6  7]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[ 8  9]]
        <BLANKLINE>
          [[10 11]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[12 13]]
        <BLANKLINE>
          [[14 15]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[16 17]]
        <BLANKLINE>
          [[18 19]]]]
    """
    shape_mode = shape_mode.lower()
    window = fc.base.auto_repeat(window, arr.ndim, check=True)
    steps = fc.base.auto_repeat(steps, arr.ndim, check=True)
    window_steps = fc.base.auto_repeat(window_steps, arr.ndim, check=True)
    assert (all(step > 0 for step in steps))
    assert (all(w_step >= 0 for w_step in window_steps))
    assert (all(dim >= size for dim, size in zip(arr.shape, window)))
    reduced_shape = tuple(
        (dim - w_step * (size - 1) + step - 1) // step
        for dim, size, step, w_step
        in zip(arr.shape, window, steps, window_steps))
    reduced_strides = tuple(
        stride * max(1, step) for stride, step in zip(arr.strides, steps))
    window_strides = tuple(
        stride * max(0, w_step)
        for stride, w_step in zip(arr.strides, window_steps))
    if shape_mode == 'begin':
        shape = tuple(window) + reduced_shape
        strides = window_strides + reduced_strides
    elif shape_mode == 'end':
        shape = reduced_shape + tuple(window)
        strides = reduced_strides + window_strides
    elif shape_mode == 'mix':
        shape = tuple(
            x for xs in zip(reduced_shape, window) for x in xs)
        strides = tuple(
            x for xs in zip(reduced_strides, window_strides) for x in xs)
    elif shape_mode == 'mix_r':
        shape = tuple(
            x for xs in zip(window, reduced_shape) for x in xs)
        strides = tuple(
            x for xs in zip(window_strides, reduced_strides) for x in xs)
    else:
        raise ValueError('shape_mode `{}` not supported.'.format(shape_mode))
    if 0 in shape:
        shape = tuple(
            dim for dim in shape if dim > 0)
        strides = tuple(
            stride for dim, stride in zip(shape, strides) if dim > 0)
    result = np.lib.stride_tricks.as_strided(
        arr, shape=shape, strides=strides, writeable=writeable)
    if not as_view:
        result = result.copy()
    return result


# ======================================================================
def separate(
        arr,
        size=2,
        axis=0,
        keepdims=False,
        truncate=False,
        fill=0,
        shape_mode='end'):
    """
    Separate an array into blocks with fixed size along a specific axis.

    This is equivalent to `flyingcircus.extra.nd_windowing()` with
    `window == steps` and `window_steps == 1`.

    Additionally, this function offers better control over the handling of
    the array shape not aligned with size.

    This is a multidimensional version of `flyingcircus.base.separate()`.

    Args:
        arr (np.ndarray): The input array.
        size (int): The size of the separated blocks.
        axis (int): The axis along which to separate.
        keepdims (bool): Keep all the dimensions for the results.
        truncate (bool): Determine how to handle uneven splits.
            If True, last block is omitted if smaller than `size`.
        fill (Any): Value to use for filling the last block.
            This is only used when `truncate` is False.
        shape_mode (str): Determine the shape of the result.
            See `flyingcircus.extra.nd_windowing()` for more details.

    Returns:
        result (np.ndarray): The output array.
            If size of `arr` along `axis` is a multiple of `size`, the
            result is a non-writable view of the input array.

    Examples:
        >>> print(separate(np.arange(10)))
        [[0 1]
         [2 3]
         [4 5]
         [6 7]
         [8 9]]
        >>> print(separate(np.arange(9)))
        [[0 1]
         [2 3]
         [4 5]
         [6 7]
         [8 0]]
        >>> print(separate(np.arange(9), truncate=True))
        [[0 1]
         [2 3]
         [4 5]
         [6 7]]

        >>> new_arr = separate(arange_nd((3, 4)), 2, 0)
        >>> print(new_arr.shape)
        (2, 4, 2)
        >>> print(new_arr)
        [[[ 0  4]
          [ 1  5]
          [ 2  6]
          [ 3  7]]
        <BLANKLINE>
         [[ 8  0]
          [ 9  0]
          [10  0]
          [11  0]]]
        >>> new_arr = separate(arange_nd((3, 4)), 2, 1)
        >>> print(new_arr.shape)
        (3, 2, 2)
        >>> print(new_arr)
        [[[ 0  1]
          [ 2  3]]
        <BLANKLINE>
         [[ 4  5]
          [ 6  7]]
        <BLANKLINE>
         [[ 8  9]
          [10 11]]]
    """
    assert(-arr.ndim <= axis < arr.ndim)
    axis %= arr.ndim
    if arr.shape[axis] % size != 0:
        aligned = fc.base.align(arr.shape[axis], size, -1 if truncate else 1)
        if truncate:
            slicing = tuple(
                slice(None) if i != axis else slice(None, aligned)
                for i in range(arr.ndim))
            arr = arr[slicing]
        else:
            fill_shape = tuple(
                dim if i != axis else aligned - dim
                for i, dim in enumerate(arr.shape))
            fill_arr = np.full(fill_shape, fill, dtype=arr.dtype)
            arr = np.concatenate([arr, fill_arr], axis=axis)
    window = tuple(1 if i != axis else size for i in range(arr.ndim))
    result = nd_windowing(arr, window, window, shape_mode=shape_mode)
    if not keepdims:
        result = np.squeeze(result)
    return result


# ======================================================================
def compute_edge_weights(
        arr,
        weighting=lambda x, y: x + y,
        weighting_kws=None,
        circular=False,
        endpoint=np.nan):
    """
    Compute the weights associate to edges for a given input.

    These are obtained by computing some weighting function over adjacent
    data elements. The computation is vectorized.

    Args:
        arr (np.ndarray): The input array.
        weighting (callable): The function for computing the weighting.
            Must have the following signature:
            weighting(np.ndarray, np.ndarray, ...) -> np.ndarray
        weighting_kws (Mappable|None): Keyword arguments.
            These are passed to the function specified in `weighting`.
            If Iterable, must be convertible to a dictionary.
            If None, no keyword arguments will be passed.
        circular (bool|Iterable[bool]): Specify if circularly connected.
            If Iterable, each axis can be specified separately.
            If True, the input array is considered circularly connected.
            If False, the input is not circularly connected and the
            index arrays are set to `-1`.
            Note that when both `orig_idx_arr` and `dest_idx_arr` elements
            are negative, these means that the edges are unconnected.
        endpoint (int|float): The value to assign to endpoint edges.
            This value is assigned to endpoint edge weights, only if
            circular is False.

    Returns:
        result (tuple): The tuple
            contains:
             - edge_weights_arr (np.ndarray): The edge weightings.
             - orig_idx_arr (np.ndarray): The indexes of the edge origins.
                   Both `edge_weights_arr` and `orig_idx_arr` must be ravelled
                   to be used.
             - dest_idx_arr (np.ndarray): The indexes of the edge destinations.
                   Both `edge_weights_arr` and `orig_idx_arr` must be ravelled
                   to be used.

    Examples:
        >>> arr = np.arange((2 * 3)).reshape(2, 3)
        >>> print(arr)
        [[0 1 2]
         [3 4 5]]

        >>> edge_weights_arr, o_idx_arr, d_idx_arr = compute_edge_weights(arr)
        >>> print(edge_weights_arr)
        [[[ 3.  1.]
          [ 5.  3.]
          [ 7. nan]]
        <BLANKLINE>
         [[nan  7.]
          [nan  9.]
          [nan nan]]]
        >>> print(o_idx_arr)
        [[[ 0  0]
          [ 1  1]
          [ 2 -1]]
        <BLANKLINE>
         [[-1  3]
          [-1  4]
          [-1 -1]]]
        >>> print(d_idx_arr)
        [[[ 3  1]
          [ 4  2]
          [ 5 -1]]
        <BLANKLINE>
         [[-1  4]
          [-1  5]
          [-1 -1]]]

        >>> edge_weights_arr, o_idx_arr, d_idx_arr = compute_edge_weights(
        ...     arr, circular=True)
        >>> print(edge_weights_arr)
        [[[3 1]
          [5 3]
          [7 2]]
        <BLANKLINE>
         [[3 7]
          [5 9]
          [7 8]]]
        >>> print(o_idx_arr)
        [[[0 0]
          [1 1]
          [2 2]]
        <BLANKLINE>
         [[3 3]
          [4 4]
          [5 5]]]
        >>> print(d_idx_arr)
        [[[3 1]
          [4 2]
          [5 0]]
        <BLANKLINE>
         [[0 4]
          [1 5]
          [2 3]]]
    """
    endpoint_idx = -1
    weighting_kws = dict(weighting_kws) if weighting_kws is not None else {}
    windows = (slice(None, -1), slice(1, None))
    # : implemented with list comprehension for speed
    edge_weights_arr = np.stack([
        np.concatenate((
            weighting(
                arr[tuple(
                    slice(None) if i != j else windows[0]
                    for j in range(arr.ndim))],
                arr[tuple(
                    slice(None) if i != j else windows[1]
                    for j in range(arr.ndim))],
                **weighting_kws),
            weighting(
                arr[tuple(
                    slice(None) if i != j else fc.base.flip(windows[0])
                    for j in range(arr.ndim))],
                arr[tuple(
                    slice(None) if i != j else fc.base.flip(windows[1])
                    for j in range(arr.ndim))],
                **weighting_kws)
            if circular else
            np.full(tuple(
                1 if i == j else d
                for j, d in enumerate(arr.shape)),
                endpoint)),
            axis=i)
        for i in range(arr.ndim)], axis=-1)
    idx_arr = np.arange(fc.base.prod(arr.shape), dtype=int).reshape(arr.shape)
    orig_idx_arr, dest_idx_arr = tuple(
        np.stack([
            np.concatenate((
                idx_arr[tuple(
                    slice(None) if i != j else window
                    for j in range(idx_arr.ndim))],
                idx_arr[tuple(
                    slice(None) if i != j else fc.base.flip(window)
                    for j in range(idx_arr.ndim))]
                if circular else
                np.full(tuple(
                    1 if i == j else d
                    for j, d in enumerate(idx_arr.shape)),
                    endpoint_idx)),
                axis=i)
            for i in range(arr.ndim)],
            axis=-1)
        for window in windows)
    return edge_weights_arr, orig_idx_arr, dest_idx_arr


# ======================================================================
def shuffle_on_axis(arr, axis=-1):
    """
    Shuffle the elements of the array separately along the specified axis.

    By contrast `numpy.random.shuffle()` shuffle **by** axis and only on the
    first axis.

    Args:
        arr (np.ndarray): The input array.
        axis (int): The axis along which to shuffle.

    Returns:
        result (np.ndarray): The shuffled array.

    Examples:
        >>> np.random.seed(0)
        >>> shape = 2, 3, 4
        >>> arr = np.arange(fc.base.prod(shape)).reshape(shape)
        >>> shuffle_on_axis(arr.copy())
        array([[[ 1,  0,  2,  3],
                [ 6,  4,  5,  7],
                [10,  8, 11,  9]],
        <BLANKLINE>
               [[12, 15, 13, 14],
                [18, 17, 16, 19],
                [21, 20, 23, 22]]])
        >>> shuffle_on_axis(arr.copy(), 0)
        array([[[ 0, 13,  2, 15],
                [16,  5,  6, 19],
                [ 8,  9, 10, 23]],
        <BLANKLINE>
               [[12,  1, 14,  3],
                [ 4, 17, 18,  7],
                [20, 21, 22, 11]]])
    """
    arr = np.swapaxes(arr, 0, axis)
    shape = arr.shape
    i = np.random.rand(*arr.shape).argsort(0).reshape(shape[0], -1)
    return arr.reshape(shape[0], -1)[i, np.arange(fc.base.prod(shape[1:]))]. \
        reshape(shape).swapaxes(axis, 0)


# ======================================================================
def is_broadcastable(items):
    """
    Check if some arrays (or their shapes) can be broadcasted together.

    Two (or more) arrays can be broadcasted together if their dims
    (starting from the last) are either the same or one of the is 1.

    Args:
        items (Iterable[Sequence|np.ndarray]): The shapes to test.
            Each item is either a NumPy array (more precisely any object
            with a `shape` attribute) or a Sequence representing the shape
            of an array.

    Returns:
        result (bool): The result of the broadcastability test.

    Examples:
        >>> is_broadcastable(((2, 3, 1), (4,)))
        True
        >>> is_broadcastable(((8, 1, 6, 1), (7, 1, 5)))
        True
        >>> is_broadcastable(((2, 1), (8, 4, 3)))
        False
        >>> is_broadcastable(
        ...      ((8, 1, 6, 1), (7, 1, 5), (7, 6, 5), (8, 1, 1, 1)))
        True
        >>> is_broadcastable((np.zeros((4, 3, 2, 1)), (3, 1, 5)))
        True
        >>> is_broadcastable(((8, 4, 3), np.zeros((2, 1))))
        False
    """
    shapes = tuple(
        item.shape if hasattr(item, 'shape') else item
        for item in items)
    return all(
        fc.base.all_equal(dim for dim in dims if dim > 1)
        for dims in zip(*tuple(shape[::-1] for shape in shapes)))


# ======================================================================
def unsqueezing(
        source_shape,
        target_shape):
    """
    Generate a broadcasting-compatible shape.

    The resulting shape contains *singletons* (i.e. `1`) for non-matching dims.
    Assumes all elements of the source shape are contained in the target shape
    (excepts for singletons) in the correct order.

    Warning! The generated shape may not be unique if some of the elements
    from the source shape are present multiple times in the target shape.

    Args:
        source_shape (Sequence): The source shape.
        target_shape (Sequence): The target shape.

    Returns:
        shape (tuple): The broadcast-safe shape.

    Raises:
        ValueError: if elements of `source_shape` are not in `target_shape`.

    Examples:
        For non-repeating elements, `unsqueezing()` is always well-defined:

        >>> unsqueezing((2, 3), (2, 3, 4))
        (2, 3, 1)
        >>> unsqueezing((3, 4), (2, 3, 4))
        (1, 3, 4)
        >>> unsqueezing((3, 5), (2, 3, 4, 5, 6))
        (1, 3, 1, 5, 1)
        >>> unsqueezing((1, 3, 5, 1), (2, 3, 4, 5, 6))
        (1, 3, 1, 5, 1)

        If there is nothing to unsqueeze, the `source_shape` is returned:

        >>> unsqueezing((1, 3, 1, 5, 1), (2, 3, 4, 5, 6))
        (1, 3, 1, 5, 1)
        >>> unsqueezing((2, 3), (2, 3))
        (2, 3)

        If some elements in `source_shape` are repeating in `target_shape`,
        a user warning will be issued:

        >>> unsqueezing((2, 2), (2, 2, 2, 2, 2))
        (2, 2, 1, 1, 1)
        >>> unsqueezing((2, 2), (2, 3, 2, 2, 2))
        (2, 1, 2, 1, 1)

        If some elements of `source_shape` are not presente in `target_shape`,
        an error is raised.

        >>> unsqueezing((2, 3), (2, 2, 2, 2, 2))
        Traceback (most recent call last):
          ...
        ValueError: Target shape must contain all source shape elements\
 (in correct order). (2, 3) -> (2, 2, 2, 2, 2)
        >>> unsqueezing((5, 3), (2, 3, 4, 5, 6))
        Traceback (most recent call last):
          ...
        ValueError: Target shape must contain all source shape elements\
 (in correct order). (5, 3) -> (2, 3, 4, 5, 6)

    """
    shape = []
    j = 0
    for i, dim in enumerate(target_shape):
        if j < len(source_shape):
            shape.append(dim if dim == source_shape[j] else 1)
            if i + 1 < len(target_shape) and dim == source_shape[j] \
                    and dim != 1 and dim in target_shape[i + 1:]:
                text = (
                    'Multiple positions (e.g. {} and {})'
                    ' for source shape element {}.'.format(
                        i, target_shape[i + 1:].index(dim) + (i + 1), dim))
                warnings.warn(text)
            if dim == source_shape[j] or source_shape[j] == 1:
                j += 1
        else:
            shape.append(1)
    if j < len(source_shape):
        raise ValueError(
            'Target shape must contain all source shape elements'
            ' (in correct order). {} -> {}'.format(source_shape, target_shape))
    return tuple(shape)


# ======================================================================
def unsqueeze(
        arr,
        axis=None,
        shape=None,
        complement=False):
    """
    Add singletons to the shape of an array to broadcast-match a given shape.

    In some sense, this function implements the inverse of `numpy.squeeze()`.

    Args:
        arr (np.ndarray): The input array.
        axis (int|Iterable|None): Axis or axes in which to operate.
            If None, a valid set axis is generated from `shape` when this is
            defined and the shape can be matched by `unsqueezing()`.
            If int or Iterable, specified how singletons are added.
            This depends on the value of `complement`.
            If `shape` is not None, the `axis` and `shape` parameters must be
            consistent.
            Values must be in the range [-(ndim+1), ndim+1]
            At least one of `axis` and `shape` must be specified.
        shape (int|Iterable|None): The target shape.
            If None, no safety checks are performed.
            If int, this is interpreted as the number of dimensions of the
            output array.
            If Iterable, the result must be broadcastable to an array with the
            specified shape.
            If `axis` is not None, the `axis` and `shape` parameters must be
            consistent.
            At least one of `axis` and `shape` must be specified.
        complement (bool): Interpret `axis` parameter as its complementary.
            If True, the dims of the input array are placed at the positions
            indicated by `axis`, and singletons are placed everywherelse and
            the `axis` length must be equal to the number of dimensions of the
            input array; the `shape` parameter cannot be `None`.
            If False, the singletons are added at the position(s) specified by
            `axis`.
            If `axis` is None, `complement` has no effect.

    Returns:
        arr (np.ndarray): The reshaped array.

    Raises:
        ValueError: if the `arr` shape cannot be reshaped correctly.

    Examples:
        Let's define some input array `arr`:

        >>> arr = np.arange(2 * 3 * 4).reshape((2, 3, 4))
        >>> arr.shape
        (2, 3, 4)

        A call to `unsqueeze()` can be reversed by `np.squeeze()`:

        >>> arr_ = unsqueeze(arr, (0, 2, 4))
        >>> arr_.shape
        (1, 2, 1, 3, 1, 4)
        >>> arr = np.squeeze(arr_, (0, 2, 4))
        >>> arr.shape
        (2, 3, 4)

        The order of the axes does not matter:

        >>> arr_ = unsqueeze(arr, (0, 4, 2))
        >>> arr_.shape
        (1, 2, 1, 3, 1, 4)

        If `shape` is an int, `axis` must be consistent with it:

        >>> arr_ = unsqueeze(arr, (0, 2, 4), 6)
        >>> arr_.shape
        (1, 2, 1, 3, 1, 4)
        >>> arr_ = unsqueeze(arr, (0, 2, 4), 7)
        Traceback (most recent call last):
          ...
        ValueError: Incompatible `[0, 2, 4]` axis and `7` shape for array of\
 shape (2, 3, 4)

        It is possible to complement the meaning to `axis` to add singletons
        everywhere except where specified (but requires `shape` to be defined
        and the length of `axis` must match the array dims):

        >>> arr_ = unsqueeze(arr, (0, 2, 4), 10, True)
        >>> arr_.shape
        (2, 1, 3, 1, 4, 1, 1, 1, 1, 1)
        >>> arr_ = unsqueeze(arr, (0, 2, 4), complement=True)
        Traceback (most recent call last):
          ...
        ValueError: When `complement` is True, `shape` cannot be None.
        >>> arr_ = unsqueeze(arr, (0, 2), 10, True)
        Traceback (most recent call last):
          ...
        ValueError: When `complement` is True, the length of axis (2) must\
 match the num of dims of array (3).

        Axes values must be valid:

        >>> arr_ = unsqueeze(arr, 0)
        >>> arr_.shape
        (1, 2, 3, 4)
        >>> arr_ = unsqueeze(arr, 3)
        >>> arr_.shape
        (2, 3, 4, 1)
        >>> arr_ = unsqueeze(arr, -1)
        >>> arr_.shape
        (2, 3, 4, 1)
        >>> arr_ = unsqueeze(arr, -4)
        >>> arr_.shape
        (1, 2, 3, 4)
        >>> arr_ = unsqueeze(arr, 10)
        Traceback (most recent call last):
          ...
        ValueError: Axis (10,) out of range.

        If `shape` is specified, `axis` can be omitted (USE WITH CARE!) or its
        value is used for additional safety checks:

        >>> arr_ = unsqueeze(arr, shape=(2, 3, 4, 5, 6))
        >>> arr_.shape
        (2, 3, 4, 1, 1)
        >>> arr_ = unsqueeze(
        ...     arr, (3, 6, 8), (2, 5, 3, 2, 7, 2, 3, 2, 4, 5, 6), True)
        >>> arr_.shape
        (1, 1, 1, 2, 1, 1, 3, 1, 4, 1, 1)
        >>> arr_ = unsqueeze(
        ...     arr, (3, 7, 8), (2, 5, 3, 2, 7, 2, 3, 2, 4, 5, 6), True)
        Traceback (most recent call last):
          ...
        ValueError: New shape [1, 1, 1, 2, 1, 1, 1, 3, 4, 1, 1] cannot be\
 broadcasted to shape (2, 5, 3, 2, 7, 2, 3, 2, 4, 5, 6)
        >>> arr = unsqueeze(arr, shape=(2, 5, 3, 7, 2, 4, 5, 6))
        >>> arr.shape
        (2, 1, 3, 1, 1, 4, 1, 1)
        >>> arr = np.squeeze(arr)
        >>> arr.shape
        (2, 3, 4)
        >>> arr = unsqueeze(arr, shape=(5, 3, 7, 2, 4, 5, 6))
        Traceback (most recent call last):
          ...
        ValueError: Target shape must contain all source shape elements\
 (in correct order). (2, 3, 4) -> (5, 3, 7, 2, 4, 5, 6)

        The behavior is consistent with other NumPy functions and the
        `keepdims` mechanism:

        >>> axis = (0, 2, 4)
        >>> arr1 = np.arange(2 * 3 * 4 * 5 * 6).reshape((2, 3, 4, 5, 6))
        >>> arr2 = np.sum(arr1, axis, keepdims=True)
        >>> arr2.shape
        (1, 3, 1, 5, 1)
        >>> arr3 = np.sum(arr1, axis)
        >>> arr3.shape
        (3, 5)
        >>> arr3 = unsqueeze(arr3, axis)
        >>> arr3.shape
        (1, 3, 1, 5, 1)
        >>> np.all(arr2 == arr3)
        True
    """
    # calculate `new_shape`
    if axis is None and shape is None:
        raise ValueError(
            'At least one of `axis` and `shape` parameters must be specified.')
    elif axis is None and shape is not None:
        new_shape = unsqueezing(arr.shape, shape)
    elif axis is not None:
        if isinstance(axis, int):
            axis = (axis,)
        # calculate the dim of the result
        if shape is not None:
            if isinstance(shape, int):
                ndim = shape
            else:  # shape is a sequence
                ndim = len(shape)
        elif not complement:
            ndim = len(axis) + arr.ndim
        else:
            raise ValueError(
                'When `complement` is True, `shape` cannot be None.')
        # check that axis is properly constructed
        if any(ax < -ndim - 1 or ax > ndim + 1 for ax in axis):
            raise ValueError('Axis {} out of range.'.format(axis))
        # normalize axis using `ndim`
        axis = sorted([ax % ndim for ax in axis])
        # manage complement mode
        if complement:
            if len(axis) == arr.ndim:
                axis = [i for i in range(ndim) if i not in axis]
            else:
                raise ValueError(
                    'When `complement` is True, the length of axis ({})'
                    ' must match the num of dims of array ({}).'.format(
                        len(axis), arr.ndim))
        elif len(axis) + arr.ndim != ndim:
            raise ValueError(
                'Incompatible `{}` axis and `{}` shape'
                ' for array of shape {}'.format(axis, shape, arr.shape))
        # generate the new shape from axis, ndim and shape
        new_shape = []
        i, j = 0, 0
        for m in range(ndim):
            if i < len(axis) and m == axis[i] or j >= arr.ndim:
                new_shape.append(1)
                i += 1
            else:
                new_shape.append(arr.shape[j])
                j += 1

    # check that `new_shape` is consistent with `shape`
    if shape is not None:
        if isinstance(shape, int):
            if len(new_shape) != ndim:
                raise ValueError(
                    'Length of new shape {} does not match '
                    'expected length ({}).'.format(len(new_shape), ndim))
        else:
            if not all(new_dim == 1 or new_dim == dim
                       for new_dim, dim in zip(new_shape, shape)):
                raise ValueError(
                    'New shape {} cannot be broadcasted to shape {}'.format(
                        new_shape, shape))

    return arr.reshape(new_shape)


# ======================================================================
def mdot(arrs):
    """
    Cumulative application of multiple `numpy.dot` operation.

    Args:
        arrs (Iterable[np.ndarray]): The input arrays.

    Returns:
        arr (np.ndarray|None): The result of the dot product.
            None is returned if *_args magic is invalid.

    Examples:
        >>> arrs = [i + np.arange(4).reshape((2, 2)) for i in range(6)]
        >>> mdot((arrs[0], arrs[1], arrs[2]))
        array([[ 22,  29],
               [ 86, 113]])
        >>> mdot(arrs[:3])
        array([[ 22,  29],
               [ 86, 113]])
        >>> mdot(arrs)
        array([[ 32303,  37608],
               [126003, 146696]])
        >>> mdot(arrs[::-1])
        array([[ 51152,  94155],
               [ 69456, 127847]])
        >>> print(mdot([]))
        None

    See Also:
        - flyingcircus.extra.ndot()
        - flyingcircus.extra.mdot_()
    """
    arr = arrs[0] if arrs else None
    for item in arrs[1:]:
        arr = np.dot(arr, item)
    return arr


# ======================================================================
def mdot_(*arrs):
    """
    Star magic version of `flyingcircus.extra.mdot()`.
    """
    return mdot(arrs)


# ======================================================================
def ndot(
        arr,
        axis=-1,
        slicing=slice(None)):
    """
    Cumulative application of `numpy.dot` operation over a given axis.

    Args:
        arr (np.ndarray): The input array.
        axis (int): The axis along which to operate.
        slicing (slice): The slicing along the operating axis.

    Returns:
        prod (np.ndarray): The result of the dot product.
            If `slicing` is empty, an empty array is returned, with length 0
            in the `axis` dimension and the other dimensions are preserved.

    Examples:
        >>> arr = np.array(
        ...     [i + np.arange(4).reshape((2, 2)) for i in range(6)])
        >>> ndot(arr, 0)
        array([[ 32303,  37608],
               [126003, 146696]])
        >>> ndot(arr, 0, slice(None, 3))
        array([[ 22,  29],
               [ 86, 113]])
        >>> ndot(arr, 0, slice(3))
        array([[ 22,  29],
               [ 86, 113]])
        >>> ndot(arr, 0, slice(0, 3))
        array([[ 22,  29],
               [ 86, 113]])
        >>> ndot(arr, 0, slice(None, None, 10))
        array([[0, 1],
               [2, 3]])
        >>> ndot(arr, 0, slice(20))
        array([[ 32303,  37608],
               [126003, 146696]])
        >>> ndot(arr, 0, slice(None, None, -1))
        array([[ 51152,  94155],
               [ 69456, 127847]])
        >>> ndot(arr, 0, slice(0, 1, -1))
        array([], shape=(0, 2, 2), dtype=int64)

    See Also:
        - flyingcircus.extra.mdot()
        - flyingcircus.extra.mdot_()
    """
    assert(-arr.ndim <= axis < arr.ndim)
    axis %= arr.ndim
    mask = tuple(
        slice(None) if j != axis else slicing for j in range(arr.ndim))
    s_dim = arr[mask].shape[axis]
    prod = arr[mask][0] if s_dim else arr[mask]
    for i in range(0, s_dim)[1:]:
        prod = np.dot(prod, arr[mask][i])
    return prod


# ======================================================================
def alternating_array(
        size,
        values=(1, -1),
        dtype=None):
    """
    Compute an alternating array.

    There are a number of alternative equivalent constructs on lists, like:

    .. code-block:: python

        [values[i % len(values)] for i in range(size)]

        [x for x in itertools.islice(itertools.cycle(values), size)]

    This method is particularly efficient for NumPy arrays.

    Args:
        size (int): The size of the resulting array.
        values (Sequence|np.ndarray): The values to alternate.
        dtype (np.dtype|None): The data type of the resulting array.

    Returns:
        result (np.ndarray): The alternating array.

    Examples:
        >>> alternating_array(10)
        array([ 1., -1.,  1., -1.,  1., -1.,  1., -1.,  1., -1.])
        >>> alternating_array(10, dtype=int)
        array([ 1, -1,  1, -1,  1, -1,  1, -1,  1, -1])
        >>> alternating_array(10, (1, 2, 3))
        array([1., 2., 3., 1., 2., 3., 1., 2., 3., 1.])
        >>> values = (1, 2, 3)
        >>> size = 100
        >>> np.all(np.isclose(
        ...     alternating_array(size, values),
        ...     [values[i % len(values)] for i in range(size)]))
        True
        >>> np.all(np.isclose(
        ...     alternating_array(size, values),
        ...     [x for x in itertools.islice(itertools.cycle(values), size)]))
        True

    """
    if dtype is None and isinstance(values, np.ndarray):
        dtype = values.dtype
    result = np.empty(size, dtype=dtype)
    for i, value in enumerate(values):
        result[i::len(values)] = value
    return result


# ======================================================================
def commutator(a, b):
    """
    Calculate the commutator of two arrays: [A,B] = AB - BA

    Args:
        a (np.ndarray): The first operand
        b (np.ndarray): The second operand

    Returns:
        c (np.ndarray): The operation result
    """
    return np.dot(a, b) - np.dot(b, a)


# ======================================================================
def anticommutator(a, b):
    """
    Calculate the anticommutator of two arrays: [A,B] = AB + BA

    Args:
        a (np.ndarray): The first operand
        b (np.ndarray): The second operand

    Returns:
        c (np.ndarray): The operation result
    """
    return np.dot(a, b) + np.dot(b, a)


# ======================================================================
def p_norm(
        arr,
        p=2):
    """
    Compute the p-norm of an array.

    The p-norm is defined as:

    .. math::
        s = \\left(\\sum_i \\left[ |x_i|^p \\right] \\right)^\\frac{1}{p}

    for any :math:`p > 0` (:math:`x_i` being an element of the array).

    When `p == 2`, the p-norm becomes the Euclidean norm.

    Args:
        arr (np.ndarray): The input array.
        p (int|float): The exponent parameter of the norm.

    Returns:
        p_norm (float): The p-norm value.

    Examples:
        >>> p_norm(np.ones(100), 1)
        100.0
        >>> p_norm(np.ones(100), 2)
        10.0
        >>> [round(p_norm(np.ones(100), p), 2) for p in range(1, 12)]
        [100.0, 10.0, 4.64, 3.16, 2.51, 2.15, 1.93, 1.78, 1.67, 1.58, 1.52]
        >>> [round(p_norm(np.ones(1000), p), 2) for p in range(1, 12)]
        [1000.0, 31.62, 10.0, 5.62, 3.98, 3.16, 2.68, 2.37, 2.15, 2.0, 1.87]
        >>> [round(p_norm(np.ones(1000), 1 / p), 2) for p in range(1, 5)]
        [1000.0, 1000000.0, 1000000000.0, 1000000000000.0]
        >>> [round(p_norm(np.arange(10), 1 / p), 2) for p in range(1, 7)]
        [45.0, 372.72, 3248.16, 28740.24, 255959.55, 2287279.92]
    """
    assert (p > 0)
    return np.sum(np.abs(arr) ** p) ** (1 / p)


# ======================================================================
def normalize(
        arr,
        normalization=np.linalg.norm,
        in_place=False):
    """
    Compute the normalized array, i.e. the array divided by its non-zero norm.

    If the norm is zero, the array is left untouched.

    Args:
        arr (np.ndarray): The input array.
        normalization (callable): The normalization function.
            Must have the following signature:
            normalization(np.ndarray) -> float
        in_place (bool): Determine if the function is applied in-place.
            If True, the input gets modified.
            If False, the modification happen on a copy of the input.

    Returns:
        arr (np.ndarray): The normalized array.

    Examples:
        >>> normalize(np.array([3, 4]))
        array([0.6, 0.8])
        >>> normalize(np.array([0, 0, 0, 1, 0]))
        array([0., 0., 0., 1., 0.])
        >>> normalize(np.array([0, 0, 0, 0, 0]))
        array([0, 0, 0, 0, 0])
        >>> normalize(np.ones(4))
        array([0.5, 0.5, 0.5, 0.5])
        >>> normalize(np.ones(8), lambda x: p_norm(x, 3))
        array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        >>> arr = np.array([3.0, 4.0])
        >>> print(arr)
        [3. 4.]
        >>> print(normalize(arr, in_place=True))
        [0.6 0.8]
        >>> print(arr)
        [0.6 0.8]
    """
    norm = normalization(arr)
    if norm:
        if in_place:
            arr /= norm
        else:
            arr = arr / norm
    return arr


# ======================================================================
def vectors2direction(
        vector1,
        vector2,
        normalized=True):
    """
    Compute the vector direction from one vector to one other.

    Args:
        vector1 (Iterable[int|float]): The first vector.
        vector2 (Iterable[int|float]): The second vector.
        normalized (bool): Normalize the result.
            If True, the vector is normalized.

    Returns:
        direction (np.ndarray): The vector direction.

    Examples:
        >>> vectors2direction([3, 2], [6, 6])
        array([0.6, 0.8])
        >>> vectors2direction([1, 2, 3, 4], [5, 6, 7, 8])
        array([0.5, 0.5, 0.5, 0.5])
    """
    direction = np.array(vector2) - np.array(vector1)
    if normalized:
        direction = normalize(direction, in_place=False)
    return direction


# ======================================================================
def pairwise_distances(
        items,
        distance=lambda x, y: p_norm(y - x, 2)):
    """
    Compute the pair-wise distances.

    Assumes that the distance function is symmetric.

    Args:
        items (Iterable): The input items.
        distance (callable): The distance function.
            Must have the following signature: distance(Any, Any) -> Any

    Returns:
        distances (tuple): The computed distances.

    Examples:
        >>> pairwise_distances((1, 2, 3, 4, 5))
        (1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 1.0, 2.0, 1.0)
    """
    return tuple(distance(x, y) for x, y in itertools.combinations(items, 2))


# ======================================================================
def distances2displacements(
        distances,
        origin=0.5,
        offset=0.0):
    """
    Compute the displacements from the distances.

    Args:
        distances (int|float|Iterable[int|float]): The distances.
        origin (int|float): The origin of the displacements.
            This is relative to the total length of the distances.
            If `origin == 0.0` the first displacement is equal to `offset`.
            If `origin == 1.0` the last displacement is equal to `offset`.
        offset (int|float)

    Returns:
        displacements (np.ndarray): The displacements.

    Examples:
        >>> distances2displacements([1, 1, 1, 1, 1, 1])
        array([-3., -2., -1.,  0.,  1.,  2.,  3.])
        >>> distances2displacements([1, 2, 3])
        array([-3., -2.,  0.,  3.])
        >>> distances2displacements([1, 2, 3], origin=0.)
        array([0., 1., 3., 6.])
        >>> distances2displacements([1, 2, 3], origin=1.)
        array([-6., -5., -3.,  0.])
        >>> distances2displacements([1, 2, 3], origin=1., offset=6)
        array([0., 1., 3., 6.])
        >>> distances2displacements(1)
        array([-0.5,  0.5])
    """
    distances = (0,) + tuple(fc.base.auto_repeat(distances, 1))
    return np.cumsum(distances) - np.sum(distances) * origin + offset


# ======================================================================
def square_size_to_num_tria(
        square_size,
        has_diag=False):
    """
    Compute the number of triangular indices from the size of a square matrix.

    Define:
     - :math:`N` the number of triangular indices
     - :math:`n` the linear size of a square matrix

    If the diagonal is excluded:

    :math:`N = \\frac{n (n - 1)}{2}`

    else:

    :math:`N = \\frac{n (n + 1)}{2}`

    Note that is useful also to determine the relationship between
    the number of rotation angles (i.e. `num_tria`) and
    the number of dimensions (i.e. `square_size`).

    Args:
        square_size (int): The linear size of the square matrix.
        has_diag (bool): Assume that the diagonal is included.

    Returns:
        num_tria (int): The number of triangular indices.

    Examples:
        >>> square_size_to_num_tria(3)
        3
        >>> square_size_to_num_tria(4)
        6
        >>> square_size_to_num_tria(5, False)
        10
        >>> square_size_to_num_tria(4, True)
        10
        >>> square_size_to_num_tria(5, True)
        15
        >>> all(square_size_to_num_tria(num_tria_to_square_size(n))
        ...     for n in range(1, 100))
        True
        >>> all(square_size_to_num_tria(num_tria_to_square_size(n, True), True)
        ...     for n in range(1, 100))
        True

    See Also:
        - flyingcircus.extra.num_tria_to_square_size()
    """
    assert (square_size > 0)
    return square_size * (square_size + (1 if has_diag else -1)) // 2


# ======================================================================
def num_tria_to_square_size(
        num_tria,
        has_diag=False,
        raise_err=False):
    """
    Compute the size of a square matrix from the number of triangular indices.

    Define:
     - :math:`N` the number of triangular indices
     - :math:`n` the linear size of a square matrix

    If the diagonal is excluded:

    :math:`N = \\frac{n (n - 1)}{2}`

    else:

    :math:`N = \\frac{n (n + 1)}{2}`

    Either of the equation is solved for `n` (assuming `n > 0`).

    The solution can be written as follows.

    If the diagonal is excluded:

    :math:`n = \\frac{\\sqrt{1 + 8 N} + 1}{2}`

    else:

    :math:`n = \\frac{\\sqrt{1 + 8 N} - 1}{2}`

    Note that is useful also to determine the relationship between
    the number of rotation angles (i.e. `num_tria`) and
    the number of dimensions (i.e. `square_size`).

    Args:
        num_tria (int): The number of triangular indices.
        has_diag (bool): Assume that the diagonal is included.
        raise_err (bool): Raise an exception if invalid number of angles.

    Returns:
        square_size (int): The linear size of the square matrix.

    Raises:
        ValueError: if the number of triangular indices is invalid
            (only if also `raise_err == True`).

    Examples:
        >>> num_tria_to_square_size(3)
        3
        >>> num_tria_to_square_size(6)
        4
        >>> num_tria_to_square_size(10, False)
        5
        >>> num_tria_to_square_size(10, True)
        4
        >>> num_tria_to_square_size(15, True)
        5
        >>> all(square_size_to_num_tria(num_tria_to_square_size(n))
        ...     for n in range(1, 100))
        True
        >>> num_tria_to_square_size(8)
        5
        >>> num_tria_to_square_size(8, raise_err=True)
        Traceback (most recent call last):
            ...
        ValueError: invalid number of triangular indices

    See Also:
        - flyingcircus.extra.square_size_to_num_tria()
    """
    square_size = (((1 + 8 * num_tria) ** 0.5 + (-1 if has_diag else 1)) / 2)
    # alternatives to `divmod()`: numpy.modf(), math.modf()
    int_part, dec_part = divmod(square_size, 1)
    if not np.isclose(dec_part, 0.0) and raise_err:
        raise ValueError('invalid number of triangular indices')
    return int(np.ceil(square_size))


# ======================================================================
def to_self_adjoint_matrix(
        numbers,
        square_size=None,
        has_diag=False,
        skew=False,
        fill=0,
        force=True):
    """
    Compute the self-adjoint matrix from numbers.

    Args:
        numbers (Iterable[int|float]): The input numbers.
        square_size (int): The linear size of the self-adjoint square matrix.
        has_diag (bool): Place the numbers also in the diagonal.
            If True, the numbers as put also in the diagonal.
            Otherwise, the diagonal contains only zeros.
        skew (bool): Compute the skew matrix.
        fill (int|float|None): The filling number.
            This is used fill missing numbers in case the length
        force (bool): Force the matrix to be self-adjoint.
            This is only useful when `had_diag == True`, else it has no effect.

    Returns:
        result (np.ndarray): The self-adjoint square matrix.

    Raises:
        ValueError: if the number of triangular indices is invalid
            (only if also `raise_err == True`).

    Examples:
        >>> to_self_adjoint_matrix([1, 2, 3])
        array([[0, 1, 2],
               [1, 0, 3],
               [2, 3, 0]])
        >>> to_self_adjoint_matrix([1, 2, 3], 4)
        array([[0, 1, 2, 3],
               [1, 0, 0, 0],
               [2, 0, 0, 0],
               [3, 0, 0, 0]])
        >>> to_self_adjoint_matrix([1, 2, 3], has_diag=True)
        array([[1, 2],
               [2, 3]])
        >>> to_self_adjoint_matrix([1, 2, 3], skew=True)
        array([[ 0,  1,  2],
               [-1,  0,  3],
               [-2, -3,  0]])
        >>> to_self_adjoint_matrix([1, 2, 3], 4, fill=9)
        array([[0, 1, 2, 3],
               [1, 0, 9, 9],
               [2, 9, 0, 9],
               [3, 9, 9, 0]])
        >>> to_self_adjoint_matrix([], 3, fill=9)
        array([[0, 9, 9],
               [9, 0, 9],
               [9, 9, 0]])
        >>> to_self_adjoint_matrix([1j, 2j, 3j])
        array([[0.+0.j, 0.+1.j, 0.+2.j],
               [0.-1.j, 0.+0.j, 0.+3.j],
               [0.-2.j, 0.-3.j, 0.+0.j]])
        >>> to_self_adjoint_matrix([1j, 2j, 3j], has_diag=True)
        array([[0.+0.j, 0.+2.j],
               [0.-2.j, 0.+0.j]])
        >>> to_self_adjoint_matrix([1j, 2j, 3j], has_diag=True, force=False)
        array([[0.+1.j, 0.+2.j],
               [0.-2.j, 0.+3.j]])
    """
    numbers = np.array(numbers).ravel()
    if square_size is None:
        square_size = num_tria_to_square_size(numbers.size, has_diag)
    num_tria = square_size_to_num_tria(square_size, has_diag)
    if num_tria != numbers.size:
        if fill is not None:
            old_numbers = numbers
            numbers = np.full(
                num_tria, fill, dtype=numbers.dtype if numbers.size else None)
            numbers[:min(num_tria, old_numbers.size)] = \
                old_numbers[:min(num_tria, old_numbers.size)]
        else:
            raise ValueError('invalid size of the input')
    # rows, cols = np.triu_indices(square_size, 0 if has_diag else 1)
    result = np.zeros((square_size, square_size), dtype=numbers.dtype)
    i = np.arange(square_size)
    mask = (i[:, None] <= i) if has_diag else (i[:, None] < i)
    del i
    result.T[mask] = (-1 * numbers.conj() if skew else numbers.conj())
    result[mask] = numbers
    del mask
    # result[cols, rows] = (-1 * numbers.conj() if skew else numbers.conj())
    # result[rows, cols] = numbers
    if has_diag and force and np.any(np.iscomplex(np.diagonal(result))):
        rows, cols = np.diag_indices_from(result)
        result[rows, cols] = np.real(result[rows, cols])
    return result


# ======================================================================
def valid_interval(interval):
    """
    Sanitize an interval to be in the standard (min, max) format.

    It will also ensure consistent types.

    Args:
        interval (Any): The input interval.
            If Iterable, must have length of 2.
            If int or float, the other interval bound is assumed to be 0.
            If complex, must be purely imaginary, and a symmetric interval is
            generated: `(-x, x)` with `x = abs(imag(interval))` using
            extrema of type float if x is non-negative, and of type int
            if x is negative.
            If range or slice (or any object supporting the `.start` and
            `.stop` properties), `.start` is used as min and `.stop` as max.

    Returns:
        interval (tuple[int|float]): The output interval.
            Has the format: (min, max)

    Examples:
        >>> print(valid_interval((0, 1)))
        (0, 1)
        >>> print(valid_interval((1, 0)))
        (0, 1)
        >>> print(valid_interval((1, 0.0)))
        (0.0, 1.0)
        >>> print(valid_interval((1.0, -1)))
        (-1.0, 1.0)
        >>> print(valid_interval(10))
        (0, 10)
        >>> print(valid_interval(10.0))
        (0.0, 10.0)
        >>> print(valid_interval(-1))
        (-1, 0)
        >>> print(valid_interval(1j))
        (-1.0, 1.0)
        >>> print(valid_interval(-1j))
        (-1, 1)
        >>> print(valid_interval(0j))
        (-0.0, 0.0)
        >>> print(valid_interval(-0j))
        (0, 0)
        >>> print(valid_interval(-5.0))
        (-5.0, 0.0)
        >>> print(valid_interval([5, -6.0]))
        (-6.0, 5.0)
        >>> print(valid_interval([1, 2, 3]))
        Traceback (most recent call last):
            ...
        AssertionError
        >>> print(valid_interval(['x', 2]))
        Traceback (most recent call last):
            ...
        AssertionError
        >>> print(valid_interval(slice(0, 100)))
        (0, 100)
        >>> print(valid_interval(range(0, 100)))
        (0, 100)
        >>> print(valid_interval(slice(0.0, 10.0)))
        (0.0, 10.0)
        >>> print(valid_interval(None))
        Traceback (most recent call last):
            ...
        TypeError: object of type 'NoneType' has no len()
    """
    try:
        interval = interval.start, interval.stop
    except AttributeError:
        pass
    if isinstance(interval, (int, float)):
        if interval < 0:
            interval = (interval, type(interval)(0))
        else:  # if bounds >= 0:
            interval = (type(interval)(0), interval)
    elif isinstance(interval, complex):
        interval = interval.imag
        if math.copysign(1.0, interval) >= 0:
            interval = (-interval, interval)
        else:
            interval = (int(interval), -int(interval))
    else:
        assert (len(interval) == 2)
        assert (all(isinstance(x, (int, float)) for x in interval))
        if interval[0] > interval[1]:
            interval = interval[::-1]
        if any(isinstance(x, float) for x in interval):
            interval = tuple(float(x) for x in interval)
    return tuple(interval)


# ======================================================================
def is_in_range(
        arr,
        interval,
        include_extrema=True):
    """
    Determine if the values of an array are within the specified interval.

    Args:
        arr (np.ndarray): The input array.
        interval (Any): The range of values to check.
            See `flyingcircus.extra.valid_interval()` for details on the
            accepted input.
        include_extrema (bool): Include extrema in the interval checks.

    Returns:
        in_range (bool): The result of the comparison.
            True if all values of the array are within the interval.
            False otherwise.

    Examples:
        >>> arr = np.arange(10)
        >>> is_in_range(arr, [-10, 10])
        True
        >>> is_in_range(arr, [1, 10])
        False
    """
    interval = valid_interval(interval)
    if include_extrema:
        in_range = np.min(arr) >= interval[0] and np.max(arr) <= interval[1]
    else:
        in_range = np.min(arr) > interval[0] and np.max(arr) < interval[1]
    return in_range


# ======================================================================
def scale(
        val,
        out_interval=None,
        in_interval=None):
    """
    Linear convert the value from input interval to output interval

    Args:
        val (float|np.ndarray): Value(s) to convert.
        out_interval (Any): Interval of the output value(s).
            If None, set to: (0, 1).
            See `flyingcircus.extra.valid_interval()` for details on the
            accepted input.
        in_interval (Any): Interval of the input value(s).
            If None, and val is Iterable, it is calculated as:
            (min(val), max(val)), otherwise set to: (0, 1).
            See `flyingcircus.extra.valid_interval()` for details on the
            accepted input.

    Returns:
        val (float|np.ndarray): The converted value(s).

    Examples:
        >>> scale(100, (0, 1000), (0, 100))
        1000.0
        >>> scale(50, (0, 1000), (-100, 100))
        750.0
        >>> scale(50, (0, 10), (0, 1))
        500.0
        >>> scale(0.5, (-10, 10))
        0.0
        >>> scale(np.pi / 3, (0, 180), (0, np.pi))
        60.0
        >>> scale(np.arange(5), (0, 1))
        array([0.  , 0.25, 0.5 , 0.75, 1.  ])
        >>> scale(np.arange(6), (0, 10))
        array([ 0.,  2.,  4.,  6.,  8., 10.])
        >>> scale(np.arange(6), (0, 10), (0, 2))
        array([ 0.,  5., 10., 15., 20., 25.])
    """
    if in_interval:
        in_min, in_max = valid_interval(in_interval)
    elif isinstance(val, np.ndarray):
        in_min, in_max = minmax(val)
    else:
        in_min, in_max = (0, 1)
    if out_interval:
        out_min, out_max = valid_interval(out_interval)
    else:
        out_min, out_max = (0, 1)
    return (val - in_min) / (in_max - in_min) * (out_max - out_min) + out_min


# ======================================================================
def combine_interval(
        interval1,
        interval2=None,
        operation='+'):
    """
    Combine two intervals with some operation to obtain a new interval.

    Args:
        interval1 (Any): Interval of first operand.
            See `flyingcircus.extra.valid_interval()` for details on the
            accepted input.
        interval2 (Any): Interval of second operand.
            See `flyingcircus.extra.valid_interval()` for details on the
            accepted input.
        operation (str): String with operation to perform.
            Supports the following operations:
                - '+' : addition
                - '-' : subtraction

    Returns:
        new_interval (tuple): Interval resulting from operation.

    Examples:
        >>> combine_interval((-1.0, 1.0), (0, 1), '+')
        (-1.0, 2.0)
        >>> combine_interval((-1.0, 1.0), (0, 1), '-')
        (-2.0, 1.0)
    """
    interval1 = valid_interval(interval1)
    interval2 = valid_interval(interval2) \
        if interval2 is not None else interval1
    if operation == '+':
        new_interval = (
            interval1[0] + interval2[0], interval1[1] + interval2[1])
    elif operation == '-':
        new_interval = (
            interval1[0] - interval2[1], interval1[1] - interval2[0])
    else:
        new_interval = (-np.inf, np.inf)
    return new_interval


# ======================================================================
def scaled_randomizer(
        val,
        interval=0.1j,
        fallback_interval=1j):
    """
    Add a random variation to a number proportional to the number itself.

    If the number is 0, uses a random number in the fallback interval.

    Args:
        val (int|float): The value to randomize.
        interval (Iterable[int|float]|Number): The scaled interval.
            Specifies the scaling interval of the random variation as relative
            to the value itself.
            This is passed to `flyingcircus.extra.valid_interval()`.
        fallback_interval (Iterable[int|float]|Number): An interval.
            Specifies the fallback interval of the random variation in
            absolute terms.
            This is only used if the value is 0 or if `interval` is None.
            This is passed to `flyingcircus.extra.valid_interval()`.
    Returns:
        result (int|float): The randomized value.

    Examples:
        >>> random.seed(0)
        >>> print([round(scaled_randomizer(x), 3) for x in range(10)])
        [0.689, 1.052, 1.968, 2.855, 4.009, 4.905, 6.341, 6.725, 7.963, 9.15]
    """
    rand_val = random.random()
    interval = valid_interval(interval)
    fallback_interval = valid_interval(fallback_interval)
    if val:
        return val * (1 + scale(rand_val, interval, (0, 1)))
    else:
        return val + scale(rand_val, fallback_interval, (0, 1))


# =====================================================================
def auto_random(val=(0.0, 1.0)):
    """
    Automatically generate a random value.

    Args:
        val (Any): The value to auto-randomize.
            If int, float, complex, str, bytes, no random value is generated.
            If slice, a random value in the (slice.start, slice.stop) range
            is generated. If both slice.start and slice.stop are int, the
            random number is also an int.
            If iterable, a random element of the iterable is picked up.

    Returns:
        rand_val (Any): The randomized value.

    Examples:
        >>> random.seed(0)
        >>> auto_random(1)
        1
        >>> auto_random(slice(0, 100))
        49
        >>> round(auto_random(slice(0, 10.0)), 4)
        7.5795
        >>> round(auto_random(), 4)
        0.4206
        >>> auto_random(['a', 'b', 'c', 'd'])
        'c'
        >>> round(auto_random(), 4)
        0.9655
        >>> round(auto_random(slice(-100.0, 100)), 4)
        -2.8145
        >>> round(auto_random(slice(-100, 100.0)), 4)
        83.6469
        >>> round(auto_random(slice(100, 100.0)), 4)
        100.0
        >>> auto_random(slice(-100, 100))
        22
        >>> auto_random(None) is None
        True
    """
    if isinstance(val, (int, float, complex, str, bytes)) or val is None:
        return val
    elif isinstance(val, (slice, range)) or (
            len(val) == 2 and fc.base.nesting_level(val) == 1):
        val = valid_interval(val)
        if all(isinstance(x, int) for x in val):
            return random.randint(val[0], val[1])
        else:
            return scale(random.random(), val, (0, 1))
    else:
        # val = list(filter(lambda i: i is not None, val))
        return val[random.randint(0, len(val) - 1)]


# ======================================================================
def midval(arr):
    """
    Calculate the middle value vector.

    Args:
        arr (np.ndarray): The input N-dim array

    Returns:
        arr (np.ndarray): The output (N-1)-dim array

    Examples:
        >>> midval(np.array([0, 1, 2, 3, 4]))
        array([0.5, 1.5, 2.5, 3.5])
    """
    return (arr[1:] - arr[:-1]) / 2.0 + arr[:-1]


# ======================================================================
def sgnlog(
        x,
        base=np.e):
    """
    Signed logarithm of x: log(abs(x)) * sign(x)

    Args:
        x (float|ndarray): The input value(s)
        base (float): The base of the logarithm.

    Returns:
        The signed logarithm

    Examples:
        >>> sgnlog(-100, 10)
        -2.0
        >>> sgnlog(-64, 2)
        -6.0
        >>> np.isclose(sgnlog(100, 2), np.log2(100))
        True
    """
    # log2 is faster than log, which is faster than log10
    return np.log2(np.abs(x)) / np.log2(base) * np.sign(x)


# ======================================================================
def sgngeomspace(
        start,
        stop,
        num=50,
        endpoint=True,
        inner_stop=None,
        inner_start=None):
    """
    Logarithmically spaced samples between signed start and stop endpoints.

    Since the logarithm has a singularity in 0, both `start` and `stop` cannot
    be 0, similarly to `linspace` not accepting infinity as extrema.

    When `start` and `stop` do not have the same sign:
     - the number of points is distributed equally between positive and
       negative values if `num` is even, otherwise the additional point is
       assigned to the largest interval.
     - the absolute value of `start` and `stop` must be greater than 1.
     - the smallest absolute values before changing sign is determined by
       the absolute values of the extrema, so that to a large extremum it
       corresponds a smaller value before changing sign. This is calculated
       by inverting the logarithm of the extremum, e.g. if `start` is:
       100, the minimum value before changing sign is: 1 / 100.

    Args:
        start (float): The starting value of the sequence.
            Cannot be 0. If start and stop have different signs, must be
            larger than 1 in absolute value.
        stop (float): The end value of the sequence.
            Cannot be 0. If start and stop have different signs, must be
            larger than 1 in absolute value.
        num (int): Number of samples to generate. Must be non-negative.
        endpoint (bool): The value of 'stop' is the last sample.
        inner_stop (float|callable|None):
        inner_start (float|callable|None):

    Returns:
        samples (ndarray): equally spaced samples on a log scale.

    Examples:
        >>> sgngeomspace(-10, 10, 3)
        array([-10. ,   0.1,  10. ])
        >>> sgngeomspace(-100, -1, 3)
        array([-100.,  -10.,   -1.])
        >>> sgngeomspace(-10, 10, 6)
        array([-10. ,  -1. ,  -0.1,   0.1,   1. ,  10. ])
        >>> sgngeomspace(-10, 10, 5)
        array([-10. ,  -0.1,   0.1,   1. ,  10. ])
        >>> sgngeomspace(2, 10, 4)
        array([ 2.        ,  3.41995189,  5.84803548, 10.        ])
        >>> sgngeomspace(-2, 10, 4)
        array([-2. , -0.5,  0.1, 10. ])
        >>> sgngeomspace(-10, 2, 6)
        array([-10. ,  -1. ,  -0.1,   0.5,   1. ,   2. ])
        >>> sgngeomspace(10, -2, 5)
        array([10. ,  1. ,  0.1, -0.5, -2. ])
        >>> sgngeomspace(10, -1, 5)
        Traceback (most recent call last):
            ...
        AssertionError
    """
    if not fc.base.is_same_sign((start, stop)):
        assert (abs(start) > 1 and abs(stop) > 1)
        bounds = ((start, 1 / start), (1 / stop, stop))
        equity = 1 if num % 2 == 1 and abs(start) > abs(stop) else 0
        nums = (num // 2 + equity, num - num // 2 - equity)
        endpoints = True, endpoint
        logspaces = tuple(
            np.geomspace(*bound, num=n, endpoint=endpoint)
            for bound, n, endpoint in zip(bounds, nums, endpoints))
        samples = np.concatenate(logspaces)
    else:
        samples = np.geomspace(start, stop, num=num, endpoint=endpoint)
    return samples


# ======================================================================
def minmax(arr):
    """
    Calculate the minimum and maximum of an array: (min, max).

    Args:
        arr (np.ndarray): The input array.

    Returns:
        min (float): the minimum value of the array
        max (float): the maximum value of the array

    Examples:
        >>> minmax(np.arange(10))
        (0, 9)
    """
    return np.min(arr), np.max(arr)


# ======================================================================
def freq2afreq(val):
    """
    Convert frequency to angular frequency (not changing time units).

    Args:
        val (float): The input value.

    Returns:
        val (float): The output value.
    """
    return (2.0 * np.pi) * val


# ======================================================================
def afreq2freq(val):
    """
    Convert angular frequency to frequency (not changing time units).

    Args:
        val (float): The input value.

    Returns:
        val (float): The output value.
    """
    return val / (2.0 * np.pi)


# ======================================================================
def subst(
        arr,
        pairs=((np.inf, 0.0), (-np.inf, 0.0), (np.nan, 0.0))):
    """
    Substitute all occurrences of a value in an array.

    Useful to replace (mask out) specific unsafe values, e.g. singularities.

    Args:
        arr (np.ndarray): The input array.
        pairs (Iterable): The substitution rules.
            Each rule consist of a value to replace and its replacement.
            Each rule is applied sequentially in the order they appear and
            modify the content of the array immediately.

    Returns:
        arr (np.ndarray): The output array.

    Examples:
        >>> a = np.arange(10)
        >>> subst(a, ((1, 100), (7, 700)))
        array([  0, 100,   2,   3,   4,   5,   6, 700,   8,   9])
        >>> a = np.tile(np.arange(4), 3)
        >>> subst(a, ((1, 100), (7, 700)))
        array([  0, 100,   2,   3,   0, 100,   2,   3,   0, 100,   2,   3])
        >>> a = np.tile(np.arange(4), 3)
        >>> subst(a, ((1, 100), (3, 300)))
        array([  0, 100,   2, 300,   0, 100,   2, 300,   0, 100,   2, 300])
        >>> a = np.array([0.0, 1.0, np.inf, -np.inf, np.nan, -np.nan])
        >>> subst(a)
        array([0., 1., 0., 0., 0., 0.])
        >>> a = np.array([0.0, 1.0, np.inf, 2.0, np.nan])
        >>> subst(a, ((np.inf, 0.0), (0.0, np.inf), (np.nan, 0.0)))
        array([inf,  1., inf,  2.,  0.])
        >>> subst(a, ((np.inf, 0.0), (np.nan, 0.0), (0.0, np.inf)))
        array([inf,  1., inf,  2., inf])
    """
    for k, v in pairs:
        if k is np.nan:
            arr[np.isnan(arr)] = v
        else:
            arr[arr == k] = v
    return arr


# ======================================================================
def ravel_clean(
        arr,
        removes=(np.nan, np.inf, -np.inf)):
    """
    Ravel and remove values to an array.

    Args:
        arr (np.ndarray): The input array.
        removes (Iterable): Values to remove.
            If empty, no values will be removed.

    Returns:
        arr (np.ndarray): The output array.

    Examples:
        >>> a = np.array([0.0, 1.0, np.inf, -np.inf, np.nan, -np.nan])
        >>> ravel_clean(a)
        array([0., 1.])

    See Also:
        - flyingcircus.base.subst()
    """
    arr = arr.ravel()
    for val in removes:
        if val is np.nan:
            arr = arr[~np.isnan(arr)]
        if len(arr) > 0:
            arr = arr[arr != val]
    return arr


# ======================================================================
def dftn(
        arr,
        axes=None):
    """
    Discrete Fourier Transform.

    Interface to fftn combined with appropriate shifts.

    Args:
        arr (np.ndarray): Input n-dim array.
        axes (Iterable[int]|None): The axes along which to operate.
            If None, operate on all axes.

    Returns:
        arr (np.ndarray): Output n-dim array.

    Examples:
        >>> a = np.arange(2)
        >>> dftn(a)
        array([1.+0.j, 1.+0.j])
        >>> print(np.allclose(a, dftn(idftn(a))))
        True

    See Also:
        numpy.fft, scipy.fftpack
    """
    return fftshift(fftn(ifftshift(arr, axes=axes), axes=axes), axes=axes)


# ======================================================================
def idftn(
        arr,
        axes=None):
    """
    Inverse Discrete Fourier transform.

    Interface to ifftn combined with appropriate shifts.

    Args:
        arr (np.ndarray): Input n-dim array.
        axes (Iterable[int]|None): The axes along which to operate.
            If None, operate on all axes.

    Returns:
        arr (np.ndarray): Output n-dim array.

    Examples:
        >>> a = np.arange(2)
        >>> idftn(a)
        array([0.5+0.j, 0.5+0.j])
        >>> print(np.allclose(a, idftn(dftn(a))))
        True

    See Also:
        numpy.fft, scipy.fftpack
    """
    return fftshift(ifftn(ifftshift(arr, axes=axes), axes=axes), axes=axes)


# ======================================================================
def ogrid2mgrid(ogrid):
    """
    Convert a sparse grid to a dense grid.

    A sparse grid is obtained from `np.ogrid[]`, while a
    dense grid is obtained from `np.mgrid[]`.

    Args:
        ogrid (Iterable[np.ndarray]): The sparse grid.
            This should be equivalent to the result of `np.ogrid[]`.
            Specifically, each array has the same number of dims, and has
            singlets in all but one dimension.

    Returns:
        mgrid (np.ndarray): The dense grid.
            This should be equivalent to the result of `np.mgrid[]`.
            Specifically, the first dim has size equal to the total number
            of dims minus one.

    Examples:
        >>> shape = (2, 3, 4)
        >>> grid = tuple(slice(0, dim) for dim in shape)
        >>> ogrid = np.ogrid[grid]
        >>> print(ogrid)
        [array([[[0]],
        <BLANKLINE>
               [[1]]]), array([[[0],
                [1],
                [2]]]), array([[[0, 1, 2, 3]]])]
        >>> mgrid = np.mgrid[grid]
        >>> print(mgrid)
        [[[[0 0 0 0]
           [0 0 0 0]
           [0 0 0 0]]
        <BLANKLINE>
          [[1 1 1 1]
           [1 1 1 1]
           [1 1 1 1]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[0 0 0 0]
           [1 1 1 1]
           [2 2 2 2]]
        <BLANKLINE>
          [[0 0 0 0]
           [1 1 1 1]
           [2 2 2 2]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[0 1 2 3]
           [0 1 2 3]
           [0 1 2 3]]
        <BLANKLINE>
          [[0 1 2 3]
           [0 1 2 3]
           [0 1 2 3]]]]
        >>> print(ogrid2mgrid(ogrid))
        [[[[0 0 0 0]
           [0 0 0 0]
           [0 0 0 0]]
        <BLANKLINE>
          [[1 1 1 1]
           [1 1 1 1]
           [1 1 1 1]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[0 0 0 0]
           [1 1 1 1]
           [2 2 2 2]]
        <BLANKLINE>
          [[0 0 0 0]
           [1 1 1 1]
           [2 2 2 2]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[0 1 2 3]
           [0 1 2 3]
           [0 1 2 3]]
        <BLANKLINE>
          [[0 1 2 3]
           [0 1 2 3]
           [0 1 2 3]]]]
        >>> np.all(np.mgrid[grid] == ogrid2mgrid(ogrid))
        True
    """
    mgrid = np.zeros(
        (len(ogrid),) + tuple(max(d for d in x.shape) for x in ogrid),
        dtype=ogrid[0].dtype)
    for i, x in enumerate(ogrid):
        mgrid[i, ...] = x
    return mgrid


# ======================================================================
def mgrid2ogrid(mgrid):
    """
    Convert a dense grid to a sparse grid.

    A sparse grid is obtained from `np.ogrid[]`, while a
    dense grid is obtained from `np.mgrid[]`.

    Args:
        mgrid (np.ndarray): The dense grid.
            This should be equivalent to the result of `np.mgrid[]`.
            Specifically, the first dim has size equal to the total number
            of dims minus one.

    Returns:
        ogrid (list[np.ndarray]): The sparse grid.
            This should be equivalent to the result of `np.ogrid[]`.
            Specifically, each array has the same number of dims, and has
            singlets in all but one dimension.

    Examples:
        >>> shape = (2, 3, 4)
        >>> grid = tuple(slice(0, dim) for dim in shape)
        >>> ogrid = np.ogrid[grid]
        >>> print(ogrid)
        [array([[[0]],
        <BLANKLINE>
               [[1]]]), array([[[0],
                [1],
                [2]]]), array([[[0, 1, 2, 3]]])]
        >>> mgrid = np.mgrid[grid]
        >>> print(mgrid)
        [[[[0 0 0 0]
           [0 0 0 0]
           [0 0 0 0]]
        <BLANKLINE>
          [[1 1 1 1]
           [1 1 1 1]
           [1 1 1 1]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[0 0 0 0]
           [1 1 1 1]
           [2 2 2 2]]
        <BLANKLINE>
          [[0 0 0 0]
           [1 1 1 1]
           [2 2 2 2]]]
        <BLANKLINE>
        <BLANKLINE>
         [[[0 1 2 3]
           [0 1 2 3]
           [0 1 2 3]]
        <BLANKLINE>
          [[0 1 2 3]
           [0 1 2 3]
           [0 1 2 3]]]]
        >>> print(mgrid2ogrid(mgrid))
        (array([[[0]],
        <BLANKLINE>
               [[1]]]), array([[[0],
                [1],
                [2]]]), array([[[0, 1, 2, 3]]]))
        >>> all(np.all(x == y)
        ...     for x, y in zip(np.ogrid[grid], mgrid2ogrid(np.mgrid[grid])))
        True
    """
    ogrid = tuple(
        mgrid[i][
            tuple(
                slice(None) if j == i else slice(0, 1)
                for j, d in enumerate(mgrid.shape[1:]))]
        for i in range(mgrid.shape[0]))
    return ogrid


# ======================================================================
def coord(
        shape,
        position=0.5,
        is_relative=True,
        use_int=True):
    """
    Calculate the coordinate in a given shape for a specified position.

    Args:
        shape (int|Iterable[int]): The shape of the container in px.
        position (float|Iterable[float]): The position of the coordinate.
            Values are relative to the lowest edge.
            The values interpretation depend on `is_relative`.
        is_relative (bool|callable): Interpret position as relative.
            If False, `position` is interpreted as absolute (in px).
            Otherwise, they are interpreted as relative, i.e. each value of
            `position` is referenced to some other value.
            If True, values of `shape` are used as the reference.
            If callable, the reference is computed from `shape` which is
            passed as the first (and only) parameter using the function.
            The signature must be: is_relative(Iterable) -> int|float|Iterable.
            Internally, uses `flyingcircus.extra.scale()`.
        use_int (bool): Force integer values for the coordinates.

    Returns:
        xx (list): The coordinates of the position.

    Examples:
        >>> coord((5, 5))
        (2, 2)
        >>> coord((4, 4))
        (2, 2)
        >>> coord((5, 5), 3, False)
        (3, 3)
    """
    xx = fc.base.auto_repeat(position, len(shape), check=True)
    if is_relative:
        refs = fc.base.auto_repeat(
            is_relative(shape), len(shape), check=True) \
            if callable(is_relative) else shape
        if use_int:
            xx = tuple(
                int(scale(x, (0, d))) for x, d in zip(xx, refs))
        else:
            xx = tuple(
                scale(x, (0, d - 1)) for x, d in zip(xx, refs))
    elif any(not isinstance(x, int) for x in xx) and use_int:
        text = 'The value of `position` must be integer ' \
               'if `is_relative == False` and `use_int == True`.'
        raise TypeError(text)
    return xx


# ======================================================================
def grid_coord(
        shape,
        position=0.5,
        is_relative=True,
        use_int=True,
        dense=False):
    """
    Calculate the generic x_i coordinates for N-dim operations.

    Args:
        shape (int|Iterable[int]): The shape of the container in px.
        position (float|Iterable[float]): Relative position of the origin.
            Values are in the [0, 1] interval.
        is_relative (bool|callable): Interpret origin as relative.
            See `flyingcircus.extra.coord()` for more info.
        dense (bool): Determine the type (dense or sparse) of the mesh-grid.
        use_int (bool): Force integer values for the coordinates.

    Returns:
        xx (list[np.ndarray]|np.ndarray): Sparse or dense mesh-grid.
            If dense is True, the result is dense (like `np.mgrid[]`).
            Specifically, the result is an `np.ndarray` where the first dim
            has size equal to the total number of dims minus one.
            Otherwise, the result is sparse (like `np.ogrid[]`).
            Specifically, the result is a list of `np.ndarray` each array
            has the same number of dims, and has singlets in all but one
            dimension.


    Returns:
        ogrid (tuple[np.ndarray]): The sparse grid.
            This should be equivalent to the result of `np.ogrid[]`.


    Examples:
        >>> grid_coord((4, 4))
        [array([[-2],
               [-1],
               [ 0],
               [ 1]]), array([[-2, -1,  0,  1]])]
        >>> grid_coord((5, 5))
        [array([[-2],
               [-1],
               [ 0],
               [ 1],
               [ 2]]), array([[-2, -1,  0,  1,  2]])]
        >>> grid_coord((2, 2))
        [array([[-1],
               [ 0]]), array([[-1,  0]])]
        >>> grid_coord((2, 2), dense=True)
        array([[[-1, -1],
                [ 0,  0]],
        <BLANKLINE>
               [[-1,  0],
                [-1,  0]]])
        >>> grid_coord((2, 3), position=(0.0, 0.5))
        [array([[0],
               [1]]), array([[-1,  0,  1]])]
        >>> grid_coord((3, 9), position=(1, 4), is_relative=False)
        [array([[-1],
               [ 0],
               [ 1]]), array([[-4, -3, -2, -1,  0,  1,  2,  3,  4]])]
        >>> grid_coord((3, 9), position=0.2, is_relative=True)
        [array([[0],
               [1],
               [2]]), array([[-1,  0,  1,  2,  3,  4,  5,  6,  7]])]
        >>> grid_coord((4, 4), use_int=False)
        [array([[-1.5],
               [-0.5],
               [ 0.5],
               [ 1.5]]), array([[-1.5, -0.5,  0.5,  1.5]])]
        >>> grid_coord((5, 5), use_int=False)
        [array([[-2.],
               [-1.],
               [ 0.],
               [ 1.],
               [ 2.]]), array([[-2., -1.,  0.,  1.,  2.]])]
        >>> grid_coord((2, 3), position=(0.0, 0.0), use_int=False)
        [array([[0.],
               [1.]]), array([[0., 1., 2.]])]
    """
    xx0 = coord(shape, position, is_relative, use_int)
    grid = tuple(slice(-x0, dim - x0) for x0, dim in zip(xx0, shape))
    return np.ogrid[grid] if not dense else np.mgrid[grid]


# ======================================================================
def grid_transform(
        xx,
        lin_mat,
        off_vec=None,
        is_dense=None):
    """
    Apply a linear or affine transform to a mesh-grid.

    The affine transform is implemented as a linear transformation followed by
    a translation (of a given shift).

    Args:
        xx (Iterable[np.ndarray]|np.ndarray): Sparse or dense mesh-grid.
            If dense is True, the result is dense (like `np.mgrid[]`).
            Specifically, the result is an `np.ndarray` where the first dim
            has size equal to the total number of dims minus one.
            Otherwise, the result is sparse (like `np.ogrid[]`).
            Specifically, the result is a list of `np.ndarray` each array
            has the same number of dims, and has singlets in all but one
            dimension.
        lin_mat (Sequence|np.ndarray): The linear transformation matrix.
            This must be a `n` by `n` matrix where `n` is the number of dims.
        off_vec (Sequence|np.ndarray): The offset vector in px.
            This must be a `n` sized array.
        is_dense (bool|None): The type (dense or sparse) of the mesh-grid.
            If bool, this is explicitly specified.
            If None, this is inferred from `xx`.

    Returns:
        xx (np.ndarray): The transformed mesh-grid (dense).
            Specifically, the result is an `np.ndarray` where the first dim
            has size equal to the total number of dims minus one.

    Examples:
        >>> shape = (2, 3)
        >>> grid = tuple(slice(0, dim) for dim in shape)
        >>> xx = np.ogrid[grid]
        >>> lin_mat = [[0, 2], [2, 1]]
        >>> off_vec = [10, 20]
        >>> xx
        [array([[0],
               [1]]), array([[0, 1, 2]])]
        >>> print(ogrid2mgrid(xx))
        [[[0 0 0]
          [1 1 1]]
        <BLANKLINE>
         [[0 1 2]
          [0 1 2]]]
        >>> grid_transform(xx, lin_mat)
        array([[[0, 2, 4],
                [0, 2, 4]],
        <BLANKLINE>
               [[0, 1, 2],
                [2, 3, 4]]])
        >>> grid_transform(xx, lin_mat, off_vec)
        array([[[10, 12, 14],
                [10, 12, 14]],
        <BLANKLINE>
               [[20, 21, 22],
                [22, 23, 24]]])
    """
    n_dim = xx[0].ndim
    max_dims = len(string.ascii_letters) - 2
    if n_dim > max_dims:
        text = 'Maximum number ({}) of dims exceeded.'.format(max_dims)
        raise ValueError(text)
    if is_dense is None:
        is_dense = isinstance(xx, np.ndarray)
    lin_mat = np.array(lin_mat)
    assert (lin_mat.shape == (n_dim, n_dim))
    xx = np.einsum(
        '{i}{j}, {i}{indexes} -> {j}{indexes}'.format(
            i=string.ascii_letters[-2], j=string.ascii_letters[-1],
            indexes=string.ascii_letters[:n_dim]),
        lin_mat, xx if is_dense else ogrid2mgrid(xx))
    if off_vec is not None:
        off_vec = np.array(off_vec)
        assert (off_vec.size == xx.shape[0])
        xx += off_vec.reshape((-1,) + (1,) * len(xx.shape[1:]))
    return xx


# ======================================================================
def rel2abs(shape, size=0.5):
    """
    Calculate the absolute size from a relative size for a given shape.

    This is a simple version of `flyingcircus.extra.scale()` and/or
    `flyingcircus.extra.coord()`.

    Args:
        shape (int|Iterable[int]): The shape of the container in px.
        size (float|Iterable[float]): Relative position (to the lowest edge).
            Each element of the tuple should be in the range [0, 1].

    Returns:
        position (float|tuple[float]): Absolute position inside the shape.
            Each element of the tuple should be in the range [0, dim - 1],
            where dim is the corresponding dimension of the shape.

    Examples:
        >>> rel2abs((100, 100, 101, 101), (0.0, 1.0, 0.0, 1.0))
        (0.0, 99.0, 0.0, 100.0)
        >>> rel2abs((100, 99, 101))
        (49.5, 49.0, 50.0)
        >>> rel2abs((100, 200, 50, 99, 37), (0.0, 1.0, 0.2, 0.3, 0.4))
        (0.0, 199.0, 9.8, 29.4, 14.4)
        >>> rel2abs((100, 100, 100), (1.0, 10.0, -1.0))
        (99.0, 990.0, -99.0)
        >>> shape = (100, 100, 100, 100, 100)
        >>> abs2rel(shape, rel2abs(shape, (0.0, 0.25, 0.5, 0.75, 1.0)))
        (0.0, 0.25, 0.5, 0.75, 1.0)

    See Also:
        - flyingcircus.extra.abs2rel()
        - flyingcircus.extra.coord()
        - flyingcircus.extra.scale()
    """
    size = fc.base.auto_repeat(size, len(shape), check=True)
    return tuple((s - 1.0) * p for p, s in zip(size, shape))


# ======================================================================
def abs2rel(shape, position=0):
    """
    Calculate the relative size from an absolute size for a given shape.

    This is a simple version of `flyingcircus.extra.scale()`.

    Args:
        shape (int|Iterable[int]): The shape of the container in px.
        position (float|Iterable[float]): Absolute position inside the shape.
            Each element of the tuple should be in the range [0, dim - 1],
            where dim is the corresponding dimension of the shape.

    Returns:
        position (float|tuple[float]): Relative position (to the lowest edge).
            Each element of the tuple should be in the range [0, 1].

    Examples:
        >>> abs2rel((100, 100, 101, 99), (0, 100, 100, 100))
        (0.0, 1.0101010101010102, 1.0, 1.0204081632653061)
        >>> abs2rel((100, 99, 101))
        (0.0, 0.0, 0.0)
        >>> abs2rel((412, 200, 37), (30, 33, 11.7))
        (0.072992700729927, 0.1658291457286432, 0.32499999999999996)
        >>> abs2rel((100, 100, 100), (250, 10, -30))
        (2.525252525252525, 0.10101010101010101, -0.30303030303030304)
        >>> shape = (100, 100, 100, 100, 100)
        >>> abs2rel(shape, rel2abs(shape, (0, 25, 50, 75, 100)))
        (0.0, 25.0, 50.0, 75.0, 100.0)

    See Also:
        - flyingcircus.extra.rel2abs()
        - flyingcircus.extra.coord()
        - flyingcircus.extra.scale()
    """
    position = fc.base.auto_repeat(position, len(shape), check=True)
    return tuple(p / (s - 1.0) for p, s in zip(position, shape))


# ======================================================================
def laplace_kernel(
        shape,
        factors=1):
    """
    Calculate the kernel to be used for the Laplacian operators.

    This is substantially `k^2`.

    This is in the Fourier domain.
    May require shifting and normalization before using in
    Discrete Fourier Transform (DFT).

    Args:
        shape (Iterable[int]): The size of the array.
        factors (int|float|Iterable[int|float]): The size conversion factors.
            If int or float, the same conversion factor is applied to all dims.
            Otherwise, the Iterable length must match the length of shape.

    Returns:
        kk2 (np.ndarray): The resulting kernel array.

    Examples:
        >>> laplace_kernel((3, 3, 3))
        array([[[3., 2., 3.],
                [2., 1., 2.],
                [3., 2., 3.]],
        <BLANKLINE>
               [[2., 1., 2.],
                [1., 0., 1.],
                [2., 1., 2.]],
        <BLANKLINE>
               [[3., 2., 3.],
                [2., 1., 2.],
                [3., 2., 3.]]])
        >>> laplace_kernel((3, 3, 3), np.sqrt(3))
        array([[[1.        , 0.66666667, 1.        ],
                [0.66666667, 0.33333333, 0.66666667],
                [1.        , 0.66666667, 1.        ]],
        <BLANKLINE>
               [[0.66666667, 0.33333333, 0.66666667],
                [0.33333333, 0.        , 0.33333333],
                [0.66666667, 0.33333333, 0.66666667]],
        <BLANKLINE>
               [[1.        , 0.66666667, 1.        ],
                [0.66666667, 0.33333333, 0.66666667],
                [1.        , 0.66666667, 1.        ]]])
        >>> laplace_kernel((2, 2, 2), 0.6)
        array([[[8.33333333, 5.55555556],
                [5.55555556, 2.77777778]],
        <BLANKLINE>
               [[5.55555556, 2.77777778],
                [2.77777778, 0.        ]]])
    """
    kk = grid_coord(shape)
    if factors and factors != 1:
        factors = fc.base.auto_repeat(factors, len(shape), check=True)
        kk = [k_i / factor for k_i, factor in zip(kk, factors)]
    kk2 = np.zeros(shape)
    for k_i, dim in zip(kk, shape):
        kk2 += k_i ** 2
    return kk2


# ======================================================================
def gradient_kernels(
        shape,
        dims=None,
        factors=1):
    """
    Calculate the kernel to be used for the gradient operators.

    This is substantially: k

    This is in the Fourier domain.
    May require shifting and normalization before using in
    Discrete Fourier Transform (DFT).

    Args:
        shape (Iterable[int]): The size of the array.
        dims (int|Iterable[int]): The direction of the gradient.
            Values must be between `len(shape)` and `-len(shape)`.
        factors (int|float|Iterable[int|float]): The size conversion factors.
            If int or float, the same conversion factor is applied to all dims.
            Otherwise, the Iterable length must match the length of shape.

    Returns:
        kks (tuple(np.ndarray)): The resulting kernel arrays.

    Examples:
        >>> gradient_kernels((2, 2))
        (array([[-1, -1],
               [ 0,  0]]), array([[-1,  0],
               [-1,  0]]))
        >>> gradient_kernels((2, 2, 2))
        (array([[[-1, -1],
                [-1, -1]],
        <BLANKLINE>
               [[ 0,  0],
                [ 0,  0]]]), array([[[-1, -1],
                [ 0,  0]],
        <BLANKLINE>
               [[-1, -1],
                [ 0,  0]]]), array([[[-1,  0],
                [-1,  0]],
        <BLANKLINE>
               [[-1,  0],
                [-1,  0]]]))
        >>> gradient_kernels((2, 2, 2), (1, 2))
        (array([[[-1, -1],
                [ 0,  0]],
        <BLANKLINE>
               [[-1, -1],
                [ 0,  0]]]), array([[[-1,  0],
                [-1,  0]],
        <BLANKLINE>
               [[-1,  0],
                [-1,  0]]]))
        >>> gradient_kernels((2, 2, 2), -1)
        (array([[[-1,  0],
                [-1,  0]],
        <BLANKLINE>
               [[-1,  0],
                [-1,  0]]]),)
        >>> gradient_kernels((2, 2), None, 3)
        (array([[-0.33333333, -0.33333333],
               [ 0.        ,  0.        ]]), array([[-0.33333333,  0.        ],
               [-0.33333333,  0.        ]]))
    """
    kk = grid_coord(shape)
    if factors and factors != 1:
        factors = fc.base.auto_repeat(factors, len(shape), check=True)
        kk = [k_i / factor for k_i, factor in zip(kk, factors)]
    if dims is None:
        dims = range(len(shape))
    else:
        if isinstance(dims, int):
            dims = (dims,)
        dims = tuple(dim % len(shape) for dim in dims)
    kks = tuple(
        np.broadcast_to(k_i, shape)
        for i, (k_i, dim) in enumerate(zip(kk, shape))
        if i in dims)
    return kks


# ======================================================================
def exp_gradient_kernels(
        shape,
        dims=None,
        factors=1):
    """
    Calculate the kernel to be used for the exponential gradient operators.

    This is substantially: :math:`1 - \\exp(2\\pi\\i k)`

    This is in the Fourier domain.
    May require shifting and normalization before using in
    Discrete Fourier Transform (DFT).

    Args:
        shape (Iterable[int]): The size of the array.
        dims (int|Iterable[int]|None): The direction of the gradient.
            Values must be between `len(shape)` and `-len(shape)`.
            If None, all dimensions are computed.
        factors (int|float|Iterable[int|float]): The size conversion factors.
            If int or float, the same conversion factor is applied to all dims.
            Otherwise, the Iterable length must match the length of shape.

    Returns:
        kks (tuple(np.ndarray)): The resulting kernel arrays.

    Examples:
        >>> exp_gradient_kernels((2, 2))
        (array([[0.-2.4492936e-16j, 0.-2.4492936e-16j],
               [0.+0.0000000e+00j, 0.+0.0000000e+00j]]),\
 array([[0.-2.4492936e-16j, 0.+0.0000000e+00j],
               [0.-2.4492936e-16j, 0.+0.0000000e+00j]]))
        >>> exp_gradient_kernels((2, 2, 2))
        (array([[[0.-2.4492936e-16j, 0.-2.4492936e-16j],
                [0.-2.4492936e-16j, 0.-2.4492936e-16j]],
        <BLANKLINE>
               [[0.+0.0000000e+00j, 0.+0.0000000e+00j],
                [0.+0.0000000e+00j, 0.+0.0000000e+00j]]]),\
 array([[[0.-2.4492936e-16j, 0.-2.4492936e-16j],
                [0.+0.0000000e+00j, 0.+0.0000000e+00j]],
        <BLANKLINE>
               [[0.-2.4492936e-16j, 0.-2.4492936e-16j],
                [0.+0.0000000e+00j, 0.+0.0000000e+00j]]]),\
 array([[[0.-2.4492936e-16j, 0.+0.0000000e+00j],
                [0.-2.4492936e-16j, 0.+0.0000000e+00j]],
        <BLANKLINE>
               [[0.-2.4492936e-16j, 0.+0.0000000e+00j],
                [0.-2.4492936e-16j, 0.+0.0000000e+00j]]]))
        >>> exp_gradient_kernels((2, 2, 2), (1, 2))
        (array([[[0.-2.4492936e-16j, 0.-2.4492936e-16j],
                [0.+0.0000000e+00j, 0.+0.0000000e+00j]],
        <BLANKLINE>
               [[0.-2.4492936e-16j, 0.-2.4492936e-16j],
                [0.+0.0000000e+00j, 0.+0.0000000e+00j]]]),\
 array([[[0.-2.4492936e-16j, 0.+0.0000000e+00j],
                [0.-2.4492936e-16j, 0.+0.0000000e+00j]],
        <BLANKLINE>
               [[0.-2.4492936e-16j, 0.+0.0000000e+00j],
                [0.-2.4492936e-16j, 0.+0.0000000e+00j]]]))
        >>> exp_gradient_kernels((2, 2, 2), -1)
        (array([[[0.-2.4492936e-16j, 0.+0.0000000e+00j],
                [0.-2.4492936e-16j, 0.+0.0000000e+00j]],
        <BLANKLINE>
               [[0.-2.4492936e-16j, 0.+0.0000000e+00j],
                [0.-2.4492936e-16j, 0.+0.0000000e+00j]]]),)
        >>> exp_gradient_kernels((2, 2), None, 3)
        (array([[1.5+0.8660254j, 1.5+0.8660254j],
               [0. +0.j       , 0. +0.j       ]]),\
 array([[1.5+0.8660254j, 0. +0.j       ],
               [1.5+0.8660254j, 0. +0.j       ]]))
    """
    kk = grid_coord(shape)
    if factors and factors != 1:
        factors = fc.base.auto_repeat(factors, len(shape), check=True)
        kk = [k_i / factor for k_i, factor in zip(kk, factors)]
    if dims is None:
        dims = range(len(shape))
    else:
        if isinstance(dims, int):
            dims = (dims,)
        dims = tuple(dim % len(shape) for dim in dims)
    kks = tuple(
        np.broadcast_to((1.0 - np.exp(2j * np.pi * k_i)), shape)
        for i, (k_i, dim) in enumerate(zip(kk, shape))
        if i in dims)
    return kks


# ======================================================================
def width_from_shapes(
        shape,
        new_shape,
        position,
        rounding=round):
    """
    Generate width values for padding a shape onto a new shape.

    Args:
        shape (Iterable[int]): The input shape.
        new_shape (int|Iterable[int]): The output shape.
            If int, uses the same value for all dimensions.
            If Iterable, must have the same length as `shape`.
            Additionally, each value of `new_shape` must be greater than or
            equal to the corresponding value of `shape`.
        position (int|float|Iterable[int|float]): Position within new shape.
            Determines the position of the array within the new shape.
            If int or float, it is considered the same in all dimensions,
            otherwise its length must match the length of shape.
            If int or Iterable of int, the values are absolute and must be
            less than or equal to the difference between the shape and
            the new shape.
            If float or Iterable of float, the values are relative to the
            difference between `new_shape` and `shape` and must bein the
            [0, 1] range.
        rounding (callable): The rounding method for the position.
            This determines the rounding to use when computing the
            *before* and *after* width values from a non-integer position
            value.
            The expected signature is:
            rounding(int|float) -> int|float

    Returns:
        width (tuple[tuple[int]]): Size of the padding to use.

    Examples:
        >>> width_from_shapes((2, 3), (4, 5), 0.5)
        ((1, 1), (1, 1))
        >>> width_from_shapes((2, 3), (4, 5), 0)
        ((0, 2), (0, 2))
        >>> width_from_shapes((2, 3), (4, 5), (2, 0))
        ((2, 0), (0, 2))
        >>> width_from_shapes((2, 3), (4, 5), (2, 0))
        ((2, 0), (0, 2))
    """
    new_shape = fc.base.auto_repeat(new_shape, len(shape), check=True)
    position = fc.base.auto_repeat(position, len(shape), check=True)
    if any(dim > new_dim for dim, new_dim in zip(shape, new_shape)):
        raise ValueError('new shape cannot be smaller than the old one.')
    position = tuple(
        (int(rounding((new_dim - dim) * offset))
         if isinstance(offset, float) else offset)
        for dim, new_dim, offset in zip(shape, new_shape, position))
    if any(dim + offset > new_dim
           for dim, new_dim, offset in zip(shape, new_shape, position)):
        raise ValueError(
            'Incompatible `shape`, `new_shape` and `position`.')
    width = tuple(
        (offset, new_dim - dim - offset)
        for dim, new_dim, offset in zip(shape, new_shape, position))
    return width


# ======================================================================
def const_padding(
        arr,
        width=0,
        values=0):
    """
    Pad an array using a constant value.

    This is equivalent to `np.pad(mode='constant')`, but should be faster.
    Also `width` and `values` parameters are interpreted in a more general way.

    Args:
        arr (np.ndarray): The input array.
        width (int|float|Iterable[int|float]): Size of the padding to use.
            This is used with `flyingcircus.base.multi_scale_to_int()`.
            The shape of the array is used for the scales.
        values (Any|Iterable[Any]): The constant value(s) to use for padding.
            If not Iterable, the value is used for all width sizes.
            If Iterable of non-iterable, the iterable must be of length equal
            to `arr.ndim` or 1 (in which case the inner iterable is repeated
            according to `arr.ndim`), while each value is repeated twice.
            If Iterable of Iterable, the inner iterables must be of length 2,
            while the outer iterable must be of length equal to `arr.ndim` or 1
            (in which case the inner iterable is repeated according to
            `arr.ndim`).
            In general, `values` should have shape `(arr.ndim, 2)` either
            directly or after appling `flyingcircus.base.strech()`.
            For each dimensions, the first value is used for the width size
            before the original input and the second value is used for the
            width size after the original input.
            Each dimension is applied incrementally from smaller to larger.

    Returns:
        result (np.ndarray): The padded array.

    Examples:
        >>> arr = arange_nd((2, 3)) + 1
        >>> print(arr)
        [[1 2 3]
         [4 5 6]]
        >>> new_arr = const_padding(arr, 2)
        >>> print(new_arr)
        [[0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0]
         [0 0 1 2 3 0 0]
         [0 0 4 5 6 0 0]
         [0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0]]
        >>> new_arr = const_padding(arr, (1,))
        >>> print(new_arr)
        [[0 0 0 0 0]
         [0 1 2 3 0]
         [0 4 5 6 0]
         [0 0 0 0 0]]
        >>> new_arr = const_padding(arr, (1, 3))
        >>> print(new_arr)
        [[0 0 0 0 0 0 0 0 0]
         [0 0 0 1 2 3 0 0 0]
         [0 0 0 4 5 6 0 0 0]
         [0 0 0 0 0 0 0 0 0]]
        >>> new_arr = const_padding(arr, ((1, 2), (3, 4)))
        >>> print(new_arr)
        [[0 0 0 0 0 0 0 0 0 0]
         [0 0 0 1 2 3 0 0 0 0]
         [0 0 0 4 5 6 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0 0]]
        >>> new_arr = const_padding(arr, ((1, 2), 3))
        >>> print(new_arr)
        [[0 0 0 0 0 0 0 0 0]
         [0 0 0 1 2 3 0 0 0]
         [0 0 0 4 5 6 0 0 0]
         [0 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 0 0]]
        >>> new_arr = const_padding(arr, ((1, 2), 1.0), -1)
        >>> print(new_arr)
        [[-1 -1 -1 -1 -1 -1 -1 -1 -1]
         [-1 -1 -1  1  2  3 -1 -1 -1]
         [-1 -1 -1  4  5  6 -1 -1 -1]
         [-1 -1 -1 -1 -1 -1 -1 -1 -1]
         [-1 -1 -1 -1 -1 -1 -1 -1 -1]]
        >>> new_arr = const_padding(arr, (0.5, 2.0), 9)
        >>> print(new_arr)
        [[9 9 9 9 9 9 9 9 9 9 9 9 9 9 9]
         [9 9 9 9 9 9 1 2 3 9 9 9 9 9 9]
         [9 9 9 9 9 9 4 5 6 9 9 9 9 9 9]
         [9 9 9 9 9 9 9 9 9 9 9 9 9 9 9]]
        >>> arr = arange_nd((5, 7, 11)) + 1
        >>> np.all(const_padding(arr, 17) == np.pad(arr, 17, 'constant'))
        True

        >>> arr = arange_nd((2, 3)) + 1
        >>> print(arr)
        [[1 2 3]
         [4 5 6]]
        >>> new_arr = const_padding(arr, 2, ((1, 2), (3, 4)))
        >>> print(new_arr)
        [[3 3 1 1 1 4 4]
         [3 3 1 1 1 4 4]
         [3 3 1 2 3 4 4]
         [3 3 4 5 6 4 4]
         [3 3 2 2 2 4 4]
         [3 3 2 2 2 4 4]]
        >>> new_arr = const_padding(arr, 2, (1, 2))
        >>> print(new_arr)
        [[2 2 1 1 1 2 2]
         [2 2 1 1 1 2 2]
         [2 2 1 2 3 2 2]
         [2 2 4 5 6 2 2]
         [2 2 1 1 1 2 2]
         [2 2 1 1 1 2 2]]
        >>> new_arr = const_padding(arr, 2, ((1, 2),))
        >>> print(new_arr)
        [[1 1 1 1 1 2 2]
         [1 1 1 1 1 2 2]
         [1 1 1 2 3 2 2]
         [1 1 4 5 6 2 2]
         [1 1 2 2 2 2 2]
         [1 1 2 2 2 2 2]]
        >>> arr = arange_nd((5, 7, 11)) + 1
        >>> vals = ((1, 2), (3, 4), (5, 6))
        >>> np.all(
        ...     const_padding(arr, 17, vals)
        ...     == np.pad(arr, 17, 'constant', constant_values=vals))
        True
    """
    width = fc.base.multi_scale_to_int(width, arr.shape)
    if callable(values):
        values = values(arr)
    if any(any(size for size in sizes) for sizes in width):
        shape = tuple(
            low + dim + up for dim, (low, up) in zip(arr.shape, width))
        if not fc.base.is_deep(values):
            result = np.full(shape, values, dtype=arr.dtype)
        else:
            values = fc.base.stretch(values, (arr.ndim, 2))
            result = np.zeros(shape, dtype=arr.dtype)
            slices = tuple(
                (slice(0, low), slice(dim - up, dim))
                for dim, (low, up) in zip(shape, width))
            for i, (slices_, values_) in enumerate(zip(slices, values)):
                for slice_, value in zip(slices_, values_):
                    slicing = tuple(
                        slice(None) if i != j else slice_ for j, dim in
                        enumerate(shape))
                    result[slicing] = value
        inner = tuple(
            slice(low, low + dim)
            for dim, (low, up) in zip(arr.shape, width))
        result[inner] = arr
    else:
        result = arr
    return result


# ======================================================================
def edge_padding(
        arr,
        width=0):
    """
    Pad an array using edge values.

    This is equivalent to `np.pad(mode='edge')`, but should be faster.
    Also, the `width` parameter is interpreted in a more general way.

    Args:
        arr (np.ndarray): The input array.
        width (int|float|Iterable[int|float]): Size of the padding to use.
            This is used with `flyingcircus.base.multi_scale_to_int()`.
            The shape of the array is used for the scales.

    Returns:
        result (np.ndarray): The padded array.

    Examples:
        >>> arr = arange_nd((2, 3)) + 1
        >>> print(arr)
        [[1 2 3]
         [4 5 6]]
        >>> new_arr = edge_padding(arr, 1)
        >>> print(new_arr)
        [[1 1 2 3 3]
         [1 1 2 3 3]
         [4 4 5 6 6]
         [4 4 5 6 6]]
        >>> new_arr = edge_padding(arr, 2)
        >>> print(new_arr)
        [[1 1 1 2 3 3 3]
         [1 1 1 2 3 3 3]
         [1 1 1 2 3 3 3]
         [4 4 4 5 6 6 6]
         [4 4 4 5 6 6 6]
         [4 4 4 5 6 6 6]]
        >>> new_arr = edge_padding(arr, (2, 3))
        >>> print(new_arr)
        [[1 1 1 1 2 3 3 3 3]
         [1 1 1 1 2 3 3 3 3]
         [1 1 1 1 2 3 3 3 3]
         [4 4 4 4 5 6 6 6 6]
         [4 4 4 4 5 6 6 6 6]
         [4 4 4 4 5 6 6 6 6]]
        >>> new_arr = edge_padding(arr, ((2, 3),))
        >>> print(new_arr)
        [[1 1 1 2 3 3 3 3]
         [1 1 1 2 3 3 3 3]
         [1 1 1 2 3 3 3 3]
         [4 4 4 5 6 6 6 6]
         [4 4 4 5 6 6 6 6]
         [4 4 4 5 6 6 6 6]
         [4 4 4 5 6 6 6 6]]
        >>> new_arr = edge_padding(arr, ((1, 2), (3, 4)))
        >>> print(new_arr)
        [[1 1 1 1 2 3 3 3 3 3]
         [1 1 1 1 2 3 3 3 3 3]
         [4 4 4 4 5 6 6 6 6 6]
         [4 4 4 4 5 6 6 6 6 6]
         [4 4 4 4 5 6 6 6 6 6]]
        >>> arr = arange_nd((5, 7, 11)) + 1
        >>> np.all(edge_padding(arr, 17) == np.pad(arr, 17, 'edge'))
        True
    """
    width = fc.base.multi_scale_to_int(width, arr.shape)
    if any(any(size for size in sizes) for sizes in width):
        shape = tuple(
            low + dim + up for dim, (low, up) in zip(arr.shape, width))
        result = np.zeros(shape, dtype=arr.dtype)
        target_slices = tuple(
            (slice(0, low), slice(low, dim - up), slice(dim - up, dim))
            for dim, (low, up) in zip(shape, width))
        source_slices = \
            ((slice(0, 1), slice(None), slice(-1, None)),) * len(shape)
        for target_slicing, source_slicing in zip(
                itertools.product(*target_slices),
                itertools.product(*source_slices)):
            result[target_slicing] = arr[source_slicing]
    else:
        result = arr
    return result


# ======================================================================
def cyclic_padding(
        arr,
        width):
    """
    Pad an array using cyclic values.

    This is equivalent to `np.pad(mode='wrap')`, but should be faster.
    Also, the `width` parameter is interpreted in a more general way.

    It uses a combination of `np.tile()` and `np.roll()` to achieve
    optimal performances.

    Args:
        arr (np.ndarray): The input array.
        width (int|float|Iterable[int|float]): Size of the padding to use.
            This is used with `flyingcircus.base.multi_scale_to_int()`.
            The shape of the array is used for the scales.

    Returns:
        result (np.ndarray): The padded array.

    Examples:
        >>> arr = arange_nd((2, 3)) + 1
        >>> print(arr)
        [[1 2 3]
         [4 5 6]]
        >>> new_arr = cyclic_padding(arr, (1, 2))
        >>> print(new_arr)
        [[5 6 4 5 6 4 5]
         [2 3 1 2 3 1 2]
         [5 6 4 5 6 4 5]
         [2 3 1 2 3 1 2]]
        >>> new_arr = cyclic_padding(arr, ((0, 1), 2))
        >>> print(new_arr)
        [[2 3 1 2 3 1 2]
         [5 6 4 5 6 4 5]
         [2 3 1 2 3 1 2]]
        >>> new_arr = cyclic_padding(arr, ((1, 0), 2))
        >>> print(new_arr)
        [[5 6 4 5 6 4 5]
         [2 3 1 2 3 1 2]
         [5 6 4 5 6 4 5]]
        >>> new_arr = cyclic_padding(arr, ((0, 1.0),))
        >>> print(new_arr)
        [[1 2 3 1 2 3]
         [4 5 6 4 5 6]
         [1 2 3 1 2 3]
         [4 5 6 4 5 6]]
        >>> arr = arange_nd((5, 7, 11)) + 1
        >>> np.all(cyclic_padding(arr, 17) == np.pad(arr, 17, 'wrap'))
        True
    """
    width = fc.base.multi_scale_to_int(width, arr.shape)
    if any(any(size for size in sizes) for sizes in width):
        offsets = tuple(
            (low + dim) % dim for dim, (low, up) in zip(arr.shape, width))
        tiling = tuple(
            (low + dim + up) // dim + (1 if (low + dim + up) % dim else 0)
            for dim, (low, up) in zip(arr.shape, width))
        slicing = tuple(
            slice(0, low + dim + up)
            for dim, (low, up) in zip(arr.shape, width))
        if any(offset != 0 for offset in offsets):
            nonzero_offsets_axes, nonzero_offsets = tuple(zip(*(
                (axis, offset) for axis, offset in enumerate(offsets)
                if offset != 0)))
            arr = np.roll(arr, nonzero_offsets, nonzero_offsets_axes)
        result = np.tile(arr, tiling)[slicing]
    else:
        result = arr
    return result


# ======================================================================
def symmetric_padding(
        arr,
        width):
    """
    Pad an array using symmetric values.

    This is equivalent to `np.pad(mode='symmetric')`, but should be faster.
    Also, the `width` parameter is interpreted in a more general way.

    Args:
        arr (np.ndarray): The input array.
        width (int|float|Iterable[int|float]): Size of the padding to use.
            This is used with `flyingcircus.base.multi_scale_to_int()`.
            The shape of the array is used for the scales.

    Returns:
        result (np.ndarray): The padded array.

    Examples:
        >>> arr = arange_nd((2, 3)) + 1
        >>> print(arr)
        [[1 2 3]
         [4 5 6]]
        >>> new_arr = symmetric_padding(arr, (1, 2))
        >>> print(new_arr)
        [[2 1 1 2 3 3 2]
         [2 1 1 2 3 3 2]
         [5 4 4 5 6 6 5]
         [5 4 4 5 6 6 5]]
        >>> new_arr = symmetric_padding(arr, ((0, 1), 2))
        >>> print(new_arr)
        [[2 1 1 2 3 3 2]
         [5 4 4 5 6 6 5]
         [5 4 4 5 6 6 5]]
        >>> new_arr = symmetric_padding(arr, ((1, 0), 2))
        >>> print(new_arr)
        [[2 1 1 2 3 3 2]
         [2 1 1 2 3 3 2]
         [5 4 4 5 6 6 5]]
        >>> new_arr = symmetric_padding(arr, ((0, 1.0),))
        >>> print(new_arr)
        [[1 2 3 3 2 1]
         [4 5 6 6 5 4]
         [4 5 6 6 5 4]
         [1 2 3 3 2 1]]
        >>> arr = arange_nd((5, 7, 11)) + 1
        >>> np.all(symmetric_padding(arr, 17) == np.pad(arr, 17, 'symmetric'))
        True
    """
    width = fc.base.multi_scale_to_int(width, arr.shape)
    if any(any(size for size in sizes) for sizes in width):
        shape = tuple(
            low + dim + up for dim, (low, up) in zip(arr.shape, width))
        result = np.zeros(shape, dtype=arr.dtype)
        target_slices = tuple(
            tuple(
                slice(
                    max((i - (1 if low % dim else 0)) * dim + low % dim, 0),
                    min((i + 1 - (1 if low % dim else 0)) * dim + low % dim,
                        low + dim + up))
                for i in range(
                    fc.base.div_ceil(low, dim) + fc.base.div_ceil(up,
                                                                  dim) + 1))
            for dim, (low, up) in zip(arr.shape, width))
        len_target_slices = tuple(len(items) for items in target_slices)
        parities = tuple(
            fc.base.div_ceil(low, dim) % 2
            for dim, (low, up) in zip(arr.shape, width))
        for i, target_slicing in enumerate(itertools.product(*target_slices)):
            ij = np.unravel_index(i, len_target_slices)
            source_slicing = []
            for idx, target_slice, parity, dim in \
                    zip(ij, target_slicing, parities, arr.shape):
                step = 1 if idx % 2 == parity else -1
                start = stop = None
                span = target_slice.stop - target_slice.start
                if span != dim:
                    if target_slice.start == 0:
                        start = \
                            (dim - span) if idx % 2 == parity else (span - 1)
                    else:
                        stop = \
                            span if idx % 2 == parity else (dim - span - 1)
                source_slicing.append(slice(start, stop, step))
            source_slicing = tuple(source_slicing)
            result[target_slicing] = arr[source_slicing]
    else:
        result = arr
    return result


# ======================================================================
def padding(
        arr,
        width=0,
        combine=None,
        mode=0,
        **_kws):
    """
    Pad an array using different padding strategies.

    The default behavior is zero-padding. Note that this function internally
    calls a specialized padding function from this package, if possible.
    Otherwise, it defaults to the `np.pad()`.
    Note that `np.pad()` is typically less performing than the specialized
    counterparts available in this package.

    Args:
        arr (np.ndarray): The input array.
        width (int|float|Iterable[int|float]): Size of the padding to use.
            This is used with `flyingcircus.base.multi_scale_to_int()`.
            The shape of the array is used for the scales.
            Note that this is more sophisticated than the interpretation from
            `np.pad()`.
             In particular the following mappings hold:
             - `padding(width=((a, b),))` == `np.pad(width=(a, b)`
             - `padding(width=(a, b))` == `np.pad(width=((a, a), (b, b))`
             - `padding(width=(a, (b, c)))` == `np.pad(width=((a, a), (b, c))`
             By contrast, e.g. `np.pad(width=(a, (b, c))` is not supported.
        combine (callable|None): The function for combining pad width values.
            Passed as `combine` to `flyingcircus.base.multi_scale_to_int()`.
        mode (Number|Iterable[Number|Iterable]|str): The padding mode.
            If int, float or complex, `mode` is set to `constant` and this is
            interpreted as the constant value to use.
            If Iterable of int, float or complex, `mode` is set to `edge` and
            the number of items must match the number of dims.
            If Iterable of Iterables, `mode` is set to `edge` and the number
            of outer iterables must match the number of dims, while the inner
            iterables can be any number but only the first and last items are
            used as the lower and upper edges for a given dimension.
            If str, this is passed directly to `np.pad()`, unless a
            specialized function from this module is available.
            See `np.pad()` for more details.
        **_kws: Keyword parameters for `np.pad()`.

    Returns:
        result (tuple): The tuple
            contains:
             - arr (np.ndarray): The padded array.
             - mask (tuple(slice)): The mask delimiting the input array.

    See Also:
        - flyingcircus.base.multi_scale_to_int()
        - flyingcircus.extra.const_padding()
        - flyingcircus.extra.edge_padding()
        - flyingcircus.extra.cyclic_padding()
        - flyingcircus.extra.symmetric_padding()
        - flyingcircus.extra.reframe()

    Examples:
        >>> arr = arange_nd((2, 3))
        >>> new_arr, mask = padding(arr, 1)
        >>> print(new_arr)
        [[0 0 0 0 0]
         [0 0 1 2 0]
         [0 3 4 5 0]
         [0 0 0 0 0]]
        >>> print(mask)
        (slice(1, -1, None), slice(1, -1, None))
    """
    if width or fc.base.is_deep(width) and any(fc.base.flatten(width)):
        width = fc.base.multi_scale_to_int(width, arr.shape, combine=combine)
        mask = tuple(slice(lower, -upper) for (lower, upper) in width)
        if isinstance(mode, (int, float, complex)):
            _kws['constant_values'] = mode
            mode = 'constant'
        if mode == 'constant':
            result = const_padding(arr, width, values=_kws['constant_values'])
        elif mode == 'edge':
            result = edge_padding(arr, width)
        elif mode in ('cyclic', 'wrap'):
            result = cyclic_padding(arr, width)
        elif mode == 'symmetric':
            result = symmetric_padding(arr, width)
        else:
            result = np.pad(arr, width, mode, **_kws)
    else:
        mask = (slice(None),) * arr.ndim
        result = arr
    return result, mask


# ======================================================================
def gradients(
        arr,
        dims=None,
        ft_factor=(2 * np.pi),
        pad_width=0.0):
    """
    Apply the gradient operator (in the Fourier domain).

    A more accurate gradient operator is provided by `np.gradient()`.

    Args:
        arr (np.ndarray): The input array.
        dims (int|Iterable[int]): The direction of the gradient.
            Values must be between `len(shape)` and `-len(shape)`.
        ft_factor (float): The Fourier factor for the gradient operator.
            Should be either 1 or 2π, depending on DFT implementation.
        pad_width (float|int): Size of the border to use.
            This is useful for mitigating border effects.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to the maximum size.

    Returns:
        arrs (np.ndarray): The output array.

    See Also:
        - flyingcircus.extra.gradient_kernels()
    """
    arr, mask = padding(arr, pad_width)
    arrs = tuple(
        (((-1j * ft_factor) ** 2) * ifftn(fftshift(kk) * fftn(arr)))[mask]
        for kk in gradient_kernels(arr.shape, dims, arr.shape))
    return arrs


# ======================================================================
def exp_gradients(
        arr,
        dims=None,
        ft_factor=(2 * np.pi),
        pad_width=0):
    """
    Apply the exponential gradient operator (in the Fourier domain).

    Args:
        arr (np.ndarray): The input array.
        dims (int|Iterable[int]|None): The direction of the gradient.
            Values must be between `len(shape)` and `-len(shape)`.
            If None, all dimensions are computed.
        ft_factor (float): The Fourier factor for the gradient operator.
            Should be either 1 or 2π, depending on DFT implementation.
        pad_width (float|int): Size of the border to use.
            This is useful for mitigating border effects.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to the maximum size.

    Returns:
        arrs (np.ndarray): The output array.

    See Also:
        - flyingcircus.extra.exp_gradient_kernels()
    """
    arr, mask = padding(arr, pad_width)
    arrs = tuple(
        (((-1j * ft_factor) ** 2) * ifftn(fftshift(kk) * fftn(arr)))[mask]
        for kk in exp_gradient_kernels(arr.shape, dims, arr.shape))
    return arrs


# ======================================================================
def laplacian(
        arr,
        ft_factor=(2 * np.pi),
        pad_width=0):
    """
    Apply the Laplacian operator (in the Fourier domain).

    Args:
        arr (np.ndarray): The input array.
        ft_factor (float): The Fourier factor for the gradient operator.
            Should be either 1 or 2π, depending on DFT implementation.
        pad_width (float|int|Iterable[float|int]): Size of the padding to use.
            This is useful for mitigating border effects.
            If Iterable, a value for each dim must be specified.
            If not Iterable, all dims will have the same value.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to the maximum size.

    Returns:
        arr (np.ndarray): The output array.
    """
    arr, mask = padding(arr, pad_width)
    kk2 = fftshift(laplace_kernel(arr.shape, arr.shape))
    arr = ((1j * ft_factor) ** 2) * ifftn(kk2 * fftn(arr))
    return arr[mask]


# ======================================================================
def inv_laplacian(
        arr,
        ft_factor=(2 * np.pi),
        pad_width=0):
    """
    Apply the inverse Laplacian operator (in the Fourier domain).

    Args:
        arr (np.ndarray): The input array.
        ft_factor (float): The Fourier factor for the gradient operator.
            Should be either 1 or 2π, depending on DFT implementation.
        pad_width (float|int): Size of the border to use.
            This is useful for mitigating border effects.
            If int, it is interpreted as absolute size.
            If float, it is interpreted as relative to the maximum size.

    Returns:
        arr (np.ndarray): The output array.
    """
    arr, mask = padding(arr, pad_width)
    kk2 = fftshift(laplace_kernel(arr.shape, arr.shape))
    kk2[kk2 != 0] = 1.0 / kk2[kk2 != 0]
    arr = ((-1j / ft_factor) ** 2) * ifftn(kk2 * fftn(arr))
    return arr[mask]


# ======================================================================
def auto_bin(
        arr,
        method='auto',
        dim=1):
    """
    Determine the optimal number of bins for histogram of an array.

    Args:
        arr (np.ndarray): The input array.
        method (str|None): The estimation method.
            Accepted values (with: N the array size, D the histogram dim):
             - 'auto': max('fd', 'sturges')
             - 'sqrt': Square-root choice (fast, independent of `dim`)
               n = sqrt(N)
             - 'sturges': Sturges' formula (tends to underestimate)
               n = 1 + log_2(N)
             - 'rice': Rice Rule (fast with `dim` dependence)
               n = 2 * N^(1/(2 + D))
             - 'riced': Modified Rice Rule (fast with strong `dim` dependence)
               n = (1 + D) * N^(1/(2 + D))
             - 'scott': Scott's normal reference rule (depends on data)
               n = N^(1/(2 + D)) *  / (3.5 * SD(arr)
             - 'fd': Freedman–Diaconis' choice (robust variant of 'scott')
               n = N^(1/(2 + D)) * range(arr) / 2 * (Q75 - Q25)
             - 'doane': Doane's formula (correction to Sturges'):
               n = 1 + log_2(N) + log_2(1 + |g1| / sigma_g1)
               where g1 = (|mean|/sigma) ** 3 is the skewness
               and sigma_g1 = sqrt(6 * (N - 2) / ((N + 1) * (N + 3))) is the
               estimated standard deviation on the skewness.
             - None: n = N
        dim (int): The dimension of the histogram.

    Returns:
        num (int): The number of bins.

    Examples:
        >>> arr = np.arange(100)
        >>> auto_bin(arr)
        8
        >>> auto_bin(arr, 'sqrt')
        10
        >>> auto_bin(arr, 'auto')
        8
        >>> auto_bin(arr, 'sturges')
        8
        >>> auto_bin(arr, 'rice')
        10
        >>> auto_bin(arr, 'riced')
        14
        >>> auto_bin(arr, 'scott')
        5
        >>> auto_bin(arr, 'fd')
        5
        >>> auto_bin(arr, None)
        100
        >>> auto_bin(arr, 'sqrt', 2)
        10
        >>> auto_bin(arr, 'auto', 2)
        8
        >>> auto_bin(arr, 'sturges', 2)
        8
        >>> auto_bin(arr, 'rice', 2)
        7
        >>> auto_bin(arr, 'riced', 2)
        13
        >>> auto_bin(arr, 'scott', 2)
        4
        >>> auto_bin(arr, 'fd', 2)
        4
        >>> auto_bin(arr, None, 2)
        100
        >>> np.random.seed(0)
        >>> arr = np.random.random(100) * 1000
        >>> arr /= np.sum(arr)
        >>> auto_bin(arr, 'scott')
        5
        >>> auto_bin(arr, 'fd')
        5
        >>> auto_bin(arr, 'scott', 2)
        4
        >>> auto_bin(arr, 'fd', 2)
        4

    References:
         - https://en.wikipedia.org/wiki/Histogram#Number_of_bins_and_width
    """
    if method == 'auto':
        num = max(auto_bin(arr, 'fd', dim), auto_bin(arr, 'sturges', dim))
    elif method == 'sqrt':
        num = int(np.ceil(arr.size ** 0.5))
    elif method == 'sturges':
        num = int(np.ceil(1 + np.log2(arr.size)))
    elif method == 'rice':
        num = int(np.ceil(2 * arr.size ** (1 / (2 + dim))))
    elif method == 'riced':
        num = int(np.ceil((2 + dim) * arr.size ** (1 / (2 + dim))))
    elif method == 'scott':
        h = 3.5 * np.std(arr) / arr.size ** (1 / (2 + dim))
        num = int(np.ceil(np.ptp(arr) / h))
    elif method == 'fd':
        q75, q25 = np.percentile(arr, [75, 25])
        h = 2 * (q75 - q25) / arr.size ** (1 / (2 + dim))
        num = int(np.ceil(np.ptp(arr) / h))
    elif method == 'doane':
        g1 = (np.abs(np.mean(arr)) / np.std(arr)) ** 3
        sigma_g1 = np.sqrt(
            6 * (arr.size - 2) / ((arr.size + 1) * (arr.size + 3)))
        num = int(np.ceil(
            1 + np.log2(arr.size) + np.log2(1 + np.abs(g1) / sigma_g1)))
    else:
        num = arr.size
    return num


# ======================================================================
def auto_bins(
        arrs,
        method='rice',
        dim=None,
        combine=max):
    """
    Determine the optimal number of bins for a histogram of a group of arrays.

    Args:
        arrs (Iterable[np.ndarray]): The input arrays.
        method (str|Iterable[str]|None): The method for calculating bins.
            If str, the same method is applied to both arrays.
            See `flyingcircus.extra.auto_bin()` for available methods.
        dim (int|None): The dimension of the histogram.
        combine (callable|None): Combine each bin using the combine function.
            combine(n_bins) -> n_bin
            n_bins is of type Iterable[int]

    Returns:
        n_bins (int|tuple[int]): The number of bins.
            If combine is None, returns a tuple of int (one for each input
            array).

    Examples:
        >>> arr1 = np.arange(100)
        >>> arr2 = np.arange(200)
        >>> arr3 = np.arange(300)
        >>> auto_bins((arr1, arr2))
        8
        >>> auto_bins((arr1, arr2, arr3))
        7
        >>> auto_bins((arr1, arr2), ('sqrt', 'sturges'))
        10
        >>> auto_bins((arr1, arr2), combine=None)
        (7, 8)
        >>> auto_bins((arr1, arr2), combine=min)
        7
        >>> auto_bins((arr1, arr2), combine=sum)
        15
        >>> auto_bins((arr1, arr2), combine=lambda x: abs(x[0] - x[1]))
        1
    """
    if isinstance(method, str) or method is None:
        method = (method,) * len(arrs)
    if not dim:
        dim = len(arrs)
    n_bins = []
    for arr, method in zip(arrs, method):
        n_bins.append(auto_bin(arr, method, dim))
    if combine:
        return combine(n_bins)
    else:
        return tuple(n_bins)


# ======================================================================
def entropy(
        hist,
        base=np.e):
    """
    Calculate the simple or joint Shannon entropy H.

    H = -sum(p(x) * log(p(x)))

    p(x) is the probability of x, where x can be N-Dim.

    Args:
        hist (np.ndarray): The probability density function p(x).
            If hist is 1-dim, the Shannon entropy is computed.
            If hist is N-dim, the joint Shannon entropy is computed.
            Zeros are handled correctly.
            The probability density function does not need to be normalized.
        base (int|float): The base units to express the result.
            Should be a number larger than 0.
            If base is 2, the unit is `bits`.
            If base is np.e (Euler's number), the unit is `nats`.

    Returns:
        h (float): The Shannon entropy H = -sum(p(x) * log(p(x)))

    Examples:
        >>>
    """
    # normalize histogram to unity
    hist = hist / np.sum(hist)
    log_hist = apply_at(hist, lambda x: np.log(x) / np.log(base), hist > 0, 0)
    h = -np.sum(hist * log_hist)
    return h


# ======================================================================
def conditional_entropy(
        hist2,
        hist,
        base=np.e):
    """
    Calculate the conditional probability: H(X|Y)

    Args:
        hist2 (np.ndarray): The joint probability density function.
            Must be the 2D histrogram of X and Y
        hist (np.ndarray): The given probability density function.
            Must be the 1D histogram of Y.
        base (int|float): The base units to express the result.
            Should be a number larger than 0.
            If base is 2, the unit is `bits`.
            If base is np.e (Euler's number), the unit is `nats`.

    Returns:
        hc (float): The conditional entropy H(X|Y)

    Examples:
        >>>
    """
    return entropy(hist2, base) - entropy(hist, base)


# ======================================================================
def variation_information(
        arr1,
        arr2,
        base=np.e,
        bins='rice'):
    """
    Calculate the variation of information between two arrays.

    Args:
        arr1 (np.ndarray): The first input array.
            Must have same shape as arr2.
        arr2 (np.ndarray): The second input array.
            Must have same shape as arr1.
        base (int|float): The base units to express the result.
            Should be a number larger than 0.
            If base is 2, the unit is `bits`.
            If base is np.e (Euler's number), the unit is `nats`.
        bins (int|str|None): The number of bins to use for the distribution.
            If int, the exact number is used.
            If str, a method accepted by `auto_bins()` is expected.
            If None, uses the `auto_bins()` default value.
    Returns:
        vi (float): The variation of information.

    Examples:
        >>> np.random.seed(0)
        >>> arr1 = np.zeros(100)
        >>> arr2 = np.arange(100)
        >>> arr3 = np.random.rand(100)
        >>> arr4 = arr3 + np.random.rand(100) / 100
        >>> variation_information(arr1, arr1)
        0.0
        >>> variation_information(arr2, arr2)
        0.0
        >>> variation_information(arr3, arr3)
        0.0
        >>> vi_12 = variation_information(arr1, arr2)
        >>> vi_21 = variation_information(arr2, arr1)
        >>> vi_31 = variation_information(arr3, arr1)
        >>> vi_34 = variation_information(arr3, arr4)
        >>> # print(vi_12, vi_21, vi_31, vi_34)
        >>> np.isclose(vi_12, vi_21)
        True
        >>> vi_34 < vi_31
        True
    """
    if not isinstance(bins, int):
        if bins is not None and not isinstance(bins, str):
            raise ValueError('Invalid value for `bins`')
        bins = auto_bins((arr1, arr2), method=bins, combine=max)

    if not np.array_equal(arr1, arr2):
        hist1, bin_edges1 = np.histogram(arr1, bins)
        hist2, bin_edges2 = np.histogram(arr2, bins)
        hist12, x_edges, y_edges = np.histogram2d(arr1, arr2, bins=bins)
        h12 = entropy(hist12, base)
        h1 = entropy(hist1, base)
        h2 = entropy(hist2, base)
        vi = 2 * h12 - h1 - h2
    else:
        vi = 0.0
    # absolute value to fix rounding errors
    return abs(vi)


# ======================================================================
def norm_mutual_information(
        arr1,
        arr2,
        bins='rice'):
    """
    Calculate a normalized mutual information between two arrays.

    Note that the numerical result depends on the number of bins.

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.
        bins (int|str|None): The number of bins to use for the distribution.
            If int, the exact number is used.
            If str, a method accepted by `auto_bin` is expected.
            If None, uses the maximum number of bins (not recommended).

    Returns:
        mi (float): The normalized mutual information.

    Examples:
        >>> np.random.seed(0)
        >>> arr1 = np.zeros(100)
        >>> arr2 = np.arange(100)
        >>> arr3 = np.random.rand(100)
        >>> arr4 = arr3 + np.random.rand(100) / 100
        >>> mi_11 = norm_mutual_information(arr1, arr1)
        >>> mi_22 = norm_mutual_information(arr2, arr2)
        >>> mi_33 = norm_mutual_information(arr3, arr3)
        >>> mi_44 = norm_mutual_information(arr4, arr4)
        >>> # print(mi_11, mi_22, mi_33, mi_44)
        >>> 1.0 == mi_11 == mi_22 == mi_33 == mi_44
        True
        >>> mi_12 = norm_mutual_information(arr1, arr2)
        >>> mi_21 = norm_mutual_information(arr2, arr1)
        >>> mi_32 = norm_mutual_information(arr3, arr2)
        >>> mi_34 = norm_mutual_information(arr3, arr4)
        >>> # print(mi_12, mi_21, mi_32, mi_34)
        >>> mi_44 > mi_34 and mi_33 > mi_34
        True
        >>> np.isclose(mi_12, mi_21)
        True
        >>> mi_34 > mi_32
        True
        >>> mi_n10 = norm_mutual_information(arr3, arr2, 10)
        >>> mi_n20 = norm_mutual_information(arr3, arr2, 20)
        >>> mi_n100 = norm_mutual_information(arr3, arr2, 100)
        >>> # print(mi_n10, mi_n20, mi_n100)
        >>> mi_n10 < mi_n20 < mi_n100
        True
    """
    # todo: check if this is correct
    if not isinstance(bins, int):
        if bins is not None and not isinstance(bins, str):
            raise ValueError('Invalid value for `bins`')
        bins = auto_bins((arr1, arr2), method=bins, combine=max)
    hist1, bin_edges1 = np.histogram(arr1, bins)
    hist2, bin_edges2 = np.histogram(arr2, bins)
    hist12, x_edges, y_edges = np.histogram2d(arr1, arr2, bins=bins)
    if not np.array_equal(arr1, arr2):
        base = np.e  # results should be independent of the base
        h12 = entropy(hist12, base)
        h1 = entropy(hist1, base)
        h2 = entropy(hist2, base)
        nmi = 1 - (2 * h12 - h1 - h2) / h12
    else:
        nmi = 1.0

    # absolute value to fix rounding errors
    return abs(nmi)


# ======================================================================
def mutual_information(
        arr1,
        arr2,
        base=np.e,
        bins='rice'):
    """
    Calculate the mutual information between two arrays.

    Note that the numerical result depends on the number of bins.

    Args:
        arr1 (np.ndarray): The first input array.
        arr2 (np.ndarray): The second input array.
        base (int|float|None): The base units to express the result.
            Should be a number larger than 1.
            If base is 2, the unit is bits.
            If base is np.e (Euler's number), the unit is `nats`.
            If base is None, the result is normalized to unity.
        bins (int|str|None): The number of bins to use for the distribution.
            If int, the exact number is used.
            If str, a method accepted by `auto_bin` is expected.
            If None, uses the maximum number of bins (not recommended).

    Returns:
        mi (float): The (normalized) mutual information.
            If base is None, the normalized version is returned.
            Otherwise returns the mutual information in the specified base.

    Examples:
        >>> np.random.seed(0)
        >>> arr1 = np.zeros(100)
        >>> arr2 = np.arange(100)
        >>> arr3 = np.random.rand(100)
        >>> arr4 = arr3 + np.random.rand(100) / 100
        >>> mi_11 = mutual_information(arr1, arr1)
        >>> mi_22 = mutual_information(arr2, arr2)
        >>> mi_33 = mutual_information(arr3, arr3)
        >>> mi_44 = mutual_information(arr4, arr4)
        >>> # print(mi_11, mi_22, mi_33, mi_44)
        >>> mi_22 > mi_33 > mi_11
        True
        >>> mi_12 = mutual_information(arr1, arr2)
        >>> mi_21 = mutual_information(arr2, arr1)
        >>> mi_32 = mutual_information(arr3, arr2)
        >>> mi_34 = mutual_information(arr3, arr4)
        >>> # print(mi_12, mi_21, mi_32, mi_34)
        >>> mi_44 > mi_34 and mi_33 > mi_34
        True
        >>> np.isclose(mi_12, mi_21)
        True
        >>> mi_34 > mi_32
        True
        >>> mi_n10 = mutual_information(arr3, arr2, np.e, 10)
        >>> mi_n20 = mutual_information(arr3, arr2, np.e, 20)
        >>> mi_n100 = mutual_information(arr3, arr2, np.e, 100)
        >>> # print(mi_n10, mi_n20, mi_n100)
        >>> mi_n10 < mi_n20 < mi_n100
        True
        >>> mi_be = mutual_information(arr3, arr4, np.e)
        >>> mi_b2 = mutual_information(arr3, arr4, 2)
        >>> mi_b10 = mutual_information(arr3, arr4, 10)
        >>> # print(mi_be, mi_b2, mi_b10)
        >>> mi_b10 < mi_be < mi_b2
        True

    See Also:
        - Cahill, Nathan D. “Normalized Measures of Mutual Information with
          General Definitions of Entropy for Multimodal Image Registration.”
          In International Workshop on Biomedical Image Registration,
          258–268. Springer, 2010.
          http://link.springer.com/chapter/10.1007/978-3-642-14366-3_23.
    """
    # todo: check implementation speed and consistency
    if not isinstance(bins, int):
        if bins is not None and not isinstance(bins, str):
            raise ValueError('Invalid value for `bins`')
        bins = auto_bins((arr1, arr2), method=bins, combine=max)

    # # scikit.learn implementation
    # hist, x_edges, y_edges = np.histogram2d(arr1, arr2, bins=bins)
    # from sklearn.metrics import mutual_info_score
    # mi = mutual_info_score(None, None, contingency=hist)
    # if base > 0 and base != np.e:
    #     mi /= np.log(base)

    # # alternate implementation
    # hist, x_edges, y_edges = np.histogram2d(arr1, arr2, bins=bins)
    # g, p, dof, expected = scipy.stats.chi2_contingency(
    #     hist + np.finfo(np.float).eps, lambda_='log-likelihood')
    # mi = g / hist.sum() / 2

    if base:
        # entropy-based implementation
        hist1, bin_edges1 = np.histogram(arr1, bins)
        hist2, bin_edges2 = np.histogram(arr2, bins)
        hist12, x_edges, y_edges = np.histogram2d(arr1, arr2, bins=bins)
        h12 = entropy(hist12, base)
        h1 = entropy(hist1, base)
        h2 = entropy(hist2, base)
        mi = h1 + h2 - h12
    else:
        mi = norm_mutual_information(arr1, arr2, bins=bins)

    # absolute value to fix rounding errors
    return abs(mi)


# ======================================================================
def gaussian_nd(
        shape,
        sigmas,
        position=0.5,
        n_dim=None,
        norm=np.sum,
        rel_position=True):
    """
    Generate a Gaussian distribution in N dimensions.

    Args:
        shape (int|Iterable[int]): The shape of the array in px.
        sigmas (Iterable[int|float]): The standard deviation in px.
        position (float|Iterable[float]): The position of the center.
            Values are relative to the lowest edge, and scaled by the
            corresponding shape size.
        n_dim (int|None): The number of dimensions.
            If None, the number of dims is guessed from the other parameters.
        norm (callable|None): Normalize using the specified function.
        rel_position (bool): Interpret positions as relative values.
            If True, position values are interpreted as relative,
            i.e. they are scaled for `shape` values.
            Otherwise, they are interpreted as absolute (in px).
            Uses `flyingcircus.base.grid_coord()` internally.

    Returns:
        arr (np.ndarray): The array containing the N-dim Gaussian.

    Examples:
        >>> gaussian_nd(8, 1)
        array([0.00087271, 0.01752886, 0.12952176, 0.35207666, 0.35207666,
               0.12952176, 0.01752886, 0.00087271])
        >>> gaussian_nd(9, 2)
        array([0.02763055, 0.06628225, 0.12383154, 0.18017382, 0.20416369,
               0.18017382, 0.12383154, 0.06628225, 0.02763055])
        >>> gaussian_nd(3, 1, n_dim=2)
        array([[0.07511361, 0.1238414 , 0.07511361],
               [0.1238414 , 0.20417996, 0.1238414 ],
               [0.07511361, 0.1238414 , 0.07511361]])
        >>> gaussian_nd(7, 2, norm=None)
        array([0.32465247, 0.60653066, 0.8824969 , 1.        , 0.8824969 ,
               0.60653066, 0.32465247])
        >>> gaussian_nd(4, 2, 1.0, norm=None)
        array([0.32465247, 0.60653066, 0.8824969 , 1.        ])
        >>> gaussian_nd(3, 2, 5.0)
        array([0.00982626, 0.10564222, 0.88453152])
        >>> gaussian_nd(3, 2, 5.0, norm=None)
        array([3.72665317e-06, 4.00652974e-05, 3.35462628e-04])
    """
    if not n_dim:
        n_dim = fc.base.combine_iter_len((shape, sigmas, position))

    shape = fc.base.auto_repeat(shape, n_dim)
    sigmas = fc.base.auto_repeat(sigmas, n_dim)
    position = fc.base.auto_repeat(position, n_dim)

    position = grid_coord(
        shape, position, is_relative=rel_position, use_int=False)
    arr = np.exp(-(sum(
        x_i ** 2 / (2 * sigma ** 2) for x_i, sigma in zip(position, sigmas))))
    if callable(norm):
        arr /= norm(arr)
    return arr


# ======================================================================
def bijective_part(arr, invert=False):
    """
    Determine the largest bijective part of an array.

    Args:
        arr (np.ndarray): The input 1D-array.
        invert (bool): Invert the selection order for equally large parts.
            The behavior of `numpy.argmax` is the default.

    Returns:
        slice (slice): The largest bijective portion of arr.
            If two equivalent parts are found, uses the `numpy.argmax` default.

    Examples:
        >>> x = np.linspace(-1 / np.pi, 1 / np.pi, 5000)
        >>> arr = np.sin(1 / x)
        >>> bijective_part(x)
        slice(None, None, None)
        >>> bijective_part(arr)
        slice(None, 833, None)
        >>> bijective_part(arr, True)
        slice(4166, None, None)
    """
    local_mins = sp.signal.argrelmin(arr.ravel())[0]
    local_maxs = sp.signal.argrelmax(arr.ravel())[0]
    # boundaries are considered pseudo-local maxima and minima
    # but are not included in local_mins / local_maxs
    # therefore they are added manually
    extrema = np.zeros((len(local_mins) + len(local_maxs)) + 2, dtype=np.int)
    extrema[-1] = len(arr) - 1
    if len(local_mins) > 0 and len(local_maxs) > 0:
        # start with smallest maxima or minima
        if np.min(local_mins) < np.min(local_maxs):
            extrema[1:-1:2] = local_mins
            extrema[2:-1:2] = local_maxs
        else:
            extrema[1:-1:2] = local_maxs
            extrema[2:-1:2] = local_mins
    elif len(local_mins) == 1 and len(local_maxs) == 0:
        extrema[1] = local_mins
    elif len(local_mins) == 0 and len(local_maxs) == 1:
        extrema[1] = local_maxs
    elif len(local_maxs) == len(local_mins) == 0:
        pass
    else:
        raise ValueError('Failed to determine maxima and/or minima.')

    part_sizes = np.diff(extrema)
    if any(part_sizes) < 0:
        raise ValueError('Failed to determine orders of maxima and minima.')
    if not invert:
        largest = np.argmax(part_sizes)
    else:
        largest = len(part_sizes) - np.argmax(part_sizes[::-1]) - 1
    min_cut, max_cut = extrema[largest:largest + 2]
    return slice(
        min_cut if min_cut > 0 else None,
        max_cut if max_cut < len(arr) - 1 else None)


# ======================================================================
def polar2complex(modulus, phase):
    """
    Calculate complex number from the polar form:
    z = R * exp(i * phi) = R * cos(phi) + i * R * sin(phi).

    Args:
        modulus (float|np.ndarray): The modulus R of the complex number.
        phase (float|np.ndarray): The argument phi of the complex number.

    Returns:
        z (complex|np.ndarray): The complex number z = R * exp(i * phi).
    """
    return modulus * np.exp(1j * phase)


# ======================================================================
def cartesian2complex(real, imag):
    """
    Calculate the complex number from the cartesian form: z = z' + i * z".

    Args:
        real (float|np.ndarray): The real part z' of the complex number.
        imag (float|np.ndarray): The imaginary part z" of the complex number.

    Returns:
        z (complex|np.ndarray): The complex number: z = z' + i * z".
    """
    return real + 1j * imag


# ======================================================================
def complex2cartesian(z):
    """
    Calculate the real and the imaginary part of a complex number.

    Args:
        z (complex|np.ndarray): The complex number or array: z = z' + i * z".

    Returns:
        tuple[float|np.ndarray]:
         - real (float|np.ndarray): The real part z' of the complex number.
         - imag (float|np.ndarray): The imaginary part z" of the complex
         number.
    """
    return np.real(z), np.imag(z)


# ======================================================================
def complex2polar(z):
    """
    Calculate the real and the imaginary part of a complex number.

    Args:
        z (complex|np.ndarray): The complex number or array: z = z' + i * z".

    Returns:
        tuple[float|np.ndarray]:
         - modulus (float|np.ndarray): The modulus R of the complex number.
         - phase (float|np.ndarray): The phase phi of the complex number.
    """
    return np.abs(z), np.angle(z)


# ======================================================================
def polar2cartesian(modulus, phase):
    """
    Calculate the real and the imaginary part of a complex number.

    Args:
        modulus (float|np.ndarray): The modulus R of the complex number.
        phase (float|np.ndarray): The phase phi of the complex number.

    Returns:
        tuple[float|np.ndarray]:
         - real (float|np.ndarray): The real part z' of the complex number.
         - imag (float|np.ndarray): The imaginary part z" of the complex
         number.
    """
    return modulus * np.cos(phase), modulus * np.sin(phase)


# ======================================================================
def cartesian2polar(real, imag):
    """
    Calculate the real and the imaginary part of a complex number.

    Args:
        real (float): The real part z' of the complex number.
        imag (float): The imaginary part z" of the complex number.

    Returns:
        tuple[float|np.ndarray]:
         - modulus (float|np.ndarray): The modulus R of the complex number.
         - argument (float|np.ndarray): The phase phi of the complex number.
    """
    return np.sqrt(real ** 2 + imag ** 2), np.arctan2(real, imag)


# ======================================================================
def filter_cx(
        arr,
        func,
        args=None,
        kws=None,
        mode='cartesian'):
    """
    Calculate a non-complex function on a complex input array.

    Args:
        arr (np.ndarray): The input array.
        func (callable): The function used to filter the input.
            Requires the first arguments to be an `np.ndarray`.
        args (Sequence|None): Positional arguments for `func`.
        kws (Mappable|None): Keyword arguments for `func`.
        mode (str): Complex calculation mode.
            Available:
             - 'cartesian': apply to real and imaginary separately.
             - 'polar': apply to magnitude and phase separately.
             - 'real': apply to real part only.
             - 'imag': apply to imaginary part only.
             - 'mag': apply to magnitude part only.
             - 'phs': apply to phase part only.
            If unknown, uses default.

    Returns:
        arr (np.ndarray): The filtered complex array.
    """
    if mode:
        mode = mode.lower()
    args = tuple(args) if args else ()
    kws = dict(kws) if kws else {}
    if mode == 'cartesian':
        arr = func(arr.real, *args, **kws) \
              + 1j * func(arr.imag, *args, **kws)
    elif mode == 'polar':
        arr = func(np.abs(arr), *args, **kws) \
              * np.exp(1j * func(np.angle(arr), *args, **kws))
    elif mode == 'real':
        arr = func(arr.real, *args, **kws) + 1j * arr.imag
    elif mode == 'imag':
        arr = arr.real + 1j * func(arr.imag, *args, **kws)
    elif mode == 'mag':
        arr = func(np.abs(arr), *args, **kws) * np.exp(1j * np.angle(arr))
    elif mode == 'phs':
        arr = np.abs(arr) * np.exp(1j * func(np.angle(arr), *args, **kws))
    else:
        warnings.warn(
            'Mode `{}` not known'.format(mode) + ' Using default.')
        arr = filter_cx(arr, func, args, kws)
    return arr


# ======================================================================
def wrap_cyclic(
        arr,
        size=2 * np.pi,
        offset=np.pi):
    """
    Cyclic wrap values to a range with a specific size and offset.

    This is useful to emulate the behavior of phase wrapping.

    Args:
        arr (int|float|np.ndarray): The input value or array.
        size (int|float): The size of the wrapped range.
        offset (int|float): The offset of the wrapped range.

    Returns:
        arr (int|float|np.ndarray): The wrapped value or array.
    """
    return (arr + offset) % size - offset


# ======================================================================
def marginal_sep_elbow(items):
    """
    Determine the marginal separation using the elbow method.

    Graphically, this is displayed as an elbow in the plot.
    Mathematically, this is defined as the first item whose (signed) global
    slope is smaller than the (signed) local slope.

    Args:
        items (Iterable): The collection of items to inspect.
            The input must be already sorted non-negative values.

    Returns:
        index (int): The position of the marginal separation value.
            If the marginal separation is not found, returns -1.

    Examples:
        >>> items = (100, 90, 70, 60, 50, 30, 20, 5, 4, 3, 2, 1)
        >>> marginal_sep_elbow(items)
        8
        >>> items = (100, 90, 70, 60, 50, 30, 20, 5)
        >>> marginal_sep_elbow(items)
        -1
    """
    if fc.base.is_increasing(items):
        sign = -1
    elif fc.base.is_decreasing(items):
        sign = 1
    else:
        sign = None
    if sign:
        index = -1
        for i_, item in enumerate(items[1:]):
            i = i_ + 1
            local_slope = item - items[i_]
            global_slope = item - items[0] / i
            if sign * global_slope < sign * local_slope:
                index = i
                break
    else:
        index = -1
    return index


# ======================================================================
def marginal_sep_quad(items):
    """
    Determine the marginal separation using the quadrature method.

    Mathematically, this is defined as the first item whose value is smaller
    than the sum of the differences of all following items.

    Args:
        items (Iterable): The collection of items to inspect.
            The input must be already sorted non-negative values.

    Returns:
        index (int): The position of the marginal separation value.
            If the marginal separation is not found, returns -1.

    Examples:
        >>> items = (100, 90, 70, 50, 30, 20, 5, 2, 1)
        >>> marginal_sep_quad(items)
        5
    """
    if fc.base.is_increasing(items):
        sign = -1
    elif fc.base.is_decreasing(items):
        sign = 1
    else:
        sign = None
    if sign:
        index = np.where(
            items[:-1] + sign * np.cumsum(np.diff(items)[::-1]) < 0)[0]
        index = int(index[0]) + 1 if len(index) > 0 else -1
    else:
        index = -1
    return index


# ======================================================================
def marginal_sep_quad_weight(items):
    """
    Determine the marginal separation using the weighted quadrature.

    Mathematically, this is defined as the first item whose value is smaller
    than the sum of the differences of all following items weighted by the
    number of items already considered.

    Args:
        items (Iterable): The collection of items to inspect.
            The input must be already sorted non-negative values.

    Returns:
        index (int): The position of the marginal separation value.
            If the marginal separation is not found, returns -1.

    Examples:
        >>> items = (100, 90, 70, 50, 30, 20, 5, 2, 1)
        >>> marginal_sep_quad_weight(items)
        7
    """
    if fc.base.is_increasing(items):
        sign = -1
    elif fc.base.is_decreasing(items):
        sign = 1
    else:
        sign = None
    if sign:
        index = np.where(
            items[:-1] + sign * np.cumsum(np.diff(items)[::-1]) /
            np.arange(1, len(items)) < 0)[0]
        index = index[0] + 1 if len(index) else -1
    else:
        index = -1
    return index


# ======================================================================
def marginal_sep_quad_inv_weight(items):
    """
    Determine the marginal separation using the inverse weighted quadrature.

    Mathematically, this is defined as the first item whose value is smaller
    than the sum of the differences of all following items weighted by the
    number of items to be considered.

    Args:
        items (Iterable): The collection of items to inspect.
            The input must be already sorted non-negative values.

    Returns:
        index (int): The position of the marginal separation value.
            If the marginal separation is not found, returns -1.

    Examples:
        >>> items = (100, 90, 70, 50, 30, 20, 5, 2, 1)
        >>> marginal_sep_quad_inv_weight(items)
        7
    """
    if fc.base.is_increasing(items):
        sign = -1
    elif fc.base.is_decreasing(items):
        sign = 1
    else:
        sign = None
    if sign:
        index = np.where(
            items[:-1] + sign * np.cumsum(np.diff(items)[::-1]) /
            np.arange(len(items), 1, -1) < 0)[0]
        index = index[0] + 1 if len(index) else -1
    else:
        index = -1
    return index


# ======================================================================
def otsu_threshold(
        items,
        bins='sqrt'):
    """
    Optimal foreground/background threshold value based on Otsu's method.

    Args:
        items (Iterable): The input items.
        bins (int|str|None): Number of bins used to calculate histogram.
            If str or None, this is automatically calculated from the data
            using `flyingcircus.extra.auto_bin()` with `method` set to
            `bins` if str,
            and using the default `flyingcircus.extra.auto_bin()` method if
            set to
            None.

    Returns:
        threshold (float): The threshold value.

    Raises:
        ValueError: If `arr` only contains a single value.

    Examples:
        >>> num = 1000
        >>> x = np.linspace(-10, 10, num)
        >>> arr = np.sin(x) ** 2
        >>> threshold = otsu_threshold(arr)
        >>> round(threshold, 1)
        0.5

    References:
        - Otsu, N., 1979. A Threshold Selection Method from Gray-Level
          Histograms. IEEE Transactions on Systems, Man, and Cybernetics 9,
          62–66. doi:10.1109/TSMC.1979.4310076
    """
    # ensure items are not identical.
    items = np.array(items)
    if items.min() == items.max():
        warnings.warn('Items are all identical!')
        threshold = items.min()
    else:
        if isinstance(bins, str):
            bins = auto_bin(items, bins)
        elif bins is None:
            bins = auto_bin(items)

        hist, bin_edges = np.histogram(items, bins)
        bin_centers = midval(bin_edges)
        hist = hist.astype(float)

        # class probabilities for all possible thresholds
        weight1 = np.cumsum(hist)
        weight2 = np.cumsum(hist[::-1])[::-1]
        # class means for all possible thresholds
        mean1 = np.cumsum(hist * bin_centers) / weight1
        mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
        # calculate the variance for all possible thresholds
        variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2

        i_max_variance = np.argmax(variance12)
        threshold = bin_centers[:-1][i_max_variance]
    return threshold


# ======================================================================
def auto_num_components(
        k,
        q=None,
        num=None,
        verbose=D_VERB_LVL):
    """
    Calculate the optimal number of principal components.

    Effectively executing a Principal Component Analysis.

    Args:
        k (int|float|str): The number of principal components.
            If int, the exact number is given. It must not exceed the size
            of the `coil_axis` dimension.
            If float, the number is interpreted as relative to the size of
            the `coil_axis` dimension, and values must be in the
            [0.1, 1] interval.
            If str, the number is automatically estimated from the magnitude
            of the eigenvalues using a specific method.
            Accepted values are:
             - 'all': use all components.
             - 'full': same as 'all'.
             - 'elbow': use `flyingcircus.base.marginal_sep_elbow()`.
             - 'quad': use `flyingcircus.base.marginal_sep_quad()`.
             - 'quad_weight': use
             `flyingcircus.base.marginal_sep_quad_weight()`.
             - 'quad_inv_weight': use
             `flyingcircus.base.marginal_sep_quad_inv_weight()`.
             - 'otsu': use `flyingcircus.segmentation.threshold_otsu()`.
             - 'X%': set the threshold at 'X' percent of the largest eigenval.
        q (Iterable[Number]|None): The values of the components.
            If None, `num` must be specified.
            If Iterable, `num` must be None.
        num (int|None): The number of components.
            If None, `q` must be specified.
            If
        verbose (int): Set level of verbosity.

    Returns:
        k (int): The optimal number of principal components.

    Examples:
        >>> q = [100, 90, 70, 10, 5, 3, 2, 1]
        >>> auto_num_components('elbow', q)
        4
        >>> auto_num_components('quad_weight', q)
        5
    """
    if (q is None and num is None) or (q is not None and num is not None):
        raise ValueError('At most one of `q` and `num` must not be `None`.')
    elif q is not None and num is None:
        q = np.array(q).ravel()
        msg('q={}'.format(q), verbose, VERB_LVL['debug'])
        num = len(q)

    msg('k={}'.format(k), verbose, VERB_LVL['debug'])
    if isinstance(k, float):
        k = max(1, int(num * min(k, 1.0)))
    elif isinstance(k, str):
        if q is not None:
            k = k.lower()
            if k == 'elbow':
                k = marginal_sep_elbow(np.abs(q / q[0])) % (num + 1)
            elif k == 'quad':
                k = marginal_sep_quad(np.abs(q / q[0])) % (num + 1)
            elif k == 'quad_weight':
                k = marginal_sep_quad_weight(np.abs(q / q[0])) % (num + 1)
            elif k == 'quad_inv_weight':
                k = marginal_sep_quad_inv_weight(np.abs(q / q[0])) % (num + 1)
            elif k.endswith('%') and (100.0 > float(k[:-1]) >= 0.0):
                k = np.abs(q[0]) * float(k[:-1]) / 100.0
                k = np.where(np.abs(q) < k)[0]
                k = k[0] if len(k) else num
            elif k == 'otsu':
                k = otsu_threshold(q)
                k = np.where(q < k)[0]
                k = k[0] if len(k) else num
            elif k == 'all' or k == 'full':
                k = num
            else:
                warnings.warn('`{}`: invalid value for `k`.'.format(k))
                k = num
        else:
            warnings.warn('`{}`: method requires `q`.'.format(k))
            k = num
    if not 0 < k <= num:
        warnings.warn('`{}` is invalid. Using: `{}`.'.format(k, num))
        k = num
    msg('k/num={}/{}'.format(k, num), verbose, VERB_LVL['medium'])
    return k


# ======================================================================
def avg(
        arr,
        axis=None,
        dtype=None,
        out=None,
        keepdims=np._NoValue,
        weights=None,
        removes=(np.inf, -np.inf)):
    """
    Calculate the (weighted) average of the array.

    The weighted average is defined as:

    .. math::
        avg(x, w) = \\frac{\\sum_i w_i x_i}{\\sum_i w_i}

    where :math:`x` is the input N-dim array, :math:`w` is the N-dim array of
    the weights, and :math:`i` runs through the dimension along which to
    compute.

    Args:
        arr (np.ndarray|Iterable): The input data.
        axis (int|Iterable[int]|None): Axis along which to compute.
            See `np.nansum()` for more info.
        dtype (np.dtype|None): The data type of the result.
            See `np.nansum()` for more info.
        out (np.ndarray|None):
            See `np.nansum()` for more info.
        keepdims (bool): Keep reduced axis in the result as dims with size 1.
            See `np.nansum()` for more info.
        weights (np.ndarray|Iterable|None): The weights.
            If np.ndarray or Iterable, the size must match with `arr`.
            If None, all wegiths are set to 1 (equivalent to no weighting).
        removes (Iterable): Values to remove.
            If empty, no values will be removed.

    Returns:
        result (np.ndarray): The computed statistics.
            Its shape depends on the value of axis.

    Examples:
        >>> arr = np.array([0, 0, 1, 0])
        >>> weights = np.array([1, 1, 3, 1])
        >>> avg(arr)
        0.25
        >>> avg(arr, weights=weights)
        0.5
        >>> avg(arr, weights=weights) == avg(np.array([0, 0, 1, 0, 1, 1]))
        True
        >>> np.mean(arr) == avg(arr)
        True
        >>> arr = np.arange(2 * 3 * 4, dtype=float).reshape((2, 3, 4))
        >>> weights = np.arange(4) + 1
        >>> avg(arr, weights=weights, axis=-1)
        array([[ 2.,  6., 10.],
               [14., 18., 22.]])
        >>> weights = np.arange(2 * 3).reshape((2, 3)) + 1
        >>> avg(arr, weights=weights, axis=(0, 1), removes=(1,))
        array([13.33333333, 15.        , 15.33333333, 16.33333333])

    See Also:
        - flyingcircus.extra.var()
        - flyingcircus.extra.std()
        - np.std()
        - np.var()
    """
    arr = np.array(arr)
    if np.issubdtype(arr.dtype, np.dtype(int).type):
        arr = arr.astype(float)
    if weights is not None:
        weights = np.array(weights, dtype=float)
        if weights.shape != arr.shape:
            weights = unsqueeze(
                weights, axis=axis, shape=arr.shape, complement=True)
            # cannot use `np.broadcast_to()` because we need to write data
            weights = np.zeros_like(arr) + weights
    for val in removes:
        mask = arr == val
        if val in arr:
            arr[mask] = np.nan
            if weights is not None:
                weights[mask] = np.nan
    if weights is None:
        weights = np.ones_like(arr)
    result = np.nansum(
        arr * weights, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    result /= np.nansum(
        weights, axis=axis, dtype=dtype, out=out, keepdims=keepdims)
    return result


# ======================================================================
def var(
        arr,
        axis=None,
        dtype=None,
        out=None,
        keepdims=np._NoValue,
        weights=None,
        removes=(np.inf, -np.inf)):
    """
    Calculate the (weighted) variance of the array.

    The weighted variance is defined as:

    .. math::
        var(x, w) = \\frac{\\sum_i (w_i x_i - avg(x, w))^2}{\\sum_i w_i}

    where :math:`x` is the input N-dim array, :math:`w` is the N-dim array of
    the weights, and :math:`i` runs through the dimension along which to
    compute.

    Args:
        arr (np.ndarray|Iterable): The input data.
        axis (int|Iterable[int]|None): Axis along which to compute.
            See `np.nansum()` for more info.
        dtype (np.dtype|None): The data type of the result.
            See `np.nansum()` for more info.
        out (np.ndarray|None):
            See `np.nansum()` for more info.
        keepdims (bool): Keep reduced axis in the result as dims with size 1.
            See `np.nansum()` for more info.
        weights (np.ndarray|Iterable|None): The weights.
            If np.ndarray or Iterable, the size must match with `arr`.
            If None, all wegiths are set to 1 (equivalent to no weighting).
        removes (Iterable): Values to remove.
            If empty, no values will be removed.

    Returns:
        result (np.ndarray): The computed statistics.
            Its shape depends on the value of axis.

    See Also:
        - flyingcircus.extra.avg()
        - flyingcircus.extra.std()
        - np.mean()
        - np.average()
        - np.std()

    Examples:
        >>> arr = np.array([0, 0, 1, 0])
        >>> weights = np.array([1, 1, 3, 1])
        >>> var(arr, weights=weights)
        0.25
        >>> var(arr, weights=weights) == var(np.array([0, 0, 1, 0, 1, 1]))
        True
        >>> np.var(arr) == var(arr)
        True
        >>> arr = np.arange(2 * 3 * 4, dtype=float).reshape((2, 3, 4))
        >>> weights = np.arange(4) + 1
        >>> var(arr, weights=weights, axis=-1)
        array([[0.8, 0.8, 0.8],
               [0.8, 0.8, 0.8]])
        >>> weights = np.arange(2 * 3).reshape((2, 3)) + 1
        >>> var(arr, weights=weights, axis=(0, 1), removes=(1,))
        array([28.44444444, 26.15384615, 28.44444444, 28.44444444])
    """
    arr = np.array(arr)
    if weights is not None:
        weights = np.array(weights, dtype=float)
    avg_arr = avg(
        arr, axis=axis, dtype=dtype, out=out, keepdims=True,
        weights=weights, removes=removes)
    result = avg(
        (arr - avg_arr) ** 2, axis=axis, dtype=dtype, out=out,
        keepdims=keepdims,
        weights=weights ** 2 if weights is not None else None, removes=removes)
    return result


# ======================================================================
def std(
        arr,
        axis=None,
        dtype=None,
        out=None,
        keepdims=np._NoValue,
        weights=None,
        removes=(np.inf, -np.inf)):
    """
    Calculate the (weighted) standard deviation of the array.

    The weighted standard deviation is defined as the square root of the
    variance.

    Args:
        arr (np.ndarray|Iterable): The input data.
        axis (int|Iterable[int]|None): Axis along which to compute.
            See `np.nansum()` for more info.
        dtype (np.dtype|None): The data type of the result.
            See `np.nansum()` for more info.
        out (np.ndarray|None):
            See `np.nansum()` for more info.
        keepdims (bool): Keep reduced axis in the result as dims with size 1.
            See `np.nansum()` for more info.
        weights (np.ndarray|Iterable|None): The weights.
            If np.ndarray or Iterable, the size must match with `arr`.
            If None, all wegiths are set to 1 (equivalent to no weighting).
        removes (Iterable): Values to remove.
            If empty, no values will be removed.

    Returns:
        result (np.ndarray): The computed statistics.
            Its shape depends on the value of axis.

    Examples:
        >>> arr = np.array([0, 0, 1, 0])
        >>> weights = np.array([1, 1, 3, 1])
        >>> std(arr, weights=weights)
        0.5
        >>> std(arr, weights=weights) == std(np.array([0, 0, 1, 0, 1, 1]))
        True
        >>> np.std(arr) == std(arr)
        True

    See Also:
        - flyingcircus.extra.avg()
        - flyingcircus.extra.std()
        - np.mean()
        - np.average()
        - np.std()
    """
    return np.sqrt(
        var(arr, axis=axis, dtype=dtype, out=out, keepdims=keepdims,
            weights=weights, removes=removes))


# ======================================================================
def gavg(
        arr,
        axis=None,
        dtype=None,
        out=None,
        keepdims=np._NoValue,
        weights=None,
        removes=(np.inf, -np.inf)):
    """
    Calculate the (weighted) geometric average of the array.

    The weighted geometric average is defined as exponential of the
    weighted average of the logarithm of the absolute value of the array.

    Args:
        arr (np.ndarray|Iterable): The input data.
        axis (int|Iterable[int]|None): Axis along which to compute.
            See `np.nansum()` for more info.
        dtype (np.dtype|None): The data type of the result.
            See `np.nansum()` for more info.
        out (np.ndarray|None):
            See `np.nansum()` for more info.
        keepdims (bool): Keep reduced axis in the result as dims with size 1.
            See `np.nansum()` for more info.
        weights (np.ndarray|Iterable|None): The weights.
            If np.ndarray or Iterable, the size must match with `arr`.
            If None, all wegiths are set to 1 (equivalent to no weighting).
        removes (Iterable): Values to remove.
            If empty, no values will be removed.

    Returns:
        result (np.ndarray): The computed statistics.
            Its shape depends on the value of axis.

    Examples:
        >>> arr = np.array([1, 1, 4, 1])
        >>> weights = np.array([1, 1, 3, 1])
        >>> gavg(arr, weights=weights)
        2.0
        >>> gavg(arr, weights=weights) == gavg(np.array([1, 1, 4, 1, 4, 4]))
        True
        >>> sp.stats.gmean(arr) == gavg(arr)
        True

    See Also:
        - scipy.stats.gmean()
        - flyingcircus.extra.avg()
        - flyingcircus.extra.std()
        - np.mean()
        - np.average()
        - np.std()
    """
    return np.exp(
        avg(np.log(np.abs(arr)), axis=axis, dtype=dtype, out=out,
            keepdims=keepdims, weights=weights, removes=removes))


# ======================================================================
def calc_stats(
        arr,
        removes=(np.nan, np.inf, -np.inf),
        val_interval=None,
        save_path=None,
        title=None,
        compact=False):
    """
    Calculate array statistical information (min, max, avg, std, sum, num).

    Args:
        arr (np.ndarray): The array to be investigated.
        removes (Iterable): Values to remove.
            If empty, no values will be removed.
        val_interval (tuple): The (min, max) values interval.
        save_path (str|None): The path to which the plot is to be saved.
            If None, no output.
        title (str|None): If title is not None, stats are printed to screen.
        compact (bool): Use a compact format string for displaying results.

    Returns:
        stats_dict (dict): Dictionary of statistical values.
            Statistical parameters calculated:
                - 'min': minimum value
                - 'max': maximum value
                - 'avg': average or mean
                - 'std': standard deviation
                - 'sum': summation
                - 'num': number of elements

    Examples:
        >>> a = np.arange(2)
        >>> d = calc_stats(a)
        >>> tuple(sorted(d.items()))
        (('avg', 0.5), ('max', 1), ('min', 0), ('num', 2), ('std', 0.5),\
 ('sum', 1))
        >>> a = np.arange(200)
        >>> d = calc_stats(a)
        >>> tuple(sorted(d.items()))
        (('avg', 99.5), ('max', 199), ('min', 0), ('num', 200),\
 ('std', 57.73430522661548), ('sum', 19900))
    """
    stats_dict = {
        'avg': None, 'std': None,
        'min': None, 'max': None,
        'sum': None, 'num': None}
    arr = ravel_clean(arr, removes)
    if val_interval is None and len(arr) > 0:
        val_interval = minmax(arr)
    if len(arr) > 0:
        arr = arr[arr >= val_interval[0]]
        arr = arr[arr <= val_interval[1]]
    if len(arr) > 0:
        stats_dict = {
            'avg': np.mean(arr),
            'std': np.std(arr),
            'min': np.min(arr),
            'max': np.max(arr),
            'sum': np.sum(arr),
            'num': np.size(arr), }
    if save_path or title:
        label_list = ['avg', 'std', 'min', 'max', 'sum', 'num']
        val_list = []
        for label in label_list:
            val_list.append(fc.base.compact_num_str(stats_dict[label]))
        if save_path:
            with open(save_path, 'wb') as csv_file:
                csv_writer = csv.writer(
                    csv_file, delimiter=str(fc.base.CSV_DELIMITER))
                csv_writer.writerow(label_list)
                csv_writer.writerow(val_list)
        if title:
            print_str = title + ': '
            for label in label_list:
                if compact:
                    print_str += '{}={}, '.format(
                        label, fc.base.compact_num_str(stats_dict[label]))
                else:
                    print_str += '{}={}, '.format(label, stats_dict[label])
            print(print_str)
    return stats_dict


# ======================================================================
def bounding_box(mask):
    """
    Find the bounding box of a mask.

    Args:
        mask (np.ndarray[bool]):

    Returns:
        result (tuple[slice]): The slices of the bounding in all dimensions.

    Examples:
        >>> arr = np.array([0, 0, 1, 0, 0, 1, 0], dtype=bool)
        >>> print(bounding_box(arr > 0))
        (slice(2, 6, None),)
        >>> arr = np.array(
        ...     [[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=bool)
        >>> print(bounding_box(arr > 0))
        (slice(1, 2, None), slice(1, 3, None))
    """
    return tuple(
        slice(np.min(indexes), np.max(indexes) + 1)
        for indexes in np.where(mask))


# ======================================================================
def apply_mask(
        arr,
        mask,
        borders=None,
        background=0.0,
        do_unsqueeze=True):
    """
    Apply a mask to an array.

    Note: this will not produced a masked array `numpy.ma` object.

    Args:
        arr (np.ndarray): The input array.
        mask (np.ndarray|None): The mask array.
            If np.ndarray, the shape of `arr` and `mask` must be identical,
            broadcastable through `np.broadcast_to()`, or unsqueezable using
            `flyingcircus.extra.unsqueeze()`.
            If None, no masking is performed.
        borders (int|float|Sequence[int|float]|None): The border size(s).
            If None, the border is not modified.
            Otherwise, a border is added to the masked array.
            If int, this is in units of pixels.
            If float, this is proportional to the initial array shape.
            If int or float, uses the same value for all dimensions,
            unless `do_unsqueeze` is set to True, in which case, the same value
            is used only for non-singletons, while 0 is used for singletons.
            If Iterable, the size must match `arr` dimensions.
        background (int|float): The value used for masked-out pixels.
        do_unsqueeze (bool): Unsqueeze mask to input.
            If True, use `flyingcircus.extra.unsqueeze()` on mask.
            Only effective when `arr` and `mask` shapes do not match and
            are not already broadcastable.
            Otherwise, shapes must match or be broadcastable.

    Returns:
        arr (np.ndarray): The output array.
            Values outside of the mask are set to background.
            Array shape may have changed (depending on `borders`).

    Raises:
        ValueError: If the mask and array shapes are not compatible.

    See Also:
        - flyingcircus.extra.padding()
    """
    if mask is not None:
        mask = mask.astype(bool)
        if arr.ndim > mask.ndim and do_unsqueeze:
            old_shape = mask.shape
            mask = unsqueeze(mask, shape=arr.shape)
            if isinstance(borders, (int, float)):
                borders = [0 if dim == 1 else borders for dim in mask.shape]
            elif borders is not None and len(borders) == len(old_shape):
                iter_borders = iter(borders)
                borders = [
                    next(iter_borders) if dim == 1 else dim
                    for dim in mask.shape]
        arr = arr.copy()
        if arr.shape != mask.shape:
            mask = np.broadcast_to(mask, arr.shape)
        if arr.shape == mask.shape:
            arr[~mask] = background
            if borders is not None:
                slicing = tuple(
                    tuple(sp.ndimage.find_objects(mask.astype(int)))[0])
                if slicing:
                    arr = arr[slicing]
                arr = frame(arr, borders, background)
        else:
            raise ValueError(
                'Cannot apply mask shaped `{}` to array shaped `{}`.'.format(
                    mask.shape, arr.shape))
    return arr


# ======================================================================
def trim(
        arr,
        mask,
        axis=None):
    """
    Trim the borders of an array along the specified dimensions.

    Args:
        arr (np.ndarray): The input array.
        mask (np.ndarray[bool]): The mask array.
            The False borders in all dimensions are trimmed away.
        axis (int|tuple[int]|None): The axis along which to operate.

    Returns:
        result (np.ndarray): The trimmed array.

    Examples:
        >>> arr = np.array([0, 0, 1, 0, 0, 1, 0])
        >>> print(trim(arr, arr > 0))
        [1 0 0 1]
        >>> arr = np.array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
        >>> print(trim(arr, arr > 0))
        [[1 1]]
        >>> print(trim(arr, arr > 0, 0))
        [[0 1 1 0]]
        >>> print(trim(arr, arr > 0, 1))
        [[0 0]
         [1 1]
         [0 0]]
    """
    assert (mask.shape == arr.shape)
    if axis is None:
        axis = set(range(mask.ndim))
    else:
        try:
            iter(axis)
        except TypeError:
            axis = (axis,)
    slices = tuple(
        slice(None) if i not in axis else slice_
        for i, slice_ in enumerate(bounding_box(mask)))
    return arr[slices]


# ======================================================================
def frame(
        arr,
        width=0.05,
        mode=0,
        combine=max,
        **_kws):
    """
    Add a background frame to an array specifying the borders.

    This is essentially a more convenient interface to
    `flyingcircus.extra.padding()`.
    Also, the output `mask` from `flyingcircus.extra.padding()` is discarded.

    Args:
        arr (np.ndarray): The input array.
        width (int|float|Iterable[int|float]): Size of the padding to use.
            Passed to `flyingcircus.base.padding()`.
        mode (Number|Iterable[Number|Iterable]|str): The padding mode.
            Passed to `flyingcircus.base.padding()`.
        combine (callable|None): The function for combining pad width values.
            Passed to `flyingcircus.base.padding()`.
        **_kws: Keyword parameters for `flyingcircus.base.padding()`.

    Returns:
        result (np.ndarray): The padded array.

    See Also:
        - flyingcircus.extra.reframe()
        - flyingcircus.extra.padding()

    Examples:
        >>> arr = arange_nd((2, 3)) + 1
        >>> print(arr)
        [[1 2 3]
         [4 5 6]]
        >>> print(frame(arr, 1))
        [[0 0 0 0 0]
         [0 1 2 3 0]
         [0 4 5 6 0]
         [0 0 0 0 0]]
    """
    result, mask = padding(arr, width, combine, mode, **_kws)
    return result


# ======================================================================
def reframe(
        arr,
        shape,
        position=0.5,
        mode=0,
        rounding=round,
        **_kws):
    """
    Add a frame to an array by centering the input array into a new shape.

    Args:
        arr (np.ndarray): The input array.
            Its shape is passed as `shape` to
            `flyingcircus.extra.shape_to_pad_width()`.
        shape (int|Iterable[int]): The shape of the output array.
            Passed as `new_shape` to `flyingcircus.extra.shape_to_pad_width()`.
        position (int|float|Iterable[int|float]): Position within new shape.
            Passed as `position` to `flyingcircus.extra.shape_to_pad_width()`.
        mode (str|Number): The padding mode.
            This is passed to `flyingcircus.extra.padding()`.
        rounding (callable): The rounding method for the position.
            Passed as `rounding` to `flyingcircus.extra.shape_to_pad_width()`.
        **_kws: Keyword parameters for `flyingcircus.base.padding()`.

    Returns:
        result (np.ndarray): The result array with added borders.

    Raises:
        IndexError: input and output shape sizes must match.
        ValueError: output shape cannot be smaller than the input shape.

    See Also:
        - flyingcircus.extra.frame()
        - flyingcircus.extra.padding()

    Examples:
        >>> arr = np.ones(4, dtype=int)
        >>> print(arr)
        [1 1 1 1]
        >>> print(reframe(arr, 5))
        [1 1 1 1 0]
        >>> print(reframe(arr, 5, 1))
        [0 1 1 1 1]
        >>> print(reframe(arr, 5, (1,)))
        [0 1 1 1 1]
        >>> print(reframe(arr, (5,)))
        [1 1 1 1 0]
        >>> print(reframe(arr, (5,), 1))
        [0 1 1 1 1]
        >>> print(reframe(arr, (5,), (1,)))
        [0 1 1 1 1]

        >>> arr = np.ones((2, 3), dtype=int)
        >>> print(arr)
        [[1 1 1]
         [1 1 1]]
        >>> print(reframe(arr, (4, 5)))
        [[0 0 0 0 0]
         [0 1 1 1 0]
         [0 1 1 1 0]
         [0 0 0 0 0]]
        >>> print(reframe(arr, (4, 5), 0))
        [[1 1 1 0 0]
         [1 1 1 0 0]
         [0 0 0 0 0]
         [0 0 0 0 0]]
        >>> print(reframe(arr, (4, 5), (2, 0)))
        [[0 0 0 0 0]
         [0 0 0 0 0]
         [1 1 1 0 0]
         [1 1 1 0 0]]
        >>> print(reframe(arr, (4, 5), (0.0, 1.0)))
        [[0 0 1 1 1]
         [0 0 1 1 1]
         [0 0 0 0 0]
         [0 0 0 0 0]]
        >>> print(reframe(arr, (4, 5), 1.0))
        [[0 0 0 0 0]
         [0 0 0 0 0]
         [0 0 1 1 1]
         [0 0 1 1 1]]

        >>> arr = arange_nd((2, 3)) + 1
        >>> print(reframe(arr, (4, 5), 0, 'wrap'))
        [[1 2 3 1 2]
         [4 5 6 4 5]
         [1 2 3 1 2]
         [4 5 6 4 5]]

        >>> print(reframe(arr, (4, 5), 1, 'wrap'))
        [[6 4 5 6 4]
         [3 1 2 3 1]
         [6 4 5 6 4]
         [3 1 2 3 1]]

        >>> print(reframe(arr, (4, 5), 0.5, 'wrap'))
        [[6 4 5 6 4]
         [3 1 2 3 1]
         [6 4 5 6 4]
         [3 1 2 3 1]]
    """
    width = width_from_shapes(arr.shape, shape, position, rounding)
    result, mask = padding(arr, width, None, mode, **_kws)
    return result


# ======================================================================
def multi_reframe(
        arrs,
        new_shape=None,
        positions=0.5,
        background=0.0,
        dtype=None,
        rounding=round):
    """
    Reframe arrays (by adding border) to match the same shape.

    Note that:
     - uses 'reframe' under the hood;
     - the sampling / resolution / voxel size will NOT change;
     - the support space / field-of-view will change.

    Args:
        arrs (Iterable[np.ndarray]): The input arrays.
        new_shape (Iterable[int]): The new base shape of the arrays.
            If None, the uses the minimum shape that would fit all input
            arrays.
        positions (int|float|Iterable[int|float]): Position within new shape.
            See `flyingcircus.extra.reframe()` for more info.
        background (Number): The background value for the frame.
        dtype (data-type): Desired output data-type.
            If None, its guessed from dtype of arrs.
            See `np.ndarray()` for more.
        rounding (callable): The rounding method for the position.
            Passed as `rounding` to `flyingcircus.extra.shape_to_pad_width()`.

    Returns:
        result (np.ndarray): The output array.
            It contains all reframed arrays from `arrs`, through the last dim.
            The shape of this array is `new_shape` + `len(arrs)`.

    Examples:
        >>> arrs = [np.arange(i) + 1 for i in range(1, 9)]
        >>> for arr in arrs: print(arr)
        [1]
        [1 2]
        [1 2 3]
        [1 2 3 4]
        [1 2 3 4 5]
        [1 2 3 4 5 6]
        [1 2 3 4 5 6 7]
        [1 2 3 4 5 6 7 8]
        >>> print(multi_reframe(arrs).transpose())
        [[0 0 0 0 1 0 0 0]
         [0 0 0 1 2 0 0 0]
         [0 0 1 2 3 0 0 0]
         [0 0 1 2 3 4 0 0]
         [0 0 1 2 3 4 5 0]
         [0 1 2 3 4 5 6 0]
         [1 2 3 4 5 6 7 0]
         [1 2 3 4 5 6 7 8]]
        >>> print(multi_reframe(arrs, positions=0.0).transpose())
        [[1 0 0 0 0 0 0 0]
         [1 2 0 0 0 0 0 0]
         [1 2 3 0 0 0 0 0]
         [1 2 3 4 0 0 0 0]
         [1 2 3 4 5 0 0 0]
         [1 2 3 4 5 6 0 0]
         [1 2 3 4 5 6 7 0]
         [1 2 3 4 5 6 7 8]]
        >>> print(multi_reframe(arrs, positions=1.0).transpose())
        [[0 0 0 0 0 0 0 1]
         [0 0 0 0 0 0 1 2]
         [0 0 0 0 0 1 2 3]
         [0 0 0 0 1 2 3 4]
         [0 0 0 1 2 3 4 5]
         [0 0 1 2 3 4 5 6]
         [0 1 2 3 4 5 6 7]
         [1 2 3 4 5 6 7 8]]
        >>> print(multi_reframe(arrs, (9,), 1).transpose())
        [[0 1 0 0 0 0 0 0 0]
         [0 1 2 0 0 0 0 0 0]
         [0 1 2 3 0 0 0 0 0]
         [0 1 2 3 4 0 0 0 0]
         [0 1 2 3 4 5 0 0 0]
         [0 1 2 3 4 5 6 0 0]
         [0 1 2 3 4 5 6 7 0]
         [0 1 2 3 4 5 6 7 8]]
        >>> print(multi_reframe(arrs, (9,), 1.0).transpose())
        [[0 0 0 0 0 0 0 0 1]
         [0 0 0 0 0 0 0 1 2]
         [0 0 0 0 0 0 1 2 3]
         [0 0 0 0 0 1 2 3 4]
         [0 0 0 0 1 2 3 4 5]
         [0 0 0 1 2 3 4 5 6]
         [0 0 1 2 3 4 5 6 7]
         [0 1 2 3 4 5 6 7 8]]
    """
    # calculate new shape
    if new_shape is None:
        shapes = [arr.shape for arr in arrs]
        new_shape = [1] * max(len(shape) for shape in shapes)
        shape_arr = np.ones((len(shapes), len(new_shape))).astype(np.int)
        for i, shape in enumerate(shapes):
            shape_arr[i, :len(shape)] = np.array(shape)
        new_shape = tuple(
            int(max(*list(shape_arr[:, i])))
            for i in range(len(new_shape)))

    positions = fc.base.auto_repeat(positions, len(arrs))

    if dtype is None:
        # : alternative to looping
        # dtype = functools.reduce(
        #     (lambda x, y: np.promote_types(x, y.dtype)), arrs)
        dtype = bool
        for arr in arrs:
            dtype = np.promote_types(dtype, arr.dtype)

    result = np.zeros(new_shape + (len(arrs),), dtype=dtype)
    for i, (arr, position) in enumerate(zip(arrs, positions)):
        result[..., i] = reframe(
            arr, new_shape, position, background, rounding)
    return result


# ======================================================================
def zoom_prepare(
        zoom_factors,
        shape,
        extra_dim=True,
        fill_dim=True):
    """
    Prepare the zoom and shape tuples to allow for non-homogeneous shapes.

    Args:
        zoom_factors (float|Iterable[float]): The factors for each direction.
        shape (int|Iterable[int]): The shape of the array to operate with.
        extra_dim (bool): Force extra dimensions in the zoom parameters.
        fill_dim (bool): Dimensions not specified are left untouched.

    Returns:
        zoom (tuple[float]): The zoom factors for each directions.
        shape (int|Iterable[int]): The shape of the array to operate with.
    """
    zoom_factors = list(fc.base.auto_repeat(zoom_factors, len(shape)))
    if extra_dim:
        shape = list(shape) + [1] * (len(zoom_factors) - len(shape))
    else:
        zoom_factors = zoom_factors[:len(shape)]
    if fill_dim and len(zoom_factors) < len(shape):
        zoom_factors[len(zoom_factors):] = \
            [1.0] * (len(shape) - len(zoom_factors))
    return zoom_factors, shape


# ======================================================================
def shape2zoom(
        old_shape,
        new_shape,
        aspect=None):
    """
    Calculate zoom (or conversion) factor between two shapes.

    Args:
        old_shape (int|Iterable[int]): The shape of the source array.
        new_shape (int|Iterable[int]): The target shape of the array.
        aspect (callable|None): Function for the manipulation of the zoom.
            Signature: aspect(Iterable[float]) -> float.
            None to leave the zoom unmodified. If specified, the function is
            applied to zoom factors tuple for fine tuning of the aspect.
            Particularly, to obtain specific aspect ratio results:
             - 'min': image strictly contained into new shape
             - 'max': new shape strictly contained into image

    Returns:
        zoom (tuple[float]): The zoom factors for each directions.
    """
    if len(old_shape) != len(new_shape):
        raise IndexError('length of tuples must match')
    zoom_factors = [new / old for old, new in zip(old_shape, new_shape)]
    if aspect:
        zoom_factors = [aspect(zoom_factors)] * len(zoom_factors)
    return zoom_factors


# ======================================================================
def zoom(
        arr,
        factors,
        window=None,
        interp_order=0,
        extra_dim=True,
        fill_dim=True,
        cx_mode='cartesian'):
    """
    Zoom the array with a specified magnification factor.

    Args:
        arr (np.ndarray): The input array.
        factors (int|float|Iterable[int|float]): The zoom factor(s).
            If int or float, uses isotropic factor along all axes.
            If Iterable, its size must match the number of dims of `arr`.
            Values larger than 1 increase `arr` size along the axis.
            Values smaller than 1 decrease `arr` size along the axis.
        window (int|Iterable[int]|None): Uniform pre-filter window size.
            This is the size of the window for the uniform filter using
            `scipy.ndimage.uniform_filter()`.
            If Iterable, its size must match the number of dims of `arr`.
            If int, uses an isotropic window with the specified size.
            If None, the window is calculated automatically from the `zoom`
            parameter.
        interp_order (int): Order of the spline interpolation.
            0: nearest. Accepted range: [0, 5].
        extra_dim (bool): Force extra dimensions in the zoom parameters.
        fill_dim (bool): Dimensions not specified are left untouched.
        cx_mode (str): Complex calculation mode.
            This is passed as `mode` to `flyingcircus.extra.filter_cx()`.

    Returns:
        result (np.ndarray): The output array.
    """
    factors, shape = zoom_prepare(factors, arr.shape, extra_dim, fill_dim)
    if window is None:
        window = [round(1.0 / (2.0 * x)) for x in factors]
    if np.issubdtype(arr.dtype, np.complexfloating):
        arr = filter_cx(
            arr, sp.ndimage.uniform_filter, (window,), mode=cx_mode)
        arr = filter_cx(
            arr.reshape(shape), sp.ndimage.zoom, (factors,),
            dict(order=interp_order), mode=cx_mode)
    else:
        arr = sp.ndimage.uniform_filter(arr, window)
        arr = sp.ndimage.zoom(
            arr.reshape(shape), factors, order=interp_order)
    return arr


# ======================================================================
def resample(
        arr,
        new_shape,
        aspect=None,
        window=None,
        interp_order=0,
        extra_dim=True,
        fill_dim=True,
        cx_mode='cartesian'):
    """
    Reshape the array to a new shape (different resolution / pixel size).

    Args:
        arr (np.ndarray): The input array.
        new_shape (Iterable[int|None]): New dimensions of the array.
        aspect (callable|Iterable[callable]|None): Zoom shape manipulation.
            Useful for obtaining specific aspect ratio effects.
            This is passed to `flyingcircus.extra.shape2zoom()`.
        window (int|Iterable[int]|None): Uniform pre-filter window size.
            This is the size of the window for the uniform filter using
            `scipy.ndimage.uniform_filter()`.
            If Iterable, its size must match the number of dims of `arr`.
            If int, uses an isotropic window with the specified size.
            If None, the window is calculated automatically from `new_shape`.
        interp_order (int|None): Order of the spline interpolation.
            0: nearest. Accepted range: [0, 5].
        extra_dim (bool): Force extra dimensions in the zoom parameters.
        fill_dim (bool): Dimensions not specified are left untouched.
        cx_mode (str): Complex calculation mode.
            This is passed as `mode` to `flyingcircus.extra.filter_cx()`.

    Returns:
        arr (np.ndarray): The output array.

    See Also:
        - flyingcircus.extra.zoom()
    """
    factors = shape2zoom(arr.shape, new_shape, aspect)
    factors, shape = zoom_prepare(
        factors, arr.shape, extra_dim, fill_dim)
    arr = zoom(
        arr, factors, window=window, interp_order=interp_order,
        cx_mode=cx_mode)
    return arr


# ======================================================================
def multi_resample(
        arrs,
        new_shape=None,
        lossless=False,
        window=None,
        interp_order=0,
        extra_dim=True,
        fill_dim=True,
        dtype=None):
    """
    Resample arrays to match the same shape.

    Note that:
     - uses 'geometry.resample()' internally;
     - the sampling / resolution / voxel size will change;
     - the support space / field-of-view will NOT change.

    Args:
        arrs (Iterable[np.ndarray]): The input arrays,
        new_shape (Iterable[int]): The new base shape of the arrays.
        lossless (bool): allow for lossy resampling.
        window (int|Iterable[int]|None): Uniform pre-filter window size.
            This is the size of the window for the uniform filter using
            `scipy.ndimage.uniform_filter()`.
            If Iterable, its size must match the number of dims of `arr`.
            If int, uses an isotropic window with the specified size.
            If None, the window is calculated automatically from `new_shape`.
        interp_order (int|None): Order of the spline interpolation.
            0: nearest. Accepted range: [0, 5].
        extra_dim (bool): Force extra dimensions in the zoom parameters.
        fill_dim (bool): Dimensions not specified are left untouched.
        dtype (data-type): Desired output data-type.
            If None, its guessed from dtype of arrs.
            See `np.ndarray()` for more.

    Returns:
        result (np.ndarray): The output array.
            It contains all reshaped arrays from `arrs`, through the last dim.
            The shape of this array is `new_shape` + `len(arrs)`.
    """
    # calculate new shape
    if new_shape is None:
        shapes = [arr.shape for arr in arrs]
        new_shape = [1] * max(len(shape) for shape in shapes)
        shape_arr = np.ones((len(shapes), len(new_shape))).astype(np.int)
        for i, shape in enumerate(shapes):
            shape_arr[i, :len(shape)] = np.array(shape)
        combiner = fc.base.lcm if lossless else max
        new_shape = tuple(
            int(combiner(list(shape_arr[:, i])))
            for i in range(len(new_shape)))
    else:
        new_shape = tuple(new_shape)

    # resample images
    if lossless:
        interp_order = 0
        window = None

    if dtype is None:
        # dtype = functools.reduce(
        #     (lambda x, y: np.promote_types(x, y.dtype)), arrs)
        dtype = bool
        for arr in arrs:
            dtype = np.promote_types(dtype, arr.dtype)

    result = np.array(new_shape + (len(arrs),), dtype=dtype)
    for i, arr in enumerate(arrs):
        # ratio should not be kept: keep_ratio_method=None
        result[..., i] = resample(
            arr, new_shape, aspect=None, window=window,
            interp_order=interp_order, extra_dim=extra_dim, fill_dim=fill_dim)
    return result


# ======================================================================
def decode_affine(aff_mat):
    """
    Decompose the affine matrix into a linear transformation and a translation.

    Args:
        aff_mat (Sequence|np.ndarray): The affine matrix.
            This must be an `n + 1` by `m + 1` matrix.

    Returns:
        result (tuple): The tuple
            contains:
             - linear (np.ndarray): The linear transformation matrix.
               This must be a `n` by `m` matrix where `n` is the number of
               dims of the domain and `m` is the number of dims of the
               co-domain.
             - off_vec (np.ndarray): The offset vector in px.
                This must be a `m` sized array.
    """
    aff_mat = np.ndarray(aff_mat)
    assert (aff_mat.ndim == 2)
    lin_mat = aff_mat[:aff_mat.shape[0] - 1, :aff_mat.shape[1] - 1]
    off_vec = aff_mat[:-1, -1]
    return lin_mat, off_vec


# ======================================================================
def encode_affine(
        lin_mat,
        off_vec):
    """
    Combine a linear transformation and a translation into the affine matrix.

    Args:
        lin_mat (Sequence|np.ndarray): The linear transformation matrix.
            This must be a `n` by `m` matrix where `n` is the number of
            dims of the domain and `m` is the number of dims of the co-domain.
        off_vec (Sequence|np.ndarray): The offset vector in px.
            This must be a `m` sized array.

    Returns:
        affine (np.ndarray): The affine matrix.
            This is an `n + 1` by `m + 1` matrix.
    """
    lin_mat = np.array(lin_mat)
    off_vec = np.array(off_vec)
    assert (lin_mat.ndim == 2)
    assert (lin_mat.shape[1] == off_vec.size)
    affine = np.eye(lin_mat.shape[0] + 1)
    affine[:lin_mat.shape[0], :lin_mat.shape[1]] = lin_mat
    affine[:-1, -1] = off_vec
    return affine


# ======================================================================
def angles2rotation(
        angles,
        n_dim=None,
        axes_list=None,
        use_degree=True,
        atol=None):
    """
    Calculate the linear transformation relative to the specified rotations.

    Args:
        angles (Iterable[float]): The angles to be used for rotation.
        n_dim (int|None): The number of dimensions to consider.
            The number of angles and `n_dim` should satisfy the relation:
            `n_angles = n_dim * (n_dim - 1) / 2`.
            If `len(angles)` is smaller than expected for a given `n_dim`,
            the remaining angles are set to 0.
            If `len(angles)` is larger than expected, the exceeding `angles`
            are ignored.
            If None, n_dim is computed from `len(angles)`.
        axes_list (Iterable[Iterable[int]]|None): The axes of rotation planes.
            If not None, for each rotation angle a pair of axes
            (i.e. a size-2 iterable of int) must be specified to define the
            associated plane of rotation.
            The number of size-2 iterables should match the number of angles
            `len(angles) == len(axes_list)`.
            If `len(angles) < len(axes_list)` or `len(angles) > len(axes_list)`
            the unspecified rotations are not performed.
            If None, generates `axes_list` using the output of
            `itertools.combinations(range(n_dim), 2)`.
        use_degree (bool): Interpret angles as expressed in degree.
            Otherwise, use radians.
        atol (float|None): Absolute tolerance in the approximation.
            If error tolerance is exceded, a warning is issued.
            If float, the specified number is used as threshold.
            If None, a threshold is computed based on the size of the linear
            transformation matrix: `dim ** 4 * np.finfo(np.double).eps`.

    Returns:
        lin_mat (np.ndarray): The rotation matrix as defined by the angles.

    See Also:
        - flyingcircus.extra.square_size_to_num_tria(),
        - flyingcircus.extra.num_tria_to_square_size(),
        - itertools.combinations()

    Examples:
        >>> print(np.round(angles2rotation([90.0]), 6))
        [[ 0. -1.]
         [ 1.  0.]]
        >>> print(np.round(angles2rotation([-30.0]), 3))
        [[ 0.866  0.5  ]
         [-0.5    0.866]]
        >>> print(np.round(angles2rotation([30.0]), 3))
        [[ 0.866 -0.5  ]
         [ 0.5    0.866]]
        >>> print(np.round(angles2rotation([30.0, 0.0, -30.0]), 3))
        [[ 0.866 -0.433 -0.25 ]
         [ 0.5    0.75   0.433]
         [ 0.    -0.5    0.866]]
        >>> print(np.round(angles2rotation([30.0, -30.0, 0.0]), 3))
        [[ 0.75  -0.5    0.433]
         [ 0.433  0.866  0.25 ]
         [-0.5    0.     0.866]]
        >>> print(np.round(angles2rotation([30.0, 45.0, -30.0]), 3))
        [[ 0.612 -0.127 -0.78 ]
         [ 0.354  0.927  0.127]
         [ 0.707 -0.354  0.612]]
        >>> print(np.round(angles2rotation([30.0], 3), 3))
        [[ 0.866 -0.5    0.   ]
         [ 0.5    0.866  0.   ]
         [ 0.     0.     1.   ]]
    """
    if n_dim is None:
        # this is the number of cartesian orthogonal planes of rotations
        # defining: N the number of dimensions and n the number of angles
        # this is given by solving: N = n! / 2! / (n - 2)!
        # the equation simplifies to: N = n * (n - 1) / 2
        n_dim = num_tria_to_square_size(len(angles))
    if not axes_list:
        axes_list = list(itertools.combinations(range(n_dim), 2))
    lin_mat = np.eye(n_dim).astype(np.double)
    for angle, axes in zip(angles, axes_list):
        if use_degree:
            angle = np.deg2rad(angle)
        rotation = np.eye(n_dim)
        rotation[axes[0], axes[0]] = np.cos(angle)
        rotation[axes[1], axes[1]] = np.cos(angle)
        rotation[axes[0], axes[1]] = -np.sin(angle)
        rotation[axes[1], axes[0]] = np.sin(angle)
        lin_mat = np.dot(lin_mat, rotation)
    # : check that this is a rotation matrix
    det = np.linalg.det(lin_mat)
    if not atol:
        atol = lin_mat.ndim ** 4 * np.finfo(np.double).eps
    if np.abs(det) - 1.0 > atol:
        text = 'rotation matrix may be inaccurate [det = {}]'.format(repr(det))
        warnings.warn(text)
    return lin_mat


# ======================================================================
def rotation2angles(
        lin_mat,
        axes_list=None,
        use_degree=True):
    """
    Compute the rotation angles compatible with the proposed linear.

    The decomposition is not, in general, unique.
    The algorithm may fail if any of the angles is half the full circle
    (i.e. pi rad or 180 deg).
    If the transformation is 2D and the angle is 180 deg, it will fail.
    For higher dimensions, a solution without the 180 deg rotation may be
    found (if exists).

    Args:
        lin_mat (np.ndarray): The rotation matrix.
        axes_list (Iterable[Iterable[int]]|None): The axes of rotation planes.
            See `flyingcircus.extra.angles2rotation()` for more details.
        use_degree (bool): Interpret angles as expressed in degree.
            Otherwise, use radians.

    Returns:
        angles (tuple[float]): The angles generating the rotation matrix.
            Units are degree if `use_degre == True` else they are radians.

    Raises:
        ValueError: If the estimated angles do not match the rotation matrix.

    Examples:
        >>> angles = [90.0]
        >>> lin_mat = angles2rotation(angles)
        >>> est_angles = rotation2angles(lin_mat)
        >>> print([round(x, 3) for x in est_angles])
        [90.0]
        >>> print(np.all(np.isclose(angles, est_angles)))
        True

        >>> angles = [30.0]
        >>> lin_mat = angles2rotation(angles)
        >>> est_angles = rotation2angles(lin_mat)
        >>> print([round(x, 3) for x in est_angles])
        [30.0]
        >>> print(np.all(np.isclose(angles, est_angles)))
        True

        >>> angles = [270.0]
        >>> lin_mat = angles2rotation(angles)
        >>> est_angles = rotation2angles(lin_mat)
        >>> print([round(x, 3) for x in est_angles])
        [-90.0]
        >>> print(np.all(np.isclose(angles, np.array(est_angles) % 360)))
        True

        >>> # Fails for angle == 180.0
        >>> angles = [180.0]
        >>> lin_mat = angles2rotation(angles)
        >>> est_angles = rotation2angles(lin_mat)
        Traceback (most recent call last):
            ...
        ValueError: Rotation angles estimation failed! One angle is 180 deg?

        >>> angles = [30.0, 0.0, -30.0]
        >>> lin_mat = angles2rotation(angles)
        >>> est_angles = rotation2angles(lin_mat)
        >>> print([round(x, 3) for x in est_angles])
        [30.0, -0.0, -30.0]
        >>> print(np.all(np.isclose(angles, est_angles)))
        True

        >>> angles = [30.0, 45.0, -30.0]
        >>> lin_mat = angles2rotation(angles)
        >>> est_angles = rotation2angles(lin_mat)
        >>> print([round(x, 3) for x in est_angles])
        [30.0, 45.0, -30.0]
        >>> print(np.all(np.isclose(angles, est_angles)))
        True

        >>> angles = [15.0, -20.0, 75.0]
        >>> lin_mat = angles2rotation(angles)
        >>> est_angles = rotation2angles(lin_mat)
        >>> print([round(x, 3) for x in est_angles])
        [15.0, -20.0, 75.0]
        >>> print(np.all(np.isclose(angles, est_angles)))
        True

        >>> angles = [15.0, 180.0, 75.0]
        >>> lin_mat = angles2rotation(angles)
        >>> est_angles = rotation2angles(lin_mat)
        >>> print([round(x, 3) for x in est_angles])
        [-165.0, 0.0, -105.0]
        >>> # not the same as the input, but the rotation matrix is the same
        >>> print(np.all(np.isclose(angles, est_angles)))
        False
        >>> print(np.all(np.isclose(angles2rotation(est_angles), lin_mat)))
        True

        >>> angles = [180.0, 180.0, 0.0]
        >>> lin_mat = angles2rotation(angles)
        >>> est_angles = rotation2angles(lin_mat)
        Traceback (most recent call last):
            ...
        ValueError: Rotation angles estimation failed! One angle is 180 deg?

        >>> angles = [180.0, 180.0, 180.0]
        >>> lin_mat = angles2rotation(angles)
        >>> est_angles = rotation2angles(lin_mat)
        >>> print([round(x, 3) for x in est_angles])
        [-0.0, 0.0, -0.0]
        >>> # not the same as the input, but the rotation matrix is the same
        >>> print(np.all(np.isclose(angles, est_angles)))
        False
        >>> print(np.all(np.isclose(angles2rotation(est_angles), lin_mat)))
        True

        >>> result = []
        >>> for x in np.arange(360):
        ...     if x == 180: continue  # fails for this value!
        ...     angles = [x]
        ...     lin_mat = angles2rotation(angles)
        ...     est_angles = rotation2angles(lin_mat)
        ...     res = np.all(np.isclose(angles, np.array(est_angles) % 360))
        ...     if not res: print(angles, est_angles)
        ...     result.append(res)
        >>> print(np.all(result))
        True
    """
    assert (lin_mat.shape[0] == lin_mat.shape[1] and lin_mat.ndim == 2)
    n_dim = lin_mat.shape[0]
    n_angles = square_size_to_num_tria(n_dim)
    angles = np.zeros(n_angles)
    res = sp.optimize.root(
        lambda x: np.ravel(
            fc.extra.angles2rotation(
                x, n_dim=n_dim, axes_list=axes_list, use_degree=use_degree)
            - lin_mat),
        angles, method='lm')
    result = res.x
    est_lin_mat = fc.extra.angles2rotation(
        result, n_dim=n_dim, axes_list=axes_list, use_degree=use_degree)
    if np.all(np.isclose(lin_mat, est_lin_mat)):
        return result.tolist()
    else:
        raise ValueError(
            'Rotation angles estimation failed! One angle is 180 deg?')


# ======================================================================
def prepare_affine(
        shape,
        lin_mat,
        off_vec=None,
        origin=None):
    """
    Prepare parameters to be used with `scipy.ndimage.affine_transform()`.

    In particular, it computes the linear matrix and the offset implementing
    an affine transformation (applied at the origin) followed by a translation
    on the array coordinates.

    The result can be passed directly as `matrix` and `offset` parameters of
    `scipy.ndimage.affine_transform()`.
    Using `scipy.ndimage.affine_transform()` naming conventions,
    `matrix`/`i_lin_mat` and `i_vec_off` describe the *pull* transform,
    while `lin_mat` and `off_vec` describe the *push* tranform.

    Args:
        shape (Iterable): The shape of the array to be transformed.
        lin_mat (np.ndarray): The N-sized linear square matrix.
        off_vec (np.ndarray|None): The offset vector in px.
            If None, no shift is performed.
        origin (np.ndarray|None): The origin of the linear transformation.
            If None, uses the center of the array.

    Returns:
        result (tuple): The tuple
            contains:
             - i_lin_mat (np.ndarray): The N-sized linear square matrix.
             - i_off_vec (np.ndarray): The offset vector in px.

    See Also:
        - scipy.ndimage.affine_transform()
    """
    ndim = len(shape)
    if off_vec is None:
        off_vec = 0
    if origin is None:
        origin = np.array(coord(shape, (0.5,) * ndim, use_int=False))
    i_lin_mat = np.linalg.pinv(lin_mat)
    i_off_vec = origin - np.dot(i_lin_mat, origin + off_vec)
    return i_lin_mat, i_off_vec


# ======================================================================
def weighted_center(
        arr,
        labels=None,
        index=None):
    """
    Determine the weighted mean of the rendered objects inside an array.

    .. math::
        \\sum_i w_i (\\vec{x}_i - \\vec{o}_i)

    for i spanning through all support space.

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|Iterable[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.

    Returns:
        center (np.ndarray): The coordinates of the weighed center.

    See Also:
        - flyingcircus.extra.tensor_of_inertia()
        - flyingcircus.extra.rotatio_axes()
        - flyingcircus.extra.auto_rotation()
        - flyingcircus.extra.realigning()
    """
    # numpy.double to improve the accuracy of the norm and the weighted center
    arr = arr.astype(np.double)
    norm = sp.ndimage.sum(arr, labels, index)
    grid = np.ogrid[[slice(0, i) for i in arr.shape]]
    # numpy.double to improve the accuracy of the result
    center = np.zeros(arr.ndim).astype(np.double)
    for i in range(arr.ndim):
        center[i] = sp.ndimage.sum(arr * grid[i], labels, index) / norm
    return center


# ======================================================================
def weighted_covariance(
        arr,
        labels=None,
        index=None,
        origin=None):
    """
    Determine the weighted covariance matrix with respect to the origin.

    .. math::
        \\sum_i w_i (\\vec{x}_i - \\vec{o}) (\\vec{x}_i - \\vec{o})^T

    for i spanning through all support space, where:
    o is the origin vector,
    x_i is the coordinate vector of the point i,
    w_i is the weight, i.e. the value of the array at that coordinate.

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|Iterable[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.
        origin (np.ndarray|None): The origin for the covariance matrix.
            If None, the weighted center is used.

    Returns:
        cov (np.ndarray): The covariance weight matrix from the origin.

    See Also:
        - flyingcircus.extra.tensor_of_inertia
        - flyingcircus.extra.rotation_axes
        - flyingcircus.extra.auto_rotation
        - flyingcircus.extra.realigning
    """
    # numpy.double to improve the accuracy of the norm and the weighted center
    arr = arr.astype(np.double)
    norm = sp.ndimage.sum(arr, labels, index)
    if origin is None:
        origin = np.array(sp.ndimage.center_of_mass(arr, labels, index))
    grid = np.ogrid[[slice(0, i) for i in arr.shape]] - origin
    # numpy.double to improve the accuracy of the result
    cov = np.zeros((arr.ndim, arr.ndim)).astype(np.double)
    for i in range(arr.ndim):
        for j in range(arr.ndim):
            if i <= j:
                cov[i, j] = sp.ndimage.sum(
                    arr * grid[i] * grid[j], labels, index) / norm
            else:
                # the covariance weight matrix is symmetric
                cov[i, j] = cov[j, i]
    return cov


# ======================================================================
def tensor_of_inertia(
        arr,
        labels=None,
        index=None,
        origin=None):
    """
    Determine the tensor of inertia with respect to the origin.

    I = Id * tr(C) - C

    where:
    C is the weighted covariance matrix,
    Id is the identity matrix.

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|Iterable[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.
        origin (np.ndarray|None): The origin for the covariance matrix.
            If None, the weighted center is used.

    Returns:
        inertia (np.ndarray): The tensor of inertia from the origin.

    See Also:
        - flyingcircus.extra.weighted_covariance()
        - flyingcircus.extra.rotation_axes()
        - flyingcircus.extra.auto_rotation()
        - flyingcircus.extra.realigning()
    """
    cov = weighted_covariance(arr, labels, index, origin)
    inertia = np.eye(arr.ndim) * np.trace(cov) - cov
    return inertia


# ======================================================================
def rotation_axes(
        arr,
        labels=None,
        index=None,
        sort_by_shape=False):
    """
    Calculate the principal axes of rotation.

    These can be found as the eigenvectors of the tensor of inertia.

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|Iterable[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.
        sort_by_shape (bool): Sort the axes by the array shape.
            This is useful in order to obtain the optimal rotations to
            align the objects to the shape.
            Otherwise, it is sorted by increasing eigenvalues.

    Returns:
        axes (list[np.ndarray]): The principal axes of rotation.

    See Also:
        - flyingcircus.extra.weighted_covariance()
        - flyingcircus.extra.tensor_of_inertia()
        - flyingcircus.extra.auto_rotation()
        - flyingcircus.extra.realigning()
    """
    # calculate the tensor of inertia with respect to the weighted center
    inertia = tensor_of_inertia(arr, labels, index, None).astype(np.double)
    # numpy.linalg only supports up to numpy.double
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    if sort_by_shape:
        tmp = [
            (size, eigenvalue, eigenvector)
            for size, eigenvalue, eigenvector
            in zip(
                sorted(arr.shape, reverse=True),
                eigenvalues,
                tuple(eigenvectors.transpose()))]
        tmp = sorted(tmp, key=lambda x: arr.shape.index(x[0]))
        axes = []
        for size, eigenvalue, eigenvector in tmp:
            axes.append(eigenvector)
    else:
        axes = [axis for axis in eigenvectors.transpose()]
    return axes


# ======================================================================
def rotation_axes_to_matrix(axes):
    """
    Compute the rotation matrix from the principal axes of rotation.

    This matrix describes the linear transformation required to bring the
    principal axes of rotation along the axes of the canonical basis.

    Args:
        axes (Iterable[np.ndarray]): The principal axes of rotation.

    Returns:
        lin_mat (np.ndarray): The linear transformation matrix.
    """
    return np.array(axes).transpose()


# ======================================================================
def auto_rotation(
        arr,
        labels=None,
        index=None,
        origin=None):
    """
    Compute the linear transformation and shift for optimal rotation.

    The principal axis of rotation will be parallel to the cartesian axes.

    The result can be passed directly as `matrix` and `offset` parameters of
    `scipy.ndimage.affine_transform()`.

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|Iterable[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.
        origin (np.ndarray|None): The origin for the covariance matrix.
            If None, the weighted center is used.

    Returns:
        i_lin_mat (np.ndarray): The linear matrix for the rotation.
        i_vec_off (np.ndarray): The offset for the translation.

    See Also:
        - scipy.ndimage.center_of_mass()
        - scipy.ndimage.affine_transform()
        - flyingcircus.extra.weighted_covariance()
        - flyingcircus.extra.tensor_of_inertia()
        - flyingcircus.extra.rotation_axes()
        - flyingcircus.extra.angles2linear()
        - flyingcircus.extra.linear2angles()
        - flyingcircus.extra.auto_rotation()
        - flyingcircus.extra.realigning()
    """
    lin_mat = rotation_axes_to_matrix(rotation_axes(arr, labels, index, True))
    return prepare_affine(arr.shape, lin_mat, origin=origin)


# ======================================================================
def auto_shifting(
        arr,
        labels=None,
        index=None,
        origin=None):
    """
    Compute the linear transformation and shift for optimal shifting.

    Weighted center will be at a given point (e.g. the middle of the support).

    The result can be passed directly as `matrix` and `offset` parameters of
    `scipy.ndimage.affine_transform()`.

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|Iterable[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.
        origin (np.ndarray|None): The origin for the covariance matrix.
            If None, the weighted center is used.

    Returns:
        i_lin_mat (np.ndarray): The linear matrix for the rotation.
        i_off_vec (np.ndarray): The offset for the translation.

    See Also:
        - scipy.ndimage.center_of_mass()
        - scipy.ndimage.affine_transform()
        - flyingcircus.extra.weighted_covariance()
        - flyingcircus.extra.tensor_of_inertia()
        - flyingcircus.extra.rotation_axes()
        - flyingcircus.extra.angles2linear()
        - flyingcircus.extra.linear2angles()
        - flyingcircus.extra.auto_rotation()
        - flyingcircus.extra.realigning()
    """
    lin_mat = np.eye(arr.ndim)
    com = np.array(sp.ndimage.center_of_mass(arr, labels, index))
    return prepare_affine(arr.shape, lin_mat, com, origin)


# ======================================================================
def realigning(
        arr,
        labels=None,
        index=None,
        origin=None):
    """
    Compute the linear transformation and shift for optimal grid alignment.

    The principal axis of rotation will be parallel to the cartesian axes.
    Weighted center will be at a given point (e.g. the middle of the support).

    The result can be passed directly as `matrix` and `offset` parameters of
    `scipy.ndimage.affine_transform()`.

    Args:
        arr (np.ndarray): The input array.
        labels (np.ndarray|None): Cumulative mask array for the objects.
            The output of `scipy.ndimage.label` is expected.
            The number of dimensions must be the same as `array`.
            Only uses the labels as indicated by `index`.
        index (int|Iterable[int]|None): Labels used for the calculation.
            If an int, uses all labels between 1 and the specified value.
            If a tuple of int, uses only the selected labels.
            If None, uses all positive labels.
        origin (np.ndarray|None): The origin for the covariance matrix.
            If None, the weighted center is used.

    Returns:
        i_lin_mat (np.ndarray): The linear matrix for the rotation.
        i_off_vec (np.ndarray): The offset for the translation.

    See Also:
        - scipy.ndimage.center_of_mass()
        - scipy.ndimage.affine_transform()
        - flyingcircus.extra.weighted_covariance()
        - flyingcircus.extra.tensor_of_inertia()
        - flyingcircus.extra.rotation_axes()
        - flyingcircus.extra.angles2linear()
        - flyingcircus.extra.linear2angles()
        - flyingcircus.extra.auto_rotation()
        - flyingcircus.extra.auto_shift()
    """
    com = np.array(sp.ndimage.center_of_mass(arr, labels, index))
    lin_mat = rotation_axes_to_matrix(rotation_axes(arr, labels, index, True))
    return prepare_affine(arr.shape, lin_mat, com, origin)


# ======================================================================
def rotation_3d_from_vector(
        normal,
        angle=None):
    """
    Compute the rotation matrix of a given angle around a specified vector.

    Args:
        normal (Sequence|np.ndarray): The vector around which to rotate.
            If Iterable or np.ndarray must have a length of 3.
            If its norm is 0, the identity matrix is returned.
        angle (int|float|None): The angle of rotation in deg.
            If None, the angle is inferred from the norm of the normal vector.

    Returns:
        rot_mat (np.ndarray): The rotation matrix.

    Examples:
        >>> rot_mat = rotation_3d_from_vector([1, 0, 0])
        >>> np.round(rot_mat, 3)
        array([[ 1.,  0.,  0.],
               [ 0.,  0., -1.],
               [ 0.,  1.,  0.]])
        >>> rot_mat = rotation_3d_from_vector([1, 1, 0])
        >>> np.round(rot_mat, 3)
        array([[ 0.955,  0.045,  0.293],
               [ 0.045,  0.955, -0.293],
               [-0.293,  0.293,  0.91 ]])
        >>> rot_mat = rotation_3d_from_vector([1, 1, 0], 90)
        >>> np.round(rot_mat, 3)
        array([[ 0.5  ,  0.5  ,  0.707],
               [ 0.5  ,  0.5  , -0.707],
               [-0.707,  0.707,  0.   ]])
        >>> rot_mat = rotation_3d_from_vector([0, 1, 0], 30)
        >>> np.round(rot_mat, 3)
        array([[ 0.866,  0.   ,  0.5  ],
               [ 0.   ,  1.   ,  0.   ],
               [-0.5  ,  0.   ,  0.866]])
        >>> rot_mat = rotation_3d_from_vector([1, 0, 0], 0.0)
        >>> np.round(rot_mat, 3)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
        >>> rot_mat = rotation_3d_from_vector([0, 0, 0], 90.0)
        >>> np.round(rot_mat, 3)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
        >>> rot_mat = rotation_3d_from_vector([0, 0, 0], 0.0)
        >>> np.round(rot_mat, 3)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
    """
    rot_mat = np.eye(3)
    norm = np.linalg.norm(normal)
    if norm:
        normal = np.array(normal) / norm
        angle = np.arcsin((norm % 1.0) if norm > 1.0 else norm) \
            if angle is None else np.deg2rad(angle)
        if angle != 0.0:
            signs = np.array([-1., 1., -1.])
            v_matrix = to_self_adjoint_matrix(normal[::-1] * signs, skew=True)
            w_matrix = np.outer(normal, normal)
            rot_mat = \
                rot_mat * np.cos(angle) + v_matrix * np.sin(angle) + \
                w_matrix * (1 - np.cos(angle))
    return rot_mat


# ======================================================================
def rotation_3d_from_vectors(
        vector1,
        vector2):
    """
    Compute the rotation matrix required to move one vector onto one other.

    Given two vectors :math:`\\vec{v}_1` and :math:`\\vec{v}_2` computes the
    rotation matrix :math:`R` such that: :math:`\\vec{v}_2 = R \\vec{v}_1`.

    Args:
        vector1 (Sequence|np.ndarray): The first vector.
            If Iterable or np.ndarray must have a length of 3.
        vector2 (Sequence|np.ndarray): The second vector.
            If Iterable or np.ndarray must have a length of 3.

    Returns:
        rot_matrix (np.ndarray): The rotation matrix.

    Examples:
        >>> vector1 = np.array([0, 1, 0])
        >>> vector2 = np.array([0, 0, 1])
        >>> rot_matrix12 = rotation_3d_from_vectors(vector1, vector2)
        >>> np.round(rot_matrix12, 6)
        array([[ 1.,  0.,  0.],
               [ 0.,  0., -1.],
               [ 0.,  1.,  0.]])
        >>> np.all(np.isclose(vector2, np.dot(rot_matrix12, vector1)))
        True
        >>> normal21 = np.cross(normalize(vector2), normalize(vector1))
        >>> rot_matrix21 = rotation_3d_from_vectors(vector2, vector1)
        >>> np.all(np.isclose(vector1, np.dot(rot_matrix21, vector2)))
        True
        >>> np.all(np.isclose(np.eye(3), np.dot(rot_matrix12, rot_matrix21)))
        True
    """
    normal = np.cross(normalize(vector1), normalize(vector2))
    return rotation_3d_from_vector(normal)


# ======================================================================
def random_mask(
        shape,
        density=0.5,
        dtype=bool):
    """
    Calculate a randomly distributed mask of specified density.

    Args:
        shape (Iterable[int]): The target array shape.
        density (float): The density of the mask.
            Must be in the (0, 1) interval.
        dtype (np.dtype): The data type of the resulting array.

    Returns:
        mask (np.ndarray[bool]): The rendered geometrical object.

    Examples:
        >>> import numpy as np
        >>> random.seed(0)
        >>> print(random_mask((2, 5), 0.5))
        [[ True False  True False  True]
         [False  True False False  True]]
    """
    size = fc.base.prod(shape)
    if not 0 < density < 1:
        raise ValueError('Density must be between 0 and 1')
    mask = np.zeros(shape, dtype=dtype).ravel()
    mask[random.sample(range(size), int(size * density))] = True
    return mask.reshape(shape)


# ======================================================================
def fibonacci_disk(num):
    """
    Generate the Fibonacci disc.

    There points are (weakly) uniformly distributed on the surface of a disc
    of radius 1.
    This is obtained by placing the points in a Fibonacci spiral.

    Args:
        num (int): The number of points to generate.

    Returns:
        arr (np.ndarray[float]): The coordinates of the points.
            The array has shape `(2, num)`.

    Examples:
        >>> np.round(fibonacci_disk(6), 3)
        array([[ 0.105, -0.448,  0.62 , -0.397, -0.168,  0.772],
               [-0.269,  0.221,  0.18 , -0.653,  0.849, -0.567]])
        >>> np.round(fibonacci_disk(8), 2)
        array([[ 0.09, -0.39,  0.54, -0.34, -0.15,  0.67, -0.9 ,  0.64],
               [-0.23,  0.19,  0.16, -0.57,  0.74, -0.49, -0.1 ,  0.73]])
    """
    arr = np.empty((2, num))
    i = np.arange(num, dtype=float) + 0.5
    arr[0] = np.cos(np.pi * (1 + 5 ** 0.5) * i)
    arr[1] = np.sin(np.pi * (1 + 5 ** 0.5) * i)
    return arr * np.sqrt(i / num)


# ======================================================================
def fibonacci_sphere(num):
    """
    Generate the Fibonacci sphere.

    These points are (weakly) uniformly distributed on the surface of a sphere
    of radius 1.
    This is obtained by placing the points in a Fibonacci spiral.

    Args:
        num (int): The number of points to generate.

    Returns:
        arr (np.ndarray[float]): The coordinates of the points.
            The array has shape `(3, num)`.

    Examples:
        >>> np.round(fibonacci_sphere(6), 3)
        array([[-0.833, -0.5  , -0.167,  0.167,  0.5  ,  0.833],
               [ 0.553, -0.639,  0.086,  0.6  , -0.853,  0.466],
               [ 0.   ,  0.585, -0.982,  0.783, -0.151, -0.297]])
        >>> np.round(fibonacci_sphere(8), 2)
        array([[-0.88, -0.62, -0.38, -0.12,  0.12,  0.38,  0.62,  0.88],
               [ 0.48, -0.58,  0.08,  0.6 , -0.98,  0.78, -0.2 , -0.22],
               [ 0.  ,  0.53, -0.92,  0.79, -0.17, -0.5 ,  0.75, -0.43]])
    """
    arr = np.empty((3, num))
    increment = np.pi * (3.0 - (5.0 ** 0.5))
    i = np.arange(num, dtype=float)
    arr[0] = i * (2.0 / num) + (1.0 / num - 1.0)
    arr[1] = np.cos(i * increment)
    arr[2] = np.sin(i * increment)
    arr[1:] *= np.sqrt(1 - arr[0] ** 2)
    return arr


# ======================================================================
def angles_in_ellipse(
        num,
        a=1,
        b=1,
        offset=0.0):
    """
    Generate the angles of (almost) equi-spaced (arc) points on an ellipse.

    Args:
        num (int): The number of points / angles.
        a (int|float): The 1st-dimension semi-axis of the ellipse.
        b (int|float): The 2nd-dimension semi-axis of the ellipse.
        offset (int|float): The angle offset in rad.

    Returns:
        angles (np.ndarray): The angles of the equi-spaced points in rad.

    Examples:
         >>> n = 8
         >>> a, b = 10, 20
         >>> e = (1.0 - a ** 2.0 / b ** 2.0) ** 0.5
         >>> phis = angles_in_ellipse(n, a, b)
         >>> arcs = sp.special.ellipeinc(phis, e)
         >>> np.round(np.diff(arcs), 3)
         array([0.566, 0.566, 0.566, 0.566, 0.566, 0.566, 0.566])
         >>> phis = angles_in_ellipse(n, a, b, np.deg2rad(10.0))
         >>> arcs = sp.special.ellipeinc(phis, e)
         >>> np.round(np.diff(arcs), 3)
         array([0.566, 0.566, 0.566, 0.566, 0.566, 0.566, 0.566])
         >>> phis = angles_in_ellipse(64, a, b)
         >>> arc_diffs = np.diff(sp.special.ellipeinc(phis, e))
         >>> np.round(np.mean(arc_diffs), 4), np.round(np.std(arc_diffs), 4)
         (0.0707, 0.0)
         >>> a, b = 20, 10
         >>> phis = angles_in_ellipse(n, a, b)
         >>> e = (1.0 - b ** 2.0 / a ** 2.0) ** 0.5
         >>> arcs = sp.special.ellipeinc(phis + (np.pi / 2.0), e)
         >>> np.round(np.diff(arcs), 3)
         array([0.566, 0.566, 0.566, 0.566, 0.566, 0.566, 0.566])
         >>> a, b = 10, 10
         >>> phis = angles_in_ellipse(n, a, b)
         >>> e = (1.0 - b ** 2.0 / a ** 2.0) ** 0.5
         >>> arcs = sp.special.ellipeinc(phis + (np.pi / 2.0), e)
         >>> np.round(np.diff(arcs), 3)
         array([0.785, 0.785, 0.785, 0.785, 0.785, 0.785, 0.785])
    """
    assert (num > 0)
    if a < b:
        a, b = b, a
        rot_offset = 0.0
    else:
        rot_offset = np.pi / 2.0
    angles = 2 * np.pi * np.arange(num) / num
    if a != b:
        e = (1.0 - b ** 2.0 / a ** 2.0) ** 0.5
        tot_size = sp.special.ellipeinc(2.0 * np.pi, e)
        arc_size = tot_size / num
        arcs = np.arange(num) * arc_size + sp.special.ellipeinc(offset, e)
        res = sp.optimize.root(
            lambda x: (sp.special.ellipeinc(x, e) - arcs), angles)
        angles = res.x
    return angles + rot_offset


# ======================================================================
def rolling_window_nd(
        arr,
        window,
        steps=1,
        window_steps=1,
        out_mode='view',
        pad_mode=0,
        pad_kws=None,
        writeable=False,
        shape_mode='end'):
    """
    Generate a N-dimensional rolling window of an array.

    Args:
        arr (np.ndarray): The input array.
        window (int|Iterable[int]): The window sizes.
        steps (int|Iterable[int]): The step sizes.
            This determines the step used for moving to the next window.
        window_steps (int|Iterable[int]): The window step sizes.
            This determines the step used for moving within the window.
        out_mode (str): The output mode.
            Can be one of:
            - 'valid': only values inside the array are used.
            - 'view': same as `valid`, but returns a view of the input instead.
            - 'same': must have the same size as the input.
            - 'full': the full output is provided.
        pad_mode (Number|str): The padding mode.
            If `out_mode` is `valid` or `view` this parameter is ignored.
            This is passed as `mode` to `flyingcircus.extra.padding()`.
        pad_kws (Mappable|None): Keyword parameters for padding.
            If `out_mode` is `valid` or `view` this parameter is ignored.
            This is passed as `mode` to `flyingcircus.extra.padding()`.
        writeable (bool): Determine if the result entries can be overwritten.
            This is passed to `flyingcircus.extra.nd_windowing()`.
        shape_mode (str): Determine the shape of the result.
            This is passed to `flyingcircus.extra.nd_windowing()`.

    Returns:
        result (np.ndarray): The windowing array.
    """
    pad_kws = dict(pad_kws) if pad_kws else {}
    window = fc.base.auto_repeat(window, arr.ndim, check=True)
    as_view = (out_mode == 'view')
    width = 0  # for both 'valid' and 'view' output modes
    if out_mode == 'same':
        width = tuple((size // 2, size - size // 2 - 1) for size in window)
    elif out_mode == 'full':
        width = tuple((size - 1, size - 1) for size in window)
    if width:
        arr, mask = padding(arr, width, mode=pad_mode, **pad_kws)
    return nd_windowing(
        arr, window, steps, window_steps, as_view, writeable, shape_mode)


# ======================================================================
def moving_mean(
        arr,
        num=1):
    """
    Calculate the moving mean.

    The moving average will be applied to the flattened array.
    Unless specified otherwise, the size of the array will be reduced by
    (num - 1).

    Args:
        arr (np.ndarray): The input array.
        num (int|Iterable): The running window size.
            The number of elements to group.

    Returns:
        arr (np.ndarray): The output array.

    Examples:
        >>> moving_mean(np.linspace(1, 9, 9), 1)
        array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
        >>> moving_mean(np.linspace(1, 8, 8), 1)
        array([1., 2., 3., 4., 5., 6., 7., 8.])
        >>> moving_mean(np.linspace(1, 9, 9), 2)
        array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
        >>> moving_mean(np.linspace(1, 8, 8), 2)
        array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
        >>> moving_mean(np.linspace(1, 9, 9), 5)
        array([3., 4., 5., 6., 7.])
        >>> moving_mean(np.linspace(1, 8, 8), 5)
        array([3., 4., 5., 6.])

    See Also:
        - flyingcircus.extra.moving_average()
        - flyingcircus.extra.moving_apply()
        - flyingcircus.extra.running_apply()
        - flyingcircus.extra.rolling_apply_nd()
    """
    arr = arr.ravel()
    arr = np.cumsum(arr)
    arr[num:] = arr[num:] - arr[:-num]
    arr = arr[num - 1:] / num
    return arr


# ======================================================================
def moving_average(
        arr,
        weights=1,
        **_kws):
    """
    Calculate the moving average (with optional weights).

    The moving average will be applied to the flattened array.
    Unless specified otherwise, the size of the array will be reduced by
    len(weights) - 1
    This is equivalent to passing `mode='valid'` to `scipy.signal.convolve()`.
    Please refer to `scipy.signal.convolve()` for more options.

    Args:
        arr (np.ndarray): The input array.
        weights (int|Iterable): The running weights.
            If int, the number of elements to group in the 'running' axis and
            unity weights are used.
            The size of the weights array len(weights) must be such that
            len(weights) >= 1 and len(weights) <= len(array), otherwise the
            flattened array is returned.
        **_kws: Keyword arguments for `scipy.signal.convolve()`.

    Returns:
        arr (np.ndarray): The output array.

    Examples:
        >>> moving_average(np.linspace(1, 9, 9), 1)
        array([1., 2., 3., 4., 5., 6., 7., 8., 9.])
        >>> moving_average(np.linspace(1, 8, 8), 1)
        array([1., 2., 3., 4., 5., 6., 7., 8.])
        >>> moving_average(np.linspace(1, 9, 9), 2)
        array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5])
        >>> moving_average(np.linspace(1, 8, 8), 2)
        array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
        >>> moving_average(np.linspace(1, 9, 9), 5)
        array([3., 4., 5., 6., 7.])
        >>> moving_average(np.linspace(1, 8, 8), 5)
        array([3., 4., 5., 6.])
        >>> moving_average(np.linspace(1, 8, 8), [1, 1, 1])
        array([2., 3., 4., 5., 6., 7.])
        >>> moving_average(np.linspace(1, 8, 8), [1, 0.2])
        array([1.16666667, 2.16666667, 3.16666667, 4.16666667, 5.16666667,
               6.16666667, 7.16666667])

    See Also:
        - flyingcircus.extra.moving_mean()
        - flyingcircus.extra.moving_apply()
        - flyingcircus.extra.running_apply()
        - flyingcircus.extra.rolling_apply_nd()
    """
    arr = arr.ravel()
    if isinstance(weights, int):
        weights = np.ones((weights,))
    else:
        # weights order needs to be inverted
        weights = np.array(weights)[::-1]
    num = len(weights) if isinstance(weights, np.ndarray) else 0
    if len(arr) >= num > 1:
        if 'mode' not in _kws:
            _kws['mode'] = 'valid'
        arr = sp.signal.convolve(arr, weights / len(weights), **_kws)
        arr *= len(weights) / np.sum(weights)
    return arr


# ======================================================================
def moving_apply(
        arr,
        weights=1,
        func=np.mean,
        args=None,
        kws=None,
        mode='valid',
        borders=None):
    """
    Compute a function on a moving (weighted) window of a 1D-array.

    This is especially useful for running/moving/rolling statistics.
    The function will be applied to the flattened array.
    This is calculated by running the specified statistics for each subset of
    the array of given size, including optional weightings.

    This function differs from `running_apply` in that it should be faster but
    more memory demanding.
    Also the `func` callable is required to accept an `axis` parameter.

    If the `weights` functionality is not required, then
    `flyingcircus.extra.rolling_apply_nd()` is a faster alternative.

    Args:
        arr (np.ndarray): The input array.
        weights (int|Iterable): The running weights.
            If int, the number of elements to group in the 'running' axis and
            unity weights are used.
            The size of the weights array len(weights) must be such that
            len(weights) >= 1 and len(weights) <= len(array), otherwise the
            flattened array is returned.
        func (callable): Function to calculate in the 'moving' window.
            Must accept an `axis` parameter, which will be set to -1.
        args (tuple|None): Positional arguments for `func`.
        kws (dict|None): Keyword arguments for `func`.
        mode (str): The output mode.
            Can be one of:
            - 'valid': only values inside the array are used.
            - 'same': must have the same size as the input.
            - 'full': the full output is provided.
        borders (str|complex|Iterable[complex]|None): The border parameters.
            If int or float, the value is repeated at the borders.
            If Iterable of int, float or complex, the first and last values are
            repeated to generate the head and tail, respectively.
            If str, the following values are accepted:
                - 'same': the array extrema are used to generate head / tail.
                - 'circ': the values are repeated periodically / circularly.
                - 'sym': the values are repeated periodically / symmetrically.

    Returns:
        arr (np.ndarray): The output array.

    Examples:
        >>> num = 8
        >>> arr = np.linspace(1, num, num)
        >>> all(np.allclose(
        ...                 moving_average(arr, n, mode=mode),
        ...                 moving_apply(arr, n, mode=mode))
        ...     for n in range(num) for mode in ('valid', 'same', 'full'))
        True
        >>> moving_apply(arr, 4, mode='same', borders=100)
        array([50.75, 26.5 ,  2.5 ,  3.5 ,  4.5 ,  5.5 ,  6.5 , 30.25])
        >>> moving_apply(arr, 4, mode='full', borders='same')
        array([1.  , 1.25, 1.75, 2.5 , 3.5 , 4.5 , 5.5 , 6.5 , 7.25, 7.75, 8.\
  ])
        >>> moving_apply(arr, 4, mode='full', borders='circ')
        array([5.5, 4.5, 3.5, 2.5, 3.5, 4.5, 5.5, 6.5, 5.5, 4.5, 3.5])
        >>> moving_apply(arr, 4, mode='full', borders='sym')
        array([1.75, 1.5 , 1.75, 2.5 , 3.5 , 4.5 , 5.5 , 6.5 , 7.25, 7.5 ,\
 7.25])
        >>> moving_apply(arr, 4, mode='same', borders='circ')
        array([4.5, 3.5, 2.5, 3.5, 4.5, 5.5, 6.5, 5.5])
        >>> moving_apply(arr, [1, 0.2])
        array([1.16666667, 2.16666667, 3.16666667, 4.16666667, 5.16666667,
               6.16666667, 7.16666667])

    See Also:
        - flyingcircus.extra.moving_mean()
        - flyingcircus.extra.moving_average()
        - flyingcircus.extra.running_apply()
        - flyingcircus.extra.rolling_apply_nd()
    """
    arr = arr.ravel()
    if isinstance(weights, int):
        weights = np.ones((weights,))
    else:
        # weights order needs to be inverted
        weights = np.array(weights)[::-1]
    num = len(weights) if isinstance(weights, np.ndarray) else 0
    size = len(arr)
    if size >= num > 1:
        # calculate how to extend the input array
        if borders is None:
            extension = np.zeros((num - 1,))
        elif borders == 'same':
            extension = np.concatenate(
                (np.full((num - 1,), arr[-1]),
                 np.full((num - 1,), arr[0])))
        elif borders == 'circ':
            extension = arr
        elif borders == 'sym':
            extension = arr[::-1]
        elif isinstance(borders, (int, float, complex)):
            extension = np.full((num - 1,), borders)
        elif isinstance(borders, (tuple, float)):
            extension = np.concatenate(
                (np.full((num - 1,), borders[-1]),
                 np.full((num - 1,), borders[0])))
        else:
            raise ValueError(fmtm('`borders={borders}` not understood'))

        # calculate generator for data and weights
        arr = np.concatenate((arr, extension))
        gen = np.zeros((size + num - 1, num))
        for i in range(num):
            gen[:, i] = np.roll(arr, i)[:size + num - 1]
        w_gen = np.stack([weights] * (size + num - 1))

        # calculate the running stats
        arr = func(
            gen * w_gen,
            *(args if args else ()), axis=-1,
            **(kws if kws else {}))
        arr *= len(weights) / np.sum(weights)

        # adjust output according to mode
        if mode == 'valid':
            arr = arr[num - 1:-(num - 1)]
        elif mode == 'same':
            begin = (num - 1) // 2
            arr = arr[begin:begin + size]
    return arr


# ======================================================================
def running_apply(
        arr,
        weights=1,
        func=np.mean,
        args=None,
        kws=None,
        mode='valid',
        borders=None):
    """
    Compute a function on a running (weighted) window of a 1D-array.

    This is especially useful for running/moving/rolling statistics.
    This is calculated by running the specified function for each subset of
    the array of given size, including optional weightings.
    The moving function will be applied to the flattened array.

    This function differs from `rolling_apply` in that it should be slower but
    less memory demanding.
    Also the `func` callable is not required to accept an `axis` parameter.

    Args:
        arr (np.ndarray): The input array.
        weights (int|Iterable): The running weights.
            If int, the number of elements to group in the 'running' axis and
            unity weights are used.
            The size of the weights array len(weights) must be such that
            len(weights) >= 1 and len(weights) <= len(array), otherwise the
            flattened array is returned.
        func (callable): Function to calculate in the 'running' axis.
        args (tuple|None): Positional arguments for `func()`.
        kws (dict|None): Keyword arguments for `func()`.
        mode (str): The output mode.
            Can be one of:
            - 'valid': only values inside the array are used.
            - 'same': must have the same size as the input.
            - 'full': the full output is provided.
        borders (str|complex|None): The border parameters.
            If int, float or complex, the value is repeated at the borders.
            If Iterable of int, float or complex, the first and last values are
            repeated to generate the head and tail, respectively.
            If str, the following values are accepted:
                - 'same': the array extrema are used to generate head / tail.
                - 'circ': the values are repeated periodically / circularly.
                - 'sym': the values are repeated periodically / symmetrically.

    Returns:
        arr (np.ndarray): The output array.

    Examples:
        >>> num = 8
        >>> arr = np.linspace(1, num, num)
        >>> all(np.allclose(
        ...                 moving_average(arr, n, mode=mode),
        ...                 running_apply(arr, n, mode=mode))
        ...     for n in range(num) for mode in ('valid', 'same', 'full'))
        True
        >>> running_apply(arr, 4, mode='same', borders=100)
        array([50.75, 26.5 ,  2.5 ,  3.5 ,  4.5 ,  5.5 ,  6.5 , 30.25])
        >>> running_apply(arr, 4, mode='same', borders='circ')
        array([4.5, 3.5, 2.5, 3.5, 4.5, 5.5, 6.5, 5.5])
        >>> running_apply(arr, 4, mode='full', borders='circ')
        array([5.5, 4.5, 3.5, 2.5, 3.5, 4.5, 5.5, 6.5, 5.5, 4.5, 3.5])
        >>> running_apply(arr, [1, 0.2])
        array([1.16666667, 2.16666667, 3.16666667, 4.16666667, 5.16666667,
               6.16666667, 7.16666667])

    See Also:
        - flyingcircus.extra.moving_mean()
        - flyingcircus.extra.moving_average()
        - flyingcircus.extra.moving_apply()
        - flyingcircus.extra.rolling_apply_nd()
    """
    arr = arr.ravel()
    if isinstance(weights, int):
        weights = np.ones((weights,))
    else:
        weights = np.array(weights)
    num = len(weights) if isinstance(weights, np.ndarray) else 0
    size = len(arr)
    if size >= num > 1:
        # calculate how to extend the input array
        if borders is None:
            head = tail = np.zeros((num - 1,))
        elif borders == 'same':
            head = np.full((num - 1,), arr[0])
            tail = np.full((num - 1,), arr[-1])
        elif borders == 'circ':
            tail = arr[:num - 1]
            head = arr[-num + 1:]
        elif borders == 'sym':
            tail = arr[-num + 1:]
            head = arr[:num - 1]
        elif isinstance(borders, (int, float, complex)):
            head = tail = np.full((num - 1,), borders)
        elif isinstance(borders, (tuple, float)):
            head = np.full((num - 1,), borders[0])
            tail = np.full((num - 1,), borders[-1])
        else:
            raise ValueError(fmtm('`borders={borders}` not understood'))

        # calculate generator for data and weights
        gen = np.concatenate((head, arr, tail))
        # print(gen)
        arr = np.zeros((len(gen) - num + 1))
        for i in range(len(arr)):
            arr[i] = func(
                gen[i:i + num] * weights,
                *(args if args else ()),
                **(kws if kws else {}))
        arr *= len(weights) / np.sum(weights)

        # adjust output according to mode
        if mode == 'valid':
            arr = arr[num - 1:-(num - 1)]
        elif mode == 'same':
            begin = (num - 1) // 2
            arr = arr[begin:begin + size]
    return arr


# ======================================================================
def rolling_apply_nd(
        arr,
        window,
        steps=1,
        window_steps=1,
        out_mode='valid',
        pad_mode=0,
        pad_kws=None,
        func=np.mean,
        args=None,
        kws=None):
    """
    Compute a function on a rolling N-dim window of the array.

    This is especially useful for running/moving/rolling statistics.
    Partial application along given axes can be obtained by setting a window
    size different from 1 only into the corresponding dimensions of interest.

    Args:
        arr (np.ndarray): The input array.
        window (int|Iterable[int]): The window sizes.
            This is passed to `flyingcircus.extra.rolling_window_nd()`.
        steps (int|Iterable[int]): The step sizes.
            This is passed to `flyingcircus.extra.rolling_window_nd()`.
        window_steps (int|Iterable[int]): The window step sizes.
            This is passed to `flyingcircus.extra.rolling_window_nd()`.
        out_mode (str): The output mode.
            This is passed to `flyingcircus.extra.rolling_window_nd()`.
            Note that `view` is treated identical to `valid`.
        pad_mode (Number|str): The padding mode.
            This is passed to `flyingcircus.extra.rolling_window_nd()`.
        pad_kws (dict|Iterable[Iterable]): Keyword parameters for padding.
            This is passed to `flyingcircus.extra.rolling_window_nd()`.
        func (callable): The function to apply.
            Must have the following signature:
             func(np.ndarray, axis=int|Sequence[int], *_args, **_kws)
              -> np.ndarray
            The result of `func` must be an array with the dimensions specified
            in `axis` collapsed, e.g. `np.mean()`, `np.min()`, `np.max()`, etc.
            Note that the `axis` parameter must be accepted.
        args (tuple|None): Positional arguments of `func`.
        kws (dict|None): Keyword arguments of `func`.

    Returns:
        result (np.ndarray): The rolling function.

    Examples:
        >>> arr = arange_nd((5, 7)) + 1
        >>> print(arr)
        [[ 1  2  3  4  5  6  7]
         [ 8  9 10 11 12 13 14]
         [15 16 17 18 19 20 21]
         [22 23 24 25 26 27 28]
         [29 30 31 32 33 34 35]]
        >>> print(rolling_apply_nd(arr, 2))
        [[ 5.  6.  7.  8.  9. 10.]
         [12. 13. 14. 15. 16. 17.]
         [19. 20. 21. 22. 23. 24.]
         [26. 27. 28. 29. 30. 31.]]
        >>> print(rolling_apply_nd(arr, 3))
        [[ 9. 10. 11. 12. 13.]
         [16. 17. 18. 19. 20.]
         [23. 24. 25. 26. 27.]]
        >>> print(np.round(rolling_apply_nd(arr, 2, out_mode='same'), 2))
        [[ 0.25  0.75  1.25  1.75  2.25  2.75  3.25]
         [ 2.25  5.    6.    7.    8.    9.   10.  ]
         [ 5.75 12.   13.   14.   15.   16.   17.  ]
         [ 9.25 19.   20.   21.   22.   23.   24.  ]
         [12.75 26.   27.   28.   29.   30.   31.  ]]
        >>> print(np.round(rolling_apply_nd(arr, 3, out_mode='same'), 2))
        [[ 2.22  3.67  4.33  5.    5.67  6.33  4.44]
         [ 5.67  9.   10.   11.   12.   13.    9.  ]
         [10.33 16.   17.   18.   19.   20.   13.67]
         [15.   23.   24.   25.   26.   27.   18.33]
         [11.56 17.67 18.33 19.   19.67 20.33 13.78]]
        >>> print(np.round(rolling_apply_nd(arr, 2, out_mode='full'), 2))
        [[ 0.25  0.75  1.25  1.75  2.25  2.75  3.25  1.75]
         [ 2.25  5.    6.    7.    8.    9.   10.    5.25]
         [ 5.75 12.   13.   14.   15.   16.   17.    8.75]
         [ 9.25 19.   20.   21.   22.   23.   24.   12.25]
         [12.75 26.   27.   28.   29.   30.   31.   15.75]
         [ 7.25 14.75 15.25 15.75 16.25 16.75 17.25  8.75]]
        >>> a = rolling_apply_nd(arr, 2, out_mode='full', pad_mode='wrap')
        >>> print(np.round(a, 2))
        [[18.  15.5 16.5 17.5 18.5 19.5 20.5 18. ]
         [ 7.5  5.   6.   7.   8.   9.  10.   7.5]
         [14.5 12.  13.  14.  15.  16.  17.  14.5]
         [21.5 19.  20.  21.  22.  23.  24.  21.5]
         [28.5 26.  27.  28.  29.  30.  31.  28.5]
         [18.  15.5 16.5 17.5 18.5 19.5 20.5 18. ]]
        >>> a = rolling_apply_nd(arr, 2, out_mode='full', pad_mode='reflect')
        >>> print(np.round(a, 2))
        [[ 5.  5.  6.  7.  8.  9. 10. 10.]
         [ 5.  5.  6.  7.  8.  9. 10. 10.]
         [12. 12. 13. 14. 15. 16. 17. 17.]
         [19. 19. 20. 21. 22. 23. 24. 24.]
         [26. 26. 27. 28. 29. 30. 31. 31.]
         [26. 26. 27. 28. 29. 30. 31. 31.]]
        >>> a = rolling_apply_nd(arr, 2, out_mode='full', pad_mode='symmetric')
        >>> print(np.round(a, 2))
        [[ 1.   1.5  2.5  3.5  4.5  5.5  6.5  7. ]
         [ 4.5  5.   6.   7.   8.   9.  10.  10.5]
         [11.5 12.  13.  14.  15.  16.  17.  17.5]
         [18.5 19.  20.  21.  22.  23.  24.  24.5]
         [25.5 26.  27.  28.  29.  30.  31.  31.5]
         [29.  29.5 30.5 31.5 32.5 33.5 34.5 35. ]]

    See Also:
        - flyingcircus.extra.moving_mean()
        - flyingcircus.extra.moving_average()
        - flyingcircus.extra.moving_apply()
        - flyingcircus.extra.running_apply()
    """
    args = tuple(args) if args else ()
    kws = dict(kws) if kws else {}
    arr = rolling_window_nd(
        arr, window, steps, window_steps, out_mode, pad_mode, pad_kws,
        False, 'end')
    axes = tuple(range(arr.ndim // 2, arr.ndim))
    kws['axis'] = axes if len(axes) > 1 else axes[0]
    return func(arr, *args, **kws)


# ======================================================================
def last_dim_as_sequence(
        arr,
        mode=tuple):
    """
    Convert the last dimension of an array to a sequence.

    Args:
        arr (np.ndarray): The input array.
        mode (type): The sequence type.
            Only supports `list` and `tuple`.
            If not `list`, defaults to `tuple`.

    Returns:
        arr (np.ndarray[object]): The output array.

    Examples:
        >>> arr = arange_nd((2, 3, 4))
        >>> print(arr)
        [[[ 0  1  2  3]
          [ 4  5  6  7]
          [ 8  9 10 11]]
        <BLANKLINE>
         [[12 13 14 15]
          [16 17 18 19]
          [20 21 22 23]]]
        >>> print(last_dim_as_sequence(arr))
        [[(0, 1, 2, 3) (4, 5, 6, 7) (8, 9, 10, 11)]
         [(12, 13, 14, 15) (16, 17, 18, 19) (20, 21, 22, 23)]]
    """
    if mode == list:
        n_dim = arr.ndim
        if n_dim > 1:
            arr = arr.tolist()
            temp = arr
            for _ in range(n_dim - 1):
                temp = temp[0]
            temp.append(None)
            result = np.array(arr)
            temp.pop()
        else:
            result = np.empty(1, dtype=object)
            result[0] = arr.tolist()
        return result
    else:
        dtype = [(str(i), arr.dtype) for i in range(arr.shape[-1])]
        return arr.view(dtype)[..., 0].astype(object)


# ======================================================================
def rel_err(
        arr1,
        arr2,
        use_average=False):
    """
    Calculate the element-wise relative error

    Args:
        arr1 (np.ndarray): The input array with the exact values
        arr2 (np.ndarray): The input array with the approximated values
        use_average (bool): Use the input arrays average as the exact values

    Returns:
        arr (ndarray): The relative error array

    Examples:
        >>> arr1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> arr2 = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1])
        >>> rel_err(arr1, arr2)
        array([0.1       , 0.05      , 0.03333333, 0.025     , 0.02      ,
               0.01666667])
        >>> rel_err(arr1, arr2, True)
        array([0.0952381 , 0.04878049, 0.03278689, 0.02469136, 0.01980198,
               0.01652893])
    """
    if arr2.dtype != np.complex:
        arr = (arr2 - arr1).astype(np.float)
    else:
        arr = (arr2 - arr1)
    if use_average:
        div = (arr1 + arr2) / 2.0
    else:
        div = arr1
    mask = (div != 0.0)
    arr *= mask
    arr[mask] = arr[mask] / div[mask]
    return arr


# ======================================================================
def euclid_dist(
        arr1,
        arr2,
        unsigned=True):
    """
    Calculate the element-wise correlation euclidean distance.

    This is the distance D between the identity line and the point of
    coordinates given by intensity:
        \\[D = abs(A2 - A1) / sqrt(2)\\]

    Args:
        arr1 (ndarray): The first array
        arr2 (ndarray): The second array
        unsigned (bool): Use signed distance

    Returns:
        arr (ndarray): The resulting array

    Examples:
        >>> arr1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        >>> arr2 = np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
        >>> euclid_dist(arr1, arr2)
        array([1.41421356, 2.82842712, 4.24264069, 5.65685425, 7.07106781,
               8.48528137])
        >>> euclid_dist(arr1, arr2, False)
        array([-1.41421356, -2.82842712, -4.24264069, -5.65685425, -7.07106781,
               -8.48528137])
    """
    arr = (arr2 - arr1) / 2.0 ** 0.5
    if unsigned:
        arr = np.abs(arr)
    return arr


# ======================================================================
elapsed(os.path.basename(__file__))

# ======================================================================
if __name__ == '__main__':
    run_doctests(__doc__)
