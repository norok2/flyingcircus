#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flyingcircus.util: generic numerical utilities.
"""

# ======================================================================
# :: Future Imports
from __future__ import (
    division, absolute_import, print_function, unicode_literals, )

# ======================================================================
# :: Python Standard Library Imports
import warnings  # Warning control
import csv  # CSV File Reading and Writing [CSV: Comma-Separated Values]
import itertools  # Functions creating iterators for efficient looping
import random  # Generate pseudo-random numbers

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
# import scipy.optimize  # SciPy: Optimization Algorithms
# import scipy.integrate  # SciPy: Integrations facilities
# import scipy.constants  # SciPy: Mathematal and Physical Constants
# import scipy.stats  # SciPy: Statistical functions
import scipy.signal  # SciPy: Signal processing
import scipy.ndimage  # SciPy: ND-image Manipulation

from numpy.fft import fftshift, ifftshift
from scipy.fftpack import fftn, ifftn

# :: Local Imports
import flyingcircus as fc
from flyingcircus import util

from flyingcircus import INFO, PATH
from flyingcircus import VERB_LVL, D_VERB_LVL, VERB_LVL_NAMES
from flyingcircus import elapsed, report
from flyingcircus import msg, dbg
from flyingcircus import HAS_JIT, jit


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
    axes = util.auto_repeat(axes, 1)
    if indexes is None:
        indexes = util.auto_repeat(None, len(axes))
    else:
        indexes = util.auto_repeat(indexes, 1)
    indexes = list(indexes)
    for i, (index, axis) in enumerate(zip(indexes, axes)):
        if index is None:
            indexes[i] = index = 0.5
        if isinstance(index, float) and index < 1.0:
            indexes[i] = int(arr.shape[axis] * index)
    # check index
    if any([(index >= arr.shape[axis]) or (index < 0)
            for index, axis in zip(indexes, axes)]):
        raise ValueError('Invalid array index in the specified direction')
    # determine slice index
    for index, axis in zip(indexes, axes):
        slab[axis] = index
    # print(slab)  # debug
    # slice the array
    return arr[tuple(slab)]


# ======================================================================
def slice_swap(a_slice):
    """
    Swap the `start` and `stop` attributes of a slice.

    If `step` is specified, `-step` is used as new `step`.

    Args:
        a_slice (slice): The input slice.

    Returns:
        r_slice (slice): The output slice.

    Examples:
        >>> slice_swap(slice(10, 20))
        slice(20, 10, None)
        >>> slice_swap(slice(10, 20, 2))
        slice(20, 10, -2)
    """
    return slice(
        a_slice.stop, a_slice.start, -a_slice.step if a_slice.step else None)


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
        weighting_kws (dict|tuple[tuple]|None): Keyword arguments.
            These are passed to the function specified in `weighting`.
            If tuple[tuple], must be convertible to a dictionary.
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
                    slice(None) if i != j else slice_swap(windows[0])
                    for j in range(arr.ndim))],
                arr[tuple(
                    slice(None) if i != j else slice_swap(windows[1])
                    for j in range(arr.ndim))],
                **weighting_kws)
            if circular else
            np.full(tuple(
                1 if i == j else d
                for j, d in enumerate(arr.shape)),
                endpoint)),
            axis=i)
        for i in range(arr.ndim)], axis=-1)
    idx_arr = np.arange(fc.util.prod(arr.shape), dtype=int).reshape(arr.shape)
    orig_idx_arr, dest_idx_arr = tuple(
        np.stack([
            np.concatenate((
                idx_arr[tuple(
                    slice(None) if i != j else window
                    for j in range(idx_arr.ndim))],
                idx_arr[tuple(
                    slice(None) if i != j else slice_swap(window)
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
        >>> arr = np.arange(util.prod(shape)).reshape(shape)
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
    return arr.reshape(shape[0], -1)[i, np.arange(util.prod(shape[1:]))]. \
        reshape(shape).swapaxes(axis, 0)


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
                text = ('Multiple positions (e.g. {} and {})'
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
        value is used for addiotional safety checks:

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
        if any([ax < -ndim - 1 or ax > ndim + 1 for ax in axis]):
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
            if not all([new_dim == 1 or new_dim == dim
                        for new_dim, dim in zip(new_shape, shape)]):
                raise ValueError(
                    'New shape {} cannot be broadcasted to shape {}'.format(
                        new_shape, shape))

    return arr.reshape(new_shape)


# ======================================================================
def mdot(*arrs):
    """
    Cumulative application of multiple `numpy.dot` operation.

    Args:
        *arrs (*Iterable[ndarray]): The input arrays.

    Returns:
        arr (np.ndarray): The result of the tensor product.

    Examples:
        >>>
    """
    # todo: complete docs
    arr = arrs[0]
    for item in arrs[1:]:
        arr = np.dot(arr, item)
    return arr


# ======================================================================
def ndot(
        arr,
        dim=-1,
        start=None,
        stop=None,
        step=None):
    """
    Cumulative application of `numpy.dot` operation over a given axis.

    Args:
        arr (np.ndarray): The input array.
        dim (int): The dimension along which to operate.
        start (int|None): The initial index for the dimension.
            If None, uses the minimum or maximum value depending on the
            value of
            `step`: if `step` is positive use minimum, otherwise maximum.
        stop (int|None): The final index for the dimension.
            If None, uses the minimum or maximum value depending on the
            value of
            `step`: if `step` is positive use maximum, otherwise minimum.
        step (int|None): The step for the dimension.
            If None, uses unity step.

    Returns:
        prod (np.ndarray): The result of the tensor product.

    Examples:
        >>>
    """
    # todo: complete docs
    if dim < 0:
        dim += arr.ndim
    if step is None:
        step = 1
        if start is not None and stop is not None and start > stop:
            step = -1
    if start is None:
        start = 0 if step > 0 else arr.shape[dim] - 1
    if stop is None:
        stop = arr.shape[dim] if step > 0 else -1
    prod = arr[
        [slice(None) if j != dim else start for j in range(arr.ndim)]]
    for i in range(start, stop, step)[1:]:
        indexes = [slice(None) if j != dim else i for j in range(arr.ndim)]
        prod = np.dot(prod, arr[indexes])
    return prod


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
def is_in_range(
        arr,
        interval,
        include_extrema=True):
    """
    Determine if the values of an array are within the specified interval.

    Args:
        arr (np.ndarray): The input array.
        interval (tuple[int|float]): The range of values to check.
            A 2-tuple with format (min, max) is expected.

    Returns:
        in_range (bool): The result of the comparison.
            True if all values of the array are within the interval.
            False otherwise.
    """
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
        out_interval (float,float): Interval of the output value(s).
            If None, set to: (0, 1).
        in_interval (float,float): Interval of the input value(s).
            If None, and val is Iterable, it is calculated as:
            (min(val), max(val)), otherwise set to: (0, 1).

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
        in_min, in_max = sorted(in_interval)
    elif isinstance(val, np.ndarray):
        in_min, in_max = minmax(val)
    else:
        in_min, in_max = (0, 1)
    if out_interval:
        out_min, out_max = sorted(out_interval)
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
        interval1 (tuple[float]): Interval of first operand
        interval2 (tuple[float]): Interval of second operand
        operation (str): String with operation to perform.
            Supports the following operations:
                - '+' : addition
                - '-' : subtraction

    Returns:
        new_interval (tuple[float]): Interval resulting from operation

    Examples:
        >>> combine_interval((-1.0, 1.0), (0, 1), '+')
        (-1.0, 2.0)
        >>> combine_interval((-1.0, 1.0), (0, 1), '-')
        (-2.0, 1.0)
    """
    if interval2 is None:
        interval2 = interval1
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
            ....
        AssertionError
    """
    if not util.is_same_sign((start, stop)):
        assert (abs(start) > 1 and abs(stop) > 1)
        bounds = ((start, 1 / start), (1 / stop, stop))
        equity = 1 if num % 2 == 1 and abs(start) > abs(stop) else 0
        nums = (num // 2 + equity, num - num // 2 - equity)
        endpoints = True, endpoint
        logspaces = tuple(
            np.geomspace(*bound_, num=num_, endpoint=endpoint_)
            for bound_, num_, endpoint_ in zip(bounds, nums, endpoints))
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

    Useful to remove specific values, e.g. singularities.

    Args:
        arr (np.ndarray): The input array.
        pairs (tuple[tuple]): The substitution rules.
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
        util.subst
    """
    arr = arr.ravel()
    for val in removes:
        if val is np.nan:
            arr = arr[~np.isnan(arr)]
        if len(arr) > 0:
            arr = arr[arr != val]
    return arr


# ======================================================================
def dftn(arr):
    """
    Discrete Fourier Transform.

    Interface to fftn combined with fftshift.

    Args:
        arr (np.ndarray): Input n-dim array.

    Returns:
        arr (np.ndarray): Output n-dim array.

    Examples:
        >>> a = np.arange(2)
        >>> dftn(a)
        array([-1.+0.j,  1.+0.j])
        >>> print(np.allclose(a, dftn(idftn(a))))
        True

    See Also:
        numpy.fft, scipy.fftpack
    """
    return fftshift(fftn(arr))


# ======================================================================
def idftn(arr):
    """
    Inverse Discrete Fourier transform.

    Interface to ifftn combined with ifftshift.

    Args:
        arr (np.ndarray): Input n-dim array.

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
    return ifftn(ifftshift(arr))


# ======================================================================
def coord(
        shape,
        position=0.5,
        is_relative=True,
        use_int=True):
    """
    Calculate the coordinate in a given shape for a specified position.

    Args:
        shape (Iterable[int]): The shape of the mask in px.
        position (float|Iterable[float]): Relative position of the origin.
            Values are in the [0, 1] interval.
        is_relative (bool): Interpret origin as relative.
        use_int (bool): Force interger values for the coordinates.

    Returns:
        position (list): The coordinate in the shape.

    Examples:
        >>> coord((5, 5))
        (2, 2)
        >>> coord((4, 4))
        (2, 2)
        >>> coord((5, 5), 3, False)
        (3, 3)
    """
    position = util.auto_repeat(position, len(shape), check=True)
    if is_relative:
        if use_int:
            position = tuple(
                int(scale(x, (0, dim))) for x, dim in zip(position, shape))
        else:
            position = tuple(
                scale(x, (0, dim - 1)) for x, dim in zip(position, shape))
    elif any([not isinstance(x, int) for x in position]) and use_int:
        raise TypeError('Absolute origin must be integer.')
    return position


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
        shape (Iterable[int]): The shape of the mask in px.
        position (float|Iterable[float]): Relative position of the origin.
            Values are in the [0, 1] interval.
        is_relative (bool): Interpret origin as relative.
        dense (bool): Determine the shape of the mesh-grid arrays.
        use_int (bool): Force interger values for the coordinates.

    Returns:
        coord (list[np.ndarray]): mesh-grid ndarrays.
            The shape is identical if dense is True, otherwise only one
            dimension is larger than 1.

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
    position = coord(shape, position, is_relative, use_int)
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    return np.ogrid[grid] if not dense else np.mgrid[grid]


# ======================================================================
def rel2abs(shape, size=0.5):
    """
    Calculate the absolute size from a relative size for a given shape.

    Args:
        shape (int|Iterable[int]): The shape of the container in px.
        size (float|tuple[float]): Relative position (to the lowest edge).
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
    """
    size = fc.util.auto_repeat(size, len(shape), check=True)
    return tuple((s - 1.0) * p for p, s in zip(size, shape))


# ======================================================================
def abs2rel(shape, position=0):
    """
    Calculate the relative size from an absolute size for a given shape.

    Args:
        shape (int|Iterable[int]): The shape of the container in px.
        position (float|tuple[float]): Absolute position inside the shape.
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
    """
    position = fc.util.auto_repeat(position, len(shape), check=True)
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
        kk_2 (np.ndarray): The resulting kernel array.

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
    kk_ = grid_coord(shape)
    if factors and factors != 1:
        factors = util.auto_repeat(factors, len(shape), check=True)
        kk_ = [k_i / factor for k_i, factor in zip(kk_, factors)]
    kk_2 = np.zeros(shape)
    for k_i, dim in zip(kk_, shape):
        kk_2 += k_i ** 2
    return kk_2


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
    kk_ = grid_coord(shape)
    if factors and factors != 1:
        factors = util.auto_repeat(factors, len(shape), check=True)
        kk_ = [k_i / factor for k_i, factor in zip(kk_, factors)]
    if dims is None:
        dims = range(len(shape))
    else:
        if isinstance(dims, int):
            dims = (dims,)
        dims = tuple(dim % len(shape) for dim in dims)
    kks = tuple(
        np.broadcast_to(k_i, shape)
        for i, (k_i, dim) in enumerate(zip(kk_, shape))
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
        dims (int|Iterable[int]): The direction of the gradient.
            Values must be between `len(shape)` and `-len(shape)`.
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
    kk_ = grid_coord(shape)
    if factors and factors != 1:
        factors = util.auto_repeat(factors, len(shape), check=True)
        kk_ = [k_i / factor for k_i, factor in zip(kk_, factors)]
    if dims is None:
        dims = range(len(shape))
    else:
        if isinstance(dims, int):
            dims = (dims,)
        dims = tuple(dim % len(shape) for dim in dims)
    kks = tuple(
        np.broadcast_to((1.0 - np.exp(2j * np.pi * k_i)), shape)
        for i, (k_i, dim) in enumerate(zip(kk_, shape))
        if i in dims)
    return kks


# ======================================================================
def padding(
        arr,
        pad_width=0,
        pad_mode='constant',
        pad_kws=(('constant_values', 0.0),)):
    """
    Array padding with a constant value.

    Useful for zero-padding. This is the default behavior.

    Args:
        arr (np.ndarray): The input array.
        pad_width (int|float): Size of the padding to use.
            See `fc.util.auto_pad_width()` for more details.
        pad_mode (str): The padding mode.
            See `np.pad()` for more details.
        pad_kws (dict|Iterable[Iterable]): Keyword parameters.
            These are passed to `np.pad()`.

    Returns:
        result (tuple): The tuple
            contains:
             - arr (np.ndarray): The padded array.
             - mask (tuple(slice)): The mask delimiting the input array.
    """
    pad_kws = dict(pad_kws) if pad_kws else {}
    if pad_width:
        shape = arr.shape
        pad_width = util.auto_pad_width(pad_width, shape)
        # mask = (slice(borders, -borders),) * arr.ndim
        mask = tuple(slice(lower, -upper) for (lower, upper) in pad_width)
        arr = np.pad(arr, pad_width, pad_mode, **pad_kws)
    else:
        mask = (slice(None),) * arr.ndim
    return arr, mask


# ======================================================================
def gradients(
        arr,
        dims=None,
        ft_factor=(2 * np.pi),
        pad_width=0.0):
    """
    Apply the gradient operator (in the Fourier domain).

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
        gradient_kernels()
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
        exp_gradient_kernels()
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
    kk_2 = fftshift(laplace_kernel(arr.shape, arr.shape))
    arr = ((1j * ft_factor) ** 2) * ifftn(kk_2 * fftn(arr))
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
    kk_2 = fftshift(laplace_kernel(arr.shape, arr.shape))
    kk_2[kk_2 != 0] = 1.0 / kk_2[kk_2 != 0]
    arr = ((-1j / ft_factor) ** 2) * ifftn(kk_2 * fftn(arr))
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
        num = int(np.ceil(np.sqrt(arr.size)))
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
            See `flyingcircus.util.auto_bin()` for available methods.
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
    # skip zero values
    mask = hist != 0.0
    log_hist = np.zeros_like(hist)
    log_hist[mask] = np.log(hist[mask]) / np.log(base)
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
        norm_mutual_information(arr1, arr2, bins=bins)

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
            Uses `flyingcircus.util.grid_coord()` internally.

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
        n_dim = util.combine_iter_len((shape, sigmas, position))
        if n_dim == 0:
            n_dim = 1

    shape = util.auto_repeat(shape, n_dim)
    sigmas = util.auto_repeat(sigmas, n_dim)
    position = util.auto_repeat(position, n_dim)

    position = grid_coord(
        shape, position, is_relative=rel_position, use_int=False)
    arr = np.exp(-(sum([
        x_i ** 2 / (2 * sigma ** 2) for x_i, sigma in zip(position, sigmas)])))
    if callable(norm):
        arr /= norm(arr)
    return arr


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
        **kws):
    """
    Calculate the moving average (with optional weights).

    The moving average will be applied to the flattened array.
    Unless specified otherwise, the size of the array will be reduced by
    len(weights) - 1
    This is equivalent to passing `mode='valid'` to `scipy.signal.convolve`.
    Please refer to `scipy.signal.convolve` for more options.

    Args:
        arr (np.ndarray): The input array.
        weights (int|Iterable): The running weights.
            If int, the number of elements to group in the 'running' axis and
            unity weights are used.
            The size of the weights array len(weights) must be such that
            len(weights) >= 1 and len(weights) <= len(array), otherwise the
            flattened array is returned.
        **kws (dict): Keyword arguments passed to `scipy.signal.convolve`.

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
    """
    arr = arr.ravel()
    if isinstance(weights, int):
        weights = np.ones((weights,))
    else:
        # weights order needs to be inverted
        weights = np.array(weights)[::-1]
    num = len(weights) if isinstance(weights, np.ndarray) else 0
    if len(arr) >= num > 1:
        if 'mode' not in kws:
            kws['mode'] = 'valid'
        arr = sp.signal.convolve(arr, weights / len(weights), **kws)
        arr *= len(weights) / np.sum(weights)
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
def rolling_stat(
        arr,
        weights=1,
        stat_func=np.mean,
        stat_args=None,
        stat_kws=None,
        mode='valid',
        borders=None):
    """
    Calculate the rolling statistics on an array.

    This is calculated by running the specified statistics for each subset of
    the array of given size, including optional weightings.
    The moving average will be applied to the flattened array.

    This function differs from `running_stat` in that it should be faster but
    more memory demanding.
    Also the `stat_func` callable is required to accept an `axis` parameter.

    Args:
        arr (np.ndarray): The input array.
        weights (int|Iterable): The running weights.
            If int, the number of elements to group in the 'running' axis and
            unity weights are used.
            The size of the weights array len(weights) must be such that
            len(weights) >= 1 and len(weights) <= len(array), otherwise the
            flattened array is returned.
            Note that these weights are
        stat_func (callable): Function to calculate in the 'running' axis.
            Must accept an `axis` parameter, which will be set to -1 on the
            flattened input.
        stat_args (tuple|list): Positional arguments passed to `stat_func`.
        stat_kws (dict): Keyword arguments passed to `stat_func`.
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
        >>> all([np.allclose(
        ...                  moving_average(arr, n, mode=mode),
        ...                  rolling_stat(arr, n, mode=mode))
        ...      for n in range(num) for mode in ('valid', 'same', 'full')])
        True
        >>> rolling_stat(arr, 4, mode='same', borders=100)
        array([50.75, 26.5 ,  2.5 ,  3.5 ,  4.5 ,  5.5 ,  6.5 , 30.25])
        >>> rolling_stat(arr, 4, mode='full', borders='same')
        array([1.  , 1.25, 1.75, 2.5 , 3.5 , 4.5 , 5.5 , 6.5 , 7.25, 7.75, 8.\
  ])
        >>> rolling_stat(arr, 4, mode='full', borders='circ')
        array([5.5, 4.5, 3.5, 2.5, 3.5, 4.5, 5.5, 6.5, 5.5, 4.5, 3.5])
        >>> rolling_stat(arr, 4, mode='full', borders='sym')
        array([1.75, 1.5 , 1.75, 2.5 , 3.5 , 4.5 , 5.5 , 6.5 , 7.25, 7.5 ,\
 7.25])
        >>> rolling_stat(arr, 4, mode='same', borders='circ')
        array([4.5, 3.5, 2.5, 3.5, 4.5, 5.5, 6.5, 5.5])
        >>> rolling_stat(arr, [1, 0.2])
        array([1.16666667, 2.16666667, 3.16666667, 4.16666667, 5.16666667,
               6.16666667, 7.16666667])
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
            raise ValueError(
                '`borders={borders}` not understood'.format(**locals()))

        # calculate generator for data and weights
        arr = np.concatenate((arr, extension))
        gen = np.zeros((size + num - 1, num))
        for i in range(num):
            gen[:, i] = np.roll(arr, i)[:size + num - 1]
        w_gen = np.stack([weights] * (size + num - 1))

        # calculate the running stats
        arr = stat_func(
            gen * w_gen,
            *(stat_args if stat_args else ()), axis=-1,
            **(stat_kws if stat_kws else {}))
        arr *= len(weights) / np.sum(weights)

        # adjust output according to mode
        if mode == 'valid':
            arr = arr[num - 1:-(num - 1)]
        elif mode == 'same':
            begin = (num - 1) // 2
            arr = arr[begin:begin + size]
    return arr


# ======================================================================
def running_stat(
        arr,
        weights=1,
        stat_func=np.mean,
        stat_args=None,
        stat_kws=None,
        mode='valid',
        borders=None):
    """
    Calculate the running statistics on an array.

    This is calculated by running the specified statistics for each subset of
    the array of given size, including optional weightings.
    The moving average will be applied to the flattened array.

    This function differs from `rolling_stat` in that it should be slower but
    less memory demanding.
    Also the `stat_func` callable is not required to accept an `axis`
    parameter.

    Args:
        arr (np.ndarray): The input array.
        weights (int|Iterable): The running weights.
            If int, the number of elements to group in the 'running' axis and
            unity weights are used.
            The size of the weights array len(weights) must be such that
            len(weights) >= 1 and len(weights) <= len(array), otherwise the
            flattened array is returned.
            Note that these weights are
        stat_func (callable): Function to calculate in the 'running' axis.
        stat_args (tuple|list): Positional arguments passed to `stat_func`.
        stat_kws (dict): Keyword arguments passed to `stat_func`.
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
        >>> all([np.allclose(
        ...                  moving_average(arr, n, mode=mode),
        ...                  running_stat(arr, n, mode=mode))
        ...      for n in range(num) for mode in ('valid', 'same', 'full')])
        True
        >>> running_stat(arr, 4, mode='same', borders=100)
        array([50.75, 26.5 ,  2.5 ,  3.5 ,  4.5 ,  5.5 ,  6.5 , 30.25])
        >>> running_stat(arr, 4, mode='same', borders='circ')
        array([4.5, 3.5, 2.5, 3.5, 4.5, 5.5, 6.5, 5.5])
        >>> running_stat(arr, 4, mode='full', borders='circ')
        array([5.5, 4.5, 3.5, 2.5, 3.5, 4.5, 5.5, 6.5, 5.5, 4.5, 3.5])
        >>> running_stat(arr, [1, 0.2])
        array([1.16666667, 2.16666667, 3.16666667, 4.16666667, 5.16666667,
               6.16666667, 7.16666667])
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
            raise ValueError(
                '`borders={borders}` not understood'.format(**locals()))

        # calculate generator for data and weights
        gen = np.concatenate((head, arr, tail))
        # print(gen)
        arr = np.zeros((len(gen) - num + 1))
        for i in range(len(arr)):
            arr[i] = stat_func(
                gen[i:i + num] * weights,
                *(stat_args if stat_args else ()),
                **(stat_kws if stat_kws else {}))
        arr *= len(weights) / np.sum(weights)

        # adjust output according to mode
        if mode == 'valid':
            arr = arr[num - 1:-(num - 1)]
        elif mode == 'same':
            begin = (num - 1) // 2
            arr = arr[begin:begin + size]
    return arr


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
        tuple[float]:
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
        tuple[float]:
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
        tuple[float]:
         - modulus (float): The modulus R of the complex number.
         - argument (float): The phase phi of the complex number.
    """
    return np.sqrt(real ** 2 + imag ** 2), np.arctan2(real, imag)


# ======================================================================
def filter_cx(
        arr,
        filter_func,
        filter_args=None,
        filter_kws=None,
        mode='cartesian'):
    """
    Calculate a non-complex function on a complex input array.

    Args:
        arr (np.ndarray): The input array.
        filter_func (callable): The function used to filter the input.
            Requires the first arguments to be an `np.ndarray`.
        filter_args (tuple|None): Positional arguments of `filter_func`.
        filter_kws (dict|None): Keyword arguments of `filter_func`.
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
    if not filter_args:
        filter_args = ()
    if not filter_kws:
        filter_kws = {}
    if mode == 'cartesian':
        arr = (
                filter_func(arr.real, *filter_args, **filter_kws) +
                1j * filter_func(arr.imag, *filter_args, **filter_kws))
    elif mode == 'polar':
        arr = (
                filter_func(np.abs(arr), *filter_args, **filter_kws) *
                np.exp(
                    1j * filter_func(
                        np.angle(arr), *filter_args, **filter_kws)))
    elif mode == 'real':
        arr = (
                filter_func(
                    arr.real, *filter_args, **filter_kws) + 1j * arr.imag)
    elif mode == 'imag':
        arr = (
                arr.real + 1j * filter_func(
            arr.imag, *filter_args, **filter_kws))
    elif mode == 'mag':
        arr = (
                filter_func(np.abs(arr), *filter_args, **filter_kws) *
                np.exp(1j * np.angle(arr)))
    elif mode == 'phs':
        arr = (
                np.abs(arr) * np.exp(
            1j * filter_func(np.angle(arr), *filter_args, **filter_kws)))
    else:
        warnings.warn(
            'Mode `{}` not known'.format(mode) + ' Using default.')
        arr = filter_cx(arr, filter_func, filter_args, filter_kws)
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
    if util.is_increasing(items):
        sign = -1
    elif util.is_decreasing(items):
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
    if util.is_increasing(items):
        sign = -1
    elif util.is_decreasing(items):
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
    if util.is_increasing(items):
        sign = -1
    elif util.is_decreasing(items):
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
    if util.is_increasing(items):
        sign = -1
    elif util.is_decreasing(items):
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
            using `flyingcircus.util.auto_bin()` with `method` set to
            `bins` if str,
            and using the default `flyingcircus.util.auto_bin()` method if
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
             - 'elbow': use `flyingcircus.util.marginal_sep_elbow()`.
             - 'quad': use `flyingcircus.util.marginal_sep_quad()`.
             - 'quad_weight': use
             `flyingcircus.util.marginal_sep_quad_weight()`.
             - 'quad_inv_weight': use
             `flyingcircus.util.marginal_sep_quad_inv_weight()`.
             - 'otsu': use `flyingcircus.segmentation.threshold_otsu()`.
             - 'X%': set the threshold at 'X' percent of the largest eigenval.
        q (Iterable[int|float|complex]|None): The values of the components.
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
        var(), std()
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
        avg(), std()

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
        avg(), var()
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
        avg()
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
            val_list.append(util.compact_num_str(stats_dict[label]))
    if save_path:
        with open(save_path, 'wb') as csv_file:
            csv_writer = csv.writer(
                csv_file, delimiter=str(util.CSV_DELIMITER))
            csv_writer.writerow(label_list)
            csv_writer.writerow(val_list)
    if title:
        print_str = title + ': '
        for label in label_list:
            if compact:
                print_str += '{}={}, '.format(
                    label, util.compact_num_str(stats_dict[label]))
            else:
                print_str += '{}={}, '.format(label, stats_dict[label])
        print(print_str)
    return stats_dict


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
            `fc.num.unsqueeze()`.
            If None, no masking is performed.
        borders (int|float|tuple[int|float]|None): The border size(s).
            If None, the border is not modified.
            Otherwise, a border is added to the masked array.
            If int, this is in units of pixels.
            If float, this is proportional to the initial array shape.
            If int or float, uses the same value for all dimensions,
            unless `unsqueezing` is set to True, in which case, the same value
            is used only for non-singletons, while 0 is used for singletons.
            If Iterable, the size must match `arr` dimensions.
            If 'use_longest' is True, use the longest dimension for the
            calculations.
        background (int|float): The value used for masked-out pixels.
        do_unsqueeze (bool): Unsqueeze mask to input.
            If True, use `fc.num.unsqueeze()` on mask.
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
        frame()
    """
    if mask is not None:
        mask = mask.astype(bool)
        if arr.ndim > mask.ndim and do_unsqueeze:
            old_shape = mask.shape
            mask = unsqueeze(mask, shape=arr.shape)
            if isinstance(borders, (int, float)):
                borders = [borders if dim != 1 else 0 for dim in mask.shape]
            elif borders is not None and len(borders) == len(old_shape):
                borders = list(
                    fc.util.replace_iter(
                        mask.shape, lambda x: x == 1, borders))
        arr = arr.copy()
        if arr.shape != mask.shape:
            mask = np.broadcast_to(mask, arr.shape)
        if arr.shape == mask.shape:
            arr[~mask] = background
            if borders is not None:
                container = sp.ndimage.find_objects(mask.astype(int))[0]
                if container:
                    arr = arr[container]
                arr = frame(arr, borders, background)
        else:
            raise ValueError(
                'Cannot apply mask shaped `{}` to array shaped `{}`.'.format(
                    mask.shape, arr.shape))
    return arr


# ======================================================================
def frame(
        arr,
        borders=0.05,
        background=0.0,
        use_longest=True):
    """
    Add a background frame to an array specifying the borders.

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
        reframe()
    """
    borders = fc.util.auto_repeat(borders, arr.ndim)
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
    inner = [
        slice(border, border + dim, None)
        for dim, border in zip(arr.shape, borders)]
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
        frame()

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
    new_shape = fc.util.auto_repeat(new_shape, arr.ndim, check=True)
    position = fc.util.auto_repeat(position, arr.ndim, check=True)
    if any([old > new for old, new in zip(arr.shape, new_shape)]):
        raise ValueError('new shape cannot be smaller than the old one.')
    position = [
        int(round((new - old) * x_i)) if isinstance(x_i, float) else x_i
        for old, new, x_i in zip(arr.shape, new_shape, position)]
    if any([old + x_i > new
            for old, new, x_i in zip(arr.shape, new_shape, position)]):
        raise ValueError(
            'Incompatible `new_shape`, `array shape` and `position`.')
    result = np.full(new_shape, background)
    inner = tuple(
        slice(offset, offset + dim, None)
        for dim, offset in zip(arr.shape, position))
    result[inner] = arr
    return result


# ======================================================================
def multi_reframe(
        arrs,
        new_shape=None,
        background=0.0,
        dtype=None):
    """
    Reframe arrays (by adding border) to match the same shape.

    Note that:
     - uses 'reframe' under the hood;
     - the sampling / resolution / voxel size will NOT change;
     - the support space / field-of-view will change.

    Args:
        arrs (Iterable[np.ndarray]): The input arrays.
        new_shape (Iterable[int]): The new base shape of the arrays.
        background (int|float|complex): The background value for the frame.
        dtype (data-type): Desired output data-type.
            If None, its guessed from dtype of arrs.
            See `np.ndarray()` for more.

    Returns:
        result (np.ndarray): The output array.
            It contains all reframed arrays from `arrs`, through the last dim.
            The shape of this array is `new_shape` + `len(arrs)`.
    """
    # calculate new shape
    if new_shape is None:
        shapes = [arr.shape for arr in arrs]
        new_shape = [1] * max([len(shape) for shape in shapes])
        shape_arr = np.ones((len(shapes), len(new_shape))).astype(np.int)
        for i, shape in enumerate(shapes):
            shape_arr[i, :len(shape)] = np.array(shape)
        new_shape = tuple(
            max(*list(shape_arr[:, i]))
            for i in range(len(new_shape)))

    if dtype is None:
        # dtype = functools.reduce(
        #     (lambda x, y: np.promote_types(x, y.dtype)), arrs)
        dtype = bool
        for arr in arrs:
            dtype = np.promote_types(dtype, arr.dtype)

    result = np.array(new_shape + (len(arrs),), dtype=dtype)
    for i, arr in enumerate(arrs):
        # ratio should not be kept: keep_ratio_method=None
        result[..., i] = reframe(arr, new_shape, background=background)
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
        zoom_factors (float|tuple[float]): The zoom factors for each
        directions.
        shape (int|Iterable[int]): The shape of the array to operate with.
        extra_dim (bool): Force extra dimensions in the zoom parameters.
        fill_dim (bool): Dimensions not specified are left untouched.

    Returns:
        zoom (tuple[float]): The zoom factors for each directions.
        shape (int|Iterable[int]): The shape of the array to operate with.
    """
    zoom_factors = list(fc.util.auto_repeat(zoom_factors, len(shape)))
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
        fill_dim=True):
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
            `sp.ndimage.uniform_filter()`.
            If Iterable, its size must match the number of dims of `arr`.
            If int, uses an isotropic window with the specified size.
            If None, the window is calculated automatically from the `zoom`
            parameter.
        interp_order (int): Order of the spline interpolation.
            0: nearest. Accepted range: [0, 5].
        extra_dim (bool): Force extra dimensions in the zoom parameters.
        fill_dim (bool): Dimensions not specified are left untouched.

    Returns:
        result (np.ndarray): The output array.

    See Also:
        geometry.zoom
    """
    factors, shape = zoom_prepare(factors, arr.shape, extra_dim, fill_dim)
    if window is None:
        window = [round(1.0 / (2.0 * x)) for x in factors]
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
        fill_dim=True):
    """
    Reshape the array to a new shape (different resolution / pixel size).

    Args:
        arr (np.ndarray): The input array.
        new_shape (Iterable[int|None]): New dimensions of the array.
        aspect (callable|Iterable[callable]|None): Zoom shape manipulation.
            Useful for obtaining specific aspect ratio effects.
            This is passed to `fc.num.shape2zoom()`.
        window (int|Iterable[int]|None): Uniform pre-filter window size.
            This is the size of the window for the uniform filter using
            `sp.ndimage.uniform_filter()`.
            If Iterable, its size must match the number of dims of `arr`.
            If int, uses an isotropic window with the specified size.
            If None, the window is calculated automatically from `new_shape`.
        interp_order (int|None): Order of the spline interpolation.
            0: nearest. Accepted range: [0, 5].
        extra_dim (bool): Force extra dimensions in the zoom parameters.
        fill_dim (bool): Dimensions not specified are left untouched.

    Returns:
        arr (np.ndarray): The output array.

    See Also:
        geometry.zoom
    """
    factors = shape2zoom(arr.shape, new_shape, aspect)
    factors, shape = zoom_prepare(
        factors, arr.shape, extra_dim, fill_dim)
    arr = zoom(arr, factors, window=window, interp_order=interp_order)
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
            `sp.ndimage.uniform_filter()`.
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
        new_shape = [1] * max([len(shape) for shape in shapes])
        shape_arr = np.ones((len(shapes), len(new_shape))).astype(np.int)
        for i, shape in enumerate(shapes):
            shape_arr[i, :len(shape)] = np.array(shape)
        combiner = fc.util.lcm if lossless else max
        new_shape = tuple(
            combiner(*list(shape_arr[:, i]))
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
def decode_affine(
        affine):
    """
    Decompose the affine matrix into a linear transformation and a translation.

    Args:
        affine (np.ndarray): The (N+1)-sized affine square matrix.

    Returns:
        linear (np.ndarray): The N-sized linear square matrix.
        shift (np.ndarray): The shift along each axis in px.
    """
    dims = affine.shape
    linear = affine[:dims[0] - 1, :dims[1] - 1]
    shift = affine[:-1, -1]
    return linear, shift


# ======================================================================
def encode_affine(
        linear,
        shift):
    """
    Combine a linear transformation and a translation into the affine matrix.

    Args:
        linear (np.ndarray): The N-sized linear square matrix.
        shift (np.ndarray): The shift along each axis in px.

    Returns:
        affine (np.ndarray): The (N+1)-sized affine square matrix.
    """
    dims = linear.shape
    affine = np.eye(dims[0] + 1)
    affine[:dims[0], :dims[1]] = linear
    affine[:-1, -1] = shift
    return affine


# ======================================================================
def num_angles_from_dim(n_dim):
    """
    Calculate the complete number of angles given the dimension.

    Given the dimension of an array, calculate the number of all possible
    cartesian orthogonal planes of rotations, using the formula:

    :math:`N = n * (n - 1) / 2` [ :math:`N = n! / 2! / (n - 2)!` ]
    (N: num of angles, n: num of dim)

    Args:
        n_dim (int): The number of dimensions.

    Returns:
        n_angles (int): The corresponding number of angles.

    See Also:
        fc.num.num_dim_from_angles()
    """
    return n_dim * (n_dim - 1) // 2


# ======================================================================
def num_dim_from_angles(
        n_angles,
        raise_err=False):
    """
    Computes the number of dimensions from the number of angles.

    This is the solution for `n` to the equation: :math:`n * (n - 1) / 2 = N`
    (N: num of angles, n: num of dim)

    Args:
        n_angles (int): The number of angles.
        raise_err (bool): Raise an exception if invalid number of angles.

    Returns:
        n_dim (int): The corresponding number of dimensions.

    Raises:
        ValueError: if `raise_err == True` and the number of angles is invalid!

    See Also:
        fc.num.num_angles_from_dim()
    """
    n_dim = ((1 + np.sqrt(1 + 8 * n_angles)) / 2)
    # alternatives: numpy.modf, math.modf
    int_part, dec_part = divmod(n_dim, 1)
    if not np.isclose(dec_part, 0.0) and raise_err:
        raise ValueError('cannot get the dimension from the number of angles')
    return int(np.ceil(n_dim))


# ======================================================================
def angles2linear(
        angles,
        n_dim=None,
        axes_list=None,
        use_degree=True,
        atol=None):
    """
    Calculate the linear transformation relative to the specified rotations.

    Args:
        angles (tuple[float]): The angles to be used for rotation.
        n_dim (int|None): The number of dimensions to consider.
            The number of angles and `n_dim` should satisfy the relation:
            `n_angles = n_dim * (n_dim - 1) / 2`.
            If `len(angles)` is smaller than expected for a given `n_dim`,
            the remaining angles are set to 0.
            If `len(angles)` is larger than expected, the exceeding `angles`
            are ignored.
            If None, n_dim is computed from `len(angles)`.
        axes_list (tuple[tuple[int]]|None): The axes of the rotation plane.
            If not None, for each rotation angle a pair of axes
            (i.e. a 2-tuple of int) must be specified to define the associated
            plane of rotation.
            The number of 2-tuples should match the number of of angles
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
        linear (np.ndarray): The rotation matrix as defined by the angles.

    See Also:
        fc.num.num_angles_from_dim(),
        fc.num.num_dim_from_angles(),
        itertools.combinations
    """
    if n_dim is None:
        n_dim = num_dim_from_angles(len(angles))
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
    # :: check that this is a rotation matrix
    det = np.linalg.det(lin_mat)
    if not atol:
        atol = lin_mat.ndim ** 4 * np.finfo(np.double).eps
    if np.abs(det) - 1.0 > atol:
        text = 'rotation matrix may be inaccurate [det = {}]'.format(repr(det))
        warnings.warn(text)
    return lin_mat


# ======================================================================
def linear2angles(
        linear,
        use_degree=True,
        atol=None):
    # todo: implement the inverse of angles2linear
    raise NotImplementedError


# ======================================================================
def prepare_affine(
        shape,
        lin_mat,
        shift=None,
        origin=None):
    """
    Prepare parameters to be used with `scipy.ndimage.affine_transform()`.

    In particular, it computes the linear matrix and the offset implementing
    an affine transformation followed by a translation on the array.

    The result can be passed directly as `matrix` and `offset` parameters of
    `scipy.ndimage.affine_transform()`.

    Args:
        shape (Iterable): The shape of the array to be transformed.
        lin_mat (np.ndarray): The N-sized linear square matrix.
        shift (np.ndarray|None): The shift along each axis in px.
            If None, no shift is performed.
        origin (np.ndarray|None): The origin of the linear transformation.
            If None, uses the center of the array.

    Returns:
        result (tuple): The tuple
            contains:
             - lin_mat (np.ndarray): The N-sized linear square matrix.
             - offset (np.ndarray): The offset along each axis in px.

    See Also:
        scipy.ndimage.affine_transform()
    """
    ndim = len(shape)
    if shift is None:
        shift = 0
    if origin is None:
        origin = np.array(rel2abs(shape, (0.5,) * ndim))
    offset = origin - np.dot(lin_mat, origin + shift)
    return lin_mat, offset


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
        fc.num.tensor_of_inertia(),
        fc.num.rotatio_axes(),
        fc.num.auto_rotation(),
        fc.num.realigning()
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
        fc.num.tensor_of_inertia,
        fc.num.rotation_axes,
        fc.num.auto_rotation,
        fc.num.realigning
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
        fc.num.weighted_covariance(),
        fc.num.rotation_axes(),
        fc.num.auto_rotation(),
        fc.num.realigning()
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
        fc.num.weighted_covariance(),
        fc.num.tensor_of_inertia(),
        fc.num.auto_rotation(),
        fc.num.realigning()
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
        lin_mat (np.ndarray): The linear matrix for the rotation.
        offset (np.ndarray): The offset for the translation.

    See Also:
        scipy.ndimage.center_of_mass(),
        scipy.ndimage.affine_transform(),
        fc.num.weighted_covariance(),
        fc.num.tensor_of_inertia(),
        fc.num.rotation_axes(),
        fc.num.angles2linear(),
        fc.num.linear2angles(),
        fc.num.auto_rotation(),
        fc.num.realigning()
    """
    lin_mat = rotation_axes_to_matrix(rotation_axes(arr, labels, index, True))
    lin_mat, offset = prepare_affine(arr.shape, lin_mat, origin=origin)
    return lin_mat, offset


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
        lin_mat (np.ndarray): The linear matrix for the rotation.
        offset (np.ndarray): The offset for the translation.

    See Also:
        scipy.ndimage.center_of_mass(),
        scipy.ndimage.affine_transform(),
        fc.num.weighted_covariance(),
        fc.num.tensor_of_inertia(),
        fc.num.rotation_axes(),
        fc.num.angles2linear(),
        fc.num.linear2angles(),
        fc.num.auto_rotation(),
        fc.num.realigning()
    """
    lin_mat = np.eye(arr.ndim)
    com = np.array(sp.ndimage.center_of_mass(arr, labels, index))
    lin_mat, offset = prepare_affine(arr.shape, lin_mat, com, origin)
    return lin_mat, offset


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
        lin_mat (np.ndarray): The linear matrix for the rotation.
        offset (np.ndarray): The offset for the translation.

    See Also:
        scipy.ndimage.center_of_mass(),
        scipy.ndimage.affine_transform(),
        fc.num.weighted_covariance(),
        fc.num.tensor_of_inertia(),
        fc.num.rotation_axes(),
        fc.num.angles2linear(),
        fc.num.linear2angles(),
        fc.num.auto_rotation(),
        fc.num.auto_shift()
    """
    com = np.array(sp.ndimage.center_of_mass(arr, labels, index))
    lin_mat = rotation_axes_to_matrix(rotation_axes(arr, labels, index, True))
    lin_mat, offset = prepare_affine(arr.shape, lin_mat, com, origin)
    return lin_mat, offset


# ======================================================================
def rand_mask(
        arr,
        density=0.01):
    """
    Calculate a randomly distributed mask of specified density.

    Args:
        arr (np.ndarray): The target array.
        density (float): The density of the mask.
            Must be in the (0, 1) interval.

    Returns:
        mask
    """
    if not 0 < density < 1:
        raise ValueError('Density must be between 0 and 1')
    shape = arr.shape
    mask = np.zeros_like(arr).astype(np.bool).ravel()
    mask[random.sample(range(arr.size), int(arr.size * density))] = True
    return mask.reshape(shape)


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
    arr = (arr2 - arr1) / np.sqrt(2.0)
    if unsigned:
        arr = np.abs(arr)
    return arr


# ======================================================================
elapsed(__file__[len(PATH['base']) + 1:])

# ======================================================================
if __name__ == '__main__':
    import doctest  # Test interactive Python examples

    msg(__doc__.strip())
    doctest.testmod()
    msg(report())
