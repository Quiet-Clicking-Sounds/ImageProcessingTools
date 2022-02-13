import functools
import math
from math import floor

import bottleneck
import numpy


def wrapper_f32_u8(function):
    """ Wrapper function to convert input numpy dtypes to float32 and output numpy dtypes to uint8"""

    def astype(arg, t, depth=0):
        if isinstance(arg, numpy.ndarray):
            return arg.astype(t)
        if isinstance(arg, list) and depth < 2:
            return [astype(v, t, depth + 1) for v in arg]
        if isinstance(arg, tuple) and depth < 2:
            return tuple([astype(v, t, depth + 1) for v in arg])
        return arg

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        args = [astype(v, numpy.float32) for v in args]
        kwargs = {k: astype(v, numpy.float32) for k, v in kwargs}
        data = function(*args, **kwargs)
        return astype(data, numpy.uint8)

    return wrapper


def moving_stdev_wrapped(window, array=None) -> numpy.ndarray:
    return moving_stdev(array, window)


@wrapper_f32_u8
def moving_stdev(array: numpy.ndarray, window: int, min_count: int = 1) -> numpy.ndarray:
    """
    Quickly apply a moving standard deviation calc over an array
    :param array: input array to apply over
    :param window: number of elements to apply the stdev method_types to must be > 1
    :param min_count:
    :return:
    """
    return bottleneck.move_std(array, window=window, min_count=min_count, axis=1, )[:-window + 1, window - 1:]


def resize_list_of_arrays(array_list: list[numpy.ndarray]) -> list[numpy.ndarray]:
    """ resize all arrays given  to the size of the smallest array
    will attempt to remove the same amount from each side of the array
    :param array_list:
    :return:
    """
    if len(set([i.shape for i in array_list])) <= 1:  # if arrays are already of equal size return
        return array_list
    min_x = min([arr.shape[0] for arr in array_list])  # find minimum width
    min_y = min([arr.shape[1] for arr in array_list])  # find minimum height
    out_lst = []
    for arr in array_list:
        a_x, a_y = arr.shape[0], arr.shape[1]
        if min_x == a_x and min_y and a_y:  # fix for if an array is already the correct size..
            out_lst.append(arr)
            continue
        dx1, dx2 = math.floor((a_x - min_x) / 2), math.ceil((a_x - min_x) / 2)
        dy1, dy2 = math.floor((a_y - min_y) / 2), math.ceil((a_y - min_y) / 2)
        out_lst.append(arr[dx1: -dx2, dy1: -dy2])
    first_item = out_lst[0]
    completed_array = [arr for arr in out_lst if arr.shape == first_item.shape]
    if len(array_list) != len(completed_array):
        raise AssertionError("Input and output arrays are of different size")
    return completed_array


@wrapper_f32_u8
def combine_avg(array_list: list[numpy.ndarray], *_) -> numpy.ndarray:
    if len(array_list) == 1:
        return array_list[0]
    elif len(array_list) == 0:
        raise ValueError('array_list in combine_array_list is empty')
    array_list = resize_list_of_arrays(array_list)

    if any([numpy.max(a) == 0 for a in array_list]):
        raise ZeroDivisionError('in combine method_types avg')

    array_sum = sum(array_list)
    if numpy.max(array_sum) == 0:
        raise ZeroDivisionError('in combine method_types avg')
    arr = array_sum / len(array_list)
    return arr


@wrapper_f32_u8
def combine_dist(array_list: list[numpy.ndarray], *, /, reverse=False) -> numpy.ndarray:
    """
    dist outputs: array_list = [5,10,25] len(array_list) = 3
    ( 5*1/3*1 + 10*1/3*2 + 25*1/3*3 ) / 3 = 11.111
    dist will output a combination closer to the final inputs given
    this can be called on a list of integers or floats for testing purposes
    """
    if len(array_list) == 1:
        return array_list[0]
    elif len(array_list) == 0:
        raise ValueError('array_list in combine_array_list is empty')
    array_list = resize_list_of_arrays(array_list)
    if reverse:
        array_list = array_list[::-1]
    array_list = array_list
    array_list_max = max([numpy.max(a) for a in array_list])
    dist_arr = sum([arr * 1 / len(array_list) * i + 1 for i, arr in enumerate(array_list)])
    return numpy.multiply(dist_arr, array_list_max / numpy.max(dist_arr))


@wrapper_f32_u8
def combine_pow(array_list: list[numpy.ndarray], *, /, reverse=False) -> numpy.ndarray:
    if len(array_list) == 1:
        return array_list[0]
    elif len(array_list) == 0:
        raise ValueError('array_list in combine_array_list is empty')
    array_list = resize_list_of_arrays(array_list)
    if reverse:
        array_list = array_list[::-1]
    array_list = array_list
    array_list_max = max([numpy.max(a) for a in array_list])
    pow_arr = sum([numpy.power(arr, 1 + i / len(array_list)) for i, arr in enumerate(array_list)])
    return numpy.multiply(pow_arr, array_list_max / numpy.max(pow_arr))


@wrapper_f32_u8
def minus_floor(array: numpy.ndarray, use_max: bool = True) -> numpy.ndarray:
    if use_max:
        rolled = numpy.maximum(numpy.roll(array, 2, 2), numpy.roll(array, 1, 2))
    else:
        rolled = numpy.minimum(numpy.roll(array, 2, 2), numpy.roll(array, 1, 2))
    sub = numpy.where(array > rolled, array - rolled, 0)
    return sub


@wrapper_f32_u8
def expand(array: numpy.ndarray, *_) -> numpy.ndarray:
    """
    :rtype: object
    """
    return numpy.multiply(array, 255 / numpy.max(array))


@wrapper_f32_u8
def sharpen_fft(image: numpy.ndarray, c_size: float = 0.3) -> numpy.ndarray:
    """
    :param image: numpy array image to be sharpened
    :param c_size: amount of the resulting fft to hide
    """
    assert 0 < c_size < 1
    axes = [0, 1]
    transformed: numpy.ndarray = numpy.fft.fft2(image, axes=axes)
    sh = transformed.shape

    i_size = (1 - c_size) / 2  # percent of image that lays out side the 0block on each side
    sh_mod = floor(sh[0] * i_size), floor(sh[1] * i_size)  # int size of image that lays outside the 0 block
    slicer = numpy.s_[sh_mod[0]:-sh_mod[0], sh_mod[1]:-sh_mod[1]]  # slicer[ c%:-c%, c%:-c% ]

    transformed[slicer] = numpy.zeros_like(transformed[slicer], dtype=complex)

    transformed = numpy.fft.ifft2(transformed, axes=axes).real
    return transformed
