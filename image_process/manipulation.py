import functools
import math
import typing
from math import floor

import bottleneck
import numpy
from cv2 import cv2

"""
Note: The floating point HSV representation of images within this module use:
    H values in the range 0-1 NOT the standard 0-360

BGR: (Blue, Green, Red) image format
HSV: (Hue, Saturation, Value) image format

:param f32: numpy.float32 
:param  u8_max: maximum integer allowed in a 8 bit un-singed integer 
:param  u16_max: maximum integer allowed in a 16 bit un-singed integer 
:param mod_hsv: default parameter for which hsv channels modifications should be applied to 
:param mod_bgr: default parameter for which bgr channels modifications should be applied to 
"""

T_mod = typing.Tuple[bool, bool, bool]
f32 = numpy.float32
u8_max = 255
u16_max = 65535
mod_hsv = (True, True, True)
mod_bgr = (True, True, True)


def _apply_where(arg, fn: typing.Callable, ty: type | tuple[type], d_=3):
    """
    helper function applies a function to all arguments of type[ty]\n
    will iterate through types `list` and `tuple`

    :param arg: input arguments
    :param ty: type to check against: applied as `isinstance(arg[n], ty)`
    :param fn: function to apply to matched items
    :param d_: recursive depth allowed
    :return:
    """
    if isinstance(arg, ty):
        return fn(arg)
    if isinstance(arg, list) and d_ < 2:
        return [_apply_where(v, fn, ty, d_ - 1) for v in arg]
    if isinstance(arg, tuple) and d_ < 2:
        return tuple([_apply_where(v, fn, ty, d_ - 1) for v in arg])
    return arg


class NotNumpyException(Exception):
    """something has gone very wrong, attempting to parse non-numpy arrays shouldn't happen"""

    pass


class ColourWrapperError(Exception):
    """the colour wrapper found itself lacking"""

    pass


def float_01(arg: numpy.ndarray) -> numpy.ndarray:
    """scale any input array to an array of float values between 0 and 1"""
    if not isinstance(arg, numpy.ndarray):
        raise NotNumpyException()
    if arg.dtype != f32:
        arg = arg.astype(f32)
    if numpy.min(arg) < 0:
        arg += numpy.min(arg)
    mx = numpy.max(arg)
    if mx > 1:
        if u16_max / 2 < mx <= u8_max:
            multiplier = u8_max
        elif u16_max / 2 < mx <= u16_max:
            multiplier = u16_max
        else:
            multiplier = mx
        arg *= 1 / multiplier
    return arg


def scale_u8(arg: numpy.ndarray):
    """scale any input array to an array of unsigned integers between 0 and 255"""
    if numpy.min(arg) < 0:
        arg += numpy.min(arg)
    mx = numpy.max(arg)
    if mx < 1:
        mul = u8_max
    else:
        mul = 1 / mx * u8_max
    arg *= mul
    return arg


def bgr_to_hsv1(arg: numpy.ndarray) -> numpy.ndarray:
    """convert bgr f32 image to hsv1 with ranges [0-1,0-1,0-1]"""
    arg = float_01(arg)
    arg = cv2.cvtColor(arg, cv2.COLOR_BGR2HSV)
    arg[:, :, 0] *= 1 / 360
    return arg


def hsv1_to_bgr(arg: numpy.ndarray) -> numpy.ndarray:
    """convert hsv1 f32 image with ranges [0-1,0-1,0-1] to bgr"""
    arg = float_01(arg)
    mx = numpy.max(arg[:, :, 0])
    arg[:, :, 0] *= 360 * mx
    arg = cv2.cvtColor(arg, cv2.COLOR_HSV2BGR)
    return arg


def wrapper_f32_u8(function):
    """Wrapper function to convert input numpy dtypes to float32 and output numpy dtypes to uint8"""

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        args = _apply_where(args, float_01, numpy.ndarray)
        kwargs = _apply_where(kwargs, float_01, numpy.ndarray)

        data = function(*args, **kwargs)
        return _apply_where(data, scale_u8, numpy.ndarray)

    return wrapper


def wrapper_colour_mods(function):
    """
    Wrapper function to convert bgr images to another format, then convert back\n
    Includes methods to apply methods over sliced arrays
    """

    def reform_arr(old: numpy.ndarray, new: numpy.ndarray, *_, t: T_mod):
        """
        overwrite the third dimension of array ``old`` with the same dimension of array ``new``
        based on the truth values in tuple ``t``

        example::
            t = (True, False, True)
            output_array = numpy.dstack([new[:, :, 0], old[:, :, 1], new[:, :, 2]])

        :param old: numpy array with shape (x, y, 3)
        :param new: numpy array with shape (x, y, 3)
        :param t: tuple of values showing where new should overwrite a
        :return:
        """

        # may be faster to do inplace replacement:
        # old[:, :, n] = new[:, :, n]
        arr_list = [new[:, :, n] if x else old[:, :, n] for n, x in enumerate(t)]
        arr_list = resize_list_of_arrays(arr_list)
        return numpy.dstack(arr_list)

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        as_hsv = "hsv" in kwargs
        as_bgr = "bgr" in kwargs
        if not (as_hsv or as_bgr):  # return if kwargs not found
            return function(*args, **kwargs)
        breaks = kwargs["hsv"] if as_hsv else kwargs["bgr"]
        if (not any(breaks)) or (not all([x is True or x is False for x in breaks])):
            raise ColourWrapperError(
                f"keyword argument {'hsv' if as_hsv else 'bgr'} does not match allowed values:"
                f"current:{breaks} must be in format tuple(bool,bool,bool) with >=1 True value"
            )
        if all([not x for x in breaks]):
            raise ColourWrapperError("all options selected are False")
        if all(breaks):  # return as normal if breaks aren't being used
            if as_hsv:
                args = _apply_where(args, bgr_to_hsv1, numpy.ndarray)
                kwargs = _apply_where(kwargs, bgr_to_hsv1, numpy.ndarray)
                _rt = _apply_where(
                    function(*args, **kwargs), hsv1_to_bgr, numpy.ndarray
                )
                return _rt
            if as_bgr:
                return function(*args, **kwargs)
        # at this point splitting is required (not applicable for lists of arrays
        # splitting will be required, only allow 1 in 1 our arrays
        # array much be first in arg and first in return
        args = list(args)
        if not isinstance(args[0], numpy.ndarray):
            raise ColourWrapperError("first input must be ndarray")
        if any([isinstance(x, numpy.ndarray) for x in args[1:]]):
            raise ColourWrapperError("too many arrays in input args")

        assert isinstance(args[0], numpy.ndarray)

        if as_hsv:
            # if required convert the first item (hopefully the  numpy array) to hsv
            args[0] = bgr_to_hsv1(args[0])
        r_args = function(*args, **kwargs)  # this is the main function call

        # split out the returned numpy array if it is found, or raise error
        if isinstance(r_args, numpy.ndarray):
            post_arr = r_args
            extend = None
        elif isinstance(r_args[0], numpy.ndarray):
            post_arr = r_args
            extend = r_args[1:]
        else:  # arg isnt the first, or only item returned
            raise ColourWrapperError("output ndarray not found")
        if isinstance(r_args, (tuple, list)) and any(
                [isinstance(x, numpy.ndarray) for x in r_args[1:]]
        ):
            raise ColourWrapperError("too many arrays in output args")

        returns = reform_arr(
            args[0], post_arr, t=breaks
        )  # reformat array with old elements where needed
        if as_hsv:
            returns = hsv1_to_bgr(returns)
        # return single array, or return array and single elements
        if extend is None:
            return returns
        return returns, *extend

    return wrapper


def moving_stdev_wrapped(window, array=None, *_, **__) -> numpy.ndarray:
    return moving_stdev(array, window)


@wrapper_f32_u8
@wrapper_colour_mods
def moving_stdev(array: numpy.ndarray, window: int, *_, **__) -> numpy.ndarray:
    """
    Quickly apply a moving standard deviation calc over an array
    :param array: input array to apply over
    :param window: number of elements to apply the stdev method_types to must be > 1
    :return:
    """
    return bottleneck.move_std(
        array,
        window=window,
        axis=1,
    )[: -window + 1, window - 1:]


def resize_list_of_arrays(array_list: list[numpy.ndarray]) -> list[numpy.ndarray]:
    """resize all arrays given  to the size of the smallest array
    will attempt to remove the same amount from each side of the array
    :param array_list:
    :return:
    """
    if (
            len(set([i.shape for i in array_list])) <= 1
    ):  # if arrays are already of equal size return
        return array_list
    min_x = min([arr.shape[0] for arr in array_list])  # find minimum width
    min_y = min([arr.shape[1] for arr in array_list])  # find minimum height
    out_lst = []
    for arr in array_list:
        a_x, a_y = arr.shape[0], arr.shape[1]
        if (
                min_x == a_x and min_y and a_y
        ):  # fix for if an array is already the correct size..
            out_lst.append(arr)
            continue
        dx1, dx2 = math.floor((a_x - min_x) / 2), math.ceil((a_x - min_x) / 2)
        dy1, dy2 = math.floor((a_y - min_y) / 2), math.ceil((a_y - min_y) / 2)
        out_lst.append(arr[dx1:-dx2, dy1:-dy2])
    first_item = out_lst[0]
    completed_array = [arr for arr in out_lst if arr.shape == first_item.shape]
    if len(array_list) != len(completed_array):
        raise AssertionError("Input and output arrays are of different size")
    return completed_array


def scale_image(array: numpy.ndarray, scale_factor: float):
    shape = array.shape[:2]
    shape = int(shape[1] * scale_factor), int(shape[0] * scale_factor)
    return cv2.resize(array, shape, interpolation=cv2.INTER_LANCZOS4)


@wrapper_f32_u8
def combine_avg(array_list: list[numpy.ndarray], *_, **__) -> numpy.ndarray:
    if len(array_list) == 1:
        return array_list[0]
    elif len(array_list) == 0:
        raise ValueError("array_list in combine_array_list is empty")
    array_list = resize_list_of_arrays(array_list)

    if any([numpy.max(a) == 0 for a in array_list]):
        raise ZeroDivisionError("in combine method_types avg")

    array_sum = sum(array_list)
    if numpy.max(array_sum) == 0:
        raise ZeroDivisionError("in combine method_types avg")
    arr = array_sum / len(array_list)
    return arr


@wrapper_f32_u8
def combine_dist(
        array_list: list[numpy.ndarray], reverse=False, *_, **__
) -> numpy.ndarray:
    """
    dist outputs: array_list = [5,10,25] len(array_list) = 3
    ( 5*1/3*1 + 10*1/3*2 + 25*1/3*3 ) / 3 = 11.111
    dist will output a combination closer to the final inputs given
    this can be called on a list of integers or floats for testing purposes
    """
    if len(array_list) == 1:
        return array_list[0]
    elif len(array_list) == 0:
        raise ValueError("array_list in combine_array_list is empty")
    array_list = resize_list_of_arrays(array_list)
    if reverse:
        array_list = array_list[::-1]
    array_list = array_list
    array_list_max = max([numpy.max(a) for a in array_list])
    dist_arr = sum(
        [arr * 1 / len(array_list) * i + 1 for i, arr in enumerate(array_list)]
    )
    return numpy.multiply(dist_arr, array_list_max / numpy.max(dist_arr))


@wrapper_f32_u8
def combine_pow(
        array_list: list[numpy.ndarray], reverse=False, *_, **__
) -> numpy.ndarray:
    if len(array_list) == 1:
        return array_list[0]
    elif len(array_list) == 0:
        raise ValueError("array_list in combine_array_list is empty")
    array_list = resize_list_of_arrays(array_list)
    if reverse:
        array_list = array_list[::-1]
    array_list = array_list
    array_list_max = max([numpy.max(a) for a in array_list])
    pow_arr = sum(
        [numpy.power(arr, 1 + i / len(array_list)) for i, arr in enumerate(array_list)]
    )
    return numpy.multiply(pow_arr, array_list_max / numpy.max(pow_arr))


@wrapper_f32_u8
@wrapper_colour_mods
def minus_floor(array: numpy.ndarray, use_max: bool = True, *_, **__) -> numpy.ndarray:
    if use_max:
        rolled = numpy.maximum(numpy.roll(array, 2, 2), numpy.roll(array, 1, 2))
    else:
        rolled = numpy.minimum(numpy.roll(array, 2, 2), numpy.roll(array, 1, 2))
    sub = numpy.where(array > rolled, array - rolled, 0)
    return sub


@wrapper_f32_u8
@wrapper_colour_mods
def expand(array: numpy.ndarray, *_, **__) -> numpy.ndarray:
    """
    :rtype: object
    """
    return numpy.multiply(array, 255 / numpy.max(array))


@wrapper_f32_u8
@wrapper_colour_mods
def sharpen_fft(image: numpy.ndarray, c_size: float = 0.3, *_, **__) -> numpy.ndarray:
    """
    :param image: numpy array image to be sharpened
    :param c_size: amount of the resulting fft to hide
    """
    assert 0 < c_size < 1
    axes = [0, 1]
    transformed: numpy.ndarray = numpy.fft.fft2(image, axes=axes)
    sh = transformed.shape

    i_size = (
                     1 - c_size
             ) / 2  # percent of image that lays out side the 0block on each side
    sh_mod = floor(sh[0] * i_size), floor(
        sh[1] * i_size
    )  # int size of image that lays outside the 0 block
    slicer = numpy.s_[
             sh_mod[0]: -sh_mod[0], sh_mod[1]: -sh_mod[1]
             ]  # slicer[ c%:-c%, c%:-c% ]

    transformed[slicer] = numpy.zeros_like(transformed[slicer], dtype=complex)

    transformed = numpy.fft.ifft2(transformed, axes=axes).real
    return transformed
