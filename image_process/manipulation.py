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

BGR: (Blue, Green, Red) img format
HSV: (Hue, Saturation, Value) img format

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
    :param ty: type to check against: applied as `isinstance(img[n], ty)`
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


def float_01(img: numpy.ndarray) -> numpy.ndarray:
    """scale any input img to an img of float values between 0 and 1"""
    if not isinstance(img, numpy.ndarray):
        raise NotNumpyException()
    if img.dtype != f32:
        img = img.astype(f32)
    if numpy.min(img) < 0:
        img += numpy.min(img)
    mx = numpy.max(img)
    if mx > 1:
        if u16_max / 2 < mx <= u8_max:
            multiplier = u8_max
        elif u16_max / 2 < mx <= u16_max:
            multiplier = u16_max
        else:
            multiplier = mx
        img *= 1 / multiplier
    return img


def scale_u8(img: numpy.ndarray):
    """scale any input img to an img of unsigned integers between 0 and 255"""
    if numpy.min(img) < 0:
        img += numpy.min(img)
    mx = numpy.max(img)
    if mx < 1:
        mul = u8_max
    else:
        mul = 1 / mx * u8_max
    img *= mul
    return img


def bgr_to_hsv1(img: numpy.ndarray) -> numpy.ndarray:
    """convert bgr f32 img to hsv1 with ranges [0-1,0-1,0-1]"""
    img = float_01(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img[:, :, 0] *= 1 / 360
    return img


def hsv1_to_bgr(img: numpy.ndarray) -> numpy.ndarray:
    """convert hsv1 f32 img with ranges [0-1,0-1,0-1] to bgr"""
    img = float_01(img)
    mx = numpy.max(img[:, :, 0])
    img[:, :, 0] *= 360 * mx
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


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
        overwrite the third dimension of img ``old`` with the same dimension of img ``new``
        based on the truth values in tuple ``t``

        example::
            t = (True, False, True)
            output_array = numpy.dstack([new[:, :, 0], old[:, :, 1], new[:, :, 2]])

        :param old: numpy img with shape (x, y, 3)
        :param new: numpy img with shape (x, y, 3)
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
        # at this point splitting is required (not applicable for lists of arrays)
        # splitting will be required, only allow 1 in 1 out arrays
        # Array must be the first or only item in args
        # Array must be the first or only item in the function return
        args = list(args)
        if not isinstance(args[0], numpy.ndarray):
            raise ColourWrapperError("first input must be ndarray")
        if any([isinstance(x, numpy.ndarray) for x in args[1:]]):
            raise ColourWrapperError("too many arrays in input args")

        assert isinstance(args[0], numpy.ndarray)

        if as_hsv:
            # if required convert the first item (hopefully the  numpy img) to hsv
            args[0] = bgr_to_hsv1(args[0])
        r_args = function(*args, **kwargs)  # this is the main function call

        # split out the returned numpy img if it is found, or raise error
        if isinstance(r_args, numpy.ndarray):
            post_arr = r_args
            extend = None
        elif isinstance(r_args[0], numpy.ndarray):
            post_arr = r_args
            extend = r_args[1:]
        else:  # img isnt the first, or only item returned
            raise ColourWrapperError("output ndarray not found")
        if isinstance(r_args, (tuple, list)) and any(
                [isinstance(x, numpy.ndarray) for x in r_args[1:]]
        ):
            raise ColourWrapperError("too many arrays in output args")

        returns = reform_arr(
            args[0], post_arr, t=breaks
        )  # reformat img with old elements where needed
        if as_hsv:
            returns = hsv1_to_bgr(returns)
        # return single img, or return img and single elements
        if extend is None:
            return returns
        return returns, *extend

    return wrapper


def moving_stdev_wrapped(window, img=None, *_, **__) -> numpy.ndarray:
    return moving_stdev(img, window)


@wrapper_f32_u8
@wrapper_colour_mods
def moving_stdev(img: numpy.ndarray, window: int, *_, **__) -> numpy.ndarray:
    """
    Quickly apply a moving standard deviation calc over an img
    :param img: input img to apply over
    :param window: number of elements to apply the stdev method_types to must be > 1
    :return:
    """
    return bottleneck.move_std(
        img,
        window=window,
        axis=1,
    )[: -window + 1, window - 1:]


@wrapper_f32_u8
@wrapper_colour_mods
def multi_moving_stdev(
        array: numpy.ndarray, windows=tuple[int, int, int], *_, **__
) -> numpy.ndarray:
    """
    Apply a windowed standard deviation calculation over an img
    with different window sizes for each of the last dimensions
    if any window size is 0 it will return that input without modification
    """
    m_w = max(windows)
    together = numpy.dstack(
        [
            bottleneck.move_std(array[..., i_], w_, axis=1) if w_ else array[..., i_]
            for i_, w_ in enumerate(windows)
        ]
    )[: -m_w + 1, m_w - 1:]
    return together


def resize_list_of_arrays(img_list: list[numpy.ndarray]) -> list[numpy.ndarray]:
    """resize all arrays given  to the size of the smallest img
    will attempt to remove the same amount from each side of the img
    :param img_list:
    :return:
    """

    if len(set([i.shape for i in img_list])) <= 1:
        # if arrays are already of equal size return
        return img_list
    min_x = min([arr.shape[0] for arr in img_list])  # find minimum width
    min_y = min([arr.shape[1] for arr in img_list])  # find minimum height
    out_lst = []
    for arr in img_list:
        a_x, a_y = arr.shape[0], arr.shape[1]
        if (
                min_x == a_x and min_y and a_y
        ):  # fix for if an img is already the correct size..
            out_lst.append(arr)
            continue
        dx1, dx2 = math.floor((a_x - min_x) / 2), math.ceil((a_x - min_x) / 2)
        dy1, dy2 = math.floor((a_y - min_y) / 2), math.ceil((a_y - min_y) / 2)
        out_lst.append(arr[dx1:-dx2, dy1:-dy2])
    first_item = out_lst[0]
    complete_img = [arr for arr in out_lst if arr.shape == first_item.shape]
    if len(img_list) != len(complete_img):
        raise AssertionError("Input and output arrays are of different size")
    return complete_img


def scale_image(img: numpy.ndarray, scale_factor: float):
    """ for a given img resize the img using Lanczos interpolation """
    shape = img.shape[:2]
    shape = int(shape[1] * scale_factor), int(shape[0] * scale_factor)
    return cv2.resize(img, shape, interpolation=cv2.INTER_LANCZOS4)


@wrapper_f32_u8
def combine_avg(img_list: list[numpy.ndarray], *_, **__) -> numpy.ndarray:
    """ return sum(img_list) / len(img_list) """
    if len(img_list) == 1:
        return img_list[0]
    elif len(img_list) == 0:
        raise ValueError("img_list in combine_array_list is empty")
    img_list = resize_list_of_arrays(img_list)

    if any([numpy.max(a) == 0 for a in img_list]):
        raise ZeroDivisionError("in combine method_types avg")

    array_sum = sum(img_list)
    if numpy.max(array_sum) == 0:
        raise ZeroDivisionError("in combine method_types avg")
    arr = array_sum / len(img_list)
    return arr


@wrapper_f32_u8
def combine_dist(
        img_list: list[numpy.ndarray], reverse=False, *_, **__
) -> numpy.ndarray:
    """
    dist outputs: img_list = [5,10,25] len(img_list) = 3
    ( 5*1/3*1 + 10*1/3*2 + 25*1/3*3 ) / 3 = 11.111
    dist will output a combination closer to the final inputs given
    this can be called on a list of integers or floats for testing purposes
    """
    if len(img_list) == 1:
        return img_list[0]
    elif len(img_list) == 0:
        raise ValueError("img_list in combine_array_list is empty")
    img_list = resize_list_of_arrays(img_list)
    if reverse:
        img_list = img_list[::-1]
    img_list = img_list
    array_list_max = max([numpy.max(a) for a in img_list])
    dist_arr = sum(
        [arr * 1 / len(img_list) * i + 1 for i, arr in enumerate(img_list)]
    )
    return numpy.multiply(dist_arr, array_list_max / numpy.max(dist_arr))


@wrapper_f32_u8
def combine_pow(
        img_list: list[numpy.ndarray], reverse=False, *_, **__
) -> numpy.ndarray:
    if len(img_list) == 1:
        return img_list[0]
    elif len(img_list) == 0:
        raise ValueError("img_list in combine_array_list is empty")
    img_list = resize_list_of_arrays(img_list)
    if reverse:
        img_list = img_list[::-1]
    img_list = img_list
    array_list_max = max([numpy.max(a) for a in img_list])
    pow_arr = sum(
        [numpy.power(arr, 1 + i / len(img_list)) for i, arr in enumerate(img_list)]
    )
    return numpy.multiply(pow_arr, array_list_max / numpy.max(pow_arr))


@wrapper_f32_u8
@wrapper_colour_mods
def insert_channel(
        base: numpy.ndarray, over_ride: list[numpy.ndarray | None], *_, **__
):
    over_bool = [a is not None for a in over_ride]  # if truu, value should be array item
    base, *over_ride = resize_list_of_arrays(
        [base, *[o for o in over_ride if o is not None]]
    )
    it = iter(over_ride)
    resized = [next(it) if b else None for b in over_bool]

    for i, arr in enumerate(resized):
        if arr is None:
            continue
        base[..., i] = arr[..., i]
    return base


@wrapper_f32_u8
@wrapper_colour_mods
def minus_floor(img: numpy.ndarray, use_max: bool = True, *_, **__) -> numpy.ndarray:
    if use_max:
        rolled = numpy.maximum(numpy.roll(img, 2, 2), numpy.roll(img, 1, 2))
    else:
        rolled = numpy.minimum(numpy.roll(img, 2, 2), numpy.roll(img, 1, 2))
    sub = numpy.where(img > rolled, img - rolled, 0)
    return sub


@wrapper_f32_u8
@wrapper_colour_mods
def expand(img: numpy.ndarray, *_, **__) -> numpy.ndarray:
    """
    :rtype: object
    """
    return numpy.multiply(img, 255 / numpy.max(img))


@wrapper_f32_u8
@wrapper_colour_mods
def sharpen_fft(img: numpy.ndarray, c_size: float = 0.3, *_, **__) -> numpy.ndarray:
    """
    :param img: numpy img image to be sharpened
    :param c_size: amount of the resulting fft to hide
    """
    assert 0 < c_size < 1
    axes = [0, 1]
    transformed: numpy.ndarray = numpy.fft.fft2(img, axes=axes)
    sh = transformed.shape

    i_size = (
                     1 - c_size
             ) / 2  # percent of img that lays out side the 0block on each side
    sh_mod = floor(sh[0] * i_size), floor(
        sh[1] * i_size
    )  # int size of img that lays outside the 0 block
    slicer = numpy.s_[
             sh_mod[0]: -sh_mod[0], sh_mod[1]: -sh_mod[1]
             ]  # slicer[ c%:-c%, c%:-c% ]

    transformed[slicer] = numpy.zeros_like(transformed[slicer], dtype=complex)

    transformed = numpy.fft.ifft2(transformed, axes=axes).real
    return transformed
