from __future__ import annotations

import pathlib
import typing
from functools import partial

import numpy
from cv2 import cv2

import IO
import MulticoreProcessing
from image_process import manipulation

_T_bool = tuple[bool, bool, bool]


class ImageCache:
    image: numpy.ndarray

    def __init__(
            self, file_name: pathlib.Path, scale_factor: float = 1, modify_filename=False
    ):
        # Load img on init
        try:
            self.image: numpy.ndarray = IO.load_image(file_name)
        except AttributeError as ae:
            if isinstance(file_name, numpy.ndarray):
                self.image: numpy.ndarray = file_name
                file_name = (
                    "FILE GIVEN AS numpy.ndarray UPDATE FILENAME\nuse img.filename"
                )
            else:
                raise ae
        self.file_name = file_name
        self.modify_filename = modify_filename
        print(f"Input:  {hash(self)}    Base    {self.file_name}")
        self.moving_stdev = manipulation.moving_stdev

        if scale_factor != 1:
            self.scale_image(scale_factor)

        # dictionary used to cache img versions '' Tuple[window, min_count]
        self.dev_dict: dict[tuple[int, int], numpy.ndarray] = {}

    def __call__(self, window: int, min_count: int = 1):
        method_id = window, min_count
        if method_id in self.dev_dict:
            return self.dev_dict[method_id]
        stdev: numpy.ndarray = manipulation.moving_stdev(self.image, window, min_count)
        # stdev = numpy.multiply(stdev.astype(numpy.uint8), 2)
        self.dev_dict[method_id] = stdev
        return self.dev_dict[method_id]

    def __hash__(self):
        return hash(str(self.image))

    def scale_image(self, scale_factor):
        shape = self.image.shape[:2]
        shape = int(shape[1] * scale_factor), int(shape[0] * scale_factor)
        self.image = cv2.resize(self.image, shape, interpolation=cv2.INTER_LANCZOS4)
        self.dev_dict: dict[tuple[int, int], numpy.ndarray] = {}

    def output_file(self, method_name):
        print(f"Output: {hash(self)}    {method_name}  {self.file_name}")
        if self.modify_filename:
            return (
                    self.file_name.parent
                    / "Output"
                    / f"{self.file_name.stem}_{method_name}{self.file_name.suffix}"
            )
        return self.file_name.parent / method_name / self.file_name.parts[-1]

    def calc_division_list(self, divisions: list[int]):
        divisions = list(set(divisions))
        function = partial(manipulation.moving_stdev_wrapped, array=self.image)
        devs = MulticoreProcessing.quick_pool(function=function, data_list=divisions)
        for div, ret in zip(divisions, devs):
            self.dev_dict[(div, 1)] = ret

    def image_byte_size(self):
        return self.image.nbytes


def apply_method(cache: ImageCache, method) -> numpy.ndarray:
    if isinstance(method, int):
        return cache(method)
    elif isinstance(method, DefinedTypes):
        return method.apply(image_cache=cache)


def get_method_counts(method):
    pre = list()
    if isinstance(method, (tuple, list)):
        for m in method:
            pre.extend(get_method_counts(m))
        return pre
    for m in method.children():
        if isinstance(m, int):
            pre.append(m)
        else:
            pre.extend(get_method_counts(m))
    return pre


class _Callable:
    apply: typing.Callable[[ImageCache], numpy.ndarray]
    apply_hsv: typing.Callable[[ImageCache], numpy.ndarray]
    apply_bgr: typing.Callable[[ImageCache], numpy.ndarray]
    hsv_partial: typing.Callable[[bool, bool, bool], object]
    bgr_partial: typing.Callable[[bool, bool, bool], object]
    hsv: typing.Callable[[None], _OptionalReturns]
    bgr: typing.Callable[[None], _OptionalReturns]
    _hsv_: tuple[bool, bool, bool] = (True, True, True)
    _bgr_: tuple[bool, bool, bool] = (True, True, True)

    def hsv(self) -> _OptionalReturns:
        """convert to hsv colour space before applying parent function, then return to base colour space"""
        self.apply = self.apply_hsv
        return self

    def bgr(self) -> _OptionalReturns:
        """convert to bgr colour space before applying parent function, then return to base colour space"""
        self.apply = self.apply_bgr
        return self

    def hsv_partial(self, h_=True, s_=True, v_=True) -> _OptionalReturns:
        """
        convert to hsv colour space before applying parent function\n
        `h_, s_, v_` each bool represents a colour segment, \n
        True values will have the parent method applied to them \n
        False values will be placed back into the outputted img before returning \n
        EG: apply a sharpening method over only the saturation value: `Sharpen.hsv_partial(False, True, False)`

        :return: Self
        """
        self.apply = self.apply_hsv
        self._hsv_ = (h_, s_, v_)
        return self

    def bgr_partial(self, b_=True, g_=True, r_=True) -> _OptionalReturns:
        """
        convert to bgr colour space before applying parent function \n
        `b_, g_, r_` each bool represents a colour segment, \n
        True values will have the parent method applied to them \n
        False values will be placed back into the outputted img before returning \n
        EG: apply a sharpening method over only the blue and red values: `Sharpen.bgr_partial(False, True, False)`

        :return: Self
        """
        self.apply = self.apply_bgr
        self._bgr_ = (b_, g_, r_)
        return self


class _ProtocolSingle(typing.Protocol):
    # https://mypy.readthedocs.io/en/stable/protocols.html#callback-protocols
    def __call__(self, arrays: list[numpy.ndarray], *args, **kwargs) -> numpy.ndarray:
        ...


class _CallableMulti(_Callable):
    method: tuple[DefinedTypes]
    opt: typing.Optional[typing.Any] = None
    func: _ProtocolSingle

    _func: typing.Callable[..., numpy.ndarray]

    # _func: typing.Callable[[list[numpy.ndarray], typing.Any, `kwargs of some type`], numpy.ndarray]

    def children(self) -> typing.Iterable:
        return self.method

    def __hash__(self) -> int:
        return hash((self.method, self.opt, self._func.__name__))

    def apply_hsv(self, image_cache: ImageCache):
        return self._func(
            [apply_method(image_cache, m) for m in self.method],
            self.opt,
            hsv=self._hsv_,
        )

    def apply_bgr(self, image_cache: ImageCache):
        return self._func(
            [apply_method(image_cache, m) for m in self.method],
            self.opt,
            bgr=self._bgr_,
        )

    apply = apply_bgr


class _CallableSingle(_Callable):
    method: DefinedTypes
    opt: typing.Optional[typing.Any] = None
    _func: typing.Callable[..., numpy.ndarray]

    # _func: typing.Callable[[numpy.ndarray, typing.Any, `kwargs of some type`], numpy.ndarray]

    def children(self) -> typing.Iterable:
        return [self.method]

    def __hash__(self) -> int:
        return hash((self.method, self.opt, self._func.__name__))

    def apply_hsv(self, image_cache: ImageCache):
        return self._func(
            apply_method(image_cache, self.method), self.opt, hsv=self._hsv_
        )

    def apply_bgr(self, image_cache: ImageCache):
        return self._func(
            apply_method(image_cache, self.method), self.opt, bgr=self._bgr_
        )

    apply = apply_bgr


# --- MULTI ---


class Average(_CallableMulti):
    hsv_partial = None
    bgr_partial = None

    def __init__(self, *method: DefinedTypes):
        self.method = method
        self._func = manipulation.combine_avg


class Distribute(_CallableMulti):
    hsv_partial = None
    bgr_partial = None

    def __init__(self, *method: DefinedTypes, reverse: bool = False):
        """
        Distribute will take a list of image methods and return a weighted average of them,
        the last item will have the strongest impact on the output image

        pseudo code: sum([ im/len(method)*n for n, im in enumerate(method)]

        :param reverse: reverse the list of items before application
        """
        self.method = method
        self.opt = reverse
        self._func = manipulation.combine_dist


class Power(_CallableMulti):
    hsv_partial = None
    bgr_partial = None

    def __init__(self, *method: DefinedTypes, reverse: bool = False):
        """
        Power will take a list of image methods and return a weighted average of them,
        the final item will have the strongest impact on the output image

        pseudo code: sum([ im ** n+1 for n, im in enumerate(method)]

        :param reverse: reverse the list of items before application
        """
        self.method = method
        self.opt = reverse
        self._func = manipulation.combine_pow


class InsertChannelHSV(_CallableMulti):
    hsv_partial = None
    bgr_partial = None

    def __init__(
            self,
            base_: DefinedTypes,
            /,
            h: _Opt_Type = None,
            s: _Opt_Type = None,
            v: _Opt_Type = None,
    ):
        """
        override channels of an img with the same channel from another image

        :param base_: method to use as the base of this image
        :param h: method use to override the hue channel of the base image,
            None ignores this channel using the base image instead
        :param s: method use to override the saturation channel of the base image,
            None ignores this channel using the base image instead
        :param v: method use to override the value channel of the base image,
            None ignores this channel using the base image instead
        """
        self.base = base_
        self.new_channels = (h, s, v)

    def apply(self, image_cache: ImageCache) -> numpy.ndarray:
        base_ = apply_method(image_cache, self.base)
        new_channels = [
            apply_method(image_cache, method) if method is not None else None
            for method in self.new_channels
        ]
        return manipulation.insert_channel(base_, new_channels, _hsv_=self._hsv_)


class InsertChannelBGR(_CallableMulti):
    hsv_partial = None
    bgr_partial = None

    def __init__(
            self,
            base_: DefinedTypes,
            /,
            b: _Opt_Type = None,
            g: _Opt_Type = None,
            r: _Opt_Type = None,
    ):
        """
        override channels of an img with the same channel from another image

        :param base_: method to use as the base of this image
        :param b: method use to override the blue channel of the base image,
            None ignores this channel using the base image instead
        :param g: method use to override the green channel of the base image,
            None ignores this channel using the base image instead
        :param r: method use to override the red channel of the base image,
            None ignores this channel using the base image instead
        """
        self.base = base_
        self.new_channels = (b, g, r)

    def apply(self, image_cache: ImageCache) -> numpy.ndarray:
        base_ = apply_method(image_cache, self.base)
        new_channels = [
            apply_method(image_cache, method) if method is not None else None
            for method in self.new_channels
        ]
        return manipulation.insert_channel(base_, new_channels, _bgr_=self._bgr_)


# --- SINGLE ---


class MaxContrast(_CallableSingle):
    def __init__(self, method, /, *_):
        self.method = method
        self.opt = None
        self._func = manipulation.expand


class RollContrast(_CallableSingle):
    def __init__(self, method, /, invert: bool = False, *_):
        self.method = method
        self.opt = invert
        self._func = manipulation.minus_floor


class Sharpen(_CallableSingle):
    def __init__(self, method, /, strength: float = 0.3, *_):
        self.method = method
        self.opt = strength
        self._func = manipulation.sharpen_fft


class HSV(_CallableSingle):
    def __init__(self, n: int, /, h_=True, s_=True, v_=True, *_):
        self.method = 0
        self.window = n
        self.opt = None
        self._func = manipulation.moving_stdev
        self._hsv_ = (h_, s_, v_)

    def apply(self, image_cache: ImageCache) -> numpy.ndarray:
        pass
        return manipulation.moving_stdev(image_cache.image, self.window, hsv=self._hsv_)


class MixedHSV(_CallableSingle):
    def __init__(self, h: int, s: int, v: int, /, h_=True, s_=True, v_=True, *_):
        """allows `0 for int values, will use input values"""
        self.method = 0
        self.window = h, s, v
        self.opt = None
        self._func = manipulation.moving_stdev
        self._hsv_ = (h_, s_, v_)

    def apply(self, image_cache: ImageCache) -> numpy.ndarray:
        return manipulation.multi_moving_stdev(
            image_cache.image, self.window, hsv=self._hsv_
        )


class MixedBGR(_CallableSingle):
    def __init__(self, b: int, g: int, r: int, /, b_=True, g_=True, r_=True, *_):
        """allows `0 for int values, will use input values"""
        self.method = 0
        self.window = b, g, r
        self.opt = None
        self._func = manipulation.moving_stdev
        self._bgr_ = (b_, g_, r_)

    def apply(self, image_cache: ImageCache) -> numpy.ndarray:
        return manipulation.multi_moving_stdev(
            image_cache.image, self.window, bgr=self._bgr_
        )


class AsInput(_Callable):
    def __init__(self, /, *_, **__):
        pass

    def apply(self, image_cache: ImageCache) -> numpy.ndarray:
        return image_cache.image


_OptionalReturns = typing.Union[
    _Callable,
    Average,
    Distribute,
    Power,
    MaxContrast,
    RollContrast,
    Sharpen,
    MixedHSV,
    MixedBGR,
    InsertChannelHSV,
    InsertChannelBGR,
    HSV,
    AsInput,
]

DefinedTypes = typing.Union[
    Average,
    Distribute,
    Power,
    MaxContrast,
    RollContrast,
    Sharpen,
    MixedHSV,
    MixedBGR,
    HSV,
    InsertChannelHSV,
    InsertChannelBGR,
    AsInput,
    int,
]
_Opt_Type = typing.Optional[DefinedTypes]
MethodsMulti = {"Average": Average, "Distribute": Distribute, "Power": Power}
MethodsSingle = {
    "MaxContrast": MaxContrast,
    "RollContrast": RollContrast,
    "Sharpen": Sharpen,
}
