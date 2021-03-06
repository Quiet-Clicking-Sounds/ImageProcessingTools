import math
from collections.abc import Callable

import bottleneck
import numpy
from cv2 import cv2


class ImageCache:
    def __init__(self, image: numpy.ndarray):
        self.image: numpy.ndarray = image

        self.moving_stdev = moving_stdev
        if len(self.image.shape) == 3:
            self.moving_stdev = rgb_moving_stdev

        # dictionary used to cache image versions
        self.dev_dict: dict[tuple[int, int, int], numpy.ndarray] = {}

    def __call__(self, window: int, min_count: int = 1, axis: int = -1):
        method_id = window, min_count, axis
        if method_id in self.dev_dict:
            return self.dev_dict[method_id]
        self.dev_dict[method_id] = moving_stdev(self.image, window, min_count, axis)
        return self.dev_dict[method_id]


def apply(function: Callable, array: numpy.ndarray, *args, **kwargs):
    """
    Apply a fuinction to an RGB image, uses cv2 split and merge, everything except function and array is
    passed to the function as an argument
    :param function: function to apply over each colour of the array
    :param array: image array
    :param kwargs: cl_args for the function
    :return:
    """
    if len(array.shape) == 3:
        return cv2.merge([function(a, *args, **kwargs) for a in cv2.split(array)])
    return function(array, *args, **kwargs)


def moving_stdev(array: numpy.ndarray, window: int, min_count: int = 1, axis: int = -1) -> numpy.ndarray:
    """
    Quickly apply a moving standard deviation calc over an array
    :param array: input array to apply over
    :param window: number of elements to apply the stdev method to must be > 1
    :param min_count:
    :param axis:
    :return:
    """
    return bottleneck.move_std(array,
                               window=window,
                               min_count=min_count,
                               axis=-axis,
                               )[:-window + 1, window - 1:]


def rgb_moving_stdev(array: numpy.ndarray, window: int, min_count: int = 1, axis: int = -1) -> numpy.ndarray:
    """
    apply moving stdev over an rgb array
    """
    return cv2.merge([moving_stdev(a, window, min_count, axis) for a in cv2.split(array)])


def resize_list_of_arrays(array_list: list[numpy.ndarray]) -> list[numpy.ndarray]:
    """ resize all arrays given  to the size of the smallest array
    will attempt to remove the same amount from each side of the array
    :param array_list:
    :return:
    """
    min_x = min([arr.shape[0] for arr in array_list])  # find minimum width
    min_y = min([arr.shape[1] for arr in array_list])  # find minimum height
    out_lst = []
    for arr in array_list:
        a_x, a_y = arr.shape[0], arr.shape[1]
        if min_x == a_x and min_y and a_y:  # fix for if an array is of correct size..
            out_lst.append(arr)
            continue
        dx1, dx2 = math.floor((a_x - min_x) / 2), math.ceil((a_x - min_x) / 2)
        dy1, dy2 = math.floor((a_y - min_y) / 2), math.ceil((a_y - min_y) / 2)
        out_lst.append(arr[dx1: -dx2, dy1: -dy2])
    first_item = out_lst[0]
    comepleted_list = [arr for arr in out_lst if arr.shape == first_item.shape]
    if len(array_list) != len(comepleted_list):
        raise AssertionError("Input and output arrays are of different size")
    return comepleted_list


def combine_method_options(as_str=False) -> str or list[str]:
    """ Function for use in print statements
    :return:
    """
    part = ["sum", "avg"]
    reversible_part = ["dist"]
    part.extend(reversible_part)
    part.extend([f'-{s}' for s in reversible_part])
    if as_str:
        return ', '.join(part)
    return part


def combine_array_list(array_list: list[numpy.ndarray], method: str = "sum") -> numpy.ndarray:
    """
    :param array_list:
    :param method: ['sum','avg','dist'] prepend '-' to reverse list before application
    :return:
    """
    if method[0] == '-':
        method = method[1:]
        array_list = array_list[::-1]

    if method == 'sum':
        return sum(array_list)
    if method == 'avg':
        return sum(array_list) / len(array_list)
    if method == 'dist':
        """ dist outputs: array_list = [5,10,25] len(array_list = 3  
            ( 5*1/3*1 + 10*1/3*2 + 25*1/3*3 ) / 3 = 11.111 
            dist will output a combination closer to the final inputs given
            this can be called on a list of integers or floats for testing purposes
        """
        return sum([arr * 1 / len(array_list) * i + 1 for i, arr in enumerate(array_list)])
    else:
        raise ValueError(f"method argument invalid: {method}")
