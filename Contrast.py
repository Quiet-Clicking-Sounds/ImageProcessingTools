import math
from collections.abc import Callable

import bottleneck
import cv2
import numpy


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


def moving_stdev(array: numpy.ndarray,
                 window: int,
                 min_count: int = 1,
                 axis: int = -1) -> numpy.ndarray:
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
        dx1, dx2 = math.floor((a_x - min_x) / 2), math.ceil((a_x - min_x) / 2)
        dy1, dy2 = math.floor((a_y - min_y) / 2), math.ceil((a_y - min_y) / 2)
        out_lst.append(arr[dx1: -dx2, dy1: -dy2])
    first_item = out_lst[0]
    return [arr for arr in out_lst if arr.shape == first_item.shape]


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
    :param method: ['sum','avg','dist']
    :param inverse: reverse list direction
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
        return sum([arr * 1 / len(array_list) * i for i, arr in enumerate(array_list)])
    else:
        raise ValueError(f"method argument invalid: {method}")
