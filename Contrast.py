import math
from collections.abc import Callable

import bottleneck
import cv2
import numpy


def apply_rgb(function: Callable, array: numpy.ndarray, *args, **kwargs):
    """
    Apply a fuinction to an RGB image, uses cv2 split and merge, everything except function and array is
    passed to the function as an argument
    :param function: function to apply over each colour of the array
    :param array: image array
    :param kwargs: args for the function
    :return:
    """
    return cv2.merge([function(a, *args, **kwargs) for a in cv2.split(array)])


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

