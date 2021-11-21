import math

import bottleneck
import numpy
from cv2 import cv2

import IO


class ImageCache:
    def __init__(self, file_name: IO.Path):
        # Load image on init
        try:
            self.image: numpy.ndarray = IO.load_image(file_name)
        except AttributeError as ae:
            if isinstance(file_name, numpy.ndarray):
                self.image: numpy.ndarray = file_name
            else:
                raise ae
        self.file_name = file_name
        print(f'Input:  {hash(self)}    Base    {self.file_name}')
        self.moving_stdev = moving_stdev
        if len(self.image.shape) == 3:
            self.moving_stdev = rgb_moving_stdev

        # dictionary used to cache image versions '' Tuple[window, min_count]
        self.dev_dict: dict[tuple[int, int], numpy.ndarray] = {}

    def __call__(self, window: int, min_count: int = 1, double_std=False, cast_to=False, **kwargs):
        method_id = window, min_count
        if method_id in self.dev_dict:
            return self.dev_dict[method_id]
        stdev: numpy.ndarray = moving_stdev(self.image, window, min_count)
        if double_std:
            stdev = numpy.multiply(stdev, 2, )
        if cast_to:
            stdev = stdev.astype(numpy.uint8)
        self.dev_dict[method_id] = stdev
        return self.dev_dict[method_id]

    def __hash__(self):
        return hash(str(self.image))

    def output_file(self, sub_folder_name):
        print(f'Output: {hash(self)}    {sub_folder_name}  {self.file_name}')
        return self.file_name.parent / sub_folder_name / self.file_name.parts[-1]


def moving_stdev(array: numpy.ndarray, window: int, min_count: int = 1) -> numpy.ndarray:
    """
    Quickly apply a moving standard deviation calc over an array
    :param array: input array to apply over
    :param window: number of elements to apply the stdev method to must be > 1
    :param min_count:
    :return:
    """
    return bottleneck.move_std(array, window=window, min_count=min_count, axis=1, )[:-window + 1, window - 1:]


def rgb_moving_stdev(array: numpy.ndarray, window: int, min_count: int = 1) -> numpy.ndarray:
    """
    apply moving stdev over an rgb array
    """
    return cv2.merge([moving_stdev(a, window, min_count) for a in cv2.split(array)])


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
        if min_x == a_x and min_y and a_y:  # fix for if an array is of correct size..
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


def combine_array_list(array_list: list[numpy.ndarray], method: str = "avg") -> numpy.ndarray:
    """
    :param array_list:
    :param method: ['sum','avg','dist'] prepend '-' to reverse list before application
    :return:
    """
    array_list = resize_list_of_arrays(array_list)
    if method[0] == '-':
        method = method[1:]
        array_list = array_list[::-1]

    if method == 'avg':
        return sum(array_list) / len(array_list)
    if method == 'dist':
        """ dist outputs: array_list = [5,10,25] len(array_list) = 3  
            ( 5*1/3*1 + 10*1/3*2 + 25*1/3*3 ) / 3 = 11.111 
            dist will output a combination closer to the final inputs given
            this can be called on a list of integers or floats for testing purposes
        """
        dist_arr = sum([arr * 1 / len(array_list) * i + 1 for i, arr in enumerate(array_list)])
        return numpy.multiply(dist_arr, 256 / numpy.max(dist_arr))
    if method == 'pow':
        """ pow outputs: array_list = [5,10,25] len(array_list = 3
            
        """
        pow_arr = sum([numpy.power(arr, 1 + i / len(array_list)) for i, arr in enumerate(array_list)])
        return numpy.multiply(pow_arr, 256 / numpy.max(pow_arr))
    else:
        raise ValueError(f"method argument invalid: {method}")
