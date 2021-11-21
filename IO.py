import pathlib
from pathlib import Path

from cv2 import cv2
import numpy


def get_list_of_files(directory: pathlib.Path) -> list[pathlib.Path]:
    """ Return a list of files below the given directory """
    assert directory.is_dir()
    file_list = []
    for item in directory.iterdir():
        if item.is_file():
            file_list.append(item)
        elif item.is_dir():
            file_list.extend(get_list_of_files(item))
    return file_list


test_file_list: list[pathlib.Path] = get_list_of_files(pathlib.Path('TestFiles'))


def load_image(file: pathlib.Path, rgb=True) -> numpy.ndarray:
    """ get image from disk using cv2 """
    if rgb:
        return cv2.imread(file.as_posix(), cv2.IMREAD_COLOR)
    return cv2.imread(file.as_posix(), cv2.IMREAD_GRAYSCALE)


def export_image(url: pathlib.Path, data: numpy.ndarray):
    """
    Export image to disk using cv2 imwrite
    :param url: url to place the file, including extension "here/this.jpg"
    :param data: numpy array containing image data
    """
    url.parent.mkdir(parents=True, exist_ok=True)  # make sure theres somewhere to save the image
    cv2.imwrite(url.as_posix(), data)


def assign_path(path_string: str, assert_file: bool = False, assert_extension: str or None = None):
    """
    :param path_string:
    :param assert_file: raise ValueError if the path does not a have a file extension
    :param assert_extension: return ValueError if the path does not have the given file extension eg: ".jpg"
    :return:
    """
    file = Path(path_string)
    if assert_file and file.suffix == "":
        raise ValueError(f"Extension not found")
    if assert_extension and file.suffix != assert_extension:
        raise ValueError(f"Extension does not match. got {file.suffix} expected {assert_extension}")
    return file
