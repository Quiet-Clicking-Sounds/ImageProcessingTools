import pathlib

import cv2
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


def load_image(file: pathlib.Path, RGB=True) -> numpy.ndarray:
    """ get image from disk using cv2 """
    if RGB:
        return cv2.imread(file.as_posix(), cv2.IMREAD_COLOR)
    return cv2.imread(file.as_posix(), cv2.IMREAD_GRAYSCALE)


def export_to_image(url: pathlib.Path, data: numpy.ndarray):
    """
    Export image to disk using cv2 imwrite
    :param url: url to place the file, including extension "here/this.jpg"
    :param data: numpy array containing image data
    """
    url.parent.mkdir(parents=True, exist_ok=True)  # make sure theres somewhere to save the image
    cv2.imwrite(url.as_posix(), data)
