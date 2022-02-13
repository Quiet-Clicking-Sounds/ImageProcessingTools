from pathlib import Path

import numpy
from cv2 import cv2

file_extensions = ('.jpg', '.jpeg', '.gif')
tk_file_extensions = (
    ('Image Files', ' '.join([f'*{a}' for a in file_extensions]))
)


def list_files_in_directory_tree(directory: Path) -> list[Path]:
    """ Return a list of files below the given directory, searches sub-folders """
    assert directory.is_dir()
    file_list = []
    for item in directory.iterdir():
        if item.is_file():
            file_list.append(item)
        elif item.is_dir():
            file_list.extend(list_files_in_directory_tree(item))
    return file_list


def load_image(file: Path, rgb=True) -> numpy.ndarray:
    """ get image from disk using cv2 """
    if file.suffix not in ['.jpg', '.jpeg', '.gif']:
        raise FileExistsError
    if rgb:
        return cv2.imread(file.as_posix(), cv2.IMREAD_COLOR)
    return cv2.imread(file.as_posix(), cv2.IMREAD_GRAYSCALE)


def export_image(url: Path, data: numpy.ndarray):
    """
        Export image to disk using cv2.imwrite
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
