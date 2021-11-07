import pathlib
from pathlib import Path

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


def load_video(file: pathlib.Path) -> numpy.ndarray:
    try:
        capture = cv2.VideoCapture(file.__str__())
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buffer = numpy.empty((frame_count, frame_height, frame_width, 3))

        frame = 0
        return_val = True
        while frame < frame_count and return_val:
            return_val, buffer[frame] = capture.read()
            frame += 1
        capture.release()
        print(f'Image: {buffer.shape}')
        return buffer
    except FileNotFoundError as fnf:
        print(f'FileNotFoundError: \n\t\t{file}  \n{fnf}')
        exit(1)


def export_video(url: pathlib.Path, data: numpy.ndarray, fps=30):
    url.parent.mkdir(parents=True, exist_ok=True)
    if url.suffix != '.avi':
        print(f"Fixing suffix without error: from {url.suffix}")
        url = url.with_suffix('.avi')

    try:
        data = data.astype('uint8')
        out = cv2.VideoWriter(filename=url.__str__(),
                              fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                              fps=fps,
                              frameSize=(data.shape[2], data.shape[1]))
        for frame in data:
            out.write(frame)

        out.release()

    except FileNotFoundError as fnf:
        print(f'FileNotFoundError: \n\t\t{url}  \n{fnf}')
        exit(1)


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
