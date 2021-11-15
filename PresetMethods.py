import pathlib
from collections.abc import Callable
from functools import partial

import Contrast
import IO

use_multi_core_processing = False
named_methods = {
    'cont_8': (
        ([7, 9, 11, 13, ], 'dist'), ([7, 9, 11, 13], '-dist'), ([5, 7, 9], 'avg'), ([3, 5, 7], 'avg'),
        'dist'),
    'cont_13': (
        ([7, 9, 11, 13, ], 'dist'), ([7, 9, 11, 13], '-dist'), ([5, 7, 9], 'avg'), ([3, 5, 7], 'avg'),
        '-dist'),
    'cont_24': (
        ([3, 5, 7, 9, 11, 13], 'dist'), ([3, 5, 7, 9, 11], 'dist'), ([3, 5, 7], 'dist'), 15, 25,
        '-dist'),
    'cont_48': (
        ([3, 5, 7, 9, 11, 13, 15], 'dist'), ([3, 5, 7, 9, 11, 13, 15], '-dist'), ([3, 5, 7], 'avg'), ([3, 5], 'avg'),
        25,
        'sum'),
    'add_area': (
        ([13, 19, 21], 'dist'), ([7, 9, 11], 'dist'), ([5, 7, 9], 'dist'), ([3, 5, 7], 'dist'),
        'dist'),
}


def apply_in_folder(folder: str, function: Callable, allow_sub_folders=False, **kwargs):
    def get_list_of_files(directory: pathlib.Path):
        """ return a list of files within a directory
        :param directory:
        :return:
        """
        file_list = []
        for item in directory.iterdir():
            if item.is_file():
                file_list.append(item)
            elif item.is_dir() and allow_sub_folders:
                file_list.extend(get_list_of_files(item))
        return file_list

    file_list = get_list_of_files(IO.assign_path(folder))
    if use_multi_core_processing:
        function = partial(function, **kwargs)
        quick_pool(function, file_list)
    else:
        for file in file_list:
            function(file, **kwargs)


def apply_to_file(file: pathlib.Path, method: list, sub_folder_name: str):
    image = IO.load_image(file)
    image = Contrast.ImageCache(image)
    data_list = []

    for parse in method:
        if isinstance(parse, tuple):
            data = [image(w) for w in parse[0]]
            data = Contrast.resize_list_of_arrays(data)
            data_list.append(Contrast.combine_array_list(data, parse[1]))
        elif isinstance(parse, int):
            data_list.append(image(window=parse))

    data = Contrast.resize_list_of_arrays(data_list)
    output_image = Contrast.combine_array_list(data, method[-1])

    file_output = file.parent / sub_folder_name / file.parts[-1]
    IO.export_image(file_output, output_image)


# --------------------------------------------------------------------------------------------------------------
# ---------------------------Multi Processing Function ---------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
from multiprocessing import Pool
from os import cpu_count

core_count = max(cpu_count() - 2, 1)


def quick_pool(function: Callable, data_list: list) -> list:
    with Pool(core_count) as p:
        return p.map(function, data_list)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", dest="directory", help="directory of files to apply methods to")
    parser.add_argument("-f", "--function", dest="function",
                        help="functions to apply over each file 'add area' use ',' as separator or 'all'",
                        default='all')
    parser.add_argument("-asf", "--allow_sub_folders", dest="sub_folder",
                        help="run function on all sub-folders of the given directory",
                        action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-m", "--multicore", dest="multicore", help="use multiple CPU cores",
                        action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    print(args)
    use_multi_core_processing = args.multicore
    if args.function.strip().lower() == 'all':
        for name in named_methods:
            apply_in_folder(folder=args.directory, function=apply_to_file, method=named_methods[name],
                            allow_sub_folders=args.sub_folder, sub_folder_name=name)
    else:
        func_list = [a for a in args.function.split(',') if a in named_methods]
        for name in func_list:
            apply_in_folder(folder=args.directory, function=apply_to_file, method=named_methods[name],
                            allow_sub_folders=args.sub_folder, sub_folder_name=name)
