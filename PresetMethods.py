import IO
import Contrast

use_multi_core_processing = False
"""
Named methods: 
for multi part sub items use:    tuple[list,str] 
for single part sub items use:   int
final item must be:              str 
"""
named_methods = {
    'high_contrast': (
        ([3, 5, 7, 9, 11, 13, 15], 'dist'), ([3, 5, 7, 9, 11, 13, 15], '-dist'), ([3, 5, 7], 'avg'), ([3, 5], 'avg'),
        25,
        'sum'),
    'low_contrast': (
        ([3, 5, 7, 9, 11, 13], 'dist'), ([3, 5, 7, 9, 11, 13], '-dist'), ([3, 5, 7], 'sum'), ([3, 5], 'avg'), 15,
        'sum'),
    'add_area': (
        ([7, 9, 11, 13, 19, 21], 'dist'), ([7, 9, 11, 13, 19, 21], '-dist'), ([5, 7, 9], 'avg'), ([3, 5, 7], 'avg'),
        'sum')
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
    data_list = []

    for parse in method:
        if isinstance(parse, tuple):
            data = [Contrast.apply(Contrast.moving_stdev, image, window=w) for w in parse[0]]
            data = Contrast.resize_list_of_arrays(data)
            data_list.append(Contrast.combine_array_list(data, parse[1]))
        elif isinstance(parse, int):
            data_list.append(Contrast.apply(Contrast.moving_stdev, image, window=parse))

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


