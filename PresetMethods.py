import pathlib
from functools import partial
from typing import Union

import Contrast
import IO
import NamedMethods

use_multi_core_processing = False


def apply_in_folder(folder: str, method_names: list[str], allow_sub_folders=False, use_multi_core_processing=False,
                    **kwargs):
    def get_list_of_files(directory: pathlib.Path):
        """ return a list of files within a directory
        :param directory:
        :return:
        """
        found_files = []
        for item in directory.iterdir():
            if item.is_file():
                found_files.append(item)
            elif item.is_dir() and allow_sub_folders:
                found_files.extend(get_list_of_files(item))
        return found_files

    file_list = get_list_of_files(IO.assign_path(folder))

    if use_multi_core_processing:
        try:
            import MulticoreProcessing
            print("MultiCore Method")
            function = partial(apply_methods, method_names=method_names)
            MulticoreProcessing.quick_pool(function, file_list)
            return None
        except ModuleNotFoundError as module_error:
            print("Multicore failed, module not found\n", module_error, "\n\n fail-over to slow method")

    print('Slow Method')
    for file in file_list:
        apply_methods(file, method_names=method_names)


def sub_method(image: Contrast.ImageCache, method: Union[tuple, int, str], data=None, **kwargs):
    if data is None:
        data = []
    if isinstance(method, tuple):
        for me in method:
            data.append(sub_method(image, me, data, **kwargs))
    if isinstance(method, int):
        return image(method, **kwargs)
    if isinstance(method, str):
        data = Contrast.combine_array_list(data, method=method)


def apply_methods(file: pathlib.Path, method_names: list[str]):
    image = Contrast.ImageCache(file)
    for method_name in method_names:
        data_list = []  # reset var
        method = NamedMethods.named_methods[method_name]
        image_args = {x: x in method for x in NamedMethods.image_args}
        for parse in method:
            if isinstance(parse, tuple):
                data = [image(window=w, **image_args) for w in parse[0]]
                data_list.append(Contrast.combine_array_list(data, parse[1]))
            elif isinstance(parse, int):
                data_list.append(image(window=parse, **image_args))

        output_image = Contrast.combine_array_list(data_list, method[-1])
        IO.export_image(url=image.output_file(method_name), data=output_image)


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

    method_dict = list(NamedMethods.named_methods.keys())
    if args.function.strip().lower() != 'all':
        method_args = args.function.split(',')
        method_dict = [a for a in NamedMethods.named_methods if a in method_args]

    apply_in_folder(folder=args.directory,
                    method_names=method_dict,
                    allow_sub_folders=args.sub_folder,
                    use_multi_core_processing=args.multicore)
