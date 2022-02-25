import pathlib
from functools import partial

import IO
import MulticoreProcessing
import image_process
from _predefined_methods import named_methods

use_multi_core_processing = False


def apply_in_folder(
        folder: str,
        method_names: list[str],
        allow_sub_folders=False,
        use_multi_core=False,
        scale_factor=1,
        modify_filename=False,
):
    def get_list_of_files(directory: pathlib.Path):
        """return a list of files within a directory
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

    if use_multi_core:
        print("MultiCore Method")
    else:
        print("Slow Method")
    for file in file_list:
        apply_methods(
            file,
            method_names=method_names,
            scale_factor=scale_factor,
            modify_filename=modify_filename,
            multicore=use_multi_core,
        )


def _apply_predefined_method_(method_name, image=None):
    method = named_methods[method_name]
    output_image = image_process.apply_method(image, method)
    IO.export_image(url=image.output_file(method_name), data=output_image)


def apply_methods(
        file: pathlib.Path,
        method_names: list[str],
        scale_factor: float,
        modify_filename: bool,
        multicore: bool = False,
):
    try:
        image = image_process.ImageCache(
            file, scale_factor=scale_factor, modify_filename=modify_filename
        )
    except FileExistsError as _:
        print(f"File Exists {file}")
        return
    if multicore:
        methods = [named_methods[m] for m in method_names]

        # img.calc_division_list(image_process.get_method_counts(methods))
        if image.image_byte_size() < 100e6:
            MulticoreProcessing.quick_pool(
                partial(_apply_predefined_method_, image=image), method_names
            )
            return

    for method_name in method_names:
        _apply_predefined_method_(method_name, image)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--directory",
        dest="directory",
        help="directory of files to apply methods to",
    )
    parser.add_argument(
        "-f",
        "--function",
        dest="function",
        help="functions to apply over each file 'add area' use ',' as separator or 'all'",
        default="all",
    )
    parser.add_argument(
        "--allow_sub_folders",
        dest="sub_folder",
        help="run function on all sub-folders of the given directory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "-m",
        "--multicore",
        dest="multicore",
        help="use multiple CPU cores",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "-s",
        "--scale",
        dest="scale_factor",
        type=float,
        help="Scale factor to use, 2 for double size, 0.5 for half size, etc",
        default=1.0,
    )
    parser.add_argument(
        "-t",
        "--timer",
        dest="timer",
        help="print timer stats -- Not usable with multicore --",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "-fn",
        "--filename",
        dest="filename",
        help="modify filename for output file instead of folder",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "-i",
        "--interactive",
        dest="interactive",
        help="open the GUI - this over_ride all other arguments.",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()
    print(args)
    use_multi_core_processing = args.multicore
    method_dict = list(named_methods.keys())
    if args.interactive:
        import interface

        interface.mainloop()
        exit(1)

    if args.function.strip().lower() != "all":
        method_args = args.function.split(",")
        method_dict = [a for a in named_methods if a in method_args]

    apply_in_folder(
        folder=args.directory,
        method_names=method_dict,
        allow_sub_folders=args.sub_folder,
        use_multi_core=args.multicore,
        scale_factor=args.scale_factor,
        modify_filename=args.filename,
    )
