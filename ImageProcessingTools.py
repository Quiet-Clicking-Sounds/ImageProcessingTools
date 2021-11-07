import argparse
from pathlib import Path

import Contrast
import IO


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


def list_from_input(in_var: str) -> list[int] or int:
    """
    convert string of comma separated integers to list[int] OR convert a single string integer to a single int
    "3,5,7" -> list(3,5,7) OR "5" -> int(5)
    :param in_var:  "3,5,7" or "5"
    :return: [3, 5, 7] or 5
    """
    in_var = ''.join([i for i in in_var if i in "1234567890,"])
    in_var = [int(a) for a in in_var.split(',')]
    if len(in_var) == 1:
        return in_var[0]
    return in_var


def single_pass(file_in: str, file_out: str, rgb: bool, window: int, return_image=False):
    """

    :param file_in: target input file as string
    :param file_out: target output file as string
    :param rgb: True to load image as rgb, False for greyscale
    :param window: window size for standard deviation calculation
    :param return_image: ignore file_out and return the image instead
    :return:
    """
    data = IO.load_image(assign_path(file_in, True), rgb=rgb)
    data = Contrast.apply(Contrast.moving_stdev, data, window=window)
    if return_image: return data
    IO.export_image(assign_path(file_out, True), data)
    print(f"Operation Complete\n{'-' * 20}")


def multi_pass(file_in: str, file_out: str, rgb: bool, window: list[int], combine_method: str, return_image=False):
    """

    :param file_in: target input file as string
    :param file_out: target output file as string
    :param rgb: True to load image as rgb, False for greyscale
    :param window: window size for standard deviation calculation
    :param combine_method: method used to combine the passes
    :param return_image: ignore file_out and return the image instead
    :return:
    """
    data = IO.load_image(assign_path(file_in, True), rgb=rgb)
    data = [Contrast.apply(Contrast.moving_stdev, data, window=w) for w in window]
    data = Contrast.resize_list_of_arrays(data)
    data = Contrast.combine_array_list(data, combine_method)
    if return_image: return data
    IO.export_image(assign_path(file_out, True), data)
    print(f"Operation Complete\n{'-' * 20}")


def commandline_mode(cl_args: argparse.Namespace):
    """  Reads args from the commandline interface when not using the interactive mode
    :param cl_args: argparse inputs
    :return:
    """
    window = list_from_input(cl_args.window)
    print(f"{'-' * 20}\nBeginning operation")
    if isinstance(window, int):
        single_pass(file_in=cl_args.filename,
                    file_out=cl_args.output,
                    rgb=cl_args.rgb,
                    window=window)
    else:
        multi_pass(file_in=cl_args.filename,
                   file_out=cl_args.output,
                   rgb=cl_args.rgb,
                   window=window,
                   combine_method=cl_args.combine_options)


def interactive_mode():
    """
    interactive mode for user input
    :return:
    """
    filename = input("Target input file: ")
    while True:
        output = input("Target output file: ")
        window = list_from_input(input("Window size, int or list[int]: "))
        sub_args = input("Additional arguments, --no-rgb: ")
        rgb = "--no-rgb" not in sub_args

        print(f"{'-' * 20}\nBeginning operation")
        if isinstance(window, int):
            single_pass(file_in=filename,
                        file_out=output,
                        rgb=rgb,
                        window=window)
        else:
            combine_options = input(
                "method used to combine multi-pass images: 'sum', 'avg', 'dist' - prepend '-' to invert list: ")
            inverse = False
            multi_pass(file_in=filename,
                       file_out=output,
                       rgb=rgb,
                       window=window,
                       combine_method=combine_options)
        if input("run again on same file? y/n: ") != "y":
            break


def get_first_valid_combination_type(string: str) -> str:
    """
    :param string: input string
    :return: First occurrence of a combination method option
    """
    return [c for c in Contrast.combine_method_options() if c in string][0]


def interactive_complex_mode():
    """
    Interactive complex mode for user input
    This mode gives additional methods but is less user friendly.
    :return:
    """
    filename = input("Target input file: ")
    output = input("Target output file: ")
    sub_args = input("Additional arguments, --no-rgb: ")
    rgb = "--no-rgb" not in sub_args

    window_list = []
    print(f"{'-' * 20}\nWindow syntax: [int] or [list[int] combine] e.g. '5' or '3,5,7 dist'")
    print(f"combine options: {Contrast.combine_method_options(True)} \nUse 'q' or 'quit' to exit window inputs\n")
    while True:
        current = input(f"Window {len(window_list)}: ")
        if current.lower() in ['q', 'quit']:
            break
        else:
            window_list.append(current)

    final_combination_method = get_first_valid_combination_type(input(f"Final combination method: "))
    print(f"{'-' * 20}\nBeginning operations")
    image_list = []
    for window in window_list:
        try:
            image_list.append(
                single_pass(file_in=filename,
                            file_out=output,
                            rgb=rgb,
                            window=int(window),
                            return_image=True))

        except ValueError:  # calls this if single_pass() fails due to int(window)
            window_sizes = list_from_input(window)
            combine_opt = get_first_valid_combination_type(window)
            image_list.append(
                multi_pass(file_in=filename,
                           file_out=output,
                           rgb=rgb,
                           window=window_sizes,
                           combine_method=combine_opt,
                           return_image=True))
    image_list = Contrast.resize_list_of_arrays(image_list)
    image = Contrast.combine_array_list(image_list, final_combination_method)
    IO.export_image(assign_path(output), image)


if __name__ == '__main__':
    # argparse info:
    # https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()

    # allow access to the interactive mode
    parser.add_argument("-i", "--interactive", dest="interactive", help="Interactive mode",
                        action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("-ic", "--interactive_complex", dest="interactive_complex", help="interactive complex mode",
                        action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("-f", "--file", dest="filename", help="input file name")
    parser.add_argument("-o", "--output", dest="output", help="output file name")

    parser.add_argument("-w", "--window", dest="window", help="window size, int or list[int]", default=3)

    # used in Contrast.combine_array_list
    parser.add_argument("-combine", "--combiner_options", dest="combine_options",
                        help="method used to combine multi-pass images, prepend '-' to invert list",
                        choices=Contrast.combine_method_options(), default="sum")

    # additional options
    parser.add_argument("-rgb", "--rgb", dest="rgb", help="Use RGB image functions",
                        action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    print(args)

    if args.interactive:
        interactive_mode()
    elif args.interactive_complex:
        interactive_complex_mode()
    else:
        commandline_mode(args)
