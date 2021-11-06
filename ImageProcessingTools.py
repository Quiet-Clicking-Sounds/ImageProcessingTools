import argparse
from pathlib import Path

import Contrast
import IO


def list_from_input(string: str) -> list[int] or int:
    l = [int(a) for a in string.split(',')]
    if len(l) == 1:
        return l[0]
    return l


def single_pass(file_in: str, file_out: str, rgb: bool, window: int):
    data = IO.load_image(Path(file_in), rgb=rgb)
    data = Contrast.apply(Contrast.moving_stdev, data, window=window)
    IO.export_image(Path(file_out), data)


def multi_pass(file_in: str, file_out: str, rgb: bool, window: list[int], combine_method: str):
    data = IO.load_image(Path(file_in), rgb=rgb)
    data = [Contrast.apply(Contrast.moving_stdev, data, window=w) for w in window]
    data = Contrast.resize_list_of_arrays(data)
    data = Contrast.combine_array_list(data, combine_method)
    IO.export_image(Path(file_out), data)




def commandline_mode(args:argparse.Namespace):
    window = list_from_input(args.window)
    if isinstance(window, int):
        single_pass(file_in=args.filename,
                    file_out=args.output,
                    rgb=args.rgb,
                    window=window)
    else:
        multi_pass(file_in=args.filename,
                   file_out=args.output,
                   rgb=args.rgb,
                   window=window,
                   combine_method=args.combine_options)

def interactive_mode():
    filename = input("Target input file:")
    output = input("Target output file:")
    window = list_from_input(input("Window size, int or list[int]"))
    sub_args = input("Additional arguments, --no-rgb")
    rgb = "--no-rgb" not in sub_args
    if isinstance(window, int):
        single_pass(file_in=filename,
                    file_out=output,
                    rgb=rgb,
                    window=window)
    else:
        combine_options = input("method used to combine multi-pass images: 'sum', 'avg', 'dist'")
        multi_pass(file_in=filename,
                   file_out=output,
                   rgb=rgb,
                   window=window,
                   combine_method=combine_options)




if __name__ == '__main__':
    # argparse info:
    # https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()

    parser.add_argument("-i" , "--interactive",dest="interactive", help="Interactive mode",
                        action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("-f", "--file", dest="filename", help="input file name")
    parser.add_argument("-o", "--output", dest="output", help="output file name")

    parser.add_argument("-w", "--window", dest="window", help="window size, int or list[int]", default=3)

    # used in Contrast.combine_array_list
    parser.add_argument("-combine", "--combiner_options", dest="combine_options",
                        help="method used to combine multi-pass images", choices=["sum", "avg", "dist"], default="sum")

    # additional options
    parser.add_argument("-rgb", "--rgb", dest="rgb", help="Use RGB image functions",
                        action=argparse.BooleanOptionalAction, default=True)

    args = parser.parse_args()
    print(args)

    if args.interactive:
        interactive_mode()
    else:
        commandline_mode(args)
