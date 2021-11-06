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


if __name__ == '__main__':
    # argparse info:
    # https://docs.python.org/3/library/argparse.html
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", dest="filename", help="input file name", required=True)
    parser.add_argument("-o", "--output", dest="output", help="output file name", required=True)

    parser.add_argument("-w", "--window", dest="window", help="window size, int or list[int]", default=3)

    # used in Contrast.combine_array_list
    parser.add_argument("-combine", "--combiner_options", dest="combine_options",
                        help="method used to combine multi-pass images", choices=["sum", "avg", "dist"], default="sum")

    # additional options
    parser.add_argument("-rgb", "--rgb", dest="rgb", help="Use Greyscale",
                        action=argparse.BooleanOptionalAction, default=True)
    # no idea if this is required or for testing yet
    args = parser.parse_args('-f TestFiles/7nshji.jpg -o TestFiles/output/7nshji.jpg -w 3,5,7'.split())
    print(args)
    window = list_from_input(args.window)
    if isinstance(args.window, int):
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
int()
