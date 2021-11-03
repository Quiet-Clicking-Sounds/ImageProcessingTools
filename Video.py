import math
import pathlib
import time
from collections.abc import Callable

import cv2
import numpy

import Contrast


def apply_without_ram_buffer(in_file: pathlib.Path, out_file: pathlib.Path, function: Callable, fps=30,
                             output_size=(0, 0), padding_=0, view_=False,
                             **kwargs) -> None:
    """
    @param in_file: input filepath
    @param out_file: output filepath
    @param function: function to be applied over a 2d array size[x,y] or 3d array size[x,y,z] defined by split_BGR
    @param fps: output fps
    @param output_size: size of the output (W,H), if (0,0) input size will be used and padding added
    @param padding_: int in range 0,255 - padding unit to be used in outputs when output is set to (0,0)
    @param view_: show output in realtime
    @param kwargs: parsed directly to function as **kwargs
    """
    padding_ = numpy.ubyte(padding_)
    # File name and position fixes
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.suffix != '.avi':
        print(f"Fixing suffix without error: from {out_file.suffix} to .avi")
        out_file = out_file.with_suffix('.avi')

    # main components
    try:
        # Input Settings
        capture = cv2.VideoCapture(in_file.__str__())
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f'Input Width:{frame_width} Height:{frame_height}')
        # Output Settings
        if output_size == (0, 0):
            output_size = (frame_width, frame_height)

        def padding_size(frame_size):
            diff_w = (frame_width - frame_size[1]) / 2
            diff_h = (frame_height - frame_size[0]) / 2
            return (math.floor(diff_h), math.ceil(diff_h)), (math.floor(diff_w), math.ceil(diff_w)), (0, 0)

        output = cv2.VideoWriter(filename=out_file.__str__(),
                                 fourcc=cv2.VideoWriter_fourcc(*'MJPG'),
                                 fps=fps,
                                 frameSize=(output_size[0], output_size[1]))

        # Settings
        frame = 0
        return_value = True
        while frame < frame_count and return_value:
            return_value, current_frame = capture.read()
            frame += 1
            current_frame = Contrast.apply_rgb(function, current_frame, **kwargs)
            current_frame = current_frame.astype('uint8')
            if current_frame.shape != (frame_height, frame_width, 3):
                current_frame = numpy.pad(current_frame,
                                          padding_size(current_frame.shape),
                                          mode='constant',
                                          constant_values=(padding_, padding_))
            output.write(current_frame)
            if view_:
                cv2.imshow('frame', current_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(1 / fps)

        # Release all items
        capture.release()
        output.release()

    except FileNotFoundError as fnf:
        print(f'FileNotFoundError: \n\t\t{in_file}\n\t\t{out_file}  \n{fnf}')
        exit(1)
    finally:
        cv2.destroyAllWindows()
