from image_process.process_handling import (
    Average,
    Distribute,
    Power,
    MaxContrast,
    RollContrast,
    Sharpen,
    HSV,
)
from image_process.process_handling import DefinedTypes

"""
Methods: 

group methods, these allow multiple argument inputs
Average(*args, )
Distribute(*args, reverse=False)
Power(*args, reverse=False)

single methods, these allow a single argument as an input
MaxContrast
RollContrast
Sharpen

base method, applies a moving standard deviation calculation over windows of the target image: 
integers in range 2 - min(image width, image height)
HSV - the base method applied to a hsv encoded image


Extensions:
Method.hsv() 
    convert image to hsv, then return it to bgr 
Method.bgr()
    opposite of the above, used to undo .hsv() as all images default to the bgr colour space 

Method.hsv_partial(h_=True, s_=True, v_=True)
    conver to hsv, optionally remove items from 
Method.bgr_partial(b_=True, g_=True, r_=True)

"""

named_methods: dict[str, DefinedTypes] = {
    "cont_24": Distribute(
        Distribute(3, 5, 7, 9, 11, 13),
        Distribute(3, 5, 7, 9, 11),
        Distribute(3, 5, 7),
        15,
        25,
        reverse=True,
    ),
    "add_area": Distribute(
        Distribute(13, 19, 21),
        Distribute(7, 9, 11),
        Distribute(5, 7, 9),
        Distribute(3, 5, 7),
    ),
    "pair_test3": Average(
        MaxContrast(RollContrast(Average(2, 15))),
        MaxContrast(RollContrast(2)),
        MaxContrast(RollContrast(Average(8, 10, 12))),
    ),
    "all_items": Average(
        Distribute(3, 4, reverse=True),
        Power(3, 4, reverse=True),
        MaxContrast(6),
        RollContrast(6, invert=True),
        Sharpen(6, strength=0.3),
    ),
    "wrapper_test": Average(
        Distribute(4, 6).hsv(),
        Power(4, 6).hsv(),
        MaxContrast(6).hsv(),
        RollContrast(6).hsv_partial(True, False, True),
        Sharpen(6).bgr_partial(False, True, False),
    ),
    "summing_hsv": Average(
        HSV(2),
        HSV(3),
        HSV(4),
        HSV(5),
        HSV(6),
        HSV(7),
        HSV(8),
        HSV(9),
        HSV(10),
    ),
}
a = named_methods
_named_methods: dict[str, DefinedTypes] = {
    "hsv_2": HSV(2),
    "hsv_ftt_2": HSV(2, False, True, True),
    "hsv_tft_2": HSV(2, True, False, True),
    "hsv_ttf_2": HSV(2, True, True, False),
    "Sharpen_ftt_2": Sharpen(2).hsv_partial(False, True, True),
    "Sharpen_tft_2": Sharpen(2).hsv_partial(True, False, True),
    "Sharpen_ttf_2": Sharpen(2).hsv_partial(True, True, False),
    "MaxContrast_ftt_2": MaxContrast(2).hsv_partial(False, True, True),
    "MaxContrast_tft_2": MaxContrast(2).hsv_partial(True, False, True),
    "MaxContrast_ttf_2": MaxContrast(2).hsv_partial(True, True, False),
    "RollContrast_ftt_2": RollContrast(2).hsv_partial(False, True, True),
    "RollContrast_tft_2": RollContrast(2).hsv_partial(True, False, True),
    "RollContrast_ttf_2": RollContrast(2).hsv_partial(True, True, False),
    "hsv_tff_2": HSV(2, True, False, False),
    "Sharpen_tff_2": Sharpen(2).hsv_partial(True, False, False),
    "MaxContrast_tff_2": MaxContrast(2).hsv_partial(True, False, False),
    "RollContrast_tff_2": RollContrast(2).hsv_partial(True, False, False),
    "hsv_ftf_2": HSV(2, False, True, False),
    "Sharpen_ftf_2": Sharpen(2).hsv_partial(False, True, False),
    "MaxContrast_ftf_2": MaxContrast(2).hsv_partial(False, True, False),
    "RollContrast_ftf_2": RollContrast(2).hsv_partial(False, True, False),
    "hsv_fft_2": HSV(2, False, False, True),
    "Sharpen_fft_2": Sharpen(2).hsv_partial(False, False, True),
    "MaxContrast_fft_2": MaxContrast(2).hsv_partial(False, False, True),
    "RollContrast_fft_2": RollContrast(2).hsv_partial(False, False, True),
    "Average_hsv_hsvtft_2": Average(
        HSV(2),
        HSV(2, True, False, True),
    ),
}
