from image_process.process_handling import (
    Average,
    Distribute,
    Power,
    MaxContrast,
    RollContrast,
    Sharpen,
    MixedHSV,
    MixedBGR,
    HSV,
    InsertChannelHSV,
    InsertChannelBGR,
    AsInput,
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

base method, applies a moving standard deviation calculation over windows of the target img: 
integers in range 2 - min(img width, img height)
HSV - the base method applied to a hsv encoded img


Extensions:
Method.hsv() 
    convert img to hsv, then return it to bgr 
Method.bgr()
    opposite of the above, used to undo .hsv() as all images default to the bgr colour space 

Method.hsv_partial(h_=True, s_=True, v_=True)
    conver to hsv, optionally remove items from 
Method.bgr_partial(b_=True, g_=True, r_=True)

"""

possible_methods: dict[str, DefinedTypes] = {
    "hsv_tff": HSV(4, True, False, False),
    "Sharpen_tff": Sharpen(4).hsv_partial(True, False, False),
    "MaxContrast_tff": MaxContrast(4).hsv_partial(True, False, False),
    "RollContrast_tff": RollContrast(4).hsv_partial(True, False, False),
    "Sharpen_ftt": Sharpen(4).hsv_partial(False, True, True),
    "MaxContrast_ftt": MaxContrast(4).hsv_partial(False, True, True),
    "RollContrast_ftt": RollContrast(4).hsv_partial(False, True, True),

    "MixedHSV_4_6_8": MixedHSV(4, 6, 8),
    "MixedBGR_4_6_8": MixedBGR(4, 6, 8),
    "InsertChannelHSV": InsertChannelHSV(4, None, MixedHSV(0, 4, 0), AsInput()),
    "InsertChannelBGR": InsertChannelBGR(4, None, 6, AsInput()),
    "Average": Average(2, 4, 6),
    "Distribute": Distribute(2, 4, 6),
    "Power": Power(2, 4, 6),
    "MaxContrast": MaxContrast(2, ),
    "RollContrast": RollContrast(2, invert=True),
    "Sharpen": Sharpen(2, strength=0.3),
}

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