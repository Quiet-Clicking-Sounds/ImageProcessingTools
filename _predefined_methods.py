from image_process.process_handling import Average, Distribute, Power, MaxContrast, RollContrast, Sharpen
from image_process.process_handling import DefinedTypes

named_methods: dict[str, DefinedTypes] = {
    'cont_24': Distribute(
        Distribute(3, 5, 7, 9, 11, 13),
        Distribute(3, 5, 7, 9, 11),
        Distribute(3, 5, 7),
        15,
        25,
        reverse=True
    ),
    'add_area': Distribute(
        Distribute(13, 19, 21),
        Distribute(7, 9, 11),
        Distribute(5, 7, 9),
        Distribute(3, 5, 7),
    ),
    'pair_test3': Average(
        MaxContrast(
            RollContrast(
                Average(2, 15)
            )
        ),
        MaxContrast(
            RollContrast(2)
        ),
        MaxContrast(
            RollContrast(
                Average(8, 10, 12)
            )
        ),
    ),
    'all_items': Average(
        Distribute(3, 4, reverse=True),
        Power(3, 4, reverse=True),
        MaxContrast(6),
        RollContrast(6, invert=True),
        Sharpen(6, strength=0.3),
    )
}
