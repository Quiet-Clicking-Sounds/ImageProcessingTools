# --------------------------------------------------------------------------------------------------------------
# ---------------------------Multi Processing Function ---------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
from multiprocessing import Pool
from os import cpu_count
from collections.abc import Callable

core_count = max(cpu_count() - 2, 1)


def quick_pool(function: Callable, data_list: list) -> list:
    with Pool(core_count) as p:
        return p.map(function, data_list)

