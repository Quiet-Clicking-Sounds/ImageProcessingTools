# --------------------------------------------------------------------------------------------------------------
# ---------------------------Multi Processing Function ---------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
import multiprocessing as mp
from collections.abc import Callable
from os import cpu_count

core_count = max(cpu_count() - 2, 1)


def quick_pool(function: Callable, data_list: list) -> list:
    with mp.Pool(core_count) as p:
        return p.map(function, data_list)
