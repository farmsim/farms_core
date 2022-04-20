"""Array"""

import numpy as np
from .types import NDARRAY


def to_array(
        array: NDARRAY,
        iteration: int = None,
) -> NDARRAY:
    """To array or None"""
    if array is not None:
        array = np.array(array)
        if iteration is not None:
            array = array[:iteration]
    return array
