"""Array"""

from typing import Any
import numpy as np
from nptyping import NDArray


def to_array(
        array: NDArray[Any, float],
        iteration: int = None,
) -> NDArray[Any, float]:
    """To array or None"""
    if array is not None:
        array = np.array(array)
        if iteration is not None:
            array = array[:iteration]
    return array
