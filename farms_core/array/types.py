"""Types"""

from typing import Any
from importlib.metadata import version
import numpy as np
from nptyping import NDArray
from ..sensors.sensor_convention import sc

# pylint: disable=invalid-name


SC_LINK_SIZE = sc.link_size
if version('nptyping') < '2.0.0':
    NDARRAY = NDArray[Any, float]
    NDARRAY_3 = NDArray[(3,), float]
    NDARRAY_4 = NDArray[(4,), float]
    NDARRAY_6 = NDArray[(6,), float]
    NDARRAY_V1 = NDArray[(Any,), float]
    NDARRAY_3_D = NDArray[(3,), np.double]
    NDARRAY_4_D = NDArray[(4,), np.double]
    NDARRAY_V1_I = NDArray[(Any,), np.uintc]
    NDARRAY_V1_D = NDArray[(Any,), np.double]
    NDARRAY_33 = NDArray[(3, 3), float]
    NDARRAY_44 = NDArray[(4, 4), float]
    NDARRAY_V2_D = NDArray[(Any, Any), np.double]
    NDARRAY_X3_D = NDArray[(Any, 3), np.double]
    NDARRAY_V3_D = NDArray[(Any, Any, Any), np.double]
    NDARRAY_XX3_D = NDArray[(Any, Any, 3), np.double]
    NDARRAY_XX4_D = NDArray[(Any, Any, 4), np.double]
    NDARRAY_LINKS_ARRAY = NDArray[(Any, Any, SC_LINK_SIZE), np.double]
else:
    from nptyping import Shape, Float
    NDARRAY = NDArray[Any, Float]
    NDARRAY_3 = NDArray[Shape['3'], Float]
    NDARRAY_4 = NDArray[Shape['4'], Float]
    NDARRAY_6 = NDArray[Shape['6'], Float]
    NDARRAY_V1 = NDArray[Shape['*'], Float]
    NDARRAY_3_D = NDArray[Shape['3'], np.double]
    NDARRAY_4_D = NDArray[Shape['4'], np.double]
    NDARRAY_V1_I = NDArray[Shape['*'], np.uintc]
    NDARRAY_V1_D = NDArray[Shape['*'], np.double]
    NDARRAY_33 = NDArray[Shape['3, 3'], Float]
    NDARRAY_44 = NDArray[Shape['4, 4'], Float]
    NDARRAY_V2_D = NDArray[Shape['*, *'], np.double]
    NDARRAY_X3_D = NDArray[Shape['*, 3'], np.double]
    NDARRAY_V3_D = NDArray[Shape['*, *, *'], np.double]
    NDARRAY_XX3_D = NDArray[Shape['*, *, 3'], np.double]
    NDARRAY_XX4_D = NDArray[Shape['*, *, 4'], np.double]
    NDARRAY_LINKS_ARRAY = NDArray[Shape['*, *, SC_LINK_SIZE'], np.double]
