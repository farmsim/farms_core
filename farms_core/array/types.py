"""Types"""

from typing import TYPE_CHECKING, Any
import numpy as np
import numpy.typing as npt
from ..sensors.sensor_convention import sc  # pylint: disable=no-name-in-module


if TYPE_CHECKING:
    from typing import Annotated, TypeVar
    DType = TypeVar("DType", bound=np.generic)  # pylint: disable=invalid-name
    def shaped_array(dtype: type[DType], shape: tuple):
        """Type-level helper for shape-annotated NDArray."""
        return Annotated[npt.NDArray[dtype], shape]
else:
    def shaped_array(_dtype, _shape):  # type: ignore
        """Runtime stub for shaped_array (returns NDArray[Any])."""
        return npt.NDArray[Any]


SC_LINK_SIZE = sc.link_size
NDARRAY = shaped_array(np.float32, ...)
NDARRAY_3 = shaped_array(np.float32, (3,))
NDARRAY_4 = shaped_array(np.float32, (4,))
NDARRAY_6 = shaped_array(np.float32, (6,))
NDARRAY_V1 = shaped_array(np.float32, (...,))
NDARRAY_V2 = shaped_array(np.float32, (..., ...))
NDARRAY_3_D = shaped_array(np.double, (3,))
NDARRAY_4_D = shaped_array(np.double, (4,))
NDARRAY_V1_I = shaped_array(np.int64, (...,))
NDARRAY_V1_D = shaped_array(np.double, (...,))
NDARRAY_33 = shaped_array(np.float32, (3, 3))
NDARRAY_44 = shaped_array(np.float32, (4, 4))
NDARRAY_V2_D = shaped_array(np.double, (..., ...))
NDARRAY_X3_D = shaped_array(np.double, (..., 3))
NDARRAY_V3_D = shaped_array(np.double, (..., ..., ...))
NDARRAY_XX3_D = shaped_array(np.double, (..., ..., 3))
NDARRAY_XX4_D = shaped_array(np.double, (..., ..., 4))
NDARRAY_XXXXX_UI8 = shaped_array(np.uint8, (..., ..., ..., ..., ...))
NDARRAY_LINKS_ARRAY = shaped_array(np.double, (..., ..., SC_LINK_SIZE))
