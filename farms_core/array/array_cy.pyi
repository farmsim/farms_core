"""Arrays"""

import numpy as np


class Array:
    """Array"""
    array: np.ndarray
    def size(self, index: int) -> int: ...
    def copy_array(self) -> np.ndarray: ...
    def log(self, times: int, folder: str, name: str, extension: str) -> None: ...


class DoubleArray1D(Array):
    """Double array"""
    array: np.ndarray
    def __init__(self, array: np.ndarray) -> None: ...


class DoubleArray2D(Array):
    """Double array"""
    array: np.ndarray
    def __init__(self, array: np.ndarray) -> None: ...


class DoubleArray3D(Array):
    """Double array"""
    array: np.ndarray
    def __init__(self, array: np.ndarray) -> None: ...


class IntegerArray1D(Array):
    """Integer array"""
    array: np.ndarray
    def __init__(self, array: np.ndarray) -> None: ...


class IntegerArray2D(Array):
    """Integer array"""
    array: np.ndarray
    def __init__(self, array: np.ndarray) -> None: ...


class Integer8Array5D(Array):
    """Integer8 array"""
    array: np.ndarray
    def __init__(self, array: np.ndarray) -> None: ...
