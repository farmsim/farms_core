"""Animat data"""

import numpy as np
import numpy.typing as npt


class LinkSensorArrayCy:
    """Links array"""
    array: np.ndarray
    def __init__(self, array: np.ndarray) -> None: ...
    def size(self, index: int) -> int: ...
    def copy_array(self) -> np.ndarray: ...


class JointSensorArrayCy:
    """Joint sensor array"""
    array: np.ndarray
    def __init__(self, array: np.ndarray) -> None: ...
    def size(self, index: int) -> int: ...
    def copy_array(self) -> np.ndarray: ...


class ContactsArrayCy:
    """Sensor array"""
    array: np.ndarray
    def __init__(self, array: np.ndarray) -> None: ...
    def size(self, index: int) -> int: ...
    def copy_array(self) -> np.ndarray: ...


class XfrcArrayCy:
    """Xfrc array"""
    array: np.ndarray
    def __init__(self, array: np.ndarray) -> None: ...
    def size(self, index: int) -> int: ...
    def copy_array(self) -> np.ndarray: ...


class MusclesArrayCy:
    """Muscles array"""
    array: np.ndarray
    def __init__(self, array: np.ndarray) -> None: ...
    def size(self, index: int) -> int: ...
    def copy_array(self) -> np.ndarray: ...


class AdhesionsArrayCy:
    """Adhesions array"""
    array: np.ndarray
    def __init__(self, array: np.ndarray) -> None: ...
    def size(self, index: int) -> int: ...
    def copy_array(self) -> np.ndarray: ...


class VisualsArrayCy:
    """Visuals array"""
    array: np.ndarray
    def __init__(self, array: np.ndarray) -> None: ...
    def size(self, index: int) -> int: ...
    def copy_array(self) -> np.ndarray: ...


class CameraArrayCy:
    """Camera array - Iteration, sensor, x, y, color"""
    array: np.ndarray
    def __init__(self, array: np.ndarray) -> None: ...
    def size(self, index: int) -> int: ...
    def copy_array(self) -> np.ndarray: ...


class SensorsDataCy:
    """Sensors data"""
    links: LinkSensorArrayCy
    joints: JointSensorArrayCy
    contacts: ContactsArrayCy
    xfrc: XfrcArrayCy
    muscles: MusclesArrayCy
    adhesions: AdhesionsArrayCy
    visuals: VisualsArrayCy
    cameras: CameraArrayCy

    def __init__(
        self,
        links: LinkSensorArrayCy | None = ...,
        joints: JointSensorArrayCy | None = ...,
        contacts: ContactsArrayCy | None = ...,
        xfrc: XfrcArrayCy | None = ...,
        muscles: MusclesArrayCy | None = ...,
        adhesions: AdhesionsArrayCy | None = ...,
        visuals: VisualsArrayCy | None = ...,
        cameras: CameraArrayCy | None = ...,
    ) -> None: ...
