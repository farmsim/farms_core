"""Control"""

from enum import IntEnum
import numpy as np
from numpy.typing import NDArray
from ..array.types import NDARRAY_V1
from ..experiment.options import ExperimentOptions
from .data import AnimatData
from .options import AnimatOptions


class ControlType(IntEnum):
    """Control type"""
    POSITION = 0
    VELOCITY = 1
    TORQUE = 2
    SPRINGREF = 3
    SPRINGCOEF = 4
    DAMPINGCOEF = 5
    MUSCLE = 6

    @staticmethod
    def to_string(control: int) -> str:
        """To string"""
        return {
            ControlType.POSITION: 'position',
            ControlType.VELOCITY: 'velocity',
            ControlType.TORQUE: 'torque',
            ControlType.SPRINGREF: 'springref',
            ControlType.SPRINGCOEF: 'springcoef',
            ControlType.DAMPINGCOEF: 'dampingcoef',
            ControlType.MUSCLE: 'muscle',
        }[control]

    @staticmethod
    def from_string(string: str) -> int:
        """From string"""
        return {
            'position': ControlType.POSITION,
            'velocity': ControlType.VELOCITY,
            'torque': ControlType.TORQUE,
            'springref': ControlType.SPRINGREF,
            'springcoef': ControlType.SPRINGCOEF,
            'dampingcoef': ControlType.DAMPINGCOEF,
            'muscle': ControlType.MUSCLE,
        }[string]

    @staticmethod
    def from_string_list(string_list: list[str]) -> list[int]:
        """From string"""
        return [
            ControlType.from_string(control_string)
            for control_string in string_list
        ]


class AnimatController:
    """Animat controller"""

    def __init__(
            self,
            joints_names: tuple[list[str], ...],
            muscles_names: tuple[str, ...],
            max_torques: tuple[NDARRAY_V1, ...],
    ):
        super().__init__()
        self.joints_names = joints_names
        self.muscles_names = muscles_names
        self.max_torques = max_torques
        self.indices: tuple[NDArray] = None
        self.position_args: tuple[NDArray] = None
        self.velocity_args: tuple[NDArray] = None
        self.excitations_args: tuple[NDArray] = None
        assert len(self.joints_names) == len(ControlType), (
            f'{len(self.joints_names)} != {len(ControlType)}'
        )
        assert len(self.max_torques) == len(ControlType), (
            f'{len(self.max_torques)} != {len(ControlType)}'
        )

    @classmethod
    def from_options(
            cls,
            animat_data: AnimatData,
            animat_options: AnimatOptions,
            experiment_options: ExperimentOptions,
            animat_i: int,  # Animat index
    ):
        joints_names = [
            joint.name
            for joint in animat_options.morphology.joints
        ]
        return cls(
            joints_names=[[]]*7,
            muscles_names=[],
            max_torques=[[]]*7,
        )

    @staticmethod
    def joints_from_control_types(
            joints_names: list[str],
            joints_control_types: dict[str, list[ControlType]],
    ) -> tuple[list[str], ...]:
        """From control types"""
        return tuple(
            [
                joint
                for joint in joints_names
                if control_type in joints_control_types[joint]
            ]
            for control_type in list(ControlType)
        )

    @staticmethod
    def max_torques_from_control_types(
            joints_names: list[str],
            max_torques: dict[str, float],
            joints_control_types: dict[str, list[ControlType]],
    ) -> tuple[NDArray, ...]:
        """From control types"""
        return tuple(
            np.array([
                max_torques[joint]
                for joint in joints_names
                if control_type in joints_control_types[joint]
            ])
            for control_type in list(ControlType)
        )

    @classmethod
    def from_control_types(
            cls,
            joints_names: list[str],
            max_torques: dict[str, float],
            joints_control_types: dict[str, list[ControlType]],
    ):
        """From control types"""
        return cls(
            joints_names=cls.joints_from_control_types(
                joints_names=joints_names,
                joints_control_types=joints_control_types,
            ),
            max_torques=cls.max_torques_from_control_types(
                joints_names=joints_names,
                max_torques=max_torques,
                joints_control_types=joints_control_types,
            ),
        )

    def step(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ):
        """Step"""

    def positions(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Positions"""
        assert iteration >= 0
        assert time >= 0
        assert timestep > 0
        return {
            joint: 0
            for joint in self.joints_names[ControlType.POSITION]
        }

    def velocities(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Velocities"""
        assert iteration >= 0
        assert time >= 0
        assert timestep > 0
        return {
            joint: 0
            for joint in self.joints_names[ControlType.VELOCITY]
        }

    def torques(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Torques"""
        assert iteration >= 0
        assert time >= 0
        assert timestep > 0
        return {
            joint: 0
            for joint in self.joints_names[ControlType.TORQUE]
        }

    def springrefs(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Spring references"""
        assert iteration >= 0
        assert time >= 0
        assert timestep > 0
        return {}

    def springcoefs(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Spring coefficients"""
        assert iteration >= 0
        assert time >= 0
        assert timestep > 0
        return {}

    def dampingcoefs(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Damping coefficients"""
        assert iteration >= 0
        assert time >= 0
        assert timestep > 0
        return {}

    def excitations(
            self,
            iteration: int,
            time: float,
            timestep: float,
    ) -> dict[str, float]:
        """Muscle excitations"""
        assert iteration >= 0
        assert time >= 0
        assert timestep > 0
        return {
            muscle: 0.05
            for muscle in self.muscles_names
        }
