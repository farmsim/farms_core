"""Control"""

from enum import IntEnum
from typing import Any, TYPE_CHECKING, TypeAlias

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from dm_control.rl.control import Task
    from dm_control.mjcf.physics import Physics
else:
    Task: TypeAlias = Any
    Physics: TypeAlias = Any

from ..array.types import NDARRAY_V1
from ..experiment.options import ExperimentOptions
from .data import AnimatData
from .options import AnimatOptions
from .extensions import AnimatExtension


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


class AnimatController(AnimatExtension):
    """Animat controller

    The controller is written in such a way where it is possible to control the
    animat's actuators, i.e. its joints and muscles. They list of actuators to
    control are both provided using the ´joints_names´ and ´muscle_torques´
    arguments.

    :param joints_names: The names of the joints
    :param muscles_names: The names of the muscles

    """

    def __init__(
            self,
            animat_i: int,
            joints_names: tuple[list[str], ...],
            muscles_names: tuple[str, ...],
            substep=True,
    ):
        super().__init__(animat_i=animat_i, substep=substep)
        self.joints_names = joints_names
        self.muscles_names = muscles_names
        assert len(self.joints_names) == len(ControlType), (
            f'{len(self.joints_names)} != {len(ControlType)}'
        )

    @classmethod
    def from_options(
            cls,
            config: dict,
            experiment_options: ExperimentOptions,
            animat_i: int,
            animat_data: AnimatData,
            animat_options: AnimatOptions,
    ):
        """From options"""
        return cls(
            animat_i=animat_i,
            joints_names=[[]]*7,
            muscles_names=[],
            # max_torques=[[]]*7,
        )

    @staticmethod
    def joints_from_control_types(
            joints_names: list[str],
            joints_control_types: dict[str, list[ControlType]],
    ) -> tuple[list[str], ...]:
        """Joints from control types

        This is a helper function for writing the joints_names argument in the
        class.

        :param joints_names: The names of the joints
        :param muscles_names: The names of the muscles
        :returns:

        """
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
