"""Animat extensions"""

from abc import ABC, abstractmethod

from ..simulation.extensions import TaskExtension
from ..experiment.options import ExperimentOptions
from .options import AnimatOptions
from .data import AnimatData


class AnimatExtension(TaskExtension, ABC):
    """Task extension"""

    @classmethod
    @abstractmethod
    def from_options(
            cls,
            config: dict,
            experiment_options: ExperimentOptions,
            animat_i: int,
            animat_data: AnimatData,
            animat_options: AnimatOptions,
    ):
        """From options"""
        raise NotImplementedError
