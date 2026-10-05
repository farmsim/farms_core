"""Animat extensions"""

from abc import ABC, abstractmethod

from ..simulation.extensions import TaskExtension
from ..experiment.options import ExperimentOptions
from .options import AnimatOptions
from .data import AnimatData


class AnimatExtension(TaskExtension, ABC):
    """Task extension"""

    def __init__(self, animat_i, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.animat_i = animat_i

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
