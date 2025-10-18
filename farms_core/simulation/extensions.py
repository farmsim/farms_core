"""Simulation extensions"""

import os
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING, TypeAlias

from .. import pylog
from ..options import Options
from ..doc import ClassDoc, get_inherited_doc_children
from ..experiment.options import ExperimentOptions
from ..experiment.data import ExperimentData

if TYPE_CHECKING:
    from dm_control.rl.control import Task
    from dm_control.mjcf.physics import Physics
else:
    Task: TypeAlias = Any
    Physics: TypeAlias = Any


class TaskExtension(ABC):
    """Task extension"""

    @classmethod
    @abstractmethod
    def from_options(cls, config: dict, experiment_options: ExperimentOptions):
        """From options"""
        raise NotImplementedError

    def __init__(self, substep=False):
        self.substep = substep

    def initialize_episode(self, task: Task, physics: Physics):
        """Initialize episode"""

    def before_step(self, task: Task, action, physics: Physics):
        """Before step"""

    def after_step(self, task: Task, physics: Physics):
        """After step"""

    def action_spec(self, task: Task, physics: Physics):
        """Action specifications"""

    def step_spec(self, task: Task, physics: Physics):
        """Timestep specifications"""

    def get_observation(self, task: Task, physics: Physics):
        """Environment observation"""

    def get_reward(self, task: Task, physics: Physics):
        """Reward"""

    def get_termination(self, task: Task, physics: Physics):
        """Return final discount if episode should end, else None"""

    def observation_spec(self, task: Task, physics: Physics):
        """Observation specifications"""

    def end_episode(self, task: Task, physics: Physics):
        """End episode"""


@dataclass
class ExtensionDoc(ClassDoc):
    """Extension documentation"""

    def __init__(self, *args, **kwargs):
        self.extensions = kwargs.pop('extensions', [])
        super().__init__(*args, **kwargs)


class ExperimentLoggerOptions(Options):
    """Experiment logger"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ExtensionDoc(
            name="experiment logger extension options",
            description="Options for logging simulations.",
            class_type=cls,
            children=get_inherited_doc_children(cls),
            extensions=[ExperimentLogger],
        )

    def __init__(self, log_path, skip):
        super().__init__()
        self.log_path = log_path
        self.skip = skip


class ExperimentLogger(TaskExtension):
    """Experiment logger extension"""

    def __init__(
            self,
            experiment_options: ExperimentOptions,
            log_path: str,
            skip: int,
    ):
        super().__init__()
        self.experiment_options = experiment_options
        self.log_path = log_path
        self.skip = skip
        self.data: ExperimentData | None = None

    @classmethod
    def from_options(
            cls,
            config: ExperimentLoggerOptions,
            experiment_options: ExperimentOptions,
    ):
        """From options"""
        config = ExperimentLoggerOptions(**config)
        return cls(
            experiment_options=experiment_options,
            log_path=config.log_path,
            skip=config.skip,
        )

    def initialize_episode(self, task: Task, physics: Physics):
        """Iteration 0"""
        del physics
        self.data = task.data

    def end_episode(self, task: Task, physics: Physics):
        """End simulation"""
        del physics
        if self.data is None:
            raise ValueError('Data was not updated during first iteration')
        pylog.info('Saving data to %s', self.log_path)
        os.makedirs(self.log_path, exist_ok=True)
        self.data.to_file(
            os.path.join(self.log_path, 'simulation.hdf5'),
            task.iteration,
        )


class ExperimentOptionsLoggerOptions(Options):
    """Experiment logger"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="experiment simulation logger extension",
            description="Options for logging simulations.",
            class_type=cls,
            children=get_inherited_doc_children(cls),
        )

    def __init__(self, log_path):
        super().__init__()
        self.log_path = log_path


class ExperimentOptionsLogger(TaskExtension):
    """Experiment logger extension"""

    def __init__(
            self,
            experiment_options: ExperimentOptions,
            log_path: str,
    ):
        super().__init__()
        self.experiment_options = experiment_options
        self.log_path = log_path

    @classmethod
    def from_options(
            cls,
            config: ExperimentOptionsLoggerOptions,
            experiment_options: ExperimentOptions,
    ):
        """From options"""
        config = ExperimentOptionsLoggerOptions(**config)
        return cls(
            experiment_options=experiment_options,
            log_path=config.log_path,
        )

    def initialize_episode(self, task: Task, physics: Physics):
        del task, physics
        pylog.info('Saving data to %s', self.log_path)
        os.makedirs(self.log_path, exist_ok=True)
        self.experiment_options.simulation.save(
            os.path.join(self.log_path, 'simulation_options.yaml')
        )
        for animat_i, animat in enumerate(self.experiment_options.animats):
            animat.save(
                os.path.join(self.log_path, f'animat_{animat_i}_options.yaml')
            )
        for arena_i, arena in enumerate(self.experiment_options.arenas):
            arena.save(
                os.path.join(self.log_path, f'arena_{arena_i}_options.yaml')
            )
