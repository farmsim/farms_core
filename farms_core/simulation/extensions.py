"""Simulation extensions"""

import os
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from dm_control.rl.control import Task
    from dm_control.mjcf.physics import Physics
else:
    Task: TypeAlias = Any
    Physics: TypeAlias = Any

from .. import pylog
from ..options import Options
from ..doc import ClassDoc, ExtensionDoc, get_inherited_doc_children
from ..experiment.options import ExperimentOptions
from ..experiment.data import ExperimentData


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
    """Experiment logger extension

    Saves simulation data to HDF5. Two modes of operation:

    * **Full-buffer (default):** When ``buffer_size >= n_iterations`` the
      entire simulation fits in memory and data is saved once at the end
      of the episode in write mode (``'w'``).  This is the backwards-
      compatible behaviour.

    * **Incremental:** When ``buffer_size < n_iterations`` the in-memory
      buffer is smaller than the simulation length.  Data is saved
      periodically (every ``buffer_size`` iterations) by appending to the
      HDF5 file.  Only the new data since the last save is written,
      keeping memory low while the full dataset accumulates on disk.

    The ``skip`` parameter controls **decimation** of the saved data:
    when ``skip > 1``, only every ``skip``-th iteration is written to
    the HDF5 file (e.g. ``skip=2`` saves iterations 0, 2, 4, ...).
    The buffer is still saved every ``buffer_size`` iterations to
    prevent overflow, but each saved chunk is sub-sampled by ``skip``.
    """

    def __init__(
            self,
            experiment_options: ExperimentOptions,
            log_path: str,
            skip: int,
    ):
        super().__init__()
        self.experiment_options = experiment_options
        self.log_path = log_path
        self.skip = max(1, skip)
        self.mode = 'w'
        self.data: ExperimentData | None = None
        self.buffer_size = experiment_options.simulation.runtime.buffer_size
        self.last_saved_iteration = 0
        self._filepath = os.path.join(self.log_path, 'simulation.hdf5')

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
        if self.data is None:
            raise ValueError('Data was not updated during first iteration')
        self.mode = 'w'
        self.last_saved_iteration = 0

    def _save_iteration_range(
            self,
            start_iteration: int,
            iteration: int,
    ):
        """Save data for iterations [start_iteration, iteration) to HDF5.

        On the first call ``self.mode`` is ``'w'`` (create/overwrite the
        file); it is switched to ``'a'`` (append) afterwards so that
        subsequent calls extend the on-disk datasets.
        """
        if iteration <= start_iteration:
            return
        pylog.info(
            'Saving data iterations %s-%s to %s',
            start_iteration,
            iteration,
            self.log_path,
        )
        os.makedirs(self.log_path, exist_ok=True)
        self.data.to_file(
            self._filepath,
            iteration=iteration,
            start_iteration=start_iteration,
            mode=self.mode,
            skip=self.skip,
        )
        self.mode = 'a'
        self.last_saved_iteration = iteration

    def after_step(self, task: Task, physics: Physics):
        """After step — periodically save buffer to disk when full"""
        del physics
        # Only trigger incremental saves when the buffer is smaller
        # than the total simulation (otherwise, save only at the end).
        if self.buffer_size >= task.n_iterations and task.n_iterations > 0:
            return
        # Save every buffer_size iterations to prevent buffer overflow.
        # The skip parameter controls decimation (sub-sampling) of the
        # saved data, not the save frequency.
        if (
                task.iteration > 0
                and not task.iteration % self.buffer_size
        ):
            self._save_iteration_range(
                start_iteration=self.last_saved_iteration,
                iteration=task.iteration,
            )

    def end_episode(self, task: Task, physics: Physics):
        """End simulation — flush any remaining unsaved data"""
        del physics
        if task.iteration > self.last_saved_iteration:
            self._save_iteration_range(
                start_iteration=self.last_saved_iteration,
                iteration=task.iteration,
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
        pylog.info(
            'Saving experiment options (sim, animats, arenas) to %s',
            self.log_path,
        )
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
