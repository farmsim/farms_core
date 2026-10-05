"""Simulation data"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..doc import ClassDoc, ChildDoc
from ..array.array import to_array
from ..array.types import NDARRAY_V1, NDARRAY_V2


class SimulationData:
    """Simulation

    Contains logs from simulation data such as number physics engine iterations
    and system energy levels.

    """

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="simulation data",
            description="Provides and logs the simulation data.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="ncon",
                    class_type="1DArray[int, [n_iterations]]",
                    description="Number of constraints during iteration.",
                ),
                ChildDoc(
                    name="niter",
                    class_type="1DArray[int, [n_iterations]]",
                    description="Number of physics engine iterations.",
                ),
                ChildDoc(
                    name="energy",
                    class_type="2DArray[int, [2, n_iterations]]",
                    description=(
                        "Potential (first index) and kinetic"
                        " (second index) energy."
                    ),
                ),
            ],
        )

    def __init__(
            self,
            ncon: NDARRAY_V1,
            niter: NDARRAY_V1,
            energy: NDARRAY_V2,
    ):
        super().__init__()
        self.ncon = ncon
        self.niter = niter
        self.energy = energy

    @classmethod
    def from_size(cls, size: int, buffer_size: int | None = None):
        """Animat data from animat and simulation options

        Parameters
        ----------
        size:
            The total number of simulation iterations
            (``n_iterations``).  Used as the fallback buffer size when
            *buffer_size* is not provided or is larger than *size*.
        buffer_size:
            The in-memory circular buffer size.  When smaller than *size*,
            the time-varying arrays (``ncon``, ``niter``, ``energy``) are
            allocated at this length to reduce memory usage.  Data is
            written cyclically at ``index = iteration % buffer_size`` and
            extracted with :func:`to_array` for saving.
        """
        if buffer_size is None or buffer_size <= 0 or buffer_size > size:
            buffer_size = size
        return cls(
            ncon=np.zeros(buffer_size),
            niter=np.zeros(buffer_size),
            energy=np.zeros([buffer_size, 2]),
        )

    @classmethod
    def from_dict(
            cls,
            dictionary: dict,
    ):
        """Load data from dictionary"""
        return cls(
            ncon=dictionary['ncon'],
            niter=dictionary['niter'],
            energy=dictionary['energy'],
        )

    def to_dict(
            self,
            iteration: int | None = None,
            start_iteration: int | None = None,
            skip: int = 1,
    ) -> dict:
        """Convert data to dictionary"""
        return {
            'ncon': to_array(self.ncon, iteration, start_iteration, skip),
            'niter': to_array(self.niter, iteration, start_iteration, skip),
            'energy': to_array(self.energy, iteration, start_iteration, skip),
        }

    def plot(
            self,
            times: NDARRAY_V1,
    ) -> dict:
        """Plot"""
        plots = {}
        plots['ncon'] = self.plot_ncon(times)
        plots['niter'] = self.plot_niter(times)
        plots['energy'] = self.plot_energy(times)
        plots['energy_potential'] = self.plot_energy(times)
        plots['energy_kinetic'] = self.plot_energy(times)
        return plots

    def plot_ncon(self, times: NDARRAY_V1) -> Figure:
        """Plot"""
        fig = plt.figure('ncon')
        plt.plot(times, self.ncon)
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Number of constraints')
        plt.grid(True)
        return fig

    def plot_niter(self, times: NDARRAY_V1) -> Figure:
        """Plot"""
        fig = plt.figure('niter')
        plt.plot(times, self.niter)
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Number of physics engine iterations')
        plt.grid(True)
        return fig

    def plot_energy(self, times: NDARRAY_V1) -> Figure:
        """Plot"""
        fig = plt.figure('energy')
        plt.plot(times, self.energy[:, 0], label='Potential')
        plt.plot(times, self.energy[:, 1], label='Kinetic')
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Energy [J]')
        plt.grid(True)
        return fig

    def plot_potential_energy(self, times: NDARRAY_V1) -> Figure:
        """Plot"""
        fig = plt.figure('energy')
        plt.plot(times, self.energy[:, 0], label='Potential')
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Energy [J]')
        plt.grid(True)
        return fig

    def plot_kinetic_energy(self, times: NDARRAY_V1) -> Figure:
        """Plot"""
        fig = plt.figure('energy')
        plt.plot(times, self.energy[:, 1], label='Kinetic')
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Energy [J]')
        plt.grid(True)
        return fig
