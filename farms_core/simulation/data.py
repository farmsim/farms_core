"""Simulation data"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..array.array import to_array
from ..array.types import NDARRAY_V1, NDARRAY_V2


class SimulationData:
    """Simulation

    Contains logs from simulation data such as number physics engine iterations
    and system energy levels.

    """

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
    def from_size(cls, size: int):
        """Animat data from animat and simulation options"""
        return cls(
            ncon=np.zeros(size),
            niter=np.zeros(size),
            energy=np.zeros([size, 2]),
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
    ) -> dict:
        """Convert data to dictionary"""
        return {
            'ncon': to_array(self.ncon, iteration),
            'niter': to_array(self.niter, iteration),
            'energy': to_array(self.energy, iteration),
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
