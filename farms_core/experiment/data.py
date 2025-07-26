"""Experiment data"""

from matplotlib.figure import Figure

from .. import pylog
from ..array.types import NDARRAY_V1
from ..simulation.data import SimulationData
from ..model.data import AnimatData
from ..io.hdf5 import hdf5_to_dict, dict_to_hdf5

from .options import ExperimentOptions


class ExperimentData:
    """Experiment data"""

    def __init__(
            self,
            times: NDARRAY_V1,
            timestep: float,
            simulation: SimulationData,
            animats: list[AnimatData],
    ):
        super().__init__()
        self.times = times
        self.timestep = timestep
        self.simulation = simulation
        self.animats = animats

    @classmethod
    def from_options(
            cls,
            experiment_options: ExperimentOptions,
            animat_class=AnimatData,
    ):
        """Experiment data from experiment and simulation options"""
        simulation_options = experiment_options.simulation
        times = simulation_options.times()
        assert len(times) == simulation_options.n_iterations
        return cls(
            times=times,
            timestep=simulation_options.timestep,
            simulation=SimulationData.from_size(len(times)),
            animats=[
                animat_class.from_options(animat_options, simulation_options)
                for animat_options in experiment_options.animats
            ],
        )

    @classmethod
    def from_file(cls, filename: str):
        """From file"""
        pylog.info('Loading data from %s', filename)
        data = hdf5_to_dict(filename=filename)
        pylog.info('loaded data from %s', filename)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, dictionary: dict):
        """Load data from dictionary"""
        times = dictionary.get('times', [])
        assert 'animats' in dictionary, f'"animats" not in {dictionary.keys()=}'
        return cls(
            times=times,
            timestep=dictionary['timestep'],
            animats=[
                AnimatData.from_dict(animat)
                for animat in dictionary['animats']
            ],
            simulation=(
                SimulationData.from_dict(dictionary['simulation'])
                if 'simulation' in dictionary
                else SimulationData.from_size(len(times))
            ),
        )

    def to_dict(self, iteration: int | None = None) -> dict:
        """Convert data to dictionary"""
        return {
            'times': self.times,
            'timestep': self.timestep,
            'simulation': self.simulation.to_dict(iteration),
            'animats': [animat.to_dict(iteration) for animat in self.animats],
        }

    def to_file(self, filename: str, iteration: int | None = None):
        """Save data to file"""
        pylog.info('Exporting to dictionary')
        data_dict = self.to_dict(iteration)
        pylog.info('Saving data to %s', filename)
        dict_to_hdf5(filename=filename, data=data_dict)
        pylog.info('Saved data to %s', filename)

    def plot(self) -> dict[str, Figure]:
        """Plot all data"""
        figs: dict[str, Figure] = {}
        figs.update(self.plot_animats())
        figs.update(self.plot_simulation())
        return figs

    def plot_simulation(self) -> dict[str, Figure]:
        """Plot simulation data"""
        return self.simulation.plot(self.times)

    def plot_animats(self) -> dict[str, Figure]:
        """Plot animats data"""
        figs: dict[str, Figure] = {}
        for animat in self.animats:
            figs.update(animat.plot(self.times))
        return figs
