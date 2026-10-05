"""Experiment data"""

import numpy as np
from matplotlib.figure import Figure

from .. import pylog
from ..doc import ClassDoc, ChildDoc
from ..array.array import to_array
from ..array.types import NDARRAY_V1
from ..array.array_cy import DoubleArray1D
from ..simulation.data import SimulationData
from ..extensions.extensions import import_item
from ..model.data import AnimatData
from ..io.hdf5 import hdf5_to_dict, dict_to_hdf5

from .options import ExperimentOptions


class ExperimentData:
    """Experiment data"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="experiment data",
            description="Provides and logs the experiment data.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="times",
                    class_type=DoubleArray1D,
                    description="The vector of logged times across all data.",
                ),
                ChildDoc(
                    name="timestep",
                    class_type=float,
                    description="The simulation timestep (Must be positive).",
                ),
                ChildDoc(
                    name="simulation",
                    class_type=SimulationData,
                    description=(
                        "The simulation data, mostly related to the physics"
                        " engine."
                    ),
                ),
                ChildDoc(
                    name="simulation",
                    class_type="list[AnimatData]",
                    class_link=AnimatData,
                    description="List of data for each animat",
                ),
            ],
        )

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
    ):
        """Experiment data from experiment and simulation options"""
        simulation_options = experiment_options.simulation
        times = simulation_options.times()
        n_iterations = simulation_options.runtime.n_iterations
        assert len(times) == n_iterations
        buffer_size = simulation_options.runtime.buffer_size
        if buffer_size == 0:
            buffer_size = n_iterations
        animat_data_loaders = [
            import_item(animat_data_loader)
            for animat_data_loader in experiment_options.loaders.animats_data
        ]
        return cls(
            times=np.zeros(buffer_size),
            timestep=simulation_options.physics.timestep,
            simulation=SimulationData.from_size(
                n_iterations,
                buffer_size=buffer_size,
            ),
            animats=[
                animat_loader.from_options(animat_options, simulation_options)
                for animat_loader, animat_options in zip(
                        animat_data_loaders,
                        experiment_options.animats,
                )
            ],
        )

    # @classmethod
    # def from_animats_names(
    #         cls,
    #         times: NDARRAY_V1,
    #         timestep: float,
    #         buffer_size: int,
    #         **kwargs,
    # ):
    #     """Experiment data from animats names"""
    #     return cls(
    #         times=times,
    #         timestep=timestep,
    #         animats=[AnimatData.from_names(
    #             buffer_size=buffer_size,
    #             links_names=kwargs.pop('links'),
    #             joints_names=kwargs.pop('joints'),
    #             contacts_names=kwargs.pop('contacts', []),
    #             xfrc_names=kwargs.pop('xfrc', []),
    #             muscles_names=kwargs.pop('muscles', [])
    #         )],
    #         simulation=SimulationData.from_size(len(times)),
    #     )

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

    def to_dict(
            self,
            iteration: int | None = None,
            start_iteration: int | None = None,
            skip: int = 1,
    ) -> dict:
        """Convert data to dictionary"""
        return {
            'times': to_array(self.times, iteration, start_iteration, skip),
            'timestep': self.timestep,
            'simulation': self.simulation.to_dict(iteration, start_iteration, skip),
            'animats': [
                animat.to_dict(iteration, start_iteration, skip)
                for animat in self.animats
            ],
        }

    def to_file(
            self,
            filename: str,
            iteration: int | None = None,
            start_iteration: int | None = None,
            mode: str = 'w',
            skip: int = 1,
    ):
        """Save data to file"""
        pylog.info('Exporting to dictionary')
        data_dict = self.to_dict(iteration, start_iteration, skip)
        pylog.info('Saving data to %s', filename)
        dict_to_hdf5(filename=filename, data=data_dict, mode=mode)
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
