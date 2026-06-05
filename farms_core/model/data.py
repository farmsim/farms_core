"""Model data"""

from .. import pylog
from ..doc import ClassDoc, ChildDoc
from ..array.types import NDARRAY_V1
from ..simulation.options import SimulationOptions
from ..io.hdf5 import hdf5_to_dict, dict_to_hdf5
from ..sensors.data import SensorsData

from .options import AnimatOptions
from .data_cy import AnimatDataCy


class AnimatData(AnimatDataCy):
    """Animat data"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="animat data",
            description="Provides and logs the animat data.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="sensors",
                    class_type=SensorsData,
                    description="Contains the logged sensors data.",
                ),
                ChildDoc(
                    name="network",
                    class_type='NetworkLog',
                    description="Contains the logged network data.",
                ),
            ],
        )

    def __init__(
            self,
            sensors: SensorsData,
            network: 'NetworkLog'=None,  # ty:ignore[invalid-parameter-default]
    ):
        super().__init__()
        self.sensors = sensors
        self.network = network

    @classmethod
    def from_options(
            cls,
            animat_options: AnimatOptions,
            simulation_options: SimulationOptions,
    ):
        """Animat data from animat and simulation options"""
        return cls(
            sensors=SensorsData.from_options(
                animat_options=animat_options,
                simulation_options=simulation_options,
            ),
        )

    @classmethod
    def from_sensors_names(
            cls,
            buffer_size: int,
            **kwargs,
    ):
        """Animat data from sensors names"""
        return cls(
            sensors=SensorsData.from_names(
                buffer_size=buffer_size,
                links_names=kwargs.pop('links'),
                joints_names=kwargs.pop('joints'),
                contacts_names=kwargs.pop('contacts', []),
                xfrc_names=kwargs.pop('xfrc', []),
                muscles_names=kwargs.pop('muscles', []),
                adhesions_names=kwargs.pop('adhesions', []),
                visuals_names=kwargs.pop('visuals', []),
            ),
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
        network_data = None
        if "network" in dictionary:
            from farms_network.core.data import NetworkLog
            network_data = NetworkLog.from_dict(dictionary['network'])
        return cls(
            sensors=SensorsData.from_dict(dictionary['sensors']),
            network=network_data,
        )

    def to_dict(self, iteration: int | None = None) -> dict:
        """Convert data to dictionary"""
        _data = {'sensors': self.sensors.to_dict(iteration)}
        if self.network is not None:
            _data['network'] = self.network.to_dict(iteration)
        return _data

    def to_file(self, filename: str, iteration: int | None = None):
        """Save data to file"""
        pylog.info('Exporting to dictionary')
        data_dict = self.to_dict(iteration)
        pylog.info('Saving data to %s', filename)
        dict_to_hdf5(filename=filename, data=data_dict)
        pylog.info('Saved data to %s', filename)

    def plot_sensors(self, times: NDARRAY_V1) -> dict:
        """Plot"""
        return self.sensors.plot(times)
