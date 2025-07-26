"""Model data"""

from .. import pylog
from ..array.types import NDARRAY_V1
from ..simulation.options import SimulationOptions
from ..io.hdf5 import hdf5_to_dict, dict_to_hdf5
from ..sensors.data import SensorsData
from ..doc import ClassDoc

from .options import AnimatOptions
from .data_cy import AnimatDataCy


class AnimatData(AnimatDataCy):
    """Animat data"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name='sensors',
            description='Contains the logged sensors data.',
            class_type=cls,
            children=[SensorsData],
        )

    def __init__(
            self,
            sensors: SensorsData,
    ):
        super().__init__()
        self.sensors = sensors

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
        return cls(
            sensors=SensorsData.from_dict(dictionary['sensors']),
        )

    def to_dict(self, iteration: int | None = None) -> dict:
        """Convert data to dictionary"""
        return {
            'sensors': self.sensors.to_dict(iteration),
        }

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
