"""Model data"""

from typing import Dict, Any
from nptyping import NDArray
import farms_pylog as pylog

from ..io.hdf5 import (
    hdf5_to_dict,
    dict_to_hdf5,
)
from ..sensors.data import (
    SensorsData,
    LinkSensorArray,
    JointSensorArray,
    ContactsArray,
    HydrodynamicsArray,
)

from .data_cy import AnimatDataCy


class ModelData(AnimatDataCy):
    """Model data"""

    def __init__(
            self,
            timestep: float,
            sensors: SensorsData,
    ):
        super().__init__()
        self.timestep = timestep
        self.sensors = sensors

    @classmethod
    def from_sensors_names(
            cls,
            timestep: float,
            n_iterations: int,
            **kwargs,
    ):
        """Default amphibious newtwork parameters"""
        sensors = SensorsData(
            links=LinkSensorArray.from_names(
                kwargs.pop('links'),
                n_iterations,
            ),
            joints=JointSensorArray.from_names(
                kwargs.pop('joints'),
                n_iterations,
            ),
            contacts=ContactsArray.from_names(
                kwargs.pop('contacts', []),
                n_iterations,
            ),
            hydrodynamics=HydrodynamicsArray.from_names(
                kwargs.pop('hydrodynamics', []),
                n_iterations,
            ),
        )
        return cls(timestep=timestep, sensors=sensors)

    @classmethod
    def from_file(cls, filename: str):
        """From file"""
        pylog.info('Loading data from %s', filename)
        data = hdf5_to_dict(filename=filename)
        pylog.info('loaded data from %s', filename)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Load data from dictionary"""
        return cls(
            timestep=dictionary['timestep'],
            sensors=SensorsData.from_dict(dictionary['sensors']),
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        return {
            'timestep': self.timestep,
            'sensors': self.sensors.to_dict(iteration),
        }

    def to_file(self, filename: str, iteration: int = None):
        """Save data to file"""
        pylog.info('Exporting to dictionary')
        data_dict = self.to_dict(iteration)
        pylog.info('Saving data to %s', filename)
        dict_to_hdf5(filename=filename, data=data_dict)
        pylog.info('Saved data to %s', filename)

    def plot_sensors(self, times: NDArray[(Any,), float]) -> Dict:
        """Plot"""
        return self.sensors.plot(times)
