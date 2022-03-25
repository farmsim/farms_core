"""Amphibious data"""

from typing import Any, Dict
import numpy as np
from nptyping import NDArray

import farms_pylog as pylog

from ..io.hdf5 import hdf5_to_dict
from ..array.array import to_array
from ..model.data import ModelData
from ..model.options import ControlOptions
from ..sensors.data import (
    SensorsData,
    LinkSensorArray,
    JointSensorArray,
    ContactsArray,
    HydrodynamicsArray,
)

from .data_cy import AmphibiousDataCy, ConnectivityCy, JointsControlArrayCy
from .network import (
    OscillatorNetworkState,
    NetworkParameters,
    DriveArray,
    Oscillators,
    OscillatorConnectivity,
    JointsConnectivity,
    ContactsConnectivity,
    HydroConnectivity,
)


class JointsControlArray(JointsControlArrayCy):
    """Oscillator array"""

    @classmethod
    def from_options(cls, control: ControlOptions):
        """Default"""
        return cls(np.array([
            [
                offset['gain'],
                offset['bias'],
                offset['low'],
                offset['high'],
                offset['saturation'],
                rate,
            ]
            for offset, rate in zip(
                control.joints_offsets(),
                control.joints_offset_rates(),
            )
        ], dtype=np.double))


class AmphibiousData(AmphibiousDataCy, ModelData):
    """Animat data"""

    def __init__(
            self,
            state: OscillatorNetworkState,
            network: NetworkParameters,
            joints: JointsControlArray,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.state = state
        self.network = network
        self.joints = joints

    @classmethod
    def from_file(cls, filename: str):
        """From file"""
        pylog.info('Loading data from %s', filename)
        data = hdf5_to_dict(filename=filename)
        pylog.info('loaded data from %s', filename)
        data['n_oscillators'] = len(data['network']['oscillators']['names'])
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, dictionary: Dict):
        """Load data from dictionary"""
        n_oscillators = dictionary.pop('n_oscillators')
        return cls(
            timestep=dictionary['timestep'],
            state=OscillatorNetworkState(dictionary['state'], n_oscillators),
            network=NetworkParameters.from_dict(dictionary['network']),
            joints=JointsControlArrayCy(dictionary['joints']),
            sensors=SensorsData.from_dict(dictionary['sensors']),
        )

    def to_dict(self, iteration: int = None) -> Dict:
        """Convert data to dictionary"""
        data_dict = super().to_dict(iteration=iteration)
        data_dict.update({
            'state': to_array(self.state.array),
            'network': self.network.to_dict(iteration),
            'joints': to_array(self.joints.array),
        })
        return data_dict

    def plot(self, times: NDArray[(Any,), float]) -> Dict:
        """Plot"""
        plots = {}
        plots.update(self.state.plot(times))
        plots.update(self.plot_sensors(times))
        plots['drives'] = self.network.drives.plot(times)
        return plots

    @classmethod
    def from_options(
            cls,
            control: ControlOptions,
            initial_state: NDArray[(Any,), np.double],
            n_iterations: int,
            timestep: float,
    ):
        """Default amphibious newtwork parameters"""
        assert isinstance(control.n_oscillators, int), control.n_oscillators
        oscillators = Oscillators.from_options(
            network=control.network,
        ) if control.network is not None else None
        joints_sensors = JointSensorArray.from_names(
            names=control.sensors.joints,
            n_iterations=n_iterations,
        )
        contacts = ContactsArray.from_names(
            names=control.sensors.contacts,
            n_iterations=n_iterations,
        )
        hydrodynamics = HydrodynamicsArray.from_names(
            names=control.sensors.hydrodynamics,
            n_iterations=n_iterations,
        )
        oscillators_map, joints_map, contacts_map, hydrodynamics_map = [
            {
                name: element_i
                for element_i, name in enumerate(element.names)
            }
            for element in (
                    oscillators,
                    joints_sensors,
                    contacts,
                    hydrodynamics,
            )
        ] if control.network is not None else (None, None, None, None)
        network = (
            NetworkParameters(
                drives=DriveArray.from_initial_drive(
                    initial_drives=control.network.drives_init(),
                    n_iterations=n_iterations,
                ),
                oscillators=oscillators,
                osc_connectivity=OscillatorConnectivity.from_connectivity(
                    connectivity=control.network.osc2osc,
                    map1=oscillators_map,
                    map2=oscillators_map,
                ),
                drive_connectivity=ConnectivityCy(
                    connections=control.network.drive2osc,
                ),
                joints_connectivity=JointsConnectivity.from_connectivity(
                    connectivity=control.network.joint2osc,
                    map1=oscillators_map,
                    map2=joints_map,
                ),
                contacts_connectivity=(
                    ContactsConnectivity.from_connectivity(
                        connectivity=control.network.contact2osc,
                        map1=oscillators_map,
                        map2=contacts_map,
                    )
                ),
                hydro_connectivity=HydroConnectivity.from_connectivity(
                    connectivity=control.network.hydro2osc,
                    map1=oscillators_map,
                    map2=hydrodynamics_map,
                ),
            )
            if control.network is not None
            else None
        )
        return cls(
            timestep=timestep,
            sensors=SensorsData(
                links=LinkSensorArray.from_names(
                    names=control.sensors.links,
                    n_iterations=n_iterations,
                ),
                joints=joints_sensors,
                contacts=contacts,
                hydrodynamics=hydrodynamics,
            ),
            state=(
                OscillatorNetworkState.from_initial_state(
                    initial_state=initial_state,
                    n_iterations=n_iterations,
                    n_oscillators=control.n_oscillators,
                )
                if control.network is not None
                else None
            ),
            network=network,
            joints=JointsControlArray.from_options(control),
        )
