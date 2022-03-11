"""Animat data"""

from typing import Any
import numpy as np
from nptyping import NDArray
from ..sensors.data import (
    SensorsData,
    LinkSensorArray,
    JointSensorArray,
    ContactsArray,
    HydrodynamicsArray,
)
from ..model.options import ControlOptions
from .animat_data_cy import ConnectivityCy, JointsControlArrayCy
from .animat_data import (
    OscillatorNetworkState,
    AnimatData,
    NetworkParameters,
    DriveArray,
    Oscillators,
    OscillatorConnectivity,
    JointsConnectivity,
    ContactsConnectivity,
    HydroConnectivity,
)


class JointsArray(JointsControlArrayCy):
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


class AmphibiousData(AnimatData):
    """Amphibious network parameter"""

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
            joints=JointsArray.from_options(control),
            sensors=SensorsData(
                links=LinkSensorArray.from_names(
                    names=control.sensors.links,
                    n_iterations=n_iterations,
                ),
                joints=joints_sensors,
                contacts=contacts,
                hydrodynamics=hydrodynamics,
            ),
        )
