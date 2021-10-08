"""Animat data"""

from ..sensors.data import (
    SensorsData,
    LinkSensorArray,
    JointSensorArray,
    ContactsArray,
    HydrodynamicsArray,
)
from .animat_data_cy import ConnectivityCy
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
    JointsArray,
)


class AmphibiousData(AnimatData):
    """Amphibious network parameter"""

    @classmethod
    def from_options(
            cls,
            control,
            n_iterations,
            timestep,
    ):
        """Default amphibious newtwork parameters"""
        state = OscillatorNetworkState.from_initial_state(
            control.network.state_init(),
            n_iterations,
        ) if control.network is not None else None
        oscillators = Oscillators.from_options(
            control.network,
        ) if control.network is not None else None
        joints_sensors = JointSensorArray.from_names(
            control.sensors.joints,
            n_iterations,
        )
        contacts = ContactsArray.from_names(
            control.sensors.contacts,
            n_iterations,
        )
        hydrodynamics = HydrodynamicsArray.from_names(
            control.sensors.hydrodynamics,
            n_iterations,
        )
        oscillators_map, joints_map, contacts_map, hydrodynamics_map = [
            {
                name: element_i
                for element_i, name in enumerate(element.names)
            }
            for element in (oscillators, joints_sensors, contacts, hydrodynamics)
        ] if control.network is not None else (None, None, None, None)
        network = NetworkParameters(
            drives=DriveArray.from_initial_drive(
                control.network.drives_init(),
                n_iterations,
            ),
            oscillators=oscillators,
            osc_connectivity=OscillatorConnectivity.from_connectivity(
                control.network.osc2osc,
                map1=oscillators_map,
                map2=oscillators_map,
            ),
            drive_connectivity=ConnectivityCy(
                control.network.drive2osc,
            ),
            joints_connectivity=JointsConnectivity.from_connectivity(
                control.network.joint2osc,
                map1=oscillators_map,
                map2=joints_map,
            ),
            contacts_connectivity=ContactsConnectivity.from_connectivity(
                control.network.contact2osc,
                map1=oscillators_map,
                map2=contacts_map,
            ),
            hydro_connectivity=HydroConnectivity.from_connectivity(
                control.network.hydro2osc,
                map1=oscillators_map,
                map2=hydrodynamics_map,
            ),
        ) if control.network is not None else None
        joints_control = JointsArray.from_options(control)
        sensors = SensorsData(
            links=LinkSensorArray.from_names(
                control.sensors.links,
                n_iterations,
            ),
            joints=joints_sensors,
            contacts=contacts,
            hydrodynamics=hydrodynamics,
        )
        return cls(timestep, state, network, joints_control, sensors)
