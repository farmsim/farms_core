"""Animat data"""

import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
import farms_pylog as pylog
from ..sensors.array import DoubleArray1D
from ..sensors.data import SensorsData
from .animat_data_cy import (
    ConnectionType,
    AnimatDataCy,
    NetworkParametersCy,
    OscillatorNetworkStateCy,
    DriveArrayCy,
    DriveDependentArrayCy,
    OscillatorsCy,
    ConnectivityCy,
    OscillatorsConnectivityCy,
    JointsConnectivityCy,
    ContactsConnectivityCy,
    HydroConnectivityCy,
    JointsControlArrayCy,
)


NPDTYPE = np.float64
NPUITYPE = np.uintc


CONNECTIONTYPENAMES = [
    'OSC2OSC',
    'DRIVE2OSC',
    'POS2FREQ',
    'VEL2FREQ',
    'TOR2FREQ',
    'POS2AMP',
    'VEL2AMP',
    'TOR2AMP',
    'REACTION2FREQ',
    'REACTION2AMP',
    'FRICTION2FREQ',
    'FRICTION2AMP',
    'LATERAL2FREQ',
    'LATERAL2AMP',
]
CONNECTIONTYPE2NAME = dict(zip(ConnectionType, CONNECTIONTYPENAMES))
NAME2CONNECTIONTYPE = dict(zip(CONNECTIONTYPENAMES, ConnectionType))


def connections_from_connectivity(connectivity, map1=None, map2=None):
    """Connections from connectivity"""
    if map1 or map2:
        for connection in connectivity:
            if map1:
                assert connection['in'] in map1, '{} not in {}'.format(
                    connection['in'],
                    map1,
                )
            if map2:
                assert connection['out'] in map2, '{} not in {}'.format(
                    connection['out'],
                    map2,
                )
    return [
        [
            map1[connection['in']] if map1 else connection['in'],
            map2[connection['out']] if map2 else connection['out'],
            NAME2CONNECTIONTYPE[connection['type']]
        ]
        for connection in connectivity
    ]


def to_array(array, iteration=None):
    """To array or None"""
    if array is not None:
        array = np.array(array)
        if iteration is not None:
            array = array[:iteration]
    return array


def _dict_to_hdf5(handler, dict_data, group=None):
    """Dictionary to HDF5"""
    if group is not None and group not in handler:
        handler = handler.create_group(group)
    for key, value in dict_data.items():
        if isinstance(value, dict):
            _dict_to_hdf5(handler, value, key)
        elif value is None:
            handler.create_dataset(name=key, data=h5py.Empty(None))
        else:
            handler.create_dataset(name=key, data=value)


def _hdf5_to_dict(handler, dict_data):
    """HDF5 to dictionary"""
    for key, value in handler.items():
        if isinstance(value, h5py.Group):
            new_dict = {}
            dict_data[key] = new_dict
            _hdf5_to_dict(value, new_dict)
        else:
            if value.shape:
                if value.dtype == np.dtype('O'):
                    data = [val.decode("utf-8") for val in value]
                else:
                    data = np.array(value)
            elif value.shape is not None:
                data = np.array(value).item()
            else:
                data = None
            dict_data[key] = data


def hdf5_open(filename, mode='w', max_attempts=10, attempt_delay=0.1):
    """Open HDF5 file with delayed attempts"""
    for attempt in range(max_attempts):
        try:
            hfile = h5py.File(name=filename, mode=mode)
            break
        except OSError as err:
            if attempt == max_attempts - 1:
                pylog.error('File {} was locked during more than {} [s]'.format(
                    filename,
                    max_attempts*attempt_delay,
                ))
                raise err
            pylog.warning('File {} seems locked during attempt {}/{}'.format(
                filename,
                attempt+1,
                max_attempts,
            ))
            time.sleep(attempt_delay)
    return hfile


def dict_to_hdf5(filename, data, mode='w', **kwargs):
    """Dictionary to HDF5"""
    hfile = hdf5_open(filename, mode=mode, **kwargs)
    _dict_to_hdf5(hfile, data)
    hfile.close()


def hdf5_to_dict(filename, **kwargs):
    """HDF5 to dictionary"""
    data = {}
    hfile = hdf5_open(filename, mode='r', **kwargs)
    _hdf5_to_dict(hfile, data)
    hfile.close()
    return data


def hdf5_keys(filename, **kwargs):
    """HDF5 to dictionary"""
    hfile = hdf5_open(filename, mode='r', **kwargs)
    keys = list(hfile.keys())
    hfile.close()
    return keys


def hdf5_get(filename, key, **kwargs):
    """HDF5 to dictionary"""
    dict_data = {}
    hfile = hdf5_open(filename, mode='r', **kwargs)
    handler = hfile
    for _key in key:
        handler = handler[_key]
    _hdf5_to_dict(handler, dict_data)
    hfile.close()
    return dict_data


class ModelData(AnimatDataCy):
    """Model data"""

    def __init__(self, timestep, sensors=None):
        super(ModelData, self).__init__()
        self.timestep = timestep
        self.sensors = sensors

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            timestep=dictionary['timestep'],
            sensors=SensorsData.from_dict(dictionary['sensors']),
        )

    @classmethod
    def from_file(cls, filename, n_oscillators=0):
        """From file"""
        pylog.info('Loading data from {}'.format(filename))
        data = hdf5_to_dict(filename=filename)
        pylog.info('loaded data from {}'.format(filename))
        return cls.from_dict(data)

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        return {
            'timestep': self.timestep,
            'sensors': self.sensors.to_dict(iteration),
        }

    def to_file(self, filename, iteration=None):
        """Save data to file"""
        pylog.info('Exporting to dictionary')
        data_dict = self.to_dict(iteration)
        pylog.info('Saving data to {}'.format(filename))
        dict_to_hdf5(filename=filename, data=data_dict)
        pylog.info('Saved data to {}'.format(filename))

    def plot_sensors(self, times):
        """Plot"""
        self.sensors.plot(times)


class AnimatData(ModelData):
    """Animat data"""

    def __init__(
            self, timestep,
            state=None, network=None,
            joints=None, sensors=None,
    ):
        super(AnimatData, self).__init__(timestep=timestep, sensors=sensors)
        self.state = state
        self.network = network
        self.joints = joints

    @classmethod
    def from_dict(cls, dictionary, n_oscillators=0):
        """Load data from dictionary"""
        return cls(
            timestep=dictionary['timestep'],
            state=OscillatorNetworkState(dictionary['state'], n_oscillators),
            network=NetworkParameters.from_dict(dictionary['network']),
            joints=JointsArray(dictionary['joints']),
            sensors=SensorsData.from_dict(dictionary['sensors']),
        )

    @classmethod
    def from_file(cls, filename):
        """From file"""
        pylog.info('Loading data from {}'.format(filename))
        data = hdf5_to_dict(filename=filename)
        pylog.info('loaded data from {}'.format(filename))
        n_oscillators = len(data['network']['oscillators']['names'])
        return cls.from_dict(data, n_oscillators)

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        data_dict = super().to_dict(iteration=iteration)
        data_dict.update({
            'state': to_array(self.state.array) if self.state is not None else None,
            'network': self.network.to_dict(iteration) if self.network is not None else None,
            'joints': to_array(self.joints.array),
        })
        return data_dict

    def plot(self, times):
        """Plot"""
        self.state.plot(times)
        self.plot_sensors(times)


class NetworkParameters(NetworkParametersCy):
    """Network parameter"""

    def __init__(
            self,
            drives,
            oscillators,
            osc_connectivity,
            drive_connectivity,
            joints_connectivity,
            contacts_connectivity,
            hydro_connectivity
    ):
        super(NetworkParameters, self).__init__()
        self.drives = drives
        self.oscillators = oscillators
        self.drive_connectivity = drive_connectivity
        self.joints_connectivity = joints_connectivity
        self.osc_connectivity = osc_connectivity
        self.contacts_connectivity = contacts_connectivity
        self.hydro_connectivity = hydro_connectivity

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            drives=DriveArray(
                dictionary['drives']
            ),
            oscillators=Oscillators.from_dict(
                dictionary['oscillators']
            ),
            osc_connectivity=OscillatorConnectivity.from_dict(
                dictionary['osc_connectivity']
            ),
            drive_connectivity=ConnectivityCy(
                dictionary['drive_connectivity']
            ),
            joints_connectivity=JointsConnectivity.from_dict(
                dictionary['joints_connectivity']
            ),
            contacts_connectivity=ContactsConnectivity.from_dict(
                dictionary['contacts_connectivity']
            ),
            hydro_connectivity=HydroConnectivity.from_dict(
                dictionary['hydro_connectivity']
            ),
        ) if dictionary else None

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {
            'drives': to_array(self.drives.array),
            'oscillators': self.oscillators.to_dict(),
            'osc_connectivity': self.osc_connectivity.to_dict(),
            'drive_connectivity': self.drive_connectivity.connections.array,
            'joints_connectivity': self.joints_connectivity.to_dict(),
            'contacts_connectivity': self.contacts_connectivity.to_dict(),
            'hydro_connectivity': self.hydro_connectivity.to_dict(),
        }


class OscillatorNetworkState(OscillatorNetworkStateCy):
    """Network state"""

    def __init__(self, state, n_oscillators):
        super(OscillatorNetworkState, self).__init__(state)
        self.n_oscillators = n_oscillators

    @classmethod
    def from_options(cls, state, animat_options):
        """From options"""
        return cls(
            state=state,
            n_oscillators=2*animat_options.morphology.n_joints()
        )

    @classmethod
    def from_solver(cls, solver, n_oscillators):
        """From solver"""
        return cls(solver.state, n_oscillators, solver.iteration)

    def phases(self, iteration):
        """Phases"""
        return self.array[iteration, :self.n_oscillators]

    def phases_all(self):
        """Phases"""
        return self.array[:, :self.n_oscillators]

    def amplitudes(self, iteration):
        """Amplitudes"""
        return self.array[iteration, self.n_oscillators:]

    def amplitudes_all(self):
        """Phases"""
        return self.array[:, self.n_oscillators:]

    @classmethod
    def from_initial_state(cls, initial_state, n_iterations):
        """From initial state"""
        state_size = len(initial_state)
        state_array = np.zeros([n_iterations, state_size], dtype=NPDTYPE)
        state_array[0, :] = initial_state
        return cls(state_array, n_oscillators=2*state_size//5)

    def plot(self, times):
        """Plot"""
        self.plot_phases(times)
        self.plot_amplitudes(times)
        self.plot_neural_activity_normalised(times)

    def plot_phases(self, times):
        """Plot phases"""
        plt.figure('Network state phases')
        for data in np.transpose(self.phases_all()):
            plt.plot(times, data[:len(times)])
        plt.xlabel('Times [s]')
        plt.ylabel('Phases [rad]')
        plt.grid(True)

    def plot_amplitudes(self, times):
        """Plot amplitudes"""
        plt.figure('Network state amplitudes')
        for data in np.transpose(self.amplitudes_all()):
            plt.plot(times, data[:len(times)])
        plt.xlabel('Times [s]')
        plt.ylabel('Amplitudes')
        plt.grid(True)

    def plot_phases(self, times):
        """Plot phases"""
        plt.figure('Network state phases')
        for data in np.transpose(self.phases_all()):
            plt.plot(times, data[:len(times)])
        plt.xlabel('Times [s]')
        plt.ylabel('Phases [rad]')
        plt.grid(True)

    def plot_neural_activity_normalised(self, times):
        """Plot amplitudes"""
        plt.figure('Neural activities (normalised)')
        for data_i, data in enumerate(np.transpose(self.phases_all())):
            plt.plot(times, 2*data_i + 0.5*(1 + np.cos(data[:len(times)])))
        plt.xlabel('Times [s]')
        plt.ylabel('Neural activity')
        plt.grid(True)


class DriveArray(DriveArrayCy):
    """Drive array"""

    @classmethod
    def from_initial_drive(cls, initial_drives, n_iterations):
        """From initial drive"""
        drive_size = len(initial_drives)
        drive_array = np.zeros([n_iterations, drive_size], dtype=NPDTYPE)
        drive_array[0, :] = initial_drives
        return cls(drive_array)


class DriveDependentArray(DriveDependentArrayCy):
    """Drive dependent array"""

    @classmethod
    def from_vectors(cls, gain, bias, low, high, saturation):
        """From each parameter"""
        return cls(np.array([gain, bias, low, high, saturation]))


class Oscillators(OscillatorsCy):
    """Oscillator array"""

    def __init__(
            self, names, intrinsic_frequencies, nominal_amplitudes, rates,
            modular_phases, modular_amplitudes,
    ):
        super(Oscillators, self).__init__()
        self.names = names
        self.intrinsic_frequencies = DriveDependentArray(intrinsic_frequencies)
        self.nominal_amplitudes = DriveDependentArray(nominal_amplitudes)
        self.rates = DoubleArray1D(rates)
        self.modular_phases = DoubleArray1D(modular_phases)
        self.modular_amplitudes = DoubleArray1D(modular_amplitudes)

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            names=dictionary['names'],
            intrinsic_frequencies=dictionary['intrinsic_frequencies'],
            nominal_amplitudes=dictionary['nominal_amplitudes'],
            rates=dictionary['rates'],
            modular_phases=dictionary['modular_phases'],
            modular_amplitudes=dictionary['modular_amplitudes'],
        )

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {
            'names': self.names,
            'intrinsic_frequencies': to_array(self.intrinsic_frequencies.array),
            'nominal_amplitudes': to_array(self.nominal_amplitudes.array),
            'rates': to_array(self.rates.array),
            'modular_phases': to_array(self.modular_phases.array),
            'modular_amplitudes': to_array(self.modular_amplitudes.array),
        }

    @classmethod
    def from_options(cls, network):
        """Default"""
        freqs, amplitudes = [
            np.array([
                [
                    freq['gain'],
                    freq['bias'],
                    freq['low'],
                    freq['high'],
                    freq['saturation'],
                ]
                for freq in option
            ], dtype=NPDTYPE)
            for option in [network.osc_frequencies(), network.osc_amplitudes()]
        ]
        return cls(
            network.osc_names(),
            freqs,
            amplitudes,
            np.array(network.osc_rates(), dtype=NPDTYPE),
            np.array(network.osc_modular_phases(), dtype=NPDTYPE),
            np.array(network.osc_modular_amplitudes(), dtype=NPDTYPE),
        )


class OscillatorConnectivity(OscillatorsConnectivityCy):
    """Connectivity array"""

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            connections=dictionary['connections'],
            weights=dictionary['weights'],
            desired_phases=dictionary['desired_phases'],
        )

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {
            'connections': to_array(self.connections.array),
            'weights': to_array(self.weights.array),
            'desired_phases': to_array(self.desired_phases.array),
        }

    @classmethod
    def from_connectivity(cls, connectivity, **kwargs):
        """From connectivity"""
        connections = connections_from_connectivity(connectivity, **kwargs)
        weights = [
            connection['weight']
            for connection in connectivity
        ]
        phase_bias = [
            connection['phase_bias']
            for connection in connectivity
        ]
        return cls(
            connections=np.array(connections, dtype=NPUITYPE),
            weights=np.array(weights, dtype=NPDTYPE),
            desired_phases=np.array(phase_bias, dtype=NPDTYPE),
        )


class JointsConnectivity(JointsConnectivityCy):
    """Connectivity array"""

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            connections=dictionary['connections'],
            weights=dictionary['weights'],
        )

    def to_dict(self, _iteration=None):
        """Convert data to dictionary"""
        return {
            'connections': to_array(self.connections.array),
            'weights': to_array(self.weights.array),
        }

    @classmethod
    def from_connectivity(cls, connectivity, **kwargs):
        """From connectivity"""
        connections = connections_from_connectivity(connectivity, **kwargs)
        weights = [
            connection['weight']
            for connection in connectivity
        ]
        return cls(
            np.array(connections, dtype=NPUITYPE),
            np.array(weights, dtype=NPDTYPE),
        )


class ContactsConnectivity(ContactsConnectivityCy):
    """Connectivity array"""

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            connections=dictionary['connections'],
            weights=dictionary['weights'],
        )

    def to_dict(self, _iteration=None):
        """Convert data to dictionary"""
        return {
            'connections': to_array(self.connections.array),
            'weights': to_array(self.weights.array),
        }

    @classmethod
    def from_connectivity(cls, connectivity, **kwargs):
        """From connectivity"""
        connections = connections_from_connectivity(connectivity, **kwargs)
        weights = [
            connection['weight']
            for connection in connectivity
        ]
        return cls(
            np.array(connections, dtype=NPUITYPE),
            np.array(weights, dtype=NPDTYPE),
        )


class HydroConnectivity(HydroConnectivityCy):
    """Connectivity array"""

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            connections=dictionary['connections'],
            weights=dictionary['weights'],
        )

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        assert iteration is None or isinstance(iteration, int)
        return {
            'connections': to_array(self.connections.array),
            'weights': to_array(self.weights.array),
        }

    @classmethod
    def from_connectivity(cls, connectivity, **kwargs):
        """From connectivity"""
        connections = connections_from_connectivity(connectivity, **kwargs)
        weights = [
            connection['weight']
            for connection in connectivity
        ]
        return cls(
            connections=np.array(connections, dtype=NPUITYPE),
            weights=np.array(weights, dtype=NPDTYPE),
        )


class JointsArray(JointsControlArrayCy):
    """Oscillator array"""

    @classmethod
    def from_options(cls, control):
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
                control.joints_rates(),
            )
        ], dtype=NPDTYPE))
