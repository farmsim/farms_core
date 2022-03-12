"""Animat data"""

from typing import Any
import numpy as np
from nptyping import NDArray
cimport numpy as np


cdef class AnimatDataCy:
    """Network parameter"""
    pass


cdef class NetworkParametersCy:
    """Network parameter"""
    pass


cdef class OscillatorNetworkStateCy(DoubleArray2D):
    """Network state"""

    def __init__(
            self,
            array: NDArray[(Any, Any), np.double],
            n_oscillators: int,
    ):
        super().__init__(array=array)
        self.n_oscillators = n_oscillators

    cpdef DoubleArray1D phases(self, unsigned int iteration):
        """Phases"""
        return self.array[iteration, :self.n_oscillators]

    cpdef DoubleArray2D phases_all(self):
        """Phases"""
        return self.array[:, :self.n_oscillators]

    cpdef DoubleArray1D amplitudes(self, unsigned int iteration):
        """Amplitudes"""
        return self.array[iteration, self.n_oscillators:]

    cpdef DoubleArray2D amplitudes_all(self):
        """Phases"""
        return self.array[:, self.n_oscillators:]


cdef class DriveDependentArrayCy(DoubleArray2D):
    """Drive dependent array"""

    def __init__(
            self,
            array: NDArray[(Any, Any), np.double],
    ):
        super().__init__(array=array)
        self.n_nodes = np.shape(array)[0]

    cdef DTYPE value(self, unsigned int index, DTYPE drive):
        """Value for a given drive"""
        return (
            self.gain[index]*drive + self.bias[index]
            if self.low[index] <= drive <= self.high[index]
            else self.saturation[index]
        )


cdef class OscillatorsCy:
    """Oscillator array"""

    def __init__(
            self,
            n_oscillators: int,
    ):
        super().__init__()
        self.n_oscillators = n_oscillators


cdef class ConnectivityCy:
    """Connectivity array"""

    def __init__(
            self,
            connections: NDArray[(Any, 3), Any],
    ):
        super(ConnectivityCy, self).__init__()
        if connections is not None and list(connections):
            shape = np.shape(connections)
            assert shape[1] == 3, (
                f'Connections should be of dim 3, got {shape[1]}'
            )
            self.n_connections = shape[0]
            self.connections = IntegerArray2D(connections)
        else:
            self.n_connections = 0
            self.connections = IntegerArray2D(None)

    cpdef UITYPE input(self, unsigned int connection_i):
        """Node input"""
        self.array[connection_i, 0]

    cpdef UITYPE output(self, unsigned int connection_i):
        """Node output"""
        self.array[connection_i, 1]

    cpdef UITYPE connection_type(self, unsigned int connection_i):
        """Connection type"""
        self.array[connection_i, 2]


cdef class OscillatorsConnectivityCy(ConnectivityCy):
    """Oscillator connectivity array"""

    def __init__(
            self,
            connections: NDArray[(Any, 3), Any],
            weights: NDArray[(Any,), np.double],
            desired_phases: NDArray[(Any,), np.double],
    ):
        super(OscillatorsConnectivityCy, self).__init__(connections)
        if connections is not None and list(connections):
            size = np.shape(connections)[0]
            assert size == len(weights), (
                f'Size of connections {size}'
                f' != size of size of weights {len(weights)}'
            )
            assert size == len(desired_phases), (
                f'Size of connections {size}'
                f' != size of size of phases {len(desired_phases)}'
            )
            self.weights = DoubleArray1D(weights)
            self.desired_phases = DoubleArray1D(desired_phases)
        else:
            self.weights = DoubleArray1D(None)
            self.desired_phases = DoubleArray1D(None)


cdef class JointsConnectivityCy(ConnectivityCy):
    """Joint connectivity array"""

    def __init__(
            self,
            connections: NDArray[(Any, 3), Any],
            weights: NDArray[(Any,), np.double],
    ):
        super(JointsConnectivityCy, self).__init__(connections)
        if connections is not None and list(connections):
            size = np.shape(connections)[0]
            assert size == len(weights), (
                f'Size of connections {size}'
                f' != size of size of weights {len(weights)}'
            )
            self.weights = DoubleArray1D(weights)
        else:
            self.weights = DoubleArray1D(None)


cdef class ContactsConnectivityCy(ConnectivityCy):
    """Contact connectivity array"""

    def __init__(
            self,
            connections: NDArray[(Any, 3), Any],
            weights: NDArray[(Any,), np.double],
    ):
        super(ContactsConnectivityCy, self).__init__(connections)
        if connections is not None and list(connections):
            size = np.shape(connections)[0]
            assert size == len(weights), (
                f'Size of connections {size}'
                f' != size of size of weights {len(weights)}'
            )
            self.weights = DoubleArray1D(weights)
        else:
            self.weights = DoubleArray1D(None)


cdef class HydroConnectivityCy(ConnectivityCy):
    """Connectivity array"""

    def __init__(
            self,
            connections: NDArray[(Any, 3), Any],
            weights: NDArray[(Any,), np.double],
    ):
        super(HydroConnectivityCy, self).__init__(connections)
        if connections is not None and list(connections):
            size = np.shape(connections)[0]
            assert size == len(weights), (
                f'Size of connections {size}'
                f' != size of size of weights {len(weights)}'
            )
            self.weights = DoubleArray1D(weights)
        else:
            self.weights = DoubleArray1D(None)
