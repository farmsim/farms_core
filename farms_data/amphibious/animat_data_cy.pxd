"""Animat data"""

include 'types.pxd'
from ..sensors.data_cy cimport SensorsDataCy
from ..array.array_cy cimport (
    DoubleArray1D,
    DoubleArray2D,
    IntegerArray2D,
)


cpdef enum ConnectionType:
    OSC2OSC
    DRIVE2OSC
    POS2FREQ
    VEL2FREQ
    TOR2FREQ
    POS2AMP
    VEL2AMP
    TOR2AMP
    STRETCH2FREQ
    STRETCH2AMP
    REACTION2FREQ
    REACTION2AMP
    REACTION2FREQTEGOTAE
    FRICTION2FREQ
    FRICTION2AMP
    LATERAL2FREQ
    LATERAL2AMP


cdef class AnimatDataCy:
    """Network parameter"""
    cdef public OscillatorNetworkStateCy state
    cdef public NetworkParametersCy network
    cdef public JointsControlArrayCy joints
    cdef public SensorsDataCy sensors


cdef class NetworkParametersCy:
    """Network parameter"""
    cdef public DriveArrayCy drives
    cdef public OscillatorsCy oscillators
    cdef public OscillatorsConnectivityCy osc_connectivity
    cdef public ConnectivityCy drive_connectivity
    cdef public JointsConnectivityCy joints_connectivity
    cdef public ContactsConnectivityCy contacts_connectivity
    cdef public HydroConnectivityCy hydro_connectivity


cdef class OscillatorNetworkStateCy(DoubleArray2D):
    """Network state"""
    cdef public unsigned int n_oscillators


cdef class DriveArrayCy(DoubleArray2D):
    """Drive array"""

    cdef inline DTYPE c_speed(self, unsigned int iteration) nogil:
        """Value"""
        return self.array[iteration, 0]

    cdef inline DTYPE c_turn(self, unsigned int iteration) nogil:
        """Value"""
        return self.array[iteration, 1]


cdef class DriveDependentArrayCy(DoubleArray2D):
    """Drive dependent array"""

    cdef public DTYPE value(self, unsigned int index, DTYPE drive)

    cdef inline unsigned int c_n_nodes(self) nogil:
        """Number of nodes"""
        return self.array.shape[0]

    cdef inline DTYPE c_gain(self, unsigned int index) nogil:
        """Gain"""
        return self.array[index, 0]

    cdef inline DTYPE c_bias(self, unsigned int index) nogil:
        """Bias"""
        return self.array[index, 1]

    cdef inline DTYPE c_low(self, unsigned int index) nogil:
        """Low"""
        return self.array[index, 2]

    cdef inline DTYPE c_high(self, unsigned int index) nogil:
        """High"""
        return self.array[index, 3]

    cdef inline DTYPE c_saturation(self, unsigned int index) nogil:
        """Saturation"""
        return self.array[index, 4]

    cdef inline DTYPE c_value(self, unsigned int index, DTYPE drive) nogil:
        """Value"""
        return (
            (self.c_gain(index)*drive + self.c_bias(index))
            if self.c_low(index) <= drive <= self.c_high(index)
            else self.c_saturation(index)
        )

    cdef inline DTYPE c_value_mod(self, unsigned int index, DTYPE drive1, DTYPE drive2) nogil:
        """Value"""
        return (
            (self.c_gain(index)*drive1 + self.c_bias(index))
            if self.c_low(index) <= drive2 <= self.c_high(index)
            else self.c_saturation(index)
        )


cdef class OscillatorsCy:
    """Oscillator array"""
    cdef public DriveDependentArrayCy intrinsic_frequencies
    cdef public DriveDependentArrayCy nominal_amplitudes
    cdef public DoubleArray1D rates
    cdef public DoubleArray1D modular_phases
    cdef public DoubleArray1D modular_amplitudes

    cdef inline unsigned int c_n_oscillators(self) nogil:
        """Number of oscillators"""
        return self.rates.array.shape[0]

    cdef inline DTYPE c_angular_frequency(self, unsigned int index, DTYPE drive) nogil:
        """Angular frequency"""
        return self.intrinsic_frequencies.c_value(index, drive)

    cdef inline DTYPE c_nominal_amplitude(self, unsigned int index, DTYPE drive) nogil:
        """Nominal amplitude"""
        return self.nominal_amplitudes.c_value(index, drive)

    cdef inline DTYPE c_rate(self, unsigned int index) nogil:
        """Rate"""
        return self.rates.array[index]

    cdef inline DTYPE c_modular_phases(self, unsigned int index) nogil:
        """Modular phase"""
        return self.modular_phases.array[index]

    cdef inline DTYPE c_modular_amplitudes(self, unsigned int index) nogil:
        """Modular amplitude"""
        return self.modular_amplitudes.array[index]


cdef class ConnectivityCy:
    """Connectivity array"""

    cdef readonly IntegerArray2D connections

    cpdef UITYPE input(self, unsigned int connection_i)
    cpdef UITYPE output(self, unsigned int connection_i)
    cpdef UITYPE connection_type(self, unsigned int connection_i)

    cdef inline UITYPE c_n_connections(self) nogil:
        """Number of connections"""
        return self.connections.array.shape[0]


cdef class OscillatorsConnectivityCy(ConnectivityCy):
    """oscillator connectivity array"""

    cdef readonly DoubleArray1D weights
    cdef readonly DoubleArray1D desired_phases

    cdef inline DTYPE c_weight(self, unsigned int index) nogil:
        """Weight"""
        return self.weights.array[index]

    cdef inline DTYPE c_desired_phase(self, unsigned int index) nogil:
        """Desired phase"""
        return self.desired_phases.array[index]


cdef class JointsConnectivityCy(ConnectivityCy):
    """Joint connectivity array"""

    cdef readonly DoubleArray1D weights

    cdef inline DTYPE c_weight(self, unsigned int index) nogil:
        """Weight"""
        return self.weights.array[index]


cdef class ContactsConnectivityCy(ConnectivityCy):
    """Contact connectivity array"""

    cdef readonly DoubleArray1D weights

    cdef inline DTYPE c_weight(self, unsigned int index) nogil:
        """Weight"""
        return self.weights.array[index]


cdef class HydroConnectivityCy(ConnectivityCy):
    """Hydrodynamics connectivity array"""

    cdef readonly DoubleArray1D weights

    cdef inline DTYPE c_weights(self, unsigned int index) nogil:
        """Weight for hydrodynamics frequency"""
        return self.weights.array[index]


cdef class JointsControlArrayCy(DriveDependentArrayCy):
    """Drive dependent joints"""

    cdef inline unsigned int c_n_joints(self) nogil:
        """Number of joints"""
        return self.c_n_nodes()

    cdef inline DTYPE c_offset_desired(self, unsigned int index, DTYPE drive1, DTYPE drive2) nogil:
        """Desired offset"""
        return self.c_value_mod(index, drive1, drive2)

    cdef inline DTYPE c_rate(self, unsigned int index) nogil:
        """Rate"""
        return self.array[index, 5]
