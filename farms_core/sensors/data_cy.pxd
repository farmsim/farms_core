"""Animat data"""

include 'types.pxd'
include 'sensor_convention.pxd'
from ..array.array_cy cimport DoubleArray3D


cdef class SensorsDataCy:
    """Sensors data"""
    cdef public LinkSensorArrayCy links
    cdef public JointSensorArrayCy joints
    cdef public ContactsArrayCy contacts
    cdef public XfrcArrayCy xfrc
    cdef public MusclesArrayCy muscles


cdef class ContactsArrayCy(DoubleArray3D):
    """Sensor array"""

    cdef inline DTYPEv1 c_all(self, unsigned iteration, unsigned int index) nogil:
        """Reaction"""
        return self.array[iteration, index, :]

    cdef inline DTYPEv1 c_reaction(self, unsigned iteration, unsigned int index) nogil:
        """Reaction"""
        return self.array[iteration, index, CONTACT_REACTION_X:CONTACT_REACTION_Z+1]

    cdef inline DTYPE c_reaction_x(self, unsigned iteration, unsigned int index) nogil:
        """Reaction x"""
        return self.array[iteration, index, CONTACT_REACTION_X]

    cdef inline DTYPE c_reaction_y(self, unsigned iteration, unsigned int index) nogil:
        """Reaction y"""
        return self.array[iteration, index, CONTACT_REACTION_Y]

    cdef inline DTYPE c_reaction_z(self, unsigned iteration, unsigned int index) nogil:
        """Reaction z"""
        return self.array[iteration, index, CONTACT_REACTION_Z]

    cdef inline DTYPEv1 c_friction(self, unsigned iteration, unsigned int index) nogil:
        """Friction"""
        return self.array[iteration, index, CONTACT_FRICTION_X:CONTACT_FRICTION_Z+1]

    cdef inline DTYPE c_friction_x(self, unsigned iteration, unsigned int index) nogil:
        """Friction x"""
        return self.array[iteration, index, CONTACT_FRICTION_X]

    cdef inline DTYPE c_friction_y(self, unsigned iteration, unsigned int index) nogil:
        """Friction y"""
        return self.array[iteration, index, CONTACT_FRICTION_Y]

    cdef inline DTYPE c_friction_z(self, unsigned iteration, unsigned int index) nogil:
        """Friction z"""
        return self.array[iteration, index, CONTACT_FRICTION_Z]

    cdef inline DTYPEv1 c_total(self, unsigned iteration, unsigned int index) nogil:
        """Total"""
        return self.array[iteration, index, CONTACT_TOTAL_X:CONTACT_TOTAL_Z+1]

    cdef inline DTYPE c_total_x(self, unsigned iteration, unsigned int index) nogil:
        """Total x"""
        return self.array[iteration, index, CONTACT_TOTAL_X]

    cdef inline DTYPE c_total_y(self, unsigned iteration, unsigned int index) nogil:
        """Total y"""
        return self.array[iteration, index, CONTACT_TOTAL_Y]

    cdef inline DTYPE c_total_z(self, unsigned iteration, unsigned int index) nogil:
        """Total z"""
        return self.array[iteration, index, CONTACT_TOTAL_Z]


cdef class JointSensorArrayCy(DoubleArray3D):
    """Joint sensor array"""

    cdef inline DTYPE position_cy(self, unsigned int iteration, unsigned int joint_i) nogil:
        """Joint position"""
        return self.array[iteration, joint_i, JOINT_POSITION]

    cdef inline DTYPEv1 positions_cy(self, unsigned int iteration) nogil:
        """Joints positions"""
        return self.array[iteration, :, JOINT_POSITION]

    cdef inline DTYPEv2 positions_all_cy(self) nogil:
        """Joints positions"""
        return self.array[:, :, JOINT_POSITION]

    cdef inline DTYPE velocity_cy(self, unsigned int iteration, unsigned int joint_i) nogil:
        """Joint velocity"""
        return self.array[iteration, joint_i, JOINT_VELOCITY]

    cdef inline DTYPEv1 velocities_cy(self, unsigned int iteration) nogil:
        """Joints velocities"""
        return self.array[iteration, :, JOINT_VELOCITY]

    cdef inline DTYPEv2 velocities_all_cy(self) nogil:
        """Joints velocities"""
        return self.array[:, :, JOINT_VELOCITY]

    cdef inline DTYPE motor_torque_cy(self, unsigned int iteration, unsigned int joint_i) nogil:
        """Joint velocity"""
        return self.array[iteration, joint_i, JOINT_TORQUE]

    cdef inline DTYPEv2 motor_torques_cy(self) nogil:
        """Joint velocity"""
        return self.array[:, :, JOINT_TORQUE]

    cdef inline DTYPEv1 force_cy(self, unsigned int iteration, unsigned int joint_i) nogil:
        """Joint force"""
        return self.array[iteration, joint_i, JOINT_FORCE_X:JOINT_FORCE_Z+1]

    cdef inline DTYPEv3 forces_all_cy(self) nogil:
        """Joints forces"""
        return self.array[:, :, JOINT_FORCE_X:JOINT_FORCE_Z+1]

    cdef inline DTYPEv1 torque_cy(self, unsigned int iteration, unsigned int joint_i) nogil:
        """Joint torque"""
        return self.array[iteration, joint_i, JOINT_TORQUE_X:JOINT_TORQUE_Z+1]

    cdef inline DTYPEv3 torques_all_cy(self) nogil:
        """Joints torques"""
        return self.array[:, :, JOINT_TORQUE_X:JOINT_TORQUE_Z+1]

    cdef inline DTYPE active_cy(self, unsigned int iteration, unsigned int joint_i) nogil:
        """Active torque"""
        return self.array[iteration, joint_i, JOINT_TORQUE_ACTIVE]

    cdef inline DTYPEv2 active_torques_cy(self) nogil:
        """Active torques"""
        return self.array[:, :, JOINT_TORQUE_ACTIVE]

    cdef inline DTYPE spring_cy(self, unsigned int iteration, unsigned int joint_i) nogil:
        """Passive spring torque"""
        return self.array[iteration, joint_i, JOINT_TORQUE_STIFFNESS]

    cdef inline DTYPEv2 spring_torques_cy(self) nogil:
        """Spring torques"""
        return self.array[:, :, JOINT_TORQUE_STIFFNESS]

    cdef inline DTYPE damping_cy(self, unsigned int iteration, unsigned int joint_i) nogil:
        """passive damping torque"""
        return self.array[iteration, joint_i, JOINT_TORQUE_DAMPING]

    cdef inline DTYPEv2 damping_torques_cy(self) nogil:
        """Damping torques"""
        return self.array[:, :, JOINT_TORQUE_DAMPING]

    cdef inline DTYPEv2 limit_force_all_cy(self) nogil:
        """joint limit forces"""
        return self.array[:, :, JOINT_LIMIT_FORCE]

    cdef inline DTYPE limit_force_cy(self, unsigned int iteration, unsigned int joint_i) nogil:
        """Joint force"""
        return self.array[iteration, joint_i, JOINT_LIMIT_FORCE]

    cdef inline DTYPEv1 limit_forces_cy(self, unsigned int joint_i) nogil:
        """Joint forces"""
        return self.array[:, joint_i, JOINT_LIMIT_FORCE]

    cdef inline DTYPEv2 limit_forces_all_cy(self) nogil:
        """joint limit forces"""
        return self.array[:, :, JOINT_LIMIT_FORCE]


cdef class LinkSensorArrayCy(DoubleArray3D):
    """Links array"""

    cdef inline DTYPEv1 com_position_cy(self, unsigned int iteration, unsigned int link_i) nogil:
        """CoM position of a link"""
        return self.array[iteration, link_i, LINK_COM_POSITION_X:LINK_COM_POSITION_Z+1]

    cdef inline DTYPEv1 com_orientation_cy(self, unsigned int iteration, unsigned int link_i) nogil:
        """CoM orientation of a link"""
        return self.array[iteration, link_i, LINK_COM_ORIENTATION_X:LINK_COM_ORIENTATION_W+1]

    cdef inline DTYPEv1 urdf_position_cy(self, unsigned int iteration, unsigned int link_i) nogil:
        """URDF position of a link"""
        return self.array[iteration, link_i, LINK_URDF_POSITION_X:LINK_URDF_POSITION_Z+1]

    cdef inline DTYPEv3 urdf_positions_cy(self) nogil:
        """URDF position of a link"""
        return self.array[:, :, LINK_URDF_POSITION_X:LINK_URDF_POSITION_Z+1]

    cdef inline DTYPEv1 urdf_orientation_cy(self, unsigned int iteration, unsigned int link_i) nogil:
        """URDF orientation of a link"""
        return self.array[iteration, link_i, LINK_URDF_ORIENTATION_X:LINK_URDF_ORIENTATION_W+1]

    cdef inline DTYPEv1 com_lin_velocity_cy(self, unsigned int iteration, unsigned int link_i) nogil:
        """CoM linear velocity of a link"""
        return self.array[iteration, link_i, LINK_COM_VELOCITY_LIN_X:LINK_COM_VELOCITY_LIN_Z+1]

    cdef inline DTYPEv3 com_lin_velocities_cy(self) nogil:
        """CoM linear velocities"""
        return self.array[:, :, LINK_COM_VELOCITY_LIN_X:LINK_COM_VELOCITY_LIN_Z+1]

    cdef inline DTYPEv1 com_ang_velocity_cy(self, unsigned int iteration, unsigned int link_i) nogil:
        """CoM angular velocity of a link"""
        return self.array[iteration, link_i, LINK_COM_VELOCITY_ANG_X:LINK_COM_VELOCITY_ANG_Z+1]


cdef class XfrcArrayCy(DoubleArray3D):
    """Xfrc array"""

    cdef inline DTYPE c_force_x(self, unsigned iteration, unsigned int index) nogil:
        """Force x"""
        return self.array[iteration, index, XFRC_FORCE_X]

    cdef inline DTYPE c_force_y(self, unsigned iteration, unsigned int index) nogil:
        """Force y"""
        return self.array[iteration, index, XFRC_FORCE_Y]

    cdef inline DTYPE c_force_z(self, unsigned iteration, unsigned int index) nogil:
        """Force z"""
        return self.array[iteration, index, XFRC_FORCE_Z]

    cdef inline DTYPE c_torque_x(self, unsigned iteration, unsigned int index) nogil:
        """Torque x"""
        return self.array[iteration, index, XFRC_TORQUE_X]

    cdef inline DTYPE c_torque_y(self, unsigned iteration, unsigned int index) nogil:
        """Torque y"""
        return self.array[iteration, index, XFRC_TORQUE_Y]

    cdef inline DTYPE c_torque_z(self, unsigned iteration, unsigned int index) nogil:
        """Torque z"""
        return self.array[iteration, index, XFRC_TORQUE_Z]


cdef class MusclesArrayCy(DoubleArray3D):
    """Muscles array"""

    cdef inline DTYPE c_muscle_excitation(self, unsigned iteration, unsigned int index) nogil:
        """Muscle activation"""
        return self.array[iteration, index, MUSCLE_EXCITATION]

    cdef inline DTYPE c_muscle_activation(self, unsigned iteration, unsigned int index) nogil:
        """Muscle activation"""
        return self.array[iteration, index, MUSCLE_ACTIVATION]

    cdef inline DTYPE c_muscle_tendon_unit_length(self, unsigned iteration, unsigned int index) nogil:
        """Muscle tendon unit length"""
        return self.array[iteration, index, MUSCLE_TENDON_UNIT_LENGTH]

    cdef inline DTYPE c_muscle_tendon_unit_velocity(self, unsigned iteration, unsigned int index) nogil:
        """Muscle tendon unit velocity"""
        return self.array[iteration, index, MUSCLE_TENDON_UNIT_VELOCITY]

    cdef inline DTYPE c_muscle_tendon_unit_force(self, unsigned iteration, unsigned int index) nogil:
        """Muscle tendon unit  force"""
        return self.array[iteration, index, MUSCLE_TENDON_UNIT_FORCE]

    cdef inline DTYPE c_muscle_fiber_length(self, unsigned iteration, unsigned int index) nogil:
        """Muscle fiber length"""
        return self.array[iteration, index, MUSCLE_FIBER_LENGTH]

    cdef inline DTYPE c_muscle_fiber_velocity(self, unsigned iteration, unsigned int index) nogil:
        """Muscle fiber velocity"""
        return self.array[iteration, index, MUSCLE_FIBER_VELOCITY]

    cdef inline DTYPE c_muscle_pennation_angle(self, unsigned iteration, unsigned int index) nogil:
        """Muscle pennation angle"""
        return self.array[iteration, index, MUSCLE_PENNATION_ANGLE]

    cdef inline DTYPE c_muscle_active_force(self, unsigned iteration, unsigned int index) nogil:
        """Muscle active force"""
        return self.array[iteration, index, MUSCLE_ACTIVE_FORCE]

    cdef inline DTYPE c_muscle_passive_force(self, unsigned iteration, unsigned int index) nogil:
        """Muscle passive force"""
        return self.array[iteration, index, MUSCLE_PASSIVE_FORCE]

    cdef inline DTYPE c_muscle_tendon_length(self, unsigned iteration, unsigned int index) nogil:
        """Muscle tendon length"""
        return self.array[iteration, index, MUSCLE_TENDON_LENGTH]

    cdef inline DTYPE c_muscle_tendon_force(self, unsigned iteration, unsigned int index) nogil:
        """Muscle tendon force"""
        return self.array[iteration, index, MUSCLE_TENDON_FORCE]

    cdef inline DTYPE c_spindle_Ia_feedback(self, unsigned iteration, unsigned int index) nogil:
        """Muscle spindle Type Ia feedback """
        return self.array[iteration, index, MUSCLE_IA_FEEDBACK]

    cdef inline DTYPE c_spindle_II_feedback(self, unsigned iteration, unsigned int index) nogil:
        """Muscle spindle Type II feedback """
        return self.array[iteration, index, MUSCLE_II_FEEDBACK]

    cdef inline DTYPE c_golgi_Ib_feedback(self, unsigned iteration, unsigned int index) nogil:
        """Muscle golgi tendon Type Ib feedback """
        return self.array[iteration, index, MUSCLE_IB_FEEDBACK]
