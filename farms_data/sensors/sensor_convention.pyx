"""Sensor index convention"""


cpdef enum sc:

    # Links
    link_size = LINK_SIZE
    link_com_position_x = LINK_COM_POSITION_X
    link_com_position_y = LINK_COM_POSITION_Y
    link_com_position_z = LINK_COM_POSITION_Z
    link_com_orientation_x = LINK_COM_ORIENTATION_X
    link_com_orientation_y = LINK_COM_ORIENTATION_Y
    link_com_orientation_z = LINK_COM_ORIENTATION_Z
    link_com_orientation_w = LINK_COM_ORIENTATION_W
    link_urdf_position_x = LINK_URDF_POSITION_X
    link_urdf_position_y = LINK_URDF_POSITION_Y
    link_urdf_position_z = LINK_URDF_POSITION_Z
    link_urdf_orientation_x = LINK_URDF_ORIENTATION_X
    link_urdf_orientation_y = LINK_URDF_ORIENTATION_Y
    link_urdf_orientation_z = LINK_URDF_ORIENTATION_Z
    link_urdf_orientation_w = LINK_URDF_ORIENTATION_W
    link_com_velocity_lin_x = LINK_COM_VELOCITY_LIN_X
    link_com_velocity_lin_y = LINK_COM_VELOCITY_LIN_Y
    link_com_velocity_lin_z = LINK_COM_VELOCITY_LIN_Z
    link_com_velocity_ang_x = LINK_COM_VELOCITY_ANG_X
    link_com_velocity_ang_y = LINK_COM_VELOCITY_ANG_Y
    link_com_velocity_ang_z = LINK_COM_VELOCITY_ANG_Z

    # Joints
    joint_size = JOINT_SIZE
    joint_position = JOINT_POSITION
    joint_velocity = JOINT_VELOCITY
    joint_force_x = JOINT_FORCE_X
    joint_force_y = JOINT_FORCE_Y
    joint_force_z = JOINT_FORCE_Z
    joint_torque_x = JOINT_TORQUE_X
    joint_torque_y = JOINT_TORQUE_Y
    joint_torque_z = JOINT_TORQUE_Z
    joint_torque = JOINT_TORQUE
    joint_torque_active = JOINT_TORQUE_ACTIVE
    joint_torque_stiffness = JOINT_TORQUE_STIFFNESS
    joint_torque_damping = JOINT_TORQUE_DAMPING

    # Contacts
    contact_size = CONTACT_SIZE
    contact_reaction_x = CONTACT_REACTION_X
    contact_reaction_y = CONTACT_REACTION_Y
    contact_reaction_z = CONTACT_REACTION_Z
    contact_friction_x = CONTACT_FRICTION_X
    contact_friction_y = CONTACT_FRICTION_Y
    contact_friction_z = CONTACT_FRICTION_Z
    contact_total_x = CONTACT_TOTAL_X
    contact_total_y = CONTACT_TOTAL_Y
    contact_total_z = CONTACT_TOTAL_Z
    contact_position_x = CONTACT_POSITION_X
    contact_position_y = CONTACT_POSITION_Y
    contact_position_z = CONTACT_POSITION_Z

    # Hydrodynamics
    hydrodynamics_size = HYDRODYNAMICS_SIZE
    hydrodynamics_force_x = HYDRODYNAMICS_FORCE_X
    hydrodynamics_force_y = HYDRODYNAMICS_FORCE_Y
    hydrodynamics_force_z = HYDRODYNAMICS_FORCE_Z
    hydrodynamics_torque_x = HYDRODYNAMICS_TORQUE_X
    hydrodynamics_torque_y = HYDRODYNAMICS_TORQUE_Y
    hydrodynamics_torque_z = HYDRODYNAMICS_TORQUE_Z
