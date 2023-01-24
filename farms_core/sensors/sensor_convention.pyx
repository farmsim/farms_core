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
    joint_torque = JOINT_TORQUE
    joint_force_x = JOINT_FORCE_X
    joint_force_y = JOINT_FORCE_Y
    joint_force_z = JOINT_FORCE_Z
    joint_torque_x = JOINT_TORQUE_X
    joint_torque_y = JOINT_TORQUE_Y
    joint_torque_z = JOINT_TORQUE_Z
    joint_cmd_position = JOINT_CMD_POSITION
    joint_cmd_velocity = JOINT_CMD_VELOCITY
    joint_cmd_torque = JOINT_CMD_TORQUE
    joint_torque_active = JOINT_TORQUE_ACTIVE
    joint_torque_stiffness = JOINT_TORQUE_STIFFNESS
    joint_torque_damping = JOINT_TORQUE_DAMPING
    joint_torque_friction = JOINT_TORQUE_FRICTION
    joint_limit_force = JOINT_LIMIT_FORCE

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

    # Xfrc
    xfrc_size = XFRC_SIZE
    xfrc_force_x = XFRC_FORCE_X
    xfrc_force_y = XFRC_FORCE_Y
    xfrc_force_z = XFRC_FORCE_Z
    xfrc_torque_x = XFRC_TORQUE_X
    xfrc_torque_y = XFRC_TORQUE_Y
    xfrc_torque_z = XFRC_TORQUE_Z

    # Muscles
    muscle_size = MUSCLE_SIZE
    muscle_excitation = MUSCLE_EXCITATION
    muscle_activation = MUSCLE_ACTIVATION
    muscle_tendon_unit_length = MUSCLE_TENDON_UNIT_LENGTH
    muscle_tendon_unit_velocity = MUSCLE_TENDON_UNIT_VELOCITY
    muscle_tendon_unit_force = MUSCLE_TENDON_UNIT_FORCE
    muscle_fiber_length = MUSCLE_FIBER_LENGTH
    muscle_fiber_velocity = MUSCLE_FIBER_VELOCITY
    muscle_pennation_angle = MUSCLE_PENNATION_ANGLE
    muscle_active_force = MUSCLE_ACTIVE_FORCE
    muscle_passive_force = MUSCLE_PASSIVE_FORCE
    muscle_tendon_length = MUSCLE_TENDON_LENGTH
    muscle_tendon_force = MUSCLE_TENDON_FORCE
    muscle_Ia_feedback = MUSCLE_IA_FEEDBACK
    muscle_II_feedback = MUSCLE_II_FEEDBACK
    muscle_Ib_feedback = MUSCLE_IB_FEEDBACK
