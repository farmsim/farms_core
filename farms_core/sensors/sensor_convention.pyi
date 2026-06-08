"""Sensor index convention"""

from enum import IntEnum

class sc(IntEnum):
    # Links
    link_size: int
    link_com_position_x: int
    link_com_position_y: int
    link_com_position_z: int
    link_com_orientation_x: int
    link_com_orientation_y: int
    link_com_orientation_z: int
    link_com_orientation_w: int
    link_urdf_position_x: int
    link_urdf_position_y: int
    link_urdf_position_z: int
    link_urdf_orientation_x: int
    link_urdf_orientation_y: int
    link_urdf_orientation_z: int
    link_urdf_orientation_w: int
    link_com_velocity_lin_x: int
    link_com_velocity_lin_y: int
    link_com_velocity_lin_z: int
    link_com_velocity_ang_x: int
    link_com_velocity_ang_y: int
    link_com_velocity_ang_z: int

    # Joints
    joint_size: int
    joint_position: int
    joint_velocity: int
    joint_torque: int
    joint_force_x: int
    joint_force_y: int
    joint_force_z: int
    joint_torque_x: int
    joint_torque_y: int
    joint_torque_z: int
    joint_cmd_position: int
    joint_cmd_velocity: int
    joint_cmd_torque: int
    joint_torque_active: int
    joint_torque_stiffness: int
    joint_torque_damping: int
    joint_torque_friction: int
    joint_limit_force: int

    # Contacts
    contact_size: int
    contact_reaction_x: int
    contact_reaction_y: int
    contact_reaction_z: int
    contact_friction_x: int
    contact_friction_y: int
    contact_friction_z: int
    contact_total_x: int
    contact_total_y: int
    contact_total_z: int
    contact_position_x: int
    contact_position_y: int
    contact_position_z: int

    # Xfrc
    xfrc_size: int
    xfrc_force_x: int
    xfrc_force_y: int
    xfrc_force_z: int
    xfrc_torque_x: int
    xfrc_torque_y: int
    xfrc_torque_z: int

    # Muscles
    muscle_size: int
    muscle_excitation: int
    muscle_activation: int
    muscle_tendon_unit_length: int
    muscle_tendon_unit_velocity: int
    muscle_tendon_unit_force: int
    muscle_fiber_length: int
    muscle_fiber_velocity: int
    muscle_pennation_angle: int
    muscle_force_length: int
    muscle_force_velocity: int
    muscle_active_force: int
    muscle_passive_force: int
    muscle_tendon_length: int
    muscle_tendon_force: int
    muscle_Ia_feedback: int
    muscle_II_feedback: int
    muscle_Ib_feedback: int

    # Adhesions
    adhesion_size: int
    adhesion_force: int

    # Visuals
    visual_size: int
    visual_color_r: int
    visual_color_g: int
    visual_color_b: int
    visual_color_a: int
    visual_emission_r: int
    visual_emission_g: int
    visual_emission_b: int
    visual_emission_i: int
