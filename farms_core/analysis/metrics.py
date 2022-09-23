"""Metrics"""

import numpy as np
from ..sensors.sensor_convention import sc


def get_limb_swings(contacts_array, contact_indices, threshold=1e-16):
    """Get limb swing (True if in swing, False otherwise)"""
    swings = np.linalg.norm(
        contacts_array[
            :,
            contact_indices,
            sc.contact_total_x:sc.contact_total_z+1,
        ],
        axis=-1,
    ) < threshold
    return swings


def analyse_gait(animat_data, animat_options, contact_indices, joint_indices):
    """Analyse gait"""
    contacts_array = np.array(animat_data.sensors.contacts.array)
    swing = get_limb_swings(contacts_array, contact_indices)

    joints_array = np.array(animat_data.sensors.joints.array)
    swing = np.logical_or(
        swing,
        joints_array[:, joint_indices, sc.joint_velocity] > 0,
    )
    gait = {}
    n_legs = animat_options.morphology.n_legs
    not_all_ground_or_air_indices = np.where(np.logical_or(
        np.sum(swing, axis=1) != 0,
        np.sum(swing, axis=1) != animat_options.morphology.n_legs,
    ))[0]
    contacts_gait = swing[not_all_ground_or_air_indices, :]
    gait['Stand'] = np.mean(np.sum(swing, axis=1) == 0)
    if n_legs == 4:
        gait['Trotting'] = np.mean(
            np.logical_xor(
                np.logical_and(contacts_gait[:, 0], contacts_gait[:, 3]),
                np.logical_and(contacts_gait[:, 1], contacts_gait[:, 2]),
            ),
        )
        # gait['Sequence'] = np.mean(np.logical_or(
        #     np.sum(contacts_gait, axis=1) == 1,
        #     np.sum(contacts_gait, axis=1) == 0,
        # ))
        gait['Sequence'] = np.mean(np.sum(contacts_gait, axis=1) == 1)
        gait['Bound'] = np.mean(
            np.logical_xor(
                np.logical_and(contacts_gait[:, 0], contacts_gait[:, 1]),
                np.logical_and(contacts_gait[:, 2], contacts_gait[:, 3]),
            ),
        )
    gait['DF'] = np.mean(swing == 0)
    if n_legs == 4:
        gait['LF'] = np.mean(swing[:, 0] == 0)
        gait['RF'] = np.mean(swing[:, 1] == 0)
        gait['LH'] = np.mean(swing[:, 2] == 0)
        gait['RH'] = np.mean(swing[:, 3] == 0)

    return gait
