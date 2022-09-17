"""Metrics"""

import numpy as np
from ..sensors.sensor_convention import sc


def get_limb_swings(contacts_array, feet_indices, threshold=1e-16):
    """Get limb swing"""
    swings = np.linalg.norm(
        contacts_array[:, feet_indices, sc.contact_total_x:sc.contact_total_z+1],
        axis=-1,
    ) < threshold  # True if in swing, False otherwise
    return swings
