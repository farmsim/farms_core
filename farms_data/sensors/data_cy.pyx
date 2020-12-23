"""Animat data"""

import numpy as np
cimport numpy as np


cdef class SensorsDataCy:
    """SensorsData"""

    def __init__(
            self,
            GpsArrayCy gps=None,
            JointSensorArrayCy joints=None,
            ContactsArrayCy contacts=None,
            HydrodynamicsArrayCy hydrodynamics=None
    ):
        super(SensorsDataCy, self).__init__()
        self.gps = gps
        self.joints = joints
        self.contacts = contacts
        self.hydrodynamics = hydrodynamics
