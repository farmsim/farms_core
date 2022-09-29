"""Animat data"""

import numpy as np
cimport numpy as np


cdef class SensorsDataCy:
    """Sensors data"""

    def __init__(
            self,
            LinkSensorArrayCy links=None,
            JointSensorArrayCy joints=None,
            ContactsArrayCy contacts=None,
            XfrcArrayCy xfrc=None,
            MusclesArrayCy muscles=None,
    ):
        super(SensorsDataCy, self).__init__()
        self.links = links
        self.joints = joints
        self.contacts = contacts
        self.xfrc = xfrc
        self.muscles = muscles
