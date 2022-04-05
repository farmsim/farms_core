"""Animat data"""

from ..sensors.data_cy cimport SensorsDataCy


cdef class AnimatDataCy:
    """Animat data"""
    cdef public double timestep
    cdef public SensorsDataCy sensors
