"""Simulation options"""

from enum import IntEnum


class Simulator(IntEnum):
    """Simulator"""
    MUJOCO = 0
    PYBULLET = 1
