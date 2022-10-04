"""Simulation options"""

from typing import List
from enum import IntEnum

import numpy as np

from ..options import Options
from ..array.types import NDARRAY_V1
from ..units import SimulationUnitScaling
from .parse_args import config_parse_args


class Simulator(IntEnum):
    """Simulator"""
    MUJOCO = 0
    PYBULLET = 1


class SimulationOptions(Options):
    """Simulation options"""
    # pylint: disable=too-many-instance-attributes

    def __init__(self, **kwargs):
        super().__init__()

        # Units
        units = kwargs.pop('units', None)
        self.units: SimulationUnitScaling = SimulationUnitScaling(
            meters=units.pop('meters', 1),
            seconds=units.pop('seconds', 1),
            kilograms=units.pop('kilograms', 1),
        ) if isinstance(units, dict) else SimulationUnitScaling(
            meters=kwargs.pop('meters', 1),
            seconds=kwargs.pop('seconds', 1),
            kilograms=kwargs.pop('kilograms', 1),
        )

        # Simulation
        self.timestep: float = kwargs.pop('timestep', 1e-3)
        self.n_iterations: int = kwargs.pop('n_iterations', 1000)
        self.play: bool = kwargs.pop('play', True)
        self.rtl: float = kwargs.pop('rtl', 1.0)
        self.fast: bool = kwargs.pop('fast', False)
        self.headless: bool = kwargs.pop('headless', False)
        self.show_progress: bool = kwargs.pop('show_progress', True)

        # Camera
        self.zoom: float = kwargs.pop('zoom', 1)
        self.free_camera: bool = kwargs.pop('free_camera', False)
        self.top_camera: bool = kwargs.pop('top_camera', False)
        self.rotating_camera: bool = kwargs.pop('rotating_camera', False)

        # Video recording
        self.video: str = kwargs.pop('video', '')
        self.video_fps: bool = kwargs.pop('video_fps', False)
        self.video_name: str = kwargs.pop('video_name', 'video')
        self.video_yaw: float = kwargs.pop('video_yaw', 30)
        self.video_pitch: float = kwargs.pop('video_pitch', 45)
        self.video_distance: float = kwargs.pop('video_distance', 1)
        self.video_offset: List[float] = kwargs.pop('video_offset', [0, 0, 0])
        self.video_filter = kwargs.pop('video_filter', None)
        self.video_resolution: List[float] = kwargs.pop(
            'video_resolution',
            (1280, 720),
        )

        # Rendering
        self.visual_scale: float = kwargs.pop('visual_scale', 1.0)

        # Mujoco
        self.cone: str = kwargs.pop('cone', 'pyramidal')

        # Pybullet
        self.gravity: List[float] = kwargs.pop('gravity', [0, 0, -9.81])
        self.opengl2: bool = kwargs.pop('opengl2', False)
        self.lcp: str = kwargs.pop('lcp', 'dantzig')
        self.n_solver_iters: int = kwargs.pop('n_solver_iters', 50)
        self.cfm: float = kwargs.pop('cfm', 1e-10)
        self.erp: float = kwargs.pop('erp', 0)
        self.contact_erp: float = kwargs.pop('contact_erp', 0)
        self.friction_erp: float = kwargs.pop('friction_erp', 0)
        self.num_sub_steps: int = kwargs.pop('num_sub_steps', 0)
        self.max_num_cmd_per_1ms: int = kwargs.pop('max_num_cmd_per_1ms', int(1e8))
        self.residual_threshold: float = kwargs.pop('residual_threshold', 1e-6)
        self.report_solver_analytics: int = kwargs.pop('report_solver_analytics', 0)
        assert not kwargs, kwargs

    @classmethod
    def with_clargs(cls, **kwargs):
        """Create simulation options and consider command-line arguments"""
        clargs = config_parse_args()
        timestep = kwargs.pop('timestep', clargs.timestep)
        assert timestep > 0, f'Timestep={timestep} should be > 0'
        return cls(
            # Simulation
            timestep=timestep,
            n_iterations=kwargs.pop(
                'n_iterations',
                round(clargs.duration/timestep)+1,
            ),
            play=kwargs.pop('play', not clargs.pause),
            rtl=kwargs.pop('rtl', clargs.rtl),
            fast=kwargs.pop('fast', clargs.fast),
            headless=kwargs.pop('headless', clargs.headless),
            show_progress=kwargs.pop('show_progress', clargs.show_progress),

            # Units
            meters=kwargs.pop('meters', clargs.meters),
            seconds=kwargs.pop('seconds', clargs.seconds),
            kilograms=kwargs.pop('kilograms', clargs.kilograms),

            # Camera
            zoom=kwargs.pop('zoom', clargs.zoom),
            free_camera=kwargs.pop('free_camera', clargs.free_camera),
            top_camera=kwargs.pop('top_camera', clargs.top_camera),
            rotating_camera=kwargs.pop('rotating_camera', clargs.rotating_camera),

            # Video recording
            video=kwargs.pop('video', clargs.video),
            video_fps=kwargs.pop('video_fps', clargs.video_fps),
            video_yaw=kwargs.pop('video_yaw', clargs.video_yaw),
            video_pitch=kwargs.pop('video_pitch', clargs.video_pitch),
            video_distance=kwargs.pop('video_distance', clargs.video_distance),
            video_offset=kwargs.pop('video_offset', clargs.video_offset),
            video_resolution=kwargs.pop('video_resolution', clargs.video_resolution),
            video_filter=kwargs.pop('video_filter', clargs.video_motion_filter),

            # Pybullet
            gravity=kwargs.pop('gravity', clargs.gravity),
            opengl2=kwargs.pop('opengl2', clargs.opengl2),
            lcp=kwargs.pop('lcp', clargs.lcp),
            n_solver_iters=kwargs.pop('n_solver_iters', clargs.n_solver_iters),
            cfm=kwargs.pop('cfm', clargs.cfm),
            erp=kwargs.pop('erp', clargs.erp),
            contact_erp=kwargs.pop('contact_erp', clargs.contact_erp),
            friction_erp=kwargs.pop('friction_erp', clargs.friction_erp),
            num_sub_steps=kwargs.pop('num_sub_steps', clargs.num_sub_steps),
            max_num_cmd_per_1ms=kwargs.pop(
                'max_num_cmd_per_1ms',
                clargs.max_num_cmd_per_1ms,
            ),
            residual_threshold=kwargs.pop(
                'residual_threshold',
                clargs.residual_threshold,
            ),
            **kwargs,
        )

    def duration(self) -> float:
        """Simulation duraiton"""
        return self.timestep*(self.n_iterations-1)

    def times(self) -> NDARRAY_V1:
        """Simulation times"""
        return np.arange(
            start=0,
            stop=self.timestep*(self.n_iterations-0.5),
            step=self.timestep,
        )
