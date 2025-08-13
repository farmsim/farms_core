"""Simulation options"""

from enum import IntEnum

import numpy as np

from ..options import Options
from ..array.types import NDARRAY_V1
from ..units import SimulationUnitScaling
from ..doc import ClassDoc, ChildDoc
from .parse_args import config_parse_args


MSG_MUJOCO_OPTION = (
    " This options is for the MuJoCo physics engine, refer to"
    " [MuJoCo's documentation]"
    "(https://mujoco.readthedocs.io/en/stable/XMLreference.html)"
    " for additional information."
)
MSG_PYBULLET_OPTION = (
    " This options is for the Bullet physics engine, refer to"
    " [Pybullet's documentation]"
    "(https://docs.google.com/document/d/"
    "10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA)"
    " for additional information."
)


class Simulator(IntEnum):
    """Simulator"""
    MUJOCO = 0
    PYBULLET = 1


class SimulationOptions(Options):
    """Simulation options"""
    # pylint: disable=too-many-instance-attributes

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="simulation",
            description="Describes the simulation options.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="units",
                    class_type=SimulationUnitScaling,
                    description=(
                        "The simulation units used in the physics engine"
                        " (All parameters defined in the config files are in"
                        " SI units)."
                    ),
                ),
                ChildDoc(
                    name="timestep",
                    class_type=float,
                    description="The simulation timestep (Must be positive).",
                ),
                ChildDoc(
                    name="n_iterations",
                    class_type=int,
                    description=(
                        "The number of simulations iterations to run"
                        " (Must be positive)"
                    ),
                ),
                ChildDoc(
                    name="buffer_size",
                    class_type=int,
                    description=(
                        "The number of simulations itertions buffered,"
                        " after which the data will be overwritten."
                    ),
                ),
                ChildDoc(
                    name="play",
                    class_type=bool,
                    description=(
                        "Whether to play the simulation, or keep it paused when"
                        " started. Only relevent when running the simulation"
                        " in interactive mode."
                    ),
                ),
                ChildDoc(
                    name="rtl",
                    class_type=float,
                    description=(
                        "Real-time limiter to limit the simulation speed."
                        " Only relevent when running the simulation"
                        " in interactive mode."
                    ),
                ),
                ChildDoc(
                    name="fast",
                    class_type=bool,
                    description=(
                        "Whether to run the simulation as fast as possible,"
                        " bypasses rtl."
                    ),
                ),
                ChildDoc(
                    name="headless",
                    class_type=bool,
                    description=(
                        "Whether to run the simulation headless, with no."
                        " external interaction."
                    ),
                ),
                ChildDoc(
                    name="show_progress",
                    class_type=bool,
                    description="Whether to display a progress bar.",
                ),
                # Camera
                ChildDoc(
                    name="zoom",
                    class_type=float,
                    description="Camera zoom.",
                ),
                ChildDoc(
                    name="free_camera",
                    class_type=bool,
                    description=(
                        "Whether the camera should be free moving instead of"
                        " following the animat."
                    ),
                ),
                ChildDoc(
                    name="top_camera",
                    class_type=bool,
                    description=(
                        "Whether the camera should look at the animat from"
                        " above."
                    ),
                ),
                ChildDoc(
                    name="rotating_camera",
                    class_type=bool,
                    description=(
                        "Whether the camera should turn around the model."
                    ),
                ),
                # Video recording
                ChildDoc(
                    name="video",
                    class_type=str,
                    description=(
                        "Path to where the video should be saved. Empty string"
                        " to disable recording."
                    ),
                ),
                ChildDoc(
                    name="video_fps",
                    class_type=str,
                    description=(
                        "Path to where the video should be saved. Empty string"
                        " to disable recording."
                    ),
                ),
                ChildDoc(
                    name="video_speed",
                    class_type=float,
                    description=(
                        "Speed factor at which the video should be played."
                    ),
                ),
                ChildDoc(
                    name="video_name",
                    class_type=str,
                    description="Video name.",
                ),
                ChildDoc(
                    name="video_yaw",
                    class_type=float,
                    description="Video yaw angle.",
                ),
                ChildDoc(
                    name="video_pitch",
                    class_type=float,
                    description="Video yaw pitch.",
                ),
                ChildDoc(
                    name="video_distance",
                    class_type=float,
                    description="Video distance from animat.",
                ),
                ChildDoc(
                    name="video_offset",
                    class_type=float,
                    description="Video position offset with respect to animat.",
                ),
                ChildDoc(
                    name="video_filter",
                    class_type=float,
                    description="Video motion filter.",
                ),
                ChildDoc(
                    name="video_resolution",
                    class_type="list[int]",
                    description="Video resolution (e.g. [1280, 720]).",
                ),
                # Physics engine
                ChildDoc(
                    name="gravity",
                    class_type="list[float]",
                    description="Gravity vector (e.g. [0, 0, -9.81]).",
                ),
                ChildDoc(
                    name="num_sub_steps",
                    class_type=int,
                    description="Number of physics substeps.",
                ),
                ChildDoc(
                    name="cb_sub_steps",
                    class_type=int,
                    description="Number of callback substeps.",
                ),
                ChildDoc(
                    name="n_solver_iters",
                    class_type=int,
                    description="Number of maximum solver iterations per step.",
                ),
                ChildDoc(
                    name="residual_threshold",
                    class_type=float,
                    description="Residual threshold (e.g. 1e-6).",
                ),
                ChildDoc(
                    name="visual_scale",
                    class_type=float,
                    description="Visual scale.",
                ),
                ),
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()

        strict = kwargs.pop('strict', True)

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
        self.buffer_size: int = kwargs.pop('buffer_size', self.n_iterations)
        if self.buffer_size == 0:
            self.buffer_size = self.n_iterations
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
        self.video_speed: float = kwargs.pop('video_speed', 1.0)
        self.video_name: str = kwargs.pop('video_name', 'video')
        self.video_yaw: float = kwargs.pop('video_yaw', 30)
        self.video_pitch: float = kwargs.pop('video_pitch', 45)
        self.video_distance: float = kwargs.pop('video_distance', 1)
        self.video_offset: list[float] = kwargs.pop('video_offset', [0, 0, 0])
        self.video_filter = kwargs.pop('video_filter', None)
        self.video_resolution: list[float] = kwargs.pop(
            'video_resolution',
            (1280, 720),
        )

        # Physics engine
        self.gravity: list[float] = kwargs.pop('gravity', [0, 0, -9.81])
        self.num_sub_steps: int = kwargs.pop('num_sub_steps', 1)  # Physics engine substeps
        self.cb_sub_steps: int = kwargs.pop('cb_sub_steps', 0)  # FARMS substep (Callbacks)
        self.n_solver_iters: int = kwargs.pop('n_solver_iters', 50)
        self.residual_threshold: float = kwargs.pop('residual_threshold', 1e-6)
        self.visual_scale: float = kwargs.pop('visual_scale', 1.0)
        self.mujoco: MuJoCoSimulationOptions = kwargs.pop('mujoco', {})
        if not isinstance(self.mujoco, MuJoCoSimulationOptions):
            self.mujoco = MuJoCoSimulationOptions(
                **self.mujoco,
                strict=strict,
            )
        self.pybullet: PybulletSimulationOptions = kwargs.pop('pybullet', {})
        if not isinstance(self.pybullet, PybulletSimulationOptions):
            self.pybullet = PybulletSimulationOptions(
                **self.pybullet,
                strict=strict,
            )

        if strict:
            assert not kwargs, kwargs

    @classmethod
    def with_clargs(cls, **kwargs):
        """Create simulation options and consider command-line arguments"""
        clargs = config_parse_args()
        timestep = kwargs.pop('timestep', clargs.timestep)
        assert timestep > 0, f'Timestep={timestep} should be > 0'
        n_iteration_default = round(clargs.duration/timestep)+1
        return cls(
            # Simulation
            timestep=timestep,
            n_iterations=kwargs.pop('n_iterations', n_iteration_default),
            buffer_size=kwargs.pop('buffer_size', clargs.buffer_size),
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
            video_speed=kwargs.pop('video_speed', clargs.video_speed),
            video_yaw=kwargs.pop('video_yaw', clargs.video_yaw),
            video_pitch=kwargs.pop('video_pitch', clargs.video_pitch),
            video_distance=kwargs.pop('video_distance', clargs.video_distance),
            video_offset=kwargs.pop('video_offset', clargs.video_offset),
            video_resolution=kwargs.pop('video_resolution', clargs.video_resolution),
            video_filter=kwargs.pop('video_filter', clargs.video_motion_filter),

            # Physics engine
            gravity=kwargs.pop('gravity', clargs.gravity),
            num_sub_steps=kwargs.pop('num_sub_steps', clargs.num_sub_steps),
            cb_sub_steps=kwargs.pop('cb_sub_steps', clargs.cb_sub_steps),
            n_solver_iters=kwargs.pop('n_solver_iters', clargs.n_solver_iters),
            residual_threshold=kwargs.pop('residual_threshold', clargs.residual_threshold),
            mujoco=kwargs.pop('mujoco', MuJoCoSimulationOptions.with_clargs(**kwargs)),
            bullet=kwargs.pop('bullet', PybulletSimulationOptions.with_clargs(**kwargs)),


            # Additional kwargs
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
