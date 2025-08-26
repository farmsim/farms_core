"""Simulation options"""

from enum import IntEnum

import numpy as np

from ..options import Options
from ..array.types import NDARRAY_V1
from ..units import SimulationUnitScaling
from ..doc import ClassDoc, ChildDoc
from .parse_args import config_parse_args


MSG_MUJOCO_OPTIONS = (
    " These options are for the MuJoCo physics engine, refer to"
    " [MuJoCo's documentation]"
    "(https://mujoco.readthedocs.io/en/stable/XMLreference.html)"
    " for additional information."
)
MSG_PYBULLET_OPTIONS = (
    " These options are for the Bullet physics engine, refer to"
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
                    name="runtime",
                    class_type=RuntimeSimulationOptions,
                    description="Runtime simulation options",
                ),
                ChildDoc(
                    name="camera",
                    class_type=CameraInterfaceOptions,
                    description="Camera options.",
                ),
                ChildDoc(
                    name="video",
                    class_type=VideoRecordingOptions,
                    description="Video recording options.",
                ),
                ChildDoc(
                    "physics",
                    class_type=PhysicsSimulationOptions,
                    description="Common physics simulation options."
                ),
                ChildDoc(
                    name="mujoco",
                    class_type=MuJoCoSimulationOptions,
                    description="MuJoCo options.",
                ),
                ChildDoc(
                    name="pybullet",
                    class_type=PybulletSimulationOptions,
                    description="Pybullet options.",
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

        # Runtime
        self.runtime: RuntimeSimulationOptions = kwargs.pop('runtime', {})
        if not isinstance(self.runtime, RuntimeSimulationOptions):
            self.runtime = RuntimeSimulationOptions(
                **self.runtime,
                strict=strict,
            )

        # Camera
        self.camera: CameraInterfaceOptions = kwargs.pop('camera', {})
        if not isinstance(self.camera, CameraInterfaceOptions):
            self.camera = CameraInterfaceOptions(
                **self.camera,
                strict=strict,
            )

        # Video recording
        self.video: VideoRecordingOptions = kwargs.pop('video', {})
        if not isinstance(self.video, VideoRecordingOptions):
            self.video = VideoRecordingOptions(
                **self.video,
                strict=strict,
            )

        # Physics engine
        self.physics: PhysicsSimulationOptions = kwargs.pop('physics', {})
        if not isinstance(self.physics, PhysicsSimulationOptions):
            self.physics = PhysicsSimulationOptions(
                **self.physics,
                strict=strict,
            )
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
        clargs = kwargs.pop('clargs', None)
        if clargs is None:
            clargs = config_parse_args()
        timestep = kwargs.get('timestep', clargs.timestep)
        assert timestep > 0, f'Timestep={timestep} should be > 0'
        if 'n_iterations' not in kwargs:
            kwargs['n_iterations'] = round(clargs.duration/timestep)+1
        return cls(
            # Runtime
            runtime=kwargs.pop('runtime', RuntimeSimulationOptions.with_clargs(
                clargs=clargs,
                **kwargs,
            )),

            # Units
            meters=kwargs.pop('meters', clargs.meters),
            seconds=kwargs.pop('seconds', clargs.seconds),
            kilograms=kwargs.pop('kilograms', clargs.kilograms),

            # Camera
            camera=kwargs.pop('camera', CameraInterfaceOptions.with_clargs(
                clargs=clargs,
                **kwargs,
            )),

            # Video recording
            video=kwargs.pop('video', VideoRecordingOptions.with_clargs(
                clargs=clargs,
                **kwargs,
            )),

            # Physics engine
            physics=kwargs.pop('physics', PhysicsSimulationOptions.with_clargs(
                clargs=clargs,
                **kwargs,
            )),
            mujoco=kwargs.pop('mujoco', MuJoCoSimulationOptions.with_clargs(
                clargs=clargs,
                **kwargs,
            )),
            bullet=kwargs.pop('bullet', PybulletSimulationOptions.with_clargs(
                clargs=clargs,
                **kwargs,
            )),

            # Additional kwargs
            **kwargs,
        )

    def duration(self) -> float:
        """Simulation duraiton"""
        return self.physics.timestep*(self.runtime.n_iterations-1)

    def times(self) -> NDARRAY_V1:
        """Simulation times"""
        return np.arange(
            start=0,
            stop=self.physics.timestep*(self.runtime.n_iterations-0.5),
            step=self.physics.timestep,
        )


class RuntimeSimulationOptions(Options):
    """Runtime simulation options"""
    # pylint: disable=too-many-instance-attributes

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="Runtime simulation options",
            description="Describes the runtime simulation options.",
            class_type=cls,
            children=[
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
                        " bypasses real-time limiter (rtl)."
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
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        self.n_iterations: int = kwargs.pop('n_iterations', 1000)
        self.buffer_size: int = kwargs.pop('buffer_size', self.n_iterations)
        if self.buffer_size == 0:
            self.buffer_size = self.n_iterations
        self.play: bool = kwargs.pop('play', True)
        self.rtl: float = kwargs.pop('rtl', 1.0)
        self.fast: bool = kwargs.pop('fast', False)
        self.headless: bool = kwargs.pop('headless', False)
        self.show_progress: bool = kwargs.pop('show_progress', True)
        if kwargs.pop('strict', True):
            assert not kwargs, kwargs

    @classmethod
    def with_clargs(cls, **kwargs):
        """Create simulation options and consider command-line arguments"""
        clargs = kwargs.pop('clargs', None)
        if clargs is None:
            clargs = config_parse_args()
        return cls(
            n_iterations=kwargs.pop('n_iterations', 0),
            buffer_size=kwargs.pop('buffer_size', clargs.buffer_size),
            play=kwargs.pop('play', not clargs.pause),
            rtl=kwargs.pop('rtl', clargs.rtl),
            fast=kwargs.pop('fast', clargs.fast),
            headless=kwargs.pop('headless', clargs.headless),
            show_progress=kwargs.pop('show_progress', clargs.show_progress),
            **kwargs,
        )


class CameraInterfaceOptions(Options):
    """Camera recording options"""
    # pylint: disable=too-many-instance-attributes

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="Camera options",
            description="Describes the camera options.",
            class_type=cls,
            children=[
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
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        self.zoom: float = kwargs.pop('zoom', 1)
        self.free_camera: bool = kwargs.pop('free_camera', False)
        self.top_camera: bool = kwargs.pop('top_camera', False)
        self.rotating_camera: bool = kwargs.pop('rotating_camera', False)
        if kwargs.pop('strict', True):
            assert not kwargs, kwargs

    @classmethod
    def with_clargs(cls, **kwargs):
        """Create simulation options and consider command-line arguments"""
        clargs = kwargs.pop('clargs', None)
        if clargs is None:
            clargs = config_parse_args()
        return cls(
            zoom=kwargs.pop('zoom', clargs.zoom),
            free_camera=kwargs.pop('free_camera', clargs.free_camera),
            top_camera=kwargs.pop('top_camera', clargs.top_camera),
            rotating_camera=kwargs.pop('rotating_camera', clargs.rotating_camera),
            **kwargs,
        )


class VideoRecordingOptions(Options):
    """Video recording options"""
    # pylint: disable=too-many-instance-attributes

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="Video recording options",
            description="Describes the video recording options.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="video",
                    class_type=str,
                    description=(
                        "Path to where the video should be saved. Empty string"
                        " to disable recording."
                    ),
                ),
                ChildDoc(
                    name="fps",
                    class_type=float,
                    description="Video framerate",
                ),
                ChildDoc(
                    name="speed",
                    class_type=float,
                    description=(
                        "Speed factor at which the video should be played."
                    ),
                ),
                ChildDoc(
                    name="name",
                    class_type=str,
                    description="Video name.",
                ),
                ChildDoc(
                    name="yaw",
                    class_type=float,
                    description="Video yaw angle.",
                ),
                ChildDoc(
                    name="pitch",
                    class_type=float,
                    description="Video yaw pitch.",
                ),
                ChildDoc(
                    name="distance",
                    class_type=float,
                    description="Video distance from animat.",
                ),
                ChildDoc(
                    name="offset",
                    class_type=float,
                    description="Video position offset with respect to animat.",
                ),
                ChildDoc(
                    name="motion_filter",
                    class_type=float,
                    description="Video motion filter.",
                ),
                ChildDoc(
                    name="resolution",
                    class_type="list[int]",
                    description="Video resolution (e.g. [1280, 720]).",
                ),
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        self.path: str = kwargs.pop('path', '')
        self.fps: float | bool = kwargs.pop('fps', False)
        self.speed: float = kwargs.pop('speed', 1.0)
        self.name: str = kwargs.pop('name', 'video')
        self.yaw: float = kwargs.pop('yaw', 30)
        self.pitch: float = kwargs.pop('pitch', 45)
        self.distance: float = kwargs.pop('distance', 1)
        self.offset: list[float] = kwargs.pop('offset', [0, 0, 0])
        self.motion_filter = kwargs.pop('motion_filter', None)
        self.resolution: list[float] = kwargs.pop(
            'resolution',
            (1280, 720),
        )
        if kwargs.pop('strict', True):
            assert not kwargs, kwargs

    @classmethod
    def with_clargs(cls, **kwargs):
        """Create simulation options and consider command-line arguments"""
        clargs = kwargs.pop('clargs', None)
        if clargs is None:
            clargs = config_parse_args()
        return cls(
            path=kwargs.pop('path', clargs.video),
            fps=kwargs.pop('fps', clargs.video_fps),
            speed=kwargs.pop('speed', clargs.video_speed),
            yaw=kwargs.pop('yaw', clargs.video_yaw),
            pitch=kwargs.pop('pitch', clargs.video_pitch),
            distance=kwargs.pop('distance', clargs.video_distance),
            offset=kwargs.pop('offset', clargs.video_offset),
            motion_filter=kwargs.pop('motion_filter', clargs.video_motion_filter),
            resolution=kwargs.pop('resolution', clargs.video_resolution),
            **kwargs,
        )


class PhysicsSimulationOptions(Options):
    """Physics simulation options"""
    # pylint: disable=too-many-instance-attributes

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="Physics simulation options",
            description="Describes the physics simulation options.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="timestep",
                    class_type=float,
                    description="The simulation timestep (Must be positive).",
                ),
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
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        self.timestep: float = kwargs.pop('timestep', 1e-3)
        self.gravity: list[float] = kwargs.pop('gravity', [0, 0, -9.81])
        self.num_sub_steps: int = kwargs.pop('num_sub_steps', 1)  # Physics engine substeps
        self.cb_sub_steps: int = kwargs.pop('cb_sub_steps', 0)  # FARMS substep (Callbacks)
        self.n_solver_iters: int = kwargs.pop('n_solver_iters', 50)
        if kwargs.pop('strict', True):
            assert not kwargs, kwargs

    @classmethod
    def with_clargs(cls, **kwargs):
        """Create simulation options and consider command-line arguments"""
        clargs = kwargs.pop('clargs', None)
        if clargs is None:
            clargs = config_parse_args()
        return cls(
            timestep=kwargs.pop('timestep', clargs.timestep),
            gravity=kwargs.pop('gravity', clargs.gravity),
            num_sub_steps=kwargs.pop('num_sub_steps', clargs.num_sub_steps),
            cb_sub_steps=kwargs.pop('cb_sub_steps', clargs.cb_sub_steps),
            n_solver_iters=kwargs.pop('n_solver_iters', clargs.n_solver_iters),
            **kwargs,
        )


class MuJoCoSimulationOptions(Options):
    """MuJoCo simulation options"""
    # pylint: disable=too-many-instance-attributes

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="MuJoCo simulation options",
            description=(
                "Describes the MuJoCo simulation options. "
                + MSG_MUJOCO_OPTIONS
            ),
            class_type=cls,
            children=[
                ChildDoc(
                    name="cone",
                    class_type=str,
                    description="Friction cone (e.g. pyramidal or elliptic).",
                ),
                ChildDoc(
                    name="solver",
                    class_type=str,
                    description="Physics solver (e.g. PGS, CG or Newton).",
                ),
                ChildDoc(
                    name="integrator",
                    class_type=str,
                    description=(
                        "Physics integrator (e.g. Euler, RK4, implicit,"
                        " implicitfast)."
                    ),
                ),
                ChildDoc(
                    name="impratio",
                    class_type=float,
                    description="Frictional-to-normal constraint impedance.",
                ),
                ChildDoc(
                    name="ccd_iterations",
                    class_type=int,
                    description="Convex Collision Detection (CCD) iterations.",
                ),
                ChildDoc(
                    name="ccd_tolerance",
                    class_type=float,
                    description="Convex Collision Detection (CCD) tolerance.",
                ),
                ChildDoc(
                    name="noslip_iterations",
                    class_type=int,
                    description="No slip iterations.",
                ),
                ChildDoc(
                    name="noslip_tolerance",
                    class_type=float,
                    description="No slip tolerance.",
                ),
                ChildDoc(
                    name="texture_repeat",
                    class_type=int,
                    description="Repeating texture.",
                ),
                ChildDoc(
                    name="shadow_size",
                    class_type=int,
                    description="Shadow size.",
                ),
                ChildDoc(
                    name="visual_scale",
                    class_type=float,
                    description="Visual scale.",
                ),
                ChildDoc(
                    name="extent",
                    class_type=float,
                    description="View extent.",
                ),
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        self.cone: str = kwargs.pop('cone', 'pyramidal')
        self.solver: str = kwargs.pop('solver', 'Newton')
        self.integrator: str = kwargs.pop('integrator', 'Euler')
        self.impratio: int = kwargs.pop('impratio', 1)
        self.ccd_iterations: int = kwargs.pop('ccd_iterations', 50)
        self.ccd_tolerance: float = kwargs.pop('ccd_tolerance', 1e-6)
        self.noslip_iterations: int = kwargs.pop('noslip_iterations', 0)
        self.noslip_tolerance: float = kwargs.pop('noslip_tolerance', 1e-6)
        self.texture_repeat: int = kwargs.pop('texture_repeat', 1)
        self.shadow_size: int = kwargs.pop('shadow_size', 1024)
        self.visual_scale: float = kwargs.pop('visual_scale', 1.0)
        self.extent: float = kwargs.pop('extent', 100.0)
        if kwargs.pop('strict', True):
            assert not kwargs, kwargs

    @classmethod
    def with_clargs(cls, **kwargs):
        """Create simulation options and consider command-line arguments"""
        clargs = kwargs.pop('clargs', None)
        if clargs is None:
            clargs = config_parse_args()
        return cls(
            cone=kwargs.pop('cone', clargs.cone),
            solver=kwargs.pop('solver', clargs.solver),
            integrator=kwargs.pop('integrator', clargs.integrator),
            impratio=kwargs.pop('impratio', clargs.impratio),
            ccd_iterations=kwargs.pop('ccd_iterations', clargs.ccd_iterations),
            ccd_tolerance=kwargs.pop('ccd_tolerance', clargs.ccd_tolerance),
            noslip_iterations=kwargs.pop('noslip_iterations', clargs.noslip_iterations),
            noslip_tolerance=kwargs.pop('noslip_tolerance', clargs.noslip_tolerance),
            extent=kwargs.pop('extent', clargs.mujoco_extent),
            **kwargs,
        )


class PybulletSimulationOptions(Options):
    """Pybullet simulation options"""
    # pylint: disable=too-many-instance-attributes

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="Pybullet simulation options",
            description=(
                "Describes the Pybullet simulation options. "
                + MSG_PYBULLET_OPTIONS
            ),
            class_type=cls,
            children=[
                ChildDoc(
                    name="opengl2",
                    class_type=bool,
                    description="Whether to use OpenGL2 instead of OpenGL3.",
                ),
                ChildDoc(
                    name="lcp",
                    class_type=str,
                    description=(
                        "Linear Complementarity Problem (LCP) constraint"
                        " solver (e.g. dantzig)."
                    ),
                ),
                ChildDoc(
                    name="cfm",
                    class_type=float,
                    description="Constraint Force Mixing (CFM).",
                ),
                ChildDoc(
                    name="erp",
                    class_type=float,
                    description="Error Reduction Parameter (ERP).",
                ),
                ChildDoc(
                    name="contact_erp",
                    class_type=float,
                    description="Contact Error Reduction Parameter (ERP).",
                ),
                ChildDoc(
                    name="friction_erp",
                    class_type=float,
                    description="Friction Error Reduction Parameter (ERP).",
                ),
                ChildDoc(
                    name="residual_threshold",
                    class_type=float,
                    description="Residual threshold (e.g. 1e-6).",
                ),
                ChildDoc(
                    name="max_num_cmd_per_1ms",
                    class_type=int,
                    description="Max number of commands per 1ms.",
                ),
                ChildDoc(
                    name="report_solver_analytics",
                    class_type=int,
                    description="Whether to report the solver analytics.",
                ),
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        self.opengl2: bool = kwargs.pop('opengl2', False)
        self.lcp: str = kwargs.pop('lcp', 'dantzig')
        self.cfm: float = kwargs.pop('cfm', 1e-10)
        self.erp: float = kwargs.pop('erp', 0)
        self.contact_erp: float = kwargs.pop('contact_erp', 0)
        self.friction_erp: float = kwargs.pop('friction_erp', 0)
        self.residual_threshold: float = kwargs.pop('residual_threshold', 1e-6)
        self.max_num_cmd_per_1ms: int = kwargs.pop('max_num_cmd_per_1ms', int(1e8))
        self.report_solver_analytics: int = kwargs.pop('report_solver_analytics', 0)
        if kwargs.pop('strict', True):
            assert not kwargs, kwargs

    @classmethod
    def with_clargs(cls, **kwargs):
        """Create simulation options and consider command-line arguments"""
        clargs = kwargs.pop('clargs', None)
        if clargs is None:
            clargs = config_parse_args()
        return cls(
            opengl2=kwargs.pop('opengl2', clargs.opengl2),
            lcp=kwargs.pop('lcp', clargs.lcp),
            cfm=kwargs.pop('cfm', clargs.cfm),
            erp=kwargs.pop('erp', clargs.erp),
            contact_erp=kwargs.pop('contact_erp', clargs.contact_erp),
            friction_erp=kwargs.pop('friction_erp', clargs.friction_erp),
            residual_threshold=kwargs.pop('residual_threshold', clargs.residual_threshold),
            max_num_cmd_per_1ms=kwargs.pop('max_num_cmd_per_1ms', clargs.max_num_cmd_per_1ms),
            **kwargs,
        )
