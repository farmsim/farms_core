"""Animat options"""

from enum import IntEnum, Enum  # StrEnum, auto
from ..doc import ClassDoc, ChildDoc, get_inherited_doc_children
from ..options import Options


class SpawnLoader(IntEnum):
    """Spawn loader"""
    FARMS = 0
    PYBULLET = 1

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="Spawn loader",
            description="Recommened to use the FARMS loader.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="FARMS",
                    class_type=IntEnum,
                    description="FARMS loader (Default).",
                ),
                ChildDoc(
                    name="PYBULLET",
                    class_type=IntEnum,
                    description="Use Pybullet's SDF loader.",
                ),
            ],
        )


class SpawnMode(str, Enum):  # Not using StrEnum until Python 3.10 EOL
    """Spawn mode, with support for base link constraits"""
    FREE = 'free'
    FIXED = 'fixed'
    ROTX = 'rotx'
    ROTY = 'roty'
    ROTZ = 'rotz'
    SAGITTAL = 'sagittal'       # Longitudinal
    SAGITTAL0 = 'sagittal0'
    SAGITTAL3 = 'sagittal3'
    CORONAL = 'coronal'        # Frontal
    CORONAL0 = 'coronal0'
    CORONAL3 = 'coronal3'
    TRANSVERSE = 'transverse'  # Horizontal
    TRANSVERSE0 = 'transverse0'
    TRANSVERSE3 = 'transverse3'

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="Spawn mode",
            description=(
                "Spawn mode providing the ability to apply constraints."
                " Default is FREE. The constraints apply to the base link."
            ),
            class_type=cls,
            children=[
                ChildDoc(
                    name=name,
                    class_type=Enum,
                    description=description,
                )
                for name, description in [
                    ["FREE", "Free-floating body"],
                    ["FIXED", "Fixed base link"],
                    ["ROTX", "Rotate along X-axis"],
                    ["ROTY", "Rotate along Y-axis"],
                    ["ROTZ", "Rotate along Z-axis"],
                    [
                        "SAGITTAL",
                        "Move along sagital plane (XZ, rotate around Y)",
                    ],
                    [
                        "SAGITTAL0",
                        "Move along sagital plane (XZ, no rotations)",
                    ],
                    [
                        "SAGITTAL3",
                        "Move along sagital plane (XZ, all rotations)",
                    ],
                    [
                        "CORONAL",
                        "Move along coronal plane (YZ, rotate around X)",
                    ],
                    [
                        "CORONAL0",
                        "Move along coronal plane (YZ, no rotations)",
                    ],
                    [
                        "CORONAL3",
                        "Move along coronal plane (YZ, all rotations)",
                    ],
                    [
                        "TRANSVERSE",
                        "Move along transversal plane (XY, rotate around Z)",
                    ],
                    [
                        "TRANSVERSE0",
                        "Move along transversal plane (XY, no rotations)",
                    ],
                    [
                        "TRANSVERSE3",
                        "Move along transversal plane (XY, all rotations)",
                    ],
                ]
            ],
        )


class MorphologyOptions(Options):
    """Morphology options"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="moprhology",
            description="Describes the morphological properties.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="links",
                    class_type="list[LinkOptions]",
                    class_link=LinkOptions,
                    description="Provides options for each link.",
                ),
                ChildDoc(
                    name="joints",
                    class_type="list[JointOptions]",
                    class_link=JointOptions,
                    description="Provides options for each joint.",
                ),
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        strict: bool = kwargs.pop('strict', True)
        links = kwargs.pop('links')
        self.links: list[LinkOptions] = (
            links
            if all(isinstance(link, LinkOptions) for link in links)
            else [LinkOptions(**link, strict=strict) for link in links]
        )
        self.self_collisions: list[list[str]] = kwargs.pop('self_collisions')
        joints = kwargs.pop('joints')
        self.joints: list[JointOptions] = (
            joints
            if all(isinstance(joint, JointOptions) for joint in joints)
            else [JointOptions(**joint, strict=strict) for joint in joints]
        )
        if strict and kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    def links_names(self) -> list[str]:
        """Links names"""
        return [link.name for link in self.links]

    def joints_names(self) -> list[str]:
        """Joints names"""
        return [joint.name for joint in self.joints]

    def n_joints(self) -> int:
        """Number of joints"""
        return len(self.joints)

    def n_links(self) -> int:
        """Number of links"""
        return len(self.links)


class LinkOptions(Options):
    """Link options"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="link",
            description="Describes the link properties.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="name",
                    class_type=str,
                    description="The link name from the SDF file.",
                ),
                ChildDoc(
                    name="collisions",
                    class_type=bool,
                    description="Whether the link should collide.",
                ),
                ChildDoc(
                    name="friction",
                    class_type="list[float]",
                    description="A list of values describing the friction coefficients.",
                ),
                ChildDoc(
                    name="extras",
                    class_type=dict,
                    description="Extra options (Deprecated).",
                ),
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        self.name: str = kwargs.pop('name')
        self.collisions: bool = kwargs.pop('collisions')
        self.friction: list[float] = kwargs.pop('friction')
        self.extras: dict = kwargs.pop('extras', {})
        if kwargs.pop('strict', True) and kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')


class JointOptions(Options):
    """Joint options"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="joint",
            description="Describes the joint properties.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="name",
                    class_type=str,
                    description="The joint name from the SDF file.",
                ),
                ChildDoc(
                    name="initial",
                    class_type="list[float]",
                    description=(
                        "A list 2 float values describing the initial position"
                        " (rad) and velocity (rad/s)."
                    ),
                ),
                ChildDoc(
                    name="limits",
                    class_type="list[float]",
                    description=(
                        "Two lists of 2 float values describing the limit value"
                        " of the joint (the first value must be smaller than or"
                        " equal to the second)."
                    ),
                ),
                ChildDoc(
                    name="stiffness",
                    class_type=float,
                    description=(
                        "The spring stifness value"
                        " (Nm/rad for revolute joints)."
                    ),
                ),
                ChildDoc(
                    name="springref",
                    class_type=float,
                    description=(
                        "The spring reference value"
                        " (rad for revolute joints)."
                    ),
                ),
                ChildDoc(
                    name="damping",
                    class_type=float,
                    description=(
                        "The damping value"
                        " (N·m·s/rad for revolute joints)."
                    ),
                ),
                ChildDoc(
                    name="extras",
                    class_type=dict,
                    description="Extra options (Deprecated).",
                ),
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        self.name: str = kwargs.pop("name")
        self.initial: list[float] = kwargs.pop("initial")
        self.limits: list[float] = kwargs.pop("limits")
        self.stiffness: float = kwargs.pop("stiffness")
        self.springref: float = kwargs.pop("springref")
        self.damping: float = kwargs.pop("damping")
        self.extras: dict = kwargs.pop("extras", {})
        for i, state in enumerate(["position", "velocity"]):
            assert self.limits[i][0] <= self.limits[i][1], (
                f"Minimum must be smaller than maximum for {state} limits"
            )
        if kwargs.pop("strict", True) and kwargs:
            raise Exception(f"Unknown kwargs: {kwargs}")


class SpawnOptions(Options):
    """Spawn options

    Provides the following:

    - loader: Defines the method to use for loading the model.
    - mode: Defines the spawn mode, free of constraints by default.
    - pose: Defines the spawn pose in world coordinates, this is a 6 item vector
      with X, Y, Z position coordinates and alpha, beta, gamma orientation.
    - velocity: Defines the spawn velocity, this is a 6 item vector with Vx, Vy,
      Vz for the linear velocity and omega_x, omega_y, omega_z for angular
      velocity.
    - extras: A dictionary containing extra options.

    """

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="spawn",
            description="Describes the spawn properties.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="loader",
                    class_type=SpawnLoader,
                    description="Choses the loader for loading the animat.",
                ),
                ChildDoc(
                    name="mode",
                    class_type=SpawnMode,
                    description="Mode and constraints to apply to spawn.",
                ),
                ChildDoc(
                    name="pose",
                    class_type="list[float]",
                    description="Spawn pose ([X, Y, Z, Rx, Ry, Rz]).",
                ),
                ChildDoc(
                    name="pose",
                    class_type="list[float]",
                    description="Spawn velocity ([Vx, Vy, Vz, Wx, Wy, Wz]).",
                ),
                ChildDoc(
                    name="extras",
                    class_type=dict,
                    description="Extra options (Deprecated).",
                ),
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        self.loader: SpawnLoader = kwargs.pop('loader')
        self.mode: SpawnMode = kwargs.pop('mode', SpawnMode.FREE)
        self.pose: list[float] = [*kwargs.pop('pose')]
        self.velocity: list[float] = [*kwargs.pop('velocity')]
        self.extras: dict = kwargs.pop('extras', {})
        assert len(self.pose) == 6, f'{self.pose=} should be list of size 6'
        assert len(self.velocity) == 6, f'{self.velocity=} should be of size 6'
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: dict):
        """Spawn options from a dictionary"""
        options = {}
        options['loader'] = kwargs.pop('spawn_loader', SpawnLoader.FARMS)
        options['mode'] = kwargs.pop('spawn_mode', SpawnMode.FREE)
        options['pose'] = (
            # Position in [m]
            list(kwargs.pop('spawn_position', [0, 0, 0]))
            # Orientation in [rad] (Euler angles)
            + list(kwargs.pop('spawn_orientation', [0, 0, 0]))
        )
        options['velocity'] = (
            # Linear velocity in [m/s]
            list(kwargs.pop('spawn_velocity_lin', [0, 0, 0]))
            # Angular velocity in [rad/s]
            + list(kwargs.pop('spawn_velocity_ang', [0, 0, 0]))
        )
        return cls(**options)


class ControlOptions(Options):
    """Control options"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="control",
            description="Describes the control options.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="sensors",
                    class_type=SensorsOptions,
                    description="Sensors options.",
                ),
                ChildDoc(
                    name="motors",
                    class_type="list[MotorOptions]",
                    class_link=MotorOptions,
                    description="List of options for each joint actuator.",
                ),
                ChildDoc(
                    name="hill_muscles",
                    class_type="list[MuscleOptions]",
                    class_link=MuscleOptions,
                    description="List of options for each Hill-type muscle.",
                ),
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        strict: bool = kwargs.pop('strict', True)
        sensors = kwargs.pop('sensors')
        self.sensors: SensorsOptions = (
            sensors
            if isinstance(sensors, SensorsOptions)
            else SensorsOptions(**sensors, strict=strict)
        )
        motors = kwargs.pop('motors')
        self.motors: list[MotorOptions] = (
            motors
            if all(isinstance(motor, MotorOptions) for motor in motors)
            else [MotorOptions(**motor, strict=strict) for motor in motors]
        )
        muscles = kwargs.pop('hill_muscles', [])
        self.hill_muscles: list[MuscleOptions] = (
            muscles
            if all(isinstance(muscle, MuscleOptions) for muscle in muscles)
            else [MuscleOptions(**muscle, strict=strict) for muscle in muscles]
        )
        if strict and kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @staticmethod
    def options_from_kwargs(kwargs):
        """Options from kwargs"""
        options = {}
        options['sensors'] = kwargs.pop(
            'sensors',
            SensorsOptions.from_options(kwargs).to_dict()
        )
        options['motors'] = kwargs.pop('motors', [])
        options['muscles'] = kwargs.pop('muscles', [])
        return options

    @classmethod
    def from_options(cls, kwargs: dict):
        """From options"""
        return cls(**cls.options_from_kwargs(kwargs))

    def joints_names(self) -> list[str]:
        """Joints names"""
        return [motor.joint_name for motor in self.motors]

    def motors_limits_torque(self) -> list[float]:
        """Motors max torques"""
        return [motor.limits_torque for motor in self.motors]


class MotorOptions(Options):
    """Motor options"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="motor",
            description="Describes the motor options.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="joint_name",
                    class_type=str,
                    description="Joint name to actuate.",
                ),
                ChildDoc(
                    name="control_types",
                    class_type="list[str]",
                    description=(
                        "List of control types"
                        " (position/velocity/torque)."
                    ),
                ),
                ChildDoc(
                    name="limits_torques",
                    class_type="list[float]",
                    description="List of torques limits ([min, max])",
                ),
                ChildDoc(
                    name="gains",
                    class_type="list[float]",
                    # TODO FIXME Document for velocity and torque control
                    description=(
                        "Proportional and Derivative gain ([Kp, Kd])"
                        " for position control. Proceed with caution when using"
                        " this for velocity and torque contol."
                    ),
                ),
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        self.joint_name: str = kwargs.pop('joint_name')
        self.control_types: list[str] = kwargs.pop('control_types')
        self.limits_torque: list[float] = kwargs.pop('limits_torque')
        self.gains: list[float] = kwargs.pop('gains')
        if kwargs.pop('strict', True) and kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')


class SensorsOptions(Options):
    """Sensors options"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="sensors",
            description="Describes the sensor options.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="links",
                    class_type="list[str]",
                    description="List of links to track.",
                ),
                ChildDoc(
                    name="joints",
                    class_type="list[str]",
                    description="List of joints to track.",
                ),
                ChildDoc(
                    name="contacts",
                    class_type="list[str] | list[list[str]]",
                    description="List of links / link pairs to track contacts.",
                ),
                ChildDoc(
                    name="xfrc",
                    class_type="list[str]",
                    description="List of links to track external forces.",
                ),
                ChildDoc(
                    name="muscles",
                    class_type="list[str]",
                    description="List of muscles to track.",
                ),
                ChildDoc(
                    name="adhesions",
                    class_type="list[str]",
                    description="List of adhesions to track.",
                ),
                ChildDoc(
                    name="visuals",
                    class_type="list[str]",
                    description="List of visuals to track.",
                ),
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        self.links: list[str] = kwargs.pop('links')
        self.joints: list[str] = kwargs.pop('joints')
        self.contacts: list[str] | list[list[str]] = kwargs.pop('contacts')
        self.xfrc: list[str] = kwargs.pop('xfrc')
        self.muscles: list[str] = kwargs.pop('muscles')
        self.adhesions: list[str] = kwargs.pop('adhesions', [])
        self.visuals: list[str] = kwargs.pop('visuals', [])
        if kwargs.pop('strict', True) and kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @staticmethod
    def options_from_kwargs(kwargs):
        """Options from kwargs"""
        options = {}
        options['links'] = kwargs.pop('sens_links', [])
        options['joints'] = kwargs.pop('sens_joints', [])
        options['contacts'] = kwargs.pop('sens_contacts', [])
        options['xfrc'] = kwargs.pop('sens_xfrc', [])
        options['muscles'] = kwargs.pop('sens_muscles', [])
        options['adhesions'] = kwargs.pop('sens_adhesions', [])
        options['visuals'] = kwargs.pop('sens_visuals', [])
        return options

    @classmethod
    def from_options(cls, kwargs: dict):
        """From options"""
        return cls(**cls.options_from_kwargs(kwargs))


class ModelOptions(Options):
    """Model options"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="animat",
            description="Describes the animat properties.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="sdf",
                    class_type=str,
                    description="Path to SDF file.",
                ),
                ChildDoc(
                    name="spawn",
                    class_type=SpawnOptions,
                    description="Provides options for spawning the animat.",
                ),
            ],
        )

    def __init__(
            self,
            sdf: str,
            spawn: SpawnOptions | dict,
    ):
        super().__init__()
        self.sdf: str = sdf
        self.spawn: SpawnOptions = (
            spawn
            if isinstance(spawn, SpawnOptions)
            else SpawnOptions(**spawn)
        )


class AnimatOptions(ModelOptions):
    """Animat options"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="animat",
            description="Describes the animat properties.",
            class_type=cls,
            children=get_inherited_doc_children(cls) + [
                ChildDoc(
                    name="morphology",
                    class_type=MorphologyOptions,
                    description="Provides animat morphology options.",
                ),
                ChildDoc(
                    name="control",
                    class_type=ControlOptions,
                    description="Provides control options.",
                ),
            ],
        )

    def __init__(
            self,
            sdf: str,
            spawn: SpawnOptions | dict,
            morphology: MorphologyOptions | dict,
            control: ControlOptions | dict,
            **kwargs,
    ):
        strict: bool = kwargs.pop('strict', True)
        super().__init__(
            sdf=sdf,
            spawn=spawn,
        )
        self.morphology: MorphologyOptions = (
            morphology
            if isinstance(morphology, MorphologyOptions)
            else MorphologyOptions(**morphology, strict=strict)
        )
        self.control: ControlOptions = (
            control
            if isinstance(control, ControlOptions)
            else ControlOptions(**control, strict=strict)
        )
        if strict and kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')


class WaterOptions(Options):
    """Water options"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="water",
            description="Describes the water options.",
            class_type=cls,
            children=[
                ChildDoc(
                    name='sdf',
                    class_type=str,
                    description="Path to SDF file.",
                ),
                ChildDoc(
                    name="drag",
                    class_type=bool,
                    description="Whether to apply drag forces.",
                ),
                ChildDoc(
                    name="buoyancy",
                    class_type=bool,
                    description="Whether to apply buoyancy forces.",
                ),
                ChildDoc(
                    name="height",
                    class_type=float,
                    description="Height of water surface.",
                ),
                ChildDoc(
                    name="velocity",
                    class_type=float,
                    description="Water velocity in 3 directions [Vx, Vy, Vz].",
                ),
                ChildDoc(
                    name="viscosity",
                    class_type=float,
                    description="Viscosity of water.",
                ),
                ChildDoc(
                    name="density",
                    class_type=float,
                    description="Density of water.",
                ),
                ChildDoc(
                    name="maps",
                    class_type="list[str]",
                    description="Provides water maps from images.",
                ),
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        self.sdf: str = kwargs.pop('sdf')
        self.drag: bool = kwargs.pop('drag')
        self.buoyancy: bool = kwargs.pop('buoyancy')
        self.height: float = kwargs.pop('height')
        self.velocity: list[float] = [*kwargs.pop('velocity')]
        self.viscosity: float = kwargs.pop('viscosity')
        self.density: float = kwargs.pop('density')
        self.maps: list = kwargs.pop('maps')
        if kwargs.pop('strict', True) and kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')


class ArenaOptions(ModelOptions):
    """Arena options"""

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="arena",
            description="Describes the arena properties.",
            class_type=cls,
            children=get_inherited_doc_children(cls) + [
                ChildDoc(
                    name="water",
                    class_type=WaterOptions,
                    description="Provides water options.",
                ),
                ChildDoc(
                    name="ground_height",
                    class_type=float,
                    description="Height offset at which to place the arena.",
                ),
            ],
        )

    def __init__(
            self,
            sdf: str,
            spawn: SpawnOptions | dict,
            water: WaterOptions | dict,
            ground_height: float,
            **kwargs,
    ):
        super().__init__(sdf=sdf, spawn=spawn)
        strict: bool = kwargs.pop('strict', True)
        self.water: WaterOptions = (
            water
            if isinstance(water, WaterOptions)
            else WaterOptions(**water, strict=strict)
        )
        self.ground_height = ground_height
        if strict and kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')


class MuscleOptions(Options):
    """ Muscle Options """

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name="muscle",
            description="Describes the properties of Hill-type muscles.",
            class_type=cls,
            children=[
                ChildDoc(
                    name="name",
                    class_type=str,
                    description="Muscle name.",
                ),
                ChildDoc(
                    name="model",
                    class_type=str,
                    description="Muscle model.",
                ),
                ChildDoc(
                    name="max_force",
                    class_type=float,
                    description="Maximum force the muscle can exert.",
                ),
                ChildDoc(
                    name="optimal_fiber",
                    class_type=float,
                    description="Optimal fiber length.",
                ),
                ChildDoc(
                    name="tendon_slack",
                    class_type=float,
                    description="Tendon slack.",
                ),
                ChildDoc(
                    name="max_velocity",
                    class_type=float,
                    description="Maximum velocity.",
                ),
                ChildDoc(
                    name="pennation_angle",
                    class_type=float,
                    description="Pennation angle.",
                ),
                ChildDoc(
                    name="lmtu_min",
                    class_type=float,
                    description="Minimum muscle-tendon unit length.",
                ),
                ChildDoc(
                    name="lmtu_max",
                    class_type=float,
                    description="Maximum muscle-tendon unit length.",
                ),
                ChildDoc(
                    name="waypoints",
                    class_type="list[list[float]]",
                    description="Muscle waypoints.",
                ),
                ChildDoc(
                    name="act_tconst",
                    class_type=float,
                    description="Activation time constant.",
                ),
                ChildDoc(
                    name="deact_tconst",
                    class_type=float,
                    description="Deactivation time constant.",
                ),
                ChildDoc(
                    name="lmin",
                    class_type=float,
                    description="Minimum muscle length.",
                ),
                ChildDoc(
                    name="lmax",
                    class_type=float,
                    description="Maximum muscle length.",
                ),
                # initialization
                ChildDoc(
                    name="init_activation",
                    class_type=float,
                    description="Initial activation.",
                ),
                ChildDoc(
                    name="init_fiber",
                    class_type=float,
                    description="Initial fiber length.",
                ),
                # type I afferent constants
                ChildDoc(
                    name="type_I_kv",
                    class_type=float,
                    description="Type I kv.",
                ),
                ChildDoc(
                    name="type_I_pv",
                    class_type=float,
                    description="Type I pv.",
                ),
                ChildDoc(
                    name="type_I_k_dI",
                    class_type=float,
                    description="Type I k dI.",
                ),
                ChildDoc(
                    name="type_I_k_nI",
                    class_type=float,
                    description="Type I k nI.",
                ),
                ChildDoc(
                    name="type_I_const_I",
                    class_type=float,
                    description="Type I constant I.",
                ),
                ChildDoc(
                    name="type_I_l_ce_th",
                    class_type=float,
                    description="Type I l ce th.",
                ),
                # type Ib afferent constants
                ChildDoc(
                    name="type_Ib_kf",
                    class_type=float,
                    description="Type Ib kf.",
                ),
                # type II afferent constants
                ChildDoc(
                    name="type_II_k_dII",
                    class_type=float,
                    description="Type II k dII.",
                ),
                ChildDoc(
                    name="type_II_k_nII",
                    class_type=float,
                    description="Type II k nII.",
                ),
                ChildDoc(
                    name="type_II_const_II",
                    class_type=float,
                    description="Type II constant II.",
                ),
                ChildDoc(
                    name="type_II_l_ce_th",
                    class_type=float,
                    description="Type II l ce th.",
                ),
            ],
        )

    def __init__(self, **kwargs):
        super().__init__()
        self.name: str = kwargs.pop('name')
        self.model: str = kwargs.pop('model')
        # muscle properties
        self.max_force: float = kwargs.pop('max_force')
        self.optimal_fiber: float = kwargs.pop('optimal_fiber')
        self.tendon_slack: float = kwargs.pop('tendon_slack')
        self.max_velocity: float = kwargs.pop('max_velocity')
        self.pennation_angle: float = kwargs.pop('pennation_angle')
        self.lmtu_min: float = kwargs.pop('lmtu_min')
        self.lmtu_max: float = kwargs.pop('lmtu_max')
        self.waypoints: list[list] = kwargs.pop('waypoints')
        self.act_tconst: float = kwargs.pop('act_tconst', 0.001)
        self.deact_tconst: float = kwargs.pop('deact_tconst', 0.001)
        self.lmin: float = kwargs.pop(
            'lmin',
            self.lmtu_min-self.tendon_slack/self.optimal_fiber
        )
        self.lmax: float = kwargs.pop(
            'lmax',
            self.lmtu_max-self.tendon_slack/self.optimal_fiber
        )
        # initialization
        self.init_activation: float = kwargs.pop('init_activation', 0.0)
        self.init_fiber: float = kwargs.pop('init_fiber', self.optimal_fiber)
        # type I afferent constants
        self.type_I_kv = kwargs.pop('type_I_kv', 6.2/6.2)
        self.type_I_pv = kwargs.pop('type_I_pv', 0.6)
        self.type_I_k_dI = kwargs.pop('type_I_k_dI', 2.0/6.2)
        self.type_I_k_nI = kwargs.pop('type_I_k_nI', 0.06)
        self.type_I_const_I = kwargs.pop('type_I_const_I', 0.05)
        self.type_I_l_ce_th = kwargs.pop('type_I_l_ce_th', 0.85)
        # type Ib afferent constants
        self.type_Ib_kF = kwargs.pop('type_Ib_kF', 1.0)
        # type II afferent constants
        self.type_II_k_dII = kwargs.pop('type_II_k_dII', 1.5)
        self.type_II_k_nII = kwargs.pop('type_II_k_nII', 0.06)
        self.type_II_const_II = kwargs.pop('type_II_const_II', 0.05)
        self.type_II_l_ce_th = kwargs.pop('type_II_l_ce_th', 0.85)
        if kwargs.pop('strict', True) and kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')
