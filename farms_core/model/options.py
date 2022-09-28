"""Animat options"""

from enum import IntEnum
from typing import List, Dict, Union
from ..options import Options


class SpawnLoader(IntEnum):
    """Spawn loader"""
    FARMS = 0
    PYBULLET = 1


class MorphologyOptions(Options):
    """Morphology options"""

    def __init__(self, **kwargs):
        super().__init__()
        links = kwargs.pop('links')
        self.links: List[LinkOptions] = (
            links
            if all(isinstance(link, LinkOptions) for link in links)
            else [LinkOptions(**link) for link in kwargs.pop('links')]
        )
        self.self_collisions: List[List[str]] = kwargs.pop('self_collisions')
        joints = kwargs.pop('joints')
        self.joints: List[JointOptions] = (
            joints
            if all(isinstance(joint, JointOptions) for joint in joints)
            else [JointOptions(**joint) for joint in joints]
        )
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    def links_names(self) -> List[str]:
        """Links names"""
        return [link.name for link in self.links]

    def joints_names(self) -> List[str]:
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

    def __init__(self, **kwargs):
        super().__init__()
        self.name: str = kwargs.pop('name')
        self.collisions: bool = kwargs.pop('collisions')
        self.friction: List[float] = kwargs.pop('friction')
        self.extras: Dict = kwargs.pop('extras', {})
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')


class JointOptions(Options):
    """Joint options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.name: str = kwargs.pop('name')
        self.initial: List[float] = kwargs.pop('initial')
        self.limits: List[float] = kwargs.pop('limits')
        self.stiffness: float = kwargs.pop('stiffness')
        self.damping: float = kwargs.pop('damping')
        self.extras: Dict = kwargs.pop('extras', {})
        for i, state in enumerate(['position', 'velocity']):
            assert self.limits[i][0] <= self.limits[i][1], (
                f'Minimum must be smaller than maximum for {state} limits'
            )
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')


class SpawnOptions(Options):
    """Spawn options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.loader: SpawnLoader = kwargs.pop('loader')
        self.pose: List[float] = [*kwargs.pop('pose')]
        self.velocity: List[float] = [*kwargs.pop('velocity')]
        assert len(self.pose) == 6
        assert len(self.velocity) == 6
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """From options"""
        options = {}
        # Loader
        options['loader'] = kwargs.pop('spawn_loader', SpawnLoader.FARMS)
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

    def __init__(self, **kwargs):
        super().__init__()
        sensors = kwargs.pop('sensors')
        self.sensors: SensorsOptions = (
            sensors
            if isinstance(sensors, SensorsOptions)
            else SensorsOptions(**kwargs.pop('sensors'))
        )
        motors = kwargs.pop('motors')
        self.motors: List[MotorOptions] = (
            motors
            if all(isinstance(motor, MotorOptions) for motor in motors)
            else [MotorOptions(**motor) for motor in motors]
        )
        if kwargs:
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
        return options

    @classmethod
    def from_options(cls, kwargs: Dict):
        """From options"""
        return cls(**cls.options_from_kwargs(kwargs))

    def joints_names(self) -> List[str]:
        """Joints names"""
        return [motor.joint_name for motor in self.motors]

    def motors_limits_torque(self) -> List[float]:
        """Motors max torques"""
        return [motor.limits_torque for motor in self.motors]


class MotorOptions(Options):
    """Motor options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.joint_name: str = kwargs.pop('joint_name')
        self.control_types: List[str] = kwargs.pop('control_types')
        self.limits_torque: List[float] = kwargs.pop('limits_torque')
        self.gains: List[float] = kwargs.pop('gains')
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')


class SensorsOptions(Options):
    """Sensors options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.links: List[str] = kwargs.pop('links')
        self.joints: List[str] = kwargs.pop('joints')
        self.contacts: List[List[str]] = kwargs.pop('contacts')
        self.xfrc: List[str] = kwargs.pop('xfrc')
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @staticmethod
    def options_from_kwargs(kwargs):
        """Options from kwargs"""
        options = {}
        options['links'] = kwargs.pop('sens_links', [])
        options['joints'] = kwargs.pop('sens_joints', [])
        options['contacts'] = kwargs.pop('sens_contacts', [])
        options['xfrc'] = kwargs.pop('sens_xfrc', [])
        return options

    @classmethod
    def from_options(cls, kwargs: Dict):
        """From options"""
        return cls(**cls.options_from_kwargs(kwargs))


class ModelOptions(Options):
    """Model options"""

    def __init__(
            self,
            sdf: str,
            spawn: Union[SpawnOptions, Dict],
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

    def __init__(
            self,
            sdf: str,
            spawn: Union[SpawnOptions, Dict],
            morphology: Union[MorphologyOptions, Dict],
            control: Union[ControlOptions, Dict],
    ):
        super().__init__(
            sdf=sdf,
            spawn=spawn,
        )
        self.morphology: MorphologyOptions = (
            morphology
            if isinstance(morphology, MorphologyOptions)
            else MorphologyOptions(**morphology)
        )
        self.control: ControlOptions = (
            control
            if isinstance(control, ControlOptions)
            else ControlOptions(**control)
        )


class WaterOptions(Options):
    """Water options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.sdf: str = kwargs.pop('sdf')
        self.drag: bool = kwargs.pop('drag')
        self.buoyancy: bool = kwargs.pop('buoyancy')
        self.height: float = kwargs.pop('height')
        self.velocity: List[float] = [*kwargs.pop('velocity')]
        self.viscosity: float = kwargs.pop('viscosity')
        self.density: float = kwargs.pop('density')
        self.maps: List = kwargs.pop('maps')


class ArenaOptions(ModelOptions):
    """Arena options"""

    def __init__(
            self,
            sdf: str,
            spawn: Union[SpawnOptions, Dict],
            water: Union[WaterOptions, Dict],
            ground_height: float,
    ):
        super().__init__(sdf=sdf, spawn=spawn)
        self.water: WaterOptions = (
            water
            if isinstance(water, WaterOptions)
            else WaterOptions(**water)
        )
        self.ground_height = ground_height
