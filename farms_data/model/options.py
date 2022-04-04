"""Animat options"""

from enum import IntEnum
from typing import List, Dict, Union
from farms_data.options import Options


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
        self.position: List[float] = [*kwargs.pop('position')]
        self.orientation: List[float] = [*kwargs.pop('orientation')]
        self.velocity_lin: List[float] = [*kwargs.pop('velocity_lin')]
        self.velocity_ang: List[float] = [*kwargs.pop('velocity_ang')]
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """From options"""
        options = {}
        # Loader
        options['loader'] = kwargs.pop('spawn_loader', SpawnLoader.FARMS)
        # Position in [m]
        options['position'] = kwargs.pop('spawn_position', [0, 0, 0.1])
        # Orientation in [rad] (Euler angles)
        options['orientation'] = kwargs.pop('spawn_orientation', [0, 0, 0])
        # Linear velocity in [m/s]
        options['velocity_lin'] = kwargs.pop('spawn_velocity_lin', [0, 0, 0])
        # Angular velocity in [rad/s] (Euler angles)
        options['velocity_ang'] = kwargs.pop('spawn_velocity_ang', [0, 0, 0])
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
        joints = kwargs.pop('joints')
        self.joints: List[JointControlOptions] = (
            joints
            if all(isinstance(joint, JointControlOptions) for joint in joints)
            else [JointControlOptions(**joint) for joint in joints]
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
        options['joints'] = kwargs.pop('joints', [])
        return options

    @classmethod
    def from_options(cls, kwargs: Dict):
        """From options"""
        return cls(**cls.options_from_kwargs(kwargs))

    def joints_names(self) -> List[str]:
        """Joints names"""
        return [joint.joint_name for joint in self.joints]

    def joints_limits_torque(self) -> List[float]:
        """Joints max torques"""
        return [joint.limits_torque for joint in self.joints]


class JointControlOptions(Options):
    """Joint options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.joint_name: str = kwargs.pop('joint_name')
        self.control_types: List[str] = kwargs.pop('control_types')
        self.limits_torque: List[float] = kwargs.pop('limits_torque')
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')


class SensorsOptions(Options):
    """Sensors options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.links: List[str] = kwargs.pop('links')
        self.joints: List[str] = kwargs.pop('joints')
        self.contacts: List[str] = kwargs.pop('contacts')
        if kwargs:
            raise Exception(f'Unknown kwargs: {kwargs}')

    @staticmethod
    def options_from_kwargs(kwargs):
        """Options from kwargs"""
        options = {}
        options['links'] = kwargs.pop('sens_links', [])
        options['joints'] = kwargs.pop('sens_joints', [])
        options['contacts'] = kwargs.pop('sens_contacts', [])
        return options

    @classmethod
    def from_options(cls, kwargs: Dict):
        """From options"""
        return cls(**cls.options_from_kwargs(kwargs))


class AnimatOptions(Options):
    """Animat options"""

    def __init__(
            self,
            spawn: Union[SpawnOptions, Dict],
            morphology: Union[MorphologyOptions, Dict],
            control: Union[ControlOptions, Dict],
    ):
        super().__init__()
        self.spawn: SpawnOptions = (
            spawn
            if isinstance(spawn, SpawnOptions)
            else SpawnOptions(**spawn)
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


class ArenaOptions(Options):
    """Arena options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.sdf: str = kwargs.pop('sdf')
        self.loader: SpawnLoader = kwargs.pop('loader')
        self.position: List[float] = [*kwargs.pop('position')]
        self.orientation: List[float] = [*kwargs.pop('orientation')]
        self.ground_height = kwargs.pop('ground_height')
        self.water: WaterOptions = (
            WaterOptions(**kwargs.pop('water'))
            if 'water' in kwargs
            else WaterOptions.from_options(kwargs)
        )


class WaterOptions(Options):
    """Water options"""

    def __init__(self, **kwargs):
        super().__init__()
        self.sdf: str = kwargs.pop('sdf')
        self.height: float = kwargs.pop('height')
        self.velocity: List[float] = [*kwargs.pop('velocity')]
        self.viscosity: float = kwargs.pop('viscosity')
        self.maps: List = kwargs.pop('maps')

    @classmethod
    def from_options(cls, kwargs: Dict):
        """From options"""
        return cls(
            sdf=kwargs.pop('water_sdf', None),
            height=kwargs.pop('water_height', None),
            velocity=kwargs.pop('water_velocity', None),
            viscosity=kwargs.pop('viscosity', None),
            maps=kwargs.pop('water_maps', None),
        )
