"""Animat data"""

from typing import List, Dict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import circmean
from ..array.array import to_array
from ..array.array_cy import DoubleArray3D
from ..array.types import (
    NDARRAY_V1,
    NDARRAY_V1_D,
    NDARRAY_V2_D,
    NDARRAY_3_D,
    NDARRAY_4_D,
    NDARRAY_X3_D,
    NDARRAY_V3_D,
    NDARRAY_XX3_D,
    NDARRAY_XX4_D,
    NDARRAY_LINKS_ARRAY,
)
from ..model.options import AnimatOptions
from ..simulation.options import SimulationOptions
from ..utils.transform import quat2euler
from .sensor_convention import sc
from .data_cy import (
    SensorsDataCy,
    LinkSensorArrayCy,
    JointSensorArrayCy,
    ContactsArrayCy,
    XfrcArrayCy,
    MusclesArrayCy
)

# pylint: disable=no-member,unsubscriptable-object
# pylint: disable=unsupported-assignment-operation


NPDTYPE = np.float64
NPUITYPE = np.uintc


class ClassDoc:
    """Class documentation"""

    def __init__(self, name, description, class_type, children):
        super().__init__()
        self.name: str = name
        self.description: str = description
        self.class_type = class_type
        self.children: List[ClassDoc] = children


class SensorData(DoubleArray3D):
    """SensorData

    This class inherits from the 3-dimensional array, with the following meaning
    for each dimension:

    1) Buffer iteration
    2) Sensor number, in the order of self.names
    3) Data type (referring to the sensor conventions)

    """

    def __init__(
            self,
            array: NDARRAY_V3_D,
            names: List[str],
    ):
        super().__init__(array)
        self.names = names

    @classmethod
    def from_dict(
            cls,
            dictionary: Dict,
    ):
        """Load data from dictionary"""
        return cls(
            array=dictionary['array'],
            names=dictionary['names'],
        )

    def to_dict(
            self,
            iteration: int = None,
    ) -> Dict:
        """Convert data to dictionary"""
        return {
            'array': to_array(self.array, iteration),
            'names': self.names,
        }


class SensorsData(SensorsDataCy):
    """SensorsData

    Class for containing all the different sensors of a rigid-body animat. This
    includes links, joints, contacts, external forces and muscles sensors.

    """

    @classmethod
    def doc(cls):
        """Doc"""
        return ClassDoc(
            name='sensors',
            description='Contains the sensors data logged from the physics engine.',
            class_type=SensorsData,
            children=[
                LinkSensorArray,
                JointSensorArray,
                ContactsArray,
                XfrcArray,
                MusclesArray,
            ]
        )

    @classmethod
    def from_names(
            cls,
            buffer_size: int,
            links_names: List[str],
            joints_names: List[str],
            contacts_names: List[str],
            xfrc_names: List[str],
            muscles_names: List[str],
    ):
        """From options"""
        return SensorsData(
            links=LinkSensorArray.from_names(
                names=links_names,
                buffer_size=buffer_size,
            ),
            joints=JointSensorArray.from_names(
                names=joints_names,
                buffer_size=buffer_size,
            ),
            contacts=ContactsArray.from_names(
                names=contacts_names,
                buffer_size=buffer_size,
            ),
            xfrc=XfrcArray.from_names(
                names=xfrc_names,
                buffer_size=buffer_size,
            ),
            muscles=MusclesArray.from_names(
                names=muscles_names,
                buffer_size=buffer_size,
            ),
        )

    @classmethod
    def from_options(
            cls,
            animat_options: AnimatOptions,
            simulation_options: SimulationOptions,
    ):
        """From options"""
        return cls.from_names(
            buffer_size=simulation_options.buffer_size,
            links_names=animat_options.control.sensors.links,
            joints_names=animat_options.control.sensors.joints,
            contacts_names=animat_options.control.sensors.contacts,
            xfrc_names=animat_options.control.sensors.xfrc,
            muscles_names=animat_options.control.sensors.muscles,
        )

    @classmethod
    def from_dict(
            cls,
            dictionary: Dict,
    ):
        """Load data from dictionary"""
        return cls(
            links=LinkSensorArray.from_dict(dictionary['links']),
            joints=JointSensorArray.from_dict(dictionary['joints']),
            contacts=ContactsArray.from_dict(dictionary['contacts']),
            xfrc=XfrcArray.from_dict(dictionary['xfrc']),
            muscles=(
                MusclesArray.from_dict(dictionary['muscles'])
                if 'muscles' in dictionary
                else MusclesArray.from_names(names=[], buffer_size=0)
            ),
        )

    def to_dict(
            self,
            iteration: int = None,
    ) -> Dict:
        """Convert data to dictionary"""
        return {
            name: data.to_dict(iteration)
            for name, data in [
                ['links', self.links],
                ['joints', self.joints],
                ['contacts', self.contacts],
                ['xfrc', self.xfrc],
                ['muscles', self.muscles],
            ]
            if data is not None
        }

    def plot(
            self,
            times: NDARRAY_V1,
    ) -> Dict:
        """Plot"""
        plots = {}
        plots.update(self.links.plot(times))
        plots.update(self.joints.plot(times))
        plots.update(self.contacts.plot(times))
        plots.update(self.xfrc.plot(times))
        return plots


def _sensor_array_doc(class_type, name, array_type, description=''):
    """Sensor array doc"""
    return ClassDoc(
            name=name,
            description=(
                description
                if description
                else 'Contains the sensors data logged from the physics engine.'
            ),
            class_type=class_type,
            children=[
                ClassDoc(
                    name='names',
                    description=(
                        f'List of {name} names, in order of indices'
                        ' in the array'
                    ),
                    class_type=List[str],
                    children=[],
                ),
                ClassDoc(
                    name='array',
                    description=(
                        f'Array containing the {name} data, refer to the'
                        ' farms_core/sensor/sensor_convention for information'
                        ' about indices.'
                    ),
                    class_type=array_type,
                    children=[],
                ),
            ]
        )


class LinkSensorArray(SensorData, LinkSensorArrayCy):
    """Links array"""

    def __init__(
            self,
            array: NDARRAY_LINKS_ARRAY,
            names: List[str],
    ):
        super().__init__(array, names)
        self.masses = None

    @classmethod
    def doc(cls):
        """Doc"""
        return _sensor_array_doc(
            cls,
            'links',
            array_type=NDARRAY_LINKS_ARRAY,
            # Array[np.double, '[n_iterations, n_{name}, {size}], double']
            description=(
                'Links positions, orientations, velocities,'
                ' angular velocities, ...'
            ),
        )

    @classmethod
    def from_dict(
            cls,
            dictionary: Dict,
    ):
        """Load data from dictionary"""
        links = super(cls, cls).from_dict(dictionary)
        links.masses = dictionary['masses']
        return links

    def to_dict(
            self,
            iteration: int = None,
    ):
        """Convert data to dictionary"""
        links = super().to_dict(iteration=iteration)
        links['masses'] = self.masses
        return links

    @classmethod
    def from_names(
            cls,
            names: List[str],
            buffer_size: int,
    ):
        """From names"""
        n_sensors = len(names)
        array = np.full(
            shape=[buffer_size, n_sensors, sc.link_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(array, names)

    @classmethod
    def from_size(
            cls, n_links: int,
            buffer_size: int,
            names: List[str],
    ):
        """From size"""
        links = np.full(
            shape=[buffer_size, n_links, sc.link_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(links, names)

    @classmethod
    def from_parameters(
            cls,
            buffer_size: int,
            n_links: int,
            names: List[str],
    ):
        """From parameters"""
        return cls(
            np.full(
                shape=[buffer_size, n_links, sc.link_size],
                fill_value=0,
                dtype=NPDTYPE,
            ),
            names,
        )

    def com_position(
            self,
            iteration: int,
            link_i: int,
    ) -> NDARRAY_3_D:
        """CoM position of a link"""
        return self.array[iteration, link_i, sc.link_com_position_x:sc.link_com_position_z+1]

    def com_orientation(
            self,
            iteration: int,
            link_i: int,
    ) -> NDARRAY_4_D:
        """CoM orientation of a link"""
        return self.array[iteration, link_i, sc.link_com_orientation_x:sc.link_com_orientation_w+1]

    def com_positions(self) -> NDARRAY_XX3_D:
        """CoM position of a link"""
        return self.array[:, :, sc.link_com_position_x:sc.link_com_position_z+1]

    def urdf_position(
            self,
            iteration: int,
            link_i: int,
    ) -> NDARRAY_3_D:
        """URDF position of a link"""
        return self.array[iteration, link_i, sc.link_urdf_position_x:sc.link_urdf_position_z+1]

    def urdf_positions(self) -> NDARRAY_XX3_D:
        """URDF position of a link"""
        return self.array[:, :, sc.link_urdf_position_x:sc.link_urdf_position_z+1]

    def urdf_orientation(
            self,
            iteration: int,
            link_i: int,
    ) -> NDARRAY_4_D:
        """URDF orientation of a link"""
        return self.array[iteration, link_i, sc.link_urdf_orientation_x:sc.link_urdf_orientation_w+1]

    def urdf_orientations(self) -> NDARRAY_XX4_D:
        """URDF orientation of a link"""
        return self.array[:, :, sc.link_urdf_orientation_x:sc.link_urdf_orientation_w+1]

    def com_lin_velocity(
            self,
            iteration: int,
            link_i: int,
    ) -> NDARRAY_3_D:
        """CoM linear velocity of a link"""
        return self.array[iteration, link_i, sc.link_com_velocity_lin_x:sc.link_com_velocity_lin_z+1]

    def com_lin_velocities(self) -> NDARRAY_XX3_D:
        """CoM linear velocities"""
        return self.array[:, :, sc.link_com_velocity_lin_x:sc.link_com_velocity_lin_z+1]

    def com_ang_velocity(
            self,
            iteration: int,
            link_i: int,
    ) -> NDARRAY_3_D:
        """CoM angular velocity of a link"""
        return self.array[iteration, link_i, sc.link_com_velocity_ang_x:sc.link_com_velocity_ang_z+1]

    def global_com_position(self, iteration: int):
        """Global CoM position"""
        total_mass = 0
        mass_position = np.zeros(3)
        for link_i, mass in enumerate(self.masses):
            mass_position += mass*self.com_position(
                iteration=iteration,
                link_i=link_i,
            )
            total_mass += mass
        assert total_mass > 0, 'No masses'
        mass_position /= total_mass
        return mass_position

    def heading(
            self,
            iteration: int,
            indices: List[int],
    ) -> float:
        """Heading"""
        n_indices = len(indices)
        link_orientation = np.zeros(n_indices)
        ori = np.zeros(3)
        for idx, link_idx in enumerate(indices):
            quat2euler(
                quat=self.urdf_orientation(
                    iteration=iteration,
                    link_i=link_idx,
                ),
                out=ori,
            )
            link_orientation[idx] = ori[2]
        return circmean(
            samples=link_orientation,
            low=-np.pi,
            high=np.pi,
        ) if n_indices > 1 else link_orientation[0]

    def plot(
            self,
            times: NDARRAY_V1,
    ) -> Dict:
        """Plot"""
        return {
            'base_position': self.plot_base_position(times, xaxis=0, yaxis=1),
            'base_velocity': self.plot_base_velocity(times),
            'heading': self.plot_heading(times),
        }

    def plot_base_position(
            self,
            times: NDARRAY_V1,
            xaxis: int = 0,
            yaxis: int = 1,
    ) -> Figure:
        """Plot"""
        fig = plt.figure('Links position')
        for link_i in range(self.size(1)):
            data = np.asarray(self.urdf_positions())[:len(times), link_i]
            plt.plot(
                data[:, xaxis],
                data[:, yaxis],
                label=f'Link_{link_i}',
            )
        plt.legend()
        plt.xlabel('Position [m]')
        plt.ylabel('Position [m]')
        plt.axis('equal')
        plt.grid(True)
        return fig

    def plot_base_velocity(
            self,
            times: NDARRAY_V1,
    ) -> Figure:
        """Plot"""
        fig = plt.figure('Links velocities')
        for link_i in range(self.size(1)):
            data = np.asarray(self.com_lin_velocities())[:len(times), link_i]
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1),
                label=f'Link_{link_i}',
            )
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.grid(True)
        return fig

    def plot_heading(
            self,
            times: NDARRAY_V1,
            indices: List[int] = None,
    ) -> Figure:
        """Plot"""
        if indices is None:
            indices = [0]
        fig = plt.figure('Heading')
        plt.plot(
            times,
            [self.heading(i, indices) for i, _ in enumerate(times)],
        )
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Heading [rad]')
        plt.grid(True)
        return fig


class JointSensorArray(SensorData, JointSensorArrayCy):
    """Joint sensor array"""

    @classmethod
    def doc(cls):
        """Doc"""
        return _sensor_array_doc(
            cls,
            'joints',
            array_type=DoubleArray3D,
            # size=sc.joint_size,
            description='Joints positions, velocities, forces, commands, ...',
        )

    @classmethod
    def from_names(
            cls,
            names: List[str],
            buffer_size: int,
    ):
        """From names"""
        n_sensors = len(names)
        array = np.full(
            shape=[buffer_size, n_sensors, sc.joint_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(array, names)

    @classmethod
    def from_size(
            cls,
            n_joints: int,
            buffer_size: int,
            names: List[str],
    ):
        """From size"""
        joints = np.full(
            shape=[buffer_size, n_joints, sc.joint_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(joints, names)

    @classmethod
    def from_parameters(
            cls,
            buffer_size: int,
            n_joints: int,
            names: List[str],
    ):
        """From parameters"""
        return cls(
            np.full(
                shape=[buffer_size, n_joints, sc.joint_size],
                fill_value=0,
                dtype=NPDTYPE,
            ),
            names,
        )

    def position(
            self,
            iteration: int,
            joint_i: int,
    ) -> float:
        """Joint position"""
        return self.array[iteration, joint_i, sc.joint_position]

    def positions(
            self,
            iteration: int,
    ) -> NDARRAY_V1_D:
        """Joints positions"""
        return self.array[iteration, :, sc.joint_position]

    def positions_all(self) -> NDARRAY_V2_D:
        """Joints positions"""
        return self.array[:, :, sc.joint_position]

    def velocity(
            self,
            iteration: int,
            joint_i: int,
    ) -> float:
        """Joint velocity"""
        return self.array[iteration, joint_i, sc.joint_velocity]

    def velocities(
            self,
            iteration: int,
    ) -> NDARRAY_V1_D:
        """Joints velocities"""
        return self.array[iteration, :, sc.joint_velocity]

    def velocities_all(self) -> NDARRAY_V2_D:
        """Joints velocities"""
        return self.array[:, :, sc.joint_velocity]

    def motor_torque(
            self,
            iteration: int,
            joint_i: int,
    ) -> float:
        """Joint torque"""
        return self.array[iteration, joint_i, sc.joint_torque]

    def motor_torques(
            self,
            iteration: int,
    ) -> NDARRAY_V1_D:
        """Joint torques"""
        return self.array[iteration, :, sc.joint_torque]

    def motor_torques_all(self) -> NDARRAY_V2_D:
        """Joint torque"""
        return self.array[:, :, sc.joint_torque]

    def force(
            self,
            iteration: int,
            joint_i: int,
    ) -> NDARRAY_3_D:
        """Joint force"""
        return self.array[iteration, joint_i, sc.joint_force_x:sc.joint_force_z+1]

    def forces_all(self) -> NDARRAY_XX3_D:
        """Joints forces"""
        return self.array[:, :, sc.joint_force_x:sc.joint_force_z+1]

    def torque(
            self,
            iteration: int,
            joint_i: int,
    ) -> NDARRAY_3_D:
        """Joint torque"""
        return self.array[iteration, joint_i, sc.joint_torque_x:sc.joint_torque_z+1]

    def torques_all(self) -> NDARRAY_XX3_D:
        """Joints torques"""
        return self.array[:, :, sc.joint_torque_x:sc.joint_torque_z+1]

    def cmd_position(
            self,
            iteration: int,
            joint_i: int,
    ) -> float:
        """Joint position"""
        return self.array[iteration, joint_i, sc.joint_cmd_position]

    def cmd_positions(self) -> NDARRAY_V2_D:
        """Joint position"""
        return self.array[:, :, sc.joint_cmd_position]

    def cmd_velocity(
            self,
            iteration: int,
            joint_i: int,
    ) -> float:
        """Joint velocity"""
        return self.array[iteration, joint_i, sc.joint_cmd_velocity]

    def cmd_velocities(self) -> NDARRAY_V2_D:
        """Joint velocity"""
        return self.array[:, :, sc.joint_cmd_velocity]

    def cmd_torque(
            self,
            iteration: int,
            joint_i: int,
    ) -> float:
        """Joint torque"""
        return self.array[iteration, joint_i, sc.joint_cmd_torque]

    def cmd_torques(self) -> NDARRAY_V2_D:
        """Joint torque"""
        return self.array[:, :, sc.joint_cmd_torque]

    def active(
            self,
            iteration: int,
            joint_i: int,
    ) -> float:
        """Active torque"""
        return self.array[iteration, joint_i, sc.joint_torque_active]

    def active_torques(self) -> NDARRAY_V2_D:
        """Active torques"""
        return self.array[:, :, sc.joint_torque_active]

    def spring(
            self,
            iteration: int,
            joint_i: int,
    ) -> float:
        """Passive spring torque"""
        return self.array[iteration, joint_i, sc.joint_torque_stiffness]

    def spring_torques(self) -> NDARRAY_V2_D:
        """Spring torques"""
        return self.array[:, :, sc.joint_torque_stiffness]

    def damping(
            self,
            iteration: int,
            joint_i: int,
    ) -> float:
        """passive damping torque"""
        return self.array[iteration, joint_i, sc.joint_torque_damping]

    def damping_torques(self) -> NDARRAY_V2_D:
        """Damping torques"""
        return self.array[:, :, sc.joint_torque_damping]

    def friction(
            self,
            iteration: int,
            joint_i: int,
    ) -> float:
        """passive friction torque"""
        return self.array[iteration, joint_i, sc.joint_torque_friction]

    def friction_torques(self) -> NDARRAY_V2_D:
        """Friction torques"""
        return self.array[:, :, sc.joint_torque_friction]

    def mechanical_power(self):
        """Compute mechanical power"""
        return np.array(self.cmd_torques())*np.array(self.velocities_all())

    def mechanical_power_active(self):
        """Compute active mechanical power"""
        return np.array(self.active_torques())*np.array(self.velocities_all())

    def limit_force(
            self,
            iteration: int,
            joint_i: int,
    ) -> NDARRAY_3_D:
        """Joint limit force"""
        return self.array[iteration, joint_i, sc.joint_limit_force]

    def limit_forces_all(self) -> NDARRAY_XX3_D:
        """Joints limits forces"""
        return self.array[:, :, sc.joint_limit_force]

    def plot(
            self,
            times: NDARRAY_V1,
    ) -> Dict:
        """Plot"""
        t_init = times[:50]
        return {
            'joints_positions': self.plot_positions(times),
            'joints_velocities': self.plot_velocities(times),
            'joints_motor_torques': self.plot_motor_torques(times),
            'joints_forces': self.plot_forces(times),
            'joints_torques': self.plot_torques(times),
            'joints_cmd_positions': self.plot_cmd_positions(times),
            'joints_cmd_velocities': self.plot_cmd_velocities(times),
            'joints_cmd_torques': self.plot_cmd_torques(times),
            'joints_active_torques': self.plot_active_torques(times),
            'joints_spring_torques': self.plot_spring_torques(times),
            'joints_damping_torques': self.plot_damping_torques(times),
            'joints_friction_torques': self.plot_friction_torques(times),
            'joints_ti_positions': self.plot_positions(t_init, ' init'),
            'joints_ti_velocities': self.plot_velocities(t_init, ' init'),
            'joints_ti_motor_torques': self.plot_motor_torques(t_init, ' init'),
            'joints_ti_cmd_positions': self.plot_cmd_positions(t_init, ' init'),
            'joints_ti_cmd_velocities': self.plot_cmd_velocities(t_init, ' init'),
            'joints_ti_cmd_torques': self.plot_cmd_torques(t_init, ' init'),
            'joints_ti_active': self.plot_active_torques(t_init, ' init'),
            'joints_ti_spring': self.plot_spring_torques(t_init, ' init'),
            'joints_ti_damping': self.plot_damping_torques(t_init, ' init'),
            'joints_ti_friction': self.plot_friction_torques(t_init, ' init'),
        }

    def plot_data(
            self,
            times: NDARRAY_V1,
            data: NDARRAY_V2_D,
            joint_i: int,
    ):
        """Plot data"""
        _data = np.asarray(data)
        plt.plot(times, _data[:len(times), joint_i], label=self.names[joint_i])

    @staticmethod
    def plot_end(
            ylabel: str,
            n_labels: int,
            max_labels_per_row: int = 20,
    ):
        """plot_end"""
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc='upper left',
            borderaxespad=0,
            ncol=n_labels//max_labels_per_row+1,
        )
        plt.xlabel('Times [s]')
        plt.ylabel(ylabel)
        plt.grid(True)

    def plot_generic(
            self,
            times: NDARRAY_V1,
            data: NDARRAY_V2_D,
            title: str,
            ylabel: str,
    ) -> Figure:
        """Plot joint sensor"""
        fig = plt.figure(title)
        n_joints = self.size(1)
        for joint_i in range(n_joints):
            self.plot_data(times=times, data=data, joint_i=joint_i)
        self.plot_end(ylabel=ylabel, n_labels=n_joints)
        return fig

    def plot_generic_3(
            self,
            times: NDARRAY_V1,
            data: NDARRAY_V2_D,
            title: str,
            ylabel: str,
    ) -> Figure:
        """Plot ground reaction forces"""
        fig = plt.figure(title)
        _data = np.linalg.norm(np.asarray(data), axis=-1)
        n_joints = self.size(1)
        for joint_i in range(n_joints):
            self.plot_data(times=times, data=_data, joint_i=joint_i)
        self.plot_end(ylabel=ylabel, n_labels=n_joints)
        return fig

    def plot_positions(
            self,
            times: NDARRAY_V1,
            suffix: str = '',
    ) -> Figure:
        """Plot ground reaction forces"""
        return self.plot_generic(
            times=times,
            data=self.positions_all(),
            title=f'Joints positions{suffix}',
            ylabel='Joint position [rad]',
        )

    def plot_velocities(
            self,
            times: NDARRAY_V1,
            suffix: str = '',
    ) -> Figure:
        """Plot ground reaction forces"""
        return self.plot_generic(
            times=times,
            data=self.velocities_all(),
            title=f'Joints velocities{suffix}',
            ylabel='Joint velocity [rad/s]',
        )

    def plot_motor_torques(
            self,
            times: NDARRAY_V1,
            suffix: str = '',
    ) -> Figure:
        """Plot joints motor torques"""
        return self.plot_generic(
            times=times,
            data=self.motor_torques_all(),
            title=f'Joints motor torques{suffix}',
            ylabel='Joint torque [Nm]',
        )

    def plot_forces(
            self,
            times: NDARRAY_V1,
            suffix: str = '',
    ) -> Figure:
        """Plot ground reaction forces"""
        return self.plot_generic_3(
            times=times,
            data=self.forces_all(),
            title=f'Joints forces{suffix}',
            ylabel='Joint force [N]',
        )

    def plot_torques(
            self,
            times: NDARRAY_V1,
            suffix: str = '',
    ) -> Figure:
        """Plot ground reaction torques"""
        return self.plot_generic_3(
            times=times,
            data=self.torques_all(),
            title=f'Joints torques{suffix}',
            ylabel='Joint torque [Nm]',
        )

    def plot_cmd_positions(
            self,
            times: NDARRAY_V1,
            suffix: str = '',
    ) -> Figure:
        """Plot joints command positions"""
        return self.plot_generic(
            times=times,
            data=self.cmd_positions(),
            title=f'Joints command positions{suffix}',
            ylabel='Joint position [rad]',
        )

    def plot_cmd_velocities(
            self,
            times: NDARRAY_V1,
            suffix: str = '',
    ) -> Figure:
        """Plot joints command velocities"""
        return self.plot_generic(
            times=times,
            data=self.cmd_velocities(),
            title=f'Joints command velocities{suffix}',
            ylabel='Joint velocity [rad/s]',
        )

    def plot_cmd_torques(
            self,
            times: NDARRAY_V1,
            suffix: str = '',
    ) -> Figure:
        """Plot joints command torques"""
        return self.plot_generic(
            times=times,
            data=self.cmd_torques(),
            title=f'Joints command torques{suffix}',
            ylabel='Joint torque [Nm]',
        )

    def plot_active_torques(
            self,
            times: NDARRAY_V1,
            suffix: str = '',
    ) -> Figure:
        """Plot joints active torques"""
        return self.plot_generic(
            times=times,
            data=self.active_torques(),
            title=f'Joints active torques{suffix}',
            ylabel='Joint torque [Nm]',
        )

    def plot_spring_torques(
            self,
            times: NDARRAY_V1,
            suffix: str = '',
    ) -> Figure:
        """Plot joints spring torques"""
        return self.plot_generic(
            times=times,
            data=self.spring_torques(),
            title=f'Joints stiffness torques{suffix}',
            ylabel='Joint torque [Nm]',
        )

    def plot_damping_torques(
            self,
            times: NDARRAY_V1,
            suffix: str = '',
    ) -> Figure:
        """Plot joints damping torques"""
        return self.plot_generic(
            times=times,
            data=self.damping_torques(),
            title=f'Joints damping torques{suffix}',
            ylabel='Joint torque [Nm]',
        )

    def plot_friction_torques(
            self,
            times: NDARRAY_V1,
            suffix: str = '',
    ) -> Figure:
        """Plot joints friction torques"""
        return self.plot_generic(
            times=times,
            data=self.friction_torques(),
            title=f'Joints friction torques{suffix}',
            ylabel='Joint torque [Nm]',
        )


class ContactsArray(SensorData, ContactsArrayCy):
    """Sensor array"""

    @classmethod
    def doc(cls):
        """Doc"""
        return _sensor_array_doc(
            cls,
            'contacts',
            array_type=DoubleArray3D,
            # size=sc.contact_size,
            description='Contacts forces, torques, contact position, ...',
        )

    @classmethod
    def from_names(
            cls,
            names: List[List[str]],  # List of collision pairs
            buffer_size: int,
    ):
        """From names"""
        n_sensors = len(names)
        array = np.full(
            shape=[buffer_size, n_sensors, sc.contact_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(array, names)

    @classmethod
    def from_parameters(
            cls,
            buffer_size: int,
            n_contacts: int,
            names: List[str],
    ):
        """From parameters"""
        return cls(
            np.full(
                shape=[buffer_size, n_contacts, sc.contact_size],
                fill_value=0,
                dtype=NPDTYPE,
            ),
            names,
        )

    @classmethod
    def from_size(
            cls,
            n_contacts: int,
            buffer_size: int,
            names: List[str],
    ):
        """From size"""
        contacts = np.full(
            shape=[buffer_size, n_contacts, sc.contact_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(contacts, names)

    def reaction(
            self,
            iteration: int,
            sensor_i: int,
    ) -> NDARRAY_3_D:
        """Reaction force"""
        return self.array[iteration, sensor_i, sc.contact_reaction_x:sc.contact_reaction_z+1]

    def reaction_all(
            self,
            sensor_i: int,
    ) -> NDARRAY_X3_D:
        """Reaction forces"""
        return self.array[:, sensor_i, sc.contact_reaction_x:sc.contact_reaction_z+1]

    def reactions(self) -> NDARRAY_XX3_D:
        """Reaction forces"""
        return self.array[:, :, sc.contact_reaction_x:sc.contact_reaction_z+1]

    def friction(
            self,
            iteration: int,
            sensor_i: int,
    ) -> NDARRAY_3_D:
        """Friction force"""
        return self.array[iteration, sensor_i, sc.contact_friction_x:sc.contact_friction_z+1]

    def friction_all(
            self,
            sensor_i: int,
    ) -> NDARRAY_X3_D:
        """Friction forces"""
        return self.array[:, sensor_i, sc.contact_friction_x:sc.contact_friction_z+1]

    def frictions(self) -> NDARRAY_XX3_D:
        """Friction forces"""
        return self.array[:, :, sc.contact_friction_x:sc.contact_friction_z+1]

    def total(
            self,
            iteration: int,
            sensor_i: int,
    ) -> NDARRAY_3_D:
        """Total force"""
        return self.array[iteration, sensor_i, sc.contact_total_x:sc.contact_total_z+1]

    def total_all(
            self,
            sensor_i: int,
    ) -> NDARRAY_X3_D:
        """Total forces"""
        return self.array[:, sensor_i, sc.contact_total_x:sc.contact_total_z+1]

    def totals(self) -> NDARRAY_XX3_D:
        """Total forces"""
        return self.array[:, :, sc.contact_total_x:sc.contact_total_z+1]

    def position(
            self,
            iteration: int,
            sensor_i: int,
    ) -> NDARRAY_3_D:
        """Position"""
        return self.array[iteration, sensor_i, sc.contact_position_x:sc.contact_position_z+1]

    def position_all(
            self,
            sensor_i: int,
    ) -> NDARRAY_X3_D:
        """Positions"""
        return self.array[:, sensor_i, sc.contact_position_x:sc.contact_position_z+1]

    def plot(
            self,
            times: NDARRAY_V1,
    ) -> Dict:
        """Plot"""
        plots = {}
        plots['reaction_forces'] = self.plot_ground_reaction_forces(times)
        plots.update(self.plot_ground_reaction_forces_all(times))
        plots['friction_forces'] = self.plot_friction_forces(times)
        plots['friction_forces_x'] = self.plot_friction_forces_ori(times, ori=0)
        plots['friction_forces_y'] = self.plot_friction_forces_ori(times, ori=1)
        plots['friction_forces_z'] = self.plot_friction_forces_ori(times, ori=2)
        plots['total_forces'] = self.plot_total_forces(times)
        return plots

    def plot_ground_reaction_forces(
            self,
            times: NDARRAY_V1,
    ) -> Figure:
        """Plot ground reaction forces"""
        fig = plt.figure('Ground reaction forces')
        for sensor_i in range(self.size(1)):
            data = np.asarray(self.reaction_all(sensor_i))
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1)[:len(times)],
                label=f'Leg_{sensor_i}',
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)
        return fig

    def plot_ground_reaction_forces_all(
            self,
            times: NDARRAY_V1,
    ) -> Figure:
        """Plot ground reaction forces"""
        figs = {}
        for sensor_i, name in enumerate(self.names):
            data = np.asarray(self.reaction_all(sensor_i))
            title = f'Ground reaction forces {name}'
            figs[title] = plt.figure(title)
            for direction_i, direction in enumerate(['x', 'y', 'z']):
                plt.plot(
                    times,
                    data[:len(times), direction_i],
                    label=f'{name}_{direction}',
                )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)
        return figs

    def plot_friction_forces(
            self,
            times: NDARRAY_V1,
    ) -> Figure:
        """Plot friction forces"""
        fig = plt.figure('Friction forces')
        for sensor_i in range(self.size(1)):
            data = np.asarray(self.friction_all(sensor_i))
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1)[:len(times)],
                label=f'Leg_{sensor_i}',
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)
        return fig

    def plot_friction_forces_ori(
            self,
            times: NDARRAY_V1,
            ori: int,
    ) -> Figure:
        """Plot friction forces"""
        fig = plt.figure(f'Friction forces (ori={ori})')
        for sensor_i in range(self.size(1)):
            data = np.asarray(self.friction_all(sensor_i))
            plt.plot(
                times,
                data[:len(times), ori],
                label=f'Leg_{sensor_i}',
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)
        return fig

    def plot_total_forces(
            self,
            times: NDARRAY_V1,
    ) -> Figure:
        """Plot contact forces"""
        fig = plt.figure('Contact total forces')
        for sensor_i in range(self.size(1)):
            data = np.asarray(self.total_all(sensor_i))
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1)[:len(times)],
                label=f'Leg_{sensor_i}',
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)
        return fig


class XfrcArray(SensorData, XfrcArrayCy):
    """Xfrc array"""

    @classmethod
    def doc(cls):
        """Doc"""
        return _sensor_array_doc(
            cls,
            'external forces',
            array_type=DoubleArray3D,
            # size=sc.xfrc_size,
            description=(
                'External forces and torques (typically used for e.g.'
                ' perturbations or for implementing custom hydrodynamics) '
            ),
        )

    @classmethod
    def from_names(
            cls,
            names: List[str],
            buffer_size: int,
    ):
        """From names"""
        n_sensors = len(names)
        array = np.full(
            shape=[buffer_size, n_sensors, 6],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(array, names)

    @classmethod
    def from_size(
            cls,
            n_links: int,
            buffer_size: int,
            names: List[str],
    ):
        """From size"""
        xfrc = np.full(
            shape=[buffer_size, n_links, sc.xfrc_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(xfrc, names)

    @classmethod
    def from_parameters(
            cls,
            buffer_size: int,
            n_links: int,
            names: List[str],
    ):
        """From parameters"""
        return cls(
            np.full(
                shape=[buffer_size, n_links, sc.xfrc_size],
                fill_value=0,
                dtype=NPDTYPE,
            ),
            names,
        )

    def force(
            self,
            iteration: int,
            sensor_i: int,
    ) -> NDARRAY_3_D:
        """Force"""
        return self.array[iteration, sensor_i, sc.xfrc_force_x:sc.xfrc_force_z+1]

    def forces(self) -> NDARRAY_XX3_D:
        """Forces"""
        return self.array[:, :, sc.xfrc_force_x:sc.xfrc_force_z+1]

    def set_force(
            self,
            iteration: int,
            sensor_i: int,
            force: NDARRAY_3_D,
    ):
        """Set force"""
        self.array[iteration, sensor_i, sc.xfrc_force_x:sc.xfrc_force_z+1] = force

    def torque(
            self,
            iteration: int,
            sensor_i: int,
    ) -> NDARRAY_3_D:
        """Torque"""
        return self.array[iteration, sensor_i, sc.xfrc_torque_x:sc.xfrc_torque_z+1]

    def torques(self) -> NDARRAY_XX3_D:
        """Torques"""
        return self.array[:, :, sc.xfrc_torque_x:sc.xfrc_torque_z+1]

    def set_torque(
            self,
            iteration: int,
            sensor_i: int,
            torque: NDARRAY_3_D,
    ):
        """Set torque"""
        self.array[iteration, sensor_i, sc.xfrc_torque_x:sc.xfrc_torque_z+1] = torque

    def plot(
            self,
            times: NDARRAY_V1,
    ) -> Dict:
        """Plot"""
        return {
            'forces': self.plot_forces(times),
            'torques': self.plot_torques(times),
        }

    def plot_forces(
            self,
            times: NDARRAY_V1,
    ) -> Figure:
        """Plot"""
        fig = plt.figure('External forces')
        for link_i in range(self.size(1)):
            data = np.asarray(self.forces())[:len(times), link_i]
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1),
                label=f'Link_{link_i}',
            )
        plt.xlabel('Time [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)
        return fig

    def plot_torques(
            self,
            times: NDARRAY_V1,
    ) -> Figure:
        """Plot"""
        fig = plt.figure('External torques')
        for link_i in range(self.size(1)):
            data = np.asarray(self.torques())[:len(times), link_i]
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1),
                label=f'Link_{link_i}',
            )
        plt.xlabel('Time [s]')
        plt.ylabel('Torques [Nm]')
        plt.grid(True)
        return fig


class MusclesArray(SensorData, MusclesArrayCy):
    """Muscles array"""

    @classmethod
    def doc(cls):
        """Doc"""
        return _sensor_array_doc(
            cls,
            'muscles',
            array_type=DoubleArray3D,
            # size=sc.muscle_size,
            description='Muscle length, velocity, force, ...',
        )

    @classmethod
    def from_names(
            cls,
            names: List[str],
            buffer_size: int,
    ):
        """From names"""
        n_sensors = len(names)
        array = np.full(
            shape=[buffer_size, n_sensors, sc.muscle_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(array, names)

    @classmethod
    def from_size(
            cls,
            n_links: int,
            buffer_size: int,
            names: List[str],
    ):
        """From size"""
        muscles = np.full(
            shape=[buffer_size, n_links, sc.muscle_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(muscles, names)

    @classmethod
    def from_parameters(
            cls,
            buffer_size: int,
            n_links: int,
            names: List[str],
    ):
        """From parameters"""
        return cls(
            np.full(
                shape=[buffer_size, n_links, sc.muscle_size],
                fill_value=0,
                dtype=NPDTYPE,
            ),
            names,
        )

    def excitation(
            self,
            iteration: int,
            muscle_i: int,
    ) -> float:
        """ Muscle excitation of a muscle at iteration """
        return self.array[iteration, muscle_i, sc.muscle_excitation]

    def excitations(
            self,
            iteration: int,
    ) -> NDARRAY_V1_D:
        """ Muscle excitations of all muscles at iteration """
        return self.array[iteration, :, sc.muscle_excitation]

    def excitations_all(
            self,
    ) -> NDARRAY_V2_D:
        """ Muscle excitations of all muscles """
        return self.array[:, :, sc.muscle_excitation]

    def activation(
            self,
            iteration: int,
            muscle_i: int,
    ) -> float:
        """ Muscle activation of a muscle at iteration """
        return self.array[iteration, muscle_i, sc.muscle_activation]

    def activations(
            self,
            iteration: int,
    ) -> NDARRAY_V1_D:
        """ Muscle activations of all muscles at iteration """
        return self.array[iteration, :, sc.muscle_activation]

    def activations_all(
            self,
    ) -> NDARRAY_V2_D:
        """ Muscle activations of all muscles """
        return self.array[:, :, sc.muscle_activation]

    def mtu_length(
            self,
            iteration: int,
            muscle_i: int,
    ) -> float:
        """ Muscle tendon unit length of a muscle at iteration """
        return self.array[iteration, muscle_i, sc.muscle_tendon_unit_length]

    def mtu_lengths(
            self,
            iteration: int,
    ) -> NDARRAY_V1_D:
        """ Muscle tendon unit lengths of all muscles at iteration """
        return self.array[iteration, :, sc.muscle_tendon_unit_length]

    def mtu_lengths_all(
            self,
    ) -> NDARRAY_V2_D:
        """ Muscle tendon unit lengths of all muscles """
        return self.array[:, :, sc.muscle_tendon_unit_length]

    def mtu_velocity(
            self,
            iteration: int,
            muscle_i: int,
    ) -> float:
        """ Muscle tendon unit velocity of a muscle at iteration """
        return self.array[iteration, muscle_i, sc.muscle_tendon_unit_velocity]

    def mtu_velocities(
            self,
            iteration: int,
    ) -> NDARRAY_V1_D:
        """ Muscle tendon unit velocities of all muscles at iteration """
        return self.array[iteration, :, sc.muscle_tendon_unit_velocity]

    def mtu_velocities_all(
            self,
    ) -> NDARRAY_V2_D:
        """ Muscle tendon unit velocities of all muscles """
        return self.array[:, :, sc.muscle_tendon_unit_velocity]

    def mtu_force(
            self,
            iteration: int,
            muscle_i: int,
    ) -> float:
        """ Muscle tendon unit force of a muscle at iteration """
        return self.array[iteration, muscle_i, sc.muscle_tendon_unit_force]

    def mtu_forces(
            self,
            iteration: int,
    ) -> NDARRAY_V1_D:
        """ Muscle tendon unit forces of all muscles at iteration """
        return self.array[iteration, :, sc.muscle_tendon_unit_force]

    def mtu_forces_all(
            self,
    ) -> NDARRAY_V2_D:
        """ Muscle tendon unit forces of all muscles """
        return self.array[:, :, sc.muscle_tendon_unit_force]

    def fiber_length(
            self,
            iteration: int,
            muscle_i: int,
    ) -> float:
        """ Muscle fiber length of a muscle at iteration """
        return self.array[iteration, muscle_i, sc.muscle_fiber_length]

    def fiber_lengths(
            self,
            iteration: int,
    ) -> NDARRAY_V1_D:
        """ Muscle fiber lengths of all muscles at iteration """
        return self.array[iteration, :, sc.muscle_fiber_length]

    def fiber_lengths_all(
            self,
    ) -> NDARRAY_V2_D:
        """ Muscle fiber lengths of all muscles """
        return self.array[:, :, sc.muscle_fiber_length]

    def fiber_velocity(
            self,
            iteration: int,
            muscle_i: int,
    ) -> float:
        """ Muscle fiber velocity of a muscle at iteration """
        return self.array[iteration, muscle_i, sc.muscle_fiber_velocity]

    def fiber_velocities(
            self,
            iteration: int,
    ) -> NDARRAY_V1_D:
        """ Muscle fiber velocities of all muscles at iteration """
        return self.array[iteration, :, sc.muscle_fiber_velocity]

    def fiber_velocities_all(
            self,
    ) -> NDARRAY_V2_D:
        """ Muscle fiber velocities of all muscles """
        return self.array[:, :, sc.muscle_fiber_velocity]

    def active_force(
            self,
            iteration: int,
            muscle_i: int,
    ) -> float:
        """ Muscle active force of a muscle at iteration """
        return self.array[iteration, muscle_i, sc.muscle_active_force]

    def active_forces(
            self,
            iteration: int,
    ) -> NDARRAY_V1_D:
        """ Muscle active forces of all muscles at iteration """
        return self.array[iteration, :, sc.muscle_active_force]

    def active_forces_all(
            self,
    ) -> NDARRAY_V2_D:
        """ Muscle active forces of all muscles """
        return self.array[:, :, sc.muscle_active_force]

    def passive_force(
            self,
            iteration: int,
            muscle_i: int,
    ) -> float:
        """ Muscle passive force of a muscle at iteration """
        return self.array[iteration, muscle_i, sc.muscle_passive_force]

    def passive_forces(
            self,
            iteration: int,
    ) -> NDARRAY_V1_D:
        """ Muscle passive forces of all muscles at iteration """
        return self.array[iteration, :, sc.muscle_passive_force]

    def passive_forces_all(
            self,
    ) -> NDARRAY_V2_D:
        """ Muscle passive forces of all muscles """
        return self.array[:, :, sc.muscle_passive_force]

    def tendon_length(
            self,
            iteration: int,
            muscle_i: int,
    ) -> float:
        """ Tendon unit length of a muscle at iteration """
        return self.array[iteration, muscle_i, sc.muscle_tendon_length]

    def tendon_lengths(
            self,
            iteration: int,
    ) -> NDARRAY_V1_D:
        """ Tendon unit lengths of all muscles at iteration """
        return self.array[iteration, :, sc.muscle_tendon_length]

    def tendon_lengths_all(
            self,
    ) -> NDARRAY_V2_D:
        """ Tendon unit lengths of all muscles """
        return self.array[:, :, sc.muscle_tendon_length]

    def tendon_force(
            self,
            iteration: int,
            muscle_i: int,
    ) -> float:
        """ Tendon unit force of a muscle at iteration """
        return self.array[iteration, muscle_i, sc.muscle_tendon_force]

    def tendon_forces(
            self,
            iteration: int,
    ) -> NDARRAY_V1_D:
        """ Tendon unit forces of all muscles at iteration """
        return self.array[iteration, :, sc.muscle_tendon_force]

    def tendon_forces_all(
            self,
    ) -> NDARRAY_V2_D:
        """ Tendon unit forces of all muscles """
        return self.array[:, :, sc.muscle_tendon_force]

    def Ia_feedback(
            self,
            iteration: int,
            muscle_i: int,
    ) -> float:
        """ Type Ia feedback  of a muscle at iteration """
        return self.array[iteration, muscle_i, sc.muscle_Ia_feedback]

    def Ia_feedbacks(
            self,
            iteration: int,
    ) -> NDARRAY_V1_D:
        """ Type Ia feedback of all muscles at iteration """
        return self.array[iteration, :, sc.muscle_Ia_feedback]

    def Ia_feedbacks_all(
            self,
    ) -> NDARRAY_V2_D:
        """ Type Ia feedback of all muscles """
        return self.array[:, :, sc.muscle_Ia_feedback]

    def II_feedback(
            self,
            iteration: int,
            muscle_i: int,
    ) -> float:
        """ Type II feedback  of a muscle at iteration """
        return self.array[iteration, muscle_i, sc.muscle_II_feedback]

    def II_feedbacks(
            self,
            iteration: int,
    ) -> NDARRAY_V1_D:
        """ Type II feedback of all muscles at iteration """
        return self.array[iteration, :, sc.muscle_II_feedback]

    def II_feedbacks_all(
            self,
    ) -> NDARRAY_V2_D:
        """ Type II feedback of all muscles """
        return self.array[:, :, sc.muscle_II_feedback]

    def Ib_feedback(
            self,
            iteration: int,
            muscle_i: int,
    ) -> float:
        """ Type Ib feedback  of a muscle at iteration """
        return self.array[iteration, muscle_i, sc.muscle_Ib_feedback]

    def Ib_feedbacks(
            self,
            iteration: int,
    ) -> NDARRAY_V1_D:
        """ Type Ib feedback of all muscles at iteration """
        return self.array[iteration, :, sc.muscle_Ib_feedback]

    def Ib_feedbacks_all(
            self,
    ) -> NDARRAY_V2_D:
        """ Type Ib feedback of all muscles """
        return self.array[:, :, sc.muscle_Ib_feedback]
