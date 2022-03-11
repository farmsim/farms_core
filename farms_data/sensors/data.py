"""Animat data"""

from typing import List, Dict, Any
import numpy as np
from nptyping import NDArray
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import circmean
from ..array.array import to_array
from ..array.array_cy import DoubleArray3D
from ..utils.transform import quat2euler
from .sensor_convention import sc
from .data_cy import (
    SensorsDataCy,
    LinkSensorArrayCy,
    JointSensorArrayCy,
    ContactsArrayCy,
    HydrodynamicsArrayCy
)


NPDTYPE = np.float64
NPUITYPE = np.uintc


class SensorData(DoubleArray3D):
    """SensorData"""

    def __init__(
            self,
            array: NDArray[(Any, Any, Any), float],
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
    """SensorsData"""

    @classmethod
    def from_dict(
            cls,
            dictionary: Dict,
    ):
        """Load data from dictionary"""
        return cls(
            links=LinkSensorArray.from_dict(
                dictionary['links']
            ),
            joints=JointSensorArray.from_dict(
                dictionary['joints']
            ),
            contacts=ContactsArray.from_dict(
                dictionary['contacts']
            ),
            hydrodynamics=HydrodynamicsArray.from_dict(
                dictionary['hydrodynamics']
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
                ['hydrodynamics', self.hydrodynamics],
            ]
            if data is not None
        }

    def plot(
            self,
            times: NDArray[(Any,), float],
    ) -> Dict:
        """Plot"""
        plots = {}
        plots.update(self.links.plot(times))
        plots.update(self.joints.plot(times))
        plots.update(self.contacts.plot(times))
        plots.update(self.hydrodynamics.plot(times))
        return plots


class LinkSensorArray(SensorData, LinkSensorArrayCy):
    """Links array"""

    def __init__(
            self,
            array: NDArray[(Any, Any, sc.link_size), float],
            names: List[str],
    ):
        super().__init__(array, names)
        self.masses = None

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
            n_iterations: int,
    ):
        """From names"""
        n_sensors = len(names)
        array = np.full(
            shape=[n_iterations, n_sensors, sc.link_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(array, names)

    @classmethod
    def from_size(
            cls, n_links: int,
            n_iterations: int,
            names: List[str],
    ):
        """From size"""
        links = np.full(
            shape=[n_iterations, n_links, sc.link_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(links, names)

    @classmethod
    def from_parameters(
            cls,
            n_iterations: int,
            n_links: int,
            names: List[str],
    ):
        """From parameters"""
        return cls(
            np.full(
                shape=[n_iterations, n_links, sc.link_size],
                fill_value=0,
                dtype=NPDTYPE,
            ),
            names,
        )

    def com_position(
            self,
            iteration: int,
            link_i: int,
    ) -> NDArray[(3,), float]:
        """CoM position of a link"""
        return self.array[iteration, link_i, sc.link_com_position_x:sc.link_com_position_z+1]

    def com_orientation(
            self,
            iteration: int,
            link_i: int,
    ) -> NDArray[(4,), float]:
        """CoM orientation of a link"""
        return self.array[iteration, link_i, sc.link_com_orientation_x:sc.link_com_orientation_w+1]

    def urdf_position(
            self,
            iteration: int,
            link_i: int,
    ) -> NDArray[(3,), float]:
        """URDF position of a link"""
        return self.array[iteration, link_i, sc.link_urdf_position_x:sc.link_urdf_position_z+1]

    def urdf_positions(self) -> NDArray[(Any, Any, 3), float]:
        """URDF position of a link"""
        return self.array[:, :, sc.link_urdf_position_x:sc.link_urdf_position_z+1]

    def urdf_orientation(
            self,
            iteration: int,
            link_i: int,
    ) -> NDArray[(4,), float]:
        """URDF orientation of a link"""
        return self.array[iteration, link_i, sc.link_urdf_orientation_x:sc.link_urdf_orientation_w+1]

    def urdf_orientations(self) -> NDArray[(Any, Any, 4), float]:
        """URDF orientation of a link"""
        return self.array[:, :, sc.link_urdf_orientation_x:sc.link_urdf_orientation_w+1]

    def com_lin_velocity(
            self,
            iteration: int,
            link_i: int,
    ) -> NDArray[(3,), float]:
        """CoM linear velocity of a link"""
        return self.array[iteration, link_i, sc.link_com_velocity_lin_x:sc.link_com_velocity_lin_z+1]

    def com_lin_velocities(self) -> NDArray[(Any, Any, 3,), float]:
        """CoM linear velocities"""
        return self.array[:, :, sc.link_com_velocity_lin_x:sc.link_com_velocity_lin_z+1]

    def com_ang_velocity(
            self,
            iteration: int,
            link_i: int,
    ) -> NDArray[(3,), float]:
        """CoM angular velocity of a link"""
        return self.array[iteration, link_i, sc.link_com_velocity_ang_x:sc.link_com_ang_lin_z+1]

    def heading(
            self,
            iteration: int,
            indices: List[int],
    ) -> float:
        """Heading"""
        n_indices = len(indices)
        link_orientation = np.zeros(n_indices)
        ori = np.zeros(3)
        for link_idx in indices:
            quat2euler(
                quat=self.urdf_orientation(
                    iteration=iteration,
                    link_i=link_idx,
                ),
                out=ori,
            )
            link_orientation[link_idx] = ori[2]
        return circmean(
            samples=link_orientation,
            low=-np.pi,
            high=np.pi,
        ) if n_indices > 1 else link_orientation[0]

    def plot(
            self,
            times: NDArray[(Any,), float],
    ) -> Dict:
        """Plot"""
        return {
            'base_position': self.plot_base_position(times, xaxis=0, yaxis=1),
            'base_velocity': self.plot_base_velocity(times),
            'heading': self.plot_heading(times),
        }

    def plot_base_position(
            self,
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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
    def from_names(
            cls,
            names: List[str],
            n_iterations: int,
    ):
        """From names"""
        n_sensors = len(names)
        array = np.full(
            shape=[n_iterations, n_sensors, sc.joint_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(array, names)

    @classmethod
    def from_size(
            cls,
            n_joints: int,
            n_iterations: int,
            names: List[str],
    ):
        """From size"""
        joints = np.full(
            shape=[n_iterations, n_joints, sc.joint_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(joints, names)

    @classmethod
    def from_parameters(
            cls,
            n_iterations: int,
            n_joints: int,
            names: List[str],
    ):
        """From parameters"""
        return cls(
            np.full(
                shape=[n_iterations, n_joints, sc.joint_size],
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
    ) -> NDArray[(Any,), float]:
        """Joints positions"""
        return self.array[iteration, :, sc.joint_position]

    def positions_all(self) -> NDArray[(Any, Any), float]:
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
    ) -> NDArray[(Any,), float]:
        """Joints velocities"""
        return self.array[iteration, :, sc.joint_velocity]

    def velocities_all(self) -> NDArray[(Any, Any), float]:
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
    ) -> NDArray[(Any,), float]:
        """Joint torques"""
        return self.array[iteration, :, sc.joint_torque]

    def motor_torques_all(self) -> NDArray[(Any, Any), float]:
        """Joint torque"""
        return self.array[:, :, sc.joint_torque]

    def force(
            self,
            iteration: int,
            joint_i: int,
    ) -> NDArray[(3,), float]:
        """Joint force"""
        return self.array[iteration, joint_i, sc.joint_force_x:sc.joint_force_z+1]

    def forces_all(self) -> NDArray[(Any, Any, 3), float]:
        """Joints forces"""
        return self.array[:, :, sc.joint_force_x:sc.joint_force_z+1]

    def torque(
            self,
            iteration: int,
            joint_i: int,
    ) -> NDArray[(3,), float]:
        """Joint torque"""
        return self.array[iteration, joint_i, sc.joint_torque_x:sc.joint_torque_z+1]

    def torques_all(self) -> NDArray[(Any, Any, 3), float]:
        """Joints torques"""
        return self.array[:, :, sc.joint_torque_x:sc.joint_torque_z+1]

    def cmd_position(
            self,
            iteration: int,
            joint_i: int,
    ) -> float:
        """Joint position"""
        return self.array[iteration, joint_i, sc.joint_cmd_position]

    def cmd_positions(self) -> NDArray[(Any, Any), float]:
        """Joint position"""
        return self.array[:, :, sc.joint_cmd_position]

    def cmd_velocity(
            self,
            iteration: int,
            joint_i: int,
    ) -> float:
        """Joint velocity"""
        return self.array[iteration, joint_i, sc.joint_cmd_velocity]

    def cmd_velocities(self) -> NDArray[(Any, Any), float]:
        """Joint velocity"""
        return self.array[:, :, sc.joint_cmd_velocity]

    def cmd_torque(
            self,
            iteration: int,
            joint_i: int,
    ) -> float:
        """Joint torque"""
        return self.array[iteration, joint_i, sc.joint_cmd_torque]

    def cmd_torques(self) -> NDArray[(Any, Any), float]:
        """Joint torque"""
        return self.array[:, :, sc.joint_cmd_torque]

    def active(
            self,
            iteration: int,
            joint_i: int,
    ) -> float:
        """Active torque"""
        return self.array[iteration, joint_i, sc.joint_torque_active]

    def active_torques(self) -> NDArray[(Any, Any), float]:
        """Active torques"""
        return self.array[:, :, sc.joint_torque_active]

    def spring(
            self,
            iteration: int,
            joint_i: int,
    ) -> float:
        """Passive spring torque"""
        return self.array[iteration, joint_i, sc.joint_torque_stiffness]

    def spring_torques(self) -> NDArray[(Any, Any), float]:
        """Spring torques"""
        return self.array[:, :, sc.joint_torque_stiffness]

    def damping(
            self,
            iteration: int,
            joint_i: int,
    ) -> float:
        """passive damping torque"""
        return self.array[iteration, joint_i, sc.joint_torque_damping]

    def damping_torques(self) -> NDArray[(Any, Any), float]:
        """Damping torques"""
        return self.array[:, :, sc.joint_torque_damping]

    def friction(
            self,
            iteration: int,
            joint_i: int,
    ) -> float:
        """passive friction torque"""
        return self.array[iteration, joint_i, sc.joint_torque_friction]

    def friction_torques(self) -> NDArray[(Any, Any), float]:
        """Friction torques"""
        return self.array[:, :, sc.joint_torque_friction]

    def plot(
            self,
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
            data: NDArray[(Any, Any), float],
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
            times: NDArray[(Any,), float],
            data: NDArray[(Any, Any), float],
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
            times: NDArray[(Any,), float],
            data: NDArray[(Any, Any), float],
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
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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
    def from_names(
            cls,
            names: List[str],
            n_iterations: int,
    ):
        """From names"""
        n_sensors = len(names)
        array = np.full(
            shape=[n_iterations, n_sensors, sc.contact_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(array, names)

    @classmethod
    def from_parameters(
            cls,
            n_iterations: int,
            n_contacts: int,
            names: List[str],
    ):
        """From parameters"""
        return cls(
            np.full(
                shape=[n_iterations, n_contacts, sc.contact_size],
                fill_value=0,
                dtype=NPDTYPE,
            ),
            names,
        )

    @classmethod
    def from_size(
            cls,
            n_contacts: int,
            n_iterations: int,
            names: List[str],
    ):
        """From size"""
        contacts = np.full(
            shape=[n_iterations, n_contacts, sc.contact_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(contacts, names)

    def reaction(
            self,
            iteration: int,
            sensor_i: int,
    ) -> NDArray[(3,), float]:
        """Reaction force"""
        return self.array[iteration, sensor_i, sc.contact_reaction_x:sc.contact_reaction_z+1]

    def reaction_all(
            self,
            sensor_i: int,
    ) -> NDArray[(Any, 3,), float]:
        """Reaction forces"""
        return self.array[:, sensor_i, sc.contact_reaction_x:sc.contact_reaction_z+1]

    def reactions(self) -> NDArray[(Any, Any, 3,), float]:
        """Reaction forces"""
        return self.array[:, :, sc.contact_reaction_x:sc.contact_reaction_z+1]

    def friction(
            self,
            iteration: int,
            sensor_i: int,
    ) -> NDArray[(3,), float]:
        """Friction force"""
        return self.array[iteration, sensor_i, sc.contact_friction_x:sc.contact_friction_z+1]

    def friction_all(
            self,
            sensor_i: int,
    ) -> NDArray[(Any, 3,), float]:
        """Friction forces"""
        return self.array[:, sensor_i, sc.contact_friction_x:sc.contact_friction_z+1]

    def frictions(self) -> NDArray[(Any, Any, 3,), float]:
        """Friction forces"""
        return self.array[:, :, sc.contact_friction_x:sc.contact_friction_z+1]

    def total(
            self,
            iteration: int,
            sensor_i: int,
    ) -> NDArray[(3,), float]:
        """Total force"""
        return self.array[iteration, sensor_i, sc.contact_total_x:sc.contact_total_z+1]

    def total_all(
            self,
            sensor_i: int,
    ) -> NDArray[(Any, 3,), float]:
        """Total forces"""
        return self.array[:, sensor_i, sc.contact_total_x:sc.contact_total_z+1]

    def totals(self) -> NDArray[(Any, Any, 3,), float]:
        """Total forces"""
        return self.array[:, :, sc.contact_total_x:sc.contact_total_z+1]

    def position(
            self,
            iteration: int,
            sensor_i: int,
    ) -> NDArray[(3,), float]:
        """Position"""
        return self.array[iteration, sensor_i, sc.contact_position_x:sc.contact_position_z+1]

    def position_all(
            self,
            sensor_i: int,
    ) -> NDArray[(Any, 3,), float]:
        """Positions"""
        return self.array[:, sensor_i, sc.contact_position_x:sc.contact_position_z+1]

    def plot(
            self,
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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
            times: NDArray[(Any,), float],
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


class HydrodynamicsArray(SensorData, HydrodynamicsArrayCy):
    """Hydrodynamics array"""

    @classmethod
    def from_names(
            cls,
            names: List[str],
            n_iterations: int,
    ):
        """From names"""
        n_sensors = len(names)
        array = np.full(
            shape=[n_iterations, n_sensors, 6],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(array, names)

    @classmethod
    def from_size(
            cls,
            n_links: int,
            n_iterations: int,
            names: List[str],
    ):
        """From size"""
        hydrodynamics = np.full(
            shape=[n_iterations, n_links, sc.hydrodynamics_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(hydrodynamics, names)

    @classmethod
    def from_parameters(
            cls,
            n_iterations: int,
            n_links: int,
            names: List[str],
    ):
        """From parameters"""
        return cls(
            np.full(
                shape=[n_iterations, n_links, sc.hydrodynamics_size],
                fill_value=0,
                dtype=NPDTYPE,
            ),
            names,
        )

    def force(
            self,
            iteration: int,
            sensor_i: int,
    ) -> NDArray[(3,), float]:
        """Force"""
        return self.array[iteration, sensor_i, sc.hydrodynamics_force_x:sc.hydrodynamics_force_z+1]

    def forces(self) -> NDArray[(Any, Any, 3,), float]:
        """Forces"""
        return self.array[:, :, sc.hydrodynamics_force_x:sc.hydrodynamics_force_z+1]

    def set_force(
            self,
            iteration: int,
            sensor_i: int,
            force: NDArray[(3,), float],
    ):
        """Set force"""
        self.array[iteration, sensor_i, sc.hydrodynamics_force_x:sc.hydrodynamics_force_z+1] = force

    def torque(
            self,
            iteration: int,
            sensor_i: int,
    ) -> NDArray[(3,), float]:
        """Torque"""
        return self.array[iteration, sensor_i, sc.hydrodynamics_torque_x:sc.hydrodynamics_torque_z+1]

    def torques(self) -> NDArray[(Any, Any, 3,), float]:
        """Torques"""
        return self.array[:, :, sc.hydrodynamics_torque_x:sc.hydrodynamics_torque_z+1]

    def set_torque(
            self,
            iteration: int,
            sensor_i: int,
            torque: NDArray[(3,), float],
    ):
        """Set torque"""
        self.array[iteration, sensor_i, sc.hydrodynamics_torque_x:sc.hydrodynamics_torque_z+1] = torque

    def plot(
            self,
            times: NDArray[(Any,), float],
    ) -> Dict:
        """Plot"""
        return {
            'forces': self.plot_forces(times),
            'torques': self.plot_torques(times),
        }

    def plot_forces(
            self,
            times: NDArray[(Any,), float],
    ) -> Figure:
        """Plot"""
        fig = plt.figure('Hydrodynamic forces')
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
            times: NDArray[(Any,), float],
    ) -> Figure:
        """Plot"""
        fig = plt.figure('Hydrodynamic torques')
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
