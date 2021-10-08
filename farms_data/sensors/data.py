"""Animat data"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circmean
from scipy.spatial.transform import Rotation
from .array import DoubleArray3D
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


def to_array(array, iteration=None):
    """To array or None"""
    if array is not None:
        array = np.array(array)
        if iteration is not None:
            array = array[:iteration]
    return array


class SensorData(DoubleArray3D):
    """SensorData"""

    def __init__(self, array, names):
        super().__init__(array)
        self.names = names

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        return cls(
            array=dictionary['array'],
            names=dictionary['names'],
        )

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        return {
            'array': to_array(self.array, iteration),
            'names': self.names,
        }


class SensorsData(SensorsDataCy):
    """SensorsData"""

    @classmethod
    def from_dict(cls, dictionary):
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

    def to_dict(self, iteration=None):
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

    def plot(self, times):
        """Plot"""
        plots = {}
        plots.update(self.links.plot(times))
        plots.update(self.joints.plot(times))
        plots.update(self.contacts.plot(times))
        plots.update(self.hydrodynamics.plot(times))
        return plots


class ContactsArray(SensorData, ContactsArrayCy):
    """Sensor array"""

    @classmethod
    def from_names(cls, names, n_iterations):
        """From names"""
        n_sensors = len(names)
        array = np.full(
            shape=[n_iterations, n_sensors, sc.contact_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(array, names)

    @classmethod
    def from_parameters(cls, n_iterations, n_contacts, names):
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
    def from_size(cls, n_contacts, n_iterations, names):
        """From size"""
        contacts = np.full(
            shape=[n_iterations, n_contacts, sc.contact_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(contacts, names)

    def reaction(self, iteration, sensor_i):
        """Reaction force"""
        return self.array[iteration, sensor_i, sc.contact_reaction_x:sc.contact_reaction_z+1]

    def reaction_all(self, sensor_i):
        """Reaction forces"""
        return self.array[:, sensor_i, sc.contact_reaction_x:sc.contact_reaction_z+1]

    def reactions(self):
        """Reaction forces"""
        return self.array[:, :, sc.contact_reaction_x:sc.contact_reaction_z+1]

    def friction(self, iteration, sensor_i):
        """Friction force"""
        return self.array[iteration, sensor_i, sc.contact_friction_x:sc.contact_friction_z+1]

    def friction_all(self, sensor_i):
        """Friction forces"""
        return self.array[:, sensor_i, sc.contact_friction_x:sc.contact_friction_z+1]

    def frictions(self):
        """Friction forces"""
        return self.array[:, :, sc.contact_friction_x:sc.contact_friction_z+1]

    def total(self, iteration, sensor_i):
        """Total force"""
        return self.array[iteration, sensor_i, sc.contact_total_x:sc.contact_total_z+1]

    def total_all(self, sensor_i):
        """Total forces"""
        return self.array[:, sensor_i, sc.contact_total_x:sc.contact_total_z+1]

    def totals(self):
        """Total forces"""
        return self.array[:, :, sc.contact_total_x:sc.contact_total_z+1]

    def position(self, iteration, sensor_i):
        """Position"""
        return self.array[iteration, sensor_i, sc.contact_position_x:sc.contact_position_z+1]

    def position_all(self, sensor_i):
        """Positions"""
        return self.array[:, sensor_i, sc.contact_position_x:sc.contact_position_z+1]

    def plot(self, times):
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

    def plot_ground_reaction_forces(self, times):
        """Plot ground reaction forces"""
        fig = plt.figure('Ground reaction forces')
        for sensor_i in range(self.size(1)):
            data = np.asarray(self.reaction_all(sensor_i))
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1)[:len(times)],
                label='Leg_{}'.format(sensor_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)
        return fig

    def plot_ground_reaction_forces_all(self, times):
        """Plot ground reaction forces"""
        figs = {}
        for sensor_i, name in enumerate(self.names):
            data = np.asarray(self.reaction_all(sensor_i))
            title = 'Ground reaction forces {}'.format(name)
            figs[title] = plt.figure(title)
            for direction_i, direction in enumerate(['x', 'y', 'z']):
                plt.plot(
                    times,
                    data[:len(times), direction_i],
                    label='{}_{}'.format(name, direction),
                )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)
        return figs

    def plot_friction_forces(self, times):
        """Plot friction forces"""
        fig = plt.figure('Friction forces')
        for sensor_i in range(self.size(1)):
            data = np.asarray(self.friction_all(sensor_i))
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1)[:len(times)],
                label='Leg_{}'.format(sensor_i),
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)
        return fig

    def plot_friction_forces_ori(self, times, ori):
        """Plot friction forces"""
        fig = plt.figure('Friction forces (ori={})'.format(ori))
        for sensor_i in range(self.size(1)):
            data = np.asarray(self.friction_all(sensor_i))
            plt.plot(
                times,
                data[:len(times), ori],
                label='Leg_{}'.format(sensor_i),
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)
        return fig

    def plot_total_forces(self, times):
        """Plot contact forces"""
        fig = plt.figure('Contact total forces')
        for sensor_i in range(self.size(1)):
            data = np.asarray(self.total_all(sensor_i))
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1)[:len(times)],
                label='Leg_{}'.format(sensor_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)
        return fig


class JointSensorArray(SensorData, JointSensorArrayCy):
    """Joint sensor array"""

    @classmethod
    def from_names(cls, names, n_iterations):
        """From names"""
        n_sensors = len(names)
        array = np.full(
            shape=[n_iterations, n_sensors, sc.joint_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(array, names)

    @classmethod
    def from_size(cls, n_joints, n_iterations, names):
        """From size"""
        joints = np.full(
            shape=[n_iterations, n_joints, sc.joint_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(joints, names)

    @classmethod
    def from_parameters(cls, n_iterations, n_joints, names):
        """From parameters"""
        return cls(
            np.full(
                shape=[n_iterations, n_joints, sc.joint_size],
                fill_value=0,
                dtype=NPDTYPE,
            ),
            names,
        )

    def position(self, iteration, joint_i):
        """Joint position"""
        return self.array[iteration, joint_i, sc.joint_position]

    def positions(self, iteration):
        """Joints positions"""
        return self.array[iteration, :, sc.joint_position]

    def positions_all(self):
        """Joints positions"""
        return self.array[:, :, sc.joint_position]

    def velocity(self, iteration, joint_i):
        """Joint velocity"""
        return self.array[iteration, joint_i, sc.joint_velocity]

    def velocities(self, iteration):
        """Joints velocities"""
        return self.array[iteration, :, sc.joint_velocity]

    def velocities_all(self):
        """Joints velocities"""
        return self.array[:, :, sc.joint_velocity]

    def force(self, iteration, joint_i):
        """Joint force"""
        return self.array[iteration, joint_i, sc.joint_force_x:sc.joint_force_z+1]

    def forces_all(self):
        """Joints forces"""
        return self.array[:, :, sc.joint_force_x:sc.joint_force_z+1]

    def torque(self, iteration, joint_i):
        """Joint torque"""
        return self.array[iteration, joint_i, sc.joint_torque_x:sc.joint_torque_z+1]

    def torques_all(self):
        """Joints torques"""
        return self.array[:, :, sc.joint_torque_x:sc.joint_torque_z+1]

    def motor_torque(self, iteration, joint_i):
        """Joint velocity"""
        return self.array[iteration, joint_i, sc.joint_torque]

    def motor_torques(self):
        """Joint velocity"""
        return self.array[:, :, sc.joint_torque]

    def active(self, iteration, joint_i):
        """Active torque"""
        return self.array[iteration, joint_i, sc.joint_torque_active]

    def active_torques(self):
        """Active torques"""
        return self.array[:, :, sc.joint_torque_active]

    def spring(self, iteration, joint_i):
        """Passive spring torque"""
        return self.array[iteration, joint_i, sc.joint_torque_stiffness]

    def spring_torques(self):
        """Spring torques"""
        return self.array[:, :, sc.joint_torque_stiffness]

    def damping(self, iteration, joint_i):
        """passive damping torque"""
        return self.array[iteration, joint_i, sc.joint_torque_damping]

    def damping_torques(self):
        """Damping torques"""
        return self.array[:, :, sc.joint_torque_damping]

    def plot(self, times):
        """Plot"""
        t_init = times[:50]
        return {
            'joints_positions': self.plot_positions(times),
            'joints_velocities': self.plot_velocities(times),
            'joints_velocities_init': self.plot_velocities(t_init, ' init'),
            'joints_forces': self.plot_forces(times),
            'joints_torques': self.plot_torques(times),
            'joints_motor_torques': self.plot_motor_torques(times),
            'joints_active_torques': self.plot_active_torques(times),
            'joints_spring_torques': self.plot_spring_torques(times),
            'joints_damping_torques': self.plot_damping_torques(times),
            'joints_ti_motor': self.plot_motor_torques(t_init, ' init'),
            'joints_ti_active': self.plot_active_torques(t_init, ' init'),
            'joints_ti_spring': self.plot_spring_torques(t_init, ' init'),
            'joints_ti_damping': self.plot_damping_torques(t_init, ' init'),
        }

    def plot_positions(self, times, suffix=''):
        """Plot ground reaction forces"""
        fig = plt.figure(f'Joints positions{suffix}')
        for joint_i in range(self.size(1)):
            plt.plot(
                times,
                np.asarray(self.positions_all())[:len(times), joint_i],
                label='Joint_{}'.format(joint_i),
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint position [rad]')
        plt.grid(True)
        return fig

    def plot_velocities(self, times, suffix=''):
        """Plot ground reaction forces"""
        fig = plt.figure(f'Joints velocities{suffix}')
        for joint_i in range(self.size(1)):
            plt.plot(
                times,
                np.asarray(self.velocities_all())[:len(times), joint_i],
                label='Joint_{}'.format(joint_i),
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint velocity [rad/s]')
        plt.grid(True)
        return fig

    def plot_forces(self, times, suffix=''):
        """Plot ground reaction forces"""
        fig = plt.figure(f'Joints forces{suffix}')
        for joint_i in range(self.size(1)):
            data = np.linalg.norm(np.asarray(self.forces_all()), axis=-1)
            plt.plot(
                times,
                data[:len(times), joint_i],
                label='Joint_{}'.format(joint_i),
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint force [N]')
        plt.grid(True)
        return fig

    def plot_torques(self, times, suffix=''):
        """Plot ground reaction torques"""
        fig = plt.figure(f'Joints torques{suffix}')
        for joint_i in range(self.size(1)):
            data = np.linalg.norm(np.asarray(self.torques_all()), axis=-1)
            plt.plot(
                times,
                data[:len(times), joint_i],
                label='Joint_{}'.format(joint_i),
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint torque [Nm]')
        plt.grid(True)
        return fig

    def plot_motor_torques(self, times, suffix=''):
        """Plot ground reaction forces"""
        fig = plt.figure(f'Joints motor torques{suffix}')
        for joint_i in range(self.size(1)):
            plt.plot(
                times,
                np.asarray(self.motor_torques())[:len(times), joint_i],
                label='Joint_{}'.format(joint_i),
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint torque [Nm]')
        plt.grid(True)
        return fig

    def plot_active_torques(self, times, suffix=''):
        """Plot joints active torques"""
        fig = plt.figure(f'Joints active torques{suffix}')
        for joint_i in range(self.size(1)):
            plt.plot(
                times,
                np.asarray(self.active_torques())[:len(times), joint_i],
                label='Joint_{}'.format(joint_i),
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint torque [Nm]')
        plt.grid(True)
        return fig

    def plot_spring_torques(self, times, suffix=''):
        """Plot joints spring torques"""
        fig = plt.figure(f'Joints spring torques{suffix}')
        for joint_i in range(self.size(1)):
            plt.plot(
                times,
                np.asarray(self.spring_torques())[:len(times), joint_i],
                label='Joint_{}'.format(joint_i)
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint torque [Nm]')
        plt.grid(True)
        return fig

    def plot_damping_torques(self, times, suffix=''):
        """Plot joints damping torques"""
        fig = plt.figure(f'Joints damping torques{suffix}')
        for joint_i in range(self.size(1)):
            plt.plot(
                times,
                np.asarray(self.damping_torques())[:len(times), joint_i],
                label='Joint_{}'.format(joint_i),
            )
        plt.legend()
        plt.xlabel('Times [s]')
        plt.ylabel('Joint torque [Nm]')
        plt.grid(True)
        return fig


class LinkSensorArray(SensorData, LinkSensorArrayCy):
    """Links array"""

    def __init__(self, array, names):
        super().__init__(array, names)
        self.masses = None

    @classmethod
    def from_dict(cls, dictionary):
        """Load data from dictionary"""
        links = super(cls, cls).from_dict(dictionary)
        links.masses = dictionary['masses']
        return links

    def to_dict(self, iteration=None):
        """Convert data to dictionary"""
        links = super().to_dict(iteration=iteration)
        links['masses'] = self.masses
        return links

    @classmethod
    def from_names(cls, names, n_iterations):
        """From names"""
        n_sensors = len(names)
        array = np.full(
            shape=[n_iterations, n_sensors, sc.link_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(array, names)

    @classmethod
    def from_size(cls, n_links, n_iterations, names):
        """From size"""
        links = np.full(
            shape=[n_iterations, n_links, sc.link_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(links, names)

    @classmethod
    def from_parameters(cls, n_iterations, n_links, names):
        """From parameters"""
        return cls(
            np.full(
                shape=[n_iterations, n_links, sc.link_size],
                fill_value=0,
                dtype=NPDTYPE,
            ),
            names,
        )

    def com_position(self, iteration, link_i):
        """CoM position of a link"""
        return self.array[iteration, link_i, sc.link_com_position_x:sc.link_com_position_z+1]

    def com_orientation(self, iteration, link_i):
        """CoM orientation of a link"""
        return self.array[iteration, link_i, sc.link_com_orientation_x:sc.link_com_orientation_w+1]

    def urdf_position(self, iteration, link_i):
        """URDF position of a link"""
        return self.array[iteration, link_i, sc.link_urdf_position_x:sc.link_urdf_position_z+1]

    def urdf_positions(self):
        """URDF position of a link"""
        return self.array[:, :, sc.link_urdf_position_x:sc.link_urdf_position_z+1]

    def urdf_orientation(self, iteration, link_i):
        """URDF orientation of a link"""
        return self.array[iteration, link_i, sc.link_urdf_orientation_x:sc.link_urdf_orientation_w+1]

    def urdf_orientations(self):
        """URDF orientation of a link"""
        return self.array[:, :, sc.link_urdf_orientation_x:sc.link_urdf_orientation_w+1]

    def com_lin_velocity(self, iteration, link_i):
        """CoM linear velocity of a link"""
        return self.array[iteration, link_i, sc.link_com_velocity_lin_x:sc.link_com_velocity_lin_z+1]

    def com_lin_velocities(self):
        """CoM linear velocities"""
        return self.array[:, :, sc.link_com_velocity_lin_x:sc.link_com_velocity_lin_z+1]

    def com_ang_velocity(self, iteration, link_i):
        """CoM angular velocity of a link"""
        return self.array[iteration, link_i, sc.link_com_velocity_ang_x:sc.link_com_ang_lin_z+1]

    def heading(self, iteration, indices):
        """Heading"""
        n_indices = len(indices)
        link_orientation = np.zeros(n_indices)
        for link_idx in indices:
            link_orientation[link_idx] = Rotation.from_quat(
                self.urdf_orientation(iteration=iteration, link_i=link_idx)
            ).as_euler('xyz')[2]
        return circmean(
            samples=link_orientation,
            low=-np.pi,
            high=np.pi,
        ) if n_indices > 1 else link_orientation[0]

    def plot(self, times):
        """Plot"""
        return {
            'base_position': self.plot_base_position(times, xaxis=0, yaxis=1),
            'base_velocity': self.plot_base_velocity(times),
            'heading': self.plot_heading(times),
        }

    def plot_base_position(self, times, xaxis=0, yaxis=1):
        """Plot"""
        fig = plt.figure('Links position')
        for link_i in range(self.size(1)):
            data = np.asarray(self.urdf_positions())[:len(times), link_i]
            plt.plot(
                data[:, xaxis],
                data[:, yaxis],
                label='Link_{}'.format(link_i),
            )
        plt.legend()
        plt.xlabel('Position [m]')
        plt.ylabel('Position [m]')
        plt.axis('equal')
        plt.grid(True)
        return fig

    def plot_base_velocity(self, times):
        """Plot"""
        fig = plt.figure('Links velocities')
        for link_i in range(self.size(1)):
            data = np.asarray(self.com_lin_velocities())[:len(times), link_i]
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1),
                label='Link_{}'.format(link_i),
            )
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Velocity [m/s]')
        plt.grid(True)
        return fig

    def plot_heading(self, times, indices=None):
        """Plot"""
        if indices is None:
            indices = [0]
        fig = plt.figure('Heading')
        plt.plot(times, [self.heading(i, indices) for i, _ in enumerate(times)])
        plt.legend()
        plt.xlabel('Time [s]')
        plt.ylabel('Heading [rad]')
        plt.grid(True)
        return fig


class HydrodynamicsArray(SensorData, HydrodynamicsArrayCy):
    """Hydrodynamics array"""

    @classmethod
    def from_names(cls, names, n_iterations):
        """From names"""
        n_sensors = len(names)
        array = np.full(
            shape=[n_iterations, n_sensors, 6],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(array, names)

    @classmethod
    def from_size(cls, n_links, n_iterations, names):
        """From size"""
        hydrodynamics = np.full(
            shape=[n_iterations, n_links, sc.hydrodynamics_size],
            fill_value=0,
            dtype=NPDTYPE,
        )
        return cls(hydrodynamics, names)

    @classmethod
    def from_parameters(cls, n_iterations, n_links, names):
        """From parameters"""
        return cls(
            np.full(
                shape=[n_iterations, n_links, sc.hydrodynamics_size],
                fill_value=0,
                dtype=NPDTYPE,
            ),
            names,
        )

    def force(self, iteration, sensor_i):
        """Force"""
        return self.array[iteration, sensor_i, sc.hydrodynamics_force_x:sc.hydrodynamics_force_z+1]

    def forces(self):
        """Forces"""
        return self.array[:, :, sc.hydrodynamics_force_x:sc.hydrodynamics_force_z+1]

    def set_force(self, iteration, sensor_i, force):
        """Set force"""
        self.array[iteration, sensor_i, sc.hydrodynamics_force_x:sc.hydrodynamics_force_z+1] = force

    def torque(self, iteration, sensor_i):
        """Torque"""
        return self.array[iteration, sensor_i, sc.hydrodynamics_torque_x:sc.hydrodynamics_torque_z+1]

    def torques(self):
        """Torques"""
        return self.array[:, :, sc.hydrodynamics_torque_x:sc.hydrodynamics_torque_z+1]

    def set_torque(self, iteration, sensor_i, torque):
        """Set torque"""
        self.array[iteration, sensor_i, sc.hydrodynamics_torque_x:sc.hydrodynamics_torque_z+1] = torque

    def plot(self, times):
        """Plot"""
        return {
            'forces': self.plot_forces(times),
            'torques': self.plot_torques(times),
        }

    def plot_forces(self, times):
        """Plot"""
        fig = plt.figure('Hydrodynamic forces')
        for link_i in range(self.size(1)):
            data = np.asarray(self.forces())[:len(times), link_i]
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1),
                label='Link_{}'.format(link_i),
            )
        plt.xlabel('Time [s]')
        plt.ylabel('Forces [N]')
        plt.grid(True)
        return fig

    def plot_torques(self, times):
        """Plot"""
        fig = plt.figure('Hydrodynamic torques')
        for link_i in range(self.size(1)):
            data = np.asarray(self.torques())[:len(times), link_i]
            plt.plot(
                times,
                np.linalg.norm(data, axis=-1),
                label='Link_{}'.format(link_i),
            )
        plt.xlabel('Time [s]')
        plt.ylabel('Torques [Nm]')
        plt.grid(True)
        return fig
