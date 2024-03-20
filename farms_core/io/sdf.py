"""Farms SDF"""

import numbers
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
from collections import defaultdict
from copy import deepcopy

import numpy as np
import trimesh as tri
from scipy.spatial.transform import Rotation

from .. import pylog
from ..options import Options
from ..units import SimulationUnitScaling


def get_floats_from_text(text, split=' '):
    """Get floats from text"""
    return [float(val) for val in text.split(split)]


def get_pose_from_xml(data):
    """Get pose"""
    return (
        get_floats_from_text(data.find('pose').text)
        if data.find('pose') is not None
        else (
            (
                get_floats_from_text(data.find('origin').attrib['xyz'])
                if 'xyz' in data.find('origin').attrib
                else [0, 0, 0]
            ) + (
                get_floats_from_text(data.find('origin').attrib['rpy'])
                if 'rpy' in data.find('origin').attrib
                else [0, 0, 0]
            )
        )
        if data.find('origin') is not None
        else np.zeros(6)
    )


def get_inertia_tensor_from_vector(inertia_vector: list) -> np.ndarray:
    """Get the inertia tensor from  the inertia vector of six elements"""
    inertia_vector = np.asarray(inertia_vector)
    inertia_tensor = np.identity(3)
    inertia_tensor[(0, 1, 2), (0, 1, 2)] = inertia_vector[[0, 3, 5]]
    inertia_tensor[(0, 0, 1), (1, 1, 2)] = inertia_vector[[1, 2, 4]]
    inertia_tensor[(1, 2, 2), (0, 0, 1)] = inertia_vector[[1, 2, 4]]
    return inertia_tensor


def get_inertia_vector_from_tensor(inertia_tensor: np.ndarray) -> list:
    """Get the inertia vector of six elements from the inertia tensor"""
    inertia_vector = np.ones((6,))
    inertia_vector[:] = inertia_tensor[
        (0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)
    ]
    return inertia_vector


def get_homogenous_matrix_from_pose(pose):
    homogenous = np.identity(4)
    homogenous[:3, -1] = np.array(pose[:3])
    homogenous[:3, :3] = Rotation.from_euler('xyz', pose[3:]).as_matrix()
    return homogenous


def get_pose_from_homogenous_matrix(homogenous_matrix):
    rot_mat = Rotation.from_matrix(homogenous_matrix[:3, :3]).as_euler("xyz")
    positions = homogenous_matrix[:3, -1]
    pose = [*positions.tolist(), *rot_mat.tolist()]
    return pose


def transfer_elem_in_link_a_to_link_b(link_a, elem, link_b):
    # Construct homogenous matrices
    link_a_homogenous = get_homogenous_matrix_from_pose(link_a.pose)
    link_b_homogenous = get_homogenous_matrix_from_pose(link_b.pose)
    elem_homogenous = get_homogenous_matrix_from_pose(
        elem.pose if elem is not None else np.zeros(6)
    )
    # Compute new pose in link_b
    transform = (
        np.linalg.inv(link_b_homogenous) @ link_a_homogenous @ elem_homogenous
    )
    return transform


def get_parenting(links, joints):
    """Get parenting"""
    parenting = {link: [] for link in links}
    for joint in joints.values():
        parenting[joint.parent].append(joint.child)
    return parenting


def create_tree_recursive(tree, parenting):
    """Tree recursive"""
    for parent, children in tree.items():
        for child in parenting[parent]:
            children[child] = {}
        create_tree_recursive(tree[parent], parenting)


def convert_poses_urdf2sdf_recursive(tree, pose, links, joint_parenting):
    """Convert poses urdf2sdf recursive"""
    for child, children in tree.items():

        # Joint
        joint = joint_parenting[child] if child in joint_parenting else None
        if joint:
            joint_pose = joint.pose
            joint.pose = [0, 0, 0, 0, 0, 0]
        else:
            joint_pose = [0, 0, 0, 0, 0, 0]
        # print('Pose of joint {}: {}'.format(joint.name, joint.pose))

        # Link pose
        link = links[child]
        rot_parent, rot_child = [
            Rotation.from_euler('xyz', ori).as_matrix()
            for ori in [pose[3:], joint_pose[3:]]
        ]
        loc = np.array(pose[:3]) + rot_parent @ np.array(joint_pose[:3])
        ori = Rotation.from_matrix(rot_parent @ rot_child).as_euler('xyz')
        child_pose = loc.tolist() + ori.tolist()
        link.pose = child_pose
        for i, visual in enumerate(link.visuals):
            visual.name = '{}_visual{}'.format(
                link.name,
                '_{}'.format(i+1) if i > 0 else '',
            )
        for i, collision in enumerate(link.collisions):
            collision.name = '{}_collision{}'.format(
                link.name,
                '_{}'.format(i+1) if i > 0 else '',
            )
        # print('Pose of link {}: {}'.format(child, pose))

        # Recursive
        convert_poses_urdf2sdf_recursive(
            tree=children,
            pose=np.copy(child_pose),
            links=links,
            joint_parenting=joint_parenting,
        )


def create_tree(robot):
    """Create tree"""
    links = {link.name: link for link in robot.links}
    joints = {joint.name: joint for joint in robot.joints}
    child_links = [joint.child for joint in joints.values()]
    base_link = [link for link in links.keys() if link not in child_links]
    assert len(base_link) == 1
    base_link = base_link[0]
    parenting = get_parenting(links, joints)
    tree = {base_link: {}}
    create_tree_recursive(tree, parenting)
    return tree, links, joints


def convert_urdf2sdf(robot):
    """Convert URDF2SDF"""
    tree, links, joints = create_tree(robot)
    base_link = links[list(tree.keys())[0]]
    pose = base_link.pose
    joint_parenting = {joint.child: joint for name, joint in joints.items()}
    convert_poses_urdf2sdf_recursive(tree, pose, links, joint_parenting)
    return tree, links, joints


def compute_child_inertial_in_parent(parent_link, child_link):
    """ Compute the combined inertial representation of links
    in parent frame using Steiner's formula
    """

    # Initialize inertial elements
    com = np.zeros((3,))
    inertia = np.zeros((3, 3))
    parent_inertia = get_inertia_tensor_from_vector(
        parent_link.inertial.inertias
    ) if parent_link.inertial is not None else np.zeros((3, 3))
    child_inertia = get_inertia_tensor_from_vector(
        child_link.inertial.inertias
    ) if child_link.inertial is not None else np.zeros((3, 3))

    # Compute transforms
    parent_inertial_transform = get_homogenous_matrix_from_pose(
        parent_link.inertial.pose
        if parent_link.inertial is not None
        else np.zeros(6)
    )
    parent_transform = transfer_elem_in_link_a_to_link_b(
        parent_link, parent_link.inertial, parent_link,
    )
    child_parent_transform = transfer_elem_in_link_a_to_link_b(
        child_link, child_link.inertial, parent_link,
    )

    # Total mass of the combined link
    parent_mass = parent_link.inertial.mass if parent_link.inertial else 0
    child_mass = child_link.inertial.mass if child_link.inertial else 0
    total_mass = parent_mass + child_mass
    if total_mass < 1e-12:
        return Inertial.empty()
    # New center of mass of the combined links
    com += parent_mass*parent_inertial_transform[:3, -1]
    com += child_mass*child_parent_transform[:3, -1]
    com /= total_mass

    # Inertia
    # Parallel axis theorem / Huygens–Steiner theorem / Steiner's theorem
    for local_inertia, transform, mass in (
            (parent_inertia, parent_transform, parent_mass),
            (child_inertia, child_parent_transform, child_mass),
    ):
        r = (transform[:3, -1] - com)
        inertia += (
            transform[:3, :3]@local_inertia@transform[:3, :3].T + mass*(
                np.inner(r, r)*np.eye(3) - np.outer(r, r)
            )
        )

    # Setup inertial pose
    inertial_rotations = np.array([0, 0, 0])
    inertial_pose = np.concatenate([com, inertial_rotations]).tolist()
    # Create final combined inertial element
    inertial = Inertial(
        inertial_pose, total_mass,
        get_inertia_vector_from_tensor(inertia)
    )
    # Check inertia
    # assert np.all(np.abs(inertia - inertia.T) < 1e-10), "Non-symmetric interia tensor"
    # assert np.all(np.linalg.eig(inertia)[0] > 0.0), "Negative diagonal elements"
    return inertial


# TODO: This could be moved under ModelSDF class
def remove_link(model, link_name):
    """ Remove link in the model

    Parameters
    ----------
    model: <ModelSDF>
        Model SDF
    link_name: <str>
        Name of the link to remove from the model.

    """
    for index, link in enumerate(model.links):
        if link.name == link_name:
            model.links.pop(index)
            break


def remove_joint(model, joint_name):
    """ Remove joint in the model

    Parameters
    ----------
    model: <ModelSDF>
        Model SDF
    joint_name: <str>
        Name of the joint to remove from the model.

    """
    for index, joint in enumerate(model.joints):
        if joint.name == joint_name:
            model.joints.pop(index)
            break


def create_parent_tree(model):
    """ Parenting tree """
    link_child_links = defaultdict(list)
    link_child_joints = defaultdict(list)
    for index, joint in enumerate(model.joints):
        link_child_links[joint.parent].append(joint.child)
        link_child_joints[joint.parent].append(joint.name)
    return link_child_links, link_child_joints


def merge_fixed_joints_recursive(
        model, link_child_links, link_child_joints, link
):
    """ Merge fixed joints in the model

    Parameters
    ----------
    model: <ModelSDF>
        SDF model
    link_child_links: <dict>
        Link and associated child links
    link_child_joints: <dict>
        Link and associated child joints
    link: <Link>
        Link to start the recursion from
    """

    for index, joint_name in enumerate(link_child_joints[link.name]):
        child_joint = model.get_joint(joint_name)
        if child_joint.type == "fixed":
            pylog.debug("Found fixed joint %s", child_joint.name)
            # Merge this joint
            merge_link = model.get_link(child_joint.child)
            # Merge Collisions
            for collision in merge_link.collisions:
                merge_collision = deepcopy(collision)
                merge_collision.pose = (
                    get_pose_from_homogenous_matrix(
                        transfer_elem_in_link_a_to_link_b(
                            merge_link, collision, link
                        )
                    )
                )
                link.collisions.append(merge_collision)
            # Merge Visuals
            for visual in merge_link.visuals:
                merge_visual = deepcopy(visual)
                merge_visual.pose = (
                    get_pose_from_homogenous_matrix(
                        transfer_elem_in_link_a_to_link_b(
                            merge_link, visual, link
                        )
                    )
                )
                link.visuals.append(merge_visual)
            # Merge Inertials
            link.inertial = compute_child_inertial_in_parent(
                link, merge_link
            )
            # Loop and update all the grand child joints
            for gindex in range(len(link_child_joints[merge_link.name])):
                gjoint = model.get_joint(
                    link_child_joints[merge_link.name][gindex]
                )
                glink = model.get_link(
                    link_child_links[merge_link.name][gindex]
                )
                gjoint.parent = link.name
                # changes
                link_child_joints[link.name].append(gjoint.name)
                link_child_links[link.name].append(glink.name)
            # delete the merged joint and link
            pylog.debug(f"Deleting : {merge_link.name}")
            remove_link(model, merge_link.name)
            pylog.debug(f"Deleting : {child_joint.name}")
            remove_joint(model, child_joint.name)
        else:
            # Recursive
            merge_fixed_joints_recursive(
                model, link_child_links, link_child_joints,
                model.get_link(child_joint.child)
            )


def merge_fixed_joints(model):
    """ Method to merge fixed links in the model.

    Parameters
    ----------
    model: <ModelSDF>
        SDF model

    Returns
    -------
    return: <None>
        The method operates directly on the given model
    """
    # Get the base link of the model
    base_link = model.get_base_link()
    # Create parent-tree
    child_links, child_joints = create_parent_tree(model)
    # Recursively merge and remove fixed joints
    merge_fixed_joints_recursive(
        model, child_links, child_joints, base_link
    )


def replace_file_name_in_path(file_path, new_name):
    """Replace a file name in a given path. File extension is retained

    Parameters
    ----------
    file_path : <str>
        Path to the file object
    new_name : <str>
        Name to replace the original file name

    Returns
    -------
    out : <str>
        New path with the replaced file name

    """
    full_path = os.path.split(file_path)[0]
    file_extension = (os.path.split(file_path)[-1]).split('.')[-1]
    new_name = new_name + '.' + file_extension
    return os.path.join(full_path, new_name)


class ModelSDF(Options):
    """Farms SDF"""

    def __init__(self, name, pose, **kwargs):
        super(ModelSDF, self).__init__()
        self.name = name
        self.pose = pose
        self.links = kwargs.pop('links', [])
        self.joints = kwargs.pop('joints', [])
        self.directory = kwargs.pop('directory', '')
        self.units = kwargs.pop('units', SimulationUnitScaling())
        assert not kwargs, kwargs

    def validate(self):
        """Validate model"""
        links_names = [link.name for link in self.links]
        assert len(list(set(links_names))) == len(links_names), (
            'There are links repetitions'
        )
        for link in self.links:
            assert isinstance(link, Link)
            assert len(link.pose) == 6
            for shape in link.visuals + link.collisions:
                if isinstance(shape.geometry, Mesh):
                    path = os.path.isfile(os.path.join(
                        self.directory,
                        shape.geometry.uri,
                    ))
                    assert path, '{} is not a file'.format(path)
            if link.inertial:
                assert len(link.inertial.pose) == 6
                assert len(link.inertial.inertias) == 6
                assert isinstance(link.inertial.mass, numbers.Number)
        for joint in self.joints:
            assert isinstance(joint, Joint)
            assert joint.parent in links_names, 'Parent {} not in links'
            assert joint.child in links_names, 'Child {} not in links'
            assert len(joint.pose) == 6
            if joint.type == 'fixed':
                assert joint.axis is None
            elif joint.type == 'continuous':
                assert joint.axis is not None
                assert len(joint.axis.xyz) == 3
            else:
                raise Exception
        return True

    def xml(self, use_world=True):
        """xml"""
        sdf = ET.Element('sdf', version='1.6')
        if use_world:
            world = ET.SubElement(sdf, 'world', name='world')
            model = ET.SubElement(world, 'model', name=self.name)
        else:
            model = ET.SubElement(sdf, 'model', name=self.name)
        if self.pose is not None:
            pose = ET.SubElement(model, 'pose')
            pose.text = ' '.join([
                str(element*(self.units.meters if i < 3 else 1))
                for i, element in enumerate(self.pose)
            ])
        for link in self.links:
            link.xml(model)
        for joint in self.joints:
            joint.xml(model)
        return sdf

    def xml_str(self):
        """xml string"""
        sdf = self.xml()
        xml_str = ET.tostring(
            sdf,
            encoding='utf8',
            method='xml'
        ).decode('utf8')
        # dom = xml.dom.minidom.parse(xml_fname)
        dom = xml.dom.minidom.parseString(xml_str)
        return dom.toprettyxml(indent=2*' ')

    def write(self, filename='animat.sdf'):
        """Write SDF to file"""
        # ET.ElementTree(self.xml()).write(filename)
        with open(filename, 'w+') as sdf_file:
            sdf_file.write(self.xml_str())

    @classmethod
    def create_model(cls, data, directory=''):
        """ Create ModelSDF from parsed sdf model data. """
        pose = get_pose_from_xml(data)
        links, joints = [], []
        for link in data.findall('link'):
            links.append(Link.from_xml(link))
        for joint in data.findall('joint'):
            joints.append(Joint.from_xml(joint))
        return cls(
            name=data.attrib['name'],
            pose=pose,
            links=links,
            joints=joints,
            directory=directory,
            units=SimulationUnitScaling(),
        )

    @classmethod
    def read(cls, filename):
        """ Read from an SDF FILE. """
        directory = os.path.dirname(filename)
        tree = ET.parse(os.path.expandvars(filename))
        root = tree.getroot()
        world = root.find('world') if root.find('world') else root
        return [
            cls.create_model(model, directory=directory)
            for model in world.findall('model')
        ]

    @classmethod
    def from_urdf(cls, filename):
        """Read from URDF file"""
        tree = ET.parse(os.path.expandvars(filename))
        root = tree.getroot()
        robot_xml = root.find('robot') if root.find('robot') else root
        robot = cls.create_model(robot_xml)
        convert_urdf2sdf(robot)
        return robot

    def change_units(self, units):
        """ Change the units of all elements in the model. """
        self.units = units
        #: Change units of links
        for link in self.links:
            link.units = units
            link.inertial.units = units
            for collision in link.collisions:
                collision.units = units
                collision.geometry.units = units
            for visual in link.visuals:
                visual.units = units
                visual.geometry.units = units
        #: Change units of joints
        # for joint in self.joints:
        #     pass

    def get_link(self, link_name: str):
        """ Get link object in the model

        Parameters
        ----------
        link_name: <str>
            Name of the link to find in the model.

        Returns
        -------
        link: <Link>
            Link object with name = link_name
            None if no object with link_name found

        """
        for link in self.links:
            if link.name == link_name:
                return link
        return None

    def get_link_index(self, link_name: str):
        """ Get link index in the model

        Parameters
        ----------
        link_name: <str>
            Name of the link to find in the model.

        Returns
        -------
        index: <int>
            Link index with name = link_name
            None if no object with link_name found

        """
        for index, link in enumerate(self.links):
            if link.name == link_name:
                return index
        return None

    def get_joint(self, joint_name):
        """ Get joint object in the model

        Parameters
        ----------
        joint_name: <str>
            Name of the joint to find in the model.

        Returns
        -------
        joint: <Link>
            Link object with name = joint_name
            None if no object with joint_name found

        """
        for joint in self.joints:
            if joint.name == joint_name:
                return joint
        return None

    def get_joint_index(self, joint_name: str):
        """ Get joint index in the model

        Parameters
        ----------
        joint_name: <str>
            Name of the joint to find in the model.

        Returns
        -------
        index: <int>
            Joint index with name = joint_name
            None if no object with joint_name found

        """
        for index, joint in enumerate(self.joints):
            if joint.name == joint_name:
                return index
        return None

    def get_base_links(self):
        """ Find the base link in the model

        Parameters
        ----------
        model: <ModelSDF>
            SDF model object

        Returns
        -------
        base_links: <list>
            List of names of base links in the model

        """
        child_links = set(joint.child for joint in self.joints)
        parent_links = set(
            [link.name for link in self.links]
            + [joint.parent for joint in self.joints]
        )
        return [
            self.get_link(link_name)
            for link_name in list(parent_links - child_links)
        ]

    def get_base_link(self):
        """ Find the base link in the model

        Parameters
        ----------
        model: <ModelSDF>
            SDF model object

        Returns
        -------
        base_link: <str>
            Name of the base link in the model

        """
        base_links = self.get_base_links()
        assert len(base_links) == 1, (
            f'None or more than one base link found in {self.name}'
        )
        return base_links[0]

    def get_parent_joint(self, link):
        """Get parent"""
        for joint in self.joints:
            if joint.child == link.name:
                return joint
        return None

    def get_parent(self, link):
        """Get parent"""
        joint = self.get_parent_joint(link=link)
        return self.get_link(joint.parent) if joint is not None else None

    def get_children(self, link):
        """Get children"""
        return [
            self.get_link(joint.child)
            for joint in self.joints
            if joint.parent == link.name
        ]

    def mass(self):
        """Mass"""
        mass = 0
        for link in self.links:
            if link.inertial is not None:
                mass += link.inertial.mass
        return mass


class Link(Options):
    """Link"""

    def __init__(self, name, pose, **kwargs):
        super(Link, self).__init__()
        self.name = name
        self.pose = pose
        self.inertial = kwargs.pop('inertial', None)
        self.collisions = kwargs.pop('collisions', [])
        self.visuals = kwargs.pop('visuals', [])
        self.units = kwargs.pop('units', SimulationUnitScaling())
        assert not kwargs, kwargs

    @classmethod
    def empty(cls, name, pose, **kwargs):
        """Empty"""
        units = kwargs.pop('units', SimulationUnitScaling())
        return cls(
            name,
            pose=pose,
            inertial=kwargs.pop('inertial', Inertial.empty(units)),
            collisions=[],
            visuals=[],
            units=units,
        )

    @classmethod
    def plane(cls, name, pose, **kwargs):
        """Plane"""
        visual_kwargs = {}
        if 'color' in kwargs:
            visual_kwargs['color'] = kwargs.pop('color', None)
        # inertial_pose = kwargs.pop('inertial_pose', np.zeros(6))
        shape_pose = kwargs.pop('shape_pose', np.zeros(6))
        units = kwargs.pop('units', SimulationUnitScaling())
        return cls(
            name,
            pose=pose,
            collisions=[Collision.plane(
                name,
                pose=shape_pose,
                units=units,
                **kwargs
            )],
            visuals=[Visual.plane(
                name,
                pose=shape_pose,
                units=units,
                **visual_kwargs,
                **kwargs
            )],
            units=units
        )

    @classmethod
    def box(cls, name, pose, **kwargs):
        """Box"""
        inertial_kwargs, visual_kwargs = {}, {}
        if 'color' in kwargs:
            visual_kwargs['color'] = kwargs.pop('color', None)
        for element in ['mass', 'inertias', 'density']:
            if element in kwargs:
                inertial_kwargs[element] = kwargs.pop(element, None)
        # inertial_pose = kwargs.pop('inertial_pose', np.zeros(6))
        shape_pose = kwargs.pop('shape_pose', np.zeros(6))
        units = kwargs.pop('units', SimulationUnitScaling())
        return cls(
            name,
            pose=pose,
            inertial=kwargs.pop('inertial', Inertial.box(
                pose=shape_pose,
                units=units,
                **kwargs,
                **inertial_kwargs,
            )),
            collisions=[Collision.box(
                name,
                pose=shape_pose,
                units=units,
                **kwargs
            )],
            visuals=[Visual.box(
                name,
                pose=shape_pose,
                units=units,
                **visual_kwargs,
                **kwargs
            )],
            units=units
        )

    @classmethod
    def sphere(cls, name, pose, **kwargs):
        """Sphere"""
        visual_kwargs = {}
        if 'color' in kwargs:
            visual_kwargs['color'] = kwargs.pop('color', None)
        # inertial_pose = kwargs.pop('inertial_pose', np.zeros(6))
        shape_pose = kwargs.pop('shape_pose', np.zeros(6))
        units = kwargs.pop('units', SimulationUnitScaling())
        return cls(
            name,
            pose=pose,
            inertial=kwargs.pop('inertial', Inertial.sphere(
                pose=shape_pose,
                units=units,
                **kwargs,
            )),
            collisions=[Collision.sphere(
                name,
                pose=shape_pose,
                units=units,
                **kwargs
            )],
            visuals=[Visual.sphere(
                name,
                pose=shape_pose,
                units=units,
                **visual_kwargs,
                **kwargs,
            )],
            units=units
        )

    @classmethod
    def cylinder(cls, name, pose, **kwargs):
        """Cylinder"""
        visual_kwargs = {}
        if 'color' in kwargs:
            visual_kwargs['color'] = kwargs.pop('color', None)
        # inertial_pose = kwargs.pop('inertial_pose', np.zeros(6))
        shape_pose = kwargs.pop('shape_pose', np.zeros(6))
        units = kwargs.pop('units', SimulationUnitScaling())
        return cls(
            name,
            pose=pose,
            inertial=kwargs.pop('inertial', Inertial.cylinder(
                pose=shape_pose,
                units=units,
                **kwargs
            )),
            collisions=[Collision.cylinder(
                name,
                pose=shape_pose,
                units=units,
                **kwargs
            )],
            visuals=[Visual.cylinder(
                name,
                pose=shape_pose,
                units=units,
                **visual_kwargs,
                **kwargs
            )],
            units=units
        )

    @classmethod
    def capsule(cls, name, pose, **kwargs):
        """Capsule"""
        visual_kwargs = {}
        if 'color' in kwargs:
            visual_kwargs['color'] = kwargs.pop('color', None)
        # inertial_pose = kwargs.pop('inertial_pose', np.zeros(6))
        shape_pose = kwargs.pop('shape_pose', np.zeros(6))
        units = kwargs.pop('units', SimulationUnitScaling())
        return cls(
            name,
            pose=pose,
            inertial=kwargs.pop('inertial', Inertial.capsule(
                pose=shape_pose,
                units=units,
                **kwargs
            )),
            collisions=[Collision.capsule(
                name,
                pose=shape_pose,
                units=units,
                **kwargs
            )],
            visuals=[Visual.capsule(
                name,
                pose=shape_pose,
                units=units,
                **visual_kwargs,
                **kwargs
            )],
            units=units
        )

    @classmethod
    def from_mesh(cls, name, mesh, pose, **kwargs):
        """From mesh"""
        inertial_kwargs = {}
        for element in ['mass', 'density']:
            if element in kwargs:
                inertial_kwargs[element] = kwargs.pop(element)
        visual_kwargs = {}
        if 'color' in kwargs:
            visual_kwargs['color'] = kwargs.pop('color', None)
        # inertial_pose = kwargs.pop('inertial_pose', np.zeros(6))
        inertial_from_bounding = kwargs.pop('inertial_from_bounding', False)
        shape_pose = kwargs.pop('shape_pose', np.zeros(6))
        scale = kwargs.pop('scale', 1)
        compute_inertial = kwargs.pop('compute_inertial', True)
        units = kwargs.pop('units', SimulationUnitScaling())
        inertial = kwargs.pop('inertial', Inertial(
            pose=np.zeros(6),
            mass=0,
            inertias=np.zeros(6),
            units=units,
        ))
        assert not kwargs, kwargs
        return cls(
            name,
            pose=pose,
            inertial=(
                Inertial.from_mesh(
                    mesh,
                    pose=shape_pose,
                    scale=scale,
                    units=units,
                    mesh_bounding_box=inertial_from_bounding,
                    **inertial_kwargs,
                ) if compute_inertial else inertial
            ),
            collisions=[Collision.from_mesh(
                name,
                mesh,
                pose=shape_pose,
                scale=(scale*np.ones(3)).tolist(),
                units=units
            )],
            visuals=[Visual.from_mesh(
                name,
                mesh,
                pose=shape_pose,
                scale=(scale*np.ones(3)).tolist(),
                units=units,
                **visual_kwargs
            )],
            units=units
        )

    @classmethod
    def heightmap(cls, name, uri, pose, **kwargs):
        """Heightmap"""
        visual_kwargs = {}
        if 'color' in kwargs:
            visual_kwargs['color'] = kwargs.pop('color', None)
        shape_pose = kwargs.pop('shape_pose', np.zeros(6))
        size = kwargs.pop('size', np.ones(3))
        pos = kwargs.pop('pos', np.zeros(3))
        units = kwargs.pop('units', SimulationUnitScaling())
        assert not kwargs, kwargs
        return cls(
            name,
            pose=pose,
            inertial=None,
            collisions=[Collision.heightmap(
                name,
                uri,
                pose=shape_pose,
                size=size,
                pos=pos,
                units=units
            )],
            visuals=[Visual.heightmap(
                name,
                uri,
                pose=shape_pose,
                size=size,
                pos=pos,
                units=units,
                **visual_kwargs
            )],
            units=units
        )

    def xml(self, model):
        """xml"""
        link = ET.SubElement(model, 'link', name=self.name)
        if self.pose is not None:
            pose = ET.SubElement(link, 'pose')
            pose.text = ' '.join([
                str(element*(self.units.meters if i < 3 else 1))
                for i, element in enumerate(self.pose)
            ])
        if self.inertial is not None:
            self.inertial.xml(link)
        for collision in self.collisions:
            collision.xml(link)
        for visual in self.visuals:
            visual.xml(link)

    @classmethod
    def from_xml(cls, data):
        """ Create link object from parsed xml data. """
        pose = get_pose_from_xml(data)
        return cls(
            data.attrib['name'],
            pose=pose,
            inertial=(
                Inertial.from_xml(data.find('inertial'))
                if data.find('inertial') is not None
                else None
            ),
            collisions=(
                [
                    Collision.from_xml(collision)
                    for collision in data.findall('collision')
                ]
                if data.find('collision') is not None
                else []
            ),
            visuals=(
                [
                    Visual.from_xml(visual)
                    for visual in data.findall('visual')
                ]
                if data.find('visual') is not None
                else []
            ),
            units=SimulationUnitScaling()
        )


class Inertial(Options):
    """Inertial"""

    def __init__(self, pose, mass, inertias, units=SimulationUnitScaling()):
        super(Inertial, self).__init__()
        self.mass = mass
        self.inertias = inertias
        self.units = units
        self.pose = pose

    @classmethod
    def empty(cls, units=SimulationUnitScaling()):
        """Empty"""
        return cls(
            pose=[0]*6,
            mass=0,
            inertias=[0]*6,
            units=units
        )

    @classmethod
    def box(cls, size, pose, **kwargs):
        """Box"""
        density = kwargs.pop('density', 1000)
        volume = size[0]*size[1]*size[2]
        mass = kwargs.pop('mass', volume*density)
        units = kwargs.pop('units', SimulationUnitScaling())
        return cls(
            pose=np.asarray(pose),
            mass=mass,
            inertias=[
                1/12*mass*(size[1]**2 + size[2]**2),
                0,
                0,
                1/12*mass*(size[0]**2 + size[2]**2),
                0,
                1/12*mass*(size[0]**2 + size[1]**2)
            ],
            units=units
        )

    @classmethod
    def sphere(cls, radius, pose, **kwargs):
        """Sphere"""
        density = kwargs.pop('density', 1000)
        volume = 4/3*np.pi*radius**3
        mass = volume*density
        units = kwargs.pop('units', SimulationUnitScaling())
        return cls(
            pose=np.asarray(pose),
            mass=mass,
            inertias=[
                2/5*mass*radius**2,
                0,
                0,
                2/5*mass*radius**2,
                0,
                2/5*mass*radius**2
            ],
            units=units
        )

    @classmethod
    def cylinder(cls, length, radius, pose, **kwargs):
        """Cylinder"""
        density = kwargs.pop('density', 1000)
        volume = np.pi*radius**2*length
        mass = volume*density
        units = kwargs.pop('units', SimulationUnitScaling())
        return cls(
            pose=np.asarray(pose),
            mass=mass,
            inertias=[
                1/12*mass*(3*radius**2 + length**2),
                0,
                0,
                1/12*mass*(3*radius**2 + length**2),
                0,
                1/2*mass*(radius**2)
            ],
            units=units
        )

    @classmethod
    def capsule(cls, length, radius, pose, **kwargs):
        """Capsule"""
        density = kwargs.pop('density', 1000)
        volume_sphere = 4/3*np.pi*radius**3
        volume_cylinder = np.pi*radius**2*length
        volume = volume_sphere + volume_cylinder
        mass = volume*density
        units = kwargs.pop('units', SimulationUnitScaling())
        return cls(
            pose=np.asarray(pose),
            mass=mass,
            # TODO: This is Cylinder inertia!!
            inertias=[
                1/12*mass*(3*radius**2 + length**2),
                0,
                0,
                1/12*mass*(3*radius**2 + length**2),
                0,
                1/2*mass*(radius**2)
            ],
            units=units
        )

    @classmethod
    def from_mesh(cls, path, pose, scale=1, **kwargs):
        """From mesh"""

        # Kwargs
        _from_bounding_box = kwargs.pop('mesh_bounding_box', False)
        units = kwargs.pop('units', SimulationUnitScaling())
        kwargs_mass = kwargs.pop('mass', None)
        density = kwargs.pop('density', 1000)

        # Setup inertial elements
        mass = 0
        inertia = np.zeros([3, 3])

        # Mesh
        original_mesh = tri.load_mesh(path)

        # Bounding box
        if _from_bounding_box:
            original_mesh = original_mesh.bounding_box

        # Meshes
        meshes = (
            [
                mesh1
                for mesh0 in original_mesh.geometry.values()
                for mesh1 in (mesh0.split() if mesh0.split() else [mesh0])
            ]
            if isinstance(original_mesh, tri.Scene)
            else original_mesh.split()
            if any(original_mesh.split())
            else [original_mesh]
            # else [original_mesh]
        )
        assert any(meshes), (
            'No mesh found in {} (Orinal mesh: {}, split: {})'.format(
                path,
                original_mesh,
                original_mesh.split(),
            )
        )

        # Apply transform
        for mesh in meshes:
            mesh.apply_transform(tri.transformations.scale_matrix(scale))

        # Mass
        total_mass = np.sum([mesh.mass for mesh in meshes])
        for mesh in meshes:
            if kwargs_mass is not None:
                mesh.density *= kwargs_mass/total_mass
            else:
                mesh.density = density
            mass += mesh.mass

        # Centre of mass
        total_mass = np.sum([mesh.mass for mesh in meshes])
        com = np.sum([
            mesh.mass*mesh.center_mass
            for mesh in meshes
        ], axis=0)/total_mass

        # Inertia
        for mesh in meshes:
            element_inertia = mesh.moment_inertia
            if not Inertial.valid_inertia(element_inertia):
                if np.sum(np.abs(element_inertia)) > 1e-8:
                    raise ValueError(
                        f'Composite of {path} has inappropriate inertia:'
                        f'\n{element_inertia}'
                    )
            inertia += element_inertia
            # Parallel axis theorem / Huygens–Steiner theorem / Steiner's theorem
            r = (com - mesh.center_mass)
            inertia += mesh.mass*(np.inner(r, r)*np.eye(3) - np.outer(r, r))

        # Pose
        rot = Rotation.from_euler('xyz', pose[3:])
        inertial_pose = np.concatenate([
            (rot.as_matrix() @ com + np.asarray(pose[:3])),
            pose[3:],  # [0, 0, 0]  # rot.inv().as_euler('xyz'),  # pose[3:]
        ])

        # Assertions
        if kwargs_mass is not None:
            assert np.isclose(mass, kwargs_mass), (
                f'{mass} [kg] != {kwargs_mass} [kg]'
            )
        if not Inertial.valid_mass(mass):
            raise ValueError(
                f'Mesh {path} has inappropriate mass: {mass} [kg]'
            )
        if not Inertial.valid_inertia(inertia):
            raise ValueError(
                f'Mesh {path} has inappropriate inertia:\n{inertia}'
            )
        assert not kwargs, kwargs

        # Return result
        return cls(
            pose=inertial_pose,
            mass=mass,
            inertias=[
                inertia[0, 0],
                inertia[0, 1],
                inertia[0, 2],
                inertia[1, 1],
                inertia[1, 2],
                inertia[2, 2]
            ],
            units=units,
        )

    @staticmethod
    def valid_mass(mass):
        """ Check if mass is positive, bounded and non zero. """
        if mass <= 0.0 or np.isnan(mass):
            return False
        return True

    @staticmethod
    def valid_inertia(inertia):
        """ Check if inertia matrix is positive, bounded and non zero. """
        ixx = inertia[0, 0]
        iyy = inertia[1, 1]
        izz = inertia[2, 2]
        (ixx, iyy, izz) = np.linalg.eigvals(inertia)
        positive_definite = np.all([i > 0.0 for i in (ixx, iyy, izz)])
        inequality = (
            (ixx + iyy > izz) and (ixx + izz > iyy) and (iyy + izz > ixx))
        if not inequality or not positive_definite:
            return False
        return True

    def xml(self, link):
        """xml"""
        inertial = ET.SubElement(link, 'inertial')
        if self.pose is not None:
            pose = ET.SubElement(inertial, 'pose')
            pose.text = ' '.join([
                str(element*(self.units.meters if i < 3 else 1))
                for i, element in enumerate(self.pose)
            ])
        if self.mass is not None:
            mass = ET.SubElement(inertial, 'mass')
            mass.text = str(self.mass*self.units.kilograms)
        if self.inertias is not None:
            inertia = ET.SubElement(inertial, 'inertia')
            inertias = [
                ET.SubElement(inertia, name)
                for name in ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']
            ]
            for i, inertia in enumerate(inertias):
                inertia.text = str(
                    self.inertias[i]*self.units.kilograms*self.units.meters**2
                )

    @classmethod
    def from_xml(cls, data):
        """ Create Inertial object from xml data.

        Parameters
        ----------
        cls : <cls>
            Class

        data : <ET.ElemenTree>
            Inertial data from the sdf

        Returns
        -------
        out : <Inertial>
            Inertial object from xml

        """
        pose = get_pose_from_xml(data)
        mass_xml = data.find('mass')
        mass = (
            (
                float(mass_xml.text)
                if mass_xml.text
                else float(mass_xml.attrib['value'])
                if 'value' in mass_xml.attrib
                else None
            )
            if mass_xml is not None
            else None
        )
        assert mass is not None, 'Mass not found'
        inertias_xml = data.find('inertia')
        elements = ['ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz']
        inertias = (
            [float(i.text) for i in inertias_xml]
            if list(inertias_xml)
            else [inertias_xml.attrib[element] for element in elements]
            if all([element in inertias_xml.attrib for element in elements])
            else None
        )
        assert inertias is not None, 'Inertias not found'
        return cls(
            mass=mass,
            inertias=inertias,
            pose=pose,
            units=SimulationUnitScaling()
        )


class Shape(Options):
    """Shape"""

    def __init__(self, name, geometry, suffix, **kwargs):
        super(Shape, self).__init__()
        if suffix not in name:
            self.name = f'{name}_{suffix}'
        else:
            self.name = name
        self.geometry = geometry
        self.suffix = suffix
        self.pose = kwargs.pop('pose', np.zeros(6))
        assert self.pose is not None
        self.units = kwargs.pop('units', SimulationUnitScaling())
        assert not kwargs, kwargs

    @classmethod
    def plane(cls, name, normal, size, **kwargs):
        """Plane"""
        units = kwargs.get('units', SimulationUnitScaling())
        return cls(
            name=name,
            geometry=Plane(normal, size, units),
            **kwargs
        )

    @classmethod
    def box(cls, name, size, **kwargs):
        """Box"""
        units = kwargs.get('units', SimulationUnitScaling())
        return cls(
            name=name,
            geometry=Box(size, units),
            **kwargs
        )

    @classmethod
    def sphere(cls, name, radius, **kwargs):
        """Box"""
        units = kwargs.get('units', SimulationUnitScaling())
        return cls(
            name=name,
            geometry=Sphere(radius, units),
            **kwargs
        )

    @classmethod
    def cylinder(cls, name, radius, length, **kwargs):
        """Cylinder"""
        units = kwargs.get('units', SimulationUnitScaling())
        return cls(
            name=name,
            geometry=Cylinder(radius, length, units),
            **kwargs
        )

    @classmethod
    def capsule(cls, name, radius, length, **kwargs):
        """Box"""
        units = kwargs.get('units', SimulationUnitScaling())
        return cls(
            name=name,
            geometry=Capsule(radius, length, units),
            **kwargs
        )

    @classmethod
    # TODO Change the method name to mesh
    def from_mesh(cls, name, mesh, scale, **kwargs):
        """From mesh"""
        units = kwargs.get('units', SimulationUnitScaling())
        return cls(
            name=name,
            geometry=Mesh(mesh, scale, units),
            **kwargs
        )

    @classmethod
    def bounding_from_mesh(cls, name, mesh, scale, **kwargs):
        """ Create bounding shape from mesh."""
        units = kwargs.get('units', SimulationUnitScaling())
        bounding_shape = (kwargs.get('bounding_shape', 'box')).lower()
        use_primitive = kwargs.get('use_primitive', False)
        #: Read the original mesh
        mesh_obj = tri.load(mesh)
        if bounding_shape == 'box':
            box = mesh_obj.bounding_box
            extents = box.extents
            if use_primitive:
                return cls(
                    name=name,
                    geometry=Box(extents, units),
                    units=units,
                    **kwargs
                )
            else:
                #: Export mesh
                new_mesh_path = replace_file_name_in_path(
                    mesh, name.replace('_'+cls.SUFFIX, '')+'_bounding_box'
                )
                box.export(new_mesh_path)
                return cls(
                    name=name,
                    geometry=Mesh(new_mesh_path, scale, units),
                    units=units,
                    **kwargs
                )
        elif bounding_shape == 'sphere':
            sphere = mesh_obj.bounding_sphere
            radius = sphere.primitive.radius
            if use_primitive:
                return cls(
                    name=name,
                    geometry=Sphere(radius, units),
                    units=units,
                    **kwargs
                )
            else:
                #: Export mesh
                new_mesh_path = replace_file_name_in_path(
                    mesh, name.replace('_'+cls.SUFFIX, '')+'_bounding_sphere'
                )
                box.export(new_mesh_path)
                return cls(
                    name=name,
                    geometry=Mesh(new_mesh_path, scale, units),
                    units=units,
                    **kwargs
                )
        elif bounding_shape == 'cylinder':
            cylinder = mesh_obj.bounding_cylinder
            radius = cylinder.primitive.radius
            length = cylinder.primitive.height
            if use_primitive:
                return cls(
                    name=name,
                    geometry=Cylinder(radius, length, units),
                    units=units,
                    **kwargs
                )
            else:
                #: Export mesh
                new_mesh_path = replace_file_name_in_path(
                    mesh, name.replace('_'+cls.SUFFIX, '')+'_bounding_cylinder'
                )
                box.export(new_mesh_path)
                return cls(
                    name=name,
                    geometry=Mesh(new_mesh_path, scale, units),
                    units=units,
                    **kwargs
                )
        elif bounding_shape == 'convex_hull':
            convex_hull = mesh_obj.convex_hull
            #: Export mesh
            new_mesh_path = replace_file_name_in_path(
                mesh, name.replace('_'+cls.SUFFIX, '')+'_bounding_convex_hull'
            )
            convex_hull.export(new_mesh_path)
            return cls(
                name=name,
                geometry=Mesh(new_mesh_path, scale, units),
                units=units,
                **kwargs
            )
        else:
            return cls(
                name=name,
                geometry=Mesh(mesh, scale, units),
                units=units,
                **kwargs
            )

    @classmethod
    def heightmap(cls, name, uri, size, pos, **kwargs):
        """Heightmap"""
        units = kwargs.get('units', SimulationUnitScaling())
        return cls(
            name=name,
            geometry=Heightmap(uri, size, pos, units),
            **kwargs
        )

    def xml(self, link):
        """xml"""
        shape = ET.SubElement(
            link,
            self.suffix,
            name=self.name
        )
        if self.pose is not None:
            pose = ET.SubElement(shape, 'pose')
            pose.text = ' '.join([
                str(element*(self.units.meters if i < 3 else 1))
                for i, element in enumerate(self.pose)
            ])
        self.geometry.xml(shape)
        return shape


class Collision(Shape):
    """Collision"""

    SUFFIX = 'collision'

    def __init__(self, name, **kwargs):
        super(Collision, self).__init__(
            name=name,
            suffix=self.SUFFIX,
            **kwargs
        )

    @classmethod
    def from_xml(cls, data):
        """Generate collision shape model from xml.

        Parameters
        ----------
        cls : <Shape>
            Shape class data
        data : <ET.ElementTree>
            Visual/Collision object data

        Returns
        -------
        out : <Shape>
            Shape model
        """
        geometry_types = {
            'plane': Plane,
            'box': Box,
            'sphere': Sphere,
            'cylinder': Cylinder,
            'capsule': Capsule,
            'mesh': Mesh,
            'heightmap': Heightmap,
        }
        pose = get_pose_from_xml(data)
        shape_data = {
            'geometry': geometry_types[
                data.find('geometry')[0].tag
            ].from_xml(data.find('geometry')[0]),
            'pose': pose,
            'units': SimulationUnitScaling()
        }
        #: Remove the suffix
        if 'name' in data.attrib:
            name = data.attrib['name']
            if '_collision' in name:
                name = name.replace('_collision', '')
        else:
            name = 'collision'
        return cls(
            name=name,
            **shape_data
        )


class Visual(Shape):
    """Visual"""

    SUFFIX = 'visual'

    def __init__(self, name, **kwargs):
        self.color = kwargs.pop('color', None)
        self.ambient = self.color
        self.diffuse = self.color
        self.specular = self.color
        self.emissive = self.color
        super(Visual, self).__init__(name=name, suffix=self.SUFFIX, **kwargs)

    def xml(self, link):
        """xml"""
        shape = super(Visual, self).xml(link)
        material = ET.SubElement(shape, 'material')
        # script = ET.SubElement(material, 'script')
        # uri = ET.SubElement(script, 'uri')
        # uri.text = 'skin.material'
        # name = ET.SubElement(script, 'name')
        # name.text = 'Skin'
        if self.color is not None:
            # color = ET.SubElement(material, 'color')
            # color.text = ' '.join([str(element) for element in self.color])
            ambient = ET.SubElement(material, 'ambient')
            ambient.text = ' '.join(
                [str(element) for element in self.ambient]
            )
            diffuse = ET.SubElement(material, 'diffuse')
            diffuse.text = ' '.join(
                [str(element) for element in self.diffuse]
            )
            specular = ET.SubElement(material, 'specular')
            specular.text = ' '.join(
                [str(element) for element in self.specular]
            )
            emissive = ET.SubElement(material, 'emissive')
            emissive.text = ' '.join(
                [str(element) for element in self.emissive]
            )

    @classmethod
    def from_xml(cls, data):
        """Generate visual shape model from xml.

        Parameters
        ----------
        cls : <Shape>
            Shape class data
        data : <ET.ElementTree>
            Visual/Collision object data

        Returns
        -------
        out : <Shape>
            Shape model
        """
        geometry_types = {
            'plane': Plane,
            'box': Box,
            'sphere': Sphere,
            'cylinder': Cylinder,
            'capsule': Capsule,
            'mesh': Mesh,
            'heightmap': Heightmap,
        }
        material = data.find('material')
        color = (
            get_floats_from_text(material.find('diffuse').text)
            if material
            else None
        )
        pose = get_pose_from_xml(data)
        shape_data = {
            'geometry': geometry_types[
                data.find('geometry')[0].tag
            ].from_xml(data.find('geometry')[0]),
            'pose': pose,
            'color': color,
            'units': SimulationUnitScaling()
        }
        #: Remove the suffix
        if 'name' in data.attrib:
            name = data.attrib['name']
            if '_visual' in name:
                name = name.replace('_visual', '')
        else:
            name = 'visual'
        return cls(
            name=name,
            **shape_data
        )


class Plane(Options):
    """Plane"""

    def __init__(self, normal, size, units=SimulationUnitScaling()):
        super(Plane, self).__init__()
        self.normal = normal
        self.size = size
        self.units = units

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, 'geometry')
        plane = ET.SubElement(geometry, 'plane')
        normal = ET.SubElement(plane, 'normal')
        normal.text = ' '.join([
            str(element)
            for element in self.normal
        ])
        size = ET.SubElement(plane, 'size')
        size.text = ' '.join([
            str(element*self.units.meters)
            for element in self.size
        ])

    @classmethod
    def from_xml(cls, data):
        """Generate Plane shape from xml data.

        Parameters
        ----------
        cls : <Plane>
            Plane class data
        data : <ET.ElementTree>
            Plane object data

        Returns
        -------
        out : <Plane>
            Plane model
        """
        return cls(
            normal=get_floats_from_text(data.find('normal').text),
            size=get_floats_from_text(data.find('size').text),
            units=SimulationUnitScaling()
        )


class Box(Options):
    """Box"""

    def __init__(self, size, units=SimulationUnitScaling()):
        super(Box, self).__init__()
        self.size = size
        self.units = units

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, 'geometry')
        box = ET.SubElement(geometry, 'box')
        size = ET.SubElement(box, 'size')
        size.text = ' '.join([
            str(element*self.units.meters)
            for element in self.size
        ])

    @classmethod
    def from_xml(cls, data):
        """Generate Box shape from xml data.

        Parameters
        ----------
        cls : <Box>
            Box class data
        data : <ET.ElementTree>
            Box object data

        Returns
        -------
        out : <Box>
            Box model
        """
        size = (
            data.find('size').text.split(' ')
            if data.find('size') is not None
            else data.attrib['size'].split(' ')
            if 'size' in data.attrib
            else None
        )
        return cls(
            size=([float(s) for s in size]),
            units=SimulationUnitScaling()
        )


class Sphere(Options):
    """Sphere"""

    def __init__(self, radius, units=SimulationUnitScaling()):
        super(Sphere, self).__init__()
        self.radius = radius
        self.units = units

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, 'geometry')
        sphere = ET.SubElement(geometry, 'sphere')
        radius = ET.SubElement(sphere, 'radius')
        radius.text = str(self.radius*self.units.meters)

    @classmethod
    def from_xml(cls, data):
        """Generate Sphere shape from xml data.

        Parameters
        ----------
        cls : <Sphere>
            Sphere class data
        data : <ET.ElementTree>
            Sphere object data

        Returns
        -------
        out : <Sphere>
            Sphere model
        """
        radius = (
            float(data.find('radius').text)
            if data.find('radius') is not None
            else data.attrib['radius']
            if 'radius' in data.attrib
            else None
        )
        assert radius is not None, 'Radius not found'
        return cls(radius=radius, units=SimulationUnitScaling())


class Cylinder(Options):
    """Cylinder"""

    def __init__(self, radius, length, units=SimulationUnitScaling()):
        super(Cylinder, self).__init__()
        self.radius = radius
        self.length = length
        self.units = units

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, 'geometry')
        cylinder = ET.SubElement(geometry, 'cylinder')
        radius = ET.SubElement(cylinder, 'radius')
        radius.text = str(self.radius*self.units.meters)
        length = ET.SubElement(cylinder, 'length')
        length.text = str(self.length*self.units.meters)

    @classmethod
    def from_xml(cls, data):
        """Generate Cylinder shape from xml data.

        Parameters
        ----------
        cls : <Cylinder>
            Cylinder class data
        data : <ET.ElementTree>
            Cylinder object data

        Returns
        -------
        out : <Cylinder>
            Cylinder model
        """
        radius, length = [
            float(data.find(element).text)
            if data.find(element) is not None
            else data.attrib[element]
            if element in data.attrib
            else None
            for element in ['radius', 'length']
        ]
        assert radius is not None, 'Radius not found'
        assert length is not None, 'Length not found'
        return cls(radius=radius, length=length, units=SimulationUnitScaling())


class Capsule(Options):
    """Capsule"""

    def __init__(self, radius, length, units=SimulationUnitScaling()):
        super(Capsule, self).__init__()
        self.radius = radius
        self.length = length
        self.units = units

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, 'geometry')
        capsule = ET.SubElement(geometry, 'capsule')
        radius = ET.SubElement(capsule, 'radius')
        radius.text = str(self.radius*self.units.meters)
        length = ET.SubElement(capsule, 'length')
        length.text = str(self.length*self.units.meters)

    @classmethod
    def from_xml(cls, data):
        """Generate Capsule shape from xml data.

        Parameters
        ----------
        cls : <Capsule>
            Capsule class data
        data : <ET.ElementTree>
            Capsule object data

        Returns
        -------
        out : <Capsule>
            Capsule model
        """
        radius, length = [
            float(data.find(element).text)
            if data.find(element) is not None
            else data.attrib[element]
            if element in data.attrib
            else None
            for element in ['radius', 'length']
        ]
        assert radius is not None, 'Radius not found'
        assert length is not None, 'Length not found'
        return cls(radius=radius, length=length, units=SimulationUnitScaling())


class Mesh(Options):
    """Mesh"""

    def __init__(self, uri, scale, units=SimulationUnitScaling()):
        super(Mesh, self).__init__()
        self.uri = uri
        self.scale = scale
        self.units = units

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, 'geometry')
        mesh = ET.SubElement(geometry, 'mesh')
        uri = ET.SubElement(mesh, 'uri')
        uri.text = self.uri
        if self.scale is not None:
            scale = ET.SubElement(mesh, 'scale')
            scale.text = ' '.join(
                [str(s*self.units.meters) for s in self.scale]
            )

    @classmethod
    def from_xml(cls, data):
        """Generate Mesh shape from xml data.

        Parameters
        ----------
        cls : <Mesh>
            Mesh class data
        data : <ET.ElementTree>
            Mesh object data

        Returns
        -------
        out : <Mesh>
            Mesh model
        """
        scale = (
            get_floats_from_text(data.find('scale').text)
            if data.find('scale') is not None
            else get_floats_from_text(data.attrib['scale'])
            if 'scale' in data.attrib
            else [1.0, 1.0, 1.0]
        )
        uri = (
            data.find('uri').text
            if data.find('uri') is not None
            else data.attrib['filename']
            if 'filename' in data.attrib
            else None
        )
        assert uri
        return cls(
            uri=uri,
            scale=scale,
            units=SimulationUnitScaling()
        )


class Heightmap(Options):
    """Heightmap"""

    def __init__(self, uri, size, pos, units=SimulationUnitScaling()):
        super(Heightmap, self).__init__()
        self.uri = uri
        self.size = size
        self.pos = pos
        self.units = units

    def xml(self, parent):
        """xml"""
        geometry = ET.SubElement(parent, 'geometry')
        heightmap = ET.SubElement(geometry, 'heightmap')
        uri = ET.SubElement(heightmap, 'uri')
        uri.text = self.uri
        if self.size is not None:
            size = ET.SubElement(heightmap, 'size')
            size.text = ' '.join([
                str(element*self.units.meters)
                for element in self.size
            ])
        if self.pos is not None:
            pos = ET.SubElement(heightmap, 'pos')
            pos.text = ' '.join([
                str(element*self.units.meters)
                for element in self.pos
            ])

    @classmethod
    def from_xml(cls, data):
        """Generate Heightmap shape from xml data.

        Parameters
        ----------
        cls : <Heightmap>
            Heightmap class data
        data : <ET.ElementTree>
            Heightmap object data

        Returns
        -------
        out : <Heightmap>
            Heightmap model
        """
        size = (
            get_floats_from_text(data.find('size').text)
            if data.find('size') is not None
            else 1.0
        )
        pos = (
            get_floats_from_text(data.find('pos').text)
            if data.find('pos') is not None
            else 1.0
        )
        return cls(
            uri=data.find('uri').text,
            size=size,
            pos=pos,
            units=SimulationUnitScaling()
        )


class Joint(Options):
    """Joint"""

    def __init__(self, name, joint_type, parent, child, **kwargs):
        super(Joint, self).__init__()
        self.name = name
        self.type = joint_type
        self.parent = parent.name
        self.child = child.name
        self.pose = kwargs.pop('pose', np.zeros(6))
        if kwargs.get('xyz', None) is not None:
            self.axis = Axis(**kwargs)
        else:
            self.axis = None

    def xml(self, model):
        """xml"""
        joint = ET.SubElement(model, 'joint', name=self.name, type=self.type)
        parent = ET.SubElement(joint, 'parent')
        parent.text = self.parent
        child = ET.SubElement(joint, 'child')
        child.text = self.child
        if self.pose is not None:
            pose = ET.SubElement(joint, 'pose')
            pose.text = ' '.join([str(element) for element in self.pose])
        if self.axis is not None:
            self.axis.xml(joint)

    @classmethod
    def from_xml(cls, data):
        """
        Generate joint object from xml data

        Parameters
        ----------
        cls : <cls>
            Class

        data : <ET.ElemenTree>
            Joint data from the sdf

        Returns
        -------
        out : <Joint>
            Joint object from xml
        """
        pose = get_pose_from_xml(data)
        axis_data = (
            Axis.from_xml(data.find('axis'))
            if data.find('axis') is not None
            else None
        )
        names = [None, None]
        for element_i, element in enumerate(['parent', 'child']):
            element_xml = data.find(element)
            names[element_i] = (
                element_xml.text
                if element_xml.text
                else element_xml.attrib['link']
                if 'link' in element_xml.attrib
                else None
            )
        parent, child = names
        if axis_data is None:
            return cls(
                name=data.attrib['name'],
                joint_type=data.attrib['type'],
                parent=Link.empty(parent, []),
                child=Link.empty(child, []),
                **{'pose': pose}
            )
        return cls(
            name=data.attrib['name'],
            joint_type=data.attrib['type'],
            parent=Link.empty(parent, []),
            child=Link.empty(child, []),
            **{
                'pose': pose,
                **axis_data
            }
        )


class Axis(Options):
    """Axis"""

    def __init__(self, **kwargs):
        super(Axis, self).__init__()
        self.initial_position = kwargs.pop('initial_position', None)
        self.xyz = kwargs.pop('xyz', [0, 0, 0])
        self.limits = kwargs.pop('limits', None)
        self.dynamics = kwargs.pop('dynamics', None)

    def xml(self, joint):
        """xml"""
        axis = ET.SubElement(joint, 'axis')
        if self.initial_position:
            initial_position = ET.SubElement(axis, 'initial_position')
            initial_position.text = str(self.initial_position)
        xyz = ET.SubElement(axis, 'xyz')
        xyz.text = ' '.join([str(element) for element in self.xyz])
        if self.limits is not None:
            limit = ET.SubElement(axis, 'limit')
            lower = ET.SubElement(limit, 'lower')
            lower.text = str(self.limits[0])
            upper = ET.SubElement(limit, 'upper')
            upper.text = str(self.limits[1])
            effort = ET.SubElement(limit, 'effort')
            effort.text = str(self.limits[2])
            velocity = ET.SubElement(limit, 'velocity')
            velocity.text = str(self.limits[3])
        if self.dynamics is not None:
            dynamics = ET.SubElement(axis, 'dynamics')
            damping = ET.SubElement(dynamics, 'damping')
            damping.text = str(self.dynamics[0])
            friction = ET.SubElement(dynamics, 'friction')
            friction.text = str(self.dynamics[1])
            spring_reference = ET.SubElement(dynamics, 'spring_reference')
            spring_reference.text = str(self.dynamics[2])
            spring_stiffness = ET.SubElement(dynamics, 'spring_stiffness')
            spring_stiffness.text = str(self.dynamics[3])

    @classmethod
    def from_xml(cls, data):
        """
        Generate axis object from xml data

        Parameters
        ----------
        cls : <cls>
            Class

        data : <ET.ElemenTree>
            Axis data from the sdf

        Returns
        -------
        out : <Axis>
            Axis object from xml
        """
        xyz_text = (
            data.find('xyz').text
            if data.find('xyz') is not None
            else data.attrib['xyz']
            if 'xyz' in data.attrib
            else None
        )
        assert xyz_text is not None, 'XYZ not found'
        xyz = get_floats_from_text(xyz_text)
        initial_position = (
            float(data.find('initial_position').text)
            if data.find('initial_position') is not None
            else None
        )
        limits = None
        if data.find('limit') is not None:
            limits = [-1]*4
            limits[0] = float(data.find('limit/lower').text)
            limits[1] = float(data.find('limit/upper').text)
            limits[2] = (
                float(data.find('limit/effort').text)
                if data.find('limit/effort') is not None
                else -1
            )
            limits[3] = (
                float(data.find('limit/velocity').text)
                if data.find('limit/velocity') is not None
                else -1
            )
        dynamics = None
        if data.find('dynamics') is not None:
            dynamics = [0]*4
            dynamics[0] = (
                float(data.find('damping').text)
                if data.find('damping') is not None
                else 0.0
            ),
            dynamics[1] = (
                float(data.find('friction').text)
                if data.find('friction') is not None
                else 0.0
            ),
            dynamics[2] = float(data.find('spring_reference').text)
            dynamics[3] = float(data.find('spring_stiffness').text)
        axis_data = {
            'initial_position': initial_position,
            'xyz': xyz,
            'limits': limits,
            'dynamics': dynamics
        }
        return axis_data
