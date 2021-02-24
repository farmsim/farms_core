""" Input output operations for yaml files"""

import collections

import yaml
from yaml.representer import Representer

import farms_pylog as pylog


########## YAML ##########
yaml.add_representer(collections.defaultdict, Representer.represent_dict)

def read_yaml(file_path):
    """Read the yaml data from file.
    Parameters
    ----------
    file_path: <str>
        Location of the *.yaml/yml containing the bone table.
    """
    pylog.debug(f"Reading {file_path}")
    with open(file_path, 'r') as stream:
        data = yaml.load(stream, Loader=yaml.FullLoader)
    return data


def write_yaml(data, file_path):
    """ Method that dumps the data to yaml file.

    Parameters
    ----------
    data : <dict>
        Dictionary containing the data to dump
    file_path : <str>
        File path to dump the data

    Returns
    -------
    out : <None>
    """
    pylog.debug(f"Writing {file_path}")
    with open(file_path, 'w') as stream:
        to_write = yaml.dump(
            data, default_flow_style=None,
            explicit_start=True, indent=2, width=80
        )
        stream.write(to_write)
