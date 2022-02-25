"""Input output operations for YAML files"""

# pylint: disable=no-name-in-module,import-error,no-member

import collections
import farms_pylog as pylog
import yaml
from yaml.representer import Representer
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
    pylog.warning(
        'YAML CLoader and CDumper not available'
        ', switching to Python implementation'
        '\nThis will run slower than the C alternative'
    )


yaml.add_representer(
    data_type=collections.defaultdict,
    representer=Representer.represent_dict,
)


def read_yaml(file_path):
    """Read the yaml data from file.
    Parameters
    ----------
    file_path: <str>
        Location of the *.yaml/yml containing the bone table.
    """
    pylog.debug('Reading %s', file_path)
    with open(file_path, 'r', encoding='utf-8') as stream:
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
    pylog.debug('Writing %s', file_path)
    with open(file_path, 'w', encoding='utf-8') as stream:
        to_write = yaml.dump(
            data, default_flow_style=False,
            explicit_start=True, indent=2, width=80,
            sort_keys=False
        )
        stream.write(to_write)


def pyobject2yaml(filename, pyobject, mode='w+'):
    """Pyobject to yaml"""
    with open(filename, mode, encoding='utf-8') as yaml_file:
        yaml.dump(
            pyobject,
            yaml_file,
            default_flow_style=False,
            sort_keys=False,
            Dumper=Dumper,
        )


def yaml2pyobject(filename):
    """Pyobject to yaml"""
    with open(filename, 'r', encoding='utf-8') as yaml_file:
        options = yaml.load(yaml_file, Loader=Loader)
    return options
