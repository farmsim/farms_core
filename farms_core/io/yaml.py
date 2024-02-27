"""Input output operations for YAML files"""

import collections
from typing import Dict, Any
import yaml
from .. import pylog
try:
    from yaml import CLoader as YamlLoader, CDumper as YamlDumper
except ImportError:
    from yaml import Loader as YamlLoader, Dumper as YamlDumper
    pylog.warning(
        'YAML CLoader and CDumper not available'
        ', switching to Python implementation'
        '\nThis will run slower than the C alternative'
    )

# pylint: disable=no-member


yaml.add_representer(
    data_type=collections.defaultdict,
    representer=yaml.representer.Representer,
)


def read_yaml(file_path: str) -> Any:
    """Read the yaml data from file.

    Parameters
    ----------
    file_path :
        Location of the .yaml/yml containing the bone table.

    """
    pylog.debug('Reading %s', file_path)
    with open(file_path, 'r', encoding='utf-8') as stream:
        data = yaml.load(stream, Loader=YamlLoader)
    return data


def write_yaml(data: Dict, file_path: str):
    """Method that dumps the data to yaml file.

    Parameters
    ----------
    data :
        Dictionary containing the data to dump
    file_path :
        File path to dump the data

    """
    pylog.debug('Writing %s', file_path)
    with open(file_path, 'w', encoding='utf-8') as stream:
        to_write = yaml.dump(
            data,
            default_flow_style=False,
            explicit_start=True,
            indent=2,
            width=80,
            sort_keys=False,
            Dumper=YamlDumper,
        )
        stream.write(to_write)


def pyobject2yaml(filename: str, pyobject: Any, mode='w+'):
    """Pyobject to yaml"""
    with open(filename, mode, encoding='utf-8') as yaml_file:
        yaml.dump(
            pyobject,
            yaml_file,
            default_flow_style=False,
            sort_keys=False,
            Dumper=YamlDumper,
        )


def yaml2pyobject(filename: str) -> Any:
    """Pyobject to yaml"""
    with open(filename, 'r', encoding='utf-8') as yaml_file:
        options = yaml.load(yaml_file, Loader=YamlLoader)
    return options
