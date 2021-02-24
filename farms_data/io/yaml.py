"""HDF5 operations"""

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
    import farms_pylog as pylog
    pylog.warning(
        'YAML CLoader and CDumper not available'
        ', switching to Python implementation'
        '\nThis will run slower than the C alternative'
    )



def pyobject2yaml(filename, pyobject, mode='w+'):
    """Pyobject to yaml"""
    with open(filename, mode) as yaml_file:
        yaml.dump(
            pyobject,
            yaml_file,
            default_flow_style=False,
            sort_keys=False,
            Dumper=Dumper,
        )


def yaml2pyobject(filename):
    """Pyobject to yaml"""
    with open(filename, 'r') as yaml_file:
        options = yaml.load(yaml_file, Loader=Loader)
    return options
