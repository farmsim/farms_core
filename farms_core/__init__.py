"""FARMS core"""

import os


# Main version
__version__ = "0.1.1"


def get_include_paths() -> list[str]:
    """Get include paths"""
    farms_core_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(farms_core_path)
    directories = [
        os.path.join(farms_core_path, folder)
        for folder in ['', 'array', 'sensors', 'model', 'utils']
    ]
    directories.insert(0, parent_path)
    for directory in directories:
        assert os.path.isdir(directory)
    return directories
