"""FARMS data"""

import os


def get_include_paths():
    """Get include paths"""
    farms_core_path = os.path.dirname(os.path.abspath(__file__))
    directories = [
        os.path.join(farms_core_path, folder)
        for folder in ['', 'amphibious', 'sensors', 'utils']
    ]
    for directory in directories:
        assert os.path.isdir(directory)
    return directories
