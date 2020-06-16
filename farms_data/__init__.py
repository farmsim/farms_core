"""FARMS data"""

import os


def get_include_paths():
    """Get include paths"""
    farms_data_path = os.path.dirname(os.path.abspath(__file__))
    directories = [
        os.path.join(farms_data_path, 'amphibious'),
        os.path.join(farms_data_path, 'sensors'),
    ]
    for directory in directories:
        assert os.path.isdir(directory)
    return directories
