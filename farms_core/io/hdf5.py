"""HDF5 operations"""

import time
import h5py
import numpy as np
from .. import pylog


def _dict_to_hdf5(handler, dict_data, group=None):
    """Dictionary to HDF5"""
    if group is not None and group not in handler:
        handler = handler.create_group(group)
    for key, value in dict_data.items():
        if isinstance(value, dict):
            _dict_to_hdf5(handler, value, key)
        elif value is None:
            handler.create_dataset(name=key, data=h5py.Empty(None))
        else:
            options = {} if np.isscalar(value) else {'compression': True}
            handler.create_dataset(name=key, data=value, **options)


def _hdf5_to_dict(handler, dict_data):
    """HDF5 to dictionary"""
    for key, value in handler.items():
        if isinstance(value, h5py.Group):
            new_dict = {}
            dict_data[key] = new_dict
            _hdf5_to_dict(value, new_dict)
        else:
            if value.shape:
                if value.dtype == np.dtype('O'):
                    if len(value.shape) == 1:
                        data = [val.decode("utf-8") for val in value]
                    elif len(value.shape) == 2:
                        data = [
                            tuple(val.decode("utf-8") for val in values)
                            for values in value
                        ]
                    else:
                        raise Exception(f'Cannot handle shape {value.shape}')
                else:
                    data = np.array(value)
            elif value.shape is not None:
                data = np.array(value).item()
            else:
                data = None
            dict_data[key] = data


def hdf5_open(filename, mode='w', max_attempts=10, attempt_delay=0.1):
    """Open HDF5 file with delayed attempts"""
    for attempt in range(max_attempts):
        try:
            hfile = h5py.File(name=filename, mode=mode)
            break
        except OSError as err:
            if attempt == max_attempts - 1:
                pylog.error(
                    'File %s was locked during more than %s [s]',
                    filename,
                    max_attempts*attempt_delay,
                )
                raise err
            pylog.warning(
                'File %s seems locked during attempt %s/%s',
                filename,
                attempt+1,
                max_attempts,
            )
            time.sleep(attempt_delay)
    return hfile


def dict_to_hdf5(filename, data, mode='w', **kwargs):
    """Dictionary to HDF5"""
    hfile = hdf5_open(filename, mode=mode, **kwargs)
    _dict_to_hdf5(hfile, data)
    hfile.close()


def hdf5_to_dict(filename, **kwargs):
    """HDF5 to dictionary"""
    data = {}
    hfile = hdf5_open(filename, mode='r', **kwargs)
    _hdf5_to_dict(hfile, data)
    hfile.close()
    return data


def hdf5_keys(filename, **kwargs):
    """HDF5 to dictionary"""
    hfile = hdf5_open(filename, mode='r', **kwargs)
    keys = list(hfile.keys())
    hfile.close()
    return keys


def hdf5_get(filename, key, **kwargs):
    """HDF5 to dictionary"""
    dict_data = {}
    hfile = hdf5_open(filename, mode='r', **kwargs)
    handler = hfile
    for _key in key:
        handler = handler[_key]
    _hdf5_to_dict(handler, dict_data)
    hfile.close()
    return dict_data
