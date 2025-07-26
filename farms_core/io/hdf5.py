"""HDF5 operations"""

import time
import h5py
import numpy as np
from .. import pylog


def _val_to_hdf5(handler, key, value):
    """Value to HDF5"""
    if value is None:
        handler.create_dataset(name=key, data=h5py.Empty(None))
        return
    options = {} if np.isscalar(value) else {'compression': True}
    try:
        handler.create_dataset(name=key, data=value, **options)
    except TypeError as err:
        raise TypeError(
            f'Issue when attempting to write {key=} with data:\n{value}'
        ) from err


def _dict_to_hdf5(handler, dict_data, group=None):
    """Dictionary to HDF5"""
    if group is not None and group not in handler:
        handler = handler.create_group(group)
    for key, value in dict_data.items():
        if isinstance(value, dict):
            _dict_to_hdf5(handler, value, key)
        elif (  # List of dictionaries
                isinstance(value, list)
                and value
                and all(isinstance(val, dict) for val in value)
        ):
            handler_list = handler.create_group(f'FARMSLIST{key}')
            for val_i, val in enumerate(value):
                key_list = str(val_i)
                if isinstance(val, dict):
                    _dict_to_hdf5(handler_list, val, key_list)
                else:
                    _val_to_hdf5(handler_list, key_list, value)
        else:
            _val_to_hdf5(handler, key, value)


def _hdf5_to_dict(handler, dict_data):
    """HDF5 to dictionary"""
    for key, value in handler.items():
        if isinstance(value, h5py.Group):
            new_dict = {}
            _hdf5_to_dict(value, new_dict)
            if 'FARMSLIST' in key:
                n_items = len(new_dict)
                new_list = [
                    new_dict[str(item_i)]
                    for item_i in range(n_items)
                ]
                dict_data[key.replace('FARMSLIST', '')] = new_list
            else:
                dict_data[key] = new_dict
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
                        raise TypeError(f'Cannot handle shape {value.shape}')
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
