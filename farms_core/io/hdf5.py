"""HDF5 operations"""

import time
import h5py
import numpy as np
from .. import pylog
from ..array.array import AppendableArray


def _create_resizable_dataset(handler, key, value, arr):
    """Create a resizable dataset for time-varying (appendable) data."""
    if arr.ndim > 1:
        maxshape = (None,) + arr.shape[1:]
    else:
        maxshape = (None,)
    handler.create_dataset(
        key, data=value, maxshape=maxshape,
        compression=True,
    )


def _create_static_dataset(handler, key, value, arr):
    """Create a non-resizable dataset for static data."""
    handler.create_dataset(key, data=value, compression=True)


def _val_to_hdf5(handler, key, value, mode='w'):
    """Value to HDF5, with support for appending.

    Distinguishes between **time-varying** data (wrapped in
    :class:`AppendableArray`, created with ``maxshape`` so it can be
    appended to in subsequent saves) and **static** data (plain arrays
    or scalars, overwritten on each save).

    In append mode (``mode='a'``):
    - :class:`AppendableArray` values are appended along dim 0.
    - Plain arrays and scalars are overwritten (static data).
    - ``None`` values are skipped if the key exists.
    """
    is_appendable = isinstance(value, AppendableArray)
    if is_appendable:
        value = value.array

    if mode == 'a':
        if key in handler:
            if value is None:
                return
            if isinstance(value, (list, tuple, np.ndarray)):
                val_arr = np.asarray(value)
                if val_arr.shape[0] == 0:
                    return
                ds = handler[key]
                if is_appendable:
                    # Time-varying: append along first dimension
                    old_shape = ds.shape
                    if (
                        len(old_shape) == len(val_arr.shape)
                        and old_shape[1:] == val_arr.shape[1:]
                    ):
                        new_len = old_shape[0] + val_arr.shape[0]
                        if len(old_shape) > 1:
                            new_shape = (new_len,) + old_shape[1:]
                        else:
                            new_shape = (new_len,)
                        ds.resize(new_shape)
                        ds[-val_arr.shape[0]:] = value
                    else:
                        ds[()] = value
                else:
                    # Static: overwrite
                    ds[()] = value
            else:
                handler[key][()] = value
        else:
            # New key in append mode
            if value is None:
                handler.create_dataset(name=key, data=h5py.Empty(None))
            elif isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value)
                if is_appendable:
                    _create_resizable_dataset(handler, key, value, arr)
                else:
                    _create_static_dataset(handler, key, value, arr)
            else:
                handler.create_dataset(key, data=value)
    else:
        # Write mode (mode='w')
        if value is None:
            handler.create_dataset(name=key, data=h5py.Empty(None))
            return
        if np.isscalar(value):
            handler.create_dataset(name=key, data=value)
            return
        arr = np.asarray(value)
        if is_appendable:
            _create_resizable_dataset(handler, key, value, arr)
        else:
            # Static array (plain numeric, string, or object) — not resizable
            _create_static_dataset(handler, key, value, arr)


def _dict_to_hdf5(handler, dict_data, group=None, mode='w'):
    """Dictionary to HDF5, with support for appending."""
    if group is not None:
        if group not in handler:
            handler = handler.create_group(group)
        else:
            handler = handler[group]
    for key, value in dict_data.items():
        if isinstance(value, dict):
            _dict_to_hdf5(handler, value, key, mode)
        elif (
            isinstance(value, list)
            and value
            and all(isinstance(val, dict) for val in value)
        ):
            if f'FARMSLIST{key}' not in handler:
                handler_list = handler.create_group(f'FARMSLIST{key}')
            else:
                handler_list = handler[f'FARMSLIST{key}']
            for val_i, val in enumerate(value):
                key_list = str(val_i)
                if isinstance(val, dict):
                    _dict_to_hdf5(handler_list, val, key_list, mode)
                else:
                    _val_to_hdf5(handler_list, key_list, val, mode)
        else:
            _val_to_hdf5(handler, key, value, mode)


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
    """Open HDF5 file with delayed attempts

    Returned value must be closed with hfile.close().

    """
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
    """Save or append a dictionary to HDF5 file"""
    hfile = hdf5_open(filename, mode=mode, **kwargs)
    _dict_to_hdf5(hfile, data, mode=mode)
    hfile.close()


def hdf5_to_dict(filename, **kwargs):
    """HDF5 file to dictionary"""
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
