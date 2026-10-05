"""Tests for farms_core.io.hdf5 append mode

Covers the HDF5 write/append/read round-trip that underpins the
incremental HDF5 saving feature.

Time-varying data (produced by ``to_array`` with ``iteration``) is
wrapped in :class:`AppendableArray` so that ``_val_to_hdf5`` knows to
append it along the first dimension.  Static data (plain arrays,
scalars, string arrays) is overwritten on each save.
"""

import os
import tempfile

import numpy as np
import pytest

from farms_core.io.hdf5 import dict_to_hdf5, hdf5_to_dict
from farms_core.array.array import AppendableArray


@pytest.fixture
def tmp_hdf5():
    """Provide a temporary HDF5 file path."""
    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)


class TestHDF5WriteMode:
    """Basic write mode (mode='w')."""

    def test_write_and_read_simple(self, tmp_hdf5):
        data = {'a': np.arange(10), 'b': 3.14}
        dict_to_hdf5(tmp_hdf5, data, mode='w')
        result = hdf5_to_dict(tmp_hdf5)
        np.testing.assert_array_equal(result['a'], data['a'])
        assert result['b'] == pytest.approx(data['b'])

    def test_write_nested_dict(self, tmp_hdf5):
        data = {
            'outer': {
                'inner': np.arange(5),
                'scalar': 42,
            },
            'top': np.zeros(3),
        }
        dict_to_hdf5(tmp_hdf5, data, mode='w')
        result = hdf5_to_dict(tmp_hdf5)
        np.testing.assert_array_equal(result['outer']['inner'], data['outer']['inner'])
        assert result['outer']['scalar'] == 42
        np.testing.assert_array_equal(result['top'], data['top'])

    def test_write_list_of_dicts(self, tmp_hdf5):
        data = {
            'animats': [
                {'values': np.arange(3)},
                {'values': np.arange(5)},
            ],
        }
        dict_to_hdf5(tmp_hdf5, data, mode='w')
        result = hdf5_to_dict(tmp_hdf5)
        assert len(result['animats']) == 2
        np.testing.assert_array_equal(result['animats'][0]['values'], np.arange(3))
        np.testing.assert_array_equal(result['animats'][1]['values'], np.arange(5))

    def test_write_2d_array(self, tmp_hdf5):
        data = {'matrix': np.arange(20).reshape(10, 2)}
        dict_to_hdf5(tmp_hdf5, data, mode='w')
        result = hdf5_to_dict(tmp_hdf5)
        np.testing.assert_array_equal(result['matrix'], data['matrix'])


class TestHDF5AppendMode:
    """Append mode (mode='a') for incremental saves."""

    def test_append_time_varying_1d(self, tmp_hdf5):
        """Append 1D time-varying data."""
        chunk1 = AppendableArray(np.arange(5))
        chunk2 = AppendableArray(np.arange(5, 10))

        dict_to_hdf5(tmp_hdf5, {'data': chunk1}, mode='w')
        dict_to_hdf5(tmp_hdf5, {'data': chunk2}, mode='a')

        result = hdf5_to_dict(tmp_hdf5)
        np.testing.assert_array_equal(result['data'], np.arange(10))

    def test_append_time_varying_2d(self, tmp_hdf5):
        """Append 2D time-varying data (first dim grows)."""
        chunk1 = AppendableArray(np.arange(10).reshape(5, 2))
        chunk2 = AppendableArray(np.arange(10, 20).reshape(5, 2))

        dict_to_hdf5(tmp_hdf5, {'data': chunk1}, mode='w')
        dict_to_hdf5(tmp_hdf5, {'data': chunk2}, mode='a')

        result = hdf5_to_dict(tmp_hdf5)
        expected = np.arange(20).reshape(10, 2)
        np.testing.assert_array_equal(result['data'], expected)

    def test_append_multiple_chunks(self, tmp_hdf5):
        """Append multiple chunks simulating periodic saves."""
        buffer_size = 5
        n_chunks = 4

        for chunk_i in range(n_chunks):
            chunk = AppendableArray(np.arange(
                chunk_i * buffer_size,
                (chunk_i + 1) * buffer_size,
            ))
            mode = 'w' if chunk_i == 0 else 'a'
            dict_to_hdf5(tmp_hdf5, {'data': chunk}, mode=mode)

        result = hdf5_to_dict(tmp_hdf5)
        expected = np.arange(n_chunks * buffer_size)
        np.testing.assert_array_equal(result['data'], expected)

    def test_append_overwrite_static(self, tmp_hdf5):
        """Static data (scalars, string arrays) is overwritten, not appended."""
        dict_to_hdf5(tmp_hdf5, {'scalar': 1.0, 'names': ['a', 'b']}, mode='w')
        dict_to_hdf5(tmp_hdf5, {'scalar': 2.0, 'names': ['a', 'b']}, mode='a')

        result = hdf5_to_dict(tmp_hdf5)
        assert result['scalar'] == 2.0
        assert list(result['names']) == ['a', 'b']

    def test_append_overwrite_static_numeric(self, tmp_hdf5):
        """Plain numeric arrays (not AppendableArray) are overwritten."""
        dict_to_hdf5(tmp_hdf5, {'config': np.arange(5)}, mode='w')
        dict_to_hdf5(tmp_hdf5, {'config': np.arange(5)}, mode='a')

        result = hdf5_to_dict(tmp_hdf5)
        np.testing.assert_array_equal(result['config'], np.arange(5))

    def test_append_nested_dict(self, tmp_hdf5):
        """Append data within nested groups."""
        chunk1 = {'simulation': {
            'ncon': AppendableArray(np.arange(5)),
            'energy': AppendableArray(np.zeros((5, 2))),
        }}
        chunk2 = {'simulation': {
            'ncon': AppendableArray(np.arange(5, 10)),
            'energy': AppendableArray(np.ones((5, 2))),
        }}

        dict_to_hdf5(tmp_hdf5, chunk1, mode='w')
        dict_to_hdf5(tmp_hdf5, chunk2, mode='a')

        result = hdf5_to_dict(tmp_hdf5)
        np.testing.assert_array_equal(
            result['simulation']['ncon'], np.arange(10),
        )
        expected_energy = np.vstack([np.zeros((5, 2)), np.ones((5, 2))])
        np.testing.assert_array_equal(
            result['simulation']['energy'], expected_energy,
        )

    def test_append_mixed_static_and_varying(self, tmp_hdf5):
        """Mix of static (overwritten) and time-varying (appended) data."""
        chunk1 = {
            'timestep': 0.001,
            'ncon': AppendableArray(np.arange(5)),
            'names': ['a', 'b'],
        }
        chunk2 = {
            'timestep': 0.001,
            'ncon': AppendableArray(np.arange(5, 10)),
            'names': ['a', 'b'],
        }

        dict_to_hdf5(tmp_hdf5, chunk1, mode='w')
        dict_to_hdf5(tmp_hdf5, chunk2, mode='a')

        result = hdf5_to_dict(tmp_hdf5)
        assert result['timestep'] == pytest.approx(0.001)
        np.testing.assert_array_equal(result['ncon'], np.arange(10))
        assert list(result['names']) == ['a', 'b']

    def test_append_none_values(self, tmp_hdf5):
        """None values in append mode should not crash."""
        chunk1 = {
            'map': {'connections': None, 'weights': None},
            'ncon': AppendableArray(np.arange(5)),
        }
        chunk2 = {
            'map': {'connections': None, 'weights': None},
            'ncon': AppendableArray(np.arange(5, 10)),
        }

        dict_to_hdf5(tmp_hdf5, chunk1, mode='w')
        dict_to_hdf5(tmp_hdf5, chunk2, mode='a')

        result = hdf5_to_dict(tmp_hdf5)
        np.testing.assert_array_equal(result['ncon'], np.arange(10))

    def test_append_list_of_dicts(self, tmp_hdf5):
        """Append data in list-of-dicts (FARMSLIST) groups."""
        chunk1 = {
            'animats': [
                {'sensors': {'values': AppendableArray(np.arange(3))}},
            ],
        }
        chunk2 = {
            'animats': [
                {'sensors': {'values': AppendableArray(np.arange(3, 6))}},
            ],
        }

        dict_to_hdf5(tmp_hdf5, chunk1, mode='w')
        dict_to_hdf5(tmp_hdf5, chunk2, mode='a')

        result = hdf5_to_dict(tmp_hdf5)
        assert len(result['animats']) == 1
        np.testing.assert_array_equal(
            result['animats'][0]['sensors']['values'],
            np.arange(6),
        )

    def test_append_empty_range_skipped(self, tmp_hdf5):
        """Appending an empty AppendableArray should not corrupt the dataset."""
        dict_to_hdf5(
            tmp_hdf5, {'data': AppendableArray(np.arange(5))}, mode='w',
        )
        dict_to_hdf5(
            tmp_hdf5, {'data': AppendableArray(np.array([]))}, mode='a',
        )

        result = hdf5_to_dict(tmp_hdf5)
        np.testing.assert_array_equal(result['data'], np.arange(5))


class TestHDF5IncrementalSaveSimulation:
    """Simulate the full incremental save pattern via HDF5."""

    def test_full_incremental_save_cycle(self, tmp_hdf5):
        """Simulate ExperimentLogger's save pattern end-to-end."""
        buffer_size = 5
        n_iterations = 12

        # In-memory circular buffer
        ncon = np.zeros(buffer_size)
        energy = np.zeros((buffer_size, 2))
        timestep = 0.001
        last_saved = 0
        mode = 'w'

        for i in range(n_iterations):
            idx = i % buffer_size
            ncon[idx] = float(i * 10)
            energy[idx, :] = [i * 30, i * 40]

            # Save at buffer boundaries
            if (i + 1) % buffer_size == 0:
                # Extract range [last_saved, i+1)
                n_elements = (i + 1) - last_saved
                start = last_saved % buffer_size
                end = start + n_elements
                if end <= buffer_size:
                    ncon_chunk = ncon[start:end]
                    energy_chunk = energy[start:end]
                else:
                    wrap = end - buffer_size
                    ncon_chunk = np.concatenate((ncon[start:], ncon[:wrap]))
                    energy_chunk = np.concatenate((energy[start:], energy[:wrap]))

                data = {
                    'timestep': timestep,
                    'simulation': {
                        'ncon': AppendableArray(ncon_chunk),
                        'niter': AppendableArray(ncon_chunk),  # simplified
                        'energy': AppendableArray(energy_chunk),
                    },
                }
                dict_to_hdf5(tmp_hdf5, data, mode=mode)
                mode = 'a'
                last_saved = i + 1

        # Final flush
        if n_iterations > last_saved:
            n_elements = n_iterations - last_saved
            start = last_saved % buffer_size
            end = start + n_elements
            if end <= buffer_size:
                ncon_chunk = ncon[start:end]
                energy_chunk = energy[start:end]
            else:
                wrap = end - buffer_size
                ncon_chunk = np.concatenate((ncon[start:], ncon[:wrap]))
                energy_chunk = np.concatenate((energy[start:], energy[:wrap]))

            data = {
                'timestep': timestep,
                'simulation': {
                    'ncon': AppendableArray(ncon_chunk),
                    'niter': AppendableArray(ncon_chunk),
                    'energy': AppendableArray(energy_chunk),
                },
            }
            dict_to_hdf5(tmp_hdf5, data, mode=mode)

        # Verify
        result = hdf5_to_dict(tmp_hdf5)
        assert result['timestep'] == pytest.approx(timestep)
        expected_ncon = np.arange(n_iterations) * 10
        np.testing.assert_array_equal(
            result['simulation']['ncon'], expected_ncon,
        )
        expected_energy = np.column_stack([
            np.arange(n_iterations) * 30,
            np.arange(n_iterations) * 40,
        ]).astype(float)
        np.testing.assert_array_equal(
            result['simulation']['energy'], expected_energy,
        )
