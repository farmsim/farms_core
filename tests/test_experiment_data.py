"""Tests for farms_core.experiment.data.ExperimentData

Covers the full-buffer and incremental save modes for
ExperimentData.to_file, including the buffer-size-aware allocation
of both SimulationData and times.
"""

import os
import tempfile

import numpy as np
import pytest

from farms_core.simulation.data import SimulationData
from farms_core.experiment.data import ExperimentData
from farms_core.io.hdf5 import hdf5_to_dict


@pytest.fixture
def tmp_hdf5():
    """Provide a temporary HDF5 file path."""
    with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)


def _make_experiment_data(
    n_iterations: int = 10,
    buffer_size: int | None = None,
) -> ExperimentData:
    """Create a minimal ExperimentData for testing.

    Uses no animats (empty list) to avoid requiring farms_network
    or other heavy dependencies.
    """
    if buffer_size is None or buffer_size <= 0 or buffer_size > n_iterations:
        buffer_size = n_iterations
    timestep = 0.001
    times = np.zeros(buffer_size)
    sim = SimulationData.from_size(n_iterations, buffer_size=buffer_size)
    return ExperimentData(
        times=times,
        timestep=timestep,
        simulation=sim,
        animats=[],
    )


def _fill_data(data: ExperimentData, n_iterations: int, buffer_size: int):
    """Fill data as if update_sensors was called for each iteration."""
    for i in range(n_iterations):
        idx = i % buffer_size
        data.times[idx] = i * 0.001
        data.simulation.ncon[idx] = float(i * 10)
        data.simulation.niter[idx] = float(i * 20)
        data.simulation.energy[idx, 0] = float(i * 30)
        data.simulation.energy[idx, 1] = float(i * 40)


class TestExperimentDataAllocation:
    """ExperimentData buffer-size-aware allocation."""

    def test_full_buffer_allocation(self):
        """Default: times and SimulationData at n_iterations."""
        data = _make_experiment_data(n_iterations=100, buffer_size=None)
        assert data.times.shape == (100,)
        assert data.simulation.ncon.shape == (100,)

    def test_reduced_buffer_allocation(self):
        """buffer_size < n_iterations: arrays at buffer_size."""
        data = _make_experiment_data(n_iterations=100, buffer_size=10)
        assert data.times.shape == (10,)
        assert data.simulation.ncon.shape == (10,)
        assert data.simulation.niter.shape == (10,)
        assert data.simulation.energy.shape == (10, 2)

    def test_buffer_equals_iterations(self):
        """buffer_size == n_iterations: arrays at n_iterations."""
        data = _make_experiment_data(n_iterations=50, buffer_size=50)
        assert data.times.shape == (50,)
        assert data.simulation.ncon.shape == (50,)


class TestExperimentDataFullBufferSave:
    """Full-buffer mode: save all data once at the end."""

    def test_full_buffer_save_round_trip(self, tmp_hdf5):
        """Write all data in one shot and read it back."""
        n_iterations = 10
        data = _make_experiment_data(n_iterations=n_iterations)
        _fill_data(data, n_iterations, n_iterations)

        data.to_file(tmp_hdf5, iteration=n_iterations, start_iteration=0, mode='w')

        result = hdf5_to_dict(tmp_hdf5)
        np.testing.assert_array_equal(
            result['simulation']['ncon'],
            np.arange(0, 100, 10).astype(float),
        )
        np.testing.assert_array_equal(
            result['simulation']['energy'][:, 0],
            np.arange(0, 300, 30).astype(float),
        )
        np.testing.assert_array_equal(
            result['times'],
            np.arange(0, 0.01, 0.001),
        )
        assert result['timestep'] == pytest.approx(0.001)


class TestExperimentDataIncrementalSave:
    """Incremental mode: save periodically, appending to HDF5."""

    def test_incremental_save_round_trip(self, tmp_hdf5):
        """Simulate periodic saves with append mode."""
        buffer_size = 5
        n_iterations = 12
        data = _make_experiment_data(
            n_iterations=n_iterations,
            buffer_size=buffer_size,
        )

        mode = 'w'
        last_saved = 0

        for i in range(n_iterations):
            idx = i % buffer_size
            data.times[idx] = i * 0.001
            data.simulation.ncon[idx] = float(i * 10)
            data.simulation.niter[idx] = float(i * 20)
            data.simulation.energy[idx, 0] = float(i * 30)
            data.simulation.energy[idx, 1] = float(i * 40)

            # Save at buffer boundaries
            if (i + 1) % buffer_size == 0:
                data.to_file(
                    tmp_hdf5,
                    iteration=i + 1,
                    start_iteration=last_saved,
                    mode=mode,
                )
                mode = 'a'
                last_saved = i + 1

        # Final flush
        if n_iterations > last_saved:
            data.to_file(
                tmp_hdf5,
                iteration=n_iterations,
                start_iteration=last_saved,
                mode=mode,
            )

        # Verify
        result = hdf5_to_dict(tmp_hdf5)
        expected_ncon = np.arange(n_iterations) * 10.0
        np.testing.assert_array_equal(result['simulation']['ncon'], expected_ncon)

        expected_niter = np.arange(n_iterations) * 20.0
        np.testing.assert_array_equal(result['simulation']['niter'], expected_niter)

        expected_energy_pot = np.arange(n_iterations) * 30.0
        np.testing.assert_array_equal(
            result['simulation']['energy'][:, 0],
            expected_energy_pot,
        )

        expected_energy_kin = np.arange(n_iterations) * 40.0
        np.testing.assert_array_equal(
            result['simulation']['energy'][:, 1],
            expected_energy_kin,
        )

        expected_times = np.arange(n_iterations) * 0.001
        np.testing.assert_array_almost_equal(result['times'], expected_times)

    def test_incremental_save_with_skip(self, tmp_hdf5):
        """Save every buffer_size iterations (skip=1)."""
        buffer_size = 3
        skip = 1
        n_iterations = 12
        save_interval = skip * buffer_size  # 3
        data = _make_experiment_data(
            n_iterations=n_iterations,
            buffer_size=buffer_size,
        )

        mode = 'w'
        last_saved = 0

        for i in range(n_iterations):
            idx = i % buffer_size
            data.simulation.ncon[idx] = float(i)

            if (i + 1) % save_interval == 0:
                data.to_file(
                    tmp_hdf5,
                    iteration=i + 1,
                    start_iteration=last_saved,
                    mode=mode,
                )
                mode = 'a'
                last_saved = i + 1

        # Final flush
        if n_iterations > last_saved:
            data.to_file(
                tmp_hdf5,
                iteration=n_iterations,
                start_iteration=last_saved,
                mode=mode,
            )

        result = hdf5_to_dict(tmp_hdf5)
        np.testing.assert_array_equal(
            result['simulation']['ncon'],
            np.arange(n_iterations).astype(float),
        )

    def test_incremental_save_exact_buffer_multiple(self, tmp_hdf5):
        """n_iterations is an exact multiple of buffer_size."""
        buffer_size = 5
        n_iterations = 10
        data = _make_experiment_data(
            n_iterations=n_iterations,
            buffer_size=buffer_size,
        )

        mode = 'w'
        last_saved = 0

        for i in range(n_iterations):
            idx = i % buffer_size
            data.simulation.ncon[idx] = float(i * 2)

            if (i + 1) % buffer_size == 0:
                data.to_file(
                    tmp_hdf5,
                    iteration=i + 1,
                    start_iteration=last_saved,
                    mode=mode,
                )
                mode = 'a'
                last_saved = i + 1

        # No final flush needed (exact multiple)
        assert last_saved == n_iterations

        result = hdf5_to_dict(tmp_hdf5)
        np.testing.assert_array_equal(
            result['simulation']['ncon'],
            np.arange(0, 20, 2).astype(float),
        )
