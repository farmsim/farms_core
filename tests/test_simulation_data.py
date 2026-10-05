"""Tests for farms_core.simulation.data.SimulationData

Covers the buffer-size-aware allocation and circular-buffer
to_dict extraction.
"""

import numpy as np
import pytest

from farms_core.simulation.data import SimulationData
from farms_core.array.array import to_array


class TestSimulationDataFromSize:
    """SimulationData.from_size allocation with buffer_size."""

    def test_default_no_buffer_size(self):
        """Without buffer_size, arrays are allocated at size."""
        sd = SimulationData.from_size(1000)
        assert sd.ncon.shape == (1000,)
        assert sd.niter.shape == (1000,)
        assert sd.energy.shape == (1000, 2)

    def test_buffer_size_smaller_than_size(self):
        """buffer_size < size allocates at buffer_size."""
        sd = SimulationData.from_size(1000, buffer_size=100)
        assert sd.ncon.shape == (100,)
        assert sd.niter.shape == (100,)
        assert sd.energy.shape == (100, 2)

    def test_buffer_size_equal_to_size(self):
        """buffer_size == size allocates at size."""
        sd = SimulationData.from_size(500, buffer_size=500)
        assert sd.ncon.shape == (500,)
        assert sd.niter.shape == (500,)
        assert sd.energy.shape == (500, 2)

    def test_buffer_size_larger_than_size(self):
        """buffer_size > size clamps to size."""
        sd = SimulationData.from_size(100, buffer_size=1000)
        assert sd.ncon.shape == (100,)
        assert sd.niter.shape == (100,)
        assert sd.energy.shape == (100, 2)

    def test_buffer_size_zero(self):
        """buffer_size == 0 falls back to size."""
        sd = SimulationData.from_size(1000, buffer_size=0)
        assert sd.ncon.shape == (1000,)
        assert sd.niter.shape == (1000,)
        assert sd.energy.shape == (1000, 2)

    def test_buffer_size_none(self):
        """buffer_size=None falls back to size."""
        sd = SimulationData.from_size(1000, buffer_size=None)
        assert sd.ncon.shape == (1000,)
        assert sd.niter.shape == (1000,)
        assert sd.energy.shape == (1000, 2)

    def test_arrays_are_zero_initialized(self):
        """All arrays start at zero."""
        sd = SimulationData.from_size(100, buffer_size=10)
        assert np.all(sd.ncon == 0)
        assert np.all(sd.niter == 0)
        assert np.all(sd.energy == 0)


class TestSimulationDataToDict:
    """SimulationData.to_dict with circular buffer extraction."""

    def test_to_dict_full_buffer_no_iteration(self):
        """to_dict() with no iteration returns full buffer."""
        sd = SimulationData.from_size(10)
        sd.ncon[:] = np.arange(10)
        sd.niter[:] = np.arange(10) * 2
        result = sd.to_dict()
        np.testing.assert_array_equal(result['ncon'], np.arange(10))
        np.testing.assert_array_equal(result['niter'], np.arange(10) * 2)

    def test_to_dict_prefix_slice(self):
        """to_dict(iteration=5) returns first 5 elements."""
        sd = SimulationData.from_size(10)
        sd.ncon[:] = np.arange(10)
        result = sd.to_dict(iteration=5)
        np.testing.assert_array_equal(result['ncon'], np.arange(5))

    def test_to_dict_range_extraction_no_wrap(self):
        """Range extraction without wrap-around."""
        sd = SimulationData.from_size(10, buffer_size=5)
        # Write 5 values
        for i in range(5):
            sd.ncon[i] = float(i * 10)
        result = sd.to_dict(iteration=5, start_iteration=0)
        np.testing.assert_array_equal(result['ncon'], np.arange(0, 50, 10))

    def test_to_dict_range_extraction_with_wrap(self):
        """Range extraction with wrap-around."""
        sd = SimulationData.from_size(10, buffer_size=5)
        # Write 8 values (wraps once)
        for i in range(8):
            sd.ncon[i % 5] = float(i * 10)
        # Extract [3, 8): values for iterations 3,4,5,6,7
        # Buffer: [50, 60, 70, 30, 40] (iterations 5,6,7,3,4)
        # start=3%5=3, n=5, end=8 -> wraps
        # buf[3:5] = [30, 40], buf[0:3] = [50, 60, 70]
        # Result: [30, 40, 50, 60, 70]
        result = sd.to_dict(iteration=8, start_iteration=3)
        np.testing.assert_array_equal(result['ncon'], np.array([30, 40, 50, 60, 70]))

    def test_to_dict_incremental_save_pattern(self):
        """Simulate the incremental save pattern."""
        buffer_size = 5
        n_iterations = 12
        sd = SimulationData.from_size(n_iterations, buffer_size=buffer_size)

        saved_ncon = []
        saved_energy = []

        for i in range(n_iterations):
            idx = i % buffer_size
            sd.ncon[idx] = float(i * 10)
            sd.energy[idx, :] = [i * 30, i * 40]

            if (i + 1) % buffer_size == 0:
                start = i + 1 - buffer_size
                end = i + 1
                result = sd.to_dict(iteration=end, start_iteration=start)
                saved_ncon.extend(list(result['ncon']))
                saved_energy.extend(list(result['energy'][:, 0]))

        # Final flush
        last_saved = (n_iterations // buffer_size) * buffer_size
        if n_iterations > last_saved:
            result = sd.to_dict(
                iteration=n_iterations, start_iteration=last_saved,
            )
            saved_ncon.extend(list(result['ncon']))
            saved_energy.extend(list(result['energy'][:, 0]))

        expected_ncon = [float(i * 10) for i in range(n_iterations)]
        assert saved_ncon == expected_ncon, f"{saved_ncon} != {expected_ncon}"

        expected_energy = [float(i * 30) for i in range(n_iterations)]
        assert saved_energy == expected_energy, f"{saved_energy} != {expected_energy}"


class TestSimulationDataFromDict:
    """SimulationData.from_dict round-trip."""

    def test_from_dict_round_trip(self):
        """from_dict creates SimulationData from dictionary data."""
        original = SimulationData.from_size(10)
        original.ncon[:] = np.arange(10)
        original.niter[:] = np.arange(10) * 2
        original.energy[:, 0] = np.arange(10) * 3
        original.energy[:, 1] = np.arange(10) * 4

        d = original.to_dict()
        restored = SimulationData.from_dict(d)

        np.testing.assert_array_equal(restored.ncon, original.ncon)
        np.testing.assert_array_equal(restored.niter, original.niter)
        np.testing.assert_array_equal(restored.energy, original.energy)
