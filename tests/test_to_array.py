"""Tests for farms_core.array.array.to_array

Covers the circular-buffer range extraction that underpins the
incremental HDF5 saving feature.
"""

import numpy as np
import pytest

from farms_core.array.array import to_array


class TestToArrayNoIteration:
    """to_array with iteration=None returns the full array."""

    def test_none_iteration_1d(self):
        arr = np.arange(10)
        result = to_array(arr, iteration=None)
        np.testing.assert_array_equal(result, arr)

    def test_none_iteration_2d(self):
        arr = np.arange(20).reshape(10, 2)
        result = to_array(arr, iteration=None)
        np.testing.assert_array_equal(result, arr)

    def test_none_array_returns_none(self):
        assert to_array(None, iteration=None) is None


class TestToArrayPrefixSlice:
    """Backwards-compatible prefix-slice behaviour (start_iteration=None)."""

    def test_prefix_slice_smaller_than_buffer(self):
        arr = np.arange(10)
        result = to_array(arr, iteration=5)
        np.testing.assert_array_equal(result, np.arange(5))

    def test_prefix_slice_equal_to_buffer(self):
        arr = np.arange(10)
        result = to_array(arr, iteration=10)
        np.testing.assert_array_equal(result, np.arange(10))

    def test_prefix_slice_larger_than_buffer_wraps(self):
        """When iteration > buffer_size, use iteration % buffer_size."""
        buf = np.arange(5)
        # iteration=7 -> 7 % 5 = 2 -> buf[:2]
        result = to_array(buf, iteration=7)
        np.testing.assert_array_equal(result, np.arange(2))

    def test_prefix_slice_2d(self):
        arr = np.arange(20).reshape(10, 2)
        result = to_array(arr, iteration=3)
        np.testing.assert_array_equal(result, arr[:3])


class TestToArrayRangeExtraction:
    """Range extraction with start_iteration for incremental saves."""

    def test_single_contiguous_segment(self):
        """Range that fits within the buffer without wrapping."""
        buf = np.arange(5)
        # Extract [2, 5) from buffer of size 5
        result = to_array(buf, iteration=5, start_iteration=2)
        np.testing.assert_array_equal(result, np.array([2, 3, 4]))

    def test_range_from_zero(self):
        buf = np.arange(5)
        result = to_array(buf, iteration=5, start_iteration=0)
        np.testing.assert_array_equal(result, np.arange(5))

    def test_range_wraps_around(self):
        """Range that straddles the buffer boundary."""
        # Buffer: [10, 11, 12, 13, 14] (size 5)
        # Extract global [3, 8): start=3%5=3, n=5, end=8 -> wraps
        # Segment 1: buf[3:5] = [13, 14]
        # Segment 2: buf[0:3] = [10, 11, 12]
        # Result: [13, 14, 10, 11, 12]
        buf = np.arange(10, 15)
        result = to_array(buf, iteration=8, start_iteration=3)
        np.testing.assert_array_equal(result, np.array([13, 14, 10, 11, 12]))

    def test_range_full_buffer_wrap(self):
        """Extract exactly one full buffer starting from non-zero."""
        buf = np.arange(5)
        # Extract [5, 10): start=5%5=0, n=5, end=5 -> no wrap
        result = to_array(buf, iteration=10, start_iteration=5)
        np.testing.assert_array_equal(result, np.arange(5))

    def test_range_partial_after_wrap(self):
        """Partial range after multiple wraps."""
        buf = np.arange(5)
        # Extract [12, 14): start=12%5=2, n=2, end=4 -> no wrap
        # buf[2:4] = [2, 3]
        result = to_array(buf, iteration=14, start_iteration=12)
        np.testing.assert_array_equal(result, np.array([2, 3]))

    def test_range_empty(self):
        """iteration == start_iteration returns empty array."""
        buf = np.arange(5)
        result = to_array(buf, iteration=5, start_iteration=5)
        assert result.shape[0] == 0

    def test_range_negative_returns_empty(self):
        """iteration < start_iteration returns empty array."""
        buf = np.arange(5)
        result = to_array(buf, iteration=3, start_iteration=5)
        assert result.shape[0] == 0

    def test_range_2d_wraps(self):
        """2D array range extraction with wrap-around."""
        buf = np.arange(10).reshape(5, 2)
        # Extract [3, 8): start=3, n=5, end=8 -> wraps
        # Segment 1: buf[3:5] = [[6,7],[8,9]]
        # Segment 2: buf[0:3] = [[0,1],[2,3],[4,5]]
        result = to_array(buf, iteration=8, start_iteration=3)
        expected = np.array([[6, 7], [8, 9], [0, 1], [2, 3], [4, 5]])
        np.testing.assert_array_equal(result, expected)

    def test_range_exactly_at_boundary(self):
        """Range that ends exactly at buffer boundary (no wrap)."""
        buf = np.arange(5)
        # Extract [2, 7): start=2, n=5, end=7 -> wraps (7 > 5)
        # Segment 1: buf[2:5] = [2,3,4]
        # Segment 2: buf[0:2] = [0,1]
        result = to_array(buf, iteration=7, start_iteration=2)
        np.testing.assert_array_equal(result, np.array([2, 3, 4, 0, 1]))

    def test_range_3d(self):
        """3D array range extraction."""
        buf = np.arange(20).reshape(5, 2, 2)
        # Extract [1, 4): start=1, n=3, end=4 -> no wrap
        result = to_array(buf, iteration=4, start_iteration=1)
        np.testing.assert_array_equal(result, buf[1:4])


class TestToArrayIncrementalSavePattern:
    """Simulate the incremental save pattern used by ExperimentLogger."""

    def test_incremental_save_all_segments(self):
        """Write data cyclically and extract at each buffer boundary."""
        buffer_size = 5
        n_iterations = 12
        buf = np.zeros(buffer_size)
        saved_values = []

        for i in range(n_iterations):
            buf[i % buffer_size] = float(i)

            # Save at buffer boundaries
            if (i + 1) % buffer_size == 0:
                start = i + 1 - buffer_size
                end = i + 1
                segment = to_array(buf, iteration=end, start_iteration=start)
                saved_values.extend(list(segment))

        # Final flush
        last_saved = (n_iterations // buffer_size) * buffer_size
        if n_iterations > last_saved:
            segment = to_array(
                buf, iteration=n_iterations, start_iteration=last_saved,
            )
            saved_values.extend(list(segment))

        expected = list(range(n_iterations))
        assert saved_values == expected, f"{saved_values} != {expected}"

    def test_incremental_save_with_skip(self):
        """Save every buffer_size iterations with skip=1 (default).

        Note: skip > 1 would require a buffer of skip*buffer_size to
        avoid data loss, so we only test skip=1 here.
        """
        buffer_size = 3
        n_iterations = 12
        skip = 1
        buf = np.zeros(buffer_size)
        saved_values = []
        save_interval = skip * buffer_size  # 3

        for i in range(n_iterations):
            buf[i % buffer_size] = float(i)

            # Save at every save_interval
            if (i + 1) % save_interval == 0:
                start = i + 1 - save_interval
                end = i + 1
                segment = to_array(buf, iteration=end, start_iteration=start)
                saved_values.extend(list(segment))

        # Final flush
        last_saved = (n_iterations // save_interval) * save_interval
        if n_iterations > last_saved:
            segment = to_array(
                buf, iteration=n_iterations, start_iteration=last_saved,
            )
            saved_values.extend(list(segment))

        expected = [float(v) for v in range(n_iterations)]
        assert saved_values == expected, f"{saved_values} != {expected}"


class TestToArraySkip:
    """Decimation (skip) parameter for sub-sampling output."""

    def test_skip_no_iteration(self):
        """skip with iteration=None decimates the full array."""
        arr = np.arange(10)
        result = to_array(arr, iteration=None, skip=2)
        np.testing.assert_array_equal(result, np.arange(0, 10, 2))

    def test_skip_with_range(self):
        """skip sub-samples the extracted range."""
        buf = np.arange(10)
        result = to_array(buf, iteration=10, start_iteration=0, skip=2)
        np.testing.assert_array_equal(result, np.arange(0, 10, 2))

    def test_skip_with_wrap(self):
        """skip sub-samples across a wrapped range."""
        buf = np.arange(5)
        # Extract [3, 8) with skip=2
        # Full range: [3, 4, 0, 1, 2] (wrapped)
        # skip=2: [3, 0, 2]
        result = to_array(buf, iteration=8, start_iteration=3, skip=2)
        np.testing.assert_array_equal(result, np.array([3, 0, 2]))

    def test_skip_2d(self):
        """skip sub-samples the first dimension of 2D arrays."""
        arr = np.arange(20).reshape(10, 2)
        result = to_array(arr, iteration=10, start_iteration=0, skip=2)
        np.testing.assert_array_equal(result, arr[::2])

    def test_skip_default_is_1(self):
        """Default skip=1 returns full range (no decimation)."""
        buf = np.arange(10)
        result = to_array(buf, iteration=10, start_iteration=0)
        np.testing.assert_array_equal(result, np.arange(10))

    def test_skip_3(self):
        """skip=3 takes every 3rd element."""
        buf = np.arange(12)
        result = to_array(buf, iteration=12, start_iteration=0, skip=3)
        np.testing.assert_array_equal(result, np.arange(0, 12, 3))

    def test_skip_incremental_save_pattern(self):
        """Simulate incremental saves with skip=2 (per-chunk decimation)."""
        buffer_size = 5
        n_iterations = 12
        buf = np.zeros(buffer_size)
        saved_values = []

        for i in range(n_iterations):
            buf[i % buffer_size] = float(i)
            if (i + 1) % buffer_size == 0:
                start = i + 1 - buffer_size
                segment = to_array(
                    buf, iteration=i + 1, start_iteration=start, skip=2,
                )
                saved_values.extend(list(segment))

        last_saved = (n_iterations // buffer_size) * buffer_size
        if n_iterations > last_saved:
            segment = to_array(
                buf, iteration=n_iterations, start_iteration=last_saved,
                skip=2,
            )
            saved_values.extend(list(segment))

        # skip decimates within each chunk, not globally
        # Chunk [0,5): [0,1,2,3,4] -> skip=2 -> [0,2,4]
        # Chunk [5,10): [5,6,7,8,9] -> skip=2 -> [5,7,9]
        # Chunk [10,12): [10,11] -> skip=2 -> [10]
        expected = [0.0, 2.0, 4.0, 5.0, 7.0, 9.0, 10.0]
        assert saved_values == expected, f"{saved_values} != {expected}"
