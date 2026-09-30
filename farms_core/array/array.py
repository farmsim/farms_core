"""Array"""

import numpy as np
from .types import NDARRAY


class AppendableArray:
    """Wrapper marking a numpy array as time-varying (appendable).

    In incremental save mode, to_dict methods produce a mix of
    static data (same every save, should be overwritten) and time-varying
    data (new chunk each save, should be appended).  Wrapping time-varying
    arrays in AppendableArray lets the HDF5 layer
    (:func:`farms_core.io.hdf5._val_to_hdf5`) distinguish the
    two by checking isinstance(value, AppendableArray).

    In write mode (mode='w') the wrapper is transparent — the inner
    array is stored as usual with maxshape so it can be appended to
    in subsequent saves.  In append mode (mode='a') the wrapper
    signals that the data should be appended along the first dimension.
    """

    __slots__ = ('array',)

    def __init__(self, array):
        self.array = np.asarray(array)

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.array, dtype)

    def __iter__(self):
        return iter(self.array)

    def __len__(self):
        return len(self.array)

    def __getitem__(self, key):
        return self.array[key]

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    @property
    def dtype(self):
        return self.array.dtype


def to_array(
        array: NDARRAY,
        iteration: int | None = None,
        start_iteration: int | None = None,
        skip: int = 1,
) -> NDARRAY:
    """Extract data from a (possibly circular) buffer array.

    Parameters
    ----------
    array:
        The buffer array.  The first dimension is the time/iteration axis.
    iteration:
        The global simulation iteration up to which (exclusive) data should
        be returned.  ``None`` returns the full array unchanged (as a
        plain ``numpy.ndarray``).  When not ``None``, the result is
        wrapped in :class:`AppendableArray` so that ``dict_to_hdf5``
        knows to append it along the first dimension in incremental
        (append-mode) saves.
    start_iteration:
        The global simulation iteration from which (inclusive) data should
        be returned.  Only meaningful together with *iteration*.  When
        provided, only the slice ``[start_iteration, iteration)`` is
        returned — this is used for incremental (append) saves so that
        only the new data since the previous save is written.
    skip:
        Decimation factor for the output.  When ``skip > 1``, only every
        ``skip``-th iteration is included in the result (i.e. the
        extracted range is sub-sampled with ``[::skip]``).  Default is
        1 (no decimation).

    Notes
    -----
    The buffer may be smaller than the total number of iterations
    (``buffer_size < n_iterations``).  In that case data is written
    cyclically at ``index = global_iteration % buffer_size``.  When
    extracting a range that straddles the wrap point the two segments
    are concatenated in chronological order.

    When *start_iteration* is ``None`` the behaviour is backwards
    compatible with the original prefix-slice logic.
    """
    if array is None:
        return None
    array = np.array(array)
    if iteration is None:
        return array[::skip] if skip > 1 else array
    buffer_size = array.shape[0]
    if start_iteration is not None:
        # Compute the number of elements to extract
        n_elements = iteration - start_iteration
        if n_elements <= 0:
            # Empty range (nothing new to save)
            return AppendableArray(array[0:0])
        # Map global iterations to buffer indices
        start = start_iteration % buffer_size
        end = start + n_elements
        if end <= buffer_size:
            # Single contiguous segment
            result = array[start:end]
        else:
            # Segment wraps around the end of the buffer
            wrap = end - buffer_size
            result = np.concatenate((array[start:], array[:wrap]))
        return AppendableArray(result[::skip] if skip > 1 else result)
    # Backwards-compatible prefix slice
    if iteration > buffer_size:
        iteration = iteration % buffer_size
    result = array[:iteration]
    return AppendableArray(result[::skip] if skip > 1 else result)
