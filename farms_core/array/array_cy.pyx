"""Arrays"""

import os
import numpy as np
cimport numpy as np


cdef class Array(object):
    """Array"""

    cpdef unsigned int size(self, unsigned int index):
        """Shape"""
        return self.array.shape[index]

    def copy_array(self):
        """Copy array"""
        return np.copy(self.array)

    def log(self, times, folder, name, extension):
        """Log data"""
        os.makedirs(folder, exist_ok=True)
        if extension == 'npy':
            save_function = np.save
            nosplit = True
        elif extension in ('txt', 'csv'):
            save_function = np.savetxt
            nosplit = False
        else:
            raise Exception(
                f'Format {extension} is not valid for logging array'
            )
        if nosplit or self.array.ndim == 2:
            path = folder + '/' + name + '.' + extension
            save_function(path, self.array[:len(times)])
        elif self.array.ndim == 3:
            for i in range(np.shape(self.array)[1]):
                path = folder+'/'+name+f'_{i}.'+extension
                save_function(path, self.array[:len(times), i])
        else:
            raise Exception(
                f'Dimensionality {self.array.ndim}'
                f' is not valid for extension of type {extension}'
            )


cdef class DoubleArray1D(Array):
    """Double array"""

    def __init__(self, array):
        super(DoubleArray1D, self).__init__()
        self.array = array


cdef class DoubleArray2D(Array):
    """Double array"""

    def __init__(self, array):
        super(DoubleArray2D, self).__init__()
        self.array = array


cdef class DoubleArray3D(Array):
    """Double array"""

    def __init__(self, array):
        super(DoubleArray3D, self).__init__()
        self.array = array


cdef class IntegerArray1D(Array):
    """Integer array"""

    def __init__(self, array):
        super(IntegerArray1D, self).__init__()
        self.array = array


cdef class IntegerArray2D(Array):
    """Integer array"""

    def __init__(self, array):
        super(IntegerArray2D, self).__init__()
        self.array = array
