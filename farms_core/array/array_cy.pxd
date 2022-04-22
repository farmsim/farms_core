"""Arrays"""

include 'types.pxd'


cdef class Array:
    """Array"""
    cpdef public unsigned int size(self, unsigned int index)


cdef class DoubleArray1D(Array):
    """Double array"""
    cdef readonly DTYPEv1 array


cdef class DoubleArray2D(Array):
    """Double array"""
    cdef readonly DTYPEv2 array


cdef class DoubleArray3D(Array):
    """Double array"""
    cdef readonly DTYPEv3 array


cdef class IntegerArray1D(Array):
    """Unsigned integer array"""
    cdef readonly UITYPEv1 array


cdef class IntegerArray2D(Array):
    """Unsigned integer array"""
    cdef readonly UITYPEv2 array
