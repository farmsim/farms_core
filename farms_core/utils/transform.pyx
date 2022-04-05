"""Transform"""

from libc.math cimport asin, atan2, pi, fabs, copysign


cdef void quat_conj(
    DTYPEv1 quat,
    DTYPEv1 out,
) nogil:
    """Quaternion conjugate"""
    cdef unsigned int i
    for i in range(3):
        out[i] = -quat[i]  # x, y, z
    out[3] = quat[3]  # w


cdef void quat_mult(
    DTYPEv1 q0,
    DTYPEv1 q1,
    DTYPEv1 out,
    bint full=1,
) nogil:
    """Hamilton product of two quaternions out = q0*q1"""
    out[0] = q0[3]*q1[0] + q0[0]*q1[3] + q0[1]*q1[2] - q0[2]*q1[1]  # x
    out[1] = q0[3]*q1[1] - q0[0]*q1[2] + q0[1]*q1[3] + q0[2]*q1[0]  # y
    out[2] = q0[3]*q1[2] + q0[0]*q1[1] - q0[1]*q1[0] + q0[2]*q1[3]  # z
    if full:
        out[3] = q0[3]*q1[3] - q0[0]*q1[0] - q0[1]*q1[1] - q0[2]*q1[2]  # w


cdef void quat_rot(
    DTYPEv1 vector,
    DTYPEv1 quat,
    DTYPEv1 quat_c,
    DTYPEv1 tmp4,
    DTYPEv1 out,
) nogil:
    """Quaternion rotation

    :param vector: Vector to rotate
    :param quat: Quaternion rotation
    :param quat_c: Returned quaternion conjugate
    :param tmp4: Temporary quaternion
    :param out: Rotated vector

    """
    quat_c[3] = 0
    quat_c[0] = vector[0]
    quat_c[1] = vector[1]
    quat_c[2] = vector[2]
    quat_mult(quat, quat_c, tmp4)
    quat_conj(quat, quat_c)
    quat_mult(tmp4, quat_c, out, full=0)


cpdef void quat2euler(
    DTYPEv1 quat,
    DTYPEv1 out,
) nogil:
    """Convert Quaternion to Euler angles

    :param vector: Vector to rotate
    :param quat: Quaternion rotation
    :param quat_c: Returned quaternion conjugate
    :param tmp4: Temporary quaternion
    :param out: Rotated vector

    """
    # roll (x-axis rotation)
    cdef double sinr_cosp = 2*(quat[3]*quat[0] + quat[1]*quat[2])
    cdef double cosr_cosp = 1 - 2*(quat[0]*quat[0] + quat[1]*quat[1])
    out[0] = atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    cdef double sinp = 2*(quat[3]*quat[1] - quat[2]*quat[0])
    if fabs(sinp) >= 1:
        out[1] = copysign(pi/2, sinp)  # use 90 degrees if out of range
    else:
        out[1] = asin(sinp)

    # yaw (z-axis rotation)
    cdef double siny_cosp = 2*(quat[3]*quat[2] + quat[0]*quat[1])
    cdef double cosy_cosp = 1 - 2*(quat[1]*quat[1] + quat[2]*quat[2])
    out[2] = atan2(siny_cosp, cosy_cosp)
