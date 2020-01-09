#cython: language_level=3

from libc.math cimport abs

def get_coords(tuple coords, list cof_x, list cof_y, list tan, float normal_x, float normal_y):
    """[summary]

    Parameters
    ----------
    tuplecoords : [type]
        [description]
    tuplecofx : [type]
        [description]
    tuplecofy : [type]
        [description]
    tupletan : [type]
        [description]
    floatnormal_x : [type]
        [description]
    floatnormal_y : [type]
        [description]
    """
    cdef int c0, c1
    cdef tuple res
    cdef float x_adim, y_adim, distorted_x, distorted_y, xx, yy
    cdef float xc0, xc1, xc2, xc3, yc0, yc1, yc2, yc3, t0, t1

    c0, c1 = coords
    xc0, xc1, xc2, xc3 = cof_x
    yc0, yc1, yc2, yc3 = cof_y
    t0, t1 = tan

    x_adim = (c0 - t0) / normal_x
    y_adim = (c1 - t1) / normal_y

    xx = x_adim*x_adim
    yy = y_adim*y_adim
    distorted_x = c0 + xc0*x_adim/abs(x_adim) + x_adim*(yy*xc1 + xx*xc2 + xx*y_adim*xc3)
    distorted_y = c1 + yc0*y_adim/abs(y_adim) + y_adim*(xx*yc1 + yy*yc2 + yy*x_adim*yc3)

    res = (distorted_x, distorted_y)
    return res
