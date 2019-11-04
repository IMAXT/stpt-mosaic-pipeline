from libc.math cimport sqrt, pow

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
    cdef float xc0, xc1, xc2, xc3, xc4, yc0, yc1, yc2, yc3, yc4, t0, t1

    c0, c1 = coords
    xc0, xc1, xc2, xc3, xc4 = cof_x
    yc0, yc1, yc2, yc3, yc4 = cof_y
    t0, t1 = tan

    x_adim = (c0 - t0) / normal_x
    y_adim = (c1 - t1) / normal_y

    xx = x_adim*x_adim
    yy = y_adim*y_adim
    distorted_x = c0 + x_adim*(yy*xc0 + xx*xc1 + xx*y_adim*xc2 + xx*yy*xc3 + xx*xx*xc4)
    distorted_y = c1 + y_adim*(xx*yc0 + yy*yc1 + yy*x_adim*yc2 + yy*xx*yc3 + yy*yy*yc4)

    res = (distorted_x, distorted_y)
    return res
