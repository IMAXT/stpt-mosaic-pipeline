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
    cdef float x_adim, y_adim, distorted_x, distorted_y, r
    cdef float xc0, xc1, xc2, yc0,yc1,yc2, t0, t1

    c0, c1 = coords
    xc0, xc1, xc2 = cof_x
    yc0, yc1, yc2 = cof_y
    t0, t1 = tan

    x_adim = (c0 - t0) / normal_x
    y_adim = (c1 - t1) / normal_y
    r = sqrt(x_adim * x_adim + y_adim * y_adim)

    distorted_x = c0 + x_adim * (xc0 * r + xc1 * pow(r, 3) + xc2 * pow(r, 5))
    distorted_y = c1 + y_adim * (yc0 * r + yc1 * pow(r, 3) + yc2 * pow(r, 5))

    res = (distorted_x, distorted_y)
    return res
