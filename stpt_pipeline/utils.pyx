from libc.math cimport abs


def get_coords(tuple coords, tuple cof, float center_x, float max_x):
    """[summary]

    Parameters
    ----------
    coords : [type]
        [description]
    cof : [type]
        [description]
    center_x : [type]
        [description]
    max_x : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    cdef float max_desp, dy_cof, xi, r0, r1
    cdef int c0, c1, sign
    cdef tuple res
    c0 = coords[0]
    c1 = coords[1]
    max_desp = cof[0] * c1 ** 2 + cof[1] * c1 + cof[2]
    dy_cof = max_desp / (max_x - center_x) ** 2
    if c0 > center_x:
        sign = 1
    else:
        sign = -1
    xi = abs(c0 - center_x)
    r0 = center_x + sign * (xi + dy_cof * xi ** 2)
    res = (r0, c1)
    return res
