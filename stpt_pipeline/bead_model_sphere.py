import numpy as np
from owl_dev.logging import logger

#
# Bead fitting procedures
#
# This is a more realistic model of a spherical bead of
# constant density that is cut by an optical plane of
# unknown width and under a layer of substrate of
# unknown optical depth.
#
# r,z are cylindrical coordinates from the bead center,
#  depth is the distance from the sample surface
#
#


def r_from_depth(depth, r_bead):
    """Calculates the r of the spherycal corona for a given depth

    Arguments
    ---------
    depth: float
      ???
    r_bead: float
      radius of the bead, float
    """

    return np.sqrt(r_bead ** 2 - depth ** 2)


def get_z(depth, r_bead, r):
    """Calculates the cylindrical z coordinate for a given depth

    Arguments
    ---------
    depth: float
      ???
    r_bead: float
      radius of the bead
    r: float
        vector of cylindrical coord r to evaluate z at
    """
    z = np.zeros_like(r) + depth

    z[r <= r_bead] = depth - np.sqrt(r_bead**2 - r[r <= r_bead]**2)

    if (depth >= -r_bead) and (depth <= r_bead):

        r_in = r_from_depth(depth, r_bead)

        z[r <= r_in] = 0

    return z


def get_bead_emission(depth, r_bead, bead_em, bgd_trans, r):
    """
        Calculates the bead emission profile as a function of
        cylindrical r

        Inputs:
        -----
            r: vector of cylindrical coord r to evaluate func at
            bgd_trans: transmissivity of the sample medium, float
            bead_em: emission of the bead material per surface unit, float
            r_bead: radius of the bead, float
            depth: float

    """

    z = get_z(depth, r_bead, r)

    all_em = bead_em * 10**(-z * bgd_trans)

    all_em[r > r_bead] = 0

    if depth < 0:
        r_lim = r_from_depth(depth, r_bead)
        all_em[r >= r_lim] = 0

    return all_em


def fit_2d(theta, im, x, y, conf):
    """
        Fits a bead surface brightness profile to a
        bead image.

        There are a couple of checks to avoid unrealistic
        models (negative emissivity and transmissivity), and
        the central coordinates are not allowed to be farther
        than 10px outside the input image.

        Inputs:
        -----
            im: pixel intensity values
            x, y: pixel coordinates associated with the
                values in im
            conf: confidence map associated with im
            theta: vector of parameters, as described in
                get_bead_emission:
                theta[0]: depth
                theta[1]: r_bead
                theta[2]: bead_em
                theta[3]: bgd_trans
                theta[4]: pedestal intensity level
                theta[5]: xc, x coordinate of the center of the bead
                theta[6]: yc, y coordinate of the center of the bead


    """

    xc, yc, = theta[-2:]

    if (xc < -10) or (xc > x.max() + 10):
        return np.inf
    if (yc < -10) or (yc > y.max() + 10):
        return np.inf

    if theta[2] <= 0:
        return np.inf
    if theta[3] <= 0:
        return np.inf

    r = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    f_s = get_bead_emission(*theta[0:4], r) + theta[4]
    res = np.sum(conf * ((im - f_s)**2) / (np.abs(im) + 1.0)) / \
        np.sum(conf.astype(float))

    return res
