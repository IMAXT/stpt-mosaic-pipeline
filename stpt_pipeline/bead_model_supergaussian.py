import numpy as np
from scipy.special import factorial


def sigmoid(r, rt, c):
    return 1.0 - 1.0 / (1.0 + np.exp(-c * (r ** 2 - rt ** 2)))


def bead_profile_sigmoid(x, y, x0, y0, rt, p, a, c):
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return np.abs(p - a * r ** 2) * sigmoid(r, rt, c)


def bead_profile_supergaussian(x, y, x0, y0, p, s, c):
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return p * np.exp(-(((r / s) ** 2) ** c))


def neg_log_like_conf(theta, xx, yy, zz, conf, do_full=False):
    x0, y0, p, s, c = theta
    if x0 < -10:
        return np.inf
    if y0 < -10:
        return np.inf
    if c < 1:
        return np.inf
    if s < 0.0:
        return np.inf
    if p < 0.0:
        return np.inf
    model = 1.0 + bead_profile_supergaussian(xx, yy, x0, y0, p, s, c)
    ll = (-model + (1.0 + zz) * np.log(model)) * conf

    if do_full:
        ll -= np.log(factorial(1.0 + zz))

    return -1.0 * np.nansum(ll)
