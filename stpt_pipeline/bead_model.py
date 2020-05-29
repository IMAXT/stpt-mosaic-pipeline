import numpy as np
from scipy.special import factorial
#
# Bead fitting procedures
#


def print_theta(theta):
    s = ''
    for i in theta:
        s += '{0:.3f} '.format(i)
    return s
#


def sigmoid(r, rt, c):
    return (1.0 - 1.0 / (1.0 + np.exp(-c * (r**2 - rt**2))))
#


def bead_profile_sigmoid(x, y, x0, y0, rt, p, a, c):
    #
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    return np.abs(p - a * r**2) * sigmoid(r, rt, c)
    # return (p-a*r**2)*(r<=rt)
#


def bead_profile_supergaussian(x, y, x0, y0, p, s, c):
    #
    r = np.sqrt((x - x0)**2 + (y - y0)**2)
    return p * np.exp(-((r / s)**2)**c)


def neg_log_like(theta, xx, yy, zz, do_full, do_print):
    #
    # x0,y0,rt,p,a=theta
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
    # model=1.+bead_profile_2d(xx,yy,x0,y0,rt,p,a,c)
    model = 1. + bead_profile_supergaussian(xx, yy, x0, y0, p, s, c)
    ll = -model + (1. + zz) * np.log(model)
    if do_full:
        ll -= np.log(factorial(1. + zz))
    #
    if do_print:
        print(print_theta(theta), ll.sum())
    return -1.0 * ll.sum()
#


def neg_log_like_conf(theta, xx, yy, zz, conf, do_full, do_print):
    #
    # x0,y0,rt,p,a=theta
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
    # model=1.+bead_profile_2d(xx,yy,x0,y0,rt,p,a,c)
    model = 1. + bead_profile_supergaussian(xx, yy, x0, y0, p, s, c)
    ll = (-model + (1. + zz) * np.log(model)) * conf
    if do_full:
        ll -= np.log(factorial(1. + zz))
    #
    if do_print:
        print(print_theta(theta), np.nansum(ll))
    return -1.0 * np.nansum(ll)
