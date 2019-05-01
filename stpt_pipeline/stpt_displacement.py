
import numpy as np

from .settings import Settings


# these two functions are only used to filter and defringe the flat
def med_box(y, half_box=2):
    """[summary]

    Parameters
    ----------
    y : [type]
        [description]
    half_box : int, optional
        [description], by default 2

    Returns
    -------
    [type]
        [description]
    """
    ym = []
    for i in range(len(y)):
        # too close to cero
        i_min = (i - half_box) if (i - half_box) >= 0 else 0
        # too close to end
        i_min = (
            i_min if i_min + 2 * half_box <= len(y) - 1 else len(y) - 1 - 2 * half_box
        )
        ym.append(np.median(y[i_min:i_min + 2 * half_box]))
    #
    return np.array(ym)


def defringe(img):
    """[summary]

    Parameters
    ----------
    img : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    fr_img = img.copy()
    for i in range(fr_img.shape[1]):
        if i < 5:
            t = np.median(img[:, 0:10], 1)
        elif i > fr_img.shape[1] - 5:
            t = np.median(img[:, -10:], 1)
        else:
            t = np.median(img[:, i - 5:i + 5], 1)
        #
        fr_img[:, i] = img[:, i] - med_box(t, 5)
    #
    return fr_img


#
#  get_coords is the function that feeds geometric_transform
# in order to correct for the optical distortion of the detector.
#
def get_coords(coords, cof, center_x, max_x, direct=True):
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
    direct : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """
    max_desp = cof[0] * coords[1] ** 2 + cof[1] * coords[1] + cof[2]
    dy_cof = max_desp / (max_x - center_x) ** 2
    if direct:
        sign = (coords[0] - center_x) / np.abs(coords[0] - center_x)
        if np.isnan(sign):
            sign = 1.0
        xi = np.abs(coords[0] - center_x)
        return (center_x + sign * (xi + dy_cof * xi ** 2), coords[1])
    else:
        xi = np.abs(coords[0] - center_x - cof[2])
        sign = (coords[0] - center_x - cof[2]) / np.abs(coords[0] - center_x - cof[2])
        if np.isnan(sign):
            sign = 1.0
        return (
            center_x + sign * (np.sqrt(1 + 4 * dy_cof * xi) - 1) / (2 * dy_cof),
            coords[1],
        )


def magic_function(x, flat=1):  # TODO: Call this some other name
    """[summary]

    This function transform the raw images into the ones used
    for crossmatching and mosaicing

    Parameters
    ----------
    x : [type]
        [description]
    nflat : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    x_min, x_max = Settings.x_min, Settings.x_max
    y_min, y_max = Settings.y_min, Settings.y_max
    norm_val = Settings.norm_val
    res = np.flipud(x / flat)[x_min:x_max, y_min:y_max] / norm_val
    return res
