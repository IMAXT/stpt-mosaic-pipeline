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
        ym.append(np.median(y[i_min : i_min + 2 * half_box]))  # noqa: E203

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
            t = np.median(img[:, i - 5 : i + 5], 1)  # noqa: E203

        fr_img[:, i] = img[:, i] - med_box(t, 5)

    return fr_img


def mask_image(img):
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
    mask = np.zeros_like(img)
    mask[x_min:x_max, y_min:y_max] = 1
    res = img * mask
    return res
