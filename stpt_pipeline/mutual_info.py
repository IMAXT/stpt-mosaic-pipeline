import numpy as np
from scipy.ndimage import shift


def mutual_information(
    im1: np.ndarray, im2: np.ndarray, offset: np.ndarray = None
) -> float:
    r"""Compute the mutual information between two images.

    The mutual information can be defined as:

    .. math::

        I(X; Y) = \sum_{y \in Y} \sum_{x \in X} \ p(x, y) \
        \log \left( \frac{p(x, y)}{p(x) \ p(y)} \right)

    where :math:`p(x,y)` is the joint probability function of X and
    Y and :math:`p(x)` and :math:`p(y)` are the marginal probability
    distribution functions of X and Y respectively.

    Parameters
    ----------
    im1
        reference image
    im2
        target image
    offset
        optional offset in pixels to apply to the target image

    Returns
    -------
    mutual information (MI) metric
    """
    x = offset if offset is not None else (0, 0)
    im2c = shift(im2, (x[1], x[0]), order=1, prefilter=False, mode='constant')
    mask = im2c > 0
    hgram, *_ = np.histogram2d(im1[mask], im2c[mask], bins=20)
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
    return mi


def iqr(im1: np.ndarray, im2: np.ndarray, offset: np.ndarray = None) -> float:
    r"""Compute the information quality ratio betwwen two images.

    The Information Quality Ratio is defined as:

    .. math::

        IQR(X, Y) = \frac{\sum_{y \in Y} \sum_{x \in X} \ p(x, y) \log (p(x) \ p(y))}
                         {\sum_{y \in Y} \sum_{x \in X} \ p(x, y) \log p(x, y)} - 1

    where :math:`p(x,y)` is the joint probability function of X and
    Y and :math:`p(x)` and :math:`p(y)` are the marginal probability
    distribution functions of X and Y respectively.

    Parameters
    ----------
    im1
        reference image
    im2
        target image
    offset
        optional offset in pixels to apply to the target image

    Returns
    -------
    information quality ratio (IQR) metric

    References
    ----------
    Wijiya et al. , "Information Quality Ratio as a novel metric
    for mother wavelet selection", https://doi.org/10.1016/j.chemolab.2016.11.012

    """
    x = offset if offset is not None else (0, 0)
    im2c = shift(im2, (x[1], x[0]), order=1, prefilter=False, mode='constant')
    mask = im2c > 0
    hgram, *_ = np.histogram2d(im1[mask], im2c[mask], bins=20)
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    iqr = np.sum(pxy[nzs] * np.log(px_py[nzs]))
    iqr = iqr / np.sum(pxy[nzs] * np.log(pxy[nzs]))
    return iqr
