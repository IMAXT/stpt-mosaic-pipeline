from pathlib import Path
from typing import Dict, List

import dask.array as da
import numpy as np
import xarray as xr
from dask import delayed
from skimage.transform import warp

from owl_dev.logging import logger

from .settings import Settings
from .stpt_displacement import mask_image
from .utils import get_coords


def read_calib(cal_file: Path) -> xr.DataArray:
    """Read stored flatfield image.

    Parameters
    ----------
    flat_file
        full path to flatfield image in npy format.

    Returns
    -------
    array containing flat field image

    Notes
    -----
    The original flat field file ha been converted to NetCDF4 using

    .. code-block:: python

        flat = np.load('/data/meds1_a/cgonzal/imaxt/stpt_desp_final/flat2.npy')
        ny, nx, nch = flat.shape
        flat = np.stack([flat[:, :, i] for i in range(nch)])
        nflat = flat / np.median(flat, axis=(1, 2), keepdims=True)
        nflat = np.where(nflat < 0.5, 1.0, nflat)
        arr = xr.DataArray(
            nflat.astype('float32'),
            dims=['channel', 'y', 'x'],
            coords={'y': range(ny), 'x': range(nx), 'channel': range(nch)},
        )
        arr.to_netcdf('flat.nc')

    """
    logger.debug("Reading calibration file %s", cal_file)
    cal = xr.open_dataarray(f"{cal_file}")
    return da.from_array(cal.values)


@delayed(pure=True)
def get_matrix(shape, cof_dist):
    coords0, coords1 = np.mgrid[: shape[0], : shape[1]].astype("float")
    for i in range(shape[0]):
        for j in range(shape[1]):
            res = get_coords(
                (i, j),
                cof_dist["cof_x"],
                cof_dist["cof_y"],
                cof_dist["tan"],
                Settings.normal_x,
                Settings.normal_y,
            )
            coords0[i, j] = res[0]
            coords1[i, j] = res[1]
    coords = np.array([coords0, coords1])
    return coords


def apply_geometric_transform(
    img: da.Array, dark: da.Array, flat: da.Array, cof_dist: Dict[str, List[float]]
) -> da.Array:
    """Apply geometric transformation to array

    Parameters
    ----------
    img
        Input image
    flat
        flat field image
    cof_dist
        Distortion coefficients

    Returns
    -------
    distortion corrected array
    """
    norm = da.flipud((img - dark) / da.clip(flat, 1e-6, 1e6))
    masked = delayed(mask_image)(norm)
    masked = da.from_delayed(masked, shape=norm.shape, dtype="float32")
    matrix = get_matrix(masked.shape, cof_dist)
    new = delayed(warp)(masked, matrix)
    res = da.from_delayed(new, shape=img.shape, dtype="float32")
    return res
