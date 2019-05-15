from collections import defaultdict
from os import listdir
from pathlib import Path
from typing import Dict

import dask.array as da
import numpy as np

from imaxt_image.image import TiffImage


def find_delta(dx, dy):
    """[summary]

    Parameters
    ----------
    dx : [type]
        [description]
    dy : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    #
    # In theory each position in the mosaic stage is offset
    # by a fixed value, but there's some jitter in the microscope
    # so this finds this delta. It is used later to get the column
    # and row position of each single image
    #
    r = np.sqrt((dx.astype(float) - dx[0]) ** 2 + (dy.astype(float) - dy[0]) ** 2)
    r[0] = r.max()  # avoiding minimum at itself
    # first candidate
    dx_1 = np.abs(dx[r.argmin()] - dx[0])
    dy_1 = np.abs(dy[r.argmin()] - dy[0])
    # second candidate
    r[r.argmin()] = r.max()
    dx_2 = np.abs(dx[r.argmin()] - dx[0])
    dy_2 = np.abs(dy[r.argmin()] - dy[0])
    #
    return np.max([dx_1, dx_2]), np.max([dy_1, dy_2])


def get_img_list(root_dir, mosaic_size=56):
    """[summary]

    Parameters
    ----------
    root_dir : [type]
        [description]
    mosaic_size : int, optional
        [description], by default 56

    Returns
    -------
    [type]
        [description]
    """
    #
    # Scans the directory, finds the images
    #  and returns the img names, the channel number,
    # the corresponding optical slice and the
    # running image number. This requires the naming
    # convention to be constant, so it is likely to
    #  crash periodically when the STPT machine changes
    # naming conventions.
    #
    imgs = []
    img_num = []
    optical_slice = []
    channel = []
    for this_file in listdir(root_dir):
        if (
            (this_file.find('tif') > -1)
            & (this_file.find('imaxt') == -1)
            & (this_file[0] != '.')
        ):
            imgs.append(this_file)
            temp = this_file.split('.')[0].split('-')[-1].split('_')
            img_num.append(float(temp[0]))
            channel.append(float(temp[1]))
            optical_slice.append(int(img_num[-1] / mosaic_size) + 1)
    imgs = np.array(imgs)
    i_ord = np.argsort(img_num)
    imgs = imgs[i_ord]
    img_num = np.array(img_num)[i_ord]
    channel = np.array(channel)[i_ord]
    optical_slice = np.array(optical_slice)[i_ord]
    #
    return imgs, img_num, channel, optical_slice


def get_mosaic_file(root_dir):
    """[summary]

    Parameters
    ----------
    root_dir : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # just locates the name of the stage mosaic text file
    # inside a provided dir
    for this_file in listdir(root_dir):
        if (this_file.find('osaic') > -1) & (this_file.find('txt') > -1):
            return this_file
    #
    return ''


class MosaicDict(defaultdict):
    def __init__(self):
        super().__init__()
        self.default_factory = list

    def update(self, item):
        key, val = list(item.items())[0]
        if 'XPos' in key:
            self['XPos'].append(int(val))
        elif 'YPos' in key:
            self['YPos'].append(int(val))
        else:
            super().update(item)


def parse_mosaic_file(root_dir: Path) -> Dict[str, str]:
    """Return contents of mosaic file as dictionary.

    Parameters
    ----------
    root_dir
        Directory containing mosaic file

    Returns
    -------
    dictionary of key, values with mosaic file contents
    """
    mosaic_file = get_mosaic_file(root_dir)
    with open(root_dir / mosaic_file) as fh:
        items = [item.strip().split(':', 1) for item in fh.readlines()]
        res = MosaicDict()
        [res.update({k[0]: k[1]}) for k in items]
        return res


def get_img_cube(
    root_dir,
    imgs,
    img_num,
    channel,
    optical_slice,
    channel_to_use=4,
    slice_to_use=1,
    dtype='float32',
):
    """[summary]

    Parameters
    ----------
    root_dir : [type]
        [description]
    imgs : [type]
        [description]
    img_num : [type]
        [description]
    channel : [type]
        [description]
    optical_slice : [type]
        [description]
    channel_to_use : int, optional
        [description], by default 4
    slice_to_use : int, optional
        [description], by default 1

    Returns
    -------
    [type]
        [description]
    """
    # Reads all the tiff files for a given channel and slice, and stores them into a cube.
    #
    ii = np.where(
        (channel == channel_to_use)
        & ((optical_slice - np.min(optical_slice) + 1) == slice_to_use)
    )[0]
    im_t = [TiffImage(root_dir / t) for t in imgs[ii]]
    im_t = [da.from_array(t, chunks=t.shape) for t in im_t]
    cube = da.stack(im_t)
    if dtype:
        cube = cube.astype(dtype)
    return cube
