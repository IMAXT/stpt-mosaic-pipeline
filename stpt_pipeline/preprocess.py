import logging
import re
import traceback
from contextlib import suppress
from os import listdir
from pathlib import Path

import numpy as np
import zarr
from dask import delayed
from distributed import Client, as_completed
from scipy.ndimage import geometric_transform
from zarr import blosc

from imaxt_image.image import TiffImage
from stpt_pipeline.utils import get_coords

from .mosaic_functions import parse_mosaic_file
from .retry import retry
from .settings import Settings
from .stpt_displacement import defringe, magic_function

log = logging.getLogger('owl.daemon.pipeline')
blosc.use_threads = False  # TODO: Check if this makes it quicker or slower


def read_flatfield(flat_file: Path) -> np.ndarray:
    """Read stored flatfield image.

    Parameters
    ----------
    flat_file
        Full path to flatfield image in npy format.

    Returns
    -------
    [type]
        [description]
    """
    log.info('Reading flatfield %s', flat_file)
    if Settings.do_flat:
        fl = np.load(flat_file)
        channel = Settings.channel_to_use - 1
        flat = fl[:, :, channel]
        if Settings.do_defringe:
            fr_img = defringe(flat)
            nflat = (flat - fr_img) / np.median(flat)
        else:
            nflat = flat / np.median(flat)
        nflat[nflat < 0.5] = 1.0
    else:
        nflat = 1.0
    return nflat


def list_directories(root_dir: Path):
    """[summary]

    This lists all the subdirectories. Once there is an
    standarized naming convention this will have to be edited,
    as at the moment looks for dir names with the '4t1' string
    on them, as all the experiments had this

    Parameters
    ----------
    root_dir : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    dirs = []
    for this_file in listdir(root_dir):
        if this_file.find('4t1') > -1:
            if this_file.find('.txt') > -1:
                continue
            dirs.append(root_dir / this_file)
    dirs.sort()
    return dirs


@retry(Exception)
def save_image(filename, z, section, first_offset, fovs):
    """[summary]

    Parameters
    ----------
    filename : [type]
        [description]
    z : [type]
        [description]
    section : [type]
        [description]
    first_offset : [type]
        [description]
    fovs : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    img = TiffImage(filename)
    match = (
        re.compile(r'(?P<offset>\d+)_(?P<channel>\d\d).tif$')
        .search(filename.name)
        .groupdict()
    )
    offset, channel = int(match['offset']), match['channel']
    zslice = (offset - first_offset) // fovs
    fov = offset - (first_offset + fovs * zslice)
    try:
        g = z.create_group(
            f'section={section}/fov={fov}/z={zslice}/channel={channel}')
    except ValueError:
        g = z[f'section={section}/fov={fov}/z={zslice}/channel={channel}']
    d = g.create_dataset('raw', data=img.asarray(), chunks=False)
    return d


def apply_geometric_transform(d, flat, cof_dist):
    """[summary]

    Parameters
    ----------
    d : [type]
        [description]
    flat : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    import time

    w = time.monotonic()
    if flat is not None:
        cropped = magic_function(d, flat=flat)
    else:
        cropped = d
    w = time.monotonic()
    new = geometric_transform(
        cropped.astype('float32'),
        get_coords,
        output_shape=cropped.shape,
        extra_arguments=(
            cof_dist['cof_x'],
            cof_dist['cof_y'],
            cof_dist['tan'],
            Settings.normal_x,
            Settings.normal_y,
        ),
        mode='constant',
        cval=0.0,
        order=1,
        prefilter=False,
    )
    return new


def geom(d, flat=1, cof_dist=None):
    """[summary]

    Parameters
    ----------
    d : [type]
        [description]
    flat : int, optional
        [description], by default 1

    Returns
    -------
    [type]
        [description]
    """
    assert cof_dist is not None

    new = apply_geometric_transform(d[:], flat, cof_dist)

    fh = zarr.group(store=d.store)
    path = d.path.replace('/raw', '')
    g = fh[path]
    dd = g.create_dataset(
        'geom', data=new.astype('float32'), chunks=False, overwrite=True
    )
    return dd


def preprocess(root_dir: Path, flat_file: Path, output_dir: Path):
    """[summary]

    Parameters
    ----------
    root_dir : Path
        [description]
    flat_file : Path
        [description]
    output_dir : Path
        [description]

    Returns
    -------
    [type]
        [description]
    """

    nflat = delayed(read_flatfield)(flat_file).persist()

    out = f'{output_dir / root_dir.name}.zarr'
    log.info('Storing images in %s', out)
    z = zarr.open(out, mode='a')

    groups = list(z.groups())

    dirs = list_directories(root_dir)
    for d in dirs[:1]:
        # TODO: All this should be metadata
        log.info('Preprocessing %s', d.name)
        section = re.compile(r'\d\d\d\d$').search(d.name).group()
        if f'section={section}' in [name for name, group in groups]:
            log.info('Section %s already preprocessed. Skipping.', section)
            continue
        mosaic = parse_mosaic_file(d)
        mrows, mcolumns = int(mosaic['mrows']), int(mosaic['mcolumns'])
        fovs = mrows * mcolumns
        files = sorted(list(d.glob('*.tif')))
        first_offset = [
            int(re.compile(r'-(\d+)_\d+.tif').search(f.name).groups()[0]) for f in files
        ]
        first_offset = min(first_offset)

        res = []
        for f in d.glob('*.tif'):
            r = delayed(save_image)(f, z, section, first_offset, fovs)
            g = delayed(geom)(r, flat=nflat, cof_dist=Settings.cof_dist)
            res.append(g)

        client = Client.current()
        futures = client.compute(res)

        for fut in as_completed(futures):
            if not fut.exception():
                log.info('%s', fut.result())
            else:
                log.error('%s', fut.exception())
                tb = fut.traceback()
                log.error(traceback.format_tb(tb))
                # TODO: shall we raise the error here?

        z[f'section={section}'].attrs.update(mosaic)
        z[f'section={section}'].attrs['fovs'] = fovs

    with suppress(UnboundLocalError):
        conf = np.ones((int(mosaic['columns']), int(mosaic['rows'])))
        conf = magic_function(conf)

        dist_conf = apply_geometric_transform(conf, None, Settings.cof_dist)

        z.create_dataset('conf', data=conf, chunks=False)
        z.create_dataset('dist_conf', data=dist_conf, chunks=False)

    z.attrs['sections'] = len(dirs)
    return z
