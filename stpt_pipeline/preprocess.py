import logging
import re
from os import listdir
from pathlib import Path

import numpy as np
import zarr
from dask import delayed
from distributed import Client, wait
from scipy.ndimage import geometric_transform

from imaxt_image.image import TiffImage
from stpt_pipeline.utils import get_coords

from .mosaic_functions import get_mosaic_file
from .settings import Settings
from .stpt_displacement import defringe, magic_function

log = logging.getLogger('owl.daemon.pipeline')


def read_flatfield(flat_file: Path):
    """[summary]

    Parameters
    ----------
    flat_file : [type]
        [description]

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


def save_image(filename, z, section, first_offset, layers):
    img = TiffImage(filename)
    match = (
        re.compile(r'(?P<offset>\d+)_(?P<channel>\d\d).tif$')
        .search(filename.name)
        .groupdict()
    )
    offset, channel = int(match['offset']), match['channel']
    fov = (offset - first_offset) // layers
    zslice = (offset - first_offset) - fov * layers
    g = z.create_group(f'section={section}/fov={fov}/z={zslice}/channel={channel}')
    d = g.create_dataset('raw', data=img.asarray(), chunks=False)
    print(d)
    return d


def geom(d, flat=1):
    shapes = (Settings.x_max - Settings.x_min, Settings.y_max - Settings.y_min)
    cropped = magic_function(d[:], flat=flat)
    new = geometric_transform(
        cropped.astype('float32'),
        get_coords,
        output_shape=shapes,
        extra_arguments=(Settings.cof_dist, shapes[0] * 0.5, shapes[0] * 1.0),
        mode='constant',
        cval=0.0,
        order=1,
        prefilter=False,
    )

    fh = zarr.group(store=d.store)
    path = d.path.replace('/raw', '')
    g = fh[path]
    dd = g.create_dataset(
        'geom', data=new.astype('float32'), chunks=False, overwrite=True
    )
    print(dd)
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
    z = zarr.group(out, 'w')

    dirs = list_directories(root_dir)
    for d in dirs:
        # TODO: All this should be metadata
        section = re.compile(r'\d\d\d\d$').search(d.name).group()
        mosaic_file = get_mosaic_file(d)
        res = open(d / mosaic_file, 'r').read()
        mrows = re.compile(r'mrows:(\d+)\n').search(res).groups()[0]
        mcolumns = re.compile(r'mcolumns:(\d+)\n').search(res).groups()[0]
        mrows, mcolumns = int(mrows), int(mcolumns)
        fovs = mrows * mcolumns
        files = sorted(list(d.glob('*.tif')))
        files_ch1 = sorted(list(d.glob('*_01.tif')))
        layers = len(files_ch1) // fovs
        first_offset = int(
            re.compile(r'-(\d+)_\d+.tif').search(files[0].name).groups()[0]
        )

        res = []
        for f in d.glob('*.tif'):
            r = delayed(save_image)(f, z, section, first_offset, layers)
            g = delayed(geom)(r, flat=nflat)
            res.append(g)

        client = Client.current()
        fut = client.compute(res)
        wait(fut)

    # Write metadata
    z.attrs['sections'] = len(dirs)  # TODO: Read from mosaic file
    z.attrs['fovs'] = fovs
    z.attrs['mrows'] = mrows
    z.attrs['mcolumns'] = mcolumns
    z.attrs['layers'] = layers
    return z
