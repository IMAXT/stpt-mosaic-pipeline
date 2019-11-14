import logging
import traceback
from functools import partial
from pathlib import Path
from typing import Dict, List

import dask.array as da
import scipy.ndimage as ndimage
import xarray as xr
from dask import delayed
from distributed import Client, as_completed

from .retry import retry
from .settings import Settings
from .stpt_displacement import mask_image
from .utils import get_coords

log = logging.getLogger('owl.daemon.pipeline')


def read_flatfield(flat_file: Path) -> da.Array:
    """Read stored flatfield image.

    Parameters
    ----------
    flat_file
        Full path to flatfield image in npy format.

    Returns
    -------
    [type]
        [description]

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
    log.info('Reading flatfield %s', flat_file)
    if Settings.do_flat:
        flat = xr.open_dataarray(f'{flat_file}')
        nflat = flat[Settings.channel_to_use - 1]
    else:
        nflat = 1.0
    return nflat


def apply_geometric_transform(
    img: da.Array, flat: da.Array, cof_dist: Dict[str, List[float]]
) -> da.Array:
    norm = da.flipud(img / flat)
    masked = delayed(mask_image)(norm)
    masked = da.from_delayed(masked, shape=norm.shape, dtype='float32')
    new = delayed(ndimage.geometric_transform)(
        masked,
        get_coords,
        output_shape=norm.shape,
        extra_arguments=(
            cof_dist['cof_x'],
            cof_dist['cof_y'],
            cof_dist['tan'],
            1000,
            1000,
        ),
        mode='constant',
        cval=0.0,
        order=1,
        prefilter=False,
    )
    res = da.from_delayed(new, shape=norm.shape, dtype='float32')
    return res


@retry(Exception)
def _write_dataset(arr, flat, *, output, cof_dist):
    log.debug('Applying optical distortion and normalization %s', arr.name)
    g = xr.Dataset()
    tiles = []
    for i in arr.tile:
        channels = []
        for j in arr.channel:
            img = da.stack(
                [
                    apply_geometric_transform(
                        arr.sel(tile=i, channel=j, z=k).data / Settings.norm_val,
                        flat,
                        cof_dist,
                    )
                    for k in arr.z
                ]
            )
            channels.append(img)
        channels = da.stack(channels)
        tiles.append(channels)
    tiles = da.stack(tiles)
    ntiles, nchannels, nz, ny, nx = tiles.shape
    this = xr.DataArray(
        tiles,
        dims=('tile', 'channel', 'z', 'y', 'x'),
        name=arr.name,
        coords={
            'tile': range(ntiles),
            'channel': range(nchannels),
            'z': range(nz),
            'y': range(ny),
            'x': range(nx),
        },
    )
    this.attrs = arr.attrs
    g[arr.name] = this

    g.to_zarr(output, mode='a')
    return f'{output}[{arr.name}]'


def distort(input: Path, flat_file: Path, output: Path, nparallel: int = 10):
    client = Client.current()
    ds = xr.Dataset()
    ds.to_zarr(output, mode='w')
    flat = read_flatfield(flat_file).persist()
    ds = xr.open_zarr(input)
    sections = list(ds)

    j = nparallel if len(sections) > nparallel else len(sections)
    func = partial(_write_dataset, flat=flat, output=output, cof_dist=Settings.cof_dist)
    futures = client.map(func, [ds[sections[n]] for n in range(j)])

    seq = as_completed(futures)
    for fut in seq:
        if not fut.exception():
            log.info('Saved %s', fut.result())
            fut.cancel()
            if j < len(sections):
                fut = client.submit(func, ds[sections[j]])
                seq.add(fut)
                j += 1
        else:
            log.error('%s', fut.exception())
            tb = fut.traceback()
            log.error(traceback.format_tb(tb))
