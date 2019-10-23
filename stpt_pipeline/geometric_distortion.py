from pathlib import Path
from typing import Dict, List

import dask.array as da
import scipy.ndimage as ndimage
import xarray as xr
from dask import delayed

from .utils import get_coords


def apply_geometric_transform(
    img: da.Array, flat: da.Array, cof_dist: Dict[str, List[float]]
) -> da.Array:
    norm_val = 10_000
    norm = da.flipud(img.astype('float32') / flat) / norm_val
    new = delayed(ndimage.geometric_transform)(
        norm,
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


def distort(
    input: Path, flat: da.Array, cof_dist: Dict[str, List[float]], output: Path
):
    ds = xr.open_zarr(input)
    g = xr.Dataset()
    g.to_zarr(output, mode='w')
    for k in [*ds.keys()]:
        g = xr.Dataset()
        arr = ds[k]
        tiles = []
        for i in arr.tile:
            channels = []
            for j in arr.channel:
                img = da.stack(
                    [
                        apply_geometric_transform(
                            arr.sel(tile=i, channel=j, z=k).data, 1, cof_dist
                        )
                        for k in arr.z
                    ]
                )
                channels.append(img)
            channels = da.stack(channels)
            tiles.append(channels)
        tiles = da.stack(tiles)
        ntiles, nchannels, nz, ny, nx = tiles.shape
        arr = xr.DataArray(
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
        g[k] = arr
        # TODO: transfer also the metadata
        g.to_zarr(output, mode='a')
