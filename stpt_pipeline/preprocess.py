import json
import logging
from pathlib import Path

import dask.array as da
import xarray as xr
from zarr import blosc

from imaxt_image.image import TiffImage

from .mosaic_functions import MosaicDict

log = logging.getLogger('owl.daemon.pipeline')
blosc.use_threads = False  # TODO: Check if this makes it quicker or slower


def preprocess(input_dir: Path, output: Path):
    ds = xr.Dataset()
    ds.to_zarr(output, mode='w')
    for s in sorted(input_dir.glob('S???')):
        ds = xr.Dataset()
        tiles = []
        for t in sorted(s.glob('T???')):
            channels = []
            metadata = []
            for f in sorted(t.glob('*.ome.tif')):
                with TiffImage(f) as img:
                    dimg = img.to_dask()
                    metadata = img.metadata.as_dict()
                if dimg.ndim == 2:
                    dimg = dimg[None, :, :]
                channels.append(dimg)
            channels = da.stack(channels)
            tiles.append(channels)
        tiles = da.stack(tiles)
        ntiles, nchannels, nz, ny, nx = tiles.shape
        arr = xr.DataArray(
            tiles,
            dims=('tile', 'channel', 'z', 'y', 'x'),
            name=s.name,
            coords={
                'tile': range(ntiles),
                'channel': range(nchannels),
                'z': range(nz),
                'y': range(ny),
                'x': range(nx),
            },
        )
        arr.attrs['OME'] = json.dumps(metadata)
        ds[s.name] = arr
        ds.attrs['orig_path'] = f'{input_dir}'

        m = MosaicDict()
        _ = [
            m.update({a['@K']: a['#text']})
            for a in metadata['OME']['StructuredAnnotations']['MapAnnotation']['Value'][
                'M'
            ]
            if '#text' in a.keys()
        ]
        for k in m.keys():
            ds.attrs[k] = m[k]

        ds.to_zarr(output, mode='a')
