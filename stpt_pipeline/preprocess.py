import json
import logging
import traceback
from collections import defaultdict
from functools import partial
from pathlib import Path

import dask.array as da
import xarray as xr
from distributed import Client, as_completed
from zarr import blosc

from imaxt_image.io import TiffImage

from .retry import retry

log = logging.getLogger('owl.daemon.pipeline')
blosc.use_threads = False  # TODO: Check if this makes it quicker or slower


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


@retry(Exception)
def _write_dataset(s, *, input_dir, output):
    log.debug('Preprocessing %s', s)
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

    m = MosaicDict()
    _ = [
        m.update({a['@K']: a['#text']})
        for a in metadata['OME']['StructuredAnnotations']['MapAnnotation']['Value']['M']
        if '#text' in a.keys()
    ]

    arr.attrs['OME'] = [json.loads(json.dumps(metadata))]
    for k in m.keys():
        arr.attrs[k] = m[k]

    ds[s.name] = arr

    ds.attrs['orig_path'] = f'{input_dir}'

    ds.to_zarr(output, mode='a')
    return f'{output}[{s.name}]'


def preprocess(input_dir: Path, output: Path, nparallel: int = 1):
    """Convert input OME-TIFF format to Xarray

    Read all files for each section and write an Xarr using
    the Zarr backend.

    Parameters
    ----------
    input_dir
        Directory containing sample
    output
        Output directory
    nparallel
        number of sections to run in parallel
    """
    ds = xr.Dataset()
    ds.to_zarr(output, mode='w')
    sections = sorted(input_dir.glob('S???'))
    client = Client.current()
    j = nparallel if len(sections) > nparallel else len(sections)
    func = partial(_write_dataset, input_dir=input_dir, output=output)
    futures = client.map(func, sections[:j])

    seq = as_completed(futures)
    for fut in seq:
        if not fut.exception():
            log.info('Saved %s', fut.result())
            fut.cancel()
            if j < len(sections):
                fut = client.submit(func, sections[j])
                seq.add(fut)
                j += 1
        else:
            log.error('%s', fut.exception())
            tb = fut.traceback()
            log.error(traceback.format_tb(tb))
