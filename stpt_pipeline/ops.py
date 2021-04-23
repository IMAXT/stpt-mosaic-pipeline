import cv2
import dask.array as da
import xarray as xr
import zarr
from dask import delayed
from local_log import to_log, to_debug
from pathlib import Path
import numpy as np


def transform(im):
    res = delayed(cv2.pyrDown)(im.data)
    return da.from_delayed(
        res, shape=(im.shape[0] // 2, im.shape[1] // 2), dtype="int",
    )


def downsample(arr: xr.DataArray, type: str = "uint16") -> xr.DataArray:
    """Downsample a STPT data array a factor of 2

    Parameters
    ----------
    arr
        input data array
    type
        numeric type of the output array

    Returns
    -------
    downsampled array
    """
    nt, nz, nch, ny, nx = arr.chunks
    ny, nx = min(ny), min(nx)
    arr = arr.chunk((1, 1, nch, ny, nx))
    arr_t = []
    for t in arr.type:
        arr_z = []
        for z in arr.z:
            arr_ch = []
            for ch in arr.channel:
                im = arr.sel(z=z.values, type=t.values, channel=ch.values)
                res = transform(im)
                arr_ch.append(res.rechunk(2040, 2040))
            arr_z.append(da.stack(arr_ch))
        arr_t.append(da.stack(arr_z))
    arr_t = da.stack(arr_t)
    nt, nz, nch, ny, nx = arr_t.shape
    narr = xr.DataArray(
        arr_t.astype(type),
        dims=("type", "z", "channel", "y", "x"),
        coords={
            "channel": range(1, nch + 1),
            "type": ["mosaic", "conf", "err"],
            "x": range(nx),
            "y": range(ny),
            "z": range(nz),
        },
    )
    return narr.chunk((1, 1, 10, 2040, 2040))


def build_cals(
    arr: xr.DataArray,
    output_arr: Path
) -> None:

    to_log('Building flats...')

    zarr_dir = output_arr / 'cals.zarr'
    _ = zarr.open(f"{zarr_dir}", mode="w")

    flats = []
    slices = list(arr)

    coords = arr[slices[0]].coords
    channels = coords['channel'].values

    # In case channels non-ordinal
    for i in range(len(channels)):

        this_channel = channels[i]

        to_log('Building channel #' + str(this_channel))

        imgs = da.concatenate(
            [
                arr[t].sel(channel=this_channel) for t in slices
            ],
            axis=0
        )

        # there are some empty tiles, removing, also
        # some nans

        means = imgs.mean(axis=(1, 2)).compute()

        threshold_val = np.median(means)

        flat_frames = np.where(
            (means > threshold_val) &
            (np.isnan(means) == False)
        )[0]

        dark_frames = np.where(
            (means <= threshold_val) &
            (np.isnan(means) == False)
        )[0]

        flats = []
        if len(flat_frames) < 10:
            to_log(
                'CH#' + str(channels[i]) +
                ': Too few frames for good flat'
            )
            flats.append(np.ones(imgs[0, ].shape))

        else:
            temp = imgs[flat_frames, ].mean(axis=0).compute()

            to_debug('Median c#{0:d} = {1:.3f}'.format(i, np.median(temp)))

            flats.append(temp / np.median(temp))

        darks = []
        if len(dark_frames) < 10:
            to_log(
                'CH#' + str(channels[i]) +
                ': Too few frames for good dark'
            )
            darks.append(np.zeros(imgs[0, ].shape))

        else:
            temp = imgs[dark_frames, ].median(axis=0).compute()

            to_debug('Median c#{0:d} = {1:.3f}'.format(i, np.median(temp)))

            darks.append(temp)

    xarr = xr.open_zarr(f"{zarr_dir}")

    temp = xr.DataArray(
        da.array(flats).rechunk((1, *flats[0].shape)).astype('float32'),
        dims=(
            "channel", "y", "x"
        ),
        coords={
            "y": range(flats[0].shape[0]),
            "x": range(flats[0].shape[1]),
            "channel": coords['channel'].values,
        },
    )
    xarr['FLATS'] = temp
    xarr.to_zarr(zarr_dir, mode='a')

    temp = xr.DataArray(
        da.array(darks).rechunk((1, *flats[0].shape)).astype('float32'),
        dims=(
            "channel", "y", "x"
        ),
        coords={
            "y": range(flats[0].shape[0]),
            "x": range(flats[0].shape[1]),
            "channel": coords['channel'].values,
        },
    )
    xarr['DARKS'] = temp
    xarr.to_zarr(zarr_dir, mode='a')
