import cv2
import dask.array as da
import xarray as xr
import zarr
from dask import delayed
from owl_dev.logging import logger
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
) -> Path:

    logger.debug('Building flats...')

    zarr_dir = output_arr / 'cals.zarr'
    _ = zarr.open(f"{zarr_dir}", mode="w")

    flats = []
    flat_medians = []
    n_flats = []
    darks = []
    n_darks = []
    slices = list(arr)

    coords = arr[slices[0]].coords
    channels = coords['channel'].values

    # In case channels non-ordinal
    for i in range(len(channels)):

        this_channel = channels[i]

        logger.info('Building channel #' + str(this_channel))

        _imgs = []
        for this_slice in slices:
            for this_z in coords['z'].values:
                _imgs.append(
                    arr[this_slice].sel(
                        channel=this_channel,
                        z=this_z
                    )
                )

        imgs = da.concatenate(_imgs, axis=0)

        # there are some empty tiles, removing, also
        # some nans

        means = np.array(imgs.mean(axis=(1, 2)).compute())

        threshold_vals = np.quantile(means, (1e-17, 0.5))

        flat_frames = np.where(
            (means >= threshold_vals[1])
        )[0]

        dark_frames = np.where(
            (means <= threshold_vals[0])
        )[0]

        if len(flat_frames) < 10:
            logger.info(
                'CH#' + str(channels[i]) +
                ': Too few frames for good flat'
            )
            flats.append(np.ones(imgs[0, ].shape))

        else:
            temp = imgs[flat_frames, ].mean(axis=0).compute()
            flat_median = np.median(temp)

            logger.debug('Median flat c#{0:d} = {1:.3f}'.format(i, flat_median))

            flats.append(temp / flat_median)
            flat_medians.append(flat_median)
            n_flats.append(len(flat_frames))

        if len(dark_frames) < 10:
            logger.info(
                'CH#' + str(channels[i]) +
                ': Too few frames for good dark'
            )
            darks.append(np.zeros(imgs[0, ].shape))

        else:
            temp = da.median(imgs[dark_frames, ], axis=0).compute()
            dark_median = np.median(temp)

            logger.debug('Median dark c#{0:d} = {1:.3f}'.format(i, dark_median))

            n_darks.append(len(dark_frames))
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

    temp.attrs['N_SAMPLE'] = n_flats
    temp.attrs['MEDIANS'] = flat_medians

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
    temp.attrs['N_SAMPLE'] = n_darks

    xarr['DARKS'] = temp
    xarr.to_zarr(zarr_dir, mode='a')

    return zarr_dir
