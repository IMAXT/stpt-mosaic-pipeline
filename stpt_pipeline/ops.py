import cv2
import dask.array as da
import xarray as xr


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
    arr = arr.chunk((1, 1, 1, ny, nx))
    arr_t = []
    for t in arr.type:
        arr_z = []
        for z in arr.z:
            arr_ch = []
            for ch in arr.channel:
                im = arr.sel(z=z.values, type=t.values, channel=ch.values)
                res = im.data.map_blocks(cv2.pyrDown, chunks=(ny // 2, nx // 2))
                arr_ch.append(res.rechunk(ny, nx))
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
    return narr