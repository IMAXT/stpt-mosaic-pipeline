import dask.array as da
import pytest
import xarray as xr

from stpt_pipeline import ops


@pytest.fixture
def dataset():
    def _dataset(factor, size=100):
        nx = size * factor
        ny = size * factor
        nz = 1
        nch = 4
        nz = 1
        nt = 3
        arr = xr.DataArray(
            da.ones((nt, nz, nch, ny, nx)),
            dims=("type", "z", "channel", "y", "x"),
            coords={
                "channel": range(1, nch + 1),
                "type": ["mosaic", "conf", "err"],
                "x": range(nx),
                "y": range(ny),
                "z": range(nz),
            },
        )
        return arr

    return _dataset


def test_downsample(dataset):
    ds = dataset(4)
    nt, nz, nch, ny, nx = ds.shape
    arr = ops.downsample(ds)
    assert nt == arr.shape[0]
    assert nz == arr.shape[1]
    assert nch == arr.shape[2]
    assert ny == arr.shape[3] * 2
    assert nx == arr.shape[4] * 2

    # run compute
    arr = arr.compute()
    assert nt == arr.shape[0]
    assert nz == arr.shape[1]
    assert nch == arr.shape[2]
    assert ny == arr.shape[3] * 2
    assert nx == arr.shape[4] * 2


def test_downsample_chunksize(dataset):
    ds = dataset(4, size=540)
    nt, nz, nch, ny, nx = ds.shape
    arr = ops.downsample(ds)
    arr = ops.downsample(arr)
    arr = ops.downsample(arr)
    arr = ops.downsample(arr)
    arr.compute()
    assert nt == arr.shape[0]
    assert nz == arr.shape[1]
    assert nch == arr.shape[2]
    assert ny == arr.shape[3] * 16
    assert nx == arr.shape[4] * 16
