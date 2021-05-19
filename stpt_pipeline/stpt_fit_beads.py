from pathlib import Path

import dask
import dask.array as da
import numpy as np
import scipy.ndimage as ndi
import xarray as xr
import zarr
from dask import delayed
from owl_dev.logging import logger
from scipy.optimize import minimize
from scipy.stats import median_absolute_deviation as mad
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from keras.models import load_model

from .bead_model_sphere import fit_2d
from .settings import Settings


def _rezoom(lab, final_shape):

    zoom_fac = (
        np.float(final_shape[0] / lab.shape[0]),
        np.float(final_shape[1] / lab.shape[1])
    )

    return np.round(ndi.zoom(lab, zoom_fac, order=0)).astype(int)


@delayed
def fit_bead(
    fit_im, labels,
    this_bead: dict,
    bscale: float,
    bzero: float,
    first_pass: bool,
    do_print: bool
) -> dict:
    """
    Fits a profile to a bead in an STPT mosaic image.
    If first_pass is no, this is assumed to be a downsampled image

    Parameters
    ----------
    fit_im:
        Mosaic dataarray with the beads
    labels:
        labeled bead mask
    this_bead:
        fit parameters dictionary of the bead, in case of
        first pass, inly this_bead['id'] is used
    bscale, bzero:
        scale and pedestal of fit_im
    first_pass:
        toggles first quick fit or full res fit
    do_print:
        if on, verbose fit
    """

    padding = 10  # px to add to cutouts

    fit_res = {}

    fit_res['success'] = False

    if first_pass:
        i_x, i_y = np.where(labels == this_bead['id'])

        r = 0.5 * (
            np.max(np.abs(i_x - np.mean(i_x))) +
            np.max(np.abs(i_y - np.mean(i_y)))
        )

        center = (
            int(np.mean(i_x)),
            int(np.mean(i_y))
        )

        # if too small or too big likely to be interloper
        if r < 2:
            return fit_res

        if r > 100:
            return fit_res

    else:
        r = this_bead['r'] * Settings.zoom_level

        center = (
            np.clip(
                int(this_bead['x'] * Settings.zoom_level),
                0,
                fit_im.shape[0] - 1
            ),
            np.clip(
                int(this_bead['y'] * Settings.zoom_level),
                0,
                fit_im.shape[1] - 1
            )
        )

        # if too small or too big likely to be interloper
        if r < 10:
            return fit_res

        if r > 300:
            return fit_res

    logger.debug(this_bead['id'], r)

    half_width = int(r)

    # cutout lower left corner
    ll = [
        int(np.clip(
            center[0] - half_width - padding,
            0,
            fit_im.shape[0] - 2 * (half_width + padding)
        )),
        int(np.clip(
            center[1] - half_width - padding,
            0, fit_im.shape[1] - 2 * (half_width + padding)
        ))
    ]

    im_cutout = fit_im[
        ll[0]:ll[0] + 2 * (half_width + padding),
        ll[1]:ll[1] + 2 * (half_width + padding)
    ] * bscale + bzero

    # Because labels is done over a pyrdown image,
    # in case of full fit, need to resample up

    if first_pass:

        labels_zoom = labels[
            ll[0]:ll[0] + 2 * (half_width + padding),
            ll[1]:ll[1] + 2 * (half_width + padding)
        ]

    else:

        small_corner = [
            int(np.round(ll[0] / Settings.zoom_level)),
            int(np.round(ll[1] / Settings.zoom_level)),
        ]
        small_width = int(
            np.round((2 * (half_width + padding)) / Settings.zoom_level))

        labels_small = labels[
            small_corner[0]:small_corner[0] + small_width,
            small_corner[1]:small_corner[1] + small_width
        ]

        labels_zoom = _rezoom(labels_small, im_cutout.shape)

    conf_cutout = (
        (labels_zoom == this_bead['id']) |
        (labels_zoom == 0)
    ) * 1.0

    x2d = np.array([
        [t * 1.0] * conf_cutout.shape[1]
        for t in range(conf_cutout.shape[0])
    ])
    y2d = np.array([
        np.arange(conf_cutout.shape[1])
        for t in range(conf_cutout.shape[0])
    ])

    central_val = fit_im[center[0], center[1]]
    pedestal = np.clip(im_cutout.min(), 0, 1e7)

    if first_pass:
        # because we are setting the parameters blindly,
        # the two cases are different enough that we need
        # to try for both sets of initial paremeters
        # and compare the final residuals to choose
        #
        # first try assuming cut bead
        depth_ini = 0.5 * r
        theta1 = [
            depth_ini,
            r,
            (central_val - pedestal),
            0.1,
            pedestal,
            center[0] - ll[0],
            center[1] - ll[1]
        ]

        fit_1 = minimize(
            fit_2d, theta1,
            args=(
                im_cutout.flatten(),
                x2d.flatten(), y2d.flatten(),
                conf_cutout.flatten()
            ),
            method='Powell'
        )

        # second assuming embedded bead

        depth_ini = 1.1 * r

        theta2 = [
            depth_ini,
            r,
            (central_val - pedestal) * 10,
            0.1,
            pedestal,
            center[0] - ll[0],
            center[1] - ll[1]
        ]

        fit_2 = minimize(
            fit_2d, theta2,
            args=(
                im_cutout.flatten(),
                x2d.flatten(), y2d.flatten(),
                conf_cutout.flatten()
            ),
            method='Powell'
        )

        # check for lower residuals

        res = fit_2

        if fit_1['fun'] < fit_2['fun']:
            res = fit_1

    else:

        theta = [
            this_bead['fit_pars'][0] * Settings.zoom_level,
            this_bead['fit_pars'][1] * Settings.zoom_level,
            this_bead['fit_pars'][2] * Settings.zoom_level,
            this_bead['fit_pars'][3],
            this_bead['fit_pars'][-3],
            center[0] - ll[0],
            center[1] - ll[1]
        ]

        res = minimize(
            fit_2d, theta,
            args=(
                im_cutout.flatten(),
                x2d.flatten(), y2d.flatten(),
                conf_cutout.flatten()
            ),
            method='Powell'
        )

    for _i in range(3):

        c = res['x']
        res = minimize(
            fit_2d, c,
            args=(
                im_cutout.flatten(),
                x2d.flatten(), y2d.flatten(),
                conf_cutout.flatten()
            ),
            method='Powell'
        )

    fit_res['id'] = this_bead['id']
    fit_res['x'] = ll[0] + res['x'][-2]
    fit_res['y'] = ll[1] + res['x'][-1]
    fit_res['r'] = res['x'][2]
    fit_res['fit_pars'] = res['x']
    fit_res['success'] = res['success']

    return fit_res


def _check_bead(this_bead, done_x, done_y, full_shape):

    if this_bead["success"] is False:
        return False

    if (this_bead["x"] > full_shape[0] + 50) or (this_bead["y"] > full_shape[1] + 50):
        return False

    if this_bead["rad"] > Settings.feature_size[1]:
        return False

    r = np.sqrt(
        (this_bead["x"] - np.array(done_x)) ** 2
        + (this_bead["y"] - np.array(done_y)) ** 2
    )
    if r.min() < 0.5 * Settings.feature_size[0]:
        return False


@delayed(nout=2)
def image_stats(im, conf):
    mask = conf > 0
    pedestal = np.median(im[mask])
    im_std = 1.48 * mad(im[mask], axis=None)
    logger.debug("Img bgd: {0:.1f}, std: {1:.1f}".format(pedestal, im_std))
    return pedestal, im_std


def find_beads(mos_zarr: Path):  # noqa: C901
    """
    Finds all the beads in all the slices (physical and optical) in the
    zarr, and fits the bead profile.

    Attaches all the bead info as attrs to the zarr
    """
    # this is to store the beads later
    zarr_store = zarr.open(f"{mos_zarr}", mode="a")

    # load nn
    model = load_model(Settings.nn_model_file)
    model_window = 128
    window_offset = 32

    # conversion from dict headers to more informative
    # metadata names
    bead_par_to_attr_name = {
        "id": "bead_id",
        "success": "bead_conv",
        "fit": "bead_fit_pars",
        "r": "bead_rad",
        "x": "bead_x",
        "y": "bead_y",
        "z": "bead_z"
    }

    mos_full = xr.open_zarr(f"{mos_zarr}", group="")
    mos_zoom = xr.open_zarr(f"{mos_zarr}", group=f"l.{Settings.zoom_level}")
    bscale, bzero = mos_full.attrs['bscale'], mos_full.attrs['bzero']

    full_shape = (mos_full.dims["y"], mos_full.dims["x"])

    for this_slice in list(mos_zoom):

        first_bead = True
        bead_cat = {}

        for this_optical in list(mos_full.z.values):

            logger.info(
                "Analysing beads in " + this_slice +
                " Z{0:03d}".format(this_optical)
            )

            im = (
                mos_zoom[this_slice]
                .sel(z=this_optical, type="mosaic")
                .mean(dim="channel")
            )

            conf = mos_zoom[this_slice].sel(
                z=this_optical, channel=Settings.channel_to_use,
                type="conf"
            )

            im_med, im_std = image_stats(im.data, conf.data).compute()
            im_shape = im.shape

            stage = np.clip((im - im_med) / im_std, -5, 5) / 5.

            mask = np.zeros(im_shape)

            i = 0
            while i * (model_window - window_offset) < im_shape[0]:

                d0_corner = np.clip(
                    i * (model_window - window_offset),
                    0,
                    im_shape[0] - model_window
                )

                j = 0
                while j * (model_window - window_offset) < im_shape[1]:

                    d1_corner = np.clip(
                        j * (model_window - window_offset),
                        0,
                        im_shape[1] - model_window
                    )

                    cut = (stage[
                        d0_corner:d0_corner + model_window,
                        d1_corner:d1_corner + model_window
                    ])

                    mask[
                        d0_corner:d0_corner + model_window,
                        d1_corner:d1_corner + model_window
                    ] += np.round(model.predict(
                        np.expand_dims(np.expand_dims(cut, axis=0), axis=3)
                    )[0, :, :, 0])

                    j += 1
                i += 1

            im_temp = (mask > 0.1).astype(float)
            distance = ndi.distance_transform_edt(im_temp)
            coords = peak_local_max(
                distance,
                footprint=np.ones((3, 3)),
                labels=im_temp
            )

            _m = np.zeros(distance.shape, dtype=bool)
            _m[tuple(coords.T)] = True

            marks = ndi.label(_m)[0]
            labels = watershed(-distance, marks, mask=im_temp)

            da_labels = da.from_array(labels).persist()

            logger.debug(
                "Found {0:d} preliminary beads".format(
                    int(labels.max())
                )
            )

            logger.debug("Fitting preliminary detections")

            temp = []
            for i in range(1, int(labels.max())):
                _d = {}
                _d['id'] = i
                temp.append(
                    fit_bead(
                        im,
                        da_labels,
                        _d,
                        bscale,
                        bzero,
                        True,
                        False
                    )
                )
            beads_1st_iter = dask.compute(temp)[0]
            logger.debug("First pass completed")

            full_im = (
                mos_full[this_slice]
                .sel(z=this_optical, type="mosaic")
                .mean(dim="channel")
                .data
            )

            # conf and error are the same across channels
            full_conf = (
                mos_full[this_slice]
                .sel(z=this_optical, channel=Settings.channel_to_use, type="conf")
                .data
            )

            full_err = (
                mos_full[this_slice]
                .sel(z=this_optical, channel=Settings.channel_to_use, type="err")
                .data
            )
            logger.debug(
                """"
                    labels: {0:d},{1:d}
                    im: {2:d},{3:d}
                    conf: {4:d},{5:d}
                    err: {6:d},{7:d}
                """.format(
                    *labels.shape,
                    *full_im.shape,
                    *full_conf.shape,
                    *full_err.shape,
                )
            )
            temp = []
            logger.debug("Fitting all beads...")
            for i in range(len(beads_1st_iter)):
                if beads_1st_iter[i]['success'] is False:
                    continue

                temp.append(
                    fit_bead(
                        full_im,
                        da_labels,
                        beads_1st_iter[i],
                        bscale,
                        bzero,
                        False,
                        False
                    )
                )
            all_beads = dask.compute(temp)[0]

            # now we store all good beads in a dictionary, removing
            # duplicates and bad fits
            #
            # We'll store beads in the attrs for the physical slice
            # so we need to add the optical slice
            #
            done_x = [0]
            done_y = [0]

            for this_bead, bead_err in all_beads:
                this_bead["err"] = bead_err

                if _check_bead(this_bead, done_x, done_y, full_shape) is False:
                    continue

                done_x.append(this_bead["x"])
                done_y.append(this_bead["y"])

                if first_bead:
                    for this_key in this_bead.keys():
                        bead_cat[this_key] = [this_bead[this_key]]
                    bead_cat["z"] = [float(this_optical)]
                else:
                    for this_key in this_bead.keys():
                        bead_cat[this_key].append(this_bead[this_key])
                    bead_cat["z"].append(float(this_optical))

                first_bead = False

        # Store results as attrs in the full res slice
        for this_key in bead_cat.keys():
            zarr_store[this_slice].attrs[bead_par_to_attr_name[this_key]] = bead_cat[
                this_key
            ]
