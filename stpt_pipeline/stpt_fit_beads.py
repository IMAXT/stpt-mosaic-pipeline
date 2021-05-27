from .settings import Settings
from .bead_model_sphere import fit_2d, get_bead_emission
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
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from keras.models import load_model
from distributed import Client, as_completed

import warnings
# this limits the volume of optimise/minimise overflow/sqrt(-1) warwnings
warnings.filterwarnings('ignore', 'invalid value encountered')
warnings.filterwarnings('ignore', 'overflow encountered')


def _check_size(rad):
    """
        Checks radius against feature size
        set in Settngs
    """

    if rad < 0.5 * Settings.feature_size[0]:
        return False
    if rad > 0.5 * Settings.feature_size[1]:
        return False

    return True


def _rezoom(lab, final_shape):
    """
        Upsamples the zoomed labels image to full res
    """

    zoom_fac = (
        float(final_shape[0] / lab.shape[0]),
        float(final_shape[1] / lab.shape[1])
    )

    return np.round(ndi.zoom(lab, zoom_fac, order=0)).astype(int)


def _get_center_rad(label_im, bead_dict):
    """
        Returns the initial guess for the center of a bead and
        its width based on a labelled bead mask.
    """

    _x = da.arange(label_im.shape[0])
    _y = da.arange(label_im.shape[1])

    w_x = da.sum(label_im == bead_dict['id'], axis=1)
    w_y = da.sum(label_im == bead_dict['id'], axis=0)
    x = np.sum(_x * w_x) / np.sum(w_x)
    y = np.sum(_y * w_y) / np.sum(w_y)

    d_x = np.max(_x[w_x > 0]) - np.min(_x[w_x > 0])
    d_y = np.max(_y[w_y > 0]) - np.min(_y[w_y > 0])

    return x, y, 0.5 * (d_x + d_y)


def _get_cutout_zoomed(bead_im, bead_labels, bead_dict):
    """
        Performs the cutting of a window around a bead,
        based on a labelled bead mask. This is intended for
        the downsampled image and beads that have been not
        fit before, so there's no estimation for size or
        coordinates.

        It returns a quality flag to discard false detections,
        the cutouts of the image and the labels, the coordinates
        of the lower left corner of the cutout and the input
        bead information. This is useful to pipe this output
        into other routines.
    """

    padding = 2  # px to add to cutouts

    x, y, w = _get_center_rad(bead_labels, bead_dict)

    half_width = int(0.5 * w)

    if _check_size(half_width * Settings.zoom_level) is False:
        return False, 0.0, 0.0, 0.0, 0.0

    ll_x = int(
        np.clip(
            x - half_width - padding,
            0,
            bead_im.shape[0] - 2 * (half_width + padding)
        )
    )
    ll_y = int(
        np.clip(
            y - half_width - padding,
            0, bead_im.shape[1] - 2 * (half_width + padding)
        )
    )

    im_cutout = bead_im[
        ll_x:ll_x + 2 * (half_width + padding),
        ll_y:ll_y + 2 * (half_width + padding)
    ]

    labels_zoom = bead_labels[
        ll_x:ll_x + 2 * (half_width + padding),
        ll_y:ll_y + 2 * (half_width + padding)
    ]

    return True, im_cutout, labels_zoom, ll_x, ll_y, bead_dict


def _get_cutout_full(bead_im, bead_labels, bead_dict):
    """
        Same as _get_cutout_zoomed, but for the full resolution
        image.
    """

    padding = 10  # px to add to cutouts

    r = bead_dict['r_mask'] * Settings.zoom_level
    center = (
        np.clip(
            int(bead_dict['x'] * Settings.zoom_level),
            0,
            bead_im.shape[0] - 1
        ),
        np.clip(
            int(bead_dict['y'] * Settings.zoom_level),
            0,
            bead_im.shape[1] - 1
        )
    )

    # cutout lower left corner
    half_width = int(r)

    if _check_size(half_width) is False:
        return False, 0.0, 0.0, 0.0, 0.0

    ll_x = int(
        np.clip(
            center[0] - half_width - padding,
            0,
            bead_im.shape[0] - 2 * (half_width + padding)
        )
    )
    ll_y = int(
        np.clip(
            center[1] - half_width - padding,
            0, bead_im.shape[1] - 2 * (half_width + padding)
        )
    )

    im_cutout = bead_im[
        ll_x:ll_x + 2 * (half_width + padding),
        ll_y:ll_y + 2 * (half_width + padding)
    ]

    # Because labels is done over a downsampled image,
    # we need to resample up

    small_x = int(np.round(ll_x / Settings.zoom_level))
    small_y = int(np.round(ll_y / Settings.zoom_level))

    small_width = int(
        np.round((2 * (half_width + padding)) / Settings.zoom_level)
    )

    labels_small = bead_labels[
        small_x:small_x + small_width,
        small_y:small_y + small_width
    ]

    labels_zoom = _rezoom(labels_small, im_cutout.shape)

    return True, im_cutout, labels_zoom, ll_x, ll_y, bead_dict


def _fit_bead_zoomed(cut_collection):
    """
    Fits a profile to a bead in a downsampled STPT mosaic image.

    Because this is the downsampled image, it is assumed that there
    is no info about the beads shape/size, so there are estimated
    from the labels, and the inital fit parameters are randomly chosen.

    Parameters
    ----------
    cut_collection:
        The ouput from get_cutout_zoomed
    """

    fit_res = {}

    fit_res['success'] = False

    if cut_collection[0] is False:
        return fit_res
    else:
        fit_mask = cut_collection[2]
        fit_im = cut_collection[1]
        ll_x = cut_collection[3]
        ll_y = cut_collection[4]
        this_bead = cut_collection[5]

    fit_conf = ((fit_mask == 0) + (fit_mask == this_bead['id'])).astype(float)

    x2d = np.array([
        [t * 1.0] * fit_conf.shape[1]
        for t in range(fit_conf.shape[0])
    ])
    y2d = np.array([
        np.arange(fit_conf.shape[1])
        for t in range(fit_conf.shape[0])
    ])

    pedestal = np.clip(fit_im.min().values, 0, 1e7)

    r = 0.5 * 0.5 * (fit_im.shape[0] + fit_im.shape[1])
    center = [
        0.5 * fit_im.shape[0],
        0.5 * fit_im.shape[1]
    ]
    central_val = fit_im[int(center[0]), int(center[1])].values

    n_tries = 5

    # because we are setting the parameters blindly,
    # we generate a small sample of random starting
    # points and see which combo gives the lowest
    # residuals

    theta_all = np.array([
        np.random.uniform(-1.0, 1.0, n_tries) * r,
        np.random.normal(1, 0.2, n_tries) * r,
        np.ones(n_tries) * (central_val - pedestal),
        np.random.uniform(0.0, 0.5, n_tries),
        np.ones(n_tries) * pedestal,
        np.ones(n_tries) * center[0],
        np.ones(n_tries) * center[1]
    ])

    res_fun = 1e17
    res = []

    for i in range(n_tries):

        theta = theta_all[:, i]

        fit_temp = minimize(
            fit_2d, theta,
            args=(
                fit_im.values,
                x2d, y2d,
                fit_conf,
                False
            ),
            method='Powell'
        )

        if fit_temp['fun'] < res_fun:
            res_fun = fit_temp['fun']
            res = fit_temp

    if _check_size(res['x'][1] * Settings.zoom_level) is False:
        fit_res['success'] = False
        r_mask = 0
    else:
        rs = np.arange(0, res['x'][1] * 1.5, 0.1)
        ps = get_bead_emission(*res['x'][0:-3], rs)
        ii = np.where(ps < 0.1)

        if len(ii) == 0:
            r_mask = 0.0
        else:
            r_mask = rs[np.min(ii)]

    fit_res['id'] = this_bead['id']
    fit_res['x'] = ll_x + res['x'][-2]
    fit_res['y'] = ll_y + res['x'][-1]
    fit_res['r'] = res['x'][1]
    fit_res['r_mask'] = r_mask
    fit_res['fit_pars'] = list(res['x'])
    fit_res['success'] = res['success']
    fit_res['fit'] = True

    # check for size
    if _check_size(r_mask * Settings.zoom_level) is False:
        fit_res['success'] = False

    return fit_res


def _fit_bead_full(cut_collection) -> dict:
    """
    Same as _fit_bead_zoomed, but for full resolution. The bead size/position
    and the first guess for the fit parameters are taken from the results
    of _fit_bead_zoomed and up-scaled.
    """

    fit_res = {}

    fit_res['success'] = False

    if cut_collection[0] is False:
        return fit_res
    else:
        fit_mask = cut_collection[2]
        fit_im = cut_collection[1]
        ll_x = cut_collection[3]
        ll_y = cut_collection[4]
        this_bead = cut_collection[5]

    fit_conf = ((fit_mask == 0) + (fit_mask == this_bead['id'])).astype(float)

    del fit_mask

    x2d = np.array([
        [t * 1.0] * fit_conf.shape[1]
        for t in range(fit_conf.shape[0])
    ])
    y2d = np.array([
        np.arange(fit_conf.shape[1])
        for t in range(fit_conf.shape[0])
    ])

    pedestal = np.clip(fit_im.min().values, 0, 1e7)

    n_tries = 3

    res = {'x': [
        this_bead['fit_pars'][0] * Settings.zoom_level,
        this_bead['fit_pars'][1] * Settings.zoom_level,
        this_bead['fit_pars'][2],
        this_bead['fit_pars'][3],
        pedestal,
        this_bead['x'] * Settings.zoom_level - ll_x,
        this_bead['y'] * Settings.zoom_level - ll_y
    ]
    }

    for _i in range(n_tries):

        c = res['x']

        res = minimize(
            fit_2d, c,
            args=(
                fit_im.values,
                x2d, y2d,
                fit_conf,
                False
            ),
            method='Powell'
        )
    del fit_im
    del fit_conf

    if _check_size(res['x'][1]) is False:
        fit_res['success'] = False
        r_mask = 0.0
    else:
        rs = np.arange(0, res['x'][1] * 1.5, 0.1)
        ps = get_bead_emission(*res['x'][0:-3], rs)
        ii = np.where(ps < 0.1)[0]

        if len(ii) == 0:
            r_mask = 0.0
        else:
            r_mask = rs[np.min(ii)]

    fit_res['id'] = this_bead['id']
    fit_res['x'] = ll_x + res['x'][-2]
    fit_res['y'] = ll_y + res['x'][-1]
    fit_res['r'] = res['x'][1]
    fit_res['r_mask'] = r_mask
    fit_res['fit_pars'] = list(res['x'])
    fit_res['success'] = res['success']
    fit_res['fit'] = True

    # check for size
    if _check_size(r_mask) is False:
        fit_res['success'] = False

    return fit_res


@ delayed(nout=2)
def _image_stats(im, conf):
    """
        Calculates the image statistics taking into account
        the confidence map
    """
    mask = conf > 0
    pedestal = np.median(im[mask])
    # im_std = 1.48 * mad(im[mask], axis=None)
    # std works better for NN
    im_std = im[mask].std()
    logger.debug("Img bgd: {0:.1f}, std: {1:.1f}".format(pedestal, im_std))
    return pedestal, im_std


def find_beads(mos_zarr: Path, sections: list):  # noqa: C901
    """
    Finds all the beads in all the slices (physical and optical) in the
    zarr, and fits the bead profile.

    Attaches all the bead info as attrs to the zarr
    """

    client = Client.current()

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
        "fit_pars": "bead_fit_pars",
        "r": "bead_rad",
        "r_mask": "mask_rad",
        "x": "bead_x",
        "y": "bead_y",
        "z": "bead_z"
    }

    mos_full = xr.open_zarr(f"{mos_zarr}", group="")
    mos_zoom = xr.open_zarr(f"{mos_zarr}", group=f"l.{Settings.zoom_level}")
    bscale, bzero = mos_full.attrs['bscale'], mos_full.attrs['bzero']

    for this_slice in sections:

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
            ) * bscale + bzero

            conf = mos_zoom[this_slice].sel(
                z=this_optical, channel=Settings.channel_to_use,
                type="conf"
            )

            im_med, im_std = _image_stats(im.data, conf.data).compute()
            im_shape = im.shape

            # this is how the NN was trained
            stage = np.clip((im - im_med) / im_std, -5, 5) / 5.

            mask = np.zeros(im_shape)

            # building the resulting mask by chunks
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
                footprint=np.ones((5, 5)),
                labels=im_temp
            )

            _m = np.zeros(distance.shape, dtype=bool)
            _m[tuple(coords.T)] = True

            marks = ndi.label(_m)[0]
            labels = watershed(-distance, marks, mask=im_temp)

            # by setting this, we effectively incorporate
            # the confidence map into the label mask
            labels[conf == 0] = -99

            da_labels = da.from_array(labels)

            logger.debug(
                "Found {0:d} preliminary beads".format(
                    int(labels.max())
                )
            )

            logger.debug("Fitting preliminary detections")

            temp_beads = []
            temp = []

            for i in range(1, labels.max()):
                # for i in range(1, 20):

                this_bead = {
                    'id': i,
                    'fit': False
                }

                _cut = _get_cutout_zoomed(im, da_labels, this_bead)

                temp.append(dask.delayed(_fit_bead_zoomed)(_cut))

            futures = client.compute(temp)
            done_x = [-10]
            done_y = [-10]
            for fut in as_completed(futures):
                t = fut.result()
                if t['success']:
                    # sometimes fits converge to the
                    # same bead from two neighbouring labels
                    # with this we remove repetitions
                    r_done = np.sqrt(
                        (np.array(done_x) - t['x'])**2 +
                        (np.array(done_y) - t['y'])**2
                    )

                    if r_done.min() > 1:
                        temp_beads.append(t)
                        done_x.append(t['x'])
                        done_y.append(t['y'])

            logger.debug("First pass completed")
            logger.debug("Found {0:d} beads".format(len(temp_beads)))

            full_im = mos_full[this_slice].sel(
                z=this_optical,
                type="mosaic"
            ).mean(dim='channel') * bscale + bzero

            logger.debug("Fitting full res beads...")
            all_beads = []

            temp = []
            for i in range(len(temp_beads)):

                this_bead = temp_beads[i]

                _cut = _get_cutout_full(full_im, da_labels, this_bead)

                temp.append(delayed(_fit_bead_full)(_cut))

            futures = client.compute(temp)
            done_x = [-10]
            done_y = [-10]
            for fut in as_completed(futures):
                t = fut.result()
                if t['success']:
                    r_done = np.sqrt(
                        (np.array(done_x) - t['x'])**2 +
                        (np.array(done_y) - t['y'])**2
                    )

                    if r_done.min() > 1:
                        done_x.append(t['x'])
                        done_y.append(t['y'])
                        # adding optical slice
                        t['z'] = float(this_optical)

                        all_beads.append(t)

            # now we store all good beads in a dictionary
            #
            # We'll store beads in the attrs for the physical slice
            # so we need to add the optical slice
            #
            logger.debug("Found {0:d} beads".format(len(all_beads)))
            for this_bead in all_beads:

                for this_key in this_bead.keys():
                    # this is just for internal
                    # bookeeping while fitting
                    if this_key == 'fit':
                        continue

                    if first_bead:
                        bead_cat[this_key] = [this_bead[this_key]]
                    else:
                        bead_cat[this_key].append(this_bead[this_key])

                first_bead = False

        # Store results as attrs in the full res slice
        for this_key in bead_cat.keys():

            zarr_store[this_slice].attrs[
                bead_par_to_attr_name[this_key]
            ] = bead_cat[
                this_key
            ]
