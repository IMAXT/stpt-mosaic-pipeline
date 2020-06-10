from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
from dask import compute, delayed
from scipy.ndimage import distance_transform_edt, label, zoom
from scipy.optimize import minimize
from scipy.stats import median_absolute_deviation as mad
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

import zarr
from owl_dev.logging import logger

from .bead_model import neg_log_like_conf
from .settings import Settings


@delayed
def _get_coords(lab_im, lab_num, size_lim):
    cx, cy = np.where(lab_im == lab_num)

    obj_size = np.max([np.max(cx) - np.min(cx), np.max(cy) - np.min(cy)])

    good_bead = (obj_size > size_lim[0]) and (obj_size < size_lim[1])

    return cx, cy, lab_num, good_bead


def det_features(im, pedestal, im_std):
    """
        Searchs for continous pathces of pixels over the detection threshold.
        Then calculates the bounding box for each patch and excludes those
        that are too big or too small.

        Parameters
        ----------
        im: image with features to be detected
        pedestal: bacckgorund level to substract
        im_std: representative standard deviation of the image

        Returns
        -------
        labels: an array, the same size as im, with the labels for
            each detected object
        good_objects: the ids of the objects that passed the preliminary
            filter in size
        good_cx, good_cy: centre-of-mass coordinates of the objets
    """

    clean_im = (im - pedestal).clip(0, 1e6)

    # this first pass only detects the sample, which is the largest
    # object

    labels, n_objs = label(clean_im > Settings.sample_detection_threshold * im_std)
    sample_size = 0
    sample_label = 0
    for i in range(1, n_objs):
        t = np.sum(labels == i)  # num of labelled px
        if t > sample_size:
            sample_label = i
            sample_size = t

    # now we mask the sample
    mask = 1.0 - (labels == sample_label)
    labels, n_objs = label(clean_im * mask > Settings.bead_detection_threshold * im_std)

    # Watershed segmentation for beads that are touching
    im_temp = (labels > 0).astype(float)
    distance = distance_transform_edt(im_temp)
    local_maxi = peak_local_max(
        distance, indices=False, footprint=np.ones((3, 3)), labels=im_temp
    )
    marks = label(local_maxi)[0]
    labels = watershed(-distance, marks, mask=im_temp)

    da_labels = da.from_array(labels)

    bead_limits = [
        Settings.feature_size[0] / Settings.zoom_level,
        Settings.feature_size[1] / Settings.zoom_level,
    ]

    res = []
    for this_object in range(1, np.max(labels) + 1):
        res.append(_get_coords(da_labels, this_object, bead_limits))

    beads = compute(res)

    good_objects = []
    good_cx = []
    good_cy = []
    for this_bead in beads[0]:
        if this_bead[-1]:
            good_cx.append(this_bead[0])
            good_cy.append(this_bead[1])
            good_objects.append(this_bead[2])

    return labels, good_objects, good_cx, good_cy


def mass_center(im, mask):
    """
        Gets the center-of-mass of an image weigthing pxs with brightness.

        Parameters
        ----------
        im: image with features to be detected
        mask: a 1/0 mask of pixels to be left out

        Returns
        -------
        x_c,y_c: position of the COM in image coordinates (x along columns)
    """

    x = np.arange(im.shape[0])
    y = np.arange(im.shape[1])
    t = (im * mask).sum(1)
    x_c = (x * t).sum() / t.sum()
    t = (im * mask).sum(0)
    y_c = (y * t).sum() / t.sum()
    #
    return x_c, y_c


@delayed
def _fit_bead_1stpass(im, labels, xc, yc, pedestal, im_std, bead_num):
    """
        Does the 1st fit to the bead profile,
        using the downsampled image
    """

    # The first iteration of the bead centre is the candidate pixel
    x_t = np.mean(xc)
    y_t = np.mean(yc)

    feature_size = [np.max(xc) - np.min(xc), np.max(yc) - np.min(yc)]

    n_conv = 0.0
    for _i in range(100):
        # I cut a small image around the candidate pixel,
        # calculate the centre of mass coordinates
        # and compare with the previous iteration of the
        # bead centre, until it converges below 1px diference
        #
        # these are the corners of the cutout image, taking care
        # that we stay inside the full stage image size
        ll_corner = [
            int(np.max([0, x_t - 0.5 * feature_size[0]])),
            int(np.max([0, y_t - 0.5 * feature_size[1]])),
        ]
        ur_corner = [
            int(np.min([im.shape[0], x_t + 0.5 * feature_size[0]])),
            int(np.min([im.shape[1], y_t + 0.5 * feature_size[1]])),
        ]

        im_t = im[ll_corner[0] : ur_corner[0], ll_corner[1] : ur_corner[1]] - pedestal
        label_t = labels[ll_corner[0] : ur_corner[0], ll_corner[1] : ur_corner[1]]
        mask_t = label_t == bead_num

        if (im_t * mask_t).sum() == 0.0:
            # there are no pixels left here
            convergence = False
            break

        # center of mass coordinates
        center_i = mass_center(im_t, mask_t)

        # putting them into the full stage coordinate system
        xc_i = center_i[0] + ll_corner[0]
        yc_i = center_i[1] + ll_corner[1]

        # and comparing with the previous iteration
        r = np.sqrt((y_t - yc_i) ** 2 + (x_t - xc_i) ** 2)
        if r <= 0.3:
            n_conv += 1
        if n_conv > 10:
            convergence = True
            break
        else:
            x_t = xc_i
            y_t = yc_i
            convergence = False

    # if failed to converge after 100 iters, this is not
    # likely a bead, so not fit. Tipycally, a centre
    # is found in less than 10 iterations
    if convergence:

        theta = [
            center_i[0],
            center_i[1],
            (im_t * mask_t).max(),
            0.25 * np.sum(feature_size),
            1.0,
        ]

        res = fit_bead(
            im_t, label_t, theta, 0.0, im_std, bead_num, [0, 0], simple_result=True
        )

        # total radius when profile drops to 0.1*std

        rad = res["x"][3] * (-1.0 * np.log(0.1 * im_std / res["x"][2])) ** (
            1.0 / (2 * res["x"][4])
        )
        peak = res["x"][2]
        width = res["x"][3]
        exp = res["x"][4]
        cen_x = res["x"][0] + ll_corner[0]
        cen_y = res["x"][1] + ll_corner[1]
        conv = res["success"]
    else:
        cen_x = 0.0
        cen_y = 0.0
        rad = -99
        width = -99
        exp = 1.0
        peak = -99
        conv = False

    return cen_x, cen_y, rad, peak, width, exp, bead_num, conv


def get_cutout(im, ll_corner, width, full_shape):

    ur_corner = [
        int(np.min([full_shape[0], ll_corner[0] + width])),
        int(np.min([full_shape[1], ll_corner[1] + width])),
    ]

    # these are the full images that come as a zarr pointer
    im_t = im[ll_corner[0] : ur_corner[0], ll_corner[1] : ur_corner[1]]

    return im_t


def _fit_bead_2ndpass(
    full_im,
    full_conf,
    full_err,
    full_labels,
    estimated_pars,
    pedestal,
    im_std,
    bead_num,
    full_shape,
):
    """
        Does the final fit over the full res image
    """

    cen_x = estimated_pars[0] * Settings.zoom_level
    cen_y = estimated_pars[1] * Settings.zoom_level

    window_size = 2.1 * estimated_pars[2] * Settings.zoom_level

    ll_corner = [
        int(np.max([0, estimated_pars[0] * Settings.zoom_level - 0.5 * window_size])),
        int(np.max([0, estimated_pars[1] * Settings.zoom_level - 0.5 * window_size])),
    ]

    im_t = get_cutout(full_im, ll_corner, window_size, full_shape)

    # check for image size
    if (im_t.shape[0] < 5) or (im_t.shape[1] < 5):
        empty_res = {
            "bead_id": bead_num,
            "x": 0.0,
            "y": 0.0,
            "rad": Settings.feature_size[1] * 10.0,
            "fit": [0.0, 0.0, 0.0, 0.0, 0.0],
            "conv": False,
            "corner": ll_corner,
        }
        return empty_res, 500

    conf_t = get_cutout(full_conf, ll_corner, window_size, full_shape)
    err_t = get_cutout(full_err, ll_corner, window_size, full_shape)

    label_t = get_cutout(full_labels, ll_corner, window_size, full_shape).astype(int)

    # brighter pixels dominate the fit, therefore contribute
    # more to the error
    temp = da.sqrt(err_t / conf_t.clip(1, 1e6))
    err_bead = da.sum(temp * (im_t - pedestal) ** 2) / da.sum((im_t - pedestal) ** 2)

    theta = [
        cen_x,
        cen_y,
        estimated_pars[3],
        Settings.zoom_level * estimated_pars[4],
        estimated_pars[5],
    ]

    res = delayed(fit_bead)(im_t, label_t, theta, pedestal, im_std, bead_num, ll_corner)

    return res, err_bead


def fit_bead(
    im, label, theta, pedestal, im_std, bead_label, corner, simple_result=False
):
    """
        Does the final fit to the bead profile at full resolution.
        theta has to be a good estimation of the bead parameters,
        ideally from _fit_bead_1stpass

        Parameters
        ----------
        im: image with the flux pixel values
        err: positional error, can be an image of the same size
            as im
        conf: the confidence map associated with im
        label: object labeling of the pixels in im
        theta: 1st estimation of the bead profile to be fit
        pedestal, im_std: background level and std of the image
        bead_label: object label of the bead to be fit, in case
            there are other features in the image, these will be
            flagged to 0
        corner: absolute coordinates of the upper left corner of
            the image, so that the output center is in abs. coords.
        simple_result: if on, just outputs the fit theta, without
            calculating other derivated parameters

        Returns
        -------
        Either the fitted parameters if simple_results is on, or a dictionary
        with the bead centre in absolute coordinates, the total radius, the convergence
        from optimize, the fit parameters and the estimated centering error
    """

    cen_x, cen_y, peak, width, exp = theta

    im_t = im - pedestal

    mask_t = (label == 0) + (label == bead_label)

    y_im_t = np.array([range(im_t.shape[1]) for t in range(im_t.shape[0])]).astype(
        "float32"
    )
    x_im_t = np.array([[t] * im_t.shape[1] for t in range(im_t.shape[0])]).astype(
        "float32"
    )

    # We fit to a super-gaussian, that more or less
    # accomodates for the flat core of the beads
    res = minimize(
        neg_log_like_conf,
        [cen_x - corner[0], cen_y - corner[1], peak, width, exp],
        args=(
            x_im_t.flatten(),
            y_im_t.flatten(),
            im_t.flatten(),
            mask_t.flatten(),
            False,
            False,
        ),
        method="Powell",
    )
    # total radius when profile drops to 0.01*peak
    if simple_result:
        return res
    fit_res = {
        "bead_id": float(bead_label),
        "x": res["x"][0] + corner[0],
        "y": res["x"][1] + corner[1],
        "rad": res["x"][3]
        * (-1.0 * np.log(0.01 * res["x"][2] / res["x"][2]))
        ** (1.0 / (2 * res["x"][4])),
        "fit": res["x"].tolist(),
        "conv": res["success"],
        "corner": corner,
    }
    return fit_res


class bead_collection:

    """
        This holds all the beads from all the slices of a mosaic,
        and keeps track of which ones are the same bead at different
        Z and which are new beads. Average coordinates per bead can be
        updated.
    """

    def __init__(self):
        self.id = []
        self.x = np.array([])
        self.y = np.array([])
        self.dx = np.array([])
        self.dy = np.array([])
        self.n = np.array([])

        self.id_list = []
        self.x_list = []
        self.y_list = []
        self.x_list_raw = []
        self.y_list_raw = []

        self.critical_radius = 20.0

    def add_bead(self, x_cor, y_cor, x_raw, y_raw, b_id):

        if len(self.x) == 0:
            r = np.array([self.critical_radius * 10.0])
        else:
            r = np.sqrt((x_cor - self.x) ** 2 + (y_cor - self.y) ** 2)

        if r.min() < self.critical_radius:
            self.x_list[r.argmin()].append(x_cor)
            self.y_list[r.argmin()].append(y_cor)
            self.x_list_raw[r.argmin()].append(x_raw)
            self.y_list_raw[r.argmin()].append(y_raw)
            self.id_list[r.argmin()].append(b_id)
            self.n[r.argmin()] += 1
        else:
            self.id.append(b_id)
            self.x = np.append(self.x, x_cor)
            self.y = np.append(self.y, y_cor)
            self.dx = np.append(self.dx, 0.0)
            self.dy = np.append(self.dy, 0.0)
            self.n = np.append(self.n, 1.0)

            self.x_list.append([x_cor])
            self.y_list.append([y_cor])
            self.x_list_raw.append([x_raw])
            self.y_list_raw.append([y_raw])
            self.id_list.append([b_id])

    def update_coords(self):
        for i in range(len(self.id)):
            if self.n[i] > 1:
                self.x[i] = np.mean(np.array(self.x_list[i]))
                self.y[i] = np.mean(np.array(self.y_list[i]))
                self.dx[i] = np.std(np.array(self.x_list[i]))
                self.dy[i] = np.std(np.array(self.y_list[i]))


def _check_bead(this_bead, done_x, done_y, full_shape):

    if this_bead["conv"] is False:
        return False

    if (this_bead["x"] < -50) or (this_bead["y"] < -50):
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


def find_beads(mos_zarr: Path):  # noqa: C901
    """
        Finds all the beads in all the slices (physical and optical) in the
        zarr, and fits the bead profile.

        Attaches all the bead info as attrs to the zarr
    """

    # this is to store the beads later
    zarr_store = zarr.open(f"{mos_zarr}", mode="a")

    # conversion from dict headers to more informative
    # metadata names
    bead_par_to_attr_name = {
        "bead_id": "bead_id",
        "conv": "bead_conv",
        "corner": "bead_cutout_corner",
        "err": "bead_centre_err",
        "fit": "bead_fit_pars",
        "rad": "bead_rad",
        "x": "bead_x",
        "y": "bead_y",
        "z": "bead_z",
    }

    mos_full = xr.open_zarr(f"{mos_zarr}", group="")
    mos_zoom = xr.open_zarr(f"{mos_zarr}", group="l.{0:d}".format(Settings.zoom_level))

    slices = list(mos_full)

    full_shape = (
        mos_full[slices[0]]
        .sel(z=0, channel=Settings.channel_to_use, type="mosaic")
        .shape
    )

    for this_slice in slices:

        optical_slices = list(mos_full[slices[0]].z.values)

        first_bead = True
        bead_cat = {}

        for this_optical in optical_slices:

            logger.info(
                "Analysing beads in " + this_slice + " Z{0:03d}".format(this_optical)
            )

            im = mos_zoom[this_slice].sel(
                z=this_optical, channel=Settings.channel_to_use, type="mosaic"
            )

            conf = mos_zoom[this_slice].sel(
                z=this_optical, channel=Settings.channel_to_use, type="conf"
            )

            # Img stats
            ix, iy = np.where(conf.values > 0)
            pedestal = np.median(im.values[ix, iy])
            im_std = 1.48 * mad(im.values[ix, iy], axis=None)
            logger.debug("Img bgd: {0:.1f}, std: {1:.1f}".format(pedestal, im_std))

            # Detection of all features with bead size
            labels, good_objects, good_cx, good_cy = det_features(
                im.values, pedestal, im_std
            )

            logger.debug("Found {0:d} preliminary beads".format(len(good_objects)))

            logger.debug("Filtering preliminary detections")

            temp = []
            for i in range(len(good_objects)):
                temp.append(
                    _fit_bead_1stpass(
                        im.data,
                        labels,
                        good_cx[i],
                        good_cy[i],
                        pedestal,
                        im_std,
                        good_objects[i],
                    )
                )
            beads_1st_iter = compute(temp)[0]

            # resampling to full arr
            full_labels = delayed(zoom)(labels, Settings.zoom_level, order=0)
            full_labels = da.from_delayed(full_labels, shape=full_shape, dtype="int")

            full_im = (
                mos_full[this_slice]
                .sel(z=this_optical, channel=Settings.channel_to_use, type="mosaic")
                .data
            )

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

            temp = []
            logger.debug("Fitting all beads...")
            for i in range(len(beads_1st_iter)):
                if beads_1st_iter[i][-1] is False:
                    continue
                if (
                    beads_1st_iter[i][2] * Settings.zoom_level
                    > Settings.feature_size[1]
                ):
                    continue

                temp.append(
                    _fit_bead_2ndpass(
                        full_im,
                        full_conf,
                        full_err,
                        full_labels,
                        beads_1st_iter[i],
                        pedestal,
                        im_std,
                        good_objects[i],
                        full_shape,
                    )
                )
            all_beads = compute(temp)[0]

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


def _get_beads(slice_obj, z_val):

    z = np.array(slice_obj["bead_z"][:])
    i_t = np.where(z == z_val)

    xc = np.array(slice_obj["bead_x"][:])[i_t]
    yc = np.array(slice_obj["bead_y"][:])[i_t]
    err = np.array(slice_obj["bead_centre_err"][:])[i_t]
    rad = np.array(slice_obj["bead_rad"][:])[i_t]
    ind = np.array(slice_obj["bead_id"][:])[i_t]

    return ind, xc, yc, rad, err


def _match_cats(xr, yr, er, xt, yt, et, errors=False):

    if len(xr) > len(xt):
        xs = np.array(xt)
        ys = np.array(yt)
        es = np.array(et)
        xl = np.array(xr)
        yl = np.array(yr)
        el = np.array(er)
    else:
        xs = np.array(xr)
        ys = np.array(yr)
        es = np.array(er)
        xl = np.array(xt)
        yl = np.array(yt)
        el = np.array(et)

    # simple closest match iterative
    dx = 0.0
    dy = 0.0
    for _i in range(3):
        i_sl = []
        i_ls = []
        r_sl = []
        for i in range(len(xs)):
            r = np.sqrt((xl - xs[i] - dx) ** 2 + (yl - ys[i] - dy) ** 2)
            r_sl.append(r.min())
            i_sl.append(i)
            i_ls.append(r.argmin())
        r_sl = np.array(r_sl)
        # combined error, minimum error should be 2
        e_c = np.sqrt(el[i_ls] ** 2 + es[i_sl] ** 2).clip(2)

        # only 3sigma matches
        ii = np.where(r_sl - np.median(r_sl) < 3.0 * 1.48 * mad(r_sl, axis=None))[0]

        dx = np.sum((xl[i_ls] - xs[i_sl])[ii] / e_c[ii] ** 2) / np.sum(
            1.0 / e_c[ii] ** 2
        )
        dy = np.sum((yl[i_ls] - ys[i_sl])[ii] / e_c[ii] ** 2) / np.sum(
            1.0 / e_c[ii] ** 2
        )

    if errors:
        ex = np.sqrt(1.0 / np.sum(1.0 / e_c ** 2))
        if len(xr) > len(xt):
            return dx, dy, ex, i_ls, i_sl
        else:
            return -dx, -dy, ex, i_sl, i_ls

    if len(xr) > len(xt):
        return dx, dy, i_ls, i_sl
    else:
        return -dx, -dy, i_sl, i_ls


def _get_good_beads(mos_full, phys, opt, good_beads):
    _ind, _x, _y, _, _e = _get_beads(mos_full[phys].attrs, opt)

    this_slice = phys + "_Z{0:03d}".format(opt)

    # we check which of these are repeated beads
    ind_t = np.array([])
    x_t = np.array([])
    y_t = np.array([])
    e_t = np.array([])
    _j = 0
    for this_id in _ind:
        temp = this_slice + ":{0:05d}".format(int(this_id))
        if temp in good_beads:
            ind_t = np.append(ind_t, this_id)
            x_t = np.append(x_t, _x[_j])
            y_t = np.append(y_t, _y[_j])
            e_t = np.append(e_t, _e[_j])
        _j += 1

    return ind_t, x_t, y_t, e_t


def register_slices(mos_zarr: Path):  # noqa: C901
    """
        Uses all the detected beads in each slice to cross-match
        and calculates an average displacement so that all
        slices are matched to the first one
    """

    # this is to store the beads later
    zarr_store = zarr.open(f"{mos_zarr}", mode="a")

    mos_full = xr.open_zarr(f"{mos_zarr}", group="")
    _slices = list(mos_full)

    # putting all slices on a single list
    optical_slices = []
    physical_slices = []
    for this_slice in _slices:
        for this_optical in mos_full[this_slice].z.values:
            physical_slices.append(this_slice)
            optical_slices.append(this_optical)

    # first pass crossmatch, taking into account all beads
    dx = [0.0]  # these store the slice to slice offset
    dy = [0.0]
    logger.info("1st pass slice shifts")
    for i in range(1, len(physical_slices)):

        # We compare each slice (_t) with the previous one (_r)

        _, x_t, y_t, _, e_t = _get_beads(
            mos_full[physical_slices[i]].attrs, optical_slices[i]
        )
        _, x_r, y_r, _, e_r = _get_beads(
            mos_full[physical_slices[i - 1]].attrs, optical_slices[i - 1]
        )

        dxt, dyt, i_rt, i_tr = _match_cats(x_r, y_r, e_r, x_t, y_t, e_t)
        dx.append(dxt)
        dy.append(dyt)

        dr = np.sqrt(
            (y_r[i_rt] - y_t[i_tr] - dyt) ** 2 + (x_r[i_rt] - x_t[i_tr] - dxt) ** 2
        )
        dr0 = np.sqrt((y_r[i_rt] - y_t[i_tr]) ** 2 + (x_r[i_rt] - x_t[i_tr]) ** 2)

        logger.info(
            physical_slices[i - 1]
            + "_Z{0:03d}:".format(optical_slices[i - 1])
            + physical_slices[i]
            + "_Z{0:03d}:".format(optical_slices[i])
            + " {0:d} ".format(len(i_tr))
            + "{0:.1f} {1:.1f} ".format(dxt, dyt)
            + "{0:.1f} {1:.1f} ".format(np.median(dr), np.median(dr0))
        )

    # now that we know the slice to slice offset, we construct the catalogue
    # of all the beads
    bb = bead_collection()

    for i in range(len(physical_slices)):

        ref_slice = physical_slices[i] + "_Z{0:03d}".format(optical_slices[i])

        # because dx,dy are slice to slice, the total displacement
        # is the sum of all the previous
        dx_t = np.sum(dx[0 : i + 1])
        dy_t = np.sum(dy[0 : i + 1])

        ind_r, x_r, y_r, _, _ = _get_beads(
            mos_full[physical_slices[i]].attrs, optical_slices[i]
        )
        id_str = []
        for this_id in ind_r:
            id_str.append(ref_slice + ":{0:05d}".format(int(this_id)))
        id_str = np.array(id_str)

        for j in range(len(x_r)):
            bb.add_bead(x_r[j] + dx_t, y_r[j] + dy_t, x_r[j], y_r[j], id_str[j])

        # at each slice we check matching beads and update avg coords
        bb.update_coords()

    # Now the we know wich objects can be seen in more than one slice,
    # we re-compute the offsets but only using the beads that appear
    # in at least all the optical slices and 2 physical

    min_num_dets = np.max(np.array(optical_slices)) + 2

    good_beads = []
    for i in range(len(bb.x)):
        if bb.n[i] >= min_num_dets:
            good_beads.extend(bb.id_list[i])
    # cleaning out repeated ids
    good_beads = list(set(good_beads))

    # here we store the new displacements, and a measurement of error
    dx2 = [0.0]
    dy2 = [0.0]
    dd2 = [0.0]
    logger.info("2nd pass slice shifts")
    for i in range(1, len(physical_slices)):

        this_slice = physical_slices[i] + "_Z{0:03d}".format(optical_slices[i])
        ref_slice = physical_slices[i - 1] + "_Z{0:03d}".format(optical_slices[i - 1])

        # only beads that have high reps
        _, x_t, y_t, e_t = _get_good_beads(
            mos_full, physical_slices[i], optical_slices[i], good_beads
        )
        _, x_r, y_r, e_r = _get_good_beads(
            mos_full, physical_slices[i - 1], optical_slices[i - 1], good_beads
        )

        dxt, dyt, edt, i_rt, i_tr = _match_cats(
            x_r, y_r, e_r, x_t, y_t, e_t, errors=True
        )

        dx2.append(dxt)
        dy2.append(dyt)
        dd2.append(edt)

        dr = np.sqrt(
            (y_r[i_rt] - y_t[i_tr] - dyt) ** 2 + (x_r[i_rt] - x_t[i_tr] - dxt) ** 2
        )
        dr0 = np.sqrt((y_r[i_rt] - y_t[i_tr]) ** 2 + (x_r[i_rt] - x_t[i_tr]) ** 2)
        logger.info(
            ref_slice
            + ":"
            + this_slice
            + " {0:d} ".format(len(i_tr))
            + "{0:.1f} {1:.1f} ".format(dxt, dyt)
            + "{0:.1f} {1:.1f} ".format(np.median(dr), np.median(dr0))
        )

    # Now we store all the displacements as attrs
    cube_reg = {
        "abs_dx": [],
        "abs_dy": [],
        "abs_err": [],
        "rel_dx": [],
        "rel_dy": [],
        "rel_err": [],
        "slice": [],
        "opt_z": [],
    }
    for i in range(len(physical_slices)):

        cube_reg["slice"].append(physical_slices[i])
        cube_reg["opt_z"].append(float(optical_slices[i]))

        # because dx,dy are slice to slice, the total displacement
        # is the sum of all the previous
        dx_t = np.sum(dx2[0 : i + 1])
        dy_t = np.sum(dy2[0 : i + 1])
        de_t = np.sqrt(np.sum(np.array(dd2[0 : i + 1]) ** 2))

        cube_reg["abs_dx"].append(dx_t)
        cube_reg["abs_dy"].append(dy_t)
        cube_reg["abs_err"].append(de_t)

        cube_reg["rel_dx"].append(dx2[i])
        cube_reg["rel_dy"].append(dy2[i])
        cube_reg["rel_err"].append(dd2[i])

    zarr_store.attrs["cube_reg"] = cube_reg
