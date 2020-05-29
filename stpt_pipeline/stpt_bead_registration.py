import numpy as np
import xarray as xr
import pickle as pk

from settings import Settings
from bead_model import bead_profile_supergaussian as bead_profile
from bead_model import neg_log_like_conf

from scipy.ndimage import gaussian_filter, label, distance_transform_edt, zoom
from scipy.optimize import minimize
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from dask import delayed, compute
import dask.array as da
from pathlib import Path

import logging
log = logging.getLogger("owl.daemon.pipeline")


def mad(x):

    return np.median(np.abs(x - np.median(x)))


@delayed
def _get_coords(lab_im, lab_num, size_lim):
    cx, cy = np.where(lab_im == lab_num)

    obj_size = np.max(
        [np.max(cx) - np.min(cx), np.max(cy) - np.min(cy)]
    )

    good_bead = ((obj_size > size_lim[0]) & (obj_size < size_lim[1]))

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

    labels, n_objs = label(
        clean_im
        > Settings.sample_detection_threshold * im_std
    )
    sample_size = 0
    sample_label = 0
    for i in range(1, n_objs):
        t = np.sum(labels == i)  # num of labelled px
        if t > sample_size:
            sample_label = i
            sample_size = t

    # now we mask the sample
    mask = 1.0 - (labels == sample_label)
    labels, n_objs = label(
        clean_im * mask
        > Settings.bead_detection_threshold * im_std
    )

    # Watershed segmentation for beads that are touching
    im_temp = (labels > 0).astype(float)
    distance = distance_transform_edt(im_temp)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),
                                labels=im_temp)
    marks = label(local_maxi)[0]
    labels = watershed(-distance, marks, mask=im_temp)

    da_labels = da.from_array(labels)

    bead_limits = [Settings.feature_size[0] / Settings.zoom_level,
                   Settings.feature_size[1] / Settings.zoom_level]

    res = []
    for this_object in range(1, np.max(labels) + 1):
        res.append(
            _get_coords(da_labels, this_object, bead_limits)
        )

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

    feature_size = [
        np.max(xc) - np.min(xc),
        np.max(yc) - np.min(yc)
    ]

    n_conv = 0.0
    for i in range(100):
        #
        # I cut a small image around the candidate pixel,
        # calculate the centre of mass coordinates
        # and compare with the previous iteration of the
        # bead centre, until it converges below 1px diference
        #
        # these are the corners of the cutout image, taking care
        # that we stay inside the full stage image size
        ll_corner = [int(np.max([0, x_t - 0.5 * feature_size[0]])),
                     int(np.max([0, y_t - 0.5 * feature_size[1]]))]
        ur_corner = [int(np.min([im.shape[0], x_t + 0.5 * feature_size[0]])),
                     int(np.min([im.shape[1], y_t + 0.5 * feature_size[1]]))]

        im_t = im[ll_corner[0]:ur_corner[0],
                  ll_corner[1]:ur_corner[1]] - pedestal
        label_t = labels[ll_corner[0]:ur_corner[0], ll_corner[1]:ur_corner[1]]
        mask_t = (label_t == bead_num)

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
        r = np.sqrt((y_t - yc_i)**2 + (x_t - xc_i)**2)
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

        theta = [center_i[0], center_i[1],
                 (im_t * mask_t).max(), 0.25 * np.sum(feature_size), 1.0]

        res = fit_bead(
            im_t, 0.0, 1.0, label_t, theta, 0.0, im_std, bead_num, [0, 0],
            simple_result=True
        )

        # total radius when profile drops to 0.1*std

        rad = res['x'][3] * (-1.0 * np.log(0.1 * im_std /
                                           res['x'][2]))**(1. / (2 * res['x'][4]))
        peak = res['x'][2]
        width = res['x'][3]
        cen_x = res['x'][0] + ll_corner[0]
        cen_y = res['x'][1] + ll_corner[1]
        conv = res['success']
    else:
        cen_x = 0.0
        cen_y = 0.0
        rad = -99
        width = -99
        peak = -99
        conv = False

    return cen_x, cen_y, rad, peak, width, bead_num, conv


def fit_bead(
    im, err, conf, label, theta, pedestal,
    im_std, bead_label, corner, simple_result=False
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

    cen_x, cen_y, rad, peak, width = theta

    im_t = im - pedestal

    # brighter pixels dominate the fit, therefore contribute
    # more to the error
    if simple_result:
        err_t = 1.0
    else:
        temp = np.sqrt(err / conf.clip(1, 1e6))
        err_t = np.sum(temp * im_t**2) / np.sum(im_t**2)

    mask_t = (label == 0) + (label == bead_label)

    y_im_t = np.array([range(im_t.shape[1])
                       for t in range(im_t.shape[0])]).astype('float32')
    x_im_t = np.array([[t] * im_t.shape[1]
                       for t in range(im_t.shape[0])]).astype('float32')

    # We fit to a super-gaussian, that more or less
    # accomodates for the flat core of the beads
    res = minimize(
        neg_log_like_conf,
        [cen_x - corner[0], cen_y - corner[1], peak, width, 1.0],
        args=(x_im_t, y_im_t, im_t, mask_t, False, False), method='Powell'
    )
    # total radius when profile drops to 0.01*peak
    if simple_result:
        return res
    fit_res = {
        'bead_id': bead_label,
        'x': res['x'][0] + corner[0],
        'y': res['x'][1] + corner[1],
        'rad': res['x'][3] * (-1.0 * np.log(0.01 * res['x'][2] /
                                            res['x'][2]))**(1. / (2 * res['x'][4])),
        'fit': res['x'],
        'conv': res['success'],
        'err': err_t,
        'corner': corner
    }
    return fit_res


def find_beads(mos_zarr: Path):
    """
        Finds all the beads in all the slices (physical and optical) in the
        zarr, and fits the bead profile.

        Attaches all the bead info as attrs to the zarr
    """

    mos_full = xr.open_zarr(mos_zarr)
    mos_zoom = xr.open_zarr(
        mos_zarr, group='l.{0:d}'.format(Settings.zoom_level)
    )

    slices = list(mos_full)

    for this_slice in slices:

        optical_slices = list(mos_full[slices[0]].z.values)

        for this_optical in optical_slices:

            log.info(
                'Analysing beads in' + this_slice +
                ' Z{0:03d}'.format(this_optical)
            )

            full_im = mos_full[this_slice].sel(
                z=this_optical,
                channel=Settings.channel_to_use,
                type='mosaic'
            )
            full_conf = mos_full[this_slice].sel(
                z=this_optical,
                channel=Settings.channel_to_use,
                type='conf'
            )
            full_err = mos_full[this_slice].sel(
                z=this_optical,
                channel=Settings.channel_to_use,
                type='err'
            )

            im = mos_zoom[this_slice].sel(
                z=this_optical,
                channel=Settings.channel_to_use,
                type='mosaic'
            )

            conf = mos_zoom[this_slice].sel(
                z=this_optical,
                channel=Settings.channel_to_use,
                type='conf'
            )

            # Img stats
            ix, iy = np.where(conf.values > 0)
            pedestal = np.median(im.values[ix, iy])
            im_std = 1.48 * mad(im.values[ix, iy])
            log.info('  Img bgd: {0:.1f}, std: {1:.1f}'.format(
                pedestal, im_std))

            # Detection of all features with bead size
            labels, good_objects, good_cx, good_cy = det_features(
                im.values, pedestal, im_std)

            log.info('  Found {0:d} preliminary beads'.format(
                len(good_objects)))

            log.info('  Filtering preliminary detections')

            temp = []
            for i in range(len(good_objects)):
                temp.append(_fit_bead_1stpass(
                    im.values, labels,
                    good_cx[i], good_cy[i], pedestal, im_std, good_objects[i]))
            beads_1st_iter = compute(temp)[0]

            n = 0
            for t in beads_1st_iter:
                if t[-1]:
                    n += 1

            log.info('  Found {0:d} possible beads'.format(n))

            # resizing labels to full image size
            full_labels = da.from_array(
                zoom(labels, Settings.zoom_level, order=0).astype(int)
            )

            temp = []
            log.info('  Fitting all beads...')
            for i in range(len(beads_1st_iter)):
                if beads_1st_iter[i][-1] == False:
                    continue
                if beads_1st_iter[i][2] * Settings.zoom_level > Settings.feature_size[1]:
                    continue

                feature_size = Settings.zoom_level * 2.1 * beads_1st_iter[i][2]
                cen_x = beads_1st_iter[i][0] * Settings.zoom_level
                cen_y = beads_1st_iter[i][1] * Settings.zoom_level

                ll_corner = [int(np.max([0, cen_x - 0.5 * feature_size])),
                             int(np.max([0, cen_y - 0.5 * feature_size]))]
                ur_corner = [int(np.min([full_im.shape[0], cen_x + 0.5 * feature_size])),
                             int(np.min([full_im.shape[1], cen_y + 0.5 * feature_size]))]

                im_t = full_im[ll_corner[0]:ur_corner[0],
                               ll_corner[1]:ur_corner[1]].values
                err_t = full_err[ll_corner[0]:ur_corner[0],
                                 ll_corner[1]: ur_corner[1]].values
                conf_t = full_conf[ll_corner[0]:ur_corner[0],
                                   ll_corner[1]: ur_corner[1]].values

                mask_t = np.array(
                    full_labels[ll_corner[0]:ur_corner[0],
                                ll_corner[1]:ur_corner[1]]
                )

                theta = [
                    cen_x,
                    cen_y,
                    Settings.zoom_level * beads_1st_iter[i][2],
                    beads_1st_iter[i][3],
                    Settings.zoom_level * beads_1st_iter[i][4]
                ]

                temp.append(
                    delayed(fit_bead)(
                        im_t, err_t, conf_t, mask_t,
                        theta, pedestal, im_std, good_objects[i],
                        ll_corner
                    )
                )
            all_beads = compute(temp)[0]

            # now we store all good beads in a dictionary, removing
            # duplicates and bad fits

            first_bead = True
            bead_cat = {}
            done_x = [0]
            done_y = [0]
            for this_bead in all_beads:
                if this_bead['conv'] == False:
                    continue
                if this_bead['x'] < -50:
                    continue
                if this_bead['y'] < -50:
                    continue
                if this_bead['x'] > full_im.shape[0] + 50:
                    continue
                if this_bead['y'] > full_im.shape[1] + 50:
                    continue
                if this_bead['rad'] > Settings.feature_size[1]:
                    continue

                r = np.sqrt((this_bead['x'] - np.array(done_x))
                            ** 2 + (this_bead['y'] - np.array(done_y))**2)
                if r.min() < 0.5 * Settings.feature_size[0]:
                    continue
                else:
                    done_x.append(this_bead['x'])
                    done_y.append(this_bead['y'])

                for this_key in this_bead.keys():
                    if first_bead:
                        bead_cat[this_key] = [this_bead[this_key]]
                    else:
                        bead_cat[this_key].append(this_bead[this_key])
                first_bead = False

            # conversion from dict headers to more informative
            # metadata names
            bead_par_to_attr_name = {
                'bead_id': 'bead_id',
                'conv': 'bead_conv',
                'corner': 'bead_cutout_corner',
                'err': 'bead_centre_err',
                'fit': 'bead_fit_pars',
                'rad': 'bead_rad',
                'x': 'bead_xc',
                'y': 'bead_yc'
            }

            # Store results as attrs in the full res slice
            for this_key in bead_cat.keys():
                mos_full[this_slice].sel(z=this_optical).attrs[
                    bead_par_to_attr_name[this_key]
                ] = np.array(bead_cat[this_key])
