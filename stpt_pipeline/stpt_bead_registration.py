import numpy as np
import xarray as xr

from settings import Settings
from bead_model import neg_log_like_conf

from scipy.ndimage import label, distance_transform_edt, zoom
from scipy.optimize import minimize
from scipy.stats import median_absolute_deviation as mad
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

from dask import delayed, compute
import dask.array as da
from pathlib import Path

import logging
log = logging.getLogger("owl.daemon.pipeline")


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
    for _i in range(100):
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


class bead_collection():

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

        self.critical_radius = 20.

    def add_bead(self, x_cor, y_cor, x_raw, y_raw, b_id):

        if len(self.x) == 0:
            r = np.array([self.critical_radius * 10.])
        else:
            r = np.sqrt((x_cor - self.x)**2 + (y_cor - self.y)**2)

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
                if beads_1st_iter[i][-1] is False:
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
                if this_bead['conv'] is False:
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


def _get_beads(slice_obj):

    xc = slice_obj['bead_xc'][:]
    yc = slice_obj['bead_xc'][:]
    err = slice_obj['bead_centre_err'][:]
    rad = slice_obj['bead_rad'][:]
    ind = slice_obj['bead_id'][:]

    return ind, xc, yc, rad, err


def _match_cats(xr, yr, er, xt, yt, et, errors=False):

    # simple closest match
    i_rt = []
    i_tr = []
    r_tr = []
    for i in range(len(xt)):
        r = np.sqrt((xr - xt[i])**2 + (yr - yt[i])**2)
        r_tr.append(r.min())
        i_tr.append(i)
        i_rt.append(r.argmin())
    r_tr = np.array(r_tr)
    # combined error
    e_c = np.sqrt(er[i_rt]**2 + et[i_tr]**2)

    # only 3sigma matches
    ii = np.where(r_tr - np.median(r_tr) < 3. * 1.48 * mad(r_tr))[0]

    # solid displacements in x and y, variance weight
    dx = ((xr[i_rt] - xt[i_tr])[ii] / e_c[ii]
          ** 2).sum() / (1. / e_c[ii]**2).sum()
    dy = ((yr[i_rt] - yt[i_tr])[ii] / e_c[ii]
          ** 2).sum() / (1. / e_c[ii]**2).sum()

    if errors:
        ex = np.sqrt(
            1. / np.sum(1. / e_c**2)
        )

        return dx, dy, ex, i_rt, i_tr

    return dx, dy, i_rt, i_tr


def register_slices(mos_zarr: Path):
    """
        Uses all the detected beads in each slice to cross-match
        and calculates an average displacement so that all
        slices are matched to the first one
    """

    mos_full = xr.open_zarr(mos_zarr)
    _slices = list(mos_full)

    # putting all slices on a single list
    optical_slices = []
    physical_slices = []
    for this_slice in _slices:
        physical_slices.append(this_slice)
        optical_slices.extend(list(mos_full[this_slice].z.values))

    # first pass crossmatch, taking into account all beads
    dx = [0.0]  # these store the slice to slice offset
    dy = [0.0]
    for i in range(1, len(physical_slices)):

        # We compare each slice (_t) with the previous one (_r)

        _, x_t, y_t, _, e_t = _get_beads(
            mos_full[physical_slices[i]].sel(z=optical_slices[i])
        )
        _, x_r, y_r, _, e_r = _get_beads(
            mos_full[physical_slices[i - 1]].sel(z=optical_slices[i - 1])
        )

        dxt, dyt, i_rt, i_tr = _match_cats(x_r, y_r, e_r, x_t, y_t, e_t)
        dx.append(dxt)
        dy.append(dyt)

        dr = np.sqrt((y_r[i_rt] - y_t[i_tr] - dyt)**2 +
                     (x_r[i_rt] - x_t[i_tr] - dxt)**2)
        dr0 = np.sqrt((y_r[i_rt] - y_t[i_tr])**2 + (x_r[i_rt] - x_t[i_tr])**2)
        log.info('1st pass slice shifts')
        log.info(
            physical_slices[i - 1] + '_Z{0:03d}:'.format(optical_slices[i - 1])
            + physical_slices[i] + '_Z{0:03d}:'.format(optical_slices[i])
            + ' {0:d} '.format(len(i_tr))
            + '{0:.3f} {1:.3f} '.format(dxt, dyt)
            + '{0:.3f} {1:.3f} '.format(np.median(dr), np.median(dr0))
        )

    # now that we know the slice to slice offset, we construct the catalogue
    # of all the beads
    bb = bead_collection()

    for i in range(len(physical_slices)):

        ref_slice = physical_slices[i] + '_Z{0:03d}:'.format(optical_slices[i])

        # because dx,dy are slice to slice, the total displacement
        # is the sum of all the previous
        dx_t = np.sum(dx[0:i + 1])
        dy_t = np.sum(dy[0:i + 1])

        ind_r, x_r, y_r, _, _ = _get_beads(
            mos_full[physical_slices[i]].sel(z=optical_slices[i])
        )
        id_str = []
        for this_id in ind_r:
            id_str.append(ref_slice + ':{0:05d}'.format(this_id))
        id_str = np.array(id_str)

        for j in range(len(x_r)):
            bb.add_bead(x_r[j] + dx_t, y_r[j] + dy_t, x_r[j], y_r[j], id_r[j])

        # at each slice we check matching beads and update avg coords
        bb.update_coords()

    # Now the we know wich objects can be seen in more than one slice,
    # we re-compute the offsets but only using the beads that appear
    # in at least all the optical slices and 2 physical

    min_num_dets = np.max(optical_slices) + 2

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
    for i in range(1, len(physical_slices)):

        this_slice = physical_slices[i] + '_Z{0:03d}:'.format(optical_slices[i])
        ref_slice = physical_slices[i - 1] + \
            '_Z{0:03d}:'.format(optical_slices[i - 1])

        ind_t, x_t, y_t, _, e_t = _get_beads(
            mos_full[physical_slices[i]].sel(z=optical_slices[i])
        )
        # we check which of these are repeated beads
        i_t = []
        j = 0
        for this_id in ind_t:
            temp = ref_slice + ':{0:05d}'.format(this_id)
            if temp in good_beads:
                i_t.append(j)
            j += 1

        ind_r, x_r, y_r, _, e_r = _get_beads(
            mos_full[physical_slices[i - 1]].sel(z=optical_slices[i - 1])
        )
        i_r = []
        j = 0
        for this_id in ind_r:
            temp = ref_slice + ':{0:05d}'.format(this_id)
            if temp in good_beads:
                i_r.append(j)
            j += 1

        dxt, dyt, edt, i_rt, i_tr = _match_cats(
            x_r[i_r], y_r[i_r], e_r[i_r],
            x_t[i_t], y_t[i_t], e_t[i_t],
            errors=True
        )
        dx2.append(dxt)
        dy2.append(dyt)
        dd2.append(edt)

        dr = np.sqrt((y_r[i_r][i_rt] - y_t[i_t][i_tr] - dyt) **
                     2 + (x_r[i_r][i_rt] - x_t[i_t][i_tr] - dxt)**2)
        dr0 = np.sqrt((y_r[i_r][i_rt] - y_t[i_t][i_tr]) **
                      2 + (x_r[i_r][i_rt] - x_t[i_t][i_tr])**2)
        log.info('2nd pass slice shifts')
        log.info(
            ref_slice + ':' + this_slice
            + ' {0:d} '.format(len(i_tr))
            + '{0:.3f} {1:.3f} '.format(dxt, dyt)
            + '{0:.3f} {1:.3f} '.format(np.median(dr), np.median(dr0))
        )

    # Now we store all the displacements as attrs
    for i in range(len(physical_slices)):

        ref_slice = physical_slices[i] + '_Z{0:03d}:'.format(optical_slices[i])

        # because dx,dy are slice to slice, the total displacement
        # is the sum of all the previous
        dx_t = np.sum(dx2[0:i + 1])
        dy_t = np.sum(dy2[0:i + 1])
        de_t = np.sqrt(np.sum(np.array(dd2[0:i + 1])**2))

        mos_full[physical_slices[i]].sel(z=optical_slices[i]).attrs[
            'slice_reg_dx'
        ] = dx_t
        mos_full[physical_slices[i]].sel(z=optical_slices[i]).attrs[
            'slice_reg_dy'
        ] = dy_t
        mos_full[physical_slices[i]].sel(z=optical_slices[i]).attrs[
            'slice_reg_err'
        ] = de_t
