import logging

import numpy as np

from scipy.optimize import minimize

from scipy.ndimage import gaussian_filter,shift

from .settings import Settings

from .mutual_info import mutual_information

log = logging.getLogger('owl.daemon.pipeline')

def mad(x):
    """[summary]

    Parameters
    ----------
    x : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return np.median(np.abs(x - np.median(x)))


def prepare_image_conf(img, conf, orientation, img_std=-1):
    """[summary]

    # Generates properly aligned images, as the crossmatching code works
    # assuming a fixed relative orientation of the images.
    #
    # Returns the rotated image, a rotated filtered image to be
    # used in the crossmatching plus a confidence map. If apply_filter
    # is switched on, a median substracted image (either 1D or 2D median)
    # is returned as fimg_filt, otherwise a copy of the original image

    Parameters
    ----------
    img : [type]
        [description]
    conf : [type]
        [description]
    orientation : str
        [description]
    img_std : int, optional
        [description], by default -1

    Returns
    -------
    [type]
        [description]
    """
    N_SIGMA_CLIP = 1.5

    if 'x' in orientation:
        img_fin = np.rot90(img, 1)
        conf_fin = np.rot90(conf, 1)
    else:
        img_fin = img.copy()
        conf_fin = conf.copy()

    img_filt = img_fin.copy()

    mask_fin = np.ones_like(img_fin)

    # if STD=-1 the std for the confidence mask is computed over the image,
    # otherwise the valued passed as STD is used. By default the code uses
    # a STD computed over the whole cube, that is mucho more robust that
    # local estimations
    if img_std == -1:
        mask_fin[np.abs(img_filt) <= N_SIGMA_CLIP * 1.48 * mad(img_filt)] = 0.0
    else:
        mask_fin[img_fin <= img_std] = 0.0
    # input conf map
    mask_fin[conf_fin < 0.7] = 0.0
    #
    return img_fin, img_filt, mask_fin


def find_overlap_conf_old_version(  # noqa
    img_ref,
    conf_ref,
    img_obj,
    conf_obj,
    orientation,
    blind_start=True,
    init_desp=None,
    img_std=-1,
):
    """[summary]

    This calculates the displacementes that minimize the squared difference
    between images.

    Parameters
    ----------
    img_ref : [type]
        [description]
    conf_ref : [type]
        [description]
    img_obj : [type]
        [description]
    conf_obj : [type]
        [description]
    orientation : str
        [description]
    blind_start : bool, optional
        [description], by default True
    init_desp : list, optional
        [description], by default [-50, 1858]
    img_std : int, optional
        [description], by default -1

    Returns
    -------
    [type]
        [description]
    """
    assert orientation is not None, 'orientation cannot be None'

    REFINE_CHI = True
    STEPS = 30  # The chi surface will be of STEPSxSTEPS size

    # DELTA is the step that dx and dy are increased at each
    # point of the evaliations of chi.
    #
    # If an initial estimate is provided, the (dx,dy) space
    # of possible displacements to explore is narrower
    if blind_start:
        DELTA = [16, 8, 1]
    else:
        DELTA = [8, 2, 1]
        if init_desp is None:
            init_desp = [-50, 1858]
        dx = init_desp[0]
        dy = init_desp[1]

    # By default the code assumes that img0 is the reference image and that
    # img1 is aligned along the X direction (in the python matrix coordinate
    # frame). If this is not the case, ORIENTATION is set to Y and the images
    # rotated inside prepare_image
    img0, img0_filt, mask0 = prepare_image_conf(
        img_ref, conf_ref, orientation, img_std=img_std
    )
    img1, img1_filt, mask1 = prepare_image_conf(
        img_obj, conf_obj, orientation, img_std=img_std
    )
    DET_SIZE = img0.shape[1]
    for i_delta in range(len(DELTA)):
        # the first iteration sets up the displacements to be
        # evaluated
        if (i_delta == 0) & blind_start:
            # add 10 so that at least there are 10px to compare overlap
            #  desp_large runs along the direction where the displacement is
            # the largest, so typically will have values between 1700 and 2100,
            # while desp_short runs in the other, and moves between -200 and 200.
            desp_large = DET_SIZE - 10 - \
                np.arange(STEPS + 1) * DELTA[i_delta] - 1.0
            desp_short = (
                np.arange(STEPS + 1) * DELTA[i_delta]
                - (STEPS - 1) * DELTA[i_delta] * 0.5
            )
        else:
            # in case there is a prior estimation of the displacement
            # that minimizes chi, the vectors are centered at that point
            desp_short = (
                np.arange(STEPS) * DELTA[i_delta]
                - (STEPS - 1) * DELTA[i_delta] * 0.5
                + dx
            )
            # this just makes sure that we don't run out of image
            if (STEPS - 1) * DELTA[i_delta] * 0.5 + dy >= DET_SIZE:
                delta_cor = (STEPS - 1) * DELTA[i_delta] * 0.5 + dy - DET_SIZE
            else:
                delta_cor = 0

            desp_large = (
                np.arange(STEPS) * DELTA[i_delta]
                - (STEPS - 1) * DELTA[i_delta] * 0.5
                + dy
                - delta_cor
            )
        # because these will be used as indexes, need to be
        # cast as ints.
        desp_y = desp_large.astype(int)
        desp_x = desp_short.astype(int)

        chi = np.zeros((len(desp_x), len(desp_y)))
        npx = np.ones((len(desp_x), len(desp_y)))
        for i, dy in enumerate(desp_y):
            # a slice of the reference image
            t_ref = img0[:, dy:]
            err_ref = np.sqrt(img0[:, dy:])  #  assuming poisson errors
            mask_ref = mask0[:, dy:]
            # same for the other image
            t1 = img1[:, 0:-dy]
            maskt = mask1[:, 0:-dy]
            err_1 = np.sqrt(img1[:, 0:-dy])

            for j, dx in enumerate(desp_x):
                if dx < 0:
                    # this is the difference of the overlap of the reference image
                    # and the displaced image. Depending on whether the small displacemente
                    # is positive, negative or zero the indexes need to be generated,
                    # so there's on branch of the if
                    temp = (
                        t1[abs(dx):, ] * maskt[abs(dx):, :] -  # noqa
                        t_ref[0:dx, ] * mask_ref[0:dx, :]  # noqa
                    )
                    #  combined mask
                    mask_final = maskt[abs(dx):, :] + mask_ref[0:dx, :]  # noqa
                    # and erros added in cuadrature
                    div_t = np.sqrt(
                        err_ref[0:dx, :] ** 2 + err_1[abs(dx):, ] ** 2  # noqa
                    )
                elif dx == 0:
                    temp = t1 * maskt - t_ref * mask_ref
                    div_t = np.sqrt(err_ref ** 2 + err_1 ** 2)
                    mask_final = maskt + mask_ref
                else:
                    temp = (
                        t1[0:-dx, ] * maskt[0:-dx, ]  # noqa
                        - t_ref[dx:, ] * mask_ref[dx:, ]  # noqa
                    )  # noqa

                    mask_final = maskt[0:-dx, ] + mask_ref[dx:, ]  # noqa

                    div_t = np.sqrt(
                        err_ref[np.abs(dx):, ] ** 2 + err_1[0:-dx, ] ** 2  # noqa
                    )
                # if there are not enough good pixels we set chi to 10e7
                if mask_final.sum() > 0:
                    chi[j, i] = np.mean(
                        (temp[mask_final > 0] / div_t[mask_final > 0]) ** 2
                    )
                    npx[j, i] = np.sum(mask_final)
                else:
                    chi[j, i] = 1e7

        if REFINE_CHI:
            # this generates an estimation of the "background" of the chi surface,
            # that sometimes is heplful, mostly for cosmetic reasons, to have a flat
            # chi with a very clear minumum
            temp_bg0 = np.array(list(np.median(chi, 0)) * chi.shape[1]).reshape(
                chi.shape
            )
            chi -= temp_bg0
        #  now finfing the minimum
        i_x, i_y = np.where(chi == chi.min())
        try:
            # in case of many nans, this fails, default to middle
            dx = desp_x[i_x][0]
            dy = desp_y[i_y][0]
        except IndexError:
            dx = np.median(desp_x)
            dy = np.median(desp_y)

    if 'x' in orientation:
        dx, dy = dy, -dx

    if '-' in orientation:
        dx, dy = -dx, -dy

    return int(dx), int(dy)


def compare_imgs(
    desp,
    im1,
    co1,
    im2,
    co2,
    do_print
):
    
    im2_desp = shift(im2, desp, mode='constant', cval=0, order=1)
    co2_desp = shift(co2, desp, mode='constant', cval=0, order=1)

    co_all = co2_desp * co1

    err = np.sum(
        (co_all * (im2_desp - im1)**2) / (im1 + 0.001)
    ) / np.sum(co_all)

    if np.isnan(err):
        return np.inf

    if do_print:
        t_s = ''
        for t in desp:
            if np.abs(t) < 1e-3:
                t_s += ' {0:.1e}'.format(t)
            else:
                t_s += ' {0:.3f}'.format(t)
        print(t_s + ' {0:f}'.format(err))

    return err


def find_overlap_conf(  # noqa
    img_ref,
    conf_ref,
    img_obj,
    conf_obj,
    desp
):
    """[summary]

    This calculates the displacementes that minimize the squared difference
    between images.

    Parameters
    ----------
    img_ref : [type]
        [description]
    conf_ref : [type]
        [description]
    img_obj : [type]
        [description]
    conf_obj : [type]
        [description]
    desp : []
        [description]

    Returns
    -------
    [type]
        [description]
    """
    if desp[0] < 0:
        r_low_px_0 = 0
        r_high_px_0 = img_ref.shape[0] + int(desp[0]) - 1
        o_low_px_0 = abs(int(desp[0]))
        o_high_px_0 = img_ref.shape[0] - 1
    else:
        r_low_px_0 = int(desp[0])
        r_high_px_0 = img_ref.shape[0] - 1
        o_low_px_0 = 0
        o_high_px_0 = img_ref.shape[0] - int(desp[0]) - 1

    if desp[1] < 0:
        r_low_px_1 = 0
        r_high_px_1 = img_ref.shape[1] + int(desp[1]) - 1
        o_low_px_1 = abs(int(desp[1]))
        o_high_px_1 = img_ref.shape[1] - 1
    else:
        r_low_px_1 = int(desp[1])
        r_high_px_1 = img_ref.shape[1] - 1
        o_low_px_1 = 0
        o_high_px_1 = img_ref.shape[1] - int(desp[1]) - 1

    r_img_cut = gaussian_filter(
        img_ref[r_low_px_0:r_high_px_0, r_low_px_1:r_high_px_1], 3
    )
    r_con_cut = conf_ref[r_low_px_0:r_high_px_0, r_low_px_1:r_high_px_1]
    o_img_cut = gaussian_filter(
        img_obj[o_low_px_0:o_high_px_0, o_low_px_1:o_high_px_1], 3
    )
    o_con_cut = conf_obj[o_low_px_0:o_high_px_0, o_low_px_1:o_high_px_1]

    res = minimize(
        compare_imgs, [0., 0.],
        args=(r_img_cut, r_con_cut, o_img_cut, o_con_cut, False),
        method='Powell', options={'ftol': Settings.ftol_desp}
    )

    final_desp = [desp[0] + res['x'][0], desp[1] + res['x'][1]]

    # mutual information per good pixel

    mi = mutual_information(r_img_cut, o_img_cut, res['x'])

    return final_desp[0], final_desp[1], mi, r_con_cut.astype('float64').sum()
