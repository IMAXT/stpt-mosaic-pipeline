import numpy as np
from scipy.signal import medfilt2d


def MAD(x):
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


def read_mosaicifile_stpt(filename):
    """[summary]

    Parameters
    ----------
    filename : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    #
    # Reads the mosaic.txt file that sets up the stage and
    # returns the expected positions in microns
    #
    file_p = open(filename, 'r')
    #  delta_x/_y are the positions in microns,
    # label_x/_y are the names of each position,
    # that normally indicate the column/row position
    # but are not used later
    delta_x = []
    label_x = []
    delta_y = []
    label_y = []
    for this_line in file_p.readlines():
        if this_line.find('XPos') > -1:
            temp = this_line.split(':')
            delta_x.append(float(temp[1]))
            label_x.append(temp[0])
        if this_line.find('YPos') > -1:
            temp = this_line.split(':')
            delta_y.append(float(temp[1]))
            label_y.append(temp[0])
    #
    file_p.close()
    #
    return np.array(delta_x), np.array(delta_y), np.array(label_x), np.array(label_y)


def prepare_image_conf(
    img, conf, ORIENTATION='X', apply_filter=False, DOUBLE_MEDIAN=False, IMG_STD=-1
):
    """[summary]

    Parameters
    ----------
    img : [type]
        [description]
    conf : [type]
        [description]
    ORIENTATION : str, optional
        [description], by default 'X'
    apply_filter : bool, optional
        [description], by default False
    DOUBLE_MEDIAN : bool, optional
        [description], by default False
    IMG_STD : int, optional
        [description], by default -1

    Returns
    -------
    [type]
        [description]
    """
    #
    # Generates properly aligned images, as the crossmatching code works
    # assuming a fixed relative orientation of the images.
    #
    # Returns the rotated image, a rotated filtered image to be
    # used in the crossmatching plus a confidence map. If apply_filter
    # is switched on, a median substracted image (either 1D or 2D median)
    # is returned as fimg_filt, otherwise a copy of the original image
    #
    N_SIGMA_CLIP = 1.5
    #
    if ORIENTATION == 'X':
        img_fin = np.rot90(img, 1)
        conf_fin = np.rot90(conf, 1)
    else:
        img_fin = img.copy()
        conf_fin = conf.copy()
    #
    if apply_filter:
        if DOUBLE_MEDIAN:
            img_filt = img_fin - 0.5 * (
                medfilt2d(img_fin, (51, 1)) + medfilt2d(img_fin, (1, 51))
            )
        else:
            img_filt = img_fin - medfilt2d(img_fin, (51, 1))
    else:
        img_filt = img_fin.copy()
    #
    mask_fin = np.ones_like(img_fin)
    #
    # if STD=-1 the std for the confidence mask is computed over the image,
    # otherwise the valued passed as STD is used. By default the code uses
    # a STD computed over the whole cube, that is mucho more robust that
    # local estimations
    #
    if IMG_STD == -1:
        mask_fin[np.abs(img_filt) <= N_SIGMA_CLIP * 1.48 * MAD(img_filt)] = 0.0
    else:
        mask_fin[img_fin <= IMG_STD] = 0.0
    # input conf map
    mask_fin[conf_fin < 0.7] = 0.0
    #
    return img_fin, img_filt, mask_fin


def find_overlap_conf(
    img_ref,
    conf_ref,
    img_obj,
    conf_obj,
    ORIENTATION='X',
    produce_image=False,
    blind_start=True,
    init_desp=[-50, 1858],
    DOUBLE_MEDIAN=False,
    return_chi=False,
    IMG_STD=-1,
):
    """[summary]

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
    ORIENTATION : str, optional
        [description], by default 'X'
    produce_image : bool, optional
        [description], by default False
    blind_start : bool, optional
        [description], by default True
    init_desp : list, optional
        [description], by default [-50, 1858]
    DOUBLE_MEDIAN : bool, optional
        [description], by default False
    return_chi : bool, optional
        [description], by default False
    IMG_STD : int, optional
        [description], by default -1

    Returns
    -------
    [type]
        [description]
    """
    #
    # This calculates the displacementes that minimize the squared difference
    # between images.
    #
    REFINE_CHI = True
    #
    STEPS = 30  # The chi surface will be of STEPSxSTEPS size
    #
    # DELTA is the step that dx and dy are increased at each
    # point of the evaliations of chi.
    #
    # If an initial estimate is provided, the (dx,dy) space
    # of possible displacements to explore is narrower
    #
    if blind_start:
        DELTA = [16, 8, 1]
    else:
        DELTA = [8, 2, 1]
        dx = init_desp[0]
        dy = init_desp[1]
    #
    # By default the code assumes that img0 is the reference image and that
    # img1 is aligned along the X direction (in the python matrix coordinate
    # frame). If this is not the case, ORIENTATION is set to Y and the images
    # rotated inside prepare_image
    #
    img0, img0_filt, mask0 = prepare_image_conf(
        img_ref,
        conf_ref,
        apply_filter=False,
        ORIENTATION=ORIENTATION,
        DOUBLE_MEDIAN=DOUBLE_MEDIAN,
        IMG_STD=IMG_STD,
    )
    img1, img1_filt, mask1 = prepare_image_conf(
        img_obj,
        conf_obj,
        apply_filter=False,
        ORIENTATION=ORIENTATION,
        DOUBLE_MEDIAN=DOUBLE_MEDIAN,
        IMG_STD=IMG_STD,
    )
    DET_SIZE = img0.shape[1]
    #
    #
    for i_delta in range(len(DELTA)):
        print('Iteration {0:1d}'.format(i_delta + 1))
        # the first iteration sets up the displacements to be
        # evaluated
        if (i_delta == 0) & blind_start:
            #
            # add 10 so that at least there are 10px to compare overlap
            #  desp_large runs along the direction where the displacement is
            # the largest, so typically will have values between 1700 and 2100,
            # while desp_short runs in the other, and moves between -200 and 200.
            #
            desp_large = DET_SIZE - 10 - np.arange(STEPS + 1) * DELTA[i_delta] - 1.0
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
            #
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
        #
        chi = np.zeros((len(desp_x), len(desp_y)))
        npx = np.ones((len(desp_x), len(desp_y)))
        for i in range(len(desp_y)):
            # a slice of the reference image
            t_ref = img0[:, desp_y[i]:]
            err_ref = np.sqrt(img0[:, desp_y[i]:])  #  assuming poisson errors
            mask_ref = mask0[:, desp_y[i]:]
            # same for the other image
            t1 = img1[:, 0: -desp_y[i]]
            maskt = mask1[:, 0: -desp_y[i]]
            err_1 = np.sqrt(img1[:, 0: -desp_y[i]])
            #
            for j in range(len(desp_x)):
                if desp_x[j] < 0:
                    # this is the difference of the overlap of the reference image
                    # and the displaced image. Depending on whether the small displacemente
                    # is positive, negative or zero the indexes need to be generated,
                    # so there's on branch of the if
                    temp = (
                        t1[np.abs(desp_x[j]):, ] * maskt[np.abs(desp_x[j]):, :]
                        - t_ref[0:desp_x[j], ] * mask_ref[0:desp_x[j], :]
                    )
                    #  combined mask
                    mask_final = (
                        maskt[np.abs(desp_x[j]):, :] + mask_ref[0:desp_x[j], :]
                    )
                    # and erros added in cuadrature
                    div_t = np.sqrt(
                        err_ref[0:desp_x[j], :] ** 2
                        + err_1[np.abs(desp_x[j]):, ] ** 2
                    )
                    #
                elif desp_x[j] == 0:
                    #
                    temp = t1 * maskt - t_ref * mask_ref
                    #
                    div_t = np.sqrt(err_ref ** 2 + err_1 ** 2)
                    #
                    mask_final = maskt + mask_ref
                else:
                    #
                    temp = (
                        t1[0: -desp_x[j], ] * maskt[0: -desp_x[j], ]
                        - t_ref[desp_x[j]:, ] * mask_ref[desp_x[j]:, ]
                    )
                    #
                    mask_final = maskt[0: -desp_x[j], ] + mask_ref[desp_x[j]:, ]
                    #
                    div_t = np.sqrt(
                        err_ref[np.abs(desp_x[j]):, ] ** 2 + err_1[0: -desp_x[j], ] ** 2
                    )
                    #
                # if there are not enough good pixels we set chi to 10e7
                if mask_final.sum() > 0:
                    chi[j, i] = np.mean(
                        (temp[mask_final > 0] / div_t[mask_final > 0]) ** 2
                    )
                    #
                    npx[j, i] = np.sum(mask_final)
                else:
                    chi[j, i] = 1e7
        #
        if REFINE_CHI:
            # this generates an estimation of the "background" of the chi surface,
            # that sometimes is heplful, mostly for cosmetic reasons, to have a flat
            # chi with a very clear minumum
            temp_bg0 = np.array(list(np.median(chi, 0)) * chi.shape[1]).reshape(
                chi.shape
            )
            chi -= temp_bg0
        #
        #  now finfing the minimum
        i_x, i_y = np.where(chi == chi.min())
        #
        try:
            # in case of many nans, this fails, default to middle
            dx = desp_x[i_x][0]
            dy = desp_y[i_y][0]
        except IndexError:
            dx = int(np.median(desp_x))
            dy = int(np.median(desp_y))
        #
        print('Found min at dx={0:4d},dy={1:4d}'.format(dx, dy))
    #
    # now on to generate the relevant outputs
    #
    if produce_image:
        #
        # This produces a mosaiced image, good to check
        # if the procedure worked properly, but takes time.
        #
        big_picture, overlap = overlap_images(img0, img1, dx, dy)
        #
        if return_chi:
            return dx, dy, big_picture, overlap, chi
        else:
            return dx, dy, big_picture, overlap
    else:
        if return_chi:
            return dx, dy, chi
        else:
            return dx, dy


def overlap_images(img0, img1, dx, dy):
    """[summary]

    Parameters
    ----------
    img0 : [type]
        [description]
    img1 : [type]
        [description]
    dx : [type]
        [description]
    dy : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    #
    # Produces a mosaiced image displacing img1
    # by dx,dy and overlapping it to img0. The
    # displacements are cast to integers, so therefore
    # there is no interpolation.
    #
    # big_picture is the actual added pixels of both images,
    # and overlap keeps track of how many real pixels went into
    # each of the pixels of big_picture
    #
    dx = int(dx)
    dy = int(dy)
    big_picture = np.zeros((img0.shape[0] + np.abs(dx), img0.shape[1] + np.abs(dy)))
    if dx < 0:
        delta_x = np.abs(dx)
    else:
        delta_x = 0
    if dy < 0:
        delta_y = np.abs(dy)
    else:
        delta_y = 0
    overlap = np.zeros_like(big_picture)
    big_picture[
        delta_x:delta_x + img0.shape[0], delta_y:delta_y + img0.shape[1]
    ] = img0
    overlap[delta_x:delta_x + img0.shape[0], delta_y:delta_y + img0.shape[1]] += 1
    big_picture[
        delta_x + dx:delta_x + img0.shape[0] + dx,
        delta_y + dy:delta_y + dy + img0.shape[1],
    ] += img1
    overlap[
        delta_x + dx:delta_x + img0.shape[0] + dx,
        delta_y + dy:delta_y + dy + img0.shape[1],
    ] += 1
    #
    return big_picture, overlap
