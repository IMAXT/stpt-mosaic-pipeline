from os import listdir

import numpy as np
from scipy.ndimage import geometric_transform

from chi_functions import find_overlap_conf, read_mosaicifile_stpt
from mosaic_functions import (find_delta, get_img_cube, get_img_list,
                              get_mosaic_file)


# these two functions are only used to filter and defringe the flat
def med_box(y, half_box=2):
    """[summary]

    Parameters
    ----------
    y : [type]
        [description]
    half_box : int, optional
        [description], by default 2

    Returns
    -------
    [type]
        [description]
    """
    ym = []
    for i in range(len(y)):
        # too close to cero
        i_min = (i - half_box) if (i - half_box) >= 0 else 0
        # too close to end
        i_min = (
            i_min if i_min + 2 * half_box <= len(y) - 1 else len(y) - 1 - 2 * half_box
        )
        ym.append(np.median(y[i_min : i_min + 2 * half_box]))
    #
    return np.array(ym)


def defringe(img):
    """[summary]

    Parameters
    ----------
    img : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    fr_img = img.copy()
    for i in range(fr_img.shape[1]):
        if i < 5:
            t = np.median(img[:, 0:10], 1)
        elif i > fr_img.shape[1] - 5:
            t = np.median(img[:, -10:], 1)
        else:
            t = np.median(img[:, i - 5 : i + 5], 1)
        #
        fr_img[:, i] = img[:, i] - med_box(t, 5)
    #
    return fr_img


#
#  get_coords is the function that feeds geometric_transform
# in order to correct for the optical distortion of the detector.
#
def get_coords(coords, cof, center_x, max_x, direct=True):
    """[summary]

    Parameters
    ----------
    coords : [type]
        [description]
    cof : [type]
        [description]
    center_x : [type]
        [description]
    max_x : [type]
        [description]
    direct : bool, optional
        [description], by default True

    Returns
    -------
    [type]
        [description]
    """
    max_desp = cof[0] * coords[1] ** 2 + cof[1] * coords[1] + cof[2]
    dy_cof = max_desp / (max_x - center_x) ** 2
    if direct:
        sign = (coords[0] - center_x) / np.abs(coords[0] - center_x)
        if np.isnan(sign):
            sign = 1.0
        xi = np.abs(coords[0] - center_x)
        return (center_x + sign * (xi + dy_cof * xi ** 2), coords[1])
    else:
        xi = np.abs(coords[0] - center_x - cof[2])
        sign = (coords[0] - center_x - cof[2]) / np.abs(coords[0] - center_x - cof[2])
        if np.isnan(sign):
            sign = 1.0
        return (
            center_x + sign * (np.sqrt(1 + 4 * dy_cof * xi) - 1) / (2 * dy_cof),
            coords[1],
        )


#
##################################################################################
#
# x/y_min and max are the lower/upper boundaries of the useful section of the
# detector.
#
# Norm val is a normalization value for the stored values, ideally the gain
#
# do_flat activates flat correction prior to crossmatching. defringe does that to
# the flat before dividing by it, but it is time consuming and not very
# useful at the moment
#
# cof_dist are the coefficients of the optical distortion, as measured from
# the images themselves.
#
x_min = 2
x_max = 2080
y_min = 80
y_max = 1990
norm_val = 10000.0
do_flat = True
defring = False
channel_to_use = 4
cof_dist = np.array([3.93716645e-05, -7.37696218e-02, 2.52457306e01]) / 2.0
#
# this is the directory where all the subdirectories (one per physical slice)
# are stored. Ideally this should be passed as an argument
#
root_dir = '../20190201_tumour_opticalsections_extraoverlap/'


def magic_function(x):
    """[summary]

    This function transform the raw images into the ones used
    for crossmatching and mosaicing

    Parameters
    ----------
    x : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    return np.flipud(x / nflat)[x_min:x_max, y_min:y_max] / norm_val


# Read flat
#
if do_flat:
    fl = np.load('flat2.npy')
    if defringe:
        fr_img = defringe(fl[:, :, channel_to_use - 1])
        nflat = (fl[:, :, channel_to_use - 1] - fr_img) / np.median(
            fl[:, :, channel_to_use - 1]
        )
    else:
        nflat = (fl[:, :, channel_to_use - 1]) / np.median(fl[:, :, channel_to_use - 1])
    nflat[nflat < 0.5] = 1.0
else:
    nflat = 1.0
#
# This lists all the subdirectories. Once there is an
# standarized naming convention this will have to be edited,
# as at the moment looks for dir names with the '4t1' string
# on them, as all the experiments had this
#
dirs = []
for this_file in listdir(root_dir):
    if this_file.find('4t1') > -1:
        if this_file.find('.txt') > -1:
            continue
        dirs.append(root_dir + '/' + this_file + '/')
dirs.sort()
#
# Now we loop over all subirectories
#
for this_dir in dirs:
    #
    print(this_dir)
    # read mosaic file, get delta in microns
    mosaic_file = get_mosaic_file(this_dir)
    dx, dy, lx, ly = read_mosaicifile_stpt(this_dir + mosaic_file)
    delta_x, delta_y = find_delta(dx, dy)
    # dx0,dy0 are just the coordinates of each image
    # in columns/rows
    dx0 = np.round((dx - dx.min()) / delta_x).astype(int)
    dy0 = np.round((dy - dy.min()) / delta_y).astype(int)
    # get image names and select the first optical slice
    imgs, img_num, channel, optical_slice = get_img_list(this_dir, mosaic_size=len(dx))
    optical_slice = optical_slice - optical_slice.min() + 1
    # read image cube
    img_cube = get_img_cube(
        this_dir,
        imgs,
        img_num,
        channel,
        optical_slice,
        channel_to_use=channel_to_use,
        slice_to_use=optical_slice.min(),
    )
    #
    # std_img is the per pixel std of the cube,
    # shapes is the reference detector  size
    #
    std_img = magic_function(np.std(img_cube, 0))
    shapes = magic_function(img_cube[0, ...]).shape
    #
    print(
        'Found {0:d} images, cube STD {1:.3f}'.format(
            img_cube.shape[0], np.mean(std_img)
        )
    )
    # Default confidence map, as the array is square but
    # the geometrical transform to correct distortion
    # leaves some pixels empty, so these are set to zero in
    # this conf. map. As all images go through the same initial
    # transformation, only on conf map is needed
    general_conf = np.ones_like(magic_function(img_cube[0, ...]))
    dist_conf = geometric_transform(
        general_conf,
        get_coords,
        output_shape=(int(shapes[0]), shapes[1]),
        extra_arguments=(cof_dist, shapes[0] * 0.5, shapes[0]),
        extra_keywords={'direct': True},
        mode='constant',
        cval=0.0,
        order=1,
    )
    #
    # These are the matrixes where the displacements that come out
    # of cross-correlation are stored. dx_mat[i,j] is the optimal
    # displacement between image i and j in the x direction. If these
    # images do not overlap, it is set to -9999
    #
    dx_mat = np.zeros((len(dx), len(dx))) - 9999
    dy_mat = np.zeros((len(dx), len(dx))) - 9999
    #
    for i in range(len(dx0)):
        # pick up if crash by reloading already calculated
        # displacements
        try:
            dx_mat = np.load(this_dir + 'desp_dist_x.npy')
            dy_mat = np.load(this_dir + 'desp_dist_y.npy')
        except:
            print('No saved displacements found')
        #
        # This is more or less the distance between images in detectors,
        # so we only crossmatch images withn 1 detector size of the
        # reference image
        #
        r = (dx0 - dx0[i]) ** 2 + (dy0 - dy0[i]) ** 2
        r[i] = 100
        i_t = np.where(r <= 1)[0]
        #
        for this_img in i_t:
            if dx_mat[i, this_img] != -9999:
                # already done
                print('({0:2d},{1:2d}) already done'.format(i, this_img))
                continue
            print('Finding shifts for ({0:2d},{1:2d})'.format(i, this_img))
            #
            # Find relative orientation. As the crossmatching code always assumes
            # that the images are aligned along the X axis and the reference is to
            # the left, we need to juggle things if that's not the case
            #
            ind_ref = i
            ind_obj = this_img
            if np.abs(dx0[i] - dx0[this_img]) > np.abs(dy0[i] - dy0[this_img]):
                ORIENTATION = 'X'
                # Relative position
                if dx0[i] > dx0[this_img]:
                    ind_ref = this_img
                    ind_obj = i
                    continue
            else:
                ORIENTATION = 'Y'
                # Relative position
                if dy0[i] > dy0[this_img]:
                    ind_ref = this_img
                    ind_obj = i
                    continue
            #
            # Transformed images
            #
            new_ref = geometric_transform(
                magic_function(img_cube[ind_ref, ...]),
                get_coords,
                output_shape=(int(shapes[0]), shapes[1]),
                extra_arguments=(cof_dist, shapes[0] * 0.5, shapes[0]),
                extra_keywords={'direct': True},
                mode='constant',
                cval=0.0,
                order=1,
            )
            new_obj = geometric_transform(
                magic_function(img_cube[ind_obj, ...]),
                get_coords,
                output_shape=(int(shapes[0]), shapes[1]),
                extra_arguments=(cof_dist, shapes[0] * 0.5, shapes[0]),
                extra_keywords={'direct': True},
                mode='constant',
                cval=0.0,
                order=1,
            )
            # finding shift
            dx, dy, mer = find_overlap_conf(
                new_ref,
                dist_conf,
                new_obj,
                dist_conf,
                ORIENTATION=ORIENTATION,
                produce_image=False,
                return_chi=True,
                DOUBLE_MEDIAN=False,
                IMG_STD=np.mean(std_img),
            )
            # undoing the index shifts
            if ORIENTATION == 'Y':
                dx_mat[ind_ref, ind_obj] = dx
                dy_mat[ind_ref, ind_obj] = dy
                #
                dx_mat[ind_obj, ind_ref] = -dx
                dy_mat[ind_obj, ind_ref] = -dy
            if ORIENTATION == 'X':
                dx_mat[ind_ref, ind_obj] = dy
                dy_mat[ind_ref, ind_obj] = -dx
                #
                dx_mat[ind_obj, ind_ref] = -dy
                dy_mat[ind_obj, ind_ref] = dx
            # storing
            np.save(this_dir + 'desp_dist_x', dx_mat)
            np.save(this_dir + 'desp_dist_y', dy_mat)
