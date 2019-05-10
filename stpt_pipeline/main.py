import logging
from os import listdir
from pathlib import Path

import numpy as np
from dask import delayed  # noqa: F401
from distributed import Client, wait
from scipy.ndimage import geometric_transform

from .chi_functions import find_overlap_conf, read_mosaicifile_stpt
from .mosaic_functions import find_delta, get_img_cube, get_img_list, get_mosaic_file
from .preprocess import preprocess
from .settings import Settings

log = logging.getLogger('owl.daemon.pipeline')


def main(*, root_dir: Path, flat_file: Path, output_dir: Path):
    """[summary]

    Parameters
    ----------
    root_dir : Path
        [description]
    flat_file : Path
        [description]
    output_dir : Path
        [description]
    """
    #  client = Client.current()
    z = preprocess(root_dir, flat_file, output_dir)
    sections = z.attrs['sections']
    for section in range(sections):
        log.info('Processing section %s', section)


"""
    for this_dir in dirs:
        log.info('Processing directory %s', this_dir)
        # read mosaic file, get delta in microns
        mosaic_file = get_mosaic_file(this_dir)
        dx, dy, lx, ly = read_mosaicifile_stpt(this_dir / mosaic_file)
        delta_x, delta_y = find_delta(dx, dy)
        log.debug('delta_x : %s , delta_y: %s', delta_x, delta_y)
        # dx0,dy0 are just the coordinates of each image
        # in columns/rows
        dx0 = np.round((dx - dx.min()) / delta_x).astype(int)
        dy0 = np.round((dy - dy.min()) / delta_y).astype(int)
        # get image names and select the first optical slice
        imgs, img_num, channel, optical_slice = get_img_list(
            this_dir, mosaic_size=len(dx)
        )
        optical_slice = optical_slice - optical_slice.min() + 1
        # read image cube
        img_cube = get_img_cube(
            this_dir,
            imgs,
            img_num,
            channel,
            optical_slice,
            channel_to_use=Settings.channel_to_use,
            slice_to_use=optical_slice.min(),
        ).persist()

        # std_img is the per pixel std of the cube,
        # shapes is the reference detector size
        img_cube_std = delayed(np.std)(img_cube, axis=0)
        std_img = delayed(magic_function)(img_cube_std, flat=nflat)
        std_img_mean = delayed(np.mean)(std_img)
        shapes = (Settings.x_max - Settings.x_min, Settings.y_max - Settings.y_min)

        log.info(
            'Found %d images, cube STD %.3f', img_cube.shape[0], std_img_mean.compute()
        )
        # Default confidence map, as the array is square but
        # the geometrical transform to correct distortion
        # leaves some pixels empty, so these are set to zero in
        # this conf. map. As all images go through the same initial
        # transformation, only one conf map is needed
        general_conf = np.ones_like(img_cube[0])
        dist_conf = delayed(geometric_transform)(
            general_conf,
            get_coords,
            output_shape=shapes,
            extra_arguments=(Settings.cof_dist, shapes[0] * 0.5, shapes[0]),
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
        results = []
        data_desc = []
        for i in range(len(dx0)):
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
                    log.debug('(%d, %d) already done', i, this_img)
                    continue
                log.info('Finding shifts for (%d, %d)', i, this_img)
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
                log.debug('Orientation %s', ORIENTATION)
                #
                # Transformed images
                #
                im_ref = delayed(magic_function)(img_cube[ind_ref], flat=nflat)
                new_ref = delayed(geometric_transform)(
                    im_ref,
                    get_coords,
                    output_shape=shapes,
                    extra_arguments=(Settings.cof_dist, shapes[0] * 0.5, shapes[0]),
                    extra_keywords={'direct': True},
                    mode='constant',
                    cval=0.0,
                    order=1,
                )

                im_obj = delayed(magic_function)(img_cube[ind_obj], flat=nflat)
                new_obj = delayed(geometric_transform)(
                    im_obj,
                    get_coords,
                    output_shape=shapes,
                    extra_arguments=(Settings.cof_dist, shapes[0] * 0.5, shapes[0]),
                    extra_keywords={'direct': True},
                    mode='constant',
                    cval=0.0,
                    order=1,
                )
                # finding shift
                log.debug('Finding shifts')
                res = delayed(find_overlap_conf)(
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
                results.append(res)
                data_desc.append([ORIENTATION, ind_ref, ind_obj])

            if len(results) > 5:
                futures = client.compute(results)
                wait(futures)
                for dd, fut in zip(data_desc, futures):
                    dx, dy, mer = fut.result()
                    ORIENTATION, ind_ref, ind_obj = dd
                    log.info('%s %s', dx, dy)

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

                results, data_desc = [], []

                out = output_dir / this_dir.name
                out.mkdir(exist_ok=True)
                log.debug('Saving results to %s', out)
                np.save(out / 'desp_dist_x', dx_mat)
                np.save(out / 'desp_dist_y', dy_mat)

"""
