import logging
from pathlib import Path

import dask.array as da
import numpy as np
from dask import delayed  # noqa: F401
from distributed import Client, as_completed

from .chi_functions import find_overlap_conf
from .mosaic_functions import find_delta
from .preprocess import apply_geometric_transform, preprocess

log = logging.getLogger('owl.daemon.pipeline')


@delayed
def _sink(*args):
    return args


def get_img_section(arr, offset, channel):
    img_cube = [
        arr[f'fov={i}/z={offset}/channel={channel:02d}/geom']
        for i in range(arr.attrs['fovs'])
    ]
    return img_cube


def main(*, root_dir: Path, flat_file: Path, output_dir: Path) -> None:
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
    client = Client.current()
    z = preprocess(root_dir, flat_file, output_dir)

    # convert to find_shifts
    for name, section in z.groups():
        results = []
        log.info('Processing section %s', name)
        img_cube = get_img_section(section, 0, 4)
        img_cube_stack = da.stack(img_cube)
        img_cube_std = da.std(img_cube_stack, axis=0)
        img_cube_mean_std = da.mean(img_cube_std).persist()

        dx, dy = np.array(section.attrs['XPos']), np.array(section.attrs['YPos'])
        delta_x, delta_y = find_delta(dx, dy)
        log.info('%s delta_x : %s , delta_y: %s', name, delta_x, delta_y)
        # dx0,dy0 are just the coordinates of each image
        # in columns/rows
        dx0 = np.round((dx - dx.min()) / delta_x).astype(int)
        dy0 = np.round((dy - dy.min()) / delta_y).astype(int)

        for i, img in enumerate(img_cube):
            r = np.sqrt((dx0 - dx0[i]) ** 2 + (dy0 - dy0[i]) ** 2)
            i_t = np.where(r == 1)[0].tolist()

            im_ref = da.from_zarr(img)
            # Calculate confidence map. Only needs to be done once.
            if i == 0:
                dist_conf = da.ones_like(im_ref)
                dist_conf = delayed(apply_geometric_transform)(dist_conf, None)

            for this_img in i_t:
                if i > this_img:
                    continue

                # flag is True if reference image is below or the right of the current image this_img
                flag = False
                if np.abs(dx0[i] - dx0[this_img]) > np.abs(dy0[i] - dy0[this_img]):
                    orientation = 'x'
                    if dx0[i] > dx0[this_img]:
                        flag = True
                else:
                    orientation = 'y'
                    if dy0[i] > dy0[this_img]:
                        flag = True

                log.info('Finding shifts for (%d, %d) %s', i, this_img, orientation)

                im_obj = da.from_zarr(img_cube[this_img])

                if flag:
                    res = delayed(find_overlap_conf)(
                        im_obj,
                        dist_conf,
                        im_ref,
                        dist_conf,
                        orientation,
                        img_std=img_cube_mean_std,
                    )
                    results.append(_sink(this_img, i, res))
                else:
                    res = delayed(find_overlap_conf)(
                        im_ref,
                        dist_conf,
                        im_obj,
                        dist_conf,
                        orientation,
                        img_std=img_cube_mean_std,
                    )
                    results.append(_sink(i, this_img, res))

        futures = client.compute(results)
        offsets = []
        for fut in as_completed(futures):
            i, j, res = fut.result()
            dx, dy = res
            offsets.append([i, j, dx, dy])
            log.info('%s offsets i: %d j: %d dx: %d dy: %d', name, i, j, dx, dy)

        section.attrs['offsets'] = offsets
        break

        ####################

    for name, section in z.groups():
        offsets = section.attrs['offsets']
        dx, dy = np.array(section.attrs['XPos']), np.array(section.attrs['YPos'])
        delta_x, delta_y = find_delta(dx, dy)
        dx0 = np.round((dx - dx.min()) / delta_x).astype(int)
        dy0 = np.round((dy - dy.min()) / delta_y).astype(int)

        dx_mu = np.zeros((len(dx), len(dy)))
        dy_mu = np.zeros((len(dx), len(dy)))
        dx_px = np.zeros((len(dx), len(dy)))
        dy_px = np.zeros((len(dx), len(dy)))

        for offset in offsets:
            x, y, ddx, ddy = offset
            dx_px[x, y] = ddx
            dx_px[y, x] = -ddx
            dy_px[x, y] = ddy
            dy_px[y, x] = -ddy
            dx_mu[x, y] = dx[x] - dx[y]
            dx_mu[y, x] = dx[y] - dx[x]
            dy_mu[x, y] = dy[x] - dy[y]
            dy_mu[y, x] = dy[y] - dy[x]

        # Mu to px scale
        #
        # we remove null theoretical displacements, bad measurements and
        # the random measured jitter (dx<1000) as these are not supposed to happen,
        # and they throw the ratio off
        ix, iy = np.where(
            (dx_mu != 0) & (np.abs(dx_px) < 5000) & (np.abs(dx_px) > 1000)
        )
        x_scale = np.median(dx_mu[ix, iy] / dx_px[ix, iy])
        ix, iy = np.where(
            (dy_mu != 0) & (np.abs(dy_px) < 5000) & (np.abs(dy_px) > 1000)
        )
        y_scale = np.median(dy_mu[ix, iy] / dy_px[ix, iy])
        err_px = np.sqrt(
            (dx_px - dx_mu / x_scale) ** 2 + (dy_px - dy_mu / y_scale) ** 2
        )

        dist_conf_done = False
        abs_pos_done = False
        # read image cube
        for optical_slice_to_use in range(4):
            for channel_to_use in range(4):
                print(
                    'Doing CH#{0:02d} OS#{1:03d}'.format(
                        int(channel_to_use), optical_slice_to_use
                    )
                )

                img_cube = get_img_section(
                    section, optical_slice_to_use, channel_to_use + 1
                )
                img_cube_stack = da.stack(img_cube)

            # Generate general confidence map based on distortions
            #
            if not (dist_conf_done):
                im_ref = da.from_zarr(img_cube[0])
                dist_conf = da.ones_like(im_ref)
                dist_conf = delayed(apply_geometric_transform)(
                    dist_conf, None
                ).compute()
                dist_conf_done = True
            #
            # ref image is the one with the max integrated flux
            ref_img_assemble = img_cube_stack.sum(axis=(1, 2)).argmax()
            ref_img_assemble = ref_img_assemble.compute()

            #
            ERROR_THRESHOLD = 20.0  # based on err_mat, all good ones are around 15
            #
            if not (abs_pos_done):
                abs_pos = np.zeros((len(dx0), 2))
                abs_pos[ref_img_assemble] = [0.0, 0.0]
                abs_err = np.zeros(len(dx0))
                abs_err[ref_img_assemble] = 0.0
                fixed_pos = [ref_img_assemble]
                # now for every fixed, see which ones overlap
                done = 1.0
                for i in fixed_pos:
                    # images at distance 1
                    r = (dx0 - dx0[i]) ** 2 + (dy0 - dy0[i]) ** 2
                    r[i] = 100
                    i_t = np.where(r <= 1)[0]
                    for this_img in i_t:
                        if this_img in fixed_pos:
                            # already been fixed
                            continue
                        #
                        # Now to calculate the final disp, we have to check
                        # if this has been measured more than one time
                        #
                        temp_pos = np.zeros(2)
                        temp_err = 0.0
                        n = 0.0
                        for ref_img in fixed_pos:
                            if (dx0[ref_img] - dx0[this_img]) ** 2 + (
                                dy0[ref_img] - dy0[this_img]
                            ) ** 2 <= 1.0:
                                print(
                                    'Matching {0:2d} to {1:2d}'.format(
                                        this_img, ref_img
                                    )
                                )
                                if err_px[ref_img, this_img] < ERROR_THRESHOLD:
                                    temp_pos[0] += (
                                        abs_pos[ref_img, 0] + dx_px[ref_img, this_img]
                                    )
                                    temp_pos[1] += (
                                        abs_pos[ref_img, 1] + dy_px[ref_img, this_img]
                                    )
                                    temp_err += 5.0 ** 2
                                else:
                                    temp_pos[0] += abs_pos[ref_img, 0] + dx_mu[
                                        ref_img, this_img
                                    ] / (x_scale)
                                    temp_pos[1] += abs_pos[ref_img, 1] + dy_mu[
                                        ref_img, this_img
                                    ] / (y_scale)
                                    temp_err += 15.0 ** 2
                                #
                                n += 1
                        abs_pos[this_img, 0] = temp_pos[0] / n
                        abs_pos[this_img, 1] = temp_pos[1] / n
                        abs_err[this_img] = np.sqrt(temp_err / n)
                        # this image has been fixed
                        fixed_pos.append(this_img)
                        done += 1
                    if done == len(dx0):
                        break
                abs_pos_done = True
            break
