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
        img_cube = [
            section[f'fov={i}/z=0/channel=04/geom']
            for i in range(section.attrs['fovs'])
        ]
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
                    im_ref, im_obj = im_obj, im_ref

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
