import logging
from typing import Tuple

import dask.array as da
import numpy as np
import zarr
from dask import delayed
from distributed import Client, as_completed

from .chi_functions import find_overlap_conf
from .preprocess import apply_geometric_transform
from .retry import retry
from .settings import Settings

log = logging.getLogger('owl.daemon.pipeline')

ERROR_THRESHOLD = 20.0


@delayed
def _sink(*args):
    return args


@retry(Exception)
def _get_image(group, path, imgtype, shape):
    g = group.require_group(path)
    try:
        arr = g.create_dataset(imgtype, shape=shape,
                               chunks=(250, 250), dtype='float32')
    except ValueError:
        arr = g[imgtype]
    log.info('Image %s/%s created', g.path, imgtype)
    return arr


@delayed
def _mosaic(im_t, conf, abs_pos, abs_err, out=None):
    shapes = im_t[0].shape
    x_size = int(np.max(abs_pos[:, 0]) - np.min(abs_pos[:, 0])) + shapes[0] + 1
    x_delta = np.min(abs_pos[:, 0])

    y_size = int(np.max(abs_pos[:, 1]) - np.min(abs_pos[:, 1])) + shapes[1] + 1
    y_delta = np.min(abs_pos[:, 1])

    store = im_t[0].store
    group = zarr.Group(store)

    big_picture = _get_image(group, out, 'raw', (x_size, y_size))
    overlap = _get_image(group, out, 'overlap', (x_size, y_size))
    pos_err = _get_image(group, out, 'pos_err', (x_size, y_size))

    for i in range(len(im_t)):
        x0 = int(abs_pos[i, 0] - x_delta)
        y0 = int(abs_pos[i, 1] - y_delta)
        xslice = slice(x0, x0 + im_t[i].shape[0])
        yslice = slice(y0, y0 + im_t[i].shape[1])

        big_picture[xslice, yslice] += im_t[i][:]
        overlap[xslice, yslice] += conf[:]
        pos_err[xslice, yslice] += abs_err[i]

    return out


class Section:
    """STPT section

    Parameters
    ----------
    section
        Zarr group containing the section data
    """

    def __init__(self, section: zarr.Group):
        self._section = section

    def __getitem__(self, attr):
        return self._section.attrs[attr]

    @property
    def path(self) -> str:
        """Path of the section
        """
        return self._section.path

    def save_image(self, arr, group, imgtype='raw'):
        g = self._section.require_group(group)
        try:
            g.create_dataset(imgtype, data=arr)
        except ValueError:
            g[imgtype] = arr
        log.info('Image %s/%s created', group, imgtype)

    @property
    def fovs(self) -> int:
        """Number of field of views.
        """
        return self._section.attrs['fovs']

    @property
    def channels(self) -> int:
        """Number of channels
        """
        fov = [*self._section.groups()][0][1]
        z = [*fov.groups()][0][1]
        return len([*z.groups()])

    @property
    def slices(self):
        """Number of optical sections
        """
        fov = [*self._section.groups()][0][1]
        return len([*fov.groups()])

    def get_img_section(self, offset: int, channel: int) -> zarr.Array:
        img_cube = [
            self._section[f'fov={i}/z={offset}/channel={channel:02d}/geom']
            for i in range(self.fovs)
        ]
        return img_cube

    def find_grid(self) -> Tuple[np.ndarray]:
        """Find the mosaic grid

        Returns
        -------
        dx, dy
            Coordinates of each image in columns/row number
        """
        # In theory each position in the mosaic stage is offset
        # by a fixed value, but there's some jitter in the microscope
        # so this finds this delta. It is used later to get the column
        # and row position of each single image
        dx = np.array(self['XPos'])
        dy = np.array(self['YPos'])

        r = np.sqrt((dx.astype(float) - dx[0])
                    ** 2 + (dy.astype(float) - dy[0]) ** 2)
        r[0] = r.max()  # avoiding minimum at itself
        # first candidate
        dx_1 = np.abs(dx[r.argmin()] - dx[0])
        dy_1 = np.abs(dy[r.argmin()] - dy[0])
        # second candidate
        r[r.argmin()] = r.max()
        dx_2 = np.abs(dx[r.argmin()] - dx[0])
        dy_2 = np.abs(dy[r.argmin()] - dy[0])

        delta_x, delta_y = np.max([dx_1, dx_2]), np.max([dy_1, dy_2])
        log.debug(
            'Section %s delta_x : %s , delta_y: %s',
            self._section.name,
            delta_x,
            delta_y,
        )

        dx0 = np.round((dx - dx.min()) / delta_x).astype(int)
        dy0 = np.round((dy - dy.min()) / delta_y).astype(int)
        return (dx0, dy0)

    def get_mos_pos(self) -> Tuple[np.ndarray]:
        """Find the mosaic grid

        Returns
        -------
        dx, dy
            Coordinates of each image in mosaic units
        """

        dx = np.array(self['XPos'])
        dy = np.array(self['YPos'])

        return (dx, dy)

    def find_orientation(self, dx1: int, dy1: int, dx2: int, dy2: int) -> str:
        """Find the relative orientation between grid positions of two images

        Parameters
        ----------
        dx1
            X grid position of first image
        dy1
            Y grid position of first image
        dx2
            X grid position of second image
        dy2
            Y grid position of second image

        Returns
        -------
        Relative position
        """
        if abs(dx1 - dx2) > abs(dy1 - dy2):
            orientation = 'x'
            if dx1 > dx2:
                orientation = f'-{orientation}'
        else:
            orientation = 'y'
            if dy1 > dy2:
                orientation = f'-{orientation}'
        return orientation

    def get_distconf(self):
        return self._section['/dist_conf'][:]

    def find_offsets(self):
        """Calculate offsets between all pairs of overlapping images
        """
        client = Client.current()
        # convert to find_shifts
        results = []
        log.info('Processing section %s', self._section.name)
        img_cube = self.get_img_section(0, 4)
        img_cube_stack = da.stack(img_cube)
        img_cube_std = da.std(img_cube_stack, axis=0)
        img_cube_mean_std = da.mean(img_cube_std).persist()

        # Calculate confidence map. Only needs to be done once per section
        dist_conf = self.get_distconf()

        dx0, dy0 = self.find_grid()
        dx_mos, dy_mos = self.get_mos_pos()

        for i, img in enumerate(img_cube):
            r = np.sqrt((dx0 - dx0[i]) ** 2 + (dy0 - dy0[i]) ** 2)

            # including diagonals

            i_t = np.where((r <= np.sqrt(2)) & (r > 0))[0].tolist()

            im_i = da.from_zarr(img)

            for this_img in i_t:
                if i > this_img:
                    continue

                desp = [(dx_mos[this_img] - dx_mos[i]) / Settings.mosaic_scale,
                        (dy_mos[this_img] - dy_mos[i]) / Settings.mosaic_scale]

                # trying to do always positive displacements

                if (desp[0] < -1000) | (desp[1] < -1000):
                    desp[0] *= -1
                    desp[1] *= -1
                    obj_img = i
                    im_obj = im_i
                    ref_img = this_img
                    im_ref = da.from_zarr(img_cube[this_img])
                else:
                    obj_img = this_img
                    im_obj = da.from_zarr(img_cube[this_img])
                    ref_img = i
                    im_ref = im_i

                log.info('Initial offsets i: %d j: %d dx: %d dy: %d',
                         ref_img,
                         obj_img,
                         desp[0],
                         desp[1],
                         )

                res = delayed(find_overlap_conf)(
                    im_ref,
                    dist_conf,
                    im_obj,
                    dist_conf,
                    desp,
                )
                results.append(_sink(ref_img, obj_img, res))

        futures = client.compute(results)
        offsets = []
        for fut in as_completed(futures):
            i, j, res = fut.result()
            dx, dy, mi, npx = res
            offsets.append([i, j, dx, dy, npx, mi])
            log.info(
                'Section %s offsets i: %d j: %d dx: %d dy: %d mi: %f avg_f: %f',
                self._section.name,
                i,
                j,
                dx,
                dy,
                mi,
                npx,
            )

        self._section.attrs['offsets'] = offsets
        del img_cube_mean_std

    def compute_scale(self):
        """Return the scale of the detector
        """
        offsets = self['offsets']
        dx, dy = self['XPos'], self['YPos']

        px = {f'{o[0]}:{o[1]}': o[2:] for o in offsets}
        mu = {
            f'{o[0]}:{o[1]}': [dx[o[1]] - dx[o[0]], dy[o[1]] - dy[o[0]]]
            for o in offsets
        }

        # Mu to px scale
        # we remove null theoretical displacements, bad measurements and
        # the random measured jitter (dx<1000) as these are not supposed to happen,
        # and they throw the ratio off
        x_scale = np.median(
            [
                m[0] / p[0]
                for m, p in zip(mu.values(), px.values())
                if (abs(p[0]) < 5000) & (abs(p[0] > 1000))
            ]
        )
        y_scale = np.median(
            [
                m[1] / p[1]
                for m, p in zip(mu.values(), px.values())
                if (abs(p[1]) < 5000) & (abs(p[1] > 1000))
            ]
        )

        for key in list(px):
            i, j = key.split(':')
            px[f'{j}:{i}'] = [-item for item in px[f'{i}:{j}']]

        for key in list(mu):
            i, j = key.split(':')
            mu[f'{j}:{i}'] = [-item for item in mu[f'{i}:{j}']]

        self._px, self._mu = px, mu

        log.debug('Scale %f %f', x_scale, y_scale)

        err_px = {}
        for key in mu:
            y_diff = px[key][0] - mu[key][0] / y_scale
            x_diff = px[key][1] - mu[key][1] / x_scale
            err_px[key] = np.sqrt(x_diff ** 2 + y_diff ** 2)

        return x_scale, y_scale, err_px

    def compute_abspos(self):
        """Compute absolute positions of images in the field of view.
        """
        log.info('Processing section %s', self._section.name)
        x_scale, y_scale, err_px = self.compute_scale()

        img_cube = self.get_img_section(0, 1)
        img_cube_stack = da.stack(img_cube)
        # ref image is the one with the max integrated flux
        ref_img_assemble = img_cube_stack.sum(axis=(1, 2)).argmax()
        ref_img_assemble = ref_img_assemble.compute()
        # ref_img_assemble = 0
        log.debug('Reference image %d', ref_img_assemble)
        dx0, dy0 = self.find_grid()

        abs_pos = np.zeros((len(dx0), 2))
        abs_pos[ref_img_assemble] = [0.0, 0.0]
        abs_err = np.zeros(len(dx0))
        abs_err[ref_img_assemble] = 0.0
        fixed_pos = [ref_img_assemble]
        # now for every fixed, see which ones overlap
        done = 1.0
        for i in fixed_pos:
            r = np.sqrt((dx0 - dx0[i]) ** 2 + (dy0 - dy0[i]) ** 2)
            i_t = np.where(r == 1)[0].tolist()

            for this_img in i_t:
                if this_img in fixed_pos:
                    continue

                # Now to calculate the final disp, we have to check
                # if this has been measured more than once
                temp_pos = [0, 0]
                temp_err = 0.0
                n = 0.0
                for ref_img in fixed_pos:
                    if (dx0[ref_img] - dx0[this_img]) ** 2 + (
                        dy0[ref_img] - dy0[this_img]
                    ) ** 2 == 1.0:
                        if err_px[f'{ref_img}:{this_img}'] < ERROR_THRESHOLD:
                            temp_pos[0] += (
                                abs_pos[ref_img, 0] +
                                self._px[f'{ref_img}:{this_img}'][0]
                            )
                            temp_pos[1] += (
                                abs_pos[ref_img, 1] +
                                self._px[f'{ref_img}:{this_img}'][1]
                            )
                            temp_err += 5.0 ** 2
                        else:
                            temp_pos[0] += (
                                abs_pos[ref_img, 0] +
                                self._mu[f'{ref_img}:{this_img}'][0] / y_scale
                            )
                            temp_pos[1] += (
                                abs_pos[ref_img, 1] +
                                self._mu[f'{ref_img}:{this_img}'][1] / x_scale
                            )
                            temp_err += 15.0 ** 2

                        n += 1
                abs_pos[this_img, 0] = temp_pos[0] / n
                abs_pos[this_img, 1] = temp_pos[1] / n
                abs_err[this_img] = np.sqrt(temp_err / n)
                # this image has been fixed
                fixed_pos.append(this_img)
                done += 1
            if done == len(dx0):
                break

        self._section.attrs['abs_pos'] = abs_pos.tolist()
        return abs_pos, abs_err

    def stitch(self):
        """Stitch and save all images
        """
        client = Client.current()

        abs_pos, abs_err = self.compute_abspos()

        conf = self.get_distconf()

        results = []
        for sl in range(self.slices):
            for ch in range(self.channels):
                im_t = self.get_img_section(sl, ch + 1)
                out = f'{self.path}/mosaic/z={sl}/channel={ch+1}'
                res = _mosaic(im_t, conf, abs_pos, abs_err, out=out)
                results.append(res)

        futures = client.compute(results)
        for fut in as_completed(futures):
            res = fut.result()
            log.info('Mosaic saved %s', res)


class STPTMosaic:
    """STPT Mosaic

    Parameters
    ----------
    arr
        Zarr array containing the full STPT sample.
    """

    def __init__(self, arr: zarr.Array):
        self._arr = arr

    def sections(self):
        """Sections generator
        """
        n = self._arr.attrs['sections']
        for i in range(n):
            section = self._arr[f'section={i+1:04d}']
            yield Section(section)
