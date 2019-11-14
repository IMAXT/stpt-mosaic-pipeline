import logging
import traceback
from pathlib import Path
from typing import Tuple

import dask.array as da
import numpy as np
import xarray as xr
import zarr
from dask import delayed
from distributed import Client, as_completed
from scipy.stats import median_absolute_deviation as mad

from imaxt_image.registration import find_overlap_conf

from .geometric_distortion import apply_geometric_transform
from .retry import retry
from .settings import Settings

log = logging.getLogger('owl.daemon.pipeline')


@delayed
def _sink(*args):
    return args


@retry(Exception)
def _get_image(group, imgtype, shape):
    try:
        arr = group.create_dataset(
            imgtype, shape=shape, chunks=(2080, 2080), dtype='float32'
        )
    except ValueError:
        arr = group[imgtype]
    log.info('Image %s/%s created', arr, imgtype)
    return arr


@delayed
def _mosaic(im_t, conf, abs_pos, abs_err, out=None, out_shape=None):
    y_size, x_size = out_shape
    y_delta, x_delta = np.min(abs_pos, axis=0)

    log.debug('Mosaic size: %dx%d', x_size, y_size)

    big_picture = _get_image(out, 'raw', (y_size, x_size))
    overlap = _get_image(out, 'overlap', (y_size, x_size))
    pos_err = _get_image(out, 'pos_err', (y_size, x_size))

    for i in range(len(im_t)):
        y0 = int(abs_pos[i, 0] - y_delta)
        x0 = int(abs_pos[i, 1] - x_delta)
        yslice = slice(y0, y0 + im_t[i].shape[0])
        xslice = slice(x0, x0 + im_t[i].shape[1])

        try:
            big_picture[yslice, xslice] += im_t[i][:]
            overlap[yslice, xslice] += conf[:]
            pos_err[yslice, xslice] += abs_err[i]
        except ValueError:
            log.error(traceback.format_exc())

    return out


class Section:
    """STPT section

    Parameters
    ----------
    section
        Zarr group containing the section data
    """

    def __init__(self, section: xr.DataArray):
        self._section = section

    def __getitem__(self, attr):
        res = self._section.attrs[attr]
        if isinstance(res, str) and res.isnumeric():
            res = int(res)
        elif isinstance(res, str):
            try:
                res = float(res)
            except ValueError:
                pass
        return res

    @property
    def path(self) -> str:
        """Path of the section
        """
        return self._section.path

    @property
    def shape(self) -> Tuple[int, int]:
        ncols = len(self._section.x)
        nrows = len(self._section.y)
        return (nrows, ncols)

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
        return len(self._section.tile)

    @property
    def channels(self) -> int:
        """Number of channels
        """
        return len(self._section.channel)

    @property
    def slices(self):
        """Number of optical sections
        """
        return len(self._section.z)

    def get_img_section(self, offset: int, channel: int) -> xr.DataArray:
        img_cube = self._section.sel(z=offset, channel=channel)
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

        r = np.sqrt((dx.astype(float) - dx[0]) ** 2 + (dy.astype(float) - dy[0]) ** 2)
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

    def get_mos_pos(self) -> np.ndarray:
        """Find the mosaic grid

        Returns
        -------
        dx, dy
            Coordinates of each image in mosaic units
        """
        dx, dy = self['XPos'], self['YPos']
        return np.stack((dx, dy))

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
        conf = da.ones(self.shape)
        res = apply_geometric_transform(conf, 1.0, Settings.cof_dist)
        return res

    def find_offsets(self):
        """Calculate offsets between all pairs of overlapping images
        """
        client = Client.current()
        # convert to find_shifts
        results = []
        log.info('Processing section %s', self._section.name)
        # TODO: this has to go in config
        img_cube = self.get_img_section(0, 3)

        # Calculate confidence map. Only needs to be done once per section
        dist_conf = self.get_distconf()

        dx0, dy0 = self.find_grid()
        dx_mos, dy_mos = self.get_mos_pos()

        for i, img in enumerate(img_cube):
            r = np.sqrt((dx0 - dx0[i]) ** 2 + (dy0 - dy0[i]) ** 2)

            # including no diagonals

            i_t = np.where((r <= np.sqrt(1)) & (r > 0))[0].tolist()

            im_i = img

            for this_img in i_t:
                if i > this_img:
                    continue

                desp = [
                    (dx_mos[this_img] - dx_mos[i]) / Settings.mosaic_scale,
                    (dy_mos[this_img] - dy_mos[i]) / Settings.mosaic_scale,
                ]

                # trying to do always positive displacements

                if (desp[0] < -1000) | (desp[1] < -1000):
                    desp[0] *= -1
                    desp[1] *= -1
                    obj_img = i
                    im_obj = im_i
                    ref_img = this_img
                    im_ref = img_cube[this_img]
                    sign = -1
                else:
                    obj_img = this_img
                    im_obj = img_cube[this_img]
                    ref_img = i
                    im_ref = im_i
                    sign = 1

                log.info(
                    'Initial offsets i: %d j: %d dx: %d dy: %d',
                    ref_img,
                    obj_img,
                    desp[0],
                    desp[1],
                )

                res = delayed(find_overlap_conf)(
                    im_ref.data, im_obj.data, dist_conf, dist_conf, desp
                )
                results.append(_sink(ref_img, obj_img, res, sign))

        futures = client.compute(results)
        offsets = []
        for fut in as_completed(futures):
            i, j, res, sign = fut.result()
            dx, dy, mi, npx = res.x, res.y, res.mi, res.avg_flux
            offsets.append([i, j, dy, dx, npx, mi, sign])
            log.info(
                'Section %s offsets i: %d j: %d dx: %d dy: %d mi: %f avg_f: %f sign: %d',
                self._section.name,
                i,
                j,
                dy,
                dx,
                mi,
                npx,
                sign,
            )

        self._section.attrs['offsets'] = offsets

    def compute_scale(self):
        """Return the scale of the detector
        """
        # Mu to px scale
        # we remove null theoretical displacements, bad measurements and
        # the random measured jitter (dx<1000) as these are not supposed to happen,
        # and they throw the ratio off
        x_scale = np.median(
            [
                m[0] / p[0]
                for m, p, a in zip(
                    self._mu.values(), self._px.values(), self._avg.values()
                )
                if (abs(p[0]) < 5000) & (abs(p[0] > 500) & (a > 0.5))
            ]
        )
        y_scale = np.median(
            [
                m[1] / p[1]
                for m, p, a in zip(
                    self._mu.values(), self._px.values(), self._avg.values()
                )
                if (abs(p[1]) < 5000) & (abs(p[1] > 00) & (a > 0.5))
            ]
        )

        log.debug('Scale %f %f', x_scale, y_scale)

        return x_scale, y_scale

    def compute_pairs(self):
        """Return the pairs of mags of the detector
        """
        offsets = self['offsets']
        dx, dy = self['XPos'], self['YPos']

        px = {}
        mu = {}
        mi = {}
        avg_flux = {}

        for o in offsets:
            px[f'{o[0]}:{o[1]}'] = o[2:4]
            px[f'{o[1]}:{o[0]}'] = [-1.0 * o[2], -1.0 * o[3]]

            mu[f'{o[0]}:{o[1]}'] = [dx[o[1]] - dx[o[0]], dy[o[1]] - dy[o[0]]]
            mu[f'{o[1]}:{o[0]}'] = [dx[o[0]] - dx[o[1]], dy[o[0]] - dy[o[1]]]

            mi[f'{o[0]}:{o[1]}'] = o[-2]
            mi[f'{o[1]}:{o[0]}'] = o[-2]

            avg_flux[f'{o[0]}:{o[1]}'] = o[-3]
            avg_flux[f'{o[1]}:{o[0]}'] = o[-3]

        self._px, self._mu = px, mu
        self._mi, self._avg = mi, avg_flux

    def get_default_displacement(self):
        """Calculates the default displacements in X and Y
        from the offsets estimated for every pair of images.
        Measured individual displacements are weighted with
        the average flux in the overlap region
        """

        offsets = self['offsets']

        px_x = np.zeros(len(offsets))
        px_y = np.zeros(len(offsets))
        avg_flux = np.zeros(len(offsets))
        for o in range(len(offsets)):
            px_x[o] = offsets[o][2]
            px_y[o] = offsets[o][3]
            avg_flux[o] = offsets[o][-2]

        ii = np.where(
            (np.abs(px_y) < 1000)
            & (np.abs(px_x) > 1000)
            & (avg_flux > np.median(avg_flux))
        )[0]
        default_x = [
            (px_x[ii] * avg_flux[ii] ** 2).sum() / (avg_flux[ii] ** 2).sum(),
            (px_y[ii] * avg_flux[ii] ** 2).sum() / (avg_flux[ii] ** 2).sum(),
        ]
        dev_x = np.sqrt(mad(px_x[ii]) ** 2 + mad(px_y[ii]) ** 2)
        ii = np.where(
            (np.abs(px_x) < 1000)
            & (np.abs(px_y) > 1000)
            & (avg_flux > np.median(avg_flux))
        )[0]
        default_y = [
            (px_x[ii] * avg_flux[ii] ** 2).sum() / (avg_flux[ii] ** 2).sum(),
            (px_y[ii] * avg_flux[ii] ** 2).sum() / (avg_flux[ii] ** 2).sum(),
        ]
        dev_y = np.sqrt(mad(px_x[ii]) ** 2 + mad(px_y[ii]) ** 2)

        log.debug(
            'Default displacement X:  dx:%f , dy:%f, std:%f',
            default_x[0],
            default_x[1],
            dev_x,
        )
        log.debug(
            'Default displacement Y:  dx:%f , dy:%f, std:%f',
            default_y[0],
            default_y[1],
            dev_y,
        )

        return default_x, dev_x, default_y, dev_y

    def compute_abspos(self):
        """Compute absolute positions of images in the field of view.
        """
        log.info('Processing section %s', self._section.name)
        self.compute_pairs()
        scale_x, scale_y = self.compute_scale()

        img_cube = self.get_img_section(0, 1)
        img_cube_stack = da.stack(img_cube)
        # ref image is the one with the max integrated flux
        ref_img_assemble = img_cube_stack.sum(axis=(1, 2)).argmax()
        ref_img_assemble = ref_img_assemble.compute()
        # ref_img_assemble = 0
        log.debug('Reference image %d', ref_img_assemble)
        dx0, dy0 = self.find_grid()
        default_x, dev_x, default_y, dev_y = self.get_default_displacement()

        abs_pos = np.zeros((len(dx0), 2))
        abs_pos[ref_img_assemble] = [0.0, 0.0]
        abs_err = np.zeros(len(dx0))
        abs_err[ref_img_assemble] = 0.0
        fixed_pos = [ref_img_assemble]
        pos_quality = np.zeros(len(dx0))
        pos_quality[ref_img_assemble] = 1.0
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
                temp_pos = [[], []]
                temp_err = []
                weight = []
                temp_qual = []
                prov_im = []
                for ref_img in fixed_pos:
                    desp_grid_x = dx0[ref_img] - dx0[this_img]
                    desp_grid_y = dy0[ref_img] - dy0[this_img]

                    if desp_grid_x ** 2 + desp_grid_y ** 2 == 1.0:

                        if desp_grid_x ** 2 > desp_grid_y ** 2:
                            # x displacement
                            default_desp = [-1.0 * default_x[0], -1.0 * default_x[1]]
                            if desp_grid_x < 0:
                                default_desp = [default_x[0], default_x[1]]
                            # only differences in the long displacement
                            # matter to see whether find_disp worked or not,
                            # as sometimes there are systematic differences in
                            # the other direction
                            err_px = np.sqrt(
                                (
                                    self._px[f'{ref_img}:{this_img}'][0]
                                    - self._mu[f'{ref_img}:{this_img}'][0] / scale_x
                                )
                                ** 2
                            )
                        else:
                            default_desp = [-1.0 * default_y[0], -1.0 * default_y[1]]
                            if desp_grid_y < 0:
                                default_desp = [default_y[0], default_y[1]]
                            # In the first column the displacement
                            # is offset from the rest of the sample by an unknown
                            # amount, so it is better to keep what is in the mosaic
                            if (dy0[ref_img] == 0.0) | (dy0[this_img] == 0.0):
                                default_desp[1] = (
                                    self._mu[f'{ref_img}:{this_img}'][1] / scale_y
                                )

                            err_px = np.sqrt(
                                (
                                    self._px[f'{ref_img}:{this_img}'][1]
                                    - self._mu[f'{ref_img}:{this_img}'][1] / scale_y
                                )
                                ** 2
                            )

                        if err_px < Settings.allowed_error_px:
                            # Dimensions in the mosaic and images have opposite directions
                            temp_pos[0].append(
                                abs_pos[ref_img, 0]
                                + self._px[f'{ref_img}:{this_img}'][0]
                            )
                            temp_pos[1].append(
                                abs_pos[ref_img, 1]
                                + self._px[f'{ref_img}:{this_img}'][1]
                            )
                            temp_err.append(
                                1.0
                            )  # this seems to be the error for good pairs
                            # weights
                            weight.append(self._avg[f'{ref_img}:{this_img}'] ** 2)
                            temp_qual.append(pos_quality[ref_img])
                            prov_im.append(ref_img)
                        else:
                            temp_pos[0].append(abs_pos[ref_img, 0] + default_desp[0])
                            temp_pos[1].append(abs_pos[ref_img, 1] + default_desp[1])
                            temp_err.append(Settings.allowed_error_px ** 2)
                            weight.append(0.01 ** 2)  # artificially low weight
                            temp_qual.append(0.1)
                            prov_im.append(-1.0 * ref_img)

                weight = np.array(weight) * np.array(temp_qual)

                abs_pos[this_img, 0] = int(
                    np.round((np.array(temp_pos[0]) * weight).sum() / weight.sum())
                )
                abs_pos[this_img, 1] = int(
                    np.round((np.array(temp_pos[1]) * weight).sum() / weight.sum())
                )
                abs_err[this_img] = np.sqrt(
                    (np.array(temp_err) * weight).sum() / weight.sum()
                )
                # this image has been fixed
                fixed_pos.append(this_img)
                pos_quality[this_img] = np.mean(temp_qual)
                done += 1

                prov_str = ''
                for im_temp in prov_im:
                    prov_str += '{0:+02d}'.format(int(im_temp))
                log.debug(
                    'Done Img %d from %s with quality %f',
                    this_img,
                    prov_str,
                    np.mean(temp_qual),
                )
            if done == len(dx0):
                break

        self._section.attrs['abs_pos'] = abs_pos.tolist()
        self._section.attrs['pos_quality'] = pos_quality.tolist()
        return abs_pos, abs_err

    def stitch(self, output: Path):
        """Stitch and save all images
        """
        client = Client.current()

        abs_pos, abs_err = self.compute_abspos()

        conf = self.get_distconf()

        # TODO: This is the start of staging support
        # (i.e creating the mosaic in temp storage)
        out_shape = (self['mrows'] * self.shape[0], self['mcolumns'] * self.shape[1])
        results = []
        z = zarr.open(f'{output}_temp.zarr', mode='w')

        for sl in range(self.slices):
            for ch in range(self.channels):
                im_t = self.get_img_section(sl, ch)
                g = z.create_group(f'/mosaic/{self._section.name}/z={sl}/channel={ch}')
                res = _mosaic(im_t, conf, abs_pos, abs_err, out=g, out_shape=out_shape)
                results.append(res)

        futures = client.compute(results)
        for fut in as_completed(futures):
            res = fut.result()
            log.info('Mosaic saved %s', res)

        # Move mosaic to final destination with correct format
        mos_overlap = []
        mos_raw = []
        mos_err = []
        for name, offset in z[f'mosaic/{self._section.name}'].groups():
            raw = da.stack(
                [
                    da.from_zarr(ch['raw']) / da.from_zarr(ch['overlap'])
                    for name, ch in offset.groups()
                ]
            )
            overlap = da.stack(
                [da.from_zarr(ch['overlap']) for name, ch in offset.groups()]
            )
            err = da.stack(
                [da.from_zarr(ch['pos_err']) for name, ch in offset.groups()]
            )
            mos_overlap.append(overlap)
            mos_err.append(err)
            mos_raw.append(raw)
        mos_raw = da.stack(mos_raw)
        mos_overlap = da.stack(mos_overlap)
        mos_err = da.stack(mos_err)
        mos = da.stack([mos_raw, mos_overlap, mos_err])
        nt, nz, nch, ny, nx = mos.shape

        raw = xr.DataArray(
            mos.astype('float32'),
            dims=('type', 'z', 'channel', 'y', 'x'),
            coords={
                'type': ['mosaic', 'conf', 'err'],
                'x': range(nx),
                'y': range(ny),
                'z': range(nz),
                'channel': range(nch),
            },
        )

        ds = xr.Dataset({self._section.name: raw})
        ds.to_zarr(f'{output}_mos.zarr', mode='a')


class STPTMosaic:
    """STPT Mosaic

    Parameters
    ----------
    arr
        Zarr array containing the full STPT sample.
    """

    def __init__(self, filename: Path):
        self._ds = xr.open_zarr(f'{filename}')

    def initialize_storage(self, output):
        ds = xr.Dataset()
        ds.to_zarr(f'{output}_mos.zarr', mode='w')

    def sections(self):
        """Sections generator
        """
        for section in self._ds:
            yield Section(self._ds[section])
