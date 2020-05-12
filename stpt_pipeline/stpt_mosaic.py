import logging
import traceback
from pathlib import Path
from typing import List, Tuple

import cv2
import dask
import dask.array as da
import numpy as np
import xarray as xr
import zarr
from dask import delayed
from distributed import Client, as_completed
from scipy.stats import median_absolute_deviation as mad

from imaxt_image.external.tifffile import TiffWriter
from imaxt_image.registration import find_overlap_conf

from .geometric_distortion import apply_geometric_transform, read_calib
from .retry import retry
from .settings import Settings

log = logging.getLogger("owl.daemon.pipeline")


@delayed
def _sink(*args):
    return args


@delayed
def arr2tiff(output, arr, scl):
    ch = str(arr.channel.values)
    z = str(arr.z.values)
    p = Path(f"{output}/tiff/mos{scl}/{arr.name}/Z{z}")
    p.mkdir(exist_ok=True, parents=True)
    filename = p / f"{output.name}-{arr.name}-z{z}-ch{ch}.tif"
    with TiffWriter(f"{filename}") as writer:
        writer.save(arr.astype("uint16"), compress=6)
    return filename


@retry(Exception)
def _get_image(group, imgtype, shape, dtype="float32"):
    try:
        arr = group.create_dataset(
            imgtype, shape=shape, chunks=(2080, 2080), dtype=dtype
        )
    except ValueError:
        arr = group[imgtype]
    log.info("Image %s/%s created", arr, imgtype)
    return arr


@delayed
def cwrite(im, conf, abs_err, xslice, yslice, big_picture, overlap, pos_err):
    mos = im.data * conf
    big_picture[yslice, xslice] = big_picture[yslice, xslice] + mos
    overlap[yslice, xslice] = overlap[yslice, xslice] + conf
    pos_err[yslice, xslice] = pos_err[yslice, xslice] + \
        np.sqrt(np.sum(np.array(abs_err)**2))


def _mosaic(im_t, conf, abs_pos, abs_err, out=None, out_shape=None):
    assert out_shape is not None
    assert out is not None
    y_size, x_size = out_shape
    y_delta, x_delta = np.min(abs_pos, axis=0)
    log.debug("Mosaic size: %dx%d", x_size, y_size)
    big_picture = _get_image(out, "raw", (y_size, x_size), dtype="float32")
    overlap = _get_image(out, "overlap", (y_size, x_size), dtype="float32")
    pos_err = _get_image(out, "pos_err", (y_size, x_size), dtype="float32")
    for i, im in enumerate(im_t):
        y0 = int(abs_pos[i, 0] - y_delta)
        x0 = int(abs_pos[i, 1] - x_delta)
        yslice = slice(y0, y0 + im.shape[0])
        xslice = slice(x0, x0 + im.shape[1])
        # We do each image synchronously
        # Better would be create a list of those that do not everlap,
        # make a list of delayed and compute those, then fill the gaps
        # with another list of non overlapping, etc.
        try:
            cwrite(im, conf, abs_err[i], xslice, yslice, big_picture, overlap,
                   pos_err).compute()
        except Exception:
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
        self.stage_size = [0, 0]

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
        """Mosaic shape in rows and columns
        """
        ncols = len(self._section.x)
        nrows = len(self._section.y)
        return (nrows, ncols)

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
        dx = np.array(self["XPos"])
        dy = np.array(self["YPos"])

        # r = np.sqrt((dx.astype(float) - dx[0]) **
        #             2 + (dy.astype(float) - dy[0]) ** 2)
        # r[0] = r.max()  # avoiding minimum at itself
        # # first candidate
        # dx_1 = np.abs(dx[r.argmin()] - dx[0])
        # dy_1 = np.abs(dy[r.argmin()] - dy[0])
        # # second candidate
        # r[r.argmin()] = r.max()
        # dx_2 = np.abs(dx[r.argmin()] - dx[0])
        # dy_2 = np.abs(dy[r.argmin()] - dy[0])

        t_x = dx[1:] - dx[:-1]
        corners = np.where(t_x == 0)[0]
        t_y = dy[corners + 1] - dy[corners]

        delta_x = np.median(np.abs(t_x))
        delta_y = np.median(np.abs(t_y))

        # delta_x, delta_y = np.max([dx_1, dx_2]), np.max([dy_1, dy_2])
        log.debug(
            "Section %s delta_x : %s , delta_y: %s",
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
        dx, dy = self["XPos"], self["YPos"]
        return np.stack((dx, dy))

    def get_distconf(self):
        flat = read_calib(Settings.flat_file)
        flat = flat[Settings.channel_to_use - 1]
        conf = da.where((flat < 0.3) | (flat > 5), 0, 1)
        res = apply_geometric_transform(conf, 0.0, 1.0, Settings.cof_dist)
        return res.astype("uint8")

    def find_offsets(self):
        """Calculate offsets between all pairs of overlapping images
        """
        client = Client.current()
        # convert to find_shifts
        results = []
        log.info("Processing section %s", self._section.name)
        img_cube = self.get_img_section(0, Settings.channel_to_use - 1)

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

                if (desp[0] < -100) or (desp[1] < -100):
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

                log.debug(
                    "Initial offsets i: %d j: %d dx: %d dy: %d",
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
            dx, dy, mi, avf = res.x, res.y, res.mi, res.avg_flux
            offsets.append([i, j, res, sign])
            log.debug(
                "Section %s offsets i: %d j: %d dx: %d dy: %d mi: %f avg_f: %f sign: %d",
                self._section.name,
                i,
                j,
                dx,
                dy,
                mi,
                avf,
                sign,
            )

        self._offsets = offsets

    @property
    def scale(self) -> Tuple[float]:
        """Scale of the detector in pixels per micron
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
                if (abs(p[1]) < 5000) & (abs(p[1] > 500) & (a > 0.5))
            ]
        )

        log.debug("Scale %f %f", x_scale, y_scale)

        self._x_scale, self._y_scale = x_scale, y_scale
        return (x_scale, y_scale)

    def compute_pairs(self):
        """Return the pairs of mags of the detector
        """
        offsets = self._offsets
        dx, dy = self["XPos"], self["YPos"]

        px = {}
        mu = {}
        mi = {}
        avg_flux = {}

        for i, j, res, _sign in offsets:
            px[f"{i}:{j}"] = [res.y, res.x]
            px[f"{j}:{i}"] = [-res.y, -res.x]

            mu[f"{i}:{j}"] = [dx[j] - dx[i], dy[j] - dy[i]]
            mu[f"{j}:{i}"] = [dx[i] - dx[j], dy[i] - dy[j]]

            mi[f"{i}:{j}"] = res.mi
            mi[f"{j}:{i}"] = res.mi

            avg_flux[f"{i}:{j}"] = res.avg_flux
            avg_flux[f"{j}:{i}"] = res.avg_flux

        self._px, self._mu = px, mu
        self._mi, self._avg = mi, avg_flux

    def get_default_displacement(self):
        """Calculates the default displacements in X and Y
        from the offsets estimated for every pair of images.

        Measured individual displacements are weighted with
        the average flux in the overlap region
        """

        offsets = self._offsets

        px_x = np.empty(len(offsets))
        px_y = np.empty(len(offsets))
        avg_flux = np.empty(len(offsets))
        for i, offset in enumerate(offsets):
            _i, _j, res, _sign = offset
            px_y[i] = res.y
            px_x[i] = res.x
            avg_flux[i] = res.avg_flux

        # these are the thresholds to tell when
        # the micro is doing the long move in one direction
        short_displacement_x = np.nanmean(np.abs(px_x))
        short_displacement_y = np.nanmean(np.abs(px_y))

        ii = np.where(
            (np.abs(px_x) < short_displacement_x)
            & (np.abs(px_y) > short_displacement_y)
            & (avg_flux > np.median(avg_flux))
        )[0]
        default_x = [
            (np.abs(px_y[ii]) * avg_flux[ii] ** 2).sum() /
            (avg_flux[ii] ** 2).sum(),
            (np.abs(px_x[ii]) * avg_flux[ii] ** 2).sum() /
            (avg_flux[ii] ** 2).sum(),
        ]
        dev_x = np.sqrt(mad(px_x[ii]) ** 2 + mad(px_y[ii]) ** 2)

        ii = np.where(
            (np.abs(px_y) < short_displacement_y)
            & (np.abs(px_x) > short_displacement_x)
            & (avg_flux > np.median(avg_flux))
        )[0]
        default_y = [
            (np.abs(px_y[ii]) * avg_flux[ii] ** 2).sum() /
            (avg_flux[ii] ** 2).sum(),
            (np.abs(px_x[ii]) * avg_flux[ii] ** 2).sum() /
            (avg_flux[ii] ** 2).sum(),
        ]
        dev_y = np.sqrt(mad(px_x[ii]) ** 2 + mad(px_y[ii]) ** 2)

        log.debug(
            "Default displacement X:  dx:%f , dy:%f, std:%f",
            default_x[0],
            default_x[1],
            dev_x,
        )
        log.debug(
            "Default displacement Y:  dx:%f , dy:%f, std:%f",
            default_y[0],
            default_y[1],
            dev_y,
        )

        self._section.attrs["default_displacements"] = [
            {
                "default_x": default_x,
                "dev_x": dev_x,
                "default_y": default_y,
                "dev_y": dev_y,
            }
        ]
        return default_x, dev_x, default_y, dev_y

    def compute_abspos_ref(
        self,
        dx0, dy0,
        default_x, dev_x,
        default_y, dev_y,
        scale_x, scale_y,
        ref_img_assemble,
        error_long_threshold,
        error_short_threshold
    ):

        # error_threshold = Settings.allowed_error_px

        abs_pos = np.zeros((len(dx0), 2))
        abs_pos[ref_img_assemble] = [0.0, 0.0]
        abs_err = np.zeros(len(dx0))
        abs_err[ref_img_assemble] = 0.0
        fixed_pos = [ref_img_assemble]
        pos_quality = np.zeros(len(dx0))
        pos_quality[ref_img_assemble] = 1.0
        # now for every fixed, see which ones overlap
        for i in fixed_pos:
            r = np.sqrt((dx0 - dx0[i]) ** 2 + (dy0 - dy0[i]) ** 2)
            i_t = np.where(r == 1)[0].tolist()

            for this_img in i_t:
                if this_img in fixed_pos:
                    continue

                # Now to calculate the final disp, we have to check
                # if this has been measured more than once
                temp_pos = [[], []]
                weight = []
                temp_qual = []
                prov_im = []
                for ref_img in fixed_pos:
                    if (f"{ref_img}:{this_img}" in self._px.keys()) == False:
                        continue

                    desp_grid_x = dx0[ref_img] - dx0[this_img]
                    desp_grid_y = dy0[ref_img] - dy0[this_img]

                    if desp_grid_x ** 2 + desp_grid_y ** 2 == 1.0:

                        if desp_grid_x ** 2 > desp_grid_y ** 2:
                            # x displacement
                            default_desp = [
                                -1.0 * default_x[0],
                                -1.0 * default_x[1]
                            ]
                            if desp_grid_x < 0:
                                default_desp = [default_x[0], default_x[1]]
                            # only differences in the long displacement
                            # matter to see whether find_disp worked or not,
                            # as sometimes there are systematic differences in
                            # the other direction
                            err_long_px = np.sqrt(
                                (
                                    self._px[f"{ref_img}:{this_img}"][0]
                                    - self._mu[f"{ref_img}:{this_img}"][0] / scale_x
                                ) ** 2
                            )
                            err_short_px = np.sqrt(
                                (
                                    self._px[f"{ref_img}:{this_img}"][1]
                                    - self._mu[f"{ref_img}:{this_img}"][1] / scale_y
                                ) ** 2
                            )

                        else:
                            default_desp = [
                                -1.0 * default_y[0],
                                -1.0 * default_y[1]
                            ]
                            if desp_grid_y < 0:
                                default_desp = [default_y[0], default_y[1]]
                            # In the first column the displacement
                            # is offset from the rest of the sample by an unknown
                            # amount, so it is better to keep what is in the mosaic
                            if (dy0[ref_img] == 0.0) | (dy0[this_img] == 0.0):
                                default_desp[1] = (
                                    self._mu[f"{ref_img}:{this_img}"][1] / scale_y
                                )

                            err_long_px = np.sqrt(
                                (
                                    self._px[f"{ref_img}:{this_img}"][1]
                                    - self._mu[f"{ref_img}:{this_img}"][1] / scale_y
                                )
                                ** 2
                            )
                            err_short_px = np.sqrt(
                                (
                                    self._px[f"{ref_img}:{this_img}"][0]
                                    - self._mu[f"{ref_img}:{this_img}"][0] / scale_x
                                ) ** 2
                            )

                    if (err_long_px < error_long_threshold) & (err_short_px < error_short_threshold):
                        # Dimensions in the mosaic and images have opposite directions
                        temp_pos[0].append(
                            abs_pos[ref_img, 0]
                            + self._px[f"{ref_img}:{this_img}"][0]
                        )
                        temp_pos[1].append(
                            abs_pos[ref_img, 1]
                            + self._px[f"{ref_img}:{this_img}"][1]
                        )
                        # weights
                        weight.append(
                            self._avg[f"{ref_img}:{this_img}"] ** 2)
                        temp_qual.append(pos_quality[ref_img])
                        prov_im.append(ref_img)
                    else:
                        temp_pos[0].append(
                            abs_pos[ref_img, 0] + default_desp[0])
                        temp_pos[1].append(
                            abs_pos[ref_img, 1] + default_desp[1])
                        weight.append(0.0000001 ** 2)  # artificially low weight
                        temp_qual.append(0.0)
                        prov_im.append(-1.0 * ref_img)

                weight = np.array(weight)

                abs_pos[this_img, 0] = int(
                    np.round(
                        (np.array(temp_pos[0]) * weight).sum() / weight.sum())
                )
                abs_pos[this_img, 1] = int(
                    np.round(
                        (np.array(temp_pos[1]) * weight).sum() / weight.sum())
                )

                # this image has been fixed
                fixed_pos.append(this_img)
                pos_quality[this_img] = (
                    np.array(temp_qual) * weight).sum() / weight.sum()

        return abs_pos, pos_quality

    def compute_abspos(self):  # noqa: C901
        """Compute absolute positions of images in the field of view.
        """
        log.info("Processing section %s", self._section.name)
        self.compute_pairs()
        scale_x, scale_y = self.scale
        dx0, dy0 = self.find_grid()
        default_x, dev_x, default_y, dev_y = self.get_default_displacement()

        # get error threshold scomparing scaled micron displacements
        # with measurements
        px_x_temp = []
        px_y_temp = []
        mu_x_temp = []
        mu_y_temp = []
        for this_key in self._px.keys():
            px_x_temp.append(self._px[this_key][0])
            px_y_temp.append(self._px[this_key][1])
            mu_x_temp.append(self._mu[this_key][0] / scale_x)
            mu_y_temp.append(self._mu[this_key][1] / scale_y)
        px_x_temp = np.array(px_x_temp)
        px_y_temp = np.array(px_y_temp)
        mu_x_temp = np.array(mu_x_temp)
        mu_y_temp = np.array(mu_y_temp)

        ind_temp = np.where(np.abs(px_x_temp) > np.mean(np.abs(px_x_temp)))[0]
        elongx = 1.48 * mad(np.sqrt((px_x_temp - mu_x_temp)[ind_temp] ** 2))
        eshorty = 1.48 * mad(np.sqrt((px_y_temp - mu_y_temp)[ind_temp] ** 2))
        ind_temp = np.where(np.abs(px_y_temp) > np.mean(np.abs(px_y_temp)))[0]
        elongy = 1.48 * mad(np.sqrt((px_y_temp - mu_y_temp)[ind_temp] ** 2))
        eshortx = 1.48 * mad(np.sqrt((px_x_temp - mu_x_temp)[ind_temp] ** 2))

        error_long_threshold = 3.0 * np.max([elongx, elongy])
        error_short_threshold = 3.0 * np.max([eshortx, eshorty])
        log.debug('Scaling error threshold: l:{0:3f} s:{1:.3f}'.format(
            error_long_threshold, error_short_threshold))

        img_cube = self.get_img_section(0, 1)
        img_cube_stack = da.stack(img_cube)
        # ref image is the one with the max integrated flux
        cube_totals = img_cube_stack.sum(axis=(1, 2))
        cube_totals = cube_totals.compute()
        cube_means = cube_totals / 2080.**2

        absolute_ref_img = cube_means.argmax()

        accumulated_pos = []
        accumulated_qual = []
        for this_ref in np.where(cube_means > np.median(cube_means))[0]:

            temp = self.compute_abspos_ref(
                dx0, dy0, default_x, dev_x,
                default_y, dev_y, scale_x, scale_y,
                this_ref, error_long_threshold, error_short_threshold
            )
            # using a common reference
            accumulated_pos.append(
                np.array(temp[0]) - np.array(temp[0])[absolute_ref_img]
            )
            # adding quality of global reference
            accumulated_qual.append(
                np.sqrt(
                    np.array(temp[1]) ** 2 +
                    np.array(temp[1])[absolute_ref_img] ** 2
                )
            )

        accumulated_pos = np.array(accumulated_pos)
        abs_pos = np.median(accumulated_pos, 0)

        # t = np.argmax(accumulated_qual, 0)
        # for i in range(len(t)):
        #     abs_pos[i, 0] = accumulated_pos[t[i], i, 0]
        #     abs_pos[i, 1] = accumulated_pos[t[i], i, 1]

        abs_err = np.std(accumulated_pos, 0)

        self._section.attrs["abs_pos"] = abs_pos.tolist()
        self._section.attrs["abs_err"] = abs_err.tolist()

        i = 0
        for t in abs_pos.tolist():
            log.info('  {0:d}: {1:.2f},{2:.2f}'.format(i, *t))
            i += 1

        return abs_pos, abs_err

    def compute_abspos_old(self):  # noqa: C901
        """Compute absolute positions of images in the field of view.
        """
        log.info("Processing section %s", self._section.name)
        self.compute_pairs()
        scale_x, scale_y = self.scale

        img_cube = self.get_img_section(0, 1)
        img_cube_stack = da.stack(img_cube)
        # ref image is the one with the max integrated flux
        ref_img_assemble = img_cube_stack.sum(axis=(1, 2)).argmax()
        ref_img_assemble = ref_img_assemble.compute()
        # ref_img_assemble = 0
        log.debug("Reference image %d", ref_img_assemble)
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
                            default_desp = [-1.0 *
                                            default_x[0], -1.0 * default_x[1]]
                            if desp_grid_x < 0:
                                default_desp = [default_x[0], default_x[1]]
                            # only differences in the long displacement
                            # matter to see whether find_disp worked or not,
                            # as sometimes there are systematic differences in
                            # the other direction
                            err_px = np.sqrt(
                                (
                                    self._px[f"{ref_img}:{this_img}"][0]
                                    - self._mu[f"{ref_img}:{this_img}"][0] / scale_x
                                ) **
                                2
                            )
                        else:
                            default_desp = [-1.0 *
                                            default_y[0], -1.0 * default_y[1]]
                            if desp_grid_y < 0:
                                default_desp = [default_y[0], default_y[1]]
                            # In the first column the displacement
                            # is offset from the rest of the sample by an unknown
                            # amount, so it is better to keep what is in the mosaic
                            if (dy0[ref_img] == 0.0) | (dy0[this_img] == 0.0):
                                default_desp[1] = (
                                    self._mu[f"{ref_img}:{this_img}"][1] / scale_y
                                )

                            err_px = np.sqrt(
                                (
                                    self._px[f"{ref_img}:{this_img}"][1]
                                    - self._mu[f"{ref_img}:{this_img}"][1] / scale_y
                                ) **
                                2
                            )

                        if err_px < Settings.allowed_error_px:
                            # Dimensions in the mosaic and images have opposite directions
                            temp_pos[0].append(
                                abs_pos[ref_img, 0]
                                + self._px[f"{ref_img}:{this_img}"][0]
                            )
                            temp_pos[1].append(
                                abs_pos[ref_img, 1]
                                + self._px[f"{ref_img}:{this_img}"][1]
                            )
                            temp_err.append(
                                1.0
                            )  # this seems to be the error for good pairs
                            # weights
                            weight.append(
                                self._avg[f"{ref_img}:{this_img}"] ** 2)
                            temp_qual.append(pos_quality[ref_img])
                            prov_im.append(ref_img)
                        else:
                            temp_pos[0].append(
                                abs_pos[ref_img, 0] + default_desp[0])
                            temp_pos[1].append(
                                abs_pos[ref_img, 1] + default_desp[1])
                            temp_err.append(Settings.allowed_error_px ** 2)
                            weight.append(0.01 ** 2)  # artificially low weight
                            temp_qual.append(0.1)
                            prov_im.append(-1.0 * ref_img)

                weight = np.array(weight) * np.array(temp_qual)

                abs_pos[this_img, 0] = int(
                    np.round(
                        (np.array(temp_pos[0]) * weight).sum() / weight.sum())
                )
                abs_pos[this_img, 1] = int(
                    np.round(
                        (np.array(temp_pos[1]) * weight).sum() / weight.sum())
                )
                abs_err[this_img] = np.sqrt(
                    (np.array(temp_err) * weight).sum() / weight.sum()
                )
                # this image has been fixed
                fixed_pos.append(this_img)
                pos_quality[this_img] = np.mean(temp_qual)
                done += 1

                prov_str = ""
                for im_temp in prov_im:
                    prov_str += "{0:+02d}".format(int(im_temp))
                log.debug(
                    "Done Img %d from %s with quality %f",
                    this_img,
                    prov_str,
                    np.mean(temp_qual),
                )
            if done == len(dx0):
                break

        self._section.attrs["abs_pos"] = abs_pos.tolist()
        self._section.attrs["pos_quality"] = pos_quality.tolist()
        return abs_pos, abs_err

    def stitch(self, output: Path):
        """Stitch and save all images
        """
        client = Client.current()

        abs_pos, abs_err = self.compute_abspos()

        conf = self.get_distconf()

        # TODO: This is the start of staging support
        # (i.e creating the mosaic in temp storage)
        # out_shape = (self["mrows"] * self.shape[0],
        #              self["mcolumns"] * self.shape[1])

        if self.stage_size[0] == 0:

            shape_0 = int(
                np.array(abs_pos[:, 0]).max()
                - np.array(abs_pos[:, 0]).min()
            ) + self.shape[0]
            shape_1 = int(
                np.array(abs_pos[:, 1]).max()
                - np.array(abs_pos[:, 1]).min()
            ) + self.shape[1]

            # this has to be a multiple of the chunk size
            if shape_0 % 2080 != 0:
                n_imgs = int(shape_0 / 2080.) + 1
                shape_0 = int(2080 * n_imgs)
            if shape_1 % 2080 != 0:
                n_imgs = int(shape_1 / 2080.) + 1
                shape_1 = int(2080 * n_imgs)

            self.stage_size = (shape_0, shape_1)

        results = []

        z = zarr.open(f"{output}/temp.zarr", mode="w")

        # debug
        # for sl in range(self.slices):
        for sl in [0]:
            for ch in range(self.channels):
                im_t = self.get_img_section(sl, ch)
                g = z.create_group(
                    f"/mosaic/{self._section.name}/z={sl}/channel={ch}")
                res = _mosaic(im_t, conf, abs_pos, abs_err,
                              out=g, out_shape=self.stage_size)
                results.append(res)

        # Now when _mosaic is done the file is already written,
        # so this block thros an error
        # futures = client.compute(results)
        # for fut in as_completed(futures):
        #     res = fut.result()
        #     log.debug("Temporary mosaic saved %s", res)

        log.debug("Temporary mosaic saved %s", res)

        # Move mosaic to final destination with correct format
        mos_overlap = []
        mos_raw = []
        mos_err = []

        for name, offset in z[f"mosaic/{self._section.name}"].groups():
            raw = da.stack(
                [
                    da.from_zarr(ch["raw"]) / da.from_zarr(ch["overlap"])
                    for name, ch in offset.groups()
                ]
            )
            raw = (raw + 10) * 1_000
            overlap = (
                da.stack([da.from_zarr(ch["overlap"]) for name, ch in offset.groups()])
                * 100
            )
            err = (
                da.stack([da.from_zarr(ch["pos_err"]) for name, ch in offset.groups()])
                * 100
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
            mos.astype("uint16"),
            dims=("type", "z", "channel", "y", "x"),
            coords={
                "type": ["mosaic", "conf", "err"],
                "x": range(nx),
                "y": range(ny),
                "z": range(nz),
                "channel": range(nch + 1)[1:],
            },
        )

        raw.attrs["default_displacements"] = self._section.attrs[
            "default_displacements"
        ]
        raw.attrs["abs_pos"] = self._section.attrs["abs_pos"]
        raw.attrs["abs_err"] = self._section.attrs["abs_err"]
        raw.attrs["offsets"] = self._offsets
        raw.attrs["scale"] = [self._x_scale, self._y_scale]

        ds = xr.Dataset({self._section.name: raw})
        ds.to_zarr(output / "mos1.zarr", mode="a")
        log.info("Mosaic saved %s", output / "mos1.zarr")


class STPTMosaic:
    """STPT Mosaic

    Parameters
    ----------
    arr
        Zarr array containing the full STPT sample.
    """

    def __init__(self, filename: Path):
        self._ds = xr.open_zarr(f"{filename}")

        # this is to carry over the mosaic size between
        # sections

        self.stage_size = [0, 0]

    def initialize_storage(self, output: Path):
        ds = xr.Dataset()
        ds.to_zarr(output / "mos1.zarr", mode="w")

    def sections(self):
        """Sections generator
        """
        for section in self._ds:
            yield Section(self._ds[section])

    def downsample(self, output: Path, scales: List[int] = None):
        """Downsample mosaic.

        Parameters
        ----------
        output
            Location of output directory
        scales
            List of scales to produce. Default is
            [2, 4, 8, 16]
        """
        log.info("Downsampling mosaics")
        up = output / "mos1.zarr"
        if scales is None:
            scales = [2, 4, 8, 16]
        for factor in scales:
            log.debug("Downsampling factor %d", factor)
            down = output / f"mos{factor}.zarr"
            ds = xr.open_zarr(f"{up}")
            nds = xr.Dataset()
            nds.to_zarr(down, "w")

            for s in list(ds):
                log.debug("Downsampling mos%d [%s]", factor, s)
                nds = xr.Dataset()
                arr = ds[s]
                nt, nz, nch, ny, nx = tuple(c[0] for c in arr.chunks)
                arr_t = []
                for t in arr.type:
                    arr_z = []
                    for z in arr.z:
                        arr_ch = []
                        for ch in arr.channel:
                            im = arr.sel(z=z.values, type=t.values,
                                         channel=ch.values)
                            res = im.data.map_blocks(
                                cv2.pyrDown, chunks=(ny // 2, nx // 2)
                            )
                            arr_ch.append(res)
                        arr_z.append(da.stack(arr_ch))
                    arr_t.append(da.stack(arr_z))
                arr_t = da.stack(arr_t)
                nt, nz, nch, ny, nx = arr_t.shape
                narr = xr.DataArray(
                    arr_t.astype("uint16"),
                    dims=("type", "z", "channel", "y", "x"),
                    coords={
                        "channel": range(1, nch + 1),
                        "type": ["mosaic", "conf", "err"],
                        "x": range(nx),
                        "y": range(ny),
                        "z": range(nz),
                    },
                )
                nds[s] = narr
                nds.to_zarr(down, mode="a")
            log.info("Downsampled mosaic saved %s", down)

            up = down

    def to_tiff(self, output: Path, scales: List[int] = None):
        """Export mosaic to TIFF format.

        Parameters
        ----------
        output
            Location of output directory
        scales
            List of scales to produce. Default is
            [16, 8, 4, 2, 1]
        """
        if scales is None:
            scales = [16, 8, 4, 2, 1]
        for scl in scales:
            arr = f"{output}/mos{scl}.zarr"
            ds = xr.open_zarr(arr)
            files = []
            for section in list(ds):
                files = []
                for ch in list(ds.channel):
                    for z in list(ds.z):
                        im = ds[section].sel(type="mosaic", channel=ch, z=z)
                        res = arr2tiff(output, im, scl)
                        files.append(res)
                for tfile in dask.compute(files)[0]:
                    log.info("Written %s", tfile)
