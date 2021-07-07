import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import dask
import dask.array as da
import numpy as np
import xarray as xr
import zarr
from dask import delayed
from distributed import Client, as_completed, Lock
from imaxt_image.registration import find_overlap_conf
from owl_dev.logging import logger
from scipy.stats import median_absolute_deviation as mad

from . import ops
from .geometric_distortion import apply_geometric_transform, read_calib
from .retry import retry
from .settings import Settings


CHUNK_SIZE = 2080


@delayed
def _sink(*args):
    return args


@retry(Exception)
def _get_image(group, imgtype, shape, dtype="float32"):
    try:
        arr = group.create_dataset(
            imgtype, shape=shape, chunks=(CHUNK_SIZE, CHUNK_SIZE), dtype=dtype
        )
    except ValueError:
        arr = group[imgtype]
    return arr


@delayed
def _mosaic(im, ch, conf, pos, abs_err, imgtype, out):
    assert imgtype in ["raw", "pos_err", "overlap"]
    assert out is not None
    y0, x0 = pos
    yslice = slice(y0, y0 + im.shape[0])
    xslice = slice(x0, x0 + im.shape[1])

    lock_name = f"{imgtype}-{ch}"
    with Lock(lock_name):
        if imgtype == "raw":
            out[yslice, xslice] = out[yslice, xslice] + im * conf
        elif imgtype == "overlap":
            out[yslice, xslice] = out[yslice, xslice] + conf
        elif imgtype == "pos_err":
            out[yslice, xslice] = out[yslice, xslice] + \
                conf * np.sum(np.array(abs_err) ** 2)


class Section:
    """STPT section

    Parameters
    ----------
    section
        Zarr group containing the section data
    """

    def __init__(self, section: xr.DataArray, stage_size: List[int] = None):
        self._section = section
        self.stage_size = stage_size
        self.name = section.name

    def __getitem__(self, attr):
        res = self._section.attrs["raw_meta"][0][attr]
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
        """Path of the section"""
        return self._section.path

    @property
    def shape(self) -> Tuple[int, int]:
        """Mosaic shape in rows and columns"""
        ncols = len(self._section.x)
        nrows = len(self._section.y)
        return (nrows, ncols)

    @property
    def fovs(self) -> int:
        """Number of field of views."""
        return len(self._section.tile)

    @property
    def channels(self) -> int:
        """Number of channels"""
        return len(self._section.channel)

    @property
    def slices(self):
        """Number of optical sections"""
        return len(self._section.z)

    def get_img_section(self, offset: int, channel: int) -> xr.DataArray:

        # 20200717 added switch to add all channels

        if channel == -1:
            n = 0
            for this_channel in self._section.channel.values:
                if n == 0:
                    img_cube = self._section.sel(z=offset, channel=this_channel)
                else:
                    img_cube += self._section.sel(z=offset,
                                                  channel=this_channel)
                n += 1.0
        else:
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
        logger.debug(
            "Section %s delta_x : %s , delta_y: %s",
            self._section.name,
            delta_x,
            delta_y,
        )

        dx0 = np.round((dx - dx.min()) / delta_x).astype(int)
        dy0 = np.round((dy - dy.min()) / delta_y).astype(int)
        return (dx0, dy0, delta_x, delta_y)

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

        if self.cal_type == 'sample':
            logger.info('Reading ' + self.cal_zarr.name)
            xr_cal = xr.open_zarr(self.cal_zarr)
            flat = da.array(
                xr_cal['FLATS'].sel(
                    channel=Settings.channel_to_use - 1
                ).values
            )
        else:
            flat = read_calib(Settings.flat_file)
            flat = flat[Settings.channel_to_use - 1]

        conf = da.where((flat < 0.3) | (flat > 5), 0, 1)
        res = apply_geometric_transform(conf, 0.0, 1.0, Settings.cof_dist)
        return res.astype("uint8")

    def find_offsets(self):  # noqa: C901
        """Calculate offsets between all pairs of overlapping images"""
        client = Client.current()
        # convert to find_shifts
        results = []
        logger.info("Processing section %s", self._section.name)
        # debug
        # img_cube = self.get_img_section(0, Settings.channel_to_use - 1)
        img_cube = self.get_img_section(0, -1)

        # Calculate confidence map. Only needs to be done once per section
        dist_conf = self.get_distconf()

        if self.cal_type == 'sample':
            xr_cal = xr.open_zarr(self.cal_zarr)
            flat = da.array(
                xr_cal['FLATS'].sel(
                    channel=Settings.channel_to_use - 1
                ).values
            )
            dark = da.array(
                xr_cal['DARKS'].sel(
                    channel=Settings.channel_to_use - 1
                ).values
            )
        else:
            flat = read_calib(Settings.flat_file)[Settings.channel_to_use - 1]
            dark = (
                read_calib(Settings.dark_file)[Settings.channel_to_use - 1]
                / Settings.norm_val
            )

        flat = flat.persist()
        dark = dark.persist()

        dx0, dy0, delta_x, delta_y = self.find_grid()
        dx_mos, dy_mos = self.get_mos_pos()

        # We calculate the ref_img here, for if the slice
        # has no data (very low max. avg flux), we skip calculations too
        img_cube_stack = da.stack(img_cube)
        # ref image is the one with the max integrated flux
        cube_totals = img_cube_stack.sum(axis=(1, 2))
        cube_totals = cube_totals.compute()
        self.cube_means = cube_totals / 2080.0 ** 2

        self.absolute_ref_img = self.cube_means.argmax()
        self.mean_ref_img = self.cube_means.max()

        logger.info("Max mean: {0:.5f}".format(self.mean_ref_img))

        # If the default displacements are too large
        # to for overlaps, stick with the default scale
        self.offset_mode = "sampled"

        if delta_x / Settings.mosaic_scale > 1950.0:
            logger.info(
                "Displacement in X too large: {0:.1f}".format(
                    delta_x / Settings.mosaic_scale
                )
            )
            self.offset_mode = "default"
        if delta_y / Settings.mosaic_scale > 1950.0:
            logger.info(
                "Displacement in Y too large: {0:.1f}".format(
                    delta_y / Settings.mosaic_scale
                )
            )
            self.offset_mode = "default"

        if self.mean_ref_img < 0.05:
            logger.info("Avg. flux too low: {0:.3f}<0.05".format(
                self.mean_ref_img))
            self.offset_mode = "default"

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

                if self.offset_mode == "sampled":
                    im1 = apply_geometric_transform(
                        im_ref.data / Settings.norm_val, dark, flat, Settings.cof_dist
                    )
                    im2 = apply_geometric_transform(
                        im_obj.data / Settings.norm_val, dark, flat, Settings.cof_dist
                    )
                    res = delayed(find_overlap_conf)(
                        im1, im2, dist_conf, dist_conf, desp
                    )
                    results.append(_sink(ref_img, obj_img, res, sign))

                else:
                    logger.debug(
                        "Initial offsets i: %d j: %d dx: %d dy: %d",
                        ref_img,
                        obj_img,
                        desp[0],
                        desp[1],
                    )
                    results.append([ref_img, obj_img, desp[0], desp[1], sign])

        if self.offset_mode == "sampled":
            futures = client.compute(results)
            offsets = []
            for fut in as_completed(futures):
                i, j, res, sign = fut.result()
                dx, dy, mi, avf = res.x, res.y, res.mi, res.avg_flux

                offsets.append([i, j, res, sign])

                logger.debug(
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
        else:
            offsets = []
            for t in results:
                i, j, dx, dy, sign = t

                offsets.append([i, j, [dx, dy, 0.0, 0.0], sign])

                logger.debug(
                    "Section %s offsets i: %d j: %d dx: %d dy: %d mi: %f avg_f: %f sign: %d",
                    self._section.name,
                    i,
                    j,
                    dx,
                    dy,
                    0.0,
                    0.0,
                    sign,
                )
        self._offsets = offsets

    @property
    def scale(self) -> Tuple[float]:
        """Scale of the detector in pixels per micron"""
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
                if (abs(p[0]) < 5000) & (abs(p[0] > 500) & (abs(a) > 0.5))
            ]
        )
        y_scale = np.median(
            [
                m[1] / p[1]
                for m, p, a in zip(
                    self._mu.values(), self._px.values(), self._avg.values()
                )
                if (abs(p[1]) < 5000) & (abs(p[1] > 500) & (abs(a) > 0.5))
            ]
        )

        if np.isnan(x_scale):
            x_scale = Settings.mosaic_scale
        if np.isnan(y_scale):
            y_scale = Settings.mosaic_scale

        logger.debug("Scale %f %f", x_scale, y_scale)

        self._x_scale, self._y_scale = x_scale, y_scale
        return (x_scale, y_scale)

    def compute_pairs(self):
        """Return the pairs of mags of the detector"""
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
            & (avg_flux > np.max([np.median(avg_flux), 0.05]))
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
            & (avg_flux > np.max([np.median(avg_flux), 0.05]))
        )[0]
        default_y = [
            (np.abs(px_y[ii]) * avg_flux[ii] ** 2).sum() /
            (avg_flux[ii] ** 2).sum(),
            (np.abs(px_x[ii]) * avg_flux[ii] ** 2).sum() /
            (avg_flux[ii] ** 2).sum(),
        ]
        dev_y = np.sqrt(mad(px_x[ii]) ** 2 + mad(px_y[ii]) ** 2)

        logger.debug(
            "Default displacement X:  dx:%f , dy:%f, std:%f",
            default_x[0],
            default_x[1],
            dev_x,
        )
        logger.debug(
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

    def compute_abspos_ref(  # noqa: C901
        self,
        dx0,
        dy0,
        default_x,
        dev_x,
        default_y,
        dev_y,
        scale_x,
        scale_y,
        ref_img_assemble,
        error_long_threshold,
        error_short_threshold,
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
                    if (f"{ref_img}:{this_img}" in self._px.keys()) is False:
                        continue

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
                            err_long_px = np.sqrt(
                                (
                                    self._px[f"{ref_img}:{this_img}"][0]
                                    - self._mu[f"{ref_img}:{this_img}"][0] / scale_x
                                )
                                ** 2
                            )
                            err_short_px = np.sqrt(
                                (
                                    self._px[f"{ref_img}:{this_img}"][1]
                                    - self._mu[f"{ref_img}:{this_img}"][1] / scale_y
                                )
                                ** 2
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
                                )
                                ** 2
                            )
                    # 20200710 added check for mi, avg because there are some slices
                    # that despite having overlaps, there seems to be no info
                    # and lead to nans
                    if (
                        (err_long_px < error_long_threshold)
                        & (err_short_px < error_short_threshold)
                        & (self._mi[f"{ref_img}:{this_img}"] > 0)
                        & (self._avg[f"{ref_img}:{this_img}"] > 0.05)
                    ):
                        # Dimensions in the mosaic and images have opposite directions
                        temp_pos[0].append(
                            abs_pos[ref_img, 0] +
                            self._px[f"{ref_img}:{this_img}"][0]
                        )
                        temp_pos[1].append(
                            abs_pos[ref_img, 1] +
                            self._px[f"{ref_img}:{this_img}"][1]
                        )
                        # weights
                        weight.append(self._avg[f"{ref_img}:{this_img}"] ** 2)
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
                    np.array(temp_qual) * weight
                ).sum() / weight.sum()

        return abs_pos, pos_quality

    def compute_abspos(self):  # noqa: C901
        """Compute absolute positions of images in the field of view."""
        logger.info("Processing section %s", self._section.name)

        # img_cube = self.get_img_section(0, Settings.channel_to_use)
        # img_cube_stack = da.stack(img_cube)
        # # ref image is the one with the max integrated flux
        # cube_totals = img_cube_stack.sum(axis=(1, 2))
        # cube_totals = cube_totals.compute()
        # cube_means = cube_totals / 2080.0 ** 2

        # absolute_ref_img = cube_means.argmax()
        absolute_ref_img = self.absolute_ref_img

        if self.offset_mode == "default":
            dx = np.array(self["XPos"])
            dy = np.array(self["YPos"])

            abs_pos = []
            abs_err = []

            for i in range(len(dx)):
                abs_pos.append(
                    [
                        (dx[i] - dx[absolute_ref_img]) / Settings.mosaic_scale,
                        (dy[i] - dy[absolute_ref_img]) / Settings.mosaic_scale,
                    ]
                )
                abs_err.append([15.0, 15.0])

            logger.info("Displacements too large, resorting to default grid")

            abs_pos = np.array(abs_pos)
            abs_err = np.array(abs_err)

            self._section.attrs["abs_pos"] = abs_pos.tolist()
            self._section.attrs["abs_err"] = abs_err.tolist()
            self._section.attrs["default_displacements"] = [
                {"default_x": 0.0, "dev_x": 0.0, "default_y": 0.0, "dev_y": 0.0}
            ]
            self._x_scale = Settings.mosaic_scale
            self._y_scale = Settings.mosaic_scale

            return abs_pos, abs_err

        self.compute_pairs()
        scale_x, scale_y = self.scale
        dx0, dy0, delta_x, delta_y = self.find_grid()
        default_x, dev_x, default_y, dev_y = self.get_default_displacement()

        # get error threshold scomparing scaled micron displacements
        # with measurements
        px_x_temp = []
        px_y_temp = []
        mu_x_temp = []
        mu_y_temp = []
        avg_f_temp = []
        for this_key in self._px.keys():
            px_x_temp.append(self._px[this_key][0])
            px_y_temp.append(self._px[this_key][1])
            mu_x_temp.append(self._mu[this_key][0] / scale_x)
            mu_y_temp.append(self._mu[this_key][1] / scale_y)
            avg_f_temp.append(self._avg[this_key])
        px_x_temp = np.array(px_x_temp)
        px_y_temp = np.array(px_y_temp)
        mu_x_temp = np.array(mu_x_temp)
        mu_y_temp = np.array(mu_y_temp)
        avg_f_temp = np.array(avg_f_temp)

        ind_temp = np.where(
            (np.abs(px_x_temp) > np.mean(np.abs(px_x_temp))) & (avg_f_temp > 0.05)
        )[0]
        if len(ind_temp) > 3:
            elongx = 1.48 * mad(np.sqrt((px_x_temp - mu_x_temp)[ind_temp] ** 2))
            eshorty = 1.48 * \
                mad(np.sqrt((px_y_temp - mu_y_temp)[ind_temp] ** 2))
        else:
            elongx = 10
            eshorty = 10

        ind_temp = np.where(
            (np.abs(px_y_temp) > np.mean(np.abs(px_y_temp))) & (avg_f_temp > 0.05)
        )[0]
        if len(ind_temp) > 3:
            elongy = 1.48 * mad(np.sqrt((px_y_temp - mu_y_temp)[ind_temp] ** 2))
            eshortx = 1.48 * \
                mad(np.sqrt((px_x_temp - mu_x_temp)[ind_temp] ** 2))
        else:
            elongy = 10.0
            eshortx = 10.0

        error_long_threshold = 3.0 * np.max([elongx, elongy]).clip(2, 30)
        error_short_threshold = 3.0 * np.max([eshortx, eshorty]).clip(2, 30)
        logger.debug(
            "Scaling error threshold: l:{0:3f} s:{1:.3f}".format(
                error_long_threshold, error_short_threshold
            )
        )

        accumulated_pos = []
        accumulated_qual = []
        for this_ref in np.where(self.cube_means > np.median(self.cube_means))[0]:

            temp = self.compute_abspos_ref(
                dx0,
                dy0,
                default_x,
                dev_x,
                default_y,
                dev_y,
                scale_x,
                scale_y,
                this_ref,
                error_long_threshold,
                error_short_threshold,
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
        abs_err = np.std(accumulated_pos, 0)

        self._section.attrs["abs_pos"] = abs_pos.tolist()
        self._section.attrs["abs_err"] = abs_err.tolist()

        return abs_pos, abs_err

    def _create_temporary_mosaic(self, conf, abs_pos, abs_err, output):
        logger.info("Creating temporary mosaic")
        z = zarr.open(f"{output}/{self.name}.zarr", mode="w")

        if self.cal_type == 'sample':
            cal_xr = xr.open_zarr(self.cal_zarr)
            flat = da.array(cal_xr['FLATS'].values)
            dark = da.array(cal_xr['DARKS'].values)
        else:
            flat = read_calib(Settings.flat_file)
            dark = read_calib(Settings.dark_file) / Settings.norm_val

        cof_dist = Settings.cof_dist
        y_delta, x_delta = np.min(abs_pos, axis=0)

        for sl in range(self.slices):
            results = []
            for ch in range(self.channels):
                im_t = self.get_img_section(sl, ch)
                g = z.create_group(f"/mosaic/{self.name}/z={sl}/channel={ch}")

                im_dis = [
                    apply_geometric_transform(
                        im.data /
                        Settings.norm_val, dark[ch], flat[ch], cof_dist
                    )
                    for im in im_t
                ]

                for imgtype in ["raw", "pos_err", "overlap"]:
                    nimg = _get_image(
                        g, imgtype, self.stage_size, dtype="float32")

                    for i in range(len(im_dis)):
                        y0 = int(abs_pos[i, 0] - y_delta)
                        x0 = int(abs_pos[i, 1] - x_delta)
                        res = _mosaic(
                            im_dis[i],
                            ch,
                            conf,
                            (y0, x0),
                            abs_err,
                            imgtype,
                            nimg,
                        )
                        results.append(res)
            dask.compute(results)
            logger.debug("Mosaic %s Slice %d done", self.name, sl)
        return z

    def _compute_final_mosaic(self, z):
        logger.info("Creating final mosaic")
        mos_overlap = []
        mos_raw = []
        mos_err = []

        for name, offset in z[f"mosaic/{self.name}"].groups():
            raw = da.stack(
                [
                    da.from_zarr(ch["raw"]) / da.from_zarr(ch["overlap"])
                    for name, ch in offset.groups()
                ]
            )

            raw = (raw - Settings.bzero) / Settings.bscale
            overlap = (
                da.stack([da.from_zarr(ch["overlap"])
                          for name, ch in offset.groups()])
                * 100
            )
            err = (
                da.stack([da.from_zarr(ch["pos_err"])
                          for name, ch in offset.groups()])
                * 100
            )
            mos_overlap.append(overlap)
            mos_err.append(err)
            mos_raw.append(raw)
        mos_raw = da.stack(mos_raw)
        mos_overlap = da.stack(mos_overlap)
        mos_err = da.stack(mos_err)
        mos = da.stack([mos_raw, mos_overlap, mos_err]).rechunk(
            (1, 1, 10, CHUNK_SIZE, CHUNK_SIZE))
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

        metadata = self._get_metadata()
        raw.attrs.update(metadata)

        return raw

    def _get_metadata(self):
        metadata = {
            "default_displacements": self._section.attrs["default_displacements"],
            "abs_pos": self._section.attrs["abs_pos"],
            "abs_err": self._section.attrs["abs_err"],
            "offsets": self._offsets,
            "scale": [self._x_scale, self._y_scale],
            "offset_mode": self.offset_mode,
            "raw_meta": [self._section.attrs.get("raw_meta")],
        }
        return metadata

    def _stage_size(self, abs_pos):
        # 20200717 increased security padding to 1.5xCHUNK_SIZE
        shape_0 = (
            int(np.array(abs_pos[:, 0]).max() - np.array(abs_pos[:, 0]).min())
            + 1.5 * self.shape[0]
        )
        shape_1 = (
            int(np.array(abs_pos[:, 1]).max() - np.array(abs_pos[:, 1]).min())
            + 1.5 * self.shape[1]
        )

        # this has to be a multiple of the chunk size
        if shape_0 % CHUNK_SIZE != 0:
            n_imgs = int(shape_0 / CHUNK_SIZE) + 1
            shape_0 = int(CHUNK_SIZE * n_imgs)
        if shape_1 % CHUNK_SIZE != 0:
            n_imgs = int(shape_1 / CHUNK_SIZE) + 1
            shape_1 = int(CHUNK_SIZE * n_imgs)

        return (shape_0, shape_1)

    def stitch(self, output: Path):
        """Stitch and save all images"""

        ds = xr.open_zarr(output / "mos.zarr")
        if self.name in ds:
            logger.info("Section %s already done. Skipping.", self.name)
            return

        abs_pos, abs_err = self.compute_abspos()
        conf = self.get_distconf()

        if self.stage_size is None:
            self.stage_size = self._stage_size(abs_pos)

        z = self._create_temporary_mosaic(conf, abs_pos, abs_err, output)

        arr = self._compute_final_mosaic(z)
        ds = xr.Dataset({self._section.name: arr})
        ds.to_zarr(output / "mos.zarr", mode="a")

        # clean temporary mosaic
        shutil.rmtree(f"{output}/{self.name}.zarr", ignore_errors=True)
        logger.info("Mosaic saved %s", output / "mos.zarr")


class STPTMosaic:
    """STPT Mosaic

    Parameters
    ----------
    arr
        Zarr array containing the full STPT sample.
    """

    def __init__(self, filename: Path):

        # _ds points to the raw zarr
        self._ds = xr.open_zarr(f"{filename}")

        # this is to carry over the mosaic size between
        # sections
        self.stage_size = None

    def initialize_storage(self, output: Path):
        logger.info("Preparing mosaic output")

        ds = xr.Dataset()
        ds.attrs = self._ds.attrs
        ds.attrs["raw_meta"] = [ds.attrs["raw_meta"]]
        ds.to_zarr(output / "mos.zarr", mode="w")

    def sections(self, section_list=None):
        """Sections generator"

        Parameters
        ----------
        section_list
            List of strings with section labels, if empty
            returns all the sections
        """
        labels = section_list or self._ds

        for section in labels:
            yield Section(self._ds[section], self.stage_size)

    def downsample(self, output: Path):
        """Downsample mosaic.

        Parameters
        ----------
        output
            Location of output directory
        """
        logger.info("Downsampling mosaics")
        store_name = output / "mos.zarr"
        store = zarr.DirectoryStore(store_name)
        up = ""
        for factor in Settings.scales:
            logger.debug("Downsampling factor %d", factor)
            ds = xr.open_zarr(f"{store_name}", group=up)
            nds = xr.Dataset()
            down = f"l.{factor}"
            nds.to_zarr(store, mode="w", group=down)

            slices = list(ds)
            for s in slices:
                nds = xr.Dataset()
                logger.debug("Downsampling mos%d [%s]", factor, s)
                narr = ops.downsample(ds[s])
                nds[s] = narr
                nds.to_zarr(store, mode="a", group=down)
            logger.info("Downsampled mosaic saved %s:%s", store_name, down)
            up = down
        arr = zarr.open(f"{store_name}", mode="r+")
        arr.attrs["multiscale"] = {
            "datasets": [
                {"path": "", "level": 1},
                {"path": "l.2", "level": 2},
                {"path": "l.4", "level": 4},
                {"path": "l.8", "level": 8},
                {"path": "l.16", "level": 16},
                {"path": "l.32", "level": 32},
            ],
            "metadata": {"method": "cv2.pyrDown", "version": cv2.__version__},
        }
        arr.attrs["bscale"] = Settings.bscale
        arr.attrs["bzero"] = Settings.bzero
