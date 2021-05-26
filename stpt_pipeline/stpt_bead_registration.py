from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from owl_dev.logging import logger
from scipy.stats import median_absolute_deviation as mad


class bead_collection:
    """
    This holds all the beads from all the slices of a mosaic,
    and keeps track of which ones are the same bead at different
    Z and which are new beads. Average coordinates per bead can be
    updated.
    """

    def __init__(self):
        self.id = []
        self.x = np.array([])
        self.y = np.array([])
        self.dx = np.array([])
        self.dy = np.array([])
        self.n = np.array([])

        self.id_list = []
        self.x_list = []
        self.y_list = []
        self.z_list = []
        self.x_list_raw = []
        self.y_list_raw = []
        self.r_mask = []

        self.critical_radius = 20.0

    def add_bead(self, x_cor, y_cor, x_raw, y_raw, z, r_m, b_id):
        """
            Adds a bead to the collection, and matches with
            average coordinates

            Parameters
            ----------
            x_cor, y_cor: coordinates corrected from slice displacement,
                if available.
            x_raw, y_raw: uncorrected coordinates.
            z: optical slice
            r_m: mask radius
            b_id: unique bead id
        """

        if len(self.x) == 0:
            r = np.array([self.critical_radius * 10.])
        else:
            r = np.sqrt((x_cor - self.x)**2 + (y_cor - self.y)**2)

        if r.min() < self.critical_radius:
            self.x_list[r.argmin()].append(x_cor)
            self.y_list[r.argmin()].append(y_cor)
            self.z_list[r.argmin()].append(z)
            self.x_list_raw[r.argmin()].append(x_raw)
            self.y_list_raw[r.argmin()].append(y_raw)
            self.id_list[r.argmin()].append(b_id)
            self.r_mask[r.argmin()].append(r_m)
            self.n[r.argmin()] += 1
        else:
            self.id.append(b_id)
            self.x = np.append(self.x, x_cor)
            self.y = np.append(self.y, y_cor)
            self.dx = np.append(self.dx, 0.0)
            self.dy = np.append(self.dy, 0.0)
            self.n = np.append(self.n, 1.0)

            self.x_list.append([x_cor])
            self.y_list.append([y_cor])
            self.z_list.append([z])
            self.x_list_raw.append([x_raw])
            self.y_list_raw.append([y_raw])
            self.id_list.append([b_id])
            self.r_mask.append([r_m])

    def update_coords(self):
        """
            Recalculates the average coordinates for the beads
        """
        for i in range(len(self.id)):
            if self.n[i] > 1:
                self.x[i] = np.mean(np.array(self.x_list[i]))
                self.y[i] = np.mean(np.array(self.y_list[i]))
                self.dx[i] = np.std(np.array(self.x_list[i]))
                self.dy[i] = np.std(np.array(self.y_list[i]))


def _get_beads(slice_obj, z_val):

    if "bead_z" in list(slice_obj.keys()):

        z = np.array(slice_obj["bead_z"][:])
        i_t = np.where(z == z_val)[0]

        xc = np.array(slice_obj["bead_x"][:])[i_t]
        yc = np.array(slice_obj["bead_y"][:])[i_t]
        err = np.array(slice_obj["bead_centre_err"][:])[i_t]
        rad = np.array(slice_obj["bead_rad"][:])[i_t]
        ind = np.array(slice_obj["bead_id"][:])[i_t]

        return ind, xc, yc, rad, err
    else:
        return [], [], [], [], []


def _match_cats(xr, yr, er, xt, yt, et, errors=False):
    if len(xr) > len(xt):
        xs = np.array(xt)
        ys = np.array(yt)
        es = np.array(et)
        xl = np.array(xr)
        yl = np.array(yr)
        el = np.array(er)
    else:
        xs = np.array(xr)
        ys = np.array(yr)
        es = np.array(er)
        xl = np.array(xt)
        yl = np.array(yt)
        el = np.array(et)

    # simple closest match iterative
    dx = 0.0
    dy = 0.0
    for _i in range(3):
        i_sl = []
        i_ls = []
        r_sl = []
        for i in range(len(xs)):
            r = np.sqrt((xl - xs[i] - dx) ** 2 + (yl - ys[i] - dy) ** 2)
            r_sl.append(r.min())
            i_sl.append(i)
            i_ls.append(r.argmin())
        r_sl = np.array(r_sl)
        # combined error, minimum error should be 2
        e_c = np.sqrt(el[i_ls] ** 2 + es[i_sl] ** 2).clip(2)

        # only 3sigma matches
        ii = np.where(r_sl - np.median(r_sl) < 3.0 *
                      1.48 * mad(r_sl, axis=None))[0]

        dx = np.sum((xl[i_ls] - xs[i_sl])[ii] / e_c[ii] ** 2) / np.sum(
            1.0 / e_c[ii] ** 2
        )
        dy = np.sum((yl[i_ls] - ys[i_sl])[ii] / e_c[ii] ** 2) / np.sum(
            1.0 / e_c[ii] ** 2
        )

    if errors:
        ex = np.sqrt(1.0 / np.sum(1.0 / e_c ** 2))
        if len(xr) > len(xt):
            return dx, dy, ex, i_ls, i_sl
        else:
            return -dx, -dy, ex, i_sl, i_ls

    if len(xr) > len(xt):
        return dx, dy, i_ls, i_sl
    else:
        return -dx, -dy, i_sl, i_ls


def _get_good_beads(mos_full, phys, opt, good_beads):
    _ind, _x, _y, _, _e = _get_beads(mos_full[phys].attrs, opt)

    this_slice = phys + "_Z{0:03d}".format(opt)

    # we check which of these are repeated beads
    ind_t = np.array([])
    x_t = np.array([])
    y_t = np.array([])
    e_t = np.array([])
    _j = 0
    for this_id in _ind:
        temp = this_slice + ":{0:05d}".format(int(this_id))
        if temp in good_beads:
            ind_t = np.append(ind_t, this_id)
            x_t = np.append(x_t, _x[_j])
            y_t = np.append(y_t, _y[_j])
            e_t = np.append(e_t, _e[_j])
        _j += 1

    return ind_t, x_t, y_t, e_t


def register_slices(mos_zarr: Path):  # noqa: C901
    """
    Uses all the detected beads in each slice to cross-match
    and calculates an average displacement so that all
    slices are matched to the first one
    """
    # this is to store the beads later
    zarr_store = zarr.open(f"{mos_zarr}", mode="a")

    mos_full = xr.open_zarr(f"{mos_zarr}", group="")
    _slices = list(mos_full)

    # putting all slices on a single list
    optical_slices = []
    physical_slices = []
    for this_slice in _slices:
        for this_optical in mos_full[this_slice].z.values:
            physical_slices.append(this_slice)
            optical_slices.append(this_optical)

    # first pass crossmatch, taking into account all beads
    dx = [0.0]  # these store the slice to slice offset
    dy = [0.0]
    logger.info("1st pass slice shifts")
    for i in range(1, len(physical_slices)):

        # We compare each slice (_t) with the previous one (_r)
        _, x_t, y_t, _, e_t = _get_beads(
            mos_full[physical_slices[i]].attrs, optical_slices[i]
        )

        _, x_r, y_r, _, e_r = _get_beads(
            mos_full[physical_slices[i - 1]].attrs, optical_slices[i - 1]
        )

        if (len(x_t) > 0) & (len(x_r) > 0):

            dxt, dyt, i_rt, i_tr = _match_cats(x_r, y_r, e_r, x_t, y_t, e_t)
            dx.append(dxt)
            dy.append(dyt)

            dr = np.sqrt(
                (y_r[i_rt] - y_t[i_tr] - dyt) ** 2 +
                (x_r[i_rt] - x_t[i_tr] - dxt) ** 2
            )
            dr0 = np.sqrt((y_r[i_rt] - y_t[i_tr]) ** 2 +
                          (x_r[i_rt] - x_t[i_tr]) ** 2)

        else:
            dx.append(0.0)
            dy.append(0.0)
            i_tr = []
            dxt = dyt = dr = dr0 = 0

        logger.info(
            physical_slices[i - 1]
            + "_Z{0:03d}:".format(optical_slices[i - 1])
            + physical_slices[i]
            + "_Z{0:03d}:".format(optical_slices[i])
            + " {0:d} ".format(len(i_tr))
            + "{0:.1f} {1:.1f} ".format(dxt, dyt)
            + "{0:.1f} {1:.1f} ".format(np.median(dr), np.median(dr0))
        )

    # now that we know the slice to slice offset, we construct the catalogue
    # of all the beads
    bb = bead_collection()

    for i in range(len(physical_slices)):
        ref_slice = physical_slices[i] + "_Z{0:03d}".format(optical_slices[i])

        # because dx,dy are slice to slice, the total displacement
        # is the sum of all the previous
        dx_t = np.sum(dx[0: i + 1])
        dy_t = np.sum(dy[0: i + 1])

        ind_r, x_r, y_r, _, _ = _get_beads(
            mos_full[physical_slices[i]].attrs, optical_slices[i]
        )
        id_str = []
        for this_id in ind_r:
            id_str.append(ref_slice + ":{0:05d}".format(int(this_id)))
        id_str = np.array(id_str)

        for j in range(len(x_r)):
            bb.add_bead(x_r[j] + dx_t, y_r[j] + dy_t, x_r[j], y_r[j], id_str[j])

        # at each slice we check matching beads and update avg coords
        bb.update_coords()

    # Now the we know wich objects can be seen in more than one slice,
    # we re-compute the offsets but only using the beads that appear
    # in at least all the optical slices and 2 physical

    min_num_dets = np.max(np.array(optical_slices)) + 2

    good_beads = []
    for i in range(len(bb.x)):
        if bb.n[i] >= min_num_dets:
            good_beads.extend(bb.id_list[i])
    # cleaning out repeated ids
    good_beads = list(set(good_beads))

    # here we store the new displacements, and a measurement of error
    dx2 = [0.0]
    dy2 = [0.0]
    dd2 = [0.0]
    logger.info("2nd pass slice shifts")
    for i in range(1, len(physical_slices)):
        this_slice = physical_slices[i] + "_Z{0:03d}".format(optical_slices[i])
        ref_slice = physical_slices[i - 1] + \
            "_Z{0:03d}".format(optical_slices[i - 1])

        # only beads that have high reps
        _, x_t, y_t, e_t = _get_good_beads(
            mos_full, physical_slices[i], optical_slices[i], good_beads
        )

        if len(x_t) > 0:
            _, x_r, y_r, e_r = _get_good_beads(
                mos_full, physical_slices[i -
                                          1], optical_slices[i - 1], good_beads
            )

            dxt, dyt, edt, i_rt, i_tr = _match_cats(
                x_r, y_r, e_r, x_t, y_t, e_t, errors=True
            )

            dx2.append(dxt)
            dy2.append(dyt)
            dd2.append(edt)

            dr = np.sqrt(
                (y_r[i_rt] - y_t[i_tr] - dyt) ** 2 +
                (x_r[i_rt] - x_t[i_tr] - dxt) ** 2
            )
            dr0 = np.sqrt((y_r[i_rt] - y_t[i_tr]) ** 2 +
                          (x_r[i_rt] - x_t[i_tr]) ** 2)
            logger.info(
                ref_slice
                + ":"
                + this_slice
                + " {0:d} ".format(len(i_tr))
                + "{0:.1f} {1:.1f} ".format(dxt, dyt)
                + "{0:.1f} {1:.1f} ".format(np.median(dr), np.median(dr0))
            )
        else:
            # NO beads in this slice
            dx2.append(0.0)
            dy2.append(0.0)
            dd2.append(100.0)

            logger.info(
                ref_slice
                + ":"
                + this_slice
                + " {0:d} ".format(len(i_tr))
                + "{0:.1f} {1:.1f} ".format(0.0, 0.0)
                + "{0:.1f} {1:.1f} ".format(0.0, 0.0)
            )

    # Now we store all the displacements as attrs
    cube_reg = {
        "abs_dx": [],
        "abs_dy": [],
        "abs_err": [],
        "rel_dx": [],
        "rel_dy": [],
        "rel_err": [],
    }
    for i in range(len(physical_slices)):
        # because dx,dy are slice to slice, the total displacement
        # is the sum of all the previous
        dx_t = np.sum(dx2[0: i + 1])
        dy_t = np.sum(dy2[0: i + 1])
        de_t = np.sqrt(np.sum(np.array(dd2[0: i + 1]) ** 2))

        cube_reg["abs_dx"].append(dx_t)
        cube_reg["abs_dy"].append(dy_t)
        cube_reg["abs_err"].append(de_t)

        cube_reg["rel_dx"].append(dx2[i])
        cube_reg["rel_dy"].append(dy2[i])
        cube_reg["rel_err"].append(dd2[i])

    zarr_store.attrs["cube_reg"] = cube_reg
