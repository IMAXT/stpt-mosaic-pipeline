from pathlib import Path

import numpy as np
import xarray as xr
import zarr
from owl_dev.logging import logger
from scipy.sparse import coo_matrix, dia_matrix
from scipy.stats import median_abs_deviation as mad
import scipy.optimize as op


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

    z = np.array(slice_obj['bead_z'][:])
    i_t = np.where(z == z_val)

    xc = np.array(slice_obj['bead_x'][:])[i_t]
    yc = np.array(slice_obj['bead_y'][:])[i_t]
    r_img = np.array(slice_obj['mask_rad'][:])[i_t]
    ind = np.array(slice_obj['bead_id'][:])[i_t]

    return ind, xc, yc, r_img


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
            r = np.sqrt((xl - xs[i] - dx)**2 + (yl - ys[i] - dy)**2)
            r_sl.append(r.min())
            i_sl.append(i)
            i_ls.append(r.argmin())
        r_sl = np.array(r_sl)
        # combined error
        e_c = np.sqrt(el[i_ls]**2 + es[i_sl]**2)

        # only 3sigma matches
        ii = np.where(r_sl - np.median(r_sl) < 3. *
                      1.48 * mad(r_sl, axis=None))[0]
        dx = np.median((xl[i_ls] - xs[i_sl])[ii])
        dy = np.median((yl[i_ls] - ys[i_sl])[ii])

    if errors:
        ex = np.sqrt(
            1. / np.sum(1. / e_c**2)
        )
        if len(xr) > len(xt):
            return dx, dy, ex, i_ls, i_sl
        else:
            return -dx, -dy, ex, i_sl, i_ls
    if len(xr) > len(xt):
        return dx, dy, i_ls, i_sl
    else:
        return -dx, -dy, i_sl, i_ls


def _prefilter_beads(xb, yb, x_cat, y_cat):
    """
        Before adding beads to a bead collection, checks if
        any pair of beads would match to the same object from
        the collection, and selects the closest.
    """

    ib = []
    rb = []
    for i in range(len(xb)):
        r = np.sqrt((xb[i] - x_cat)**2 + (yb[i] - y_cat)**2)
        ib.append(np.argmin(r))
        rb.append(np.min(r))
    ib = np.array(ib)
    rb = np.array(rb)

    # checking repetitions
    good_beads = []
    done_beads = []
    for i in range(len(xb)):
        if i in done_beads:
            continue
        reps = np.where(ib == ib[i])[0]
        if len(reps) > 1:
            done_beads.extend(list(reps))
            _t = np.argmin(rb[reps])
            good_beads.append(_t)
        else:
            good_beads.append(i)

    return good_beads


def _shift_func(x, A, At, d, mat_var):
    """
        Calculates the difference between the
        bead displacements according to shift vectors
        and the measured values
    """
    #
    temp_dif = d - A.dot(x)
    temp_prod = mat_var.dot(temp_dif)
    #
    f = np.dot(temp_dif.T, temp_prod)
    #
    grad_f = -2 * At.dot(temp_prod)
    #
    return f / float(len(d)), grad_f / float(len(d))


def _build_collection(mos_zarr, dx, dy, physical, optical):
    """
        Builds a bead collection from the beads on
        a zarr metadata and the calculated displacements
    """

    bb = bead_collection()

    for i in range(len(physical)):
        ref_slice = physical[i] + "_Z{0:03d}".format(optical[i])

        dx_t = dx[i]
        dy_t = dy[i]

        ind_r, x_r, y_r, r_r = _get_beads(
            mos_zarr[physical[i]].attrs, optical[i]
        )
        id_str = []
        for this_id in ind_r:
            id_str.append(ref_slice + ":{0:05d}".format(int(this_id)))
        id_str = np.array(id_str)

        if i > 0:
            g = _prefilter_beads(x_r + dx_t, y_r + dy_t, bb.x, bb.y)
        else:
            g = range(len(x_r))

        for j in g:
            bb.add_bead(
                x_r[j] + dx_t,
                y_r[j] + dy_t,
                x_r[j], y_r[j],
                optical[i],
                r_r[j],
                id_str[j]
            )

        # at each slice we check matching beads and update avg coords
        bb.update_coords()

    return bb


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

    # first pass crossmatch, taking into account all beads, these
    # estimations are good and used later to refine, but also we can
    # check if there are enough beads between slices

    dx = [0.0]  # these store the slice to slice offset
    dy = [0.0]
    logger.info("1st pass slice shifts")
    for i in range(1, len(physical_slices)):

        # We compare each slice (_t) with the previous one (_r)
        _, x_t, y_t, r_t = _get_beads(
            mos_full[physical_slices[i]].attrs, optical_slices[i]
        )

        _, x_r, y_r, r_r = _get_beads(
            mos_full[physical_slices[i - 1]].attrs, optical_slices[i - 1]
        )

        # estimate of the error based on the radius
        e_t = 0.01 * r_t
        e_r = 0.01 * r_r

        if (len(x_t) > 0) and (len(x_r) > 0):

            dxt, dyt, i_rt, i_tr = _match_cats(x_r, y_r, e_r, x_t, y_t, e_t)

            if len(i_rt) < 3:
                logger.info(
                    '*** WARNING: Not enough beads in common between ' +
                    physical_slices[i - 1] +
                    'Z{0:03d}'.format(optical_slices[i - 1]) +
                    physical_slices[i] +
                    ' and Z{0:03d}'.format(optical_slices[i])
                )
                # set desp to 0
                dxt = dyt = dr = dr0 = 0

            else:
                dr = np.sqrt(
                    (y_r[i_rt] - y_t[i_tr] - dyt) ** 2 +
                    (x_r[i_rt] - x_t[i_tr] - dxt) ** 2
                )
                dr0 = np.sqrt((y_r[i_rt] - y_t[i_tr]) ** 2 +
                              (x_r[i_rt] - x_t[i_tr]) ** 2)

        else:
            logger.info(
                '*** WARNING: Not enough beads in ' +
                physical_slices[i - 1] +
                'Z{0:03d}'.fomat(optical_slices[i - 1]) +
                physical_slices[i] +
                ' or Z{0:03d}'.fomat(optical_slices[i])
            )
            i_tr = []
            dxt = dyt = dr = dr0 = 0

        dx.append(dxt)
        dy.append(dyt)
        logger.info(
            '\t'
            + physical_slices[i - 1]
            + "_Z{0:03d}:".format(optical_slices[i - 1])
            + physical_slices[i]
            + "_Z{0:03d}:".format(optical_slices[i])
            + " {0:d} ".format(len(i_tr))
            + "{0:.1f} {1:.1f} ".format(dxt, dyt)
            + "{0:.1f} {1:.1f} ".format(np.median(dr), np.median(dr0))
        )

    # now that we know the slice to slice offset, we construct the catalogue
    # of all the beads

    # because dx,dy are slice to slice, the total displacement
    # is the sum of all the previous
    abs_dx = np.array([np.sum(dx[0:t + 1]) for t in range(len(dx))])
    abs_dy = np.array([np.sum(dy[0:t + 1]) for t in range(len(dy))])

    bb = _build_collection(
        mos_full, abs_dx, abs_dy,
        physical_slices, optical_slices
    )

    # For the 2nd pass, we calculate the shifts taking all beads into
    # account.

    # First we need to build the matrixes, to speed things up we use
    # sparse matrixes, and so the first step is just to generate vectors
    # with the coordinates and the values to fill the matrices

    logger.info("2nd pass slice shifts")
    n_op = np.max(mos_full[physical_slices[0]].z.values) + 1
    xi = []
    yi = []
    c = []
    dx = []
    dy = []
    ex = []
    row_counter = 0
    for i in range(len(bb.id)):
        if bb.n[i] < 3:
            continue

        _x = bb.x_list_raw[i]
        _y = bb.y_list_raw[i]
        _id = bb.id_list[i]
        # the z value takes into account physical and optical
        _z = [
            (float(t[1:4]) - 1) * n_op + float(t[6:9]) for t in _id
        ]
        _e = 0.01 * np.array(bb.r_mask[i])
        for j in range(1, len(_z)):
            xi.append(row_counter)
            yi.append(_z[j])
            c.append(1.0)

            xi.append(row_counter)
            yi.append(_z[j - 1])
            c.append(-1.0)

            dx.append(_x[j] - _x[j - 1])
            dy.append(_y[j] - _y[j - 1])
            ex.append(np.sqrt(_e[j]**2 + _e[j - 1]**2))

            row_counter += 1
        # adding last minus first
        xi.append(row_counter)
        yi.append(_z[0])
        c.append(1.0)

        xi.append(row_counter)
        yi.append(_z[-1])
        c.append(-1.0)

        dx.append(_x[0] - _x[-1])
        dy.append(_y[0] - _y[-1])
        ex.append(10.)

        row_counter += 1

    # building matrix/vectos
    mat = coo_matrix((c, (xi, yi)), (len(dx), int(np.max(yi)) + 1))
    dx = np.array(dx)
    dy = np.array(dy)
    ex = np.array(ex)

    # we minimise recursively trimming outliers
    for i in range(10):

        if i == 0:
            x0 = np.zeros(int(np.max(yi)) + 1)
            y0 = np.zeros(int(np.max(yi)) + 1)

        dr = (dx - mat.dot(x0))**2 + (dy - mat.dot(y0))**2
        std = 10.
        ii = np.where(dr < (13 - i) * std)[0]

        mat_0 = mat.tocsr()[ii]
        mat_0_t = mat_0.T
        mat_var = dia_matrix((ex[ii]**(-2), 0), shape=(len(ii), len(ii)))
        dx_t = dx[ii]
        dy_t = dy[ii]

        resx = op.minimize(_shift_func, x0, args=(mat_0, mat_0_t, dx_t, mat_var),
                           method='L-BFGS-B', jac=True, options={'disp': False})
        resy = op.minimize(_shift_func, y0, args=(mat_0, mat_0_t, dy_t, mat_var),
                           method='L-BFGS-B', jac=True, options={'disp': False})

        x0 = resx['x']
        y0 = resy['x']

    # to get an estimation of the errors, we update our bead collection
    # with the new displacements

    bb = _build_collection(
        mos_full, resx['x'], resy['x'],
        physical_slices, optical_slices
    )

    # now for each slice, we get the average radial offset from the beads
    # to the bead collection
    abs_err = []
    for i in range(len(physical_slices)):

        _, x_r, y_r, r_r = _get_beads(
            mos_full[physical_slices[i]].attrs, optical_slices[i]
        )

        _dr = []
        for j in range(len(x_r)):
            r = np.sqrt(
                (x_r[j] + resx['x'][i] - bb.x)**2 +
                (y_r[j] + resy['x'][i] - bb.y)**2
            )
            _dr.append(r.min())

        abs_err.append(np.median(np.array(_dr)))

        ref_slice = '\t' + physical_slices[i] + \
            "_Z{0:03d}: ".format(optical_slices[i])
        logger.info(
            ref_slice
            + "{0:.1f} ".format(resx['x'][i])
            + "{0:.1f} ".format(resy['x'][i])
            + "{0:.1f} ".format(abs_err[-1])
        )

    # Now we store all the displacements as attrs
    cube_reg = {
        "abs_dx": [],
        "abs_dy": [],
        "abs_err": [],
        "slice": [],
        "opt_z": [],
    }
    for i in range(len(physical_slices)):
        cube_reg["slice"].append(physical_slices[i])
        cube_reg["opt_z"].append(float(optical_slices[i]))

        cube_reg["abs_dx"].append(resx['x'][i])
        cube_reg["abs_dy"].append(resy['x'][i])
        cube_reg["abs_err"].append(abs_err[i])

    zarr_store.attrs["cube_reg"] = cube_reg
