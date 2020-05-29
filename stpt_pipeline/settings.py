class Settings:
    """Internal settings.

    x/y_min and max are the lower/upper boundaries of the useful section of the
    detector.

    Norm val is a normalization value for the stored values, ideally the gain

    do_flat activates flat correction prior to crossmatching. defringe does that to
    the flat before dividing by it, but it is time consuming and not very
    useful at the moment

    cof_dist are the coefficients of the optical distortion, as measured from
    the images themselves.

    """

    x_min = 12
    x_max = 2078
    y_min = 72
    y_max = 2007
    norm_val = 10_000
    channel_to_use = 4
    """Channel to use to calculate offsets
    """

    cof_dist = {
        "cof_x": [
            -0.3978768594309433,
            22.021244273079315,
            4.969194161793347,
            1.489648554969356,
        ],
        "cof_y": [
            0.24781724665108668,
            -1.908123031173702,
            -3.1947393819597947,
            1.5466333490746447,
        ],
        "tan": [974.42215511, 1037.63451947],
    }
    """distortion coefficients (filled in from the configuration at runtime)
    """

    bscale = 0.001
    bzero = -10

    scales = [2, 4, 8, 16, 32]
    """default downsample scales"""

    normal_x = 1000
    normal_y = 1000

    ftol_desp = 0.1
    """Relative tolerance for convergence when calculating offsets
    """

    mosaic_scale = 5.57  # mosaic displacement units to pixel

    # allowed_error_px = 5.
    """allowed pixel difference when comparing measured disp with mosaic
    """

    flat_file = "/data/meds1_a/eglez/imaxt/flat_dev.nc"
    dark_file = "/data/meds1_a/eglez/imaxt/dark_dev.nc"
