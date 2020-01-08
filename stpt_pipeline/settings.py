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

    cof_dist = None
    """distortion coefficients (filled in from the configuration at runtime)
    """

    normal_x = 1000
    normal_y = 1000

    ftol_desp = 0.1
    """Relative tolerance for convergence when calculating offsets
    """

    mosaic_scale = 5.55  # mosaic displacement units to pixel

    allowed_error_px = 5
    """allowed pixel difference when comparing measured disp with mosaic
    """
