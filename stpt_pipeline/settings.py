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

    x_min = 2
    x_max = 2080
    y_min = 80
    y_max = 1990
    norm_val = 10000.0
    do_flat = True
    do_defringe = False
    channel_to_use = 4
    cof_dist = None
    normal_x = 1000
    normal_y = 1000
