from pathlib import Path

import voluptuous as vo

DEFAULT_RECIPES = ["cals", "mosaic", "downsample", "beadreg"]


def check_recipes(val):
    for item in val:
        if item not in DEFAULT_RECIPES:
            raise vo.Invalid(f"Not a valid recipe {item!r}")
    return val


# TODO: make sure they are lists of floats
cof_dist_schema = vo.Schema(
    {vo.Required("cof_x"): list, vo.Required("cof_y"): list, vo.Required("tan"): list}
)

schema = vo.Schema(
    {
        vo.Required("input_dir"): vo.Coerce(Path),
        vo.Required("output_dir"): vo.Coerce(Path),
        vo.Required("recipes"): vo.All(list, check_recipes),
        vo.Optional("cof_dist"): cof_dist_schema,
        vo.Optional("sections", default=[]): list,
        vo.Optional("reset", default=True): bool,
    }
)
