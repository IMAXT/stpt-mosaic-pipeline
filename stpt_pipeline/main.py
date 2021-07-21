from pathlib import Path
from typing import Dict, List

from owl_dev import pipeline
from owl_dev.logging import logger
from numcodecs import blosc

from .settings import Settings
from .stpt_fit_beads import find_beads
from .stpt_bead_registration import register_slices
from .stpt_mosaic import STPTMosaic
from .ops import build_cals

blosc.use_threads = False


def _check_cals(cal_zarr_name: Path):

    if cal_zarr_name.is_dir():
        _type = 'sample'
        _name = cal_zarr_name
        logger.info('Using sample calibrations')
    else:
        _type = 'static'
        _name = ''
        logger.info('Using static calibrations')

    return _name, _type


@pipeline
def main(  # noqa: C901
    *,
    input_dir: Path,
    output_dir: Path,
    recipes: List,
    sections: List,
    reset: bool,
    cof_dist: Dict = None,
) -> None:
    """[summary]

    Parameters
    ----------
    input_dir : Path
        [description]
    output_dir : Path
        [description]
    """
    if cof_dist is not None:
        Settings.cof_dist = cof_dist

    if not input_dir.exists():
        raise FileNotFoundError(f"Directory {input_dir} does not exist.")

    mos = STPTMosaic(input_dir)

    if reset:
        mos.initialize_storage(output_dir)

    if "cals" in recipes:
        build_cals(mos, output_dir)

    if "mosaic" in recipes:
        if reset:
            mos.initialize_storage(output_dir)

        # need to recheck because you can launch mosaic
        # without cals
        cal_zarr_name, cal_type = _check_cals(
            output_dir / 'cals.zarr'
        )

        for section in mos.sections():
            if sections and (section.name not in sections):
                continue
            section.cal_type = cal_type
            section.cal_zarr = cal_zarr_name

            section.find_offsets()
            section.stitch(output_dir)

            # after the first section, stage size is fixed:
            mos.stage_size = section.stage_size

    if "downsample" in recipes:
        mos.downsample(output_dir)

    mos_dis = output_dir / "mos.zarr"

    if "beadfit" in recipes:
        find_beads(mos_dis)

    if "beadreg" in recipes:
        register_slices(mos_dis)
