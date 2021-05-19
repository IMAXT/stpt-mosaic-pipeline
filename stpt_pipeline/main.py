from pathlib import Path
from typing import Dict, List

from owl_dev import pipeline
from numcodecs import blosc

from .settings import Settings
from .stpt_bead_registration import find_beads, register_slices
from .stpt_mosaic import STPTMosaic
from .ops import build_cals

blosc.use_threads = False


@pipeline
def main(
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

    mos = STPTMosaic(root_dir)

    done_reset = False

    if "cals" in recipes:
        if reset:
            mos.initialize_storage(output_dir_full)
            done_reset = True

        _ = build_cals(mos._ds, output_dir_full)

    if "mosaic" in recipes:
        if reset:
            mos.initialize_storage(output_dir)

        # need to recheck because you can launch mosaic
        # without cals
        cal_zarr_name = output_dir_full / 'cals.zarr'
        if cal_zarr_name.is_dir():
            cal_type = 'sample'
            logger.info('Using sample calibrations')
        else:
            cal_type = 'static'
            cal_zarr_name = ''
            logger.info('Using static calibrations')

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
    if "beadreg" in recipes:
        find_beads(mos_dis)
        register_slices(mos_dis)
