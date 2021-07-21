from pathlib import Path
from typing import Dict, List

from owl_dev import pipeline
from numcodecs import blosc

from .settings import Settings
from .stpt_bead_registration import find_beads, register_slices
from .stpt_mosaic import STPTMosaic

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

    # TODO: compute dark and flat

    mos = STPTMosaic(input_dir)
    if "mosaic" in recipes:
        if reset:
            mos.initialize_storage(output_dir)

        for section in mos.sections():
            if sections and (section.name not in sections):
                continue
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
