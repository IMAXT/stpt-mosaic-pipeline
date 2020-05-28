from pathlib import Path
from typing import Dict, List

import owl_dev
import owl_dev.config
from owl_dev.logging import logger

from .geometric_distortion import distort
from .preprocess import preprocess
from .settings import Settings
from .stpt_mosaic import STPTMosaic


def main(  # noqa: C901
    *, root_dir: Path, output_dir: Path, recipes: List, cof_dist: Dict = None
) -> None:
    """[summary]

    Parameters
    ----------
    root_dir : Path
        [description]
    output_dir : Path
        [description]
    """
    logger.info("Pipeline started")

    output_dir_full = output_dir / root_dir.name
    owl_dev.config.set({"output_dir": f"{output_dir_full}"})
    owl_dev.setup(main.config)

    if cof_dist is not None:
        Settings.cof_dist = cof_dist

    # TODO: Preprocessing to me moved outside the pipeline
    out = output_dir_full / "raw.zarr"
    if "preprocess" in recipes:
        if not root_dir.exists():
            raise FileNotFoundError(f"Directory {root_dir} does not exist.")
        preprocess(root_dir, out)

    # TODO: compute dark and flat

    # Geometric distortion
    out_dis = output_dir_full / "dis.zarr"
    if "distortion" in recipes:
        if not out.exists():
            raise FileNotFoundError(f"Preprocessed data not found in {out}")
        distort(out, Settings.dark_file, Settings.flat_file, out_dis)

    if not out_dis.exists():
        raise FileNotFoundError(f"Distortion corrected data not found in {out_dis}")

    mos = STPTMosaic(out_dis)
    if "mosaic" in recipes:
        mos.initialize_storage(output_dir_full)

        for section in mos.sections():
            # initialize stage size:
            section.stage_size = mos.stage_size

            section.find_offsets()
            section.stitch(output_dir_full)

            # after the first section, stage size is fixed:
            mos.stage_size = section.stage_size

    if "downsample" in recipes:
        mos.downsample(output_dir_full)

    if "tiff" in recipes:
        mos.to_tiff(output_dir_full)

    logger.info("Pipeline completed")
