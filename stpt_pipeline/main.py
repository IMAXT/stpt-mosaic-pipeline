import logging
from pathlib import Path
from typing import Dict, List

from .geometric_distortion import distort
from .preprocess import preprocess
from .settings import Settings
from .stpt_mosaic import STPTMosaic
from .stpt_bead_registration import find_beads, register_slices

log = logging.getLogger("owl.daemon.pipeline")


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
    if cof_dist is not None:
        Settings.cof_dist = cof_dist

    # TODO: This to me moved outside the pipeline
    basedir = output_dir / root_dir.name
    basedir.mkdir(exist_ok=True, parents=True)
    out = basedir / "raw.zarr"
    if "preprocess" in recipes:
        if not root_dir.exists():
            raise FileNotFoundError(f"Directory {root_dir} does not exist.")
        preprocess(root_dir, out)

    # TODO: compute dark and flat

    # Geometric distortion
    out_dis = basedir / "dis.zarr"
    if "distortion" in recipes:
        if not out.exists():
            raise FileNotFoundError(f"Preprocessed data not found in {out}")
        distort(out, Settings.dark_file, Settings.flat_file, out_dis)

    mos = STPTMosaic(out_dis)
    if "mosaic" in recipes:
        if not out_dis.exists():
            raise FileNotFoundError(
                f"Distortion corrected data not found in {out_dis}")
        mos.initialize_storage(basedir)

        for section in mos.sections():
            # initialize stage size:
            section.stage_size = mos.stage_size

            section.find_offsets()
            section.stitch(basedir)

            # after the first section, stage size is fixed:
            mos.stage_size = section.stage_size

    if "downsample" in recipes:
        mos.downsample(basedir)

    if "tiff" in recipes:
        mos.to_tiff(basedir)

    mos_dis = basedir / "mos.zarr"
    if "beadreg" in recipes:
        find_beads(mos_dis)
        register_slices(mos_dis)
