import logging
from pathlib import Path
from typing import Dict

from .geometric_distortion import distort
from .preprocess import preprocess
from .settings import Settings
from .stpt_mosaic import STPTMosaic

log = logging.getLogger("owl.daemon.pipeline")


def main(
    *,
    root_dir: Path,
    dark_file: Path,
    flat_file: Path,
    output_dir: Path,
    cof_dist: Dict,
) -> None:
    """[summary]

    Parameters
    ----------
    root_dir : Path
        [description]
    flat_file : Path
        [description]
    output_dir : Path
        [description]
    """
    Settings.cof_dist = cof_dist

    # TODO: This to me moved outside the pipeline
    out = f"{output_dir / root_dir.name}.zarr"
    preprocess(root_dir, out)

    #  Compute dark and flat

    # Geometric distortion
    out_dis = f"{output_dir / root_dir.name}_dis.zarr"
    distort(out, dark_file, flat_file, out_dis)

    mos = STPTMosaic(out_dis)
    mos.initialize_storage(f"{output_dir / root_dir.name}")

    for section in mos.sections():
        section.find_offsets()
        section.stitch(f"{output_dir / root_dir.name}")
        section.downsample(f"{output_dir / root_dir.name}")
