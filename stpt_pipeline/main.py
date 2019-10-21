import logging
from pathlib import Path
from typing import Dict

from .preprocess import preprocess
from .settings import Settings
from .stpt_mosaic import STPTMosaic

log = logging.getLogger('owl.daemon.pipeline')


def main(*, root_dir: Path, flat_file: Path, output_dir: Path, cof_dist: Dict) -> None:
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

    z = preprocess(root_dir, flat_file, output_dir)

    mos = STPTMosaic(z)

    for section in mos.sections():
        section.find_offsets()
        section.stitch()
