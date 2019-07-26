import logging
from pathlib import Path

from .preprocess import preprocess
from .stpt_mosaic import STPTMosaic

log = logging.getLogger('owl.daemon.pipeline')


def main(*, root_dir: Path, flat_file: Path, output_dir: Path) -> None:
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
    z = preprocess(root_dir, flat_file, output_dir)

    mos = STPTMosaic(z)

    for section in mos.sections():
        section.find_offsets()

    for section in mos.sections():
        section.stitch()
