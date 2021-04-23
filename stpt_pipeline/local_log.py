from owl_dev.logging import logger

"""
    Convenience functions to simplify info messaging
"""


def to_log(msg: str) -> None:

    logger.info(msg)


def to_debug(msg: str) -> None:

    logger.debug(msg)
